"""Thin OpenRouter client.

OpenRouter speaks the OpenAI wire protocol at ``https://openrouter.ai/api/v1``,
so the official ``openai`` SDK works unmodified against it. That means we talk
to three endpoints through one client:

* ``POST /embeddings``       -- document and query vectors
* ``POST /chat/completions`` -- answer generation (streamed)
* ``GET  /key``              -- credential validation, used by the doctor

Everything remote lives here so the rest of the codebase stays synchronous and
easy to test.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from app.config import Settings, get_settings

#: Chat-completion message roles.
Message = dict[str, str]

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    """A remote call failed in a way the user needs to see explained."""


class MissingAPIKeyError(OpenRouterError):
    pass


class EmptyModelResponse(OpenRouterError):
    """The provider answered but with no usable choices.

    OpenRouter occasionally returns ``choices: null`` -- most often when a
    reasoning model spends its entire ``max_tokens`` budget on reasoning and
    emits no content. It is transient, so the retry loop treats it as such.
    """


@dataclass
class Usage:
    """Token accounting for one generation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def explain(error: Exception) -> str:
    """Turn an SDK exception into something a human can act on."""
    if isinstance(error, MissingAPIKeyError):
        return str(error)
    if isinstance(error, AuthenticationError):
        return (
            "OpenRouter rejected the API key (401). Check OPENROUTER_API_KEY in "
            "your .env -- it should look like 'sk-or-v1-...'."
        )
    if isinstance(error, PermissionDeniedError):
        return (
            "OpenRouter denied access to this model (403). The model may not be "
            "available on your account or under its free tier."
        )
    if isinstance(error, RateLimitError):
        return (
            "OpenRouter rate limit hit (429). Free-tier models have strict "
            "per-minute and per-day limits -- wait a moment, index fewer files "
            "at a time, or switch to a paid model."
        )
    if isinstance(error, APIStatusError):
        code = error.status_code
        if code == 402:
            return (
                "OpenRouter reports insufficient credits (402). Add credit at "
                "https://openrouter.ai/credits or switch to a :free model."
            )
        if code == 404:
            return (
                "Model not found (404). Check the model id in .env -- browse "
                "current ids at https://openrouter.ai/models"
            )
        if code in {401, 402, 403, 404}:
            return f"OpenRouter returned {code}. {error.message}"
        return f"OpenRouter returned {code}: {error.message}"
    if isinstance(error, APIConnectionError):
        return (
            "Could not reach OpenRouter. Check your internet connection and any "
            "firewall or proxy settings."
        )
    return f"{type(error).__name__}: {error}"


def build_client(settings: Settings | None = None) -> OpenAI:
    """Create the OpenAI SDK client pointed at OpenRouter."""
    settings = settings or get_settings()
    if not settings.openrouter_api_key:
        raise MissingAPIKeyError(
            "OPENROUTER_API_KEY is not set. Edit .env in the project root and "
            "paste your key from https://openrouter.ai/keys"
        )

    headers: dict[str, str] = {}
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_name:
        headers["X-Title"] = settings.openrouter_app_name

    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        timeout=settings.api_timeout_seconds,
        # Retries are handled below so that 429s can surface real guidance
        # instead of the SDK silently sleeping.
        max_retries=0,
        default_headers=headers or None,
    )


def _retry(
    operation,
    *,
    max_retries: int,
    label: str,
    sleep=time.sleep,
):
    """Run ``operation`` with exponential backoff on transient failures."""
    delay = 2.0
    last: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return operation()
        except MissingAPIKeyError:
            raise
        except EmptyModelResponse as exc:
            last = exc
            if attempt == max_retries:
                raise OpenRouterError(
                    f"{label} returned no content after {attempt + 1} attempt(s). "
                    f"For a reasoning model this usually means RAG_MAX_TOKENS "
                    f"is too small to leave room for an answer after the "
                    f"reasoning trace; try 1024 or more."
                ) from exc
            sleep(min(2.0 * (2**attempt), 30.0))
            continue
        except Exception as exc:  # noqa: BLE001 - re-raised below
            status = getattr(exc, "status_code", None)
            transient = isinstance(exc, APIConnectionError) or (
                isinstance(exc, APIStatusError)
                and (status in RETRYABLE_STATUS or status is None)
            )
            # Auth/quota/permission problems will not fix themselves.
            if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
                raise OpenRouterError(explain(exc)) from exc
            if isinstance(exc, APIStatusError) and status == 402:
                raise OpenRouterError(explain(exc)) from exc
            if not transient or attempt == max_retries:
                raise OpenRouterError(
                    f"{label} failed after {attempt + 1} attempt(s): {explain(exc)}"
                ) from exc

            last = exc
            jitter = random.uniform(0, delay * 0.25)
            sleep(delay + jitter)
            delay = min(delay * 2, 60.0)

    raise OpenRouterError(f"{label} failed: {explain(last)}")  # pragma: no cover


class OpenRouterClient:
    """Convenience wrapper around embeddings and chat completions."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: OpenAI | None = None
        #: Token counts and cost from the most recent completion.
        self.last_usage = Usage()
        self.last_cost = 0.0

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = build_client(self.settings)
        return self._client

    # ── embeddings ──────────────────────────────────────────────────────

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input.

        Input order is preserved. Empty input yields empty output without
        making a request.
        """
        if not texts:
            return []
        clean = [t if t.strip() else " " for t in texts]

        def call() -> list[list[float]]:
            response = self.client.embeddings.create(
                model=self.settings.embed_model,
                input=clean,
                encoding_format="float",
            )
            # Sort by index rather than trusting response order.
            ordered = sorted(response.data, key=lambda d: d.index)
            return [item.embedding for item in ordered]

        return _retry(
            call,
            max_retries=self.settings.embed_max_retries,
            label=f"Embedding with '{self.settings.embed_model}'",
        )

    def embed_batched(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        on_batch=None,
    ) -> list[list[float]]:
        """Embed many texts in API-sized batches.

        Args:
            texts: Strings to embed.
            batch_size: Overrides ``EMBED_BATCH_SIZE``.
            on_batch: Optional ``callback(done, total)`` for progress reporting.

        Returns:
            Vectors in the same order as ``texts``.
        """
        if not texts:
            return []
        size = max(batch_size or self.settings.embed_batch_size, 1)
        total = (len(texts) + size - 1) // size
        vectors: list[list[float]] = []

        for batch_index, start in enumerate(range(0, len(texts), size), start=1):
            batch = texts[start : start + size]
            vectors.extend(self.embed(batch))
            if on_batch is not None:
                on_batch(batch_index, total)

        return vectors

    # ── chat ────────────────────────────────────────────────────────────

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        show_reasoning: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Stream a chat completion.

        Yields ``(content, reasoning)`` tuples. ``reasoning`` is empty unless
        the model is a reasoning model and reasoning was enabled, so callers
        can render it separately or discard it.

        The whole request *and* its iteration are inside the retry loop, because
        a provider can fail after streaming has begun (OpenRouter free-tier
        endpoints regularly return "upstream temporarily overloaded"). Retrying
        is only safe while nothing has been emitted yet: once tokens have been
        handed to the caller, restarting would duplicate visible text, so the
        error propagates instead.

        Being a generator, HTTP errors surface on first iteration rather than at
        call time.
        """
        settings = self.settings
        want_reasoning = (
            settings.show_reasoning if show_reasoning is None else show_reasoning
        )
        label = f"Chat with '{settings.llm_model}'"

        def make_kwargs() -> dict:
            kwargs: dict = {
                "model": settings.llm_model,
                "messages": list(messages),
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": (
                    settings.temperature if temperature is None else temperature
                ),
                "max_tokens": settings.max_tokens if max_tokens is None else max_tokens,
            }
            if not want_reasoning:
                # Suppress the reasoning trace at the source rather than
                # filtering it out of the stream. OpenRouter-specific keys must
                # travel in extra_body so the SDK forwards them.
                kwargs["extra_body"] = {"reasoning": {"enabled": False}}
            return kwargs

        max_attempts = settings.embed_max_retries + 1

        for attempt in range(max_attempts):
            emitted = False
            try:
                stream = self.client.chat.completions.create(**make_kwargs())

                self.last_usage = Usage()
                self.last_cost = 0.0
                saw_choice = False

                for event in stream:
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        self.last_usage = Usage(
                            prompt_tokens=int(
                                getattr(usage, "prompt_tokens", 0) or 0
                            ),
                            completion_tokens=int(
                                getattr(usage, "completion_tokens", 0) or 0
                            ),
                        )
                        self.last_cost = float(getattr(usage, "cost", 0) or 0)

                    if not event.choices:
                        continue
                    saw_choice = True

                    delta = event.choices[0].delta
                    reasoning = ""
                    content = ""
                    if getattr(delta, "reasoning", None):
                        reasoning = delta.reasoning or ""
                    elif getattr(delta, "reasoning_content", None):
                        reasoning = delta.reasoning_content or ""
                    if getattr(delta, "content", None):
                        content = delta.content

                    if content or reasoning:
                        emitted = True
                        yield content, reasoning

                if not saw_choice:
                    raise EmptyModelResponse("stream contained no choices")
                return

            except (MissingAPIKeyError, AuthenticationError, PermissionDeniedError):
                raise

            except Exception as exc:  # noqa: BLE001 - classified below
                status = getattr(exc, "status_code", None)

                if isinstance(exc, APIStatusError) and status == 402:
                    raise OpenRouterError(explain(exc)) from exc

                transient = (
                    isinstance(exc, (APIConnectionError, APIError, EmptyModelResponse))
                    or (isinstance(exc, APIStatusError) and status in RETRYABLE_STATUS)
                    or isinstance(exc, TimeoutError)
                )

                if emitted or not transient or attempt == max_attempts - 1:
                    hint = ""
                    if emitted:
                        hint = (
                            " (the answer was cut off partway through; ask again "
                            "to get a complete one)"
                        )
                    elif isinstance(exc, EmptyModelResponse):
                        hint = (
                            ". For a reasoning model this usually means "
                            "RAG_MAX_TOKENS is too small to leave room for an "
                            "answer after the reasoning trace; try 1024 or more"
                        )
                    raise OpenRouterError(
                        f"{label} failed after {attempt + 1} attempt(s): "
                        f"{explain(exc)}{hint}"
                    ) from exc

                time.sleep(min(2.0 * (2**attempt), 30.0) + random.uniform(0, 1))

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
    ) -> str:
        """Non-streaming completion, used for health checks and tests."""
        settings = self.settings

        def call() -> str:
            response = self.client.chat.completions.create(
                model=settings.llm_model,
                messages=list(messages),
                max_tokens=max_tokens or 64,
                temperature=0,
            )
            if not response.choices:
                raise EmptyModelResponse("provider returned no choices")
            message = response.choices[0].message
            # A reasoning model can return reasoning but no content.
            if not (message.content or "").strip():
                if (getattr(message, "reasoning", None) or "").strip():
                    raise EmptyModelResponse(
                        "model produced only reasoning, no answer"
                    )
                raise EmptyModelResponse("provider returned an empty message")
            return message.content or ""

        return _retry(
            call,
            max_retries=1,
            label=f"Chat with '{settings.llm_model}'",
        )


# ── health checks used by scripts/doctor.py ───────────────────────────────


def check_key(client: OpenAI) -> dict:
    """Validate the credential and report its limits.

    OpenRouter exposes ``GET /api/v1/key``, which is OpenAI-compatible but not
    part of the SDK's typed surface, so ``cast_to=dict`` is used to keep the
    response unvalidated.
    """
    try:
        response = client.get("/key", cast_to=dict)
    except Exception as exc:  # noqa: BLE001
        raise OpenRouterError(f"Could not verify the API key: {explain(exc)}") from exc

    payload = response if isinstance(response, dict) else {}
    data = payload.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    return {
        "label": data.get("label") or "(unlabelled)",
        "usage": data.get("usage"),
        "limit": data.get("limit"),
        "is_free_tier": bool(data.get("is_free_tier")),
        "rate_limit": data.get("rate_limit"),
    }


def probe_embedding(client: OpenRouterClient) -> tuple[int, float]:
    """Embed a probe string. Returns ``(dimensions, elapsed_seconds)``."""
    started = time.perf_counter()
    vectors = client.embed(["rag-workflow health check"])
    elapsed = time.perf_counter() - started
    return len(vectors[0]), elapsed


def probe_chat(client: OpenRouterClient) -> tuple[str, float]:
    """Ask for a one-word reply. Returns ``(content, elapsed_seconds)``."""
    started = time.perf_counter()
    text = client.complete(
        [
            {"role": "system", "content": "Reply with one word."},
            {"role": "user", "content": "ping"},
        ],
        max_tokens=256,
    )
    return text.strip(), time.perf_counter() - started