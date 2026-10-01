"""Question answering: retrieve passages, then stream a grounded answer.

The split between this module and the UI is deliberate -- everything here is
plain synchronous Python returning strings and iterables, so it can be tested
without Streamlit and reasoned about without a browser.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.openrouter import Message, OpenRouterClient, OpenRouterError
from app.vectorstore import Collection, SearchHit, count, search

#: Per-passage character budget when assembling the context block. Keeps the
#: prompt inside small models' effective context even when top_k is raised.
MAX_CONTEXT_CHARS = 12_000


@dataclass
class Answer:
    """A completed answer plus everything needed to justify it."""

    question: str
    text: str
    reasoning: str = ""
    hits: list[SearchHit] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0

    @property
    def sources(self) -> list[SearchHit]:
        return self.hits

    def usage_line(self) -> str:
        return (
            f"{self.prompt_tokens} prompt + {self.completion_tokens} completion "
            f"tokens, {self.elapsed:.1f}s"
        )


def format_context(hits: Sequence[SearchHit]) -> str:
    """Render retrieved passages as a numbered block.

    Numbering matters: the system prompt asks the model to cite ``[n]``, and
    showing the same numbers in the UI makes those citations checkable.
    """
    blocks: list[str] = []
    used = 0

    for i, hit in enumerate(hits, start=1):
        label = hit.source
        if hit.heading:
            label = f"{label} :: {hit.heading}"

        header = f"--- Passage {i} (source: {label}) ---"
        body = hit.text.strip()
        if used + len(header) + len(body) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - used - len(header) - 2
            if remaining <= 200:
                break
            body = body[:remaining].rstrip() + "\n... (passage truncated)"
            blocks.append(f"{header}\n{body}")
            break

        blocks.append(f"{header}\n{body}")
        used += len(header) + len(body) + 2

    return "\n\n".join(blocks)


def build_messages(
    question: str,
    hits: Sequence[SearchHit],
    *,
    history: Sequence[Message] = (),
    settings: Settings | None = None,
) -> list[Message]:
    """Assemble the chat messages for one question."""
    settings = settings or get_settings()
    context = format_context(hits)

    system = settings.system_prompt
    if not context:
        user = (
            f"User Question: {question}\n\n"
            "No documents were retrieved for this question, so there is no "
            "context to answer from. Tell the user that the document "
            "collection has no relevant information."
        )
    else:
        user = (
            f"{context}\n\n"
            f"---\n"
            f"User Question: {question}\n\n"
            f"Answer the question using only the numbered passages above."
        )

    messages: list[Message] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user})
    return messages


def retrieve(
    question: str,
    collection: Collection,
    client: OpenRouterClient,
    *,
    settings: Settings | None = None,
    top_k: int | None = None,
) -> list[SearchHit]:
    """Embed the question and fetch the nearest passages.

    Raises:
        NoDocumentsIndexedError: The collection is empty.
        OpenRouterError: Embedding failed.
    """
    settings = settings or get_settings()
    k = top_k or settings.top_k

    if count(collection) == 0:
        raise NoDocumentsIndexedError

    vectors = client.embed([question])
    return search(collection, vectors[0], top_k=k)


class NoDocumentsIndexedError(RuntimeError):
    """Raised when the collection has no vectors to search."""


def stream_answer(
    question: str,
    hits: Sequence[SearchHit],
    client: OpenRouterClient,
    *,
    history: Sequence[Message] = (),
    settings: Settings | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(content, reasoning)`` deltas for a grounded answer."""
    settings = settings or get_settings()
    messages = build_messages(question, hits, history=history, settings=settings)
    yield from client.stream_chat(messages)


def ask(
    question: str,
    collection: Collection,
    client: OpenRouterClient,
    *,
    history: Sequence[Message] = (),
    settings: Settings | None = None,
) -> Answer:
    """Answer a question end to end, non-streaming.

    Returns an :class:`Answer` with the text, retrieved sources and token
    counts. Raises :class:`NoDocumentsIndexedError` when nothing is indexed.
    """
    import time

    settings = settings or get_settings()
    started = time.perf_counter()

    hits = retrieve(question, collection, client, settings=settings)
    if not hits:
        raise NoDocumentsIndexedError

    messages = build_messages(question, hits, history=history, settings=settings)

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for content, reasoning in client.stream_chat(messages):
        if content:
            content_parts.append(content)
        if reasoning:
            reasoning_parts.append(reasoning)

    text = "".join(content_parts).strip()
    if not text:
        raise OpenRouterError(
            "The model returned an empty response. This usually means the "
            "free-tier endpoint was rate limited mid-stream; try again."
        )

    return Answer(
        question=question,
        text=text,
        reasoning="".join(reasoning_parts).strip(),
        hits=hits,
        elapsed=time.perf_counter() - started,
    )


def estimate_tokens(text: str) -> int:
    """Rough token count for display when the API does not report usage."""
    from app.config import CHARS_PER_TOKEN

    return max(1, len(text) // CHARS_PER_TOKEN)