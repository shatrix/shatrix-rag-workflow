"""Central configuration.

Everything the application needs is resolved here, once, from (in order of
precedence):

  1. Real environment variables
  2. The ./.env file
  3. The defaults in this module

All filesystem paths are derived from this repository's root so the project can
be cloned or moved anywhere without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# app/config.py -> app/ -> <repo root>
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

# Real environment variables win over .env so that operators can override a
# single value without editing the file.
load_dotenv(ENV_FILE, override=False)


# ── helpers ───────────────────────────────────────────────────────────────


def _str(key: str, default: str) -> str:
    value = os.environ.get(key)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _int(key: str, default: int, *, minimum: int | None = None) -> int:
    raw = _str(key, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{key}={raw!r} is not a whole number. Fix it in {ENV_FILE}."
        ) from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{key}={value} must be at least {minimum}.")
    return value


def _float(key: str, default: float) -> float:
    raw = _str(key, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{key}={raw!r} is not a number. Fix it in {ENV_FILE}."
        ) from exc


def _bool(key: str, default: bool) -> bool:
    raw = _str(key, "true" if default else "false").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{key}={raw!r} is not a boolean (use true or false).")


def _path(key: str, default: str) -> Path:
    """Resolve a configured path against the repo root if it is relative."""
    candidate = Path(_str(key, default)).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


class ConfigError(RuntimeError):
    """Raised when the environment is missing something required."""


# ── embedding model capabilities ───────────────────────────────────────────

# Rough characters-per-token ratio for English prose. Good enough for the
# chunk-size safety check; not intended to be precise.
CHARS_PER_TOKEN = 4

# Known embedding models served by OpenRouter. `max_input_tokens` is the hard
# per-input ceiling: anything longer is truncated or rejected by the provider.
# `dims` is the natural vector width.
#
# This matters a lot. `liquid/lfm-2.5-embedding-350m:free` only accepts 512
# tokens per input, which is roughly 2000 characters, so large chunks would be
# silently truncated and retrieval quality would collapse.
EMBED_MODEL_CAPABILITIES: dict[str, dict[str, int]] = {
    "nvidia/nemotron-3-embed-1b:free": {"max_input_tokens": 32768, "dims": 2048},
    "liquid/lfm-2.5-embedding-350m:free": {"max_input_tokens": 512, "dims": 1024},
    "nvidia/llama-nemotron-embed-vl-1b-v2:free": {"max_input_tokens": 131072, "dims": 2048},
    "openai/text-embedding-3-small": {"max_input_tokens": 8192, "dims": 1536},
    "openai/text-embedding-3-large": {"max_input_tokens": 8192, "dims": 3072},
    "qwen/qwen3-embedding-8b": {"max_input_tokens": 32768, "dims": 4096},
}

# Used when the configured model is not in the table above. Deliberately
# conservative so an unknown model never silently truncates.
FALLBACK_EMBED_CAPABILITY = {"max_input_tokens": 512, "dims": 0}


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the resolved configuration."""

    # OpenRouter
    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_site_url: str
    openrouter_app_name: str

    # Models
    llm_model: str
    embed_model: str
    vision_model: str
    embed_dims: int
    embed_max_input_tokens: int

    # Retrieval
    top_k: int
    chunk_chars: int
    chunk_overlap_chars: int
    temperature: float
    max_tokens: int
    show_reasoning: bool
    show_sources: bool
    system_prompt: str

    # Vector store
    chroma_host: str
    chroma_port: int
    chroma_collection: str

    # Storage
    data_dir: Path
    docs_dir: Path
    chroma_dir: Path

    # Apps
    bind_address: str
    query_port: int
    admin_port: int
    max_upload_mb: int

    # Limits
    max_file_mb: int
    max_files_per_batch: int
    max_zip_members: int
    max_zip_total_mb: int
    max_zip_expansion_ratio: int

    # API tuning
    index_window_chunks: int
    chroma_max_payload_mb: int
    embed_batch_size: int
    embed_max_retries: int
    api_timeout_seconds: int

    # ── derived ──────────────────────────────────────────────────────────
    @property
    def max_chunk_chars(self) -> int:
        """Largest chunk size this embedding model can actually accept."""
        ceiling = int(self.embed_max_input_tokens * CHARS_PER_TOKEN)
        return min(self.chunk_chars, ceiling)

    @property
    def chunk_truncated(self) -> bool:
        """True if the configured chunk size exceeds the model's input limit."""
        return self.chunk_chars > self.max_chunk_chars

    def require_api_key(self) -> str:
        if not self.openrouter_api_key:
            raise ConfigError(
                f"OPENROUTER_API_KEY is not set.\n"
                f"  Edit {ENV_FILE} and paste your key from https://openrouter.ai/keys\n"
                f"  or export it:  export OPENROUTER_API_KEY=sk-or-v1-..."
            )
        if "your" in self.openrouter_api_key.lower() and len(
            self.openrouter_api_key
        ) < 40:
            raise ConfigError(
                "OPENROUTER_API_KEY still looks like the placeholder from "
                f"{ENV_FILE}. Replace it with a real key."
            )
        return self.openrouter_api_key

    def ensure_dirs(self) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_SYSTEM_PROMPT = (
    "You are a precise technical assistant answering questions about a "
    "collection of documents.\n"
    "Rules:\n"
    "1. Answer using ONLY the numbered context passages provided below.\n"
    "2. If the passages do not contain the answer, say exactly: "
    "\"The provided documents do not contain information about this.\"\n"
    "   Do not speculate or fall back on prior knowledge.\n"
    "3. Cite the passage numbers you relied on, inline, like [2].\n"
    "4. If several passages disagree, say so and present both.\n"
    "5. Be concise. Use Markdown for lists and emphasis where it aids clarity.\n"
    "6. Never invent passage numbers, quotes, or links."
)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build the settings object. Cached; call :func:`reload_settings` to re-read."""
    embed_model = _str("RAG_EMBED_MODEL", "nvidia/nemotron-3-embed-1b:free")
    capability = EMBED_MODEL_CAPABILITIES.get(
        embed_model, FALLBACK_EMBED_CAPABILITY
    )

    chunk_chars = _int("RAG_CHUNK_CHARS", 1800, minimum=200)
    overlap = _int("RAG_CHUNK_OVERLAP_CHARS", 200, minimum=0)
    if overlap >= chunk_chars:
        raise ConfigError(
            f"RAG_CHUNK_OVERLAP_CHARS={overlap} must be smaller than "
            f"RAG_CHUNK_CHARS={chunk_chars}."
        )

    max_file_mb = _int("MAX_FILE_MB", 200, minimum=1)

    return Settings(
        openrouter_api_key=_str("OPENROUTER_API_KEY", ""),
        openrouter_base_url=_str(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/"),
        openrouter_site_url=_str("OPENROUTER_SITE_URL", ""),
        openrouter_app_name=_str("OPENROUTER_APP_NAME", "rag-workflow"),
        llm_model=_str("RAG_LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        embed_model=embed_model,
        vision_model=_str("RAG_VISION_MODEL", ""),
        embed_dims=capability["dims"],
        embed_max_input_tokens=capability["max_input_tokens"],
        top_k=_int("RAG_TOP_K", 5, minimum=1),
        chunk_chars=chunk_chars,
        chunk_overlap_chars=overlap,
        temperature=_float("RAG_TEMPERATURE", 0.2),
        max_tokens=_int("RAG_MAX_TOKENS", 1024, minimum=1),
        show_reasoning=_bool("RAG_SHOW_REASONING", False),
        show_sources=_bool("RAG_SHOW_SOURCES", True),
        system_prompt=_str("RAG_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        chroma_host=_str("CHROMA_HOST", "127.0.0.1"),
        chroma_port=_int("CHROMA_PORT", 8888, minimum=1),
        chroma_collection=_str("CHROMA_COLLECTION", "rag_docs"),
        data_dir=_path("RAG_DATA_DIR", "data"),
        docs_dir=_path("RAG_DOCS_DIR", "data/documents"),
        chroma_dir=_path("CHROMA_DIR", "data/chroma"),
        bind_address=_str("APP_BIND_ADDRESS", "127.0.0.1"),
        query_port=_int("QUERY_APP_PORT", 8901, minimum=1),
        admin_port=_int("ADMIN_APP_PORT", 8902, minimum=1),
        max_upload_mb=_int("APP_MAX_UPLOAD_MB", max_file_mb, minimum=1),
        max_file_mb=max_file_mb,
        max_files_per_batch=_int("MAX_FILES_PER_BATCH", 50, minimum=1),
        max_zip_members=_int("MAX_ZIP_MEMBERS", 500, minimum=1),
        max_zip_total_mb=_int("MAX_ZIP_TOTAL_MB", 500, minimum=1),
        max_zip_expansion_ratio=_int("MAX_ZIP_EXPANSION_RATIO", 100, minimum=1),
        index_window_chunks=_int("INDEX_WINDOW_CHUNKS", 256, minimum=1),
        chroma_max_payload_mb=_int("CHROMA_MAX_PAYLOAD_MB", 4, minimum=1),
        embed_batch_size=_int("EMBED_BATCH_SIZE", 32, minimum=1),
        embed_max_retries=_int("EMBED_MAX_RETRIES", 5, minimum=0),
        api_timeout_seconds=_int("API_TIMEOUT_SECONDS", 120, minimum=1),
    )


def reload_settings() -> Settings:
    """Drop the cache and re-read the environment."""
    get_settings.cache_clear()
    return get_settings()


def config_warnings() -> list[str]:
    """Non-fatal configuration problems worth surfacing in the UI."""
    settings = get_settings()
    warnings: list[str] = []

    if settings.chunk_truncated:
        warnings.append(
            f"RAG_CHUNK_CHARS={settings.chunk_chars} exceeds what "
            f"`{settings.embed_model}` can accept "
            f"({settings.embed_max_input_tokens} tokens ~= "
            f"{settings.max_chunk_chars} characters). Chunks will be clamped to "
            f"{settings.max_chunk_chars} characters. Set RAG_CHUNK_CHARS to "
            f"{settings.max_chunk_chars} or lower to silence this."
        )
    if settings.chunk_overlap_chars > settings.chunk_chars // 3:
        warnings.append(
            f"RAG_CHUNK_OVERLAP_CHARS={settings.chunk_overlap_chars} is more "
            f"than a third of RAG_CHUNK_CHARS={settings.chunk_chars}. Heavy "
            f"overlap repeats text across chunks, which inflates storage and "
            f"embedding cost and barely helps retrieval. 150-250 characters is "
            f"usually plenty."
        )
    if settings.embed_dims == 0:
        warnings.append(
            f"`{settings.embed_model}` is not in the known-models table, so its "
            "vector width is unknown. This is fine, but if you change models "
            "later you must wipe the collection and re-index."
        )
    if not settings.openrouter_api_key:
        warnings.append(
            "OPENROUTER_API_KEY is empty. Add it to .env before indexing or "
            "asking questions."
        )
    if settings.bind_address not in {"127.0.0.1", "localhost", "::1"}:
        warnings.append(
            f"APP_BIND_ADDRESS is {settings.bind_address}, so these apps are "
            "reachable from other machines. The admin app has no "
            "authentication -- anyone who can reach it can upload and delete "
            "your documents. Make sure you trust the whole network."
        )
    return warnings


def redact(key: str) -> str:
    """Show enough of a secret to identify it, never enough to use it."""
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]} ({len(key)} chars)"