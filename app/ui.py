"""Shared Streamlit helpers.

Both apps need the same connection logic, config banner and footer. Keeping it
here means a fix to, say, the dimension-mismatch warning lands in both places
at once.
"""

from __future__ import annotations

import streamlit as st

from app.config import Settings, config_warnings
from app.openrouter import OpenRouterClient, OpenRouterError
from app.vectorstore import (
    Collection,
    VectorStoreError,
    connect,
    count,
    dimension_warning,
    open_collection,
)


@st.cache_resource(show_spinner=False)
def get_client(settings: Settings) -> OpenRouterClient:
    """One OpenRouter client per process.

    Cached because the underlying httpx connection pool is worth reusing
    across Streamlit reruns.
    """
    return OpenRouterClient(settings)


@st.cache_resource(show_spinner=False)
def get_collection(
    settings: Settings,
    *,
    create: bool = True,
) -> tuple[object, Collection]:
    """Connect to Chroma and open the collection. Cached per process.

    Returns ``(chroma_client, collection)``. Note this is the *vector store*
    client -- it has no ``embed`` method. Use :func:`get_client` for the
    OpenRouter client.
    """
    chroma_client = connect(settings)
    return chroma_client, open_collection(chroma_client, settings, create=create)


def show_config_banner(settings: Settings) -> None:
    """Surface configuration problems at the top of the page."""
    for warning in config_warnings():
        st.warning(warning, icon="⚠️")


def footer(settings: Settings, *, role: str) -> None:
    """Render the status footer both apps share."""
    total_chunks = "?"
    try:
        _, collection = get_collection(settings, create=True)
        total_chunks = f"{count(collection):,}"
    except VectorStoreError:
        pass

    st.markdown("---")
    st.caption(
        f"**{role}** · LLM `{settings.llm_model}` · "
        f"embeddings `{settings.embed_model}` · "
        f"chunks `{total_chunks}` · "
        f"Chroma `{settings.chroma_host}:{settings.chroma_port}` "
        f"(collection `{settings.chroma_collection}`)"
    )


def sidebar_status(settings: Settings) -> None:
    """Sidebar panel describing the current configuration."""
    with st.sidebar:
        st.subheader("Configuration")
        st.code(
            f"LLM          {settings.llm_model}\n"
            f"Embeddings   {settings.embed_model}\n"
            f"Vector dims  {settings.embed_dims or 'unknown'}\n"
            f"Top-K        {settings.top_k}\n"
            f"Chunk size   {settings.chunk_chars} chars"
            + (f" (clamped to {settings.max_chunk_chars})"
               if settings.chunk_truncated else "")
            + f"\n"
            f"Chroma       {settings.chroma_host}:{settings.chroma_port}\n"
            f"Collection   {settings.chroma_collection}\n"
            f"Documents    {settings.docs_dir}",
            language="text",
        )


def require_ready(
    settings: Settings, *, role: str
) -> tuple[OpenRouterClient, Collection] | None:
    """Check every precondition, showing the reason and stopping if unmet.

    Returns ``(router, collection)`` when ready -- ``router`` is the OpenRouter
    client, which is what :mod:`app.rag` expects for embedding and generation.
    Otherwise returns ``None`` after calling ``st.stop()``.
    """
    if not settings.openrouter_api_key:
        st.error(
            "**OPENROUTER_API_KEY is not set.**\n\n"
            "Edit the `.env` file in the project root and paste your key from "
            "[openrouter.ai/keys](https://openrouter.ai/keys), then restart "
            "this app."
        )
        st.code("OPENROUTER_API_KEY=sk-or-v1-...", language="bash")
        st.stop()
        return None

    try:
        _, collection = get_collection(settings, create=True)
    except VectorStoreError as exc:
        st.error(f"**Cannot reach the vector store.**\n\n{exc}")
        st.stop()
        return None

    warning = dimension_warning(collection, settings)
    if warning:
        st.error(f"**Embedding model mismatch.**\n\n{warning}")
        st.stop()
        return None

    show_config_banner(settings)

    if count(collection) == 0:
        st.info(
            f"The collection `{settings.chroma_collection}` is empty, so "
            f"there is nothing to search yet. Open the admin app and upload "
            f"some documents first."
        )
        st.stop()
        return None

    return get_client(settings), collection


def friendly_error(exc: Exception) -> None:
    """Show an exception as an actionable message rather than a traceback."""
    if isinstance(exc, OpenRouterError):
        st.error(str(exc))
        if "rate limit" in str(exc).lower():
            st.caption(
                "Free-tier models on OpenRouter have strict per-minute and "
                "per-day limits. Waiting a minute usually clears it."
            )
    else:
        st.error(f"{type(exc).__name__}: {exc}")
