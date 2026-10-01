"""Read-only question answering interface."""

from __future__ import annotations

import streamlit as st

import sys
from pathlib import Path

# Python puts the *script's* directory on sys.path[0], which for this file is
# app/ -- so `import app.config` would look for app/app/config.py and fail.
# Putting the repository root on the path first makes the package importable no
# matter how the app is launched (systemd, `streamlit run`, `python app/...`).
# This must stay above the `app.*` imports below.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.ui import footer, friendly_error, require_ready, sidebar_status
from app.config import get_settings

st.set_page_config(
    page_title="Document Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

settings = get_settings()

st.title("💬 Document Assistant")
st.caption(
    f"Answers are generated from the documents indexed in "
    f"`{settings.chroma_collection}` using `{settings.llm_model}`."
)

with st.sidebar:
    st.header("Ask")
    top_k = st.slider(
        "Passages to retrieve",
        min_value=1,
        max_value=20,
        value=settings.top_k,
        help="More passages means broader coverage but a longer prompt and "
        "higher cost.",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=settings.temperature,
        step=0.05,
        help="Lower is more factual and consistent, higher is more varied.",
    )
    max_tokens = st.slider(
        "Max answer length (tokens)",
        min_value=128,
        max_value=4096,
        value=settings.max_tokens,
        step=128,
    )
    show_sources = st.checkbox("Show sources", value=settings.show_sources)
    show_reasoning = st.checkbox(
        "Show model reasoning",
        value=settings.show_reasoning,
        help="Reasoning models emit a visible trace before the answer. "
        "Useful for debugging, usually noise otherwise.",
    )
    st.divider()
    if st.button("Clear conversation", width="stretch"):
        st.session_state.history = []
        st.rerun()

ready = require_ready(settings, role="Query")
if ready is not None:
    # router talks to OpenRouter (embeddings + chat); collection is the vector
    # store. They are different objects with different methods.
    router, collection = ready

    sidebar_status(settings)

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources") and settings.show_sources:
                with st.expander("Sources"):
                    for i, hit in enumerate(turn["sources"], start=1):
                        st.markdown(
                            f"**{i}. {hit.source}** · relevance "
                            f"{hit.score:.3f}"
                        )
                        st.caption(hit.text[:300] + ("…" if len(hit.text) > 300 else ""))
            if turn.get("usage"):
                st.caption(turn["usage"])

    question = st.chat_input("Ask a question about the indexed documents…")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        from app.openrouter import OpenRouterError
        from app.rag import NoDocumentsIndexedError, build_messages, retrieve

        history_for_prompt = [
            {"role": t["role"], "content": t["content"]}
            for t in st.session_state.history[-6:]
        ]

        try:
            hits = retrieve(question, collection, router, top_k=top_k)

            messages = build_messages(
                question, hits, history=history_for_prompt, settings=settings
            )

            with st.chat_message("assistant"):
                answer_parts: list[str] = []
                reasoning_parts: list[str] = []
                placeholder = st.empty()
                reasoning_box = None

                if show_reasoning:
                    reasoning_box = st.expander("Reasoning", expanded=False)

                for content, reasoning in router.stream_chat(
                    messages,
                    show_reasoning=show_reasoning,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        if reasoning_box is not None:
                            reasoning_box.write("".join(reasoning_parts))
                    if content:
                        answer_parts.append(content)
                        placeholder.markdown("".join(answer_parts))

                answer = "".join(answer_parts).strip()

                if not answer:
                    st.warning(
                        "The model returned an empty response. This usually "
                        "means the free-tier endpoint was rate limited "
                        "mid-stream — try again."
                    )

                if show_sources and hits:
                    with st.expander(f"Sources ({len(hits)})"):
                        for i, hit in enumerate(hits, start=1):
                            label = hit.source
                            if hit.heading:
                                label = f"{label} :: {hit.heading}"
                            st.markdown(f"**{i}. {label}** · relevance {hit.score:.3f}")
                            st.text(hit.text[:500] + ("…" if len(hit.text) > 500 else ""))

            usage = (
                f"{len(answer_parts)} streamed chunks · "
                f"{len(question)} char question · "
                f"{len(hits)} passages retrieved"
            )

            st.session_state.history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": answer or "_(no answer)_",
                    "sources": hits if show_sources else [],
                    "usage": usage,
                }
            )

        except (OpenRouterError, NoDocumentsIndexedError) as exc:
            friendly_error(exc)
        except Exception as exc:  # noqa: BLE001 - never crash the app
            friendly_error(exc)

footer(settings, role="Query app")