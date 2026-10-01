"""Document upload, indexing and index management."""

from __future__ import annotations

import streamlit as st

import sys
from collections.abc import Callable
from pathlib import Path

# Python puts the *script's* directory on sys.path[0], which for this file is
# app/ -- so `import app.config` would look for app/app/config.py and fail.
# Putting the repository root on the path first makes the package importable no
# matter how the app is launched (systemd, `streamlit run`, `python app/...`).
# This must stay above the `app.*` imports below.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.config import config_warnings, get_settings
from app.documents import ConversionError, store_upload
from app.formats import resolve_statuses, supported_extensions
from app.indexing import IndexReport, index_batch, rescan_directory
from app.ui import footer, friendly_error, get_client
from app.vectorstore import (
    VectorStoreError,
    connect,
    count,
    delete_by_source,
    dimension_warning,
    list_documents,
    open_collection,
    reset_collection,
    server_batch_size,
)

st.set_page_config(
    page_title="Document Admin",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

settings = get_settings()
vision_enabled = bool(settings.vision_model)
accepted = supported_extensions(vision_enabled=vision_enabled)

st.title("📚 Document Admin")
st.caption(
    f"Upload documents into `{settings.docs_dir}`, convert them to Markdown "
    f"with markitdown, embed with `{settings.embed_model}`, and store the "
    f"vectors in ChromaDB."
)

for warning in config_warnings():
    st.warning(warning, icon="⚠️")

if settings.bind_address not in {"127.0.0.1", "localhost", "::1"}:
    st.error(
        "**This admin interface has no authentication and is not bound to "
        "localhost.** Anyone who can reach it can upload and delete your "
        "documents. Set `APP_BIND_ADDRESS=127.0.0.1` in `.env` unless you "
        "fully trust the network.",
        icon="🔓",
    )

settings.ensure_dirs()

# ── connection ────────────────────────────────────────────────────────────

try:
    chroma_client = connect(settings)
    collection = open_collection(chroma_client, settings, create=True)
except VectorStoreError as exc:
    st.error(f"**Cannot reach the vector store.**\n\n{exc}")
    st.stop()

router = get_client(settings)

warning = dimension_warning(collection, settings)
if warning:
    st.error(f"**Embedding model mismatch.**\n\n{warning}")

total_chunks = count(collection)
documents = list_documents(collection)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Indexed chunks", f"{total_chunks:,}")
c2.metric("Documents", f"{len(documents):,}")
c3.metric("Embedding model", settings.embed_model.split("/")[-1])
c4.metric("Vector dimensions", settings.embed_dims or "unknown")

# ── upload ────────────────────────────────────────────────────────────────

st.subheader("1 · Upload documents")

if not accepted:
    st.error(
        "No file formats are available. Run `./scripts/setup.sh`, or "
        "reinstall with `pip install -r requirements.txt`."
    )
else:
    st.caption("**Accepted formats:** " + " ".join(f"`{e}`" for e in accepted))

uploaded = st.file_uploader(
    "Choose files to upload",
    type=accepted,
    accept_multiple_files=True,
    help=f"Up to {settings.max_files_per_batch} files per batch, "
    f"{settings.max_file_mb} MB each. ZIP archives are expanded and each "
    f"member is indexed as its own document.",
)

if uploaded and len(uploaded) > settings.max_files_per_batch:
    st.error(
        f"You selected {len(uploaded)} files; the limit is "
        f"{settings.max_files_per_batch}. Trim the batch and try again."
    )
    uploaded = None

saved: list[Path] = []
if uploaded:
    with st.spinner("Saving uploads…"):
        rejected: list[tuple[str, str]] = []
        for file in uploaded:
            try:
                saved.append(
                    store_upload(file.getvalue(), file.name, docs_dir=settings.docs_dir)
                )
            except (ConversionError, ValueError) as exc:
                rejected.append((file.name, str(exc)))

    if saved:
        st.success(
            f"Saved {len(saved)} file(s): "
            + ", ".join(f"`{p.name}`" for p in saved[:10])
            + ("…" if len(saved) > 10 else "")
        )
    if rejected:
        with st.expander(f"{len(rejected)} file(s) rejected"):
            for name, reason in rejected:
                st.error(f"`{name}`: {reason}")

# ── indexing ──────────────────────────────────────────────────────────────

st.subheader("2 · Index into the vector store")

targets: list[Path] = saved if saved else []

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Uploaded files** (upload and save a batch above, then index it)")
with col_b:
    if targets:
        st.markdown("**Ready to index:** " + ", ".join(f"`{p.name}`" for p in targets))

col1, col2, col3 = st.columns([2, 1, 1])
index_uploaded = col1.button(
    "Index the uploaded files",
    type="primary",
    disabled=not targets,
    width="stretch",
)
rescan = col2.button(
    "Re-scan the documents folder",
    width="stretch",
)
rebuild = col3.button(
    "Force re-index everything",
    width="stretch",
    help="Ignores the change-detection cache and re-embeds every supported "
    "file. Use this after changing RAG_EMBED_MODEL or the chunk settings.",
)

# Chroma's batch limit is server-side and version dependent, so ask once here
# rather than re-deriving it for every file.
try:
    chroma_batch_size = server_batch_size(chroma_client)
except VectorStoreError:
    chroma_batch_size = None

progress_log: list[str] = []


def make_progress_reporter(status_slot) -> Callable[[str], None]:
    """Build a progress callback that updates the UI live.

    The log used to be rendered only after the whole run finished, which for a
    large document meant staring at an opaque spinner for many minutes with no
    indication of whether anything was happening. Writing into a placeholder as
    work proceeds makes the run observable.

    Streamlit only repaints between script runs, so the placeholder is updated
    on every message; the text is visible as soon as the run yields.
    """

    def progress(message: str) -> None:
        progress_log.append(message)
        stamp = f"{len(progress_log)}/…"
        try:
            status_slot.code(
                f"[{stamp}] {message}\n" + "\n".join(progress_log[-12:]),
                language="text",
            )
        except Exception:  # noqa: BLE001 - never let logging break indexing
            pass

    return progress


report: IndexReport | None = None
progress: Callable[[str], None] = lambda _message: None

if index_uploaded and targets:
    with st.spinner("Indexing…"):
        status_slot = st.empty()
        progress = make_progress_reporter(status_slot)
        try:
            report = index_batch(
                targets,
                collection,
                router,
                settings=settings,
                batch_size=chroma_batch_size,
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001
            friendly_error(exc)

elif rescan:
    with st.spinner("Scanning the documents folder…"):
        status_slot = st.empty()
        progress = make_progress_reporter(status_slot)
        try:
            report = rescan_directory(
                settings.docs_dir,
                collection,
                router,
                settings=settings,
                batch_size=chroma_batch_size,
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001
            friendly_error(exc)

elif rebuild:
    with st.spinner("Rebuilding the whole index…"):
        status_slot = st.empty()
        progress = make_progress_reporter(status_slot)
        try:
            report = rescan_directory(
                settings.docs_dir,
                collection,
                router,
                settings=settings,
                force=True,
                batch_size=chroma_batch_size,
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001
            friendly_error(exc)

if report is not None:
    st.markdown(f"**Finished in {report.elapsed:.1f}s**")
    st.write(
        f"{report.indexed} indexed · {report.unchanged} unchanged · "
        f"{report.skipped} skipped · {report.failed} failed · "
        f"{report.total_chunks} chunks written"
    )

    if report.outcomes:
        st.dataframe(
            [
                {
                    "file": o.file_name,
                    "status": o.status,
                    "chunks": o.chunks_written,
                    "detail": o.detail,
                    "warnings": "; ".join(o.warnings),
                    "seconds": round(o.elapsed, 1),
                }
                for o in report.outcomes
            ],
            width="stretch",
            hide_index=True,
        )

    if progress_log:
        with st.expander(f"Progress log ({len(progress_log)} entries)"):
            st.code("\n".join(progress_log[-400:]), language="text")

# ── index management ──────────────────────────────────────────────────────

st.subheader("3 · Manage the index")

st.caption(
    f"Source files live in `{settings.docs_dir}` and are preserved when you "
    f"delete an entry here, so you can re-index later."
)

tab_docs, tab_formats, tab_danger = st.tabs(
    ["Indexed documents", "Supported formats", "Danger zone"]
)

with tab_docs:
    if not documents:
        st.info("Nothing is indexed yet.")
    else:
        st.dataframe(
            [
                {
                    "document": d.label,
                    "chunks": d.chunks,
                    "hash": d.content_hash,
                }
                for d in documents
            ],
            width="stretch",
            hide_index=True,
        )

        st.markdown("**Delete a document from the index**")
        to_delete = st.selectbox(
            "Choose a document",
            options=[d.source for d in documents],
            format_func=lambda s: next(
                (d.label for d in documents if d.source == s), s
            ),
            label_visibility="collapsed",
        )
        if st.button(f"Delete `{to_delete}` from the index", type="secondary"):
            try:
                removed = delete_by_source(collection, to_delete)
                st.success(
                    f"Removed {removed} chunk(s) for `{to_delete}`. "
                    f"Source file left in place."
                )
                st.rerun()
            except VectorStoreError as exc:
                friendly_error(exc)

with tab_formats:
    st.caption(
        "Formats are detected from what is actually installed, so this list "
        "always matches what the uploader accepts."
    )
    st.dataframe(
        [
            {
                "extensions": status.ext_display,
                "format": status.label,
                "available": "yes" if status.available else "no",
                "why": status.reason or status.spec.note,
            }
            for status in resolve_statuses(vision_enabled=vision_enabled)
        ],
        width="stretch",
        hide_index=True,
    )
    st.info(
        "**Scanned PDFs will not work.** markitdown's PDF converter reads the "
        "text layer only; there is no OCR. An image-only scan produces no "
        "text and is reported as empty rather than silently indexed."
    )

with tab_danger:
    st.warning(
        "These actions cannot be undone. Indexing is cheap to repeat, so "
        "prefer a targeted re-scan over a full rebuild."
    )
    st.write(f"Current collection: `{settings.chroma_collection}` "
             f"({count(collection):,} chunks)")

    if st.button("Delete the entire collection and start over"):
        try:
            reset_collection(chroma_client, settings)
            st.success("Collection dropped. Upload and index again.")
            st.rerun()
        except VectorStoreError as exc:
            friendly_error(exc)

    if st.button("Clear the documents folder", type="secondary"):
        removed = 0
        for path in settings.docs_dir.glob("*"):
            if path.is_file():
                path.unlink()
                removed += 1
        st.success(f"Deleted {removed} file(s) from {settings.docs_dir}")
        st.rerun()

footer(settings, role="Admin app")