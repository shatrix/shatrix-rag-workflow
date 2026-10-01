"""The indexing pipeline: file on disk -> vectors in Chroma.

Stages, in order:

    units  = convert_file(...)      # file -> Markdown (handles archives)
    chunks = chunk_text(...)        # Markdown -> embedding-sized pieces
    vectors = client.embed_batched(...)  # pieces -> vectors
    upsert_chunks(...)              # vectors -> Chroma

Change detection makes re-running cheap. Chunk ids are derived from the
SHA-256 of the source bytes, so re-indexing identical content overwrites in
place instead of doubling the collection, and files whose hash is already
recorded are skipped before any API call is made. On a rate-limited free tier
that difference is the difference between seconds and hours.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.chunking import Chunk, chunk_text
from app.config import Settings, get_settings
from app.documents import ConvertedUnit, ConversionError, convert_file
from app.openrouter import OpenRouterClient, OpenRouterError
from app.vectorstore import (
    ChunkRecord,
    Collection,
    VectorStoreError,
    delete_by_source,
    stored_hashes,
    upsert_chunks,
)


@dataclass
class IndexOutcome:
    """What happened to one uploaded file."""

    file_name: str
    status: str  # indexed | unchanged | skipped | failed
    detail: str = ""
    chunks_written: int = 0
    chunks_skipped: int = 0
    units: int = 1
    #: SHA-256 prefix of the source bytes, so callers can record change state.
    content_hash: str = ""
    warnings: list[str] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status != "failed"

    @property
    def symbol(self) -> str:
        return {
            "indexed": "OK",
            "unchanged": "--",
            "skipped": "!!",
            "failed": "XX",
        }.get(self.status, "??")


@dataclass
class IndexReport:
    """Aggregate result of indexing a batch."""

    outcomes: list[IndexOutcome] = field(default_factory=list)
    total_chunks: int = 0
    total_skipped: int = 0
    elapsed: float = 0.0

    @property
    def indexed(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "indexed")

    @property
    def unchanged(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "unchanged")

    @property
    def skipped(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "skipped")

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "failed")


def content_hash(data: bytes) -> str:
    """Short, stable content fingerprint used for change detection and ids."""
    return hashlib.sha256(data).hexdigest()[:16]


def chunk_id(doc_id: str, index: int) -> str:
    """Deterministic chunk id, so re-indexing overwrites rather than duplicates."""
    return f"{doc_id}:{index}"


def _build_records(
    units: list[ConvertedUnit],
    *,
    settings: Settings,
) -> tuple[list[tuple[Chunk, ConvertedUnit]], list[str]]:
    """Flatten converted units into chunks.

    Each chunk is paired with the unit it came from, so archive members keep
    their own attribution without a second pass.

    Returns:
        ``(chunks_with_origin, warnings)``
    """
    pairs: list[tuple[Chunk, ConvertedUnit]] = []
    warnings: list[str] = []

    limit = settings.max_chunk_chars

    for unit in units:
        for warning in unit.warnings:
            warnings.append(f"{unit.label}: {warning}")

        if not unit.markdown.strip():
            continue

        # Prefix each chunk with its origin so the embedding and the LLM both
        # know which document a passage came from. The filename is the
        # identifier: it is the one thing reliably known for every format.
        #
        # No separate title is prepended. Every converter derives one
        # differently and unreliably -- for PDFs it is whatever text sat at the
        # top of page one, which for a long manual is a licence header or a
        # stray code fragment. A wrong title prepended to every chunk
        # contaminates every embedding and every prompt, so the heading
        # breadcrumb the chunker already produces carries the structure
        # instead.
        prefix = (
            f"Source: {unit.source} :: {unit.member_path}\n\n"
            if unit.member_path
            else f"Source: {unit.source}\n\n"
        )

        body_limit = max(limit - len(prefix), 200)
        for chunk in chunk_text(
            unit.markdown,
            chunk_chars=body_limit,
            overlap_chars=min(settings.chunk_overlap_chars, body_limit // 4),
        ):
            text = prefix + chunk.text
            if len(text) > limit:
                # Header plus content exceeded the model's ceiling: trim the
                # body rather than send an over-long input.
                text = text[:limit]
            pairs.append(
                (
                    Chunk(text=text, heading=chunk.heading, index=len(pairs)),
                    unit,
                )
            )

    return pairs, warnings


def index_file(
    path: Path,
    collection: Collection,
    client: OpenRouterClient,
    *,
    settings: Settings | None = None,
    existing_hashes: dict[str, str] | None = None,
    force: bool = False,
    source: str | None = None,
    batch_size: int | None = None,
    window_chunks: int | None = None,
    max_payload_bytes: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> IndexOutcome:
    """Index a single file, skipping it when its content is already present.

    Args:
        path: The file to index.
        collection: Target Chroma collection.
        client: OpenRouter client used for embeddings.
        settings: Configuration override.
        existing_hashes: ``source -> content_hash`` map from a previous run,
            used to skip unchanged files.
        force: Re-index even when the hash matches.
        source: Display name; defaults to ``path.name``.
        progress: Optional ``callback(message)`` for UI feedback.

    Returns:
        An :class:`IndexOutcome`. Never raises for per-file problems; those are
        reported in ``status``/``detail`` so one bad file cannot abort a batch.
    """
    settings = settings or get_settings()
    name = source or path.name
    started = time.perf_counter()

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return IndexOutcome(
            file_name=name,
            status="failed",
            detail=f"cannot read file: {exc}",
            elapsed=time.perf_counter() - started,
        )

    digest = content_hash(raw)

    if not force and existing_hashes is not None:
        if existing_hashes.get(name) == digest:
            return IndexOutcome(
                file_name=name,
                status="unchanged",
                content_hash=digest,
                detail="content unchanged since last index",
                elapsed=time.perf_counter() - started,
            )

    # A changed file replaces its own chunks; any other stale chunks for the
    # same source are removed so shrinking a document cannot leave orphans.
    try:
        delete_by_source(collection, name)
    except VectorStoreError as exc:
        return IndexOutcome(
            file_name=name,
            status="failed",
            detail=str(exc),
            elapsed=time.perf_counter() - started,
        )

    try:
        say(f"converting {name}...")
        units = convert_file(path, source=name, settings=settings)
    except ConversionError as exc:
        return IndexOutcome(
            file_name=name,
            status="failed",
            detail=str(exc),
            elapsed=time.perf_counter() - started,
        )

    pairs, warnings = _build_records(units, settings=settings)
    chunks = [c for c, _ in pairs]

    if not chunks:
        return IndexOutcome(
            file_name=name,
            status="skipped",
            detail="no usable text to index",
            units=len(units),
            warnings=warnings,
            elapsed=time.perf_counter() - started,
        )

    total = len(pairs)
    window = max(1, window_chunks or settings.index_window_chunks)
    say(
        f"{name}: {total} chunk(s) to index, "
        f"{len(units)} unit(s), writing every {window}"
    )

    # Embed and write in windows rather than all at once. A 1400-page PDF
    # produces thousands of chunks; embedding everything before writing
    # anything means a failure near the end loses all of it, and leaves the
    # collection empty until the very last request. Windows make progress
    # durable and observable, and cap peak memory.
    written = 0
    for offset in range(0, total, window):
        window_pairs = pairs[offset : offset + window]
        slice_chunks = [c for c, _ in window_pairs]

        try:
            vectors = client.embed_batched(
                [c.text for c in slice_chunks],
                on_batch=lambda done, batches, o=offset: say(
                    f"{name}: embedding {o + done * settings.embed_batch_size}"
                    f"/{total} chunk(s)"
                ),
            )
        except OpenRouterError as exc:
            return IndexOutcome(
                file_name=name,
                status="failed",
                detail=(
                    f"{exc} (stopped after {written}/{total} chunks were "
                    f"already written and kept)"
                ),
                units=len(units),
                chunks_written=written,
                warnings=warnings,
                elapsed=time.perf_counter() - started,
            )

        if len(vectors) != len(slice_chunks):  # pragma: no cover - defensive
            return IndexOutcome(
                file_name=name,
                status="failed",
                detail=(
                    f"embedding count mismatch in window at {offset}: sent "
                    f"{len(slice_chunks)} chunks, received {len(vectors)} "
                    f"vectors"
                ),
                units=len(units),
                chunks_written=written,
                warnings=warnings,
                elapsed=time.perf_counter() - started,
            )

        records = [
            ChunkRecord(
                id=chunk_id(digest, chunk.index),
                text=chunk.text,
                vector=vector,
                source=name,
                doc_id=digest,
                chunk_index=chunk.index,
                content_hash=digest,
                heading=chunk.heading,
                source_zip=unit.source_zip,
                member_path=unit.member_path,
            )
            for (chunk, unit), vector in zip(window_pairs, vectors)
        ]

        try:
            written += upsert_chunks(
                collection,
                records,
                batch_size=batch_size,
                max_payload_bytes=(
                    max_payload_bytes
                    if max_payload_bytes is not None
                    else settings.chroma_max_payload_mb * 1024 * 1024
                ),
            )
        except VectorStoreError as exc:
            return IndexOutcome(
                file_name=name,
                status="failed",
                detail=(
                    f"{exc} (stopped after {written}/{total} chunks were "
                    f"already written and kept)"
                ),
                units=len(units),
                chunks_written=written,
                warnings=warnings,
                elapsed=time.perf_counter() - started,
            )

        say(f"{name}: indexed {written}/{total} chunk(s)")

    return IndexOutcome(
        file_name=name,
        status="indexed",
        detail=f"{written} chunk(s) across {len(units)} unit(s)",
        chunks_written=written,
        content_hash=digest,
        units=len(units),
        warnings=warnings,
        elapsed=time.perf_counter() - started,
    )


def index_batch(
    paths: list[Path],
    collection: Collection,
    client: OpenRouterClient,
    *,
    settings: Settings | None = None,
    force: bool = False,
    batch_size: int | None = None,
    window_chunks: int | None = None,
    max_payload_bytes: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> IndexReport:
    """Index many files, re-reading the change-detection map between files.

    Reading the map once up front and once at the end keeps the common case
    cheap while still picking up files written earlier in the same batch.
    """
    settings = settings or get_settings()
    report = IndexReport()
    started = time.perf_counter()
    hashes = stored_hashes(collection)

    for path in paths:
        outcome = index_file(
            path,
            collection,
            client,
            settings=settings,
            existing_hashes=hashes,
            force=force,
            batch_size=batch_size,
            window_chunks=window_chunks,
            max_payload_bytes=max_payload_bytes,
            progress=progress,
        )
        report.outcomes.append(outcome)
        report.total_chunks += outcome.chunks_written
        report.total_skipped += outcome.chunks_skipped
        if outcome.status == "indexed":
            # Reflect the write so a duplicate path in the same batch is caught.
            hashes[outcome.file_name] = content_hash(path.read_bytes())

    report.elapsed = time.perf_counter() - started
    return report


def rescan_directory(
    docs_dir: Path,
    collection: Collection,
    client: OpenRouterClient,
    *,
    settings: Settings | None = None,
    force: bool = False,
    extensions: list[str] | None = None,
    batch_size: int | None = None,
    window_chunks: int | None = None,
    max_payload_bytes: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> IndexReport:
    """Index every supported file sitting in the documents directory."""
    settings = settings or get_settings()

    if not docs_dir.exists():
        return IndexReport()

    from app.formats import supported_extensions

    allowed = set(extensions) if extensions else set(
        supported_extensions(vision_enabled=bool(settings.vision_model))
    )

    paths = [
        p
        for p in sorted(docs_dir.iterdir())
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in allowed
    ]

    return index_batch(
        paths,
        collection,
        client,
        settings=settings,
        force=force,
        batch_size=batch_size,
        window_chunks=window_chunks,
        max_payload_bytes=max_payload_bytes,
        progress=progress,
    )

#: Extensions whose chunk count can be estimated from a character count
#: without reading the whole file. Everything else falls back to a size proxy.
_TEXTY = {".md", ".markdown", ".txt", ".text", ".json", ".jsonl", ".csv",
          ".htm", ".html", ".xml", ".rss", ".atom", ".epub", ".ipynb",
          ".log", ".rst", ".tex", ".yaml", ".yml", ".toml", ".ini"}

#: Bytes per character assumed for the text formats above. Deliberately high so
#: the estimate over-reports slightly rather than under-reporting.
_CHARS_PER_BYTE = 0.25


def estimate_chunks(
    path: Path, *, settings: Settings | None = None
) -> tuple[int, str]:
    """Estimate how many chunks a document will produce.

    Uses only ``stat``, so it stays cheap for a very large file. The result is
    an estimate, not a promise: it is there to tell an operator whether a job
    belongs in the browser or on the command line.

    Returns:
        ``(chunks, basis)`` where ``basis`` explains how the number was reached.
    """
    settings = settings or get_settings()
    try:
        size = path.stat().st_size
    except OSError:
        return 0, "unreadable"

    if path.suffix.lower() in _TEXTY:
        characters = size * _CHARS_PER_BYTE
        return (
            max(1, int(characters / max(settings.max_chunk_chars, 1))),
            "from file size",
        )

    # Binary formats (PDF, Office, archives) are dominated by embedded media
    # and fonts, so bytes map poorly to text. Roughly 50 KB of PDF per page and
    # ~400 characters of text per page is a serviceable rule of thumb.
    bytes_per_page = 50 * 1024
    pages = max(1, size // bytes_per_page)
    characters = pages * 400
    return (
        max(1, int(characters / max(settings.max_chunk_chars, 1))),
        f"~{pages} pages from file size",
    )
