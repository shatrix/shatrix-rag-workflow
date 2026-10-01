"""ChromaDB access.

Both Streamlit apps talk to a single ``chroma run`` server over HTTP rather
than opening the on-disk store directly. Chroma's local persistence mode does
not support two processes reading and writing the same SQLite database safely,
and we genuinely have two processes (the admin app writes, the query app
reads), so a server is the correct shape rather than the convenient one.

Vectors are always computed by :mod:`app.openrouter` and passed in explicitly.
That means we must pass ``embedding_function=None`` when creating a collection:
the Chroma default is an ONNX MiniLM model that would download ~80 MB on first
use and, worse, silently produce different vectors than the query path.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.config import Settings, get_settings

#: Cosine similarity. These embedding models are trained for cosine, and
#: Chroma's default L2 distance would rank results differently for no benefit.
SPACE = "cosine"

#: Metadata keys we write onto every chunk.
META_SOURCE = "source"
META_DOC_ID = "doc_id"
META_CHUNK_INDEX = "chunk_index"
META_HEADING = "heading"
META_CONTENT_HASH = "content_hash"
META_SOURCE_ZIP = "source_zip"
META_MEMBER_PATH = "member_path"
META_TITLE = "title"

#: Keys describing how the collection was built.
META_KEY_EMBED_MODEL = "embedding_model"
META_KEY_EMBED_DIMS = "embedding_dims"
META_KEY_SPACE = "space"


class VectorStoreError(RuntimeError):
    pass


@dataclass
class ChunkRecord:
    """One chunk ready to be written."""

    id: str
    text: str
    vector: list[float]
    source: str
    doc_id: str
    chunk_index: int
    content_hash: str
    heading: str = ""
    source_zip: str = ""
    member_path: str = ""

    def metadata(self) -> dict[str, str | int]:
        # Chroma metadata accepts only str/int/float/bool. Empty values are
        # dropped rather than stored as "" to keep filters tidy.
        meta: dict[str, str | int] = {
            META_SOURCE: self.source,
            META_DOC_ID: self.doc_id,
            META_CHUNK_INDEX: self.chunk_index,
            META_CONTENT_HASH: self.content_hash,
        }
        optional = {
            META_HEADING: self.heading,
            META_SOURCE_ZIP: self.source_zip,
            META_MEMBER_PATH: self.member_path,
        }
        for key, value in optional.items():
            if value:
                meta[key] = value
        return meta


@dataclass
class SearchHit:
    """One retrieved passage."""

    text: str
    metadata: dict
    distance: float

    @property
    def source(self) -> str:
        return str(self.metadata.get(META_SOURCE, "unknown"))

    @property
    def heading(self) -> str:
        return str(self.metadata.get(META_HEADING, ""))

    @property
    def score(self) -> float:
        """Similarity in [0, 1]. Chroma returns distance; for cosine that is
        ``1 - cosine_similarity``, so this is the similarity."""
        return max(0.0, min(1.0, 1.0 - self.distance))


@dataclass
class DocumentSummary:
    """Aggregated view of one indexed source, for the admin UI."""

    source: str
    doc_id: str
    chunks: int
    content_hash: str = ""
    member_paths: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.member_paths:
            inner = self.member_paths[0]
            if len(self.member_paths) > 1:
                inner = f"{inner} +{len(self.member_paths) - 1} more"
            return f"{self.source}/{inner}"
        return self.source


def connect(settings: Settings | None = None, *, timeout: float = 10.0) -> ClientAPI:
    """Open a client against the Chroma server and verify it answers."""
    settings = settings or get_settings()
    try:
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        client.heartbeat()
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        raise VectorStoreError(
            f"Cannot reach the Chroma server at "
            f"{settings.chroma_host}:{settings.chroma_port}.\n"
            f"  Start it with:  .venv/bin/chroma run --path {settings.chroma_dir} "
            f"--port {settings.chroma_port}\n"
            f"  Or via:        ./scripts/run-local.sh\n"
            f"  Underlying error: {exc}"
        ) from exc
    return client


def open_collection(
    client: ClientAPI,
    settings: Settings | None = None,
    *,
    create: bool = True,
) -> Collection:
    """Fetch the configured collection, optionally creating it.

    ``embedding_function=None`` is essential: see the module docstring.
    """
    settings = settings or get_settings()
    metadata: dict[str, str | int] = {
        META_KEY_EMBED_MODEL: settings.embed_model,
        META_KEY_SPACE: SPACE,
    }
    if settings.embed_dims:
        metadata[META_KEY_EMBED_DIMS] = settings.embed_dims

    try:
        if create:
            return client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata=metadata,
                embedding_function=None,
            )
        return client.get_collection(
            name=settings.chroma_collection,
            embedding_function=None,
        )
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(
            f"Could not open Chroma collection "
            f"'{settings.chroma_collection}': {exc}"
        ) from exc


def collection_exists(client: ClientAPI, name: str) -> bool:
    try:
        client.get_collection(name=name, embedding_function=None)
        return True
    except Exception:  # noqa: BLE001
        return False


def dimension_warning(collection: Collection, settings: Settings) -> str | None:
    """Warn when the collection was built with a different embedding model.

    Chroma accepts vectors of any width up to the collection's configured
    dimension and only rejects mismatches at query time, so this check is what
    turns a confusing runtime error into an actionable message.
    """
    try:
        meta = collection.metadata or {}
    except Exception:  # noqa: BLE001
        return None

    stored_model = meta.get(META_KEY_EMBED_MODEL)
    stored_dims = meta.get(META_KEY_EMBED_DIMS)

    if stored_model and stored_model != settings.embed_model:
        return (
            f"This collection was indexed with '{stored_model}' but your .env "
            f"now specifies '{settings.embed_model}'. Vectors from different "
            f"models cannot be compared. Use the admin app's 'Delete entire "
            f"collection' and re-index, or point RAG_EMBED_MODEL back to "
            f"'{stored_model}'."
        )
    if (
        stored_dims
        and settings.embed_dims
        and int(stored_dims) != settings.embed_dims
    ):
        return (
            f"Collection stores {stored_dims}-dimensional vectors but "
            f"'{settings.embed_model}' produces {settings.embed_dims}. Wipe and "
            f"re-index."
        )
    return None


def actual_dimensions(collection: Collection) -> int | None:
    """Read the real vector width from the collection's HNSW index config."""
    try:
        config = collection.configuration
        index = config["hnsw"]["space"]  # touch it to force resolution
        del index
        return int(config.get("hnsw", {}).get("metadata", {}).get("hnsw:dim", 0)) or None
    except Exception:  # noqa: BLE001 - purely informational
        return None


def server_batch_size(client: ClientAPI, fallback: int = 64) -> int:
    """Ask the server what batch size it will accept.

    The limit is server-side and version-dependent, so it is queried rather
    than hard-coded, with a small conservative fallback.
    """
    try:
        size = int(client.get_max_batch_size())
        return max(1, min(size, 5_000))
    except Exception:  # noqa: BLE001
        return fallback


#: Used when the server's real limit cannot be determined.
DEFAULT_BATCH_SIZE = 64

#: Conservative ceiling on the JSON size of one upsert request.
#:
#: The server's ``max_batch_size`` is a *record count* limit, not a payload
#: limit, so it cannot be trusted on its own. Each record carries one
#: embedding, and a 2048-dimensional float vector serialises to roughly 20
#: bytes per value, so a single record is ~40 KB. Sending thousands at once is
#: what produced a "Payload too large" failure on a 1400-page PDF.
#:
#: Keep well under typical reverse-proxy body limits (nginx defaults to 1 MB).
DEFAULT_MAX_PAYLOAD_BYTES = 4 * 1024 * 1024

#: JSON bytes assumed per embedding value. ``repr`` of a float is up to 19
#: characters, plus a separator, and JSON escaping can add more, so 22 leaves
#: headroom without being wildly pessimistic.
BYTES_PER_FLOAT = 22


def estimate_record_bytes(record: ChunkRecord) -> int:
    """Approximate the JSON size of one upsert record.

    Overestimates on purpose: sizing batches from an estimate that is slightly
    too large is harmless, whereas one that is too small causes the write to
    fail.
    """
    meta_bytes = sum(len(str(k)) + len(str(v)) + 8 for k, v in record.metadata().items())
    return (
        len(record.text) * 2  # UTF-8, plus escaping of newlines and quotes
        + len(record.vector) * BYTES_PER_FLOAT
        + meta_bytes
        + len(record.id) * 2
        + 128  # field names, brackets, commas
    )


def plan_batches(
    records: Sequence[ChunkRecord],
    *,
    max_records: int,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> list[Sequence[ChunkRecord]]:
    """Split records into batches that respect both limits.

    A batch ends when it hits ``max_records`` or when adding the next record
    would exceed ``max_payload_bytes``.
    """
    if not records:
        return []

    batches: list[Sequence[ChunkRecord]] = []
    current: list[ChunkRecord] = []
    current_bytes = 0

    for record in records:
        size = estimate_record_bytes(record)
        if current and (
            len(current) >= max_records or current_bytes + size > max_payload_bytes
        ):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += size

    if current:
        batches.append(current)
    return batches


def _is_payload_error(exc: Exception) -> bool:
    """Recognise a request-body-too-large rejection."""
    message = str(exc).lower()
    return "payload too large" in message or "request entity too large" in message or (
        getattr(exc, "status_code", None) == 413
    )


def upsert_chunks(
    collection: Collection,
    records: Sequence[ChunkRecord],
    *,
    batch_size: int | None = None,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    client: ClientAPI | None = None,
    on_batch=None,
) -> int:
    """Write chunks to Chroma in batches that fit both limits.

    Chunk ids are deterministic (derived from the document hash), so
    re-indexing the same content overwrites in place instead of duplicating.

    If the server still rejects a batch for being too large, the batch is split
    in half and retried, so an unexpectedly small server limit degrades
    throughput instead of losing the write.

    Args:
        batch_size: Explicit record-count cap. Callers that already know the
            server's limit should pass it, so the HTTP layer need not be
            consulted per file.
        max_payload_bytes: Ceiling on the estimated JSON size per request.
        client: Chroma client used to query the server's record limit when
            ``batch_size`` is not given.
        on_batch: Optional ``callback(done, total)``.
    """
    if not records:
        return 0

    if batch_size is not None:
        max_records = max(1, min(int(batch_size), 5_000))
    elif client is not None:
        max_records = server_batch_size(client)
    else:
        max_records = DEFAULT_BATCH_SIZE

    batches = plan_batches(
        records, max_records=max_records, max_payload_bytes=max_payload_bytes
    )
    written = 0
    done = 0

    # A stack of batches still to write; oversized ones get pushed back split.
    pending: list[Sequence[ChunkRecord]] = list(batches)
    queue = list(reversed(pending))

    while queue:
        batch = queue.pop()
        try:
            collection.upsert(
                ids=[r.id for r in batch],
                embeddings=[r.vector for r in batch],
                documents=[r.text for r in batch],
                metadatas=[r.metadata() for r in batch],
            )
        except Exception as exc:  # noqa: BLE001
            if _is_payload_error(exc) and len(batch) > 1:
                # Halve and retry rather than losing the whole run.
                mid = len(batch) // 2
                queue.append(batch[mid:])
                queue.append(batch[:mid])
                continue
            raise VectorStoreError(
                f"Failed writing {written}/{len(records)} records to Chroma "
                f"after {done} successful batches: {exc}"
            ) from exc

        written += len(batch)
        done += 1
        if on_batch is not None:
            on_batch(written, len(records))

    return written


def delete_by_doc_id(collection: Collection, doc_id: str) -> int:
    """Remove every chunk belonging to one document."""
    try:
        existing = collection.get(where={META_DOC_ID: doc_id}, include=[])
        count = len(existing.get("ids", []))
        if count:
            collection.delete(where={META_DOC_ID: doc_id})
        return count
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Could not delete document {doc_id}: {exc}") from exc


def delete_by_source(collection: Collection, source: str) -> int:
    """Remove every chunk belonging to one source filename."""
    try:
        existing = collection.get(where={META_SOURCE: source}, include=[])
        count = len(existing.get("ids", []))
        if count:
            collection.delete(where={META_SOURCE: source})
        return count
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Could not delete source '{source}': {exc}") from exc


def list_documents(collection: Collection) -> list[DocumentSummary]:
    """Summarise the collection, one entry per document."""
    try:
        result = collection.get(include=["metadatas"])
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Could not read the collection: {exc}") from exc

    grouped: dict[str, DocumentSummary] = {}
    for meta in result.get("metadatas") or []:
        if not meta:
            continue
        source = str(meta.get(META_SOURCE, "unknown"))
        summary = grouped.get(source)
        if summary is None:
            summary = DocumentSummary(
                source=source,
                doc_id=str(meta.get(META_DOC_ID, "")),
                chunks=0,
                content_hash=str(meta.get(META_CONTENT_HASH, "")),
            )
            grouped[source] = summary
        summary.chunks += 1
        member = meta.get(META_MEMBER_PATH)
        if member and member not in summary.member_paths:
            summary.member_paths.append(str(member))

    return sorted(grouped.values(), key=lambda s: s.source.lower())


def stored_hashes(collection: Collection) -> dict[str, str]:
    """Map ``source -> content_hash`` for change detection."""
    try:
        result = collection.get(include=["metadatas"])
    except Exception:  # noqa: BLE001
        return {}

    hashes: dict[str, str] = {}
    for meta in result.get("metadatas") or []:
        if not meta:
            continue
        source = str(meta.get(META_SOURCE, ""))
        if source:
            hashes[source] = str(meta.get(META_CONTENT_HASH, ""))
    return hashes


def search(
    collection: Collection,
    vector: Sequence[float],
    *,
    top_k: int,
) -> list[SearchHit]:
    """Retrieve the ``top_k`` nearest passages."""
    try:
        result = collection.query(
            query_embeddings=[list(vector)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(
            f"Vector search failed. This usually means the stored vectors and "
            f"your query vector come from different embedding models, or the "
            f"collection is empty. ({exc})"
        ) from exc

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[SearchHit] = []
    for i, text in enumerate(documents):
        meta = metadatas[i] if i < len(metadatas) else {}
        dist = float(distances[i]) if i < len(distances) else 1.0
        hits.append(SearchHit(text=str(text), metadata=dict(meta or {}), distance=dist))

    return hits


def count(collection: Collection) -> int:
    try:
        return int(collection.count())
    except Exception:  # noqa: BLE001
        return 0


def reset_collection(client: ClientAPI, settings: Settings | None = None) -> None:
    """Drop the collection so it can be rebuilt from scratch."""
    settings = settings or get_settings()
    try:
        client.delete_collection(name=settings.chroma_collection)
    except Exception:  # noqa: BLE001 - absent collection is fine
        pass


def iter_batches(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]