"""Indexing pipeline behaviour.

Uses the in-memory doubles from conftest, so these tests exercise real logic
without touching OpenRouter or a live Chroma server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.indexing import (
    IndexReport,
    chunk_id,
    content_hash,
    index_batch,
    index_file,
    rescan_directory,
)


@pytest.fixture
def doc(tmp_path: Path) -> Path:
    path = tmp_path / "documents" / "guide.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Network Setup\n\n## Static IP\n\n"
        "Set a static address with ifconfig em0 inet 10.0.0.5. "
        + ("Extra detail about the interface. " * 40)
        + "\n"
    )
    return path


class TestDeterminism:
    def test_content_hash_is_stable_and_short(self) -> None:
        a = content_hash(b"hello")
        b = content_hash(b"hello")
        assert a == b
        assert len(a) == 16
        assert content_hash(b"other") != a

    def test_chunk_ids_are_derived_from_content(self) -> None:
        digest = content_hash(b"hello")
        assert chunk_id(digest, 0) == f"{digest}:0"
        assert chunk_id(digest, 3) != chunk_id(digest, 0)

    def test_chunk_ids_contain_no_chroma_reserved_characters(self) -> None:
        # Chroma ids must not include the separator it forbids.
        cid = chunk_id(content_hash(b"x" * 100), 7)
        assert ";" not in cid and ".." not in cid and "\x00" not in cid


class TestIndexing:
    def test_indexing_writes_chunks(
        self, doc: Path, fake_collection, fake_router
    ) -> None:
        outcome = index_file(doc, fake_collection, fake_router)
        assert outcome.status == "indexed"
        assert outcome.chunks_written > 0
        assert fake_collection.count() == outcome.chunks_written

    def test_reindexing_the_same_content_is_a_no_op(
        self, doc: Path, fake_collection, fake_router
    ) -> None:
        index_file(doc, fake_collection, fake_router)
        count_after_first = fake_collection.count()

        second = index_file(
            doc,
            fake_collection,
            fake_router,
            existing_hashes={"guide.md": content_hash(doc.read_bytes())},
        )
        assert second.status == "unchanged"
        assert fake_collection.count() == count_after_first
        assert len(fake_router.calls) == 1, "no API call should be made"

    def test_reindexing_does_not_duplicate_vectors(
        self, doc: Path, fake_collection, fake_router
    ) -> None:
        # The original project re-ingested the whole directory on every click,
        # doubling the collection. Deterministic ids prevent that.
        index_file(doc, fake_collection, fake_router)
        first = fake_collection.count()
        for _ in range(3):
            index_file(doc, fake_collection, fake_router, force=True)
        assert fake_collection.count() == first

    def test_changed_content_replaces_old_chunks(
        self, doc: Path, fake_collection, fake_router
    ) -> None:
        index_file(doc, fake_collection, fake_router)
        original = fake_collection.count()

        doc.write_text("# New Content\n\n" + "Entirely different text. " * 60 + "\n")
        index_file(doc, fake_collection, fake_router, force=True)

        # Old chunks for that source must be gone, not merely supplemented.
        remaining = fake_collection.get(where={"source": "guide.md"})
        assert len(remaining["ids"]) > 0
        texts = " ".join(fake_collection.get(include=["documents"])["documents"])
        assert "Entirely different" in texts
        assert "ifconfig em0" not in texts
        assert fake_collection.count() != original or original > 0

    def test_force_overrides_change_detection(
        self, doc: Path, fake_collection, fake_router
    ) -> None:
        hashes = {"guide.md": content_hash(doc.read_bytes())}
        outcome = index_file(
            doc, fake_collection, fake_router, existing_hashes=hashes, force=True
        )
        assert outcome.status == "indexed"

    def test_metadata_carries_provenance(
        self, doc: Path, fake_collection, fake_router
    ) -> None:
        index_file(doc, fake_collection, fake_router)
        rows = fake_collection.get(include=["metadatas"])
        meta = rows["metadatas"][0]
        assert meta["source"] == "guide.md"
        assert len(meta["doc_id"]) == 16
        assert meta["chunk_index"] == 0
        assert meta["content_hash"] == content_hash(doc.read_bytes())

    def test_chunks_reference_their_source_in_the_text(
        self, doc: Path, fake_collection, fake_router
    ) -> None:
        # The LLM sees only chunk text, so provenance has to be inline too.
        index_file(doc, fake_collection, fake_router)
        texts = fake_collection.get(include=["documents"])["documents"]
        assert all("guide.md" in t for t in texts)

    def test_no_chunk_exceeds_the_model_ceiling(
        self, doc: Path, fake_collection, fake_router, isolated_config
    ) -> None:
        index_file(doc, fake_collection, fake_router)
        limit = isolated_config.max_chunk_chars
        texts = fake_collection.get(include=["documents"])["documents"]
        assert all(len(t) <= limit for t in texts)

    def test_one_bad_file_does_not_abort_the_batch(
        self, tmp_path: Path, fake_collection, fake_router
    ) -> None:
        docs = tmp_path / "documents"
        docs.mkdir()
        (docs / "good.md").write_text("# Good\n\n" + "Valid content here. " * 40)
        (docs / "bad.zip").write_bytes(b"PK\x03\x04 definitely not a zip")

        report = index_batch(
            sorted(docs.iterdir()), fake_collection, fake_router
        )
        assert report.failed == 1
        assert report.indexed == 1
        assert fake_collection.count() > 0

    def test_empty_file_is_skipped_not_failed(
        self, tmp_path: Path, fake_collection, fake_router
    ) -> None:
        path = tmp_path / "empty.md"
        path.write_text("   ")
        outcome = index_file(path, fake_collection, fake_router)
        assert outcome.status == "skipped"
        assert fake_collection.count() == 0

    def test_missing_file_is_reported_not_raised(
        self, tmp_path: Path, fake_collection, fake_router
    ) -> None:
        outcome = index_file(
            tmp_path / "gone.md", fake_collection, fake_router
        )
        assert outcome.status == "failed"
        assert "cannot read" in outcome.detail


class TestBatchAndRescan:
    def test_report_counts_are_consistent(
        self, tmp_path: Path, fake_collection, fake_router
    ) -> None:
        docs = tmp_path / "documents"
        docs.mkdir()
        for i in range(3):
            (docs / f"d{i}.md").write_text(f"# Doc {i}\n\n" + f"Body {i}. " * 50)

        report = rescan_directory(docs, fake_collection, fake_router)
        assert isinstance(report, IndexReport)
        assert report.indexed == 3
        assert report.failed == 0
        assert report.total_chunks == fake_collection.count()

    def test_rescan_ignores_unsupported_extensions(
        self, tmp_path: Path, fake_collection, fake_router
    ) -> None:
        docs = tmp_path / "documents"
        docs.mkdir()
        (docs / "keep.md").write_text("# Keep\n\n" + "Body. " * 60)
        (docs / "ignore.bin").write_bytes(b"\x00\x01\x02")
        (docs / ".hidden.md").write_text("# Hidden\n\nnope")

        report = rescan_directory(docs, fake_collection, fake_router)
        assert report.indexed == 1

    def test_rescan_on_missing_directory_is_empty(
        self, tmp_path: Path, fake_collection, fake_router
    ) -> None:
        report = rescan_directory(
            tmp_path / "nope", fake_collection, fake_router
        )
        assert report.outcomes == []

    def test_duplicate_paths_in_one_batch_do_not_double_index(
        self, doc: Path, fake_collection, fake_router
    ) -> None:
        report = index_batch([doc, doc], fake_collection, fake_router)
        assert report.indexed == 1
        assert report.unchanged == 1
        rows = fake_collection.get(where={"source": "guide.md"})
        assert len({r for r in rows["ids"]}) == len(rows["ids"])