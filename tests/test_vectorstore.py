"""Vector store helpers and the retrieval prompt.

The Chroma HTTP calls need a live server, so these tests use the in-memory
FakeCollection for anything touching the collection API, and cover the pure
helpers directly.
"""

from __future__ import annotations

import pytest

from app.rag import build_messages, format_context
from app.vectorstore import (
    ChunkRecord,
    Collection,
    DocumentSummary,
    SearchHit,
    count,
    delete_by_doc_id,
    delete_by_source,
    dimension_warning,
    list_documents,
    search,
    server_batch_size,
    stored_hashes,
    upsert_chunks,
)


def record(**overrides) -> ChunkRecord:
    base = dict(
        id="doc:0",
        text="Some passage text.",
        vector=[0.1, 0.2, 0.3],
        source="guide.md",
        doc_id="doc",
        chunk_index=0,
        content_hash="doc",
    )
    base.update(overrides)
    return ChunkRecord(**base)


class TestChunkRecordMetadata:
    def test_required_keys_always_present(self) -> None:
        meta = record().metadata()
        for key in ("source", "doc_id", "chunk_index", "content_hash"):
            assert key in meta

    def test_empty_optionals_are_dropped(self) -> None:
        # Empty strings would clutter Chroma filters.
        meta = record(heading="", source_zip="", member_path="").metadata()
        assert "heading" not in meta
        assert "source_zip" not in meta

    def test_optionals_are_included_when_set(self) -> None:
        meta = record(heading="Setup", member_path="a/b.md").metadata()
        assert meta["heading"] == "Setup"
        assert meta["member_path"] == "a/b.md"

    def test_metadata_values_are_chroma_safe(self) -> None:
        # Chroma metadata accepts only str/int/float/bool.
        for value in record(heading="x").metadata().values():
            assert isinstance(value, (str, int, float, bool))


class TestWritesAndDeletes:
    def test_upsert_writes_all_records(self, fake_collection: Collection) -> None:
        records = [record(id=f"doc:{i}", chunk_index=i) for i in range(5)]
        assert upsert_chunks(fake_collection, records) == 5
        assert count(fake_collection) == 5

    def test_upsert_respects_batch_size(self, fake_collection: Collection) -> None:
        records = [record(id=f"doc:{i}", chunk_index=i) for i in range(7)]
        upsert_chunks(fake_collection, records, batch_size=3)
        assert len(fake_collection.upsert_batches) == 3
        assert sum(fake_collection.upsert_batches) == 7

    def test_upsert_of_nothing_is_a_no_op(
        self, fake_collection: Collection
    ) -> None:
        assert upsert_chunks(fake_collection, []) == 0

    def test_batch_size_of_zero_is_corrected(self) -> None:
        class FakeClient:
            def get_max_batch_size(self):
                return 0

        # A server reporting zero must not cause an infinite loop.
        assert server_batch_size(FakeClient()) >= 1

    def test_delete_by_source_removes_only_that_source(
        self, fake_collection: Collection
    ) -> None:
        fake_collection.upsert(
            ids=["a:0", "a:1", "b:0"],
            embeddings=[[1.0], [1.0], [1.0]],
            documents=["a", "a", "b"],
            metadatas=[
                {"source": "a.md", "doc_id": "a", "chunk_index": 0,
                 "content_hash": "a"},
                {"source": "a.md", "doc_id": "a", "chunk_index": 1,
                 "content_hash": "a"},
                {"source": "b.md", "doc_id": "b", "chunk_index": 0,
                 "content_hash": "b"},
            ],
        )
        assert delete_by_source(fake_collection, "a.md") == 2
        assert count(fake_collection) == 1

    def test_delete_by_doc_id(self, fake_collection: Collection) -> None:
        fake_collection.upsert(
            ids=["a:0"],
            embeddings=[[1.0]],
            documents=["a"],
            metadatas=[{"source": "a.md", "doc_id": "a", "chunk_index": 0,
                        "content_hash": "a"}],
        )
        assert delete_by_doc_id(fake_collection, "a") == 1
        assert count(fake_collection) == 0

    def test_deleting_a_missing_source_is_harmless(
        self, fake_collection: Collection
    ) -> None:
        assert delete_by_source(fake_collection, "never.md") == 0


class TestListingAndSearch:
    @pytest.fixture
    def populated(self, fake_collection: Collection) -> Collection:
        fake_collection.upsert(
            ids=["a:0", "a:1", "zip:0"],
            embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            documents=["first", "second", "third"],
            metadatas=[
                {"source": "a.md", "doc_id": "a", "chunk_index": 0,
                 "content_hash": "ha"},
                {"source": "a.md", "doc_id": "a", "chunk_index": 1,
                 "content_hash": "ha"},
                {"source": "b.zip", "doc_id": "b", "chunk_index": 0,
                 "content_hash": "hb", "member_path": "inner.md"},
            ],
        )
        return fake_collection

    def test_documents_are_grouped_by_source(
        self, populated: Collection
    ) -> None:
        docs = list_documents(populated)
        assert len(docs) == 2
        a = next(d for d in docs if d.source == "a.md")
        assert a.chunks == 2
        assert a.content_hash == "ha"

    def test_archive_members_are_shown_in_the_label(
        self, populated: Collection
    ) -> None:
        b = next(d for d in list_documents(populated) if d.source == "b.zip")
        assert "inner.md" in b.label

    def test_stored_hashes_map_source_to_hash(
        self, populated: Collection
    ) -> None:
        assert stored_hashes(populated) == {"a.md": "ha", "b.zip": "hb"}

    def test_search_ranks_nearest_first(self, populated: Collection) -> None:
        hits = search(populated, [1.0, 0.0], top_k=2)
        assert len(hits) == 2
        assert hits[0].text == "first"
        assert hits[0].score >= hits[1].score

    def test_search_score_is_a_similarity(
        self, populated: Collection
    ) -> None:
        for hit in search(populated, [1.0, 0.0], top_k=3):
            assert 0.0 <= hit.score <= 1.0

    def test_search_respects_top_k(self, populated: Collection) -> None:
        assert len(search(populated, [1.0, 0.0], top_k=1)) == 1

    def test_search_hit_exposes_source_and_heading(self) -> None:
        hit = SearchHit(
            text="t", metadata={"source": "s.md", "heading": "H"}, distance=0.25
        )
        assert hit.source == "s.md"
        assert hit.heading == "H"
        assert hit.score == pytest.approx(0.75)


class TestDimensionWarning:
    def test_matching_model_produces_no_warning(
        self, isolated_config, fake_collection: Collection
    ) -> None:
        fake_collection.metadata = {
            "embedding_model": isolated_config.embed_model,
            "embedding_dims": isolated_config.embed_dims,
        }
        assert dimension_warning(fake_collection, isolated_config) is None

    def test_different_model_is_flagged(self, isolated_config) -> None:
        class Coll:
            metadata = {"embedding_model": "some/other-model"}

        warning = dimension_warning(Coll(), isolated_config)
        assert warning is not None
        assert "re-index" in warning

    def test_different_dimensions_are_flagged(
        self, isolated_config
    ) -> None:
        class Coll:
            metadata = {
                "embedding_model": isolated_config.embed_model,
                "embedding_dims": isolated_config.embed_dims + 1,
            }

        assert dimension_warning(Coll(), isolated_config) is not None

    def test_missing_metadata_is_not_an_error(
        self, isolated_config
    ) -> None:
        class Coll:
            metadata = None

        assert dimension_warning(Coll(), isolated_config) is None


class TestPromptAssembly:
    def make_hit(self, text: str, source: str = "a.md", heading: str = "") -> SearchHit:
        return SearchHit(
            text=text, metadata={"source": source, "heading": heading},
            distance=0.2,
        )

    def test_passages_are_numbered_and_cited_in_order(self) -> None:
        context = format_context(
            [self.make_hit("first body"), self.make_hit("second body", "b.md")]
        )
        assert "Passage 1" in context
        assert "Passage 2" in context
        assert context.index("first body") < context.index("second body")

    def test_heading_appears_in_the_passage_label(self) -> None:
        context = format_context([self.make_hit("x", heading="A > B")])
        assert "a.md :: A > B" in context

    def test_messages_request_grounded_answering(self) -> None:
        messages = build_messages("How?", [self.make_hit("body")])
        assert messages[0]["role"] == "system"
        assert "ONLY" in messages[0]["content"]
        assert "Passage 1" in messages[1]["content"]
        assert "How?" in messages[1]["content"]

    def test_no_hits_produces_an_explicit_no_context_instruction(self) -> None:
        messages = build_messages("How?", [])
        assert "No documents were retrieved" in messages[1]["content"]

    def test_history_is_inserted_before_the_question(self) -> None:
        messages = build_messages(
            "New?",
            [self.make_hit("body")],
            history=[{"role": "user", "content": "Old"},
                     {"role": "assistant", "content": "Reply"}],
        )
        assert [m["role"] for m in messages] == [
            "system", "user", "assistant", "user"
        ]

    def test_oversized_context_is_truncated(
        self, monkeypatch
    ) -> None:
        import app.rag as rag_module

        monkeypatch.setattr(rag_module, "MAX_CONTEXT_CHARS", 500)
        hits = [self.make_hit("x" * 400, f"f{i}.md") for i in range(10)]
        context = rag_module.format_context(hits)
        assert len(context) < 1500
        assert "truncated" in context or "Passage 1" in context


class TestDocumentSummary:
    def test_label_without_members_is_the_source(self) -> None:
        assert DocumentSummary(source="a.md", doc_id="a", chunks=1).label == "a.md"

    def test_label_counts_extra_members(self) -> None:
        summary = DocumentSummary(
            source="b.zip", doc_id="b", chunks=4,
            member_paths=["one.md", "two.md", "three.md"],
        )
        assert summary.label == "b.zip/one.md +2 more"