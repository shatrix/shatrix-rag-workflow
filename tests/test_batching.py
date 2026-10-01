"""Batching and payload-size limits.

Regression cover for the failure that stopped a 1426-page PDF from indexing:
Chroma's ``max_batch_size`` is a *record count* limit (5000 on this server),
not a payload limit, so sending every chunk in one request produced a ~146 MB
JSON body and the server answered "Payload too large".
"""

from __future__ import annotations

import json
import random

import pytest

from app.vectorstore import (
    BYTES_PER_FLOAT,
    DEFAULT_MAX_PAYLOAD_BYTES,
    ChunkRecord,
    Collection,
    estimate_record_bytes,
    plan_batches,
    upsert_chunks,
)

DIMS = 2048  # nvidia/nemotron-3-embed-1b


def make_record(i: int, dims: int = DIMS, text_chars: int = 1800) -> ChunkRecord:
    rng = random.Random(i)
    return ChunkRecord(
        id=f"{i:016x}:0",
        text="x" * text_chars,
        vector=[rng.uniform(-0.05, 0.05) for _ in range(dims)],
        source="doc.pdf",
        doc_id="deadbeef",
        chunk_index=i,
        content_hash="deadbeef",
    )


def real_json_bytes(batch) -> int:
    return len(
        json.dumps(
            {
                "ids": [r.id for r in batch],
                "embeddings": [r.vector for r in batch],
                "documents": [r.text for r in batch],
                "metadatas": [r.metadata() for r in batch],
            }
        )
    )


class TestEstimate:
    def test_estimate_is_conservative(self) -> None:
        """The estimate must never be lower than reality.

        Under-estimating is what caused the failure; over-estimating only costs
        a few extra round trips.
        """
        record = make_record(0)
        assert estimate_record_bytes(record) >= real_json_bytes([record])

    def test_bytes_per_float_is_plausible(self) -> None:
        # A repr of a float such as -0.03656357558875988 is 20 characters.
        assert BYTES_PER_FLOAT >= 20

    def test_longer_text_and_more_dimensions_cost_more(self) -> None:
        small = estimate_record_bytes(make_record(0, dims=384, text_chars=200))
        large = estimate_record_bytes(make_record(0, dims=DIMS, text_chars=1800))
        assert large > small


class TestPlanBatches:
    def test_record_count_limit_is_respected(self) -> None:
        records = [make_record(i) for i in range(50)]
        for batch in plan_batches(records, max_records=7):
            assert len(batch) <= 7

    def test_payload_limit_is_respected(self) -> None:
        records = [make_record(i) for i in range(60)]
        for batch in plan_batches(
            records, max_records=5000, max_payload_bytes=2 * 1024 * 1024
        ):
            assert sum(estimate_record_bytes(r) for r in batch) <= (
                2 * 1024 * 1024
            )

    def test_thousands_of_records_produce_many_batches(self) -> None:
        """The regression: 3747 chunks must not become one request."""
        records = [make_record(i) for i in range(3747)]
        batches = plan_batches(
            records,
            max_records=5000,  # what the server reports
            max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES,
        )
        assert len(batches) > 30
        assert sum(len(b) for b in batches) == 3747

    def test_no_batch_exceeds_the_budget_when_really_serialised(self) -> None:
        records = [make_record(i) for i in range(300)]
        for batch in plan_batches(
            records,
            max_records=5000,
            max_payload_bytes=DEFAULT_MAX_PAYLOAD_BYTES,
        ):
            assert real_json_bytes(batch) <= DEFAULT_MAX_PAYLOAD_BYTES

    def test_all_records_are_preserved_in_order(self) -> None:
        records = [make_record(i) for i in range(97)]
        batches = plan_batches(records, max_records=10)
        flat = [r for batch in batches for r in batch]
        assert [r.id for r in flat] == [r.id for r in records]

    def test_empty_input_yields_nothing(self) -> None:
        assert plan_batches([], max_records=10) == []

    def test_single_record_larger_than_budget_still_lands_alone(self) -> None:
        # Truncating here would silently drop content, so an oversized record
        # must be written on its own and left for the retry-split to handle.
        huge = make_record(0, dims=DIMS, text_chars=1800)
        batches = plan_batches(
            [huge], max_records=10, max_payload_bytes=1024
        )
        assert len(batches) == 1
        assert batches[0][0] is huge

    def test_small_dimension_model_needs_fewer_requests(self) -> None:
        big = [make_record(i, dims=2048) for i in range(500)]
        small = [make_record(i, dims=384) for i in range(500)]
        n_big = len(plan_batches(big, max_records=5000))
        n_small = len(plan_batches(small, max_records=5000))
        assert n_small < n_big


class RejectingCollection:
    """Collection double that refuses oversized bodies, like the real server."""

    def __init__(self, limit_bytes: int):
        self.limit_bytes = limit_bytes
        self.calls: list[int] = []
        self.rows: dict[str, ChunkRecord] = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        records = [
            ChunkRecord(
                id=i,
                text=d,
                vector=e,
                source="s",
                doc_id="d",
                chunk_index=0,
                content_hash="d",
            )
            for i, e, d in zip(ids, embeddings, documents)
        ]
        if real_json_bytes(records) > self.limit_bytes:
            from app.vectorstore import VectorStoreError

            raise VectorStoreError(
                "Failed to send request: Payload too large (trace ID: 0)"
            )
        self.calls.append(len(records))
        for r in records:
            self.rows[r.id] = r


class TestUpsertResilience:
    def test_writes_everything_within_limits(
        self, fake_collection: Collection
    ) -> None:
        records = [make_record(i) for i in range(40)]
        written = upsert_chunks(
            fake_collection, records, batch_size=5000
        )
        assert written == 40
        assert fake_collection.count() == 40

    def test_oversized_request_is_split_and_retried(self) -> None:
        """If the server's real limit is lower than we estimated, split."""
        collection = RejectingCollection(limit_bytes=400_000)
        records = [make_record(i) for i in range(60)]
        written = upsert_chunks(
            collection, records, batch_size=5000
        )
        assert written == 60
        assert len(collection.rows) == 60
        assert max(collection.calls) * 47_243 < 2_000_000

    def test_other_errors_are_not_swallowed(self, fake_collection) -> None:
        class Broken:
            def upsert(self, **kwargs):
                from app.vectorstore import VectorStoreError

                raise VectorStoreError("disk on fire")

        with pytest.raises(Exception) as exc:
            upsert_chunks(Broken(), [make_record(0)], batch_size=10)
        assert "disk on fire" in str(exc.value)

    def test_partial_progress_is_reported_in_the_error(
        self, fake_collection
    ) -> None:
        calls = {"n": 0}

        class FailsLate:
            def upsert(self, ids, embeddings, documents, metadatas, **kw):
                calls["n"] += 1
                if calls["n"] >= 3:
                    from app.vectorstore import VectorStoreError

                    raise VectorStoreError("something broke")
                for i in ids:
                    pass

        records = [make_record(i) for i in range(30)]
        with pytest.raises(Exception) as exc:
            upsert_chunks(FailsLate(), records, batch_size=5)
        message = str(exc.value)
        assert "after 2 successful batches" in message

    def test_empty_input_is_a_no_op(self, fake_collection: Collection) -> None:
        assert upsert_chunks(fake_collection, []) == 0
        assert fake_collection.count() == 0

    def test_batch_size_zero_does_not_hang(self, fake_collection) -> None:
        assert upsert_chunks(fake_collection, [make_record(0)], batch_size=0) == 1