"""Shared fixtures.

No test in this suite is allowed to touch the network or the real vector
store, so anything that would do either is replaced by a local double.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point every path at a temporary directory and clear the API key.

    Autouse, so no test can accidentally write into the real data/ folder or
    read the developer's real .env.
    """
    from app.config import reload_settings

    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_DOCS_DIR", str(tmp_path / "data" / "documents"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "data" / "chroma"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key-for-unit-tests")

    # Pinned so assertions about the default configuration do not change
    # depending on what the developer's own .env happens to say. Without this,
    # setting APP_BIND_ADDRESS=0.0.0.0 locally would fail unrelated tests.
    monkeypatch.setenv("APP_BIND_ADDRESS", "127.0.0.1")

    settings = reload_settings()
    settings.ensure_dirs()
    yield settings

    # Individual tests deliberately set invalid values to exercise validation.
    # Those are undone before this runs, so reloading here must not raise.
    try:
        reload_settings()
    except Exception:
        pass


class FakeEmbeddings:
    """Deterministic stand-in for the OpenRouter embeddings endpoint.

    Returns unit-ish vectors derived from a hash of the input so that
    similarity ordering is at least self-consistent across calls.
    """

    def __init__(self, dims: int = 8):
        self.dims = dims
        self.calls: list[list[str]] = []

    def _vector(self, text: str) -> list[float]:
        seed = sum(ord(c) * (i + 1) for i, c in enumerate(text))
        return [((seed * (i + 3)) % 97) / 97.0 for i in range(self.dims)]

    def embed(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        return [self._vector(t) for t in texts]

    def embed_batched(self, texts, *, batch_size=None, on_batch=None):
        texts = list(texts)
        size = batch_size or 4
        out: list[list[float]] = []
        batches = [
            texts[i : i + size] for i in range(0, len(texts), size)
        ]
        for n, batch in enumerate(batches, start=1):
            out.extend(self.embed(batch))
            if on_batch is not None:
                on_batch(n, len(batches))
        return out


class FakeCollection:
    """In-memory stand-in for a Chroma collection.

    Implements only the surface vectorstore.py uses.
    """

    def __init__(self, name: str = "test"):
        self.name = name
        self.metadata: dict = {}
        self._rows: dict[str, dict] = {}
        self.upsert_batches: list[int] = []

    # -- writes --
    def upsert(self, ids, embeddings, documents, metadatas):
        assert len(ids) == len(embeddings) == len(documents) == len(metadatas)
        assert len(set(ids)) == len(ids), "duplicate ids in one batch"
        self.upsert_batches.append(len(ids))
        for i, doc, vec, meta in zip(ids, documents, embeddings, metadatas):
            self._rows[i] = {
                "document": doc,
                "embedding": vec,
                "metadata": dict(meta),
            }

    def delete(self, where=None):
        if not where:
            self._rows.clear()
            return
        for key, value in where.items():
            for row in list(self._rows.values()):
                if row["metadata"].get(key) != value:
                    continue
                for row_id, row_data in list(self._rows.items()):
                    if row_data is row:
                        del self._rows[row_id]

    # -- reads --
    def count(self) -> int:
        return len(self._rows)

    def get(self, where=None, include=None):
        rows = list(self._rows.items())
        if where:
            rows = [
                (rid, r)
                for rid, r in rows
                if all(r["metadata"].get(k) == v for k, v in where.items())
            ]
        include = include or []
        result: dict = {"ids": [rid for rid, _ in rows]}
        if "metadatas" in include:
            result["metadatas"] = [r["metadata"] for _, r in rows]
        if "documents" in include:
            result["documents"] = [r["document"] for _, r in rows]
        return result

    def query(self, query_embeddings, n_results, include=None):
        import math

        vec = query_embeddings[0]

        def similarity(row):
            other = row["embedding"]
            dot = sum(a * b for a, b in zip(vec, other))
            na = math.sqrt(sum(a * a for a in vec)) or 1.0
            nb = math.sqrt(sum(b * b for b in other)) or 1.0
            return dot / (na * nb)

        ranked = sorted(self._rows.items(), key=lambda kv: similarity(kv[1]),
                        reverse=True)[:n_results]
        return {
            "ids": [[rid for rid, _ in ranked]],
            "documents": [[r["document"] for _, r in ranked]],
            "metadatas": [[r["metadata"] for _, r in ranked]],
            "distances": [[1.0 - similarity(r) for _, r in ranked]],
        }


@pytest.fixture
def fake_collection() -> FakeCollection:
    return FakeCollection()


@pytest.fixture
def fake_router(isolated_config) -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def sample_markdown() -> str:
    return (
        "# Network Setup\n\n"
        "Intro paragraph introducing the section.\n\n"
        "## Static IP\n\n"
        "Set a static address in /etc/hosts. " + ("Filler sentence. " * 60) + "\n\n"
        "```bash\n"
        "ifconfig em0 inet 10.0.0.5 netmask 255.255.255.0\n"
        "route add default 10.0.0.1\n"
        "```\n\n"
        "## DHCP\n\nRun smpd and let it lease an address.\n"
    )