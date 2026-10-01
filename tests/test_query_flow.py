"""End-to-end query flow through the Streamlit app.

Regression cover for a wiring bug that made every question fail with
``AttributeError: 'Client' object has no attribute 'embed'``.

``require_ready`` returned the *Chroma* client where the app expected the
*OpenRouter* client, so ``retrieve()`` called ``.embed()`` on a vector-store
object. The existing tests missed it because running a Streamlit script in a
test harness never submits anything: ``chat_input`` yields ``None``, so the
entire query branch stayed unexecuted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# AppTest resolves a relative script path against the *test file*, not the
# repository root, so the entry point must be referenced absolutely.
QUERY_APP = Path(__file__).resolve().parents[1] / "app" / "query_app.py"

from app.config import get_settings
from app.openrouter import OpenRouterClient
from app.rag import NoDocumentsIndexedError, retrieve
from app.vectorstore import connect, count, open_collection
from tests.conftest import FakeCollection


class TestClientWiringContract:
    """The two clients have different methods and must not be swapped."""

    def test_router_exposes_the_openrouter_methods(self) -> None:
        from app.ui import get_client

        router = get_client(get_settings())
        assert hasattr(router, "embed")
        assert hasattr(router, "embed_batched")
        assert hasattr(router, "stream_chat")
        assert hasattr(router, "complete")

    def test_vector_store_client_does_not_look_like_a_router(
        self, fake_collection: FakeCollection
    ) -> None:
        # Documents *why* the two must be kept distinct: the Chroma client has
        # no embed method, which is exactly what broke.
        assert not hasattr(fake_collection, "embed")
        assert not hasattr(fake_collection, "stream_chat")

    def test_get_collection_returns_the_chroma_pair(
        self, monkeypatch
    ) -> None:
        import app.ui as ui

        monkeypatch.setattr(ui, "connect", lambda settings: "chroma-client")
        monkeypatch.setattr(
            ui, "open_collection", lambda client, settings, create=True: "collection"
        )
        chroma_client, collection = ui.get_collection(get_settings())
        assert chroma_client == "chroma-client"
        assert collection == "collection"

    def test_require_ready_returns_an_openrouter_client(self, monkeypatch) -> None:
        import app.ui as ui

        class FakeCollectionObj:
            metadata = {}
            def count(self):
                return 7

        monkeypatch.setattr(ui, "connect", lambda settings: "chroma-client")
        monkeypatch.setattr(
            ui, "open_collection",
            lambda client, settings, create=True: FakeCollectionObj(),
        )
        monkeypatch.setattr(ui, "dimension_warning", lambda *a: None)
        monkeypatch.setattr(ui, "config_warnings", lambda: [])
        monkeypatch.setattr(
            ui, "get_client", lambda settings: OpenRouterClient(settings)
        )

        ready = ui.require_ready(get_settings(), role="Query")
        assert ready is not None
        router, _collection = ready
        assert isinstance(router, OpenRouterClient)
        assert hasattr(router, "embed")


class TestRetrieveRejectsWrongClient:
    """retrieve() must be handed something with .embed()."""

    def test_passing_a_vector_store_client_fails_loudly(
        self, fake_collection: FakeCollection
    ) -> None:
        # Needs at least one record, otherwise retrieve() short-circuits on the
        # empty-collection check before ever calling .embed().
        fake_collection.upsert(
            ids=["d:0"],
            embeddings=[[1.0]],
            documents=["text"],
            metadatas=[{"source": "s.md", "doc_id": "d", "chunk_index": 0,
                        "content_hash": "d"}],
        )
        with pytest.raises(AttributeError):
            # Documents the original failure mode. If a future refactor swaps
            # the clients again, this still catches it at the call site.
            retrieve("anything", fake_collection, fake_collection)  # type: ignore[arg-type]


class TestQueryAppSubmitsAQuestion:
    """Drive the real script with a question actually entered."""

    def test_asking_a_question_produces_an_answer(self, monkeypatch) -> None:
        pytest.importorskip("streamlit.testing.v1")
        from streamlit.testing.v1 import AppTest

        import app.ui as ui
        from app.vectorstore import SearchHit

        collection = FakeCollection("rag_docs")
        collection.upsert(
            ids=["d:0"],
            embeddings=[[1.0, 0.0, 0.0]],
            documents=["Use IMAGE_INSTALL:append to add a package."],
            metadatas=[{"source": "guide.md", "doc_id": "d", "chunk_index": 0,
                        "content_hash": "d"}],
        )

        # Keep a handle on the instance the app actually uses, so its recorded
        # prompt can be inspected afterwards.
        created: list[StubRouter] = []

        class StubRouter(OpenRouterClient):
            def __init__(self, settings=None):
                super().__init__(settings or get_settings())
                self.prompts: list = []
                created.append(self)

            def embed(self, texts):
                return [[1.0, 0.0, 0.0] for _ in texts]

            def stream_chat(self, messages, **kwargs):
                self.prompts.append(messages)
                yield "Add it with ", ""
                yield "IMAGE_INSTALL:append", ""

        monkeypatch.setattr(ui, "get_client", lambda settings: StubRouter(settings))
        monkeypatch.setattr(ui, "get_collection", lambda settings, create=True: ("c", collection))
        monkeypatch.setattr(ui, "dimension_warning", lambda *a: None)
        monkeypatch.setattr(ui, "config_warnings", lambda: [])
        monkeypatch.setattr(ui, "count", lambda c: 1)

        app = AppTest.from_file(str(QUERY_APP), default_timeout=120)
        app.run()

        assert not app.exception, [str(e.value) for e in app.exception]

        # Enter a question and run again -- the step the old tests never took.
        app.chat_input[0].set_value("How do I add a package?")
        app.run()

        assert not app.exception, (
            "asking a question raised: "
            + str([str(e.value) for e in app.exception])
        )
        assert not app.error, [e.value for e in app.error]

        chat = [m.value for m in app.markdown if "IMAGE_INSTALL" in (m.value or "")]
        assert chat, "the answer text was not rendered"

        # Retrieval must have run and its passages must be in the prompt, which
        # is what proves the right client was used for the right job.
        assert created, "the app never asked for a router"
        prompts = created[-1].prompts
        assert prompts, "stream_chat was never called"
        sent = "\n".join(m["content"] for m in prompts[-1])
        assert "Passage 1" in sent, "retrieved passages missing from the prompt"
        assert "IMAGE_INSTALL:append" in sent, "passage text missing from prompt"


class TestEmptyCollectionMessage:
    def test_retrieve_on_empty_collection_raises_clearly(
        self, fake_collection: FakeCollection
    ) -> None:
        from app.openrouter import OpenRouterClient

        with pytest.raises(NoDocumentsIndexedError):
            retrieve("q", fake_collection, OpenRouterClient())