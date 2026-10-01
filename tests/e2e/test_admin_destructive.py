"""Destructive index operations, driven through the real admin UI.

These had no coverage at all before. ``reset_collection`` -- the "Delete the
entire collection and start over" button -- was wired to one click with no test
anywhere, on an app with no authentication.

Every test here drives Chrome against the real Streamlit app against a real
Chroma server, so it exercises the button wiring, the Streamlit session and
the Chroma round trip together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.conftest import (
    assert_no_app_error,
    button,
    click,
    collection_count,
    make_documents,
    upload,
    wait_for_app,
    wait_for_idle,
)


def index_documents(page, paths: list[Path]) -> None:
    """Upload documents and drive them through the index, like a user would."""
    wait_for_app(page)
    upload(page, paths)
    click(page, "Index the uploaded files")
    wait_for_app(page)
    assert_no_app_error(page)


@pytest.fixture
def indexed(page_factory, e2e_stack):
    """An admin page with three documents already indexed."""
    docs = make_documents(e2e_stack["docs_dir"])
    page = page_factory(e2e_stack["admin_url"])
    index_documents(page, docs)
    yield page, e2e_stack, docs
    page.close()


class TestIndexAndList:
    def test_documents_reach_the_collection(self, indexed) -> None:
        page, stack, docs = indexed
        wait_for_app(page)
        body = page.inner_text("body")

        chroma = stack["docs_dir"].parent / "chroma"
        assert collection_count(chroma, stack["collection"]) > 0, \
            "nothing was written to the collection"

        for name in ("network.md", "packages.md", "booting.md"):
            assert name in body, f"{name} missing from the document list"

    def test_indexed_documents_are_listed_with_chunk_counts(self, indexed) -> None:
        page, _stack, _docs = indexed
        wait_for_app(page)
        page.get_by_role("tab", name="Indexed documents").click()
        text = page.inner_text("body")
        assert "network.md" in text
        assert "chunks" in text.lower() or "chunk(s)" in text.lower()

    def test_reindex_is_idempotent(self, indexed) -> None:
        """Repeated scans must not keep growing the collection.

        Measured as "scan, snapshot, scan again, compare" rather than a single
        before/after pair, because the documents folder is shared across the
        module and an earlier scan may legitimately pick up a file another test
        added. What must never happen is a second scan adding more chunks.
        """
        page, stack, _docs = indexed
        chroma = stack["docs_dir"].parent / "chroma"

        wait_for_app(page)
        click(page, "Re-scan the documents folder")
        wait_for_idle(page, settle_ms=2500)
        assert_no_app_error(page)
        after_first = collection_count(chroma, stack["collection"])
        assert after_first > 0

        click(page, "Re-scan the documents folder")
        wait_for_idle(page, settle_ms=2500)
        assert_no_app_error(page)
        after_second = collection_count(chroma, stack["collection"])

        assert after_second == after_first, (
            f"a second scan changed the collection: {after_first} -> "
            f"{after_second}. Unchanged files must be skipped."
        )


class TestDeleteOneDocument:
    def test_deleting_one_leaves_the_others(
        self, page_factory, e2e_stack
    ) -> None:
        docs = make_documents(e2e_stack["docs_dir"])
        page = page_factory(e2e_stack["admin_url"])
        chroma = e2e_stack["docs_dir"].parent / "chroma"

        index_documents(page, docs)
        total = collection_count(chroma, e2e_stack["collection"])
        assert total > 0

        wait_for_app(page)
        page.get_by_role("tab", name="Indexed documents").click()
        wait_for_app(page)

        # The selectbox offers each indexed document; pick network.md.
        box = page.locator('[data-testid="stSelectbox"]').first
        box.click()
        page.get_by_role("option", name="network.md").first.click()
        wait_for_idle(page, settle_ms=1500)

        # The delete button names the chosen document, which confirms the
        # selection actually reached the app.
        delete = page.get_by_role(
            "button", name="Delete", exact=False
        ).first
        assert "network.md" in delete.inner_text(), (
            f"delete button does not mention the selection: "
            f"{delete.inner_text()!r}"
        )
        delete.click()
        wait_for_idle(page, settle_ms=2500)
        assert_no_app_error(page)

        remaining = collection_count(chroma, e2e_stack["collection"])
        assert 0 < remaining < total, (
            f"expected a partial delete, went {total} -> {remaining}"
        )

        # And the deleted document must be gone from the offered choices.
        page.get_by_role("tab", name="Indexed documents").click()
        wait_for_idle(page, settle_ms=1500)
        page.locator('[data-testid="stSelectbox"]').first.click()
        options = page.get_by_role("option").all_inner_texts()
        assert "network.md" not in options, (
            f"deleted document still offered: {options}"
        )
        page.close()

    def test_source_file_survives_deletion(self, page_factory, e2e_stack) -> None:
        """Deleting from the index must not delete the uploaded document."""
        docs = make_documents(e2e_stack["docs_dir"])
        page = page_factory(e2e_stack["admin_url"])

        index_documents(page, docs)
        wait_for_app(page)
        page.get_by_role("tab", name="Indexed documents").click()
        wait_for_app(page)
        page.locator('[data-testid="stSelectbox"]').first.click()
        page.get_by_role("option", name="booting.md").first.click()
        wait_for_idle(page, settle_ms=1500)
        click(page, "Delete")
        wait_for_idle(page, settle_ms=2500)

        assert (e2e_stack["docs_dir"] / "booting.md").exists(), (
            "the source document was deleted from disk"
        )
        page.close()


class TestDeleteEntireCollection:
    def test_button_empties_the_collection(self, indexed) -> None:
        page, stack, _docs = indexed
        chroma = stack["docs_dir"].parent / "chroma"
        before = collection_count(chroma, stack["collection"])
        assert before > 0

        wait_for_app(page)
        page.get_by_role("tab", name="Danger zone").click()
        wait_for_app(page)

        # The button must announce what it destroys.
        body = page.inner_text("body")
        assert "cannot be undone" in body.lower()

        click(page, "Delete the entire collection")
        wait_for_app(page)
        assert_no_app_error(page)

        after = collection_count(chroma, stack["collection"])
        assert after == 0, f"collection still holds {after} chunks"
        page.close() if False else None

    def test_reindexing_after_a_wipe_restores_chunks(
        self, indexed
    ) -> None:
        """The destructive path must be recoverable."""
        page, stack, _docs = indexed
        chroma = stack["docs_dir"].parent / "chroma"

        wait_for_app(page)
        page.get_by_role("tab", name="Danger zone").click()
        wait_for_app(page)
        click(page, "Delete the entire collection")
        wait_for_app(page)
        assert collection_count(chroma, stack["collection"]) == 0

        click(page, "Force re-index everything")
        wait_for_app(page)
        assert_no_app_error(page)
        assert collection_count(chroma, stack["collection"]) > 0, (
            "a forced re-index did not rebuild anything"
        )
        page.close()


class TestClearDocumentsFolder:
    def test_button_removes_files_but_leaves_the_index(
        self, indexed
    ) -> None:
        page, stack, _docs = indexed
        chroma = stack["docs_dir"].parent / "chroma"
        indexed_count = collection_count(chroma, stack["collection"])
        assert indexed_count > 0

        wait_for_app(page)
        page.get_by_role("tab", name="Danger zone").click()
        wait_for_app(page)
        click(page, "Clear the documents folder")
        wait_for_app(page)
        assert_no_app_error(page)

        remaining_files = [p for p in stack["docs_dir"].glob("*") if p.is_file()]
        assert not remaining_files, f"files survived: {remaining_files}"
        # The index is untouched, which is the documented behaviour.
        assert collection_count(chroma, stack["collection"]) == indexed_count
        page.close()


class TestFormatSupportPanel:
    def test_supported_formats_are_advertised(self, indexed) -> None:
        page, _stack, _docs = indexed
        wait_for_app(page)
        page.get_by_role("tab", name="Supported formats").click()
        wait_for_app(page)
        body = page.inner_text("body")
        for extension in (".pdf", ".docx", ".md", ".zip"):
            assert extension in body, f"{extension} not listed"
        assert "scanned" in body.lower(), "no OCR caveat shown"

class TestUploaderLifecycle:
    """Regression: the uploader used to re-save on every rerun.

    That made "Clear the documents folder" a silent no-op -- files were deleted
    and immediately rewritten by the next rerun -- and it re-saved the same batch
    on every single interaction for as long as the tab stayed open.
    """

    def test_clearing_the_folder_actually_clears_it(
        self, page_factory, e2e_stack
    ) -> None:
        docs = make_documents(e2e_stack["docs_dir"])
        page = page_factory(e2e_stack["admin_url"])
        index_documents(page, docs)
        assert list(e2e_stack["docs_dir"].glob("*.md")), "upload never landed"

        wait_for_app(page)
        page.get_by_role("tab", name="Danger zone").click()
        wait_for_app(page)
        click(page, "Clear the documents folder")
        wait_for_idle(page, settle_ms=2000)

        survivors = [p.name for p in e2e_stack["docs_dir"].glob("*") if p.is_file()]
        assert not survivors, (
            f"files came back after clearing: {survivors}. The uploader is "
            f"re-saving them on rerun."
        )
        page.close()

    def test_a_tab_switch_does_not_resave_uploads(
        self, page_factory, e2e_stack
    ) -> None:
        """Uploads are written once, not on every interaction."""
        docs = make_documents(e2e_stack["docs_dir"])
        page = page_factory(e2e_stack["admin_url"])
        index_documents(page, docs)

        before = {
            p.name: p.stat().st_mtime_ns for p in e2e_stack["docs_dir"].glob("*")
        }
        assert before

        for tab in ("Supported formats", "Indexed documents", "Danger zone"):
            page.get_by_role("tab", name=tab).click()
            wait_for_app(page)

        after = {p.name: p.stat().st_mtime_ns for p in e2e_stack["docs_dir"].glob("*")}
        assert after == before, (
            f"files were rewritten by tab switches: "
            f"{ {k: (before.get(k), v) for k, v in after.items()} }"
        )
        page.close()

    def test_index_button_still_works_after_the_uploader_resets(
        self, page_factory, e2e_stack
    ) -> None:
        """Resetting the uploader must not lose the pending index targets."""
        docs = make_documents(e2e_stack["docs_dir"])
        chroma = e2e_stack["docs_dir"].parent / "chroma"
        page = page_factory(e2e_stack["admin_url"])

        wait_for_app(page)
        upload(page, docs)
        # After the internal reset/rerun, the button must still be enabled.
        click(page, "Index the uploaded files")
        wait_for_app(page)
        assert_no_app_error(page)
        assert collection_count(chroma, e2e_stack["collection"]) > 0
        page.close()


class TestStateRefreshAfterActions:
    """Regression: the page reported pre-action state.

    ``documents`` and the chunk total were read at the top of the script, before
    the indexing button was handled. Streamlit renders tab content from the
    final run and cannot redraw what it already emitted, so a successful index
    of three documents still left "Indexed documents" saying "Nothing is
    indexed yet" and the metrics showing zero.
    """

    def test_indexing_updates_the_panel_immediately(
        self, page_factory, e2e_stack
    ) -> None:
        docs = make_documents(e2e_stack["docs_dir"])
        page = page_factory(e2e_stack["admin_url"])

        index_documents(page, docs)

        # No extra interaction: the same run that indexed must already show it.
        page.get_by_role("tab", name="Indexed documents").click()
        wait_for_idle(page, settle_ms=1500)
        text = page.inner_text("body")
        assert "Nothing is indexed yet" not in text, (
            "the panel still reports an empty index straight after indexing"
        )
        for name in ("network.md", "packages.md", "booting.md"):
            assert name in text, f"{name} missing immediately after indexing"
        page.close()

    def test_metrics_are_not_zero_after_indexing(
        self, page_factory, e2e_stack
    ) -> None:
        docs = make_documents(e2e_stack["docs_dir"])
        page = page_factory(e2e_stack["admin_url"])
        index_documents(page, docs)
        body = page.inner_text("body")
        assert "Indexed chunks" in body
        # The metric must not read zero.
        zero_reading = page.locator(
            '[data-testid="stMetricValue"]:has-text("0")'
        )
        assert zero_reading.count() == 0, (
            "a metric still reads zero immediately after a successful index"
        )
        page.close()

    def test_delete_updates_the_panel_immediately(self, indexed) -> None:
        page, _stack, _docs = indexed
        wait_for_app(page)
        page.get_by_role("tab", name="Danger zone").click()
        wait_for_app(page)
        click(page, "Delete the entire collection")
        wait_for_idle(page, settle_ms=2500)

        page.get_by_role("tab", name="Indexed documents").click()
        wait_for_idle(page, settle_ms=1500)
        text = page.inner_text("body")
        assert "Nothing is indexed yet" in text, (
            "the panel still lists documents after the collection was deleted"
        )
        page.close()
