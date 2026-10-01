"""Multi-file and archive ingestion, through the real admin UI.

Only ever tested at the function level before. Archives were never pushed
through the whole pipeline by an actual user, which is where the interesting
failures live: attribution of members, unsafe members being skipped without
aborting the batch, and a partial failure not losing the good files.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tests.e2e.conftest import (
    assert_no_app_error,
    collection_count,
    make_documents,
    upload,
    wait_for_app,
    wait_for_idle,
)


def index_through_ui(page, paths: list[Path]) -> None:
    wait_for_app(page)
    upload(page, paths)
    page.get_by_role("button", name="Index the uploaded files", exact=False).first.click()
    wait_for_idle(page, settle_ms=3000)


def chroma_dir(stack) -> Path:
    return stack["docs_dir"].parent / "chroma"


def build_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


class TestMultiFileBatch:
    def test_several_files_index_in_one_go(self, page_factory, e2e_stack) -> None:
        docs = make_documents(e2e_stack["docs_dir"])
        page = page_factory(e2e_stack["admin_url"])
        index_through_ui(page, docs)
        assert_no_app_error(page)

        assert collection_count(chroma_dir(e2e_stack), e2e_stack["collection"]) > 0
        body = page.inner_text("body")
        for name in ("network.md", "packages.md", "booting.md"):
            assert name in body
        page.close()

    def test_one_bad_file_does_not_lose_the_good_ones(
        self, page_factory, e2e_stack
    ) -> None:
        """A corrupt archive must not abort the batch."""
        docs = make_documents(e2e_stack["docs_dir"])
        broken = e2e_stack["docs_dir"] / "broken.zip"
        broken.write_bytes(b"PK\x03\x04 this is not really a zip file")
        page = page_factory(e2e_stack["admin_url"])

        index_through_ui(page, docs + [broken])
        assert_no_app_error(page)

        body = page.inner_text("body")
        assert "failed" in body.lower(), "the failure was not reported"
        # The three good documents are still indexed.
        page.get_by_role("tab", name="Indexed documents").click()
        wait_for_idle(page, settle_ms=1500)
        listed = page.inner_text("body")
        for name in ("network.md", "packages.md", "booting.md"):
            assert name in listed, f"{name} lost when a sibling file failed"
        page.close()


class TestArchiveIngestion:
    @pytest.fixture
    def archive(self, e2e_stack) -> Path:
        return build_zip(
            e2e_stack["docs_dir"] / "bundle.zip",
            {
                "readme.md": "# Bundle\n\nThis archive holds several documents.\n"
                * 4,
                "guides/network.md": "# Network\n\nSet a static address with "
                "ifconfig on the interface. " * 10,
                "guides/packages.md": "# Packages\n\nUse IMAGE_INSTALL:append "
                "to add a package to the image. " * 10,
                "data/table.csv": "name,role\nalpha,smpsched\nbeta,io-spawn\n",
            },
        )

    def test_archive_members_become_separate_documents(
        self, page_factory, e2e_stack, archive
    ) -> None:
        page = page_factory(e2e_stack["admin_url"])
        index_through_ui(page, [archive])
        assert_no_app_error(page)

        page.get_by_role("tab", name="Indexed documents").click()
        wait_for_idle(page, settle_ms=2000)
        body = page.inner_text("body")

        # Every member shares the archive as its source, so the index lists one
        # row per archive and labels it with a member path. Deleting the row
        # therefore removes all of its members at once.
        assert "bundle.zip" in body, body[:800]
        assert collection_count(chroma_dir(e2e_stack), e2e_stack["collection"]) > 0
        page.close()

    def test_members_are_retrievable_individually(
        self, page_factory, e2e_stack, archive
    ) -> None:
        import chromadb

        client = chromadb.HttpClient(host="127.0.0.1", port=18888)
        handle = client.get_collection(
            name=e2e_stack["collection"], embedding_function=None
        )
        rows = handle.get(include=["documents", "metadatas"])
        sources = {m.get("member_path") for m in rows["metadatas"] if m}
        assert any("network.md" in (s or "") for s in sources), (
            f"archive member lost: {sources}"
        )
        # Every member records which archive it came from.
        zips = {m.get("source") for m in rows["metadatas"] if m}
        assert "bundle.zip" in zips
        page = page_factory(e2e_stack["admin_url"])
        page.close()

    def test_unsafe_members_are_skipped_without_aborting(
        self, page_factory, e2e_stack
    ) -> None:
        """Traversal members must be refused, and the rest still indexed."""
        evil = build_zip(
            e2e_stack["docs_dir"] / "evil.zip",
            {
                "../../../tmp/pwned.md": "# Escape\n\nshould never be indexed\n",
                "/etc/shadow": "# Absolute\n\nalso refused\n",
                "good.md": "# Good\n\nThis member is legitimate. " * 12,
            },
        )
        page = page_factory(e2e_stack["admin_url"])
        index_through_ui(page, [evil])
        assert_no_app_error(page)

        page.get_by_role("tab", name="Indexed documents").click()
        wait_for_idle(page, settle_ms=2000)
        body = page.inner_text("body")
        assert "evil.zip" in body, body[:800]
        assert not Path("/tmp/pwned.md").exists(), "traversal wrote outside"

        # The refused members must be absent from the vectors, and the good one
        # present.
        import chromadb

        client = chromadb.HttpClient(host="127.0.0.1", port=18888)
        handle = client.get_collection(
            name=e2e_stack["collection"], embedding_function=None
        )
        rows = handle.get(include=["metadatas"])
        members = {m.get("member_path") for m in rows["metadatas"] if m}
        assert "good.md" in members, f"legitimate member not indexed: {members}"
        assert not any(
            m and ("pwned" in m or "shadow" in m) for m in members
        ), f"unsafe member was indexed: {members}"
        page.close()

    def test_empty_archive_does_not_crash_the_app(
        self, page_factory, e2e_stack
    ) -> None:
        empty = build_zip(e2e_stack["docs_dir"] / "empty.zip", {})
        page = page_factory(e2e_stack["admin_url"])
        index_through_ui(page, [empty])
        assert_no_app_error(page)
        page.close()


class TestLargeDocumentGuidance:
    def test_oversized_file_gets_pointed_at_the_cli(
        self, page_factory, e2e_stack
    ) -> None:
        """The admin app must warn before a long in-session job."""
        big = e2e_stack["docs_dir"] / "big.md"
        big.write_text("# Big\n\n" + "Filler sentence about topics. " * 40_000)
        assert big.stat().st_size > 1024 * 1024, (
            f"fixture too small: {big.stat().st_size}"
        )

        page = page_factory(e2e_stack["admin_url"])
        wait_for_app(page)

        summary = page.locator('[data-testid="stExpander"] summary').first
        assert summary.count() > 0, "no expander offered for the large file"
        label = summary.inner_text().lower()
        assert "large document" in label or "read this first" in label, (
            f"unexpected expander: {summary.inner_text()!r}"
        )

        # The guidance lives inside a collapsed expander, so open it.
        summary.click()
        wait_for_idle(page, settle_ms=1200)
        text = page.inner_text("body")
        assert "scripts/index.py" in text, (
            "the panel does not offer the command-line alternative"
        )
        assert "big.md" in text, "the large file is not named in the guidance"
        page.close()