"""Conversion and archive-safety tests.

Real markitdown conversions are exercised for the formats whose extras are
installed; the security guards around ZIP handling are tested unconditionally
because they must never regress.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from app.documents import ConversionError, convert_file, store_upload


def make_zip(path: Path, members: dict[str, str], *, compress: bool = True) -> Path:
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", mode) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


LONG_BODY = "This is a sentence of reasonable length about system configuration. " * 8


class TestPlainFormats:
    @pytest.mark.parametrize(
        ("name", "content"),
        [
            ("doc.md", "# Heading\n\n" + LONG_BODY),
            ("doc.txt", LONG_BODY),
            ("doc.json", '{"board": "i.MX8", "smp_cores": 6}'),
            ("doc.csv", "name,role\nalpha,smpsched\nbeta,io-spawn\n"),
            ("doc.html", "<html><body><h1>Notes</h1><p>" + LONG_BODY + "</p></body></html>"),
        ],
    )
    def test_text_formats_convert(self, tmp_path: Path, name, content) -> None:
        path = tmp_path / name
        path.write_text(content)
        units = convert_file(path)
        assert len(units) == 1
        assert units[0].usable
        assert units[0].markdown.strip()

    def test_markdown_heading_becomes_the_title(self, tmp_path: Path) -> None:
        path = tmp_path / "guide.md"
        path.write_text("# Network Setup\n\n" + LONG_BODY)
        assert convert_file(path)[0].title == "Network Setup"

    def test_empty_file_yields_an_empty_unit(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.md"
        path.write_text("   \n\n  ")
        unit = convert_file(path)[0]
        assert not unit.usable
        assert any("no text" in w for w in unit.warnings)

    def test_short_but_real_content_is_still_usable(
        self, tmp_path: Path
    ) -> None:
        # A one-line README is short but perfectly valid; it must not be
        # dropped just for being small.
        path = tmp_path / "readme.md"
        path.write_text("# Readme\n\nShort but real.")
        unit = convert_file(path)[0]
        assert unit.usable
        assert unit.suspicious


class TestArchiveSafety:
    def test_each_member_becomes_its_own_document(
        self, tmp_path: Path
    ) -> None:
        archive = make_zip(
            tmp_path / "bundle.zip",
            {
                "a.md": "# Alpha\n\n" + LONG_BODY,
                "b.md": "# Beta\n\n" + LONG_BODY,
            },
        )
        units = convert_file(archive)
        assert len(units) == 2
        assert {u.member_path for u in units} == {"a.md", "b.md"}
        assert all(u.source == "bundle.zip" for u in units)
        assert all(u.source_zip == "bundle.zip" for u in units)

    @pytest.mark.parametrize(
        "hostile",
        [
            "../escape.md",
            "../../etc/passwd",
            "/absolute/path.md",
            "sub/../../escape.md",
        ],
    )
    def test_traversal_members_are_rejected(
        self, tmp_path: Path, hostile: str
    ) -> None:
        archive = make_zip(
            tmp_path / "evil.zip",
            {hostile: "# Evil\n\n" + LONG_BODY, "ok.md": "# Fine\n\n" + LONG_BODY},
        )
        units = convert_file(archive)
        skipped = [u for u in units if any("skipped" in w for w in u.warnings)]
        assert len(skipped) == 1
        assert "traversal" in skipped[0].warnings[0] or "absolute" in skipped[0].warnings[0]
        assert not skipped[0].usable
        # The legitimate member still indexed.
        assert any(u.member_path == "ok.md" and u.usable for u in units)

    def test_symlink_members_are_rejected(self, tmp_path: Path) -> None:
        archive_path = tmp_path / "links.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            info = zipfile.ZipInfo("evil.md")
            info.external_attr = (0o120777 << 16)  # symlink mode bits
            archive.writestr(info, "/etc/passwd")
        units = convert_file(archive_path)
        assert any("symlink" in u.warnings[0] for u in units if u.warnings)

    def test_zip_bomb_is_rejected_by_expansion_ratio(
        self, tmp_path: Path, isolated_config
    ) -> None:
        # 5 MB of zeros compresses to a few KB: a ~1000x expansion ratio.
        archive = make_zip(
            tmp_path / "bomb.zip", {"big.bin": "\0" * (5 * 1024 * 1024)}
        )
        with pytest.raises(ConversionError, match="expands"):
            convert_file(archive)

    def test_too_many_members_is_rejected(
        self, tmp_path: Path, isolated_config, monkeypatch
    ) -> None:
        from app.config import reload_settings

        monkeypatch.setenv("MAX_ZIP_MEMBERS", "5")
        reload_settings()
        archive = make_zip(
            tmp_path / "many.zip", {f"f{i}.md": "x" for i in range(50)}
        )
        with pytest.raises(ConversionError, match="entries"):
            convert_file(archive)

    def test_corrupt_archive_is_reported_cleanly(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "broken.zip"
        path.write_bytes(b"PK\x03\x04 not really a zip")
        with pytest.raises(ConversionError, match="ZIP"):
            convert_file(path)

    def test_nested_archive_is_supported(
        self, tmp_path: Path
    ) -> None:
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as archive:
            archive.writestr("deep.md", "# Deep\n\n" + LONG_BODY)
        outer = tmp_path / "outer.zip"
        with zipfile.ZipFile(outer, "w") as archive:
            archive.writestr("inner.zip", inner.getvalue())
        units = convert_file(outer)
        assert any(u.member_path.endswith("deep.md") and u.usable for u in units)


class TestStoreUpload:
    def test_sanitises_and_writes(self, tmp_path: Path) -> None:
        docs = tmp_path / "docs"
        written = store_upload(b"hello", "../../evil.md", docs_dir=docs)
        assert written.parent == docs
        assert "/" not in written.name
        assert written.read_bytes() == b"hello"

    def test_duplicate_content_overwrites_the_same_file(
        self, tmp_path: Path
    ) -> None:
        docs = tmp_path / "docs"
        a = store_upload(b"same", "doc.md", docs_dir=docs)
        b = store_upload(b"same", "doc.md", docs_dir=docs)
        assert a == b
        assert len(list(docs.iterdir())) == 1

    def test_changed_content_gets_a_new_name(
        self, tmp_path: Path
    ) -> None:
        # Overwriting silently would orphan the previously indexed version.
        docs = tmp_path / "docs"
        store_upload(b"first", "doc.md", docs_dir=docs)
        second = store_upload(b"second", "doc.md", docs_dir=docs)
        assert second.name != "doc.md"
        assert len(list(docs.iterdir())) == 2

    def test_size_limit_is_enforced(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from app.config import reload_settings

        monkeypatch.setenv("MAX_FILE_MB", "1")
        reload_settings()
        with pytest.raises(ConversionError, match="limit"):
            store_upload(b"0" * (2 * 1024 * 1024), "big.bin", docs_dir=tmp_path)

    def test_hostile_name_cannot_escape(self, tmp_path: Path) -> None:
        docs = tmp_path / "docs"
        written = store_upload(b"x", "../../../tmp/pwned.md", docs_dir=docs)
        assert written.resolve().is_relative_to(docs.resolve())


class TestOfficeFormats:
    def test_xlsx_converts(self, tmp_path: Path) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = "Ports"
        sheet.append(["port", "protocol"])
        sheet.append([187, "tcp"])
        path = tmp_path / "ports.xlsx"
        book.save(path)
        unit = convert_file(path)[0]
        assert unit.usable
        assert "187" in unit.markdown

    def test_pptx_converts(self, tmp_path: Path) -> None:
        pptx = pytest.importorskip("pptx")
        deck = pptx.Presentation()
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = "Architecture Overview"
        slide.placeholders[1].text = "procmgr spawns io-spawn."
        path = tmp_path / "arch.pptx"
        deck.save(path)
        unit = convert_file(path)[0]
        assert unit.usable
        assert "Architecture Overview" in unit.markdown