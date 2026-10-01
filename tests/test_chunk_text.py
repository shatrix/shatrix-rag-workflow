"""Chunk text must stay clean and generic.

Two properties are asserted here, both derived from failures on real
documents and neither tied to any particular file or topic:

1. A ``#`` line inside a fenced code block is code, not a section heading.
   Shell, C, Python, BitBake, YAML and make all use ``#`` comments, so this
   affects essentially every technical document.

2. No derived document title is prepended to chunk text. Converters infer
   titles inconsistently and unreliably, and because the title lands in every
   chunk it contaminates every embedding and every prompt.
"""

from __future__ import annotations

import pytest

from app.chunking import chunk_text
from app.indexing import _build_records

LIMIT = 1800


class ConvertedStub:
    """Minimal stand-in for a ConvertedUnit."""

    def __init__(self, markdown: str, source: str = "f.md", title: str = "T",
                 member_path: str = "", source_zip: str = ""):
        self.markdown = markdown
        self.source = source
        self.title = title
        self.member_path = member_path
        self.source_zip = source_zip
        self.warnings: list[str] = []

    @property
    def label(self) -> str:
        return self.source


@pytest.mark.parametrize(
    ("language", "body"),
    [
        ("bash", "# install the toolchain\ncurl https://example.com | bash\n"
                  "# add to PATH\necho ok"),
        ("c", "#include <stdio.h>\nint main(void) { return 0; }"),
        ("python", "# a comment\nimport os\n# another\nprint(os.name)"),
        ("yaml", "# top level\nkey: value\n# nested\nother: 1"),
        ("make", "# default target\nall:\n\techo hi"),
        ("bitbake", "# comment\nSUMMARY = \"x\""),
    ],
)
class TestFencedCommentsAreNotHeadings:
    def test_hash_inside_a_fence_is_not_a_section_heading(
        self, language: str, body: str
    ) -> None:
        doc = f"# Top Level\n\nRun this:\n\n```{language}\n{body}\n```\n\nAfter.\n"
        for chunk in chunk_text(doc, chunk_chars=LIMIT):
            assert not chunk.heading or "Top Level" in chunk.heading, (
                f"{language}: fenced comment became a heading {chunk.heading!r}"
            )

    def test_code_fence_contents_survive_intact(
        self, language: str, body: str
    ) -> None:
        doc = f"# Top Level\n\n```{language}\n{body}\n```\n"
        joined = "".join(c.text for c in chunk_text(doc, chunk_chars=LIMIT))
        for line in body.splitlines():
            assert line in joined, f"{language}: lost {line!r}"

    def test_real_headings_around_the_fence_are_still_found(
        self, language: str, body: str
    ) -> None:
        doc = (
            "# Install\n\nSee below.\n\n"
            f"```{language}\n{body}\n```\n\n"
            "## Configure\n\nNow set it up.\n"
        )
        headings = {c.heading for c in chunk_text(doc, chunk_chars=LIMIT)}
        assert any(h.endswith("Install") for h in headings)
        assert any(h.endswith("Configure") for h in headings)

    def test_tilde_fences_are_honoured_too(
        self, language: str, body: str
    ) -> None:
        doc = f"# Doc\n\n~~~{language}\n{body}\n~~~\n\nEnd.\n"
        for chunk in chunk_text(doc, chunk_chars=LIMIT):
            assert not chunk.heading or "Doc" in chunk.heading


class TestUnfencedHashLinesStillCount:
    def test_plain_hash_headings_are_still_sections(self) -> None:
        doc = "# Alpha\n\nfirst\n\n## Beta\n\nsecond\n"
        headings = {c.heading for c in chunk_text(doc, chunk_chars=LIMIT)}
        assert "Alpha" in headings
        assert any(h.endswith("Beta") for h in headings)


class TestNoDerivedTitleInChunkText:
    """A wrong title contaminates every embedding, so none is prepended."""

    @pytest.mark.parametrize("junk_title", [
        "ifndef HELLOLIB_H",     # a C preprocessor line
        "ATTACH",                # a licence banner
        "Chapter 1",             # a repeated page header
        '"',                     # a stray glyph
        "",
    ])
    def test_title_never_reaches_chunk_text(self, junk_title: str) -> None:
        settings = _settings()
        unit = ConvertedStub(
            markdown="# Real Heading\n\nActual body content here.\n",
            title=junk_title,
        )
        pairs, _ = _build_records([unit], settings=settings)
        assert pairs
        for chunk, _ in pairs:
            assert "Title:" not in chunk.text
            if junk_title.strip():
                assert junk_title not in chunk.text

    def test_source_is_present(self) -> None:
        settings = _settings()
        unit = ConvertedStub(markdown="Some body text.\n", source="notes.md")
        pairs, _ = _build_records([unit], settings=settings)
        assert "Source: notes.md" in pairs[0][0].text

    def test_archive_member_keeps_its_full_provenance(self) -> None:
        settings = _settings()
        unit = ConvertedStub(
            markdown="Body text.\n", source="bundle.zip", member_path="a/b.md"
        )
        pairs, _ = _build_records([unit], settings=settings)
        assert "Source: bundle.zip :: a/b.md" in pairs[0][0].text

    def test_heading_still_provides_structure(self) -> None:
        settings = _settings()
        unit = ConvertedStub(
            markdown="# Install > Dependencies\n\nInstall the deps.\n"
        )
        pairs, _ = _build_records([unit], settings=settings)
        assert any("Install" in c.text for c, _ in pairs)


class TestChunkTextStaysWithinLimits:
    def test_no_chunk_exceeds_the_configured_ceiling(self) -> None:
        settings = _settings()
        unit = ConvertedStub(
            markdown=("A sentence of ordinary prose. " * 400),
            source="long.md",
            title="whatever",
        )
        pairs, _ = _build_records([unit], settings=settings)
        limit = settings.max_chunk_chars
        assert pairs
        for chunk, _ in pairs:
            assert len(chunk.text) <= limit


def _settings():
    from app.config import get_settings

    return get_settings()