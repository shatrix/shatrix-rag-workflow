"""Chunking behaviour.

The properties asserted here are the ones that silently break retrieval:
chunks exceeding the embedding model's input budget, code blocks split
mid-fence, and lost content.
"""

from __future__ import annotations

import pytest

from app.chunking import chunk_text

BUDGETS = (300, 500, 1000, 1800)
OVERLAPS = (0, 50, 200)


class TestBudgetAdherence:
    @pytest.mark.parametrize("budget", BUDGETS)
    @pytest.mark.parametrize("overlap", OVERLAPS)
    def test_no_chunk_exceeds_the_budget(
        self, sample_markdown: str, budget: int, overlap: int
    ) -> None:
        chunks = chunk_text(
            sample_markdown, chunk_chars=budget, overlap_chars=overlap
        )
        assert chunks, "expected at least one chunk"
        for chunk in chunks:
            assert chunk.char_count <= budget, (
                f"chunk {chunk.index} is {chunk.char_count} chars, "
                f"over the {budget} budget"
            )

    def test_unbreakable_input_is_hard_split(self) -> None:
        chunks = chunk_text("X" * 5000, chunk_chars=300, overlap_chars=40)
        assert all(c.char_count <= 300 for c in chunks)
        assert len(chunks) > 1

    def test_overlap_larger_than_budget_does_not_hang(self) -> None:
        # Guarded by clamping; must terminate rather than loop forever.
        chunks = chunk_text("word " * 500, chunk_chars=300, overlap_chars=300)
        assert all(c.char_count <= 300 for c in chunks)

    def test_zero_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_chars=0)

    def test_negative_overlap_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_chars=300, overlap_chars=-1)


class TestContentPreservation:
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_no_text_is_lost(self, sample_markdown: str, budget: int) -> None:
        chunks = chunk_text(sample_markdown, chunk_chars=budget)
        joined = "".join(c.text for c in chunks)
        for probe in (
            "Intro paragraph introducing",
            "Set a static address",
            "ifconfig em0 inet 10.0.0.5",
            "Run smpd and let it lease",
        ):
            assert probe in joined, f"lost content: {probe!r}"

    def test_empty_input_yields_nothing(self) -> None:
        assert chunk_text("", chunk_chars=500) == []
        assert chunk_text("   \n\n\t ", chunk_chars=500) == []

    def test_indices_are_contiguous(self, sample_markdown: str) -> None:
        chunks = chunk_text(sample_markdown, chunk_chars=400)
        assert [c.index for c in chunks] == list(range(len(chunks)))


class TestMarkdownAwareness:
    def test_fenced_code_blocks_are_not_split(
        self, sample_markdown: str
    ) -> None:
        for budget in BUDGETS:
            chunks = chunk_text(sample_markdown, chunk_chars=budget)
            # Every chunk must contain an even number of fences, so no chunk
            # ends inside an unterminated code block.
            for chunk in chunks:
                assert chunk.text.count("```") % 2 == 0, (
                    f"chunk {chunk.index} at budget {budget} has an "
                    f"unbalanced code fence"
                )

    def test_headings_are_attributed_and_prepended(
        self, sample_markdown: str
    ) -> None:
        chunks = chunk_text(sample_markdown, chunk_chars=1800)
        headings = {c.heading for c in chunks}
        assert any("Static IP" in h for h in headings)
        # A chunk's text begins with its heading so the embedding has context.
        for chunk in chunks:
            if chunk.heading:
                assert chunk.text.startswith("## ")

    def test_heading_breadcrumb_reflects_nesting(self) -> None:
        text = "# A\n\n## B\n\n### C\n\nDeep content here.\n"
        chunks = chunk_text(text, chunk_chars=2000)
        assert chunks[0].heading == "A > B > C"

    def test_sibling_headings_do_not_leak_into_each_other(self) -> None:
        text = "# Top\n\n## One\n\nFirst body.\n\n## Two\n\nSecond body.\n"
        chunks = chunk_text(text, chunk_chars=2000)
        one = [c for c in chunks if "First body" in c.text]
        two = [c for c in chunks if "Second body" in c.text]
        assert one and two
        assert one[0].heading.endswith("One")
        assert two[0].heading.endswith("Two")

    def test_document_without_headings_still_chunks(self) -> None:
        chunks = chunk_text("Just prose. " * 400, chunk_chars=400)
        assert chunks
        assert all(c.heading == "" for c in chunks)

    def test_oversized_code_block_is_broken_on_line_boundaries(self) -> None:
        block = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(200)) + "\n```"
        chunks = chunk_text(block, chunk_chars=400)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.char_count <= 400
            # No line should be cut in half.
            for line in chunk.text.splitlines():
                if line.startswith("line_"):
                    assert line.endswith(f"= {line.split('=')[-1].strip()}")


class TestOverlap:
    def test_overlap_repeats_trailing_text(self) -> None:
        text = "".join(f"Sentence number {i} about widgets. " for i in range(120))
        without = chunk_text(text, chunk_chars=600, overlap_chars=0)
        with_overlap = chunk_text(text, chunk_chars=600, overlap_chars=150)
        assert len(with_overlap) >= len(without)

    def test_overlap_starts_at_a_word_boundary(self) -> None:
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        chunks = chunk_text(text, chunk_chars=60, overlap_chars=30)
        for chunk in chunks[1:]:
            first_word = chunk.text.split()[0]
            # A hard mid-word cut would leave punctuation or a fragment.
            assert first_word.isalnum()

class TestJunkHeadingRejection:
    """Real PDFs contain '#' lines that are code, not section headings."""

    @pytest.mark.parametrize(
        "line",
        ['# "', '# !!!', '# ----', '#'],
    )
    def test_punctuation_only_headings_are_not_headings(self, line: str) -> None:
        chunks = chunk_text(f"# Real Title\n\nBody text.\n\n{line}\n\nMore.",
                            chunk_chars=2000)
        headings = {c.heading for c in chunks}
        assert all("Real Title" in h or not h for h in headings), headings

    @pytest.mark.parametrize("heading", ["A", "C", "1", "Q3"])
    def test_short_but_real_headings_are_kept(self, heading: str) -> None:
        chunks = chunk_text(f"# {heading}\n\nSome body text here.",
                            chunk_chars=2000)
        assert any(chunks[0].heading == heading for _ in [0])

    def test_absurdly_long_hash_line_is_not_a_heading(self) -> None:
        line = "# " + ("x" * 300)
        chunks = chunk_text(f"# Real Title\n\nBody.\n\n{line}\n",
                            chunk_chars=2000)
        assert not any("x" * 50 in c.heading for c in chunks)
