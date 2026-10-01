"""Splitting converted Markdown into embedding-sized chunks.

A naive ``text[:1000]`` splitter cuts mid-sentence and mid-table, which wrecks
both embedding quality and the readability of the passages handed to the LLM.
This splitter is aware of three Markdown structures that matter:

* **Headings** -- a section is the natural chunking boundary, and its heading
  is prepended to every chunk in that section so the embedding carries context
  ("## Configure network > ### Static IP" is far more useful than a paragraph
  that begins "Set the address to...").
* **Fenced code blocks** -- never split inside ``` fences; a half a shell
  command is worse than no chunk at all.
* **Paragraphs** -- never split mid-paragraph unless the paragraph alone
  exceeds the budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit."""

    text: str
    heading: str
    index: int

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class _Section:
    heading: str
    body: str


def _group_units(body: str) -> list[str]:
    """Split a section body into atomic units.

    Returns a list where each element is either prose (splittable further) or
    a fenced code block (atomic).
    """
    units: list[str] = []
    buffer: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in body.splitlines():
        match = _FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker
                if buffer:
                    units.append("\n".join(buffer))
                    buffer = []
                buffer.append(line)
                continue
            if marker == fence_marker:
                in_fence = False
                fence_marker = ""
                buffer.append(line)
                units.append("\n".join(buffer))
                buffer = []
                continue
        buffer.append(line)

    if buffer:
        units.append("\n".join(buffer))

    return [u for u in units if u.strip()]


def _split_prose(text: str, limit: int) -> list[str]:
    """Split prose into pieces no larger than ``limit``.

    Tries, in order: blank-line paragraph boundaries, then sentence
    boundaries, then a hard cut as a last resort.
    """
    if len(text) <= limit:
        return [text]

    pieces: list[str] = []

    # Paragraphs first.
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if len(block) <= limit:
            pieces.append(block)
            continue

        # Paragraph is oversized: fall back to sentences.
        sentences = [s.strip() for s in _SENTENCE_RE.split(block) if s.strip()]
        current = ""
        for sentence in sentences:
            # A single sentence longer than the limit gets hard-cut below.
            while len(sentence) > limit:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(sentence[:limit])
                sentence = sentence[limit:]
            if not current:
                current = sentence
            elif len(current) + 1 + len(sentence) <= limit:
                current = f"{current} {sentence}"
            else:
                pieces.append(current)
                current = sentence
        if current:
            pieces.append(current)

    return [p for p in pieces if p.strip()]


def _sections(text: str) -> list[_Section]:
    """Break a document into heading-delimited sections.

    A document with no headings yields a single unnamed section.
    """
    sections: list[_Section] = []
    heading = ""
    breadcrumb: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if current:
                sections.append(_Section(heading, "\n".join(current)))
                current = []
            level = len(match.group(1))
            title = match.group(2).strip()
            del breadcrumb[level - 1 :]
            while len(breadcrumb) < level - 1:
                breadcrumb.append("")
            breadcrumb.append(title)
            heading = " > ".join(part for part in breadcrumb if part)
        else:
            current.append(line)

    if current:
        sections.append(_Section(heading, "\n".join(current)))

    if not any(s.body.strip() for s in sections):
        return [_Section("", text)]

    return sections


def _tail_overlap(text: str, overlap: int) -> str:
    """Take roughly ``overlap`` trailing characters, starting at a word edge."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    cut = tail.find(" ")
    if 0 <= cut < len(tail) - 1:
        tail = tail[cut + 1 :]
    return tail.strip()


def chunk_text(
    text: str,
    *,
    chunk_chars: int,
    overlap_chars: int = 0,
) -> list[Chunk]:
    """Split ``text`` into chunks of at most ``chunk_chars`` characters.

    Args:
        text: Markdown produced by the document converter.
        chunk_chars: Maximum characters per chunk. Must be positive.
        overlap_chars: Characters of the previous chunk repeated at the start
            of the next, to preserve context across boundaries.

    Returns:
        Chunks in document order. Empty when there is nothing to index.
    """
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be at least 1")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must not be negative")
    overlap_chars = min(overlap_chars, chunk_chars - 1)

    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalised.strip():
        return []

    # Reserve room for a heading prefix so chunks never exceed the budget once
    # the heading is prepended.
    raw: list[tuple[str, str]] = []

    for section in _sections(normalised):
        heading_prefix = f"## {section.heading}\n\n" if section.heading else ""
        budget = max(chunk_chars - len(heading_prefix), 1)

        for unit in _group_units(section.body):
            is_code = bool(_FENCE_RE.match(unit))
            if is_code and len(unit) > budget:
                # An oversized code block has to be broken somewhere; break on
                # line boundaries so no line is corrupted.
                buffer: list[str] = []
                size = 0
                for line in unit.splitlines():
                    if size + len(line) + 1 > budget and buffer:
                        raw.append((heading_prefix, "\n".join(buffer)))
                        buffer, size = [], 0
                    buffer.append(line)
                    size += len(line) + 1
                if buffer:
                    raw.append((heading_prefix, "\n".join(buffer)))
                continue

            for piece in _split_prose(unit, budget):
                raw.append((heading_prefix, piece))

    # Pack adjacent pieces up to the budget, carrying overlap across joins.
    chunks: list[Chunk] = []
    buffer = ""
    prefix = ""

    def emit(body: str, pre: str) -> None:
        text = (pre + body).strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    heading=pre.removeprefix("## ").strip(),
                    index=len(chunks),
                )
            )

    def flush() -> None:
        nonlocal buffer
        emit(buffer, prefix)
        buffer = ""

    for piece_prefix, piece in raw:
        if not buffer:
            buffer, prefix = piece, piece_prefix
        elif piece_prefix != prefix:
            # Never merge across a heading boundary. A chunk that swallowed
            # the next section would carry a heading describing only part of
            # its own content, which misleads both the embedding and the
            # citation shown to the user.
            flush()
            buffer, prefix = piece, piece_prefix
        else:
            # Accumulate onto what we already have; only emit and start a new
            # chunk once the next piece would overflow.
            candidate = f"{buffer}\n\n{piece}"
            if len(piece_prefix + candidate) <= chunk_chars:
                buffer = candidate
            else:
                carry = _tail_overlap(buffer, overlap_chars)
                flush()
                prefix = piece_prefix
                buffer = f"{carry}\n\n{piece}" if carry else piece

        # A single piece, or a carried overlap, can still overflow; hard-cut it.
        while len(prefix + buffer) > chunk_chars:
            room = chunk_chars - len(prefix)
            if room <= 0:
                # Pathological: the heading prefix alone fills the budget.
                emit(buffer, prefix)
                buffer = ""
                break
            head, buffer = buffer[:room], buffer[room:]
            emit(head, prefix)
            carry = _tail_overlap(head, overlap_chars)
            buffer = f"{carry}\n\n{buffer}" if carry else buffer

    flush()

    return [c for c in chunks if c.text.strip()]