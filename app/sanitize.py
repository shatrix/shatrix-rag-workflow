"""Filename and path safety.

Uploads arrive as attacker-controlled strings. Streamlit hands us
``UploadedFile.name``, and naively doing ``os.path.join(DOCS_DIR, name)`` lets
``../../../etc/cron.d/evil`` escape the upload directory. Everything in this
module exists to make that impossible.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

#: Longest stored filename, in characters (not bytes). Most filesystems cap a
#: single path component at 255 bytes, and UTF-8 can use up to 4 per character.
MAX_NAME_CHARS = 180

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class UnsafeNameError(ValueError):
    """Raised when a filename cannot be made safe."""


def sanitize_filename(raw: str, *, fallback: str = "document") -> str:
    """Reduce an arbitrary upload name to a single safe path component.

    Returns a name that contains no directory separators, no ``..``, no null
    or control characters, is not hidden, and fits within a sane length.

    Raises:
        UnsafeNameError: if nothing usable survives sanitisation.
    """
    if not isinstance(raw, str):
        raise UnsafeNameError(f"expected a string filename, got {type(raw).__name__}")

    candidate = raw.strip()

    # Keep only the final component, and prefer POSIX then Windows separators
    # so a name like r"C:\docs\x.pdf" reduces to "x.pdf".
    candidate = candidate.replace("\\", "/").split("/")[-1]

    # Drop null bytes and other control characters outright rather than
    # stripping them, so "a\0b" cannot become the distinct name "ab".
    candidate = _CONTROL_CHARS.sub("", candidate)

    # Normalise so visually identical names collapse to one file, and so the
    # result is stable across filesystems.
    candidate = unicodedata.normalize("NFC", candidate).strip()

    # Collapse characters that confuse shells and path handling. \w is
    # Unicode-aware for str patterns, so legitimate non-ASCII filenames such as
    # "résumé.md" survive intact while separators, quotes and shell
    # metacharacters do not.
    candidate = re.sub(r"[^\w.\- ]+", "_", candidate, flags=re.UNICODE)
    candidate = re.sub(r"_{2,}", "_", candidate).strip(" ._-")

    if not candidate:
        # Something was there but nothing survived sanitisation (e.g. "___").
        # Falling back keeps the upload usable instead of failing it.
        candidate = fallback
    if candidate in {".", ".."}:
        raise UnsafeNameError(f"filename {raw!r} contains nothing usable")

    stem, dot, ext = candidate.rpartition(".")
    if not dot:
        stem, ext = candidate, ""
    stem = stem.strip(" ._-") or "document"
    ext = ext.strip(" ._-")[:16]

    if stem.lower() in _WINDOWS_RESERVED:
        stem = f"{stem}_file"

    # Guarantee a leading word character so the name is never hidden and never
    # collides with our own bookkeeping prefixes. \w is Unicode-aware, so
    # non-Latin filenames keep their first character.
    stem = re.sub(r"^[^\w]+", "", stem, flags=re.UNICODE).strip("._-") or "document"

    budget = MAX_NAME_CHARS - (len(ext) + 1 if ext else 0)
    stem = stem[: max(budget, 1)]

    return f"{stem}.{ext}" if ext else stem


def safe_join(base: Path, name: str, *, fallback: str = "document") -> Path:
    """Join ``name`` onto ``base``, guaranteeing the result stays inside base.

    This is the last line of defence: even if a caller somehow passes a
    traversal string, the resolved path is verified to be under ``base``
    before it is returned.
    """
    base_resolved = base.resolve()
    target = base_resolved / sanitize_filename(name, fallback=fallback)

    # resolve() collapses ".." so the containment check is meaningful.
    resolved = target.resolve()
    if not resolved.is_relative_to(base_resolved):
        raise UnsafeNameError(
            f"{name!r} resolves outside {base_resolved} and was rejected"
        )
    return target


def human_size(num_bytes: int) -> str:
    """Format a byte count for display."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def relative_display(path: Path, base: Path) -> str:
    """Render ``path`` relative to ``base`` where possible, else absolute."""
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (ValueError, OSError):
        return str(path)


def which_or_none(program: str) -> str | None:
    """Locate an executable without raising if it is absent."""
    from shutil import which

    return which(program)


def ensure_within(path: Path, root: Path) -> bool:
    """True when ``path`` resolves to a location inside ``root``."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def is_hidden(name: str) -> bool:
    return name.startswith(".")


def safe_read_text(path: Path, *, max_bytes: int = 32 * 1024 * 1024) -> str:
    """Read a text file, tolerating encoding problems and refusing huge files."""
    size = os.path.getsize(path)
    if size > max_bytes:
        raise UnsafeNameError(
            f"{path.name} is {human_size(size)}, above the "
            f"{human_size(max_bytes)} read limit"
        )
    return path.read_text(encoding="utf-8", errors="replace")