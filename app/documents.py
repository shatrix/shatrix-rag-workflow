"""Turn uploaded files into Markdown, using markitdown.

Security notes, both drawn from markitdown's own documentation:

* We only ever call ``convert_stream()`` with a stream we opened ourselves
  from a path we already sanitised. We deliberately never call ``convert()``,
  ``convert_uri()`` or ``convert_url()``, any of which would let a crafted
  input fetch a remote or ``file://`` resource.
* Archives are expanded here rather than by markitdown's zip converter. Its
  converter concatenates every member into one unnamed blob, which destroys
  source attribution and makes per-file deletion impossible. Expanding here
  also lets us apply zip-bomb and path-traversal guards that a blind
  "read everything" converter would not.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import Settings, get_settings
from app.sanitize import UnsafeNameError, sanitize_filename
from app.vectorstore import META_MEMBER_PATH, META_SOURCE_ZIP

#: Below this many non-whitespace characters an extraction is flagged as
#: suspicious. Typically means an image-only PDF with no text layer. Flagged
#: documents are still indexed -- a legitimately short README must not be
#: dropped just because it is small.
SUSPICIOUS_CHARS = 50

#: Chunks are only refused outright when there is literally nothing to embed.
MIN_USABLE_CHARS = 1

#: Guards against an archive that expands into itself indefinitely.
MAX_ARCHIVE_DEPTH = 3


@dataclass
class ConvertedUnit:
    """One indexable unit: a single file, or a single member of an archive."""

    source: str
    markdown: str
    title: str = ""
    member_path: str = ""
    source_zip: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """False only when there is literally nothing to embed."""
        return len(self.markdown.strip()) >= MIN_USABLE_CHARS

    @property
    def suspicious(self) -> bool:
        """True when the extraction is too small to be a real document.

        Worth surfacing, not worth blocking on: a one-page PDF may legitimately
        yield 80 characters, while a scanned page yields zero.
        """
        return 0 < len(self.markdown.strip()) < SUSPICIOUS_CHARS

    @property
    def label(self) -> str:
        if self.member_path:
            return f"{self.source}/{self.member_path}"
        return self.source


class ConversionError(RuntimeError):
    """A file could not be converted to text."""


def _quiet_pdf_loggers() -> None:
    """Silence a per-page pdfminer warning that floods logs on big PDFs.

    pdfminer emits "Could not get FontBBox from font descriptor because None
    cannot be parsed as 4 floats" for many pages in a normal document. On a
    1400-page PDF that buries the actual progress output under thousands of
    lines. It is benign -- pdfminer recovers and extraction succeeds.
    """
    import logging

    for name in ("pdfminer", "pdfplumber", "pdfminer.layout", "pdfminer.converter"):
        logging.getLogger(name).setLevel(logging.ERROR)


@lru_cache(maxsize=1)
def _converter(vision_model: str):
    """Build (and cache) a MarkItDown instance.

    ``vision_model`` participates in the cache key so that changing it between
    runs produces a fresh converter.
    """
    from markitdown import MarkItDown

    kwargs: dict = {"enable_plugins": False}
    if vision_model:
        # Only used for image captions and, with the markitdown-ocr plugin,
        # OCR. Requires a genuinely vision-capable model.
        from app.openrouter import build_client

        kwargs["llm_client"] = build_client()
        kwargs["llm_model"] = vision_model

    return MarkItDown(**kwargs)


def _first_heading(markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            # Only the heading line itself; multi-cell notebooks put several
            # cells in one line and we want just the first.
            return stripped.lstrip("#").strip().splitlines()[0]
    return ""


def _convert_bytes(
    data: bytes,
    *,
    extension: str,
    filename: str,
    settings: Settings,
) -> tuple[str, str]:
    """Convert raw bytes to Markdown. Returns ``(markdown, title)``."""
    from markitdown import StreamInfo

    _quiet_pdf_loggers()

    stream_info = StreamInfo(extension=extension, filename=filename)
    stream = io.BytesIO(data)

    try:
        converter = _converter(settings.vision_model)
    except Exception as exc:  # noqa: BLE001
        raise ConversionError(
            f"markitdown could not be initialised: {exc}. Reinstall with "
            f"'./scripts/setup.sh' or 'pip install -r requirements.txt'."
        ) from exc

    try:
        result = converter.convert_stream(stream, stream_info=stream_info)
    except Exception as exc:  # noqa: BLE001 - markitdown raises many types
        raise ConversionError(f"{type(exc).__name__}: {exc}") from exc

    markdown = (getattr(result, "markdown", "") or "").strip()
    title = (getattr(result, "title", "") or "").strip()
    # Some converters (notebooks, spreadsheets) return a multi-line title.
    # Titles are display metadata, so keep only the first line.
    if title:
        title = title.splitlines()[0].strip()

    # A PDF's title is inferred from the first text on its first page. For a
    # short document that is usually fine, but on a long one it is whatever
    # happened to be at the top of page one -- a header, a licence preamble, a
    # stray code fragment. Because the title is prepended to every chunk, one
    # bad title contaminates the whole document's embeddings. Prefer a real
    # Markdown heading, then the filename.
    if extension == ".pdf":
        # Fall back to the filename rather than keeping a guess derived from
        # the first line of page one.
        title = _first_heading(markdown) or Path(filename).stem
    elif not title:
        title = _first_heading(markdown) or Path(filename).stem

    return markdown, title


def _is_safe_member(info: zipfile.ZipInfo) -> tuple[bool, str]:
    """Reject unsafe archive members."""
    name = info.filename.replace("\\", "/")

    if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        return False, "absolute path"
    if any(part == ".." for part in name.split("/")):
        return False, "path traversal (..)"
    if info.is_dir():
        return False, "directory"

    # Unix symlinks live in the high 16 bits of external_attr.
    mode = (info.external_attr >> 16) & 0o170000
    if mode == 0o120000:
        return False, "symlink"

    return True, ""


def _read_archive(
    archive: zipfile.ZipFile,
    *,
    archive_name: str,
    prefix: str,
    compressed_bytes: int,
    settings: Settings,
    depth: int,
) -> list[ConvertedUnit]:
    """Convert every safe member of an open archive into units.

    Shared by top-level archives and nested ones so both paths get identical
    safety checks.
    """
    units: list[ConvertedUnit] = []
    infos = archive.infolist()

    if len(infos) > settings.max_zip_members:
        raise ConversionError(
            f"archive{' at ' + prefix if prefix else ''} holds {len(infos)} "
            f"entries, above the limit of {settings.max_zip_members}"
        )

    total_uncompressed = sum(i.file_size for i in infos)
    if total_uncompressed > settings.max_zip_total_mb * 1024 * 1024:
        raise ConversionError(
            f"archive{' at ' + prefix if prefix else ''} expands to "
            f"{total_uncompressed / 1024 / 1024:.0f} MB, above the limit of "
            f"{settings.max_zip_total_mb} MB"
        )
    if compressed_bytes > 0:
        ratio = total_uncompressed / compressed_bytes
        if ratio > settings.max_zip_expansion_ratio:
            raise ConversionError(
                f"archive{' at ' + prefix if prefix else ''} expands "
                f"{ratio:.0f}x, above the allowed "
                f"{settings.max_zip_expansion_ratio}x (possible zip bomb)"
            )

    for info in infos:
        member_name = f"{prefix}/{info.filename}" if prefix else info.filename

        safe, why = _is_safe_member(info)
        if not safe:
            units.append(
                ConvertedUnit(
                    source=archive_name,
                    member_path=member_name,
                    source_zip=archive_name,
                    markdown="",
                    warnings=[f"skipped: {why}"],
                )
            )
            continue

        member_ext = Path(info.filename).suffix.lower()

        if member_ext == ".zip":
            if depth + 1 >= MAX_ARCHIVE_DEPTH:
                units.append(
                    ConvertedUnit(
                        source=archive_name,
                        member_path=member_name,
                        source_zip=archive_name,
                        markdown="",
                        warnings=[
                            f"skipped: nested archive deeper than "
                            f"{MAX_ARCHIVE_DEPTH} levels"
                        ],
                    )
                )
                continue
            try:
                payload = archive.read(info)
                with zipfile.ZipFile(io.BytesIO(payload)) as nested:
                    units.extend(
                        _read_archive(
                            nested,
                            archive_name=archive_name,
                            prefix=member_name,
                            compressed_bytes=len(payload),
                            settings=settings,
                            depth=depth + 1,
                        )
                    )
            except zipfile.BadZipFile:
                units.append(
                    ConvertedUnit(
                        source=archive_name,
                        member_path=member_name,
                        source_zip=archive_name,
                        markdown="",
                        warnings=["skipped: nested archive is unreadable"],
                    )
                )
            except ConversionError as exc:
                units.append(
                    ConvertedUnit(
                        source=archive_name,
                        member_path=member_name,
                        source_zip=archive_name,
                        markdown="",
                        warnings=[str(exc)],
                    )
                )
            continue

        try:
            payload = archive.read(info)
        except Exception as exc:  # noqa: BLE001
            units.append(
                ConvertedUnit(
                    source=archive_name,
                    member_path=member_name,
                    source_zip=archive_name,
                    markdown="",
                    warnings=[f"could not read member: {exc}"],
                )
            )
            continue

        try:
            markdown, title = _convert_bytes(
                payload,
                extension=member_ext,
                filename=Path(info.filename).name,
                settings=settings,
            )
        except ConversionError as exc:
            units.append(
                ConvertedUnit(
                    source=archive_name,
                    member_path=member_name,
                    source_zip=archive_name,
                    markdown="",
                    warnings=[str(exc)],
                )
            )
            continue

        warnings: list[str] = []
        if not markdown.strip():
            warnings.append("no text extracted")
        elif len(markdown.strip()) < SUSPICIOUS_CHARS:
            warnings.append(
                f"only {len(markdown.strip())} characters extracted "
                "(scanned or image-only file?)"
            )

        units.append(
            ConvertedUnit(
                source=archive_name,
                markdown=markdown,
                title=title,
                member_path=member_name,
                source_zip=archive_name,
                warnings=warnings,
            )
        )

    return units


def _expand_archive(
    path: Path,
    archive_name: str,
    *,
    settings: Settings,
    depth: int = 0,
) -> list[ConvertedUnit]:
    """Expand a ZIP file on disk into one unit per safe member."""
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ConversionError(f"not a readable ZIP archive: {exc}") from exc

    with archive:
        return _read_archive(
            archive,
            archive_name=archive_name,
            prefix="",
            compressed_bytes=path.stat().st_size,
            settings=settings,
            depth=depth,
        )


def convert_file(
    path: Path,
    *,
    source: str | None = None,
    settings: Settings | None = None,
    depth: int = 0,
) -> list[ConvertedUnit]:
    """Convert a file on disk into one or more indexable units.

    A ``.zip`` yields one unit per safe member. Everything else yields exactly
    one unit.

    Args:
        path: File to read. Must already live inside the documents directory.
        source: Display name; defaults to the file's name.
        settings: Configuration override, mainly for tests.
        depth: Current archive nesting depth.

    Raises:
        ConversionError: The file could not be converted at all.
    """
    settings = settings or get_settings()
    source = source or path.name
    extension = path.suffix.lower()

    if extension == ".zip":
        return _expand_archive(path, source, settings=settings, depth=depth)

    size_bytes = path.stat().st_size
    if size_bytes > settings.max_file_mb * 1024 * 1024:
        raise ConversionError(
            f"file is {size_bytes / 1024 / 1024:.1f} MB, above the "
            f"{settings.max_file_mb} MB limit"
        )

    markdown, title = _convert_bytes(
        path.read_bytes(),
        extension=extension,
        filename=path.name,
        settings=settings,
    )

    warnings: list[str] = []
    stripped = markdown.strip()
    if not stripped:
        detail = (
            "the file may be an image-only scan and needs OCR, which this "
            "project does not include"
            if extension == ".pdf"
            else "the file may be empty or contain only images"
        )
        warnings.append(f"no text extracted ({detail})")
    elif len(stripped) < SUSPICIOUS_CHARS:
        warnings.append(
            f"only {len(stripped)} characters extracted; the file may be "
            "mostly images or a near-empty scan"
        )

    return [
        ConvertedUnit(
            source=source,
            markdown=markdown,
            title=title,
            warnings=warnings,
        )
    ]


def store_upload(data: bytes, raw_name: str, *, docs_dir: Path) -> Path:
    """Sanitise ``raw_name`` and write ``data`` into ``docs_dir``.

    Returns the path written. Raises:
        UnsafeNameError: the name could not be made safe.
        ConversionError: the payload exceeds the configured size limit.
    """
    from app.config import get_settings as _gs

    settings = _gs()
    limit = settings.max_file_mb * 1024 * 1024
    if len(data) > limit:
        raise ConversionError(
            f"{raw_name} is {len(data) / 1024 / 1024:.1f} MB, above the "
            f"{settings.max_file_mb} MB limit"
        )

    try:
        safe_name = sanitize_filename(raw_name)
    except UnsafeNameError as exc:
        raise ConversionError(f"rejected filename: {exc}") from exc

    docs_dir.mkdir(parents=True, exist_ok=True)
    target = docs_dir / safe_name

    # Resolve and confirm containment: sanitize_filename already strips
    # separators, this is the belt-and-braces check.
    resolved = target.resolve()
    if not resolved.is_relative_to(docs_dir.resolve()):
        raise ConversionError(f"refusing to write outside {docs_dir}")

    # A clash with an existing different file would silently orphan the old
    # copy, so version the name instead.
    if target.exists() and target.read_bytes() != data:
        stem, dot, ext = safe_name.rpartition(".")
        stem = stem or "document"
        for n in range(2, 100):
            candidate = target.with_name(f"{stem}-{n}{dot}{ext}" if dot else f"{stem}-{n}")
            if not candidate.exists():
                target = candidate
                break
        resolved = target.resolve()
        if not resolved.is_relative_to(docs_dir.resolve()):
            raise ConversionError(f"refusing to write outside {docs_dir}")

    target.write_bytes(data)
    return target


__all__ = [
    "ConvertedUnit",
    "ConversionError",
    "MIN_USABLE_CHARS",
    "SUSPICIOUS_CHARS",
    "META_MEMBER_PATH",
    "META_SOURCE_ZIP",
    "convert_file",
    "store_upload",
]