"""Which file formats this installation can actually handle, and why.

markitdown supports a wide range of inputs, but several of its converters are
gated behind optional pip extras. Rather than hard-coding a list of extensions
for the upload widget -- which would happily offer the user a format that then
fails three stages later -- we probe the environment and only advertise what
will work end to end.

The extension -> converter mapping below was read directly out of markitdown's
``ACCEPTED_FILE_EXTENSIONS`` constants rather than from its prose
documentation, so it reflects reality.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

#: Text-ish formats markitdown handles with no optional dependencies.
_PLAIN = (".md", ".markdown", ".txt", ".text", ".json", ".jsonl")
_STRUCTURED = (".csv", ".htm", ".html", ".rss", ".atom", ".xml", ".epub", ".ipynb")
_ARCHIVE = (".zip",)


@dataclass(frozen=True)
class FormatSpec:
    """One row of the support table."""

    extensions: tuple[str, ...]
    label: str
    #: Python modules that must be importable for this format to work.
    modules: tuple[str, ...] = ()
    #: True when useful output requires a vision-capable model.
    needs_vision: bool = False
    note: str = ""
    #: Extensions handled by this project rather than by markitdown.
    handled_by: str = "markitdown"

    @property
    def probe_modules(self) -> tuple[str, ...]:
        return self.modules


FORMAT_SPECS: tuple[FormatSpec, ...] = (
    FormatSpec(
        _PLAIN,
        "Markdown / plain text",
        note="Read directly.",
    ),
    FormatSpec(
        (".csv",),
        "CSV",
        note="Converted via pandas.",
    ),
    FormatSpec(
        (".htm", ".html"),
        "HTML",
        note="Requires beautifulsoup4 + markdownify (both ship with markitdown).",
    ),
    FormatSpec(
        (".rss", ".atom", ".xml"),
        "RSS / Atom / XML",
    ),
    FormatSpec((".epub",), "EPUB"),
    FormatSpec((".ipynb",), "Jupyter notebook"),
    FormatSpec(
        _ARCHIVE,
        "ZIP archive",
        handled_by="rag-workflow",
        note="Expanded by this project; each member becomes its own document.",
    ),
    FormatSpec(
        (".pdf",),
        "PDF",
        modules=("pdfminer", "pdfplumber"),
        note=(
            "Text layer only. Image-only scans yield nothing and are reported "
            "as empty; OCR is not included."
        ),
    ),
    FormatSpec(
        (".docx",),
        "Word",
        modules=("mammoth",),
        note="Installed via markitdown's [docx] extra.",
    ),
    FormatSpec(
        (".pptx",),
        "PowerPoint",
        modules=("pptx",),
        note="Text in shapes and notes; images are not described.",
    ),
    FormatSpec(
        (".xlsx",),
        "Excel (xlsx)",
        modules=("openpyxl",),
        note="Every sheet becomes a Markdown table; large sheets make large chunks.",
    ),
    FormatSpec(
        (".xls",),
        "Excel (legacy xls)",
        modules=("xlrd",),
        note="Needs markitdown's [xls] extra.",
    ),
    FormatSpec(
        (".msg",),
        "Outlook message",
        modules=("olefile",),
        note="Body only; no attachments, no header metadata.",
    ),
    FormatSpec(
        (".jpg", ".jpeg", ".png"),
        "Image",
        modules=(),
        needs_vision=True,
        note=(
            "EXIF metadata only unless RAG_VISION_MODEL names a "
            "vision-capable model."
        ),
    ),
    FormatSpec(
        (".mp3", ".wav", ".m4a", ".mp4"),
        "Audio / video",
        modules=("pydub", "speechrecognition"),
        note="Needs markitdown's [audio-transcription] extra; not installed by default.",
    ),
)


@dataclass
class FormatStatus:
    """A :class:`FormatSpec` resolved against the current environment."""

    spec: FormatSpec
    available: bool
    reason: str = ""
    extensions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def ext_display(self) -> str:
        return " ".join(self.extensions)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # ValueError: a parent package exists but is broken.
        return False


def markitdown_available() -> bool:
    """True if the markitdown package itself can be imported."""
    return _module_available("markitdown")


def resolve_statuses(*, vision_enabled: bool = False) -> list[FormatStatus]:
    """Probe every spec and report whether it works here.

    ``vision_enabled`` reflects whether the operator configured a
    vision-capable model; without one, image uploads cannot produce useful
    text and are reported as unavailable.
    """
    has_markitdown = markitdown_available()
    statuses: list[FormatStatus] = []

    for spec in FORMAT_SPECS:
        extensions = spec.extensions

        if not has_markitdown:
            statuses.append(
                FormatStatus(
                    spec,
                    available=False,
                    reason="markitdown is not installed",
                    extensions=extensions,
                )
            )
            continue

        if spec.handled_by == "rag-workflow":
            # ZIP expansion is our own code, so no extra dependency applies.
            statuses.append(FormatStatus(spec, available=True, extensions=extensions))
            continue

        missing = [m for m in spec.probe_modules if not _module_available(m)]
        if missing:
            statuses.append(
                FormatStatus(
                    spec,
                    available=False,
                    reason=f"missing module(s): {', '.join(missing)}",
                    extensions=extensions,
                )
            )
            continue

        if spec.needs_vision and not vision_enabled:
            statuses.append(
                FormatStatus(
                    spec,
                    available=False,
                    reason="set RAG_VISION_MODEL to a vision-capable model",
                    extensions=extensions,
                )
            )
            continue

        statuses.append(FormatStatus(spec, available=True, extensions=extensions))

    return statuses


def supported_extensions(*, vision_enabled: bool = False) -> list[str]:
    """Extensions safe to pass to ``st.file_uploader(type=[...])``."""
    exts: list[str] = []
    for status in resolve_statuses(vision_enabled=vision_enabled):
        if status.available:
            exts.extend(status.extensions)
    return sorted(set(exts))


def extension_supported(extension: str, *, vision_enabled: bool = False) -> bool:
    extension = extension.lower()
    if not extension.startswith("."):
        extension = "." + extension
    return extension in set(supported_extensions(vision_enabled=vision_enabled))


def unavailable_reason(extension: str, *, vision_enabled: bool = False) -> str:
    """Human-readable explanation for why an extension is not accepted."""
    extension = extension.lower()
    if not extension.startswith("."):
        extension = "." + extension
    for status in resolve_statuses(vision_enabled=vision_enabled):
        if extension in status.extensions:
            if status.available:
                return ""
            return status.reason
    return "format not recognised"