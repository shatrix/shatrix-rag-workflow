"""The supported-format registry.

The registry is what the upload widget advertises, so a mismatch between it
and reality is a user-visible bug: offering a format that then fails three
stages later.
"""

from __future__ import annotations

import pytest

from app.formats import (
    FORMAT_SPECS,
    FormatSpec,
    extension_supported,
    markitdown_available,
    resolve_statuses,
    supported_extensions,
    unavailable_reason,
)

# Verified against markitdown 0.1.8's ACCEPTED_FILE_EXTENSIONS constants.
EXPECTED_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".text", ".json", ".jsonl",
    ".csv", ".htm", ".html", ".rss", ".atom", ".xml", ".epub", ".ipynb",
    ".zip", ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".msg",
    ".jpg", ".jpeg", ".png", ".mp3", ".wav", ".m4a", ".mp4",
}


class TestRegistryShape:
    def test_every_known_extension_is_declared(self) -> None:
        declared = {e for spec in FORMAT_SPECS for e in spec.extensions}
        assert declared == EXPECTED_EXTENSIONS

    def test_extensions_are_lowercase_and_dotted(self) -> None:
        for spec in FORMAT_SPECS:
            for ext in spec.extensions:
                assert ext.startswith("."), ext
                assert ext == ext.lower(), ext

    def test_no_extension_is_declared_twice(self) -> None:
        seen: list[str] = []
        for spec in FORMAT_SPECS:
            seen.extend(spec.extensions)
        duplicates = {e for e in seen if seen.count(e) > 1}
        assert not duplicates, f"duplicate extensions: {duplicates}"

    def test_every_spec_has_a_label(self) -> None:
        assert all(spec.label.strip() for spec in FORMAT_SPECS)

    def test_zip_is_handled_by_this_project(self) -> None:
        zip_spec = next(s for s in FORMAT_SPECS if ".zip" in s.extensions)
        assert zip_spec.handled_by == "rag-workflow"


class TestRuntimeProbing:
    def test_statuses_cover_every_spec(self) -> None:
        statuses = resolve_statuses()
        assert len(statuses) == len(FORMAT_SPECS)

    def test_status_carries_the_extensions_and_a_reason(self) -> None:
        for status in resolve_statuses():
            assert status.extensions
            if not status.available:
                assert status.reason, f"{status.label} unavailable but silent"

    def test_supported_extensions_are_a_sorted_subset(self) -> None:
        exts = supported_extensions()
        assert exts == sorted(set(exts))
        assert all(e.startswith(".") for e in exts)

    def test_images_require_a_vision_model(self) -> None:
        without = supported_extensions(vision_enabled=False)
        with_vision = supported_extensions(vision_enabled=True)
        assert ".png" not in without
        assert ".png" in with_vision

    def test_image_reason_explains_the_fix(self) -> None:
        reason = unavailable_reason(".png")
        assert "RAG_VISION_MODEL" in reason

    def test_unknown_extension_reports_as_unrecognised(self) -> None:
        assert "not recognised" in unavailable_reason(".xyz")

    def test_extension_supported_accepts_bare_and_dotted(self) -> None:
        assert extension_supported("pdf") == extension_supported(".pdf")

    @pytest.mark.skipif(
        not markitdown_available(), reason="markitdown not installed"
    )
    def test_plain_text_formats_need_no_optional_extra(self) -> None:
        for ext in (".md", ".txt", ".csv", ".html", ".json"):
            assert extension_supported(ext), f"{ext} should always work"

    def test_available_extensions_have_no_reason(self) -> None:
        for status in resolve_statuses():
            if status.available:
                assert status.reason == ""


class TestSpecBehaviour:
    def test_probe_modules_defaults_to_modules(self) -> None:
        assert FormatSpec((".x",), "X", modules=("a",)).probe_modules == ("a",)

    def test_spec_without_modules_is_valid(self) -> None:
        spec = FormatSpec((".x",), "X")
        assert spec.probe_modules == ()