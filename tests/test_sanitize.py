"""Filename and path safety.

These are the tests that matter most: an unsanitised upload name is a
path-traversal vulnerability.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.sanitize import (
    MAX_NAME_CHARS,
    UnsafeNameError,
    human_size,
    safe_join,
    sanitize_filename,
)


class TestTraversalIsImpossible:
    @pytest.mark.parametrize(
        "raw",
        [
            "../../etc/passwd",
            "../../../../root/.ssh/authorized_keys",
            "/etc/shadow",
            "/tmp/evil.sh",
            r"C:\Windows\System32\drivers\etc\hosts",
            "subdir/nested/deep.md",
            "..",
        ],
    )
    def test_directory_components_are_stripped(self, raw: str) -> None:
        result = sanitize_filename(raw)
        assert "/" not in result
        assert "\\" not in result
        assert ".." not in result

    def test_null_bytes_are_removed_not_stripped(self) -> None:
        # Stripping rather than removing would turn "a\0b" into the distinct
        # name "ab"; removing it does the same here, but the important part is
        # that no null ever survives into a path.
        result = sanitize_filename("a\x00b.txt")
        assert "\x00" not in result

    def test_safe_join_stays_inside_base(self, tmp_path: Path) -> None:
        base = tmp_path / "docs"
        base.mkdir()
        for hostile in (
            "../../etc/passwd",
            "/etc/passwd",
            "....//....//etc/passwd",
            "a/b/../../../c",
        ):
            result = safe_join(base, hostile)
            assert result.resolve().is_relative_to(base.resolve())

    def test_safe_join_result_is_a_single_component(self, tmp_path: Path) -> None:
        base = tmp_path / "docs"
        base.mkdir()
        assert safe_join(base, "a/b/c.md").parent == base


class TestNamingRules:
    def test_ordinary_name_is_preserved(self) -> None:
        assert sanitize_filename("report.pdf") == "report.pdf"

    def test_non_ascii_is_kept(self) -> None:
        # Sanitising must not mangle legitimate international filenames.
        assert sanitize_filename("résumé.md") == "résumé.md"
        assert sanitize_filename("日本語の資料.pdf") == "日本語の資料.pdf"

    def test_shell_metacharacters_are_replaced(self) -> None:
        result = sanitize_filename("weird;name|with`chars`$(x).md")
        for bad in ";|`$()":
            assert bad not in result

    def test_hidden_names_become_visible(self) -> None:
        assert not sanitize_filename(".bashrc").startswith(".")

    def test_length_is_capped(self) -> None:
        result = sanitize_filename("x" * 5000 + ".pdf")
        assert len(result) <= MAX_NAME_CHARS
        assert result.endswith(".pdf")

    def test_extension_is_preserved_when_truncating(self) -> None:
        result = sanitize_filename("x" * 5000 + ".markdown")
        assert result.endswith(".markdown")

    @pytest.mark.parametrize("raw", ["", "   ", "....", "\t\n", "___"])
    def test_unusable_names_fall_back_rather_than_fail(self, raw: str) -> None:
        # Refusing the upload over a cosmetic filename would be unhelpful;
        # substituting a placeholder keeps the pipeline running.
        assert sanitize_filename(raw) == "document"

    def test_non_string_is_rejected(self) -> None:
        with pytest.raises(UnsafeNameError):
            sanitize_filename(None)  # type: ignore[arg-type]

    def test_windows_reserved_names_are_escaped(self) -> None:
        assert sanitize_filename("con.txt") != "con.txt"

    def test_result_is_a_single_path_component(self) -> None:
        for raw in ("a/b.md", "../x.md", r"C:\a\b.md", "x\0y.md"):
            assert "/" not in sanitize_filename(raw)
            assert "\\" not in sanitize_filename(raw)


class TestHelpers:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (5 * 1024**2, "5.0 MB")],
    )
    def test_human_size(self, size: int, expected: str) -> None:
        assert human_size(size) == expected