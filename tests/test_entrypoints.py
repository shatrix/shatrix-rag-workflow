"""Entry-point import safety.

Regression guard for a real bug: ``streamlit run app/query_app.py`` failed
with ``ModuleNotFoundError: No module named 'app'`` because Python places the
*script's* directory (``app/``) on ``sys.path[0]``, so ``import app.config``
looked for ``app/app/config.py``.

The HTTP health check did not catch it -- Streamlit serves its static shell
with a 200 while the script crashes on every session -- and neither did the
usual test harness, which runs with the repository root already importable.

These tests reproduce the failure mode by running each entry point as a real
file from an unrelated working directory with a clean environment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = [ROOT / "app" / "query_app.py", ROOT / "app" / "admin_app.py"]


def run_entrypoint(script: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    """Execute the script the way Python does when handed a file path.

    Running from ``tmp_path`` and stripping PYTHONPATH guarantees the
    repository root is *not* importable for free, so only the script's own
    bootstrap can make ``import app.*`` work.
    """
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            # Deliberately minimal: no PYTHONPATH, no inherited env config.
            "ANONYMIZED_TELEMETRY": "False",
        },
    )


@pytest.mark.parametrize("script", ENTRY_POINTS, ids=lambda p: p.name)
def test_entrypoint_imports_when_run_as_a_file(
    script: Path, tmp_path: Path
) -> None:
    result = run_entrypoint(script, tmp_path)
    combined = result.stdout + result.stderr

    assert "ModuleNotFoundError" not in combined, (
        f"{script.name} could not import its own package when run as a file:\n"
        f"{combined[-2000:]}"
    )
    assert "ImportError" not in combined, (
        f"{script.name} raised an ImportError:\n{combined[-2000:]}"
    )


@pytest.mark.parametrize("script", ENTRY_POINTS, ids=lambda p: p.name)
def test_entrypoint_puts_the_repo_root_on_sys_path(
    script: Path, tmp_path: Path
) -> None:
    # Assert the bootstrap explicitly rather than inferring it from a clean run.
    source = script.read_text(encoding="utf-8")
    assert "sys.path" in source, (
        f"{script.name} has no sys.path bootstrap; it will fail when launched "
        f"by anything that does not already have the repo root importable"
    )
    assert "parents[1]" in source, (
        f"{script.name} should derive the repo root from __file__ so the "
        f"project stays relocatable"
    )


@pytest.mark.parametrize("script", ENTRY_POINTS, ids=lambda p: p.name)
def test_bootstrap_precedes_app_imports(script: Path) -> None:
    """The bootstrap is useless if it sits below the imports that need it."""
    source = script.read_text(encoding="utf-8")
    bootstrap_at = source.index("sys.path.insert")
    first_app_import = source.index("from app.")
    assert bootstrap_at < first_app_import, (
        f"{script.name} imports from `app` before extending sys.path"
    )


def test_ui_module_no_longer_carries_dead_code() -> None:
    # inject_path() was written to solve this bug but could never run in time,
    # because the imports that needed it came first. Guard against it creeping
    # back in as an unused helper.
    ui = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    assert "def inject_path" not in ui