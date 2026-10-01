#!/usr/bin/env python3
"""Preflight checks for a rag-workflow installation.

Run this after setup and whenever something is not working. It checks, in
order, that can be checked cheaply first, so the first failure reported is
usually also the first thing that needs fixing.

    ./scripts/doctor.py            # everything
    ./scripts/doctor.py --offline  # skip calls to OpenRouter
    ./scripts/doctor.py --json     # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── terminal formatting ────────────────────────────────────────────────────

_TTY = sys.stdout.isatty()


def _colour(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


BOLD = lambda t: _colour("1", t)  # noqa: E731
GREEN = lambda t: _colour("32", t)  # noqa: E731
YELLOW = lambda t: _colour("33", t)  # noqa: E731
RED = lambda t: _colour("31", t)  # noqa: E731
DIM = lambda t: _colour("2", t)  # noqa: E731

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, level: str, name: str, detail: str = "") -> None:
        self.rows.append({"level": level, "check": name, "detail": detail})

    def ok(self, name: str, detail: str = "") -> None:
        self.add(PASS, name, detail)

    def warn(self, name: str, detail: str = "") -> None:
        self.add(WARN, name, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self.add(FAIL, name, detail)

    def info(self, name: str, detail: str = "") -> None:
        self.add(INFO, name, detail)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if r["level"] == FAIL)

    @property
    def warned(self) -> int:
        return sum(1 for r in self.rows if r["level"] == WARN)

    def render(self) -> None:
        icons = {PASS: GREEN("  ok  "), WARN: YELLOW(" warn "), FAIL: RED(" FAIL "), INFO: DIM(" info ")}
        width = max((len(r["check"]) for r in self.rows), default=0)
        for row in self.rows:
            label = icons.get(row["level"], "      ")
            name = row["check"].ljust(width)
            if row["detail"]:
                print(f"{label} {name}  {DIM(row['detail'])}")
            else:
                print(f"{label} {name}")


def port_in_use(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ── checks ─────────────────────────────────────────────────────────────────


def check_python(report: Report) -> None:
    v = sys.version_info
    if (v.major, v.minor) < (3, 10):
        report.fail(
            "Python version",
            f"{v.major}.{v.minor} is too old; 3.10 or newer is required",
        )
    elif (v.major, v.minor) > (3, 14):
        report.warn(
            "Python version",
            f"{v.major}.{v.minor} is newer than tested; some wheels may not "
            f"exist yet",
        )
    else:
        report.ok("Python version", f"{v.major}.{v.minor}.{v.micro}")


def check_executable(report: Report) -> None:
    exe = Path(sys.executable)
    if ".venv" in exe.parts:
        report.ok("Virtual environment", str(exe.parent.parent))
    else:
        report.warn(
            "Virtual environment",
            f"not running from .venv ({exe}); run ./scripts/setup.sh",
        )


def check_packages(report: Report) -> None:
    required = {
        "streamlit": "Streamlit UI",
        "chromadb": "vector store client",
        "openai": "OpenRouter transport",
        "dotenv": ".env loading",
    }
    optional = {
        "markitdown": "document conversion",
        "pdfminer": "PDF support",
        "mammoth": "Word support",
        "pptx": "PowerPoint support",
        "openpyxl": "Excel support",
        "olefile": "Outlook support",
    }

    missing_required = []
    for module, purpose in required.items():
        try:
            __import__(module)
        except ImportError:
            missing_required.append(f"{module} ({purpose})")
    if missing_required:
        report.fail(
            "Required packages",
            "missing: " + ", ".join(missing_required)
            + " -- run ./scripts/setup.sh",
        )
    else:
        report.ok("Required packages", f"{len(required)}/{len(required)} present")

    missing_optional = []
    for module, purpose in optional.items():
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(f"{module} ({purpose})")
    if missing_optional:
        report.warn(
            "Optional converters",
            "not installed: " + ", ".join(missing_optional),
        )
    else:
        report.ok("Optional converters", f"{len(optional)}/{len(optional)} present")


def check_config(report: Report) -> bool:
    from app.config import ENV_FILE, get_settings, redact

    settings = get_settings()
    report.info("Project root", str(ENV_FILE.parent))
    report.info("Documents folder", str(settings.docs_dir))
    report.info("Chroma data folder", str(settings.chroma_dir))

    if ENV_FILE.exists():
        report.ok(".env file", str(ENV_FILE))
    else:
        report.fail(
            ".env file",
            f"missing {ENV_FILE} -- copy .env.example to .env, or run "
            f"./scripts/setup.sh",
        )
        return False

    key = settings.openrouter_api_key
    if not key:
        report.fail(
            "API key",
            "OPENROUTER_API_KEY is empty in .env -- get one at "
            "https://openrouter.ai/keys",
        )
        return False
    if "your" in key.lower() and len(key) < 40:
        report.fail(
            "API key",
            "still looks like the placeholder from .env.example",
        )
        return False
    report.ok("API key", redact(key))
    return True


def check_models(report: Report) -> None:
    from app.config import get_settings

    settings = get_settings()
    report.info("LLM model", settings.llm_model)
    report.info("Embedding model", f"{settings.embed_model} "
              f"({settings.embed_dims or 'unknown'} dims)")
    if settings.chunk_truncated:
        report.warn(
            "Chunk size",
            f"{settings.chunk_chars} chars exceeds the "
            f"{settings.max_chunk_chars} this model accepts; it will be clamped",
        )
    else:
        report.ok(
            "Chunk size",
            f"{settings.chunk_chars} chars (model allows "
            f"~{settings.embed_max_input_tokens} tokens per input)",
        )
    if settings.embed_dims == 0:
        report.warn(
            "Embedding model",
            "not in the known-models table, so its vector width is unknown; "
            "switching models later will require a full re-index",
        )


def check_paths(report: Report) -> None:
    from app.config import get_settings

    settings = get_settings()
    for label, path in (
        ("Documents folder", settings.docs_dir),
        ("Chroma data folder", settings.chroma_dir),
    ):
        if path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                probe = path / ".write-probe"
                probe.touch()
                probe.unlink()
                report.ok(label, f"{path} (writable)")
            except OSError as exc:
                report.fail(label, f"{path} is not writable: {exc}")
        else:
            report.warn(label, f"{path} does not exist yet; it will be created")


def check_ports(report: Report) -> None:
    from app.config import get_settings

    settings = get_settings()
    for label, host, port, should_run in (
        ("Chroma server", settings.chroma_host, settings.chroma_port, True),
        ("Query app", "127.0.0.1", settings.query_port, False),
        ("Admin app", "127.0.0.1", settings.admin_port, False),
    ):
        listening = port_in_use(host, port)
        if listening:
            report.ok(f"{label} port {port}", "in use")
        elif should_run:
            report.warn(
                f"{label} port {port}",
                "nothing is listening -- start it with ./scripts/run-local.sh "
                "or ./scripts/install-services.sh",
            )
        else:
            report.ok(f"{label} port {port}", "free")


def check_chroma(report: Report) -> None:
    from app.config import get_settings
    from app.vectorstore import (
        VectorStoreError,
        collection_exists,
        connect,
        count,
        dimension_warning,
        open_collection,
        server_batch_size,
    )

    settings = get_settings()
    try:
        client = connect(settings)
    except VectorStoreError as exc:
        report.fail("Chroma connection", str(exc).splitlines()[0])
        return

    report.ok("Chroma connection", f"{settings.chroma_host}:{settings.chroma_port}")
    report.info("Chroma max batch", str(server_batch_size(client)))

    if not collection_exists(client, settings.chroma_collection):
        report.info(
            "Collection",
            f"'{settings.chroma_collection}' does not exist yet; the admin "
            f"app will create it on first use",
        )
        return

    collection = open_collection(client, settings, create=False)
    report.ok("Collection", f"{settings.chroma_collection}")
    report.ok("Indexed chunks", f"{count(collection):,}")

    warning = dimension_warning(collection, settings)
    if warning:
        report.fail("Model match", warning.split(".")[0])
    else:
        report.ok("Model match", "collection matches RAG_EMBED_MODEL")


def check_openrouter(report: Report) -> None:
    from app.openrouter import (
        OpenRouterClient,
        build_client,
        check_key,
        probe_chat,
        probe_embedding,
    )

    try:
        client = build_client()
    except Exception as exc:  # noqa: BLE001
        report.fail("OpenRouter client", str(exc))
        return

    try:
        info = check_key(client)
        tier = "free tier" if info["is_free_tier"] else "paid"
        limit = info.get("limit")
        limit_text = f"{limit:,} requests" if isinstance(limit, (int, float)) else ""
        report.ok("API key accepted", f"{info['label']} ({tier}) {limit_text}")
        rl = info.get("rate_limit")
        if isinstance(rl, dict):
            report.info(
                "Rate limits",
                "; ".join(f"{k}={v}" for k, v in rl.items()),
            )
    except Exception as exc:  # noqa: BLE001
        report.fail("API key accepted", str(exc))
        return

    router = OpenRouterClient()

    try:
        dims, seconds = probe_embedding(router)
        report.ok(
            "Embedding endpoint",
            f"{dims} dimensions in {seconds:.2f}s",
        )
        from app.config import get_settings

        expected = get_settings().embed_dims
        if expected and dims != expected:
            report.warn(
                "Embedding dimensions",
                f"API returned {dims}, expected {expected}; re-index required",
            )
    except Exception as exc:  # noqa: BLE001
        report.fail("Embedding endpoint", str(exc))

    try:
        text, seconds = probe_chat(router)
        report.ok(
            "Chat endpoint",
            f"replied in {seconds:.2f}s" + (f" ({text!r})" if text else ""),
        )
    except Exception as exc:  # noqa: BLE001
        report.fail("Chat endpoint", str(exc))


# ── entry point ────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip every check that calls OpenRouter",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of text"
    )
    args = parser.parse_args()

    report = Report()

    check_python(report)
    check_executable(report)
    check_packages(report)
    key_ok = check_config(report)
    check_models(report)
    check_paths(report)
    check_ports(report)
    check_chroma(report)

    if not args.offline and key_ok:
        check_openrouter(report)
    elif args.offline:
        report.info("OpenRouter", "skipped (--offline)")

    if args.json:
        print(
            json.dumps(
                {
                    "checks": report.rows,
                    "failed": report.failed,
                    "warned": report.warned,
                    "ok": report.failed == 0,
                },
                indent=2,
            )
        )
    else:
        print(BOLD("rag-workflow doctor"))
        print(DIM("=" * 60))
        report.render()
        print(DIM("=" * 60))
        if report.failed:
            print(RED(f"{report.failed} check(s) failed") + f", {report.warned} warning(s)")
        elif report.warned:
            print(YELLOW(f"all checks passed, {report.warned} warning(s)"))
        else:
            print(GREEN("all checks passed"))

    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())