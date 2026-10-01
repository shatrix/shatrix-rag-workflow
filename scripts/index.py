#!/usr/bin/env python3
"""Index documents from the command line.

The admin interface is convenient for a handful of small files, but a large
document produces thousands of chunks and can take many minutes. A long-running
job inside a Streamlit session is fragile: the run is tied to a browser
connection, and closing the tab or a reconnect can cancel it. This script does
the same work outside Streamlit, shows live progress, and can be run in the
background.

    ./scripts/index.py                     # index everything unindexed
    ./scripts/index.py --file path.pdf     # one specific file
    ./scripts/index.py --force            # ignore change detection
    ./scripts/index.py --dry-run          # report what would happen

Run it detached for a big job:

    nohup .venv/bin/python scripts/index.py --file data/documents/big.pdf \\
        > /tmp/index.log 2>&1 &

The Chroma server must be running; either via
``systemctl --user start rag-chroma.service`` or ``./scripts/run-local.sh``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.formats import supported_extensions  # noqa: E402
from app.indexing import index_file  # noqa: E402
from app.openrouter import OpenRouterClient  # noqa: E402
from app.vectorstore import (  # noqa: E402
    VectorStoreError,
    connect,
    count,
    open_collection,
    server_batch_size,
    stored_hashes,
)

_TTY = sys.stdout.isatty()
#: ANSI reset, empty when not writing to a terminal so redirected logs stay clean.
RESET = "\033[0m" if _TTY else ""


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


BOLD = lambda t: _c("1", t)  # noqa: E731
DIM = lambda t: _c("2", t)  # noqa: E731
GREEN = lambda t: _c("32", t)  # noqa: E731
YELLOW = lambda t: _c("33", t)  # noqa: E731
RED = lambda t: _c("31", t)  # noqa: E731
BLUE = lambda t: _c("34", t)  # noqa: E731


class Progress:
    """Prints the most recent message, with elapsed time and chunk counts."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.last = ""

    def __call__(self, message: str) -> None:
        if message == self.last:
            return
        self.last = message
        elapsed = time.perf_counter() - self.started
        stamp = DIM(f"{elapsed:6.1f}s") if _TTY else f"{elapsed:6.1f}s"
        print(f"  {stamp}  {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file", action="append", default=[],
                        help="index this file (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="re-index even if unchanged")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be indexed, write nothing")
    parser.add_argument("--window", type=int, default=None,
                        help="chunks per embed+write pass "
                             "(default from INDEX_WINDOW_CHUNKS)")
    args = parser.parse_args()

    settings = get_settings()

    print(BOLD("rag-workflow indexer"))
    print(DIM(f"  documents  {settings.docs_dir}"))
    print(DIM(f"  embed      {settings.embed_model}"))
    print(DIM(f"  collection {settings.chroma_collection}"))

    # ── pick targets ───────────────────────────────────────────────────────
    targets: list[Path] = []
    if args.file:
        for raw in args.file:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.is_file():
                print(RED(f"error: not a file: {path}"))
                return 2
            targets.append(path)
    else:
        allowed = set(
            supported_extensions(vision_enabled=bool(settings.vision_model))
        )
        if settings.docs_dir.exists():
            targets = sorted(
                p for p in settings.docs_dir.iterdir()
                if p.is_file()
                and not p.name.startswith(".")
                and p.suffix.lower() in allowed
            )

    if not targets:
        print(YELLOW("nothing to index"))
        return 0

    total_mb = sum(p.stat().st_size for p in targets) / 1024 / 1024
    print(
        f"  {len(targets)} file(s), {total_mb:.1f} MB total\n"
        + "".join(f"    {DIM('*' if p.stat().st_size else ' ')} {p.name} "
                  f"{DIM(f'({p.stat().st_size / 1024 / 1024:.1f} MB)')}\n"
                  for p in targets)
    )

    if args.dry_run:
        print(YELLOW("dry run: nothing written"))
        return 0

    # ── connect ────────────────────────────────────────────────────────────
    try:
        client = connect(settings)
        collection = open_collection(client, settings, create=True)
    except VectorStoreError as exc:
        print(RED(f"\nerror: {exc}"))
        return 1

    try:
        batch_size = server_batch_size(client)
    except VectorStoreError:
        batch_size = None

    router = OpenRouterClient()
    progress = Progress()
    hashes = {} if args.force else stored_hashes(collection)

    print(f"\n{BOLD('indexing')}{RESET}  (window={args.window or settings.index_window_chunks} chunks,"
          f" server batch={batch_size})")
    print(DIM("-" * 68))

    indexed = unchanged = skipped = failed = 0
    chunks_before = count(collection)

    for path in targets:
        print(f"\n{BLUE(path.name)}{RESET}  "
              f"{DIM(f'{path.stat().st_size / 1024 / 1024:.1f} MB')}")
        outcome = index_file(
            path,
            collection,
            router,
            settings=settings,
            existing_hashes=hashes,
            force=args.force,
            batch_size=batch_size,
            window_chunks=args.window,
            progress=progress,
        )
        if outcome.content_hash:
            hashes[path.name] = outcome.content_hash

        mark = {
            "indexed": GREEN("OK  "),
            "unchanged": DIM("--  "),
            "skipped": YELLOW("SKIP"),
            "failed": RED("FAIL"),
        }.get(outcome.status, "?")
        print(f"  {mark} {outcome.detail} {DIM(f'({outcome.elapsed:.1f}s)')}")
        for warning in outcome.warnings[:5]:
            print(f"       {YELLOW('!')} {warning}")
        if outcome.status == "failed":
            failed += 1
        elif outcome.status == "indexed":
            indexed += 1
        elif outcome.status == "unchanged":
            unchanged += 1
        else:
            skipped += 1

    chunks_after = count(collection)
    print("\n" + DIM("-" * 68))
    print(
        f"{BOLD('done')}{RESET}  {indexed} indexed · {unchanged} unchanged · "
        f"{skipped} skipped · {failed} failed"
    )
    print(
        f"  collection {chunks_before:,} -> {chunks_after:,} chunks "
        f"(+{chunks_after - chunks_before:,})"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())