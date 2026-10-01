"""End-to-end harness.

Boots a complete, isolated stack per session:

* the OpenRouter stub (offline, deterministic, free)
* a real ``chroma run`` server on its own port with its own data directory
* both real Streamlit apps on their own ports, as real processes

Nothing touches the developer's installation: ports, data directory, collection
and ``.env`` values are all overridden through the environment. The apps
themselves are the production files, unmodified.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "e2e"))

from fake_openrouter import FakeOpenRouter  # noqa: E402

# Unusual ports so they cannot collide with the running installation
# (8888 / 8901 / 8902) or anything else on a developer machine.
CHROMA_PORT = 18888
QUERY_PORT = 18901
ADMIN_PORT = 18902
COLLECTION = "e2e_docs"

VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
CHROMA_BIN = ROOT / ".venv" / "bin" / "chroma"
STREAMLIT_BIN = ROOT / ".venv" / "bin" / "streamlit"

BOOT_TIMEOUT = 90


def free_port_ok(port: int) -> bool:
    """True when something is accepting connections on ``port``."""
    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_http(url: str, timeout: float = BOOT_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except urllib.error.HTTPError:
            # Any HTTP response means the server is up.
            return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.4)
    return False


class Service:
    """A child process that must be shut down."""

    def __init__(self, name: str, argv: list[str], env: dict, log: Path) -> None:
        self.name = name
        self.log_path = log
        self._log = log.open("wb")
        self.process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    @property
    def pid(self) -> int:
        return self.process.pid

    def logs(self) -> str:
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def is_running(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), 15)
            except (ProcessLookupError, PermissionError):
                self.process.terminate()
            try:
                self.process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), 9)
                except (ProcessLookupError, PermissionError):
                    self.process.kill()
        try:
            self._log.close()
        except OSError:
            pass

    def __enter__(self) -> "Service":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


@pytest.fixture(scope="session")
def e2e_stack(tmp_path_factory):
    """Start the stub, Chroma and both apps; yield their URLs."""
    if not VENV_PYTHON.exists():
        pytest.skip("no virtualenv; run ./scripts/setup.sh")

    for port in (CHROMA_PORT, QUERY_PORT, ADMIN_PORT):
        if free_port_ok(port):
            pytest.skip(
                f"port {port} is already in use; stop the other stack first"
            )

    workdir = tmp_path_factory.mktemp("e2e")
    data_dir = workdir / "data"
    (data_dir / "documents").mkdir(parents=True)
    (data_dir / "chroma").mkdir(parents=True)

    stub = FakeOpenRouter().start()

    env = dict(os.environ)
    env.update(
        {
            # Route the apps at the stub instead of the real API.
            "OPENROUTER_API_KEY": "sk-or-v1-e2e-stub-key",
            "OPENROUTER_BASE_URL": stub.base_url,
            "RAG_LLM_MODEL": "stub/chat",
            "RAG_EMBED_MODEL": "stub/embed",
            "RAG_VISION_MODEL": "",
            # Isolated storage.
            "RAG_DATA_DIR": str(data_dir),
            "RAG_DOCS_DIR": str(data_dir / "documents"),
            "CHROMA_DIR": str(data_dir / "chroma"),
            "CHROMA_HOST": "127.0.0.1",
            "CHROMA_PORT": str(CHROMA_PORT),
            "CHROMA_COLLECTION": COLLECTION,
            # Speed: the stub is instant, so small batches are fine.
            # Small threshold so the large-document guidance can be exercised
            # without writing a 25 MB fixture.
            "LARGE_DOCUMENT_MB": "1",
            "INDEX_WINDOW_CHUNKS": "64",
            "EMBED_BATCH_SIZE": "8",
            "EMBED_MAX_RETRIES": "1",
            # Keep the test apps on loopback regardless of the developer's .env.
            "APP_BIND_ADDRESS": "127.0.0.1",
            "PYTHONUNBUFFERED": "1",
            "STREAMLIT_SERVER_HEADLESS": "true",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        }
    )

    logs = workdir / "logs"
    logs.mkdir()
    services: list[Service] = []

    def launch(name: str, argv: list[str]) -> Service:
        service = Service(name, argv, env, logs / f"{name}.log")
        services.append(service)
        return service

    try:
        chroma = launch(
            "chroma",
            [
                str(CHROMA_BIN), "run",
                "--path", str(data_dir / "chroma"),
                "--host", "127.0.0.1",
                "--port", str(CHROMA_PORT),
            ],
        )
        if not wait_for_http(f"http://127.0.0.1:{CHROMA_PORT}/api/v2/heartbeat"):
            pytest.fail(f"chroma never became ready:\n{chroma.logs()}")

        admin = launch(
            "admin",
            [
                str(STREAMLIT_BIN), "run", str(ROOT / "app" / "admin_app.py"),
                "--server.port", str(ADMIN_PORT),
                "--server.address", "127.0.0.1",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
            ],
        )
        query = launch(
            "query",
            [
                str(STREAMLIT_BIN), "run", str(ROOT / "app" / "query_app.py"),
                "--server.port", str(QUERY_PORT),
                "--server.address", "127.0.0.1",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
            ],
        )

        for name, port in (("admin", ADMIN_PORT), ("query", QUERY_PORT)):
            if not wait_for_http(f"http://127.0.0.1:{port}/"):
                service = admin if name == "admin" else query
                pytest.fail(f"{name} app never became ready:\n{service.logs()}")

        yield {
            "query_url": f"http://127.0.0.1:{QUERY_PORT}",
            "admin_url": f"http://127.0.0.1:{ADMIN_PORT}",
            "docs_dir": data_dir / "documents",
            "collection": COLLECTION,
            "stub_base_url": stub.base_url,
            "logs": logs,
        }
    finally:
        for service in reversed(services):
            service.stop()
        stub.stop()


@pytest.fixture(scope="session")
def page_factory(e2e_stack):
    """Yield a Playwright page factory, skipping the session if unavailable."""
    playwright = pytest.importorskip(
        "playwright.sync_api", reason="playwright not installed"
    )
    try:
        with playwright.sync_playwright() as driver:
            browser = driver.chromium.launch(
                channel="chrome",  # use the system Chrome, no download needed
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            context = browser.new_context(viewport={"width": 1600, "height": 1100})

            def open_page(url: str):
                page = context.new_page()
                page.set_default_timeout(20_000)
                page.goto(url, wait_until="domcontentloaded")
                return page

            try:
                yield open_page
            finally:
                context.close()
                browser.close()
    except Exception as exc:  # noqa: BLE001
        if "Executable doesn't exist" in str(exc):
            pytest.skip("no browser available: install Chrome or run "
                        "`playwright install chromium`")
        raise


@pytest.fixture
def admin_page(page_factory, e2e_stack):
    page = page_factory(e2e_stack["admin_url"])
    yield page
    page.close()


@pytest.fixture
def query_page(page_factory, e2e_stack):
    page = page_factory(e2e_stack["query_url"])
    yield page
    page.close()


# ── page helpers ───────────────────────────────────────────────────────────


def wait_for_idle(page, settle_ms: int = 900) -> None:
    """Wait for any in-flight Streamlit run to finish.

    Streamlit shows a status widget while the script is running. Immediately
    after an interaction the widget may not exist yet, so wait briefly for it to
    appear before waiting for it to clear -- otherwise this returns before the
    run has even started, which is a race.
    """
    page.wait_for_timeout(settle_ms)
    status = page.locator('[data-testid="stStatusWidget"]')
    try:
        status.first.wait_for(state="attached", timeout=2_000)
    except Exception:
        pass
    try:
        page.wait_for_function(
            "() => { const w = document.querySelector("
            "'[data-testid=\"stStatusWidget\"]');"
            " return !w || w.innerText.trim() === ''; }",
            timeout=60_000,
        )
    except Exception:
        pass
    page.wait_for_timeout(settle_ms)


def wait_for_app(page, marker: str = "Upload documents") -> None:
    """Wait until Streamlit has rendered the app and finished its run."""
    page.wait_for_selector(f"text={marker}", state="visible", timeout=30_000)
    wait_for_idle(page)


def wait_for_query(page) -> None:
    """Wait for the query app, identified by its own heading rather than the
    admin app's."""
    wait_for_app(page, "Document Assistant")


def button(page, label: str):
    return page.get_by_role("button", name=label, exact=False)


def click(page, label: str) -> None:
    button(page, label).first.click()


def upload(page, paths: list[Path]) -> None:
    """Attach files to the uploader and wait for them to be accepted."""
    page.wait_for_selector('input[type="file"]', state="attached", timeout=30_000)
    page.locator('input[type="file"]').first.set_input_files(
        [str(p) for p in paths]
    )
    page.wait_for_selector("text=/Saved .* file/", timeout=60_000)
    wait_for_idle(page)


def assert_no_app_error(page) -> None:
    """Fail if Streamlit rendered an error banner."""
    errors = page.locator('[data-testid="stException"], [data-testid="stAlertContainer"]')
    for index in range(errors.count()):
        text = (errors.nth(index).inner_text() or "").strip()
        if text and "error" not in text.lower():
            continue
        if "Traceback" in text or "Error" in text or "error" in text:
            raise AssertionError(f"app rendered an error: {text[:400]}")


def make_documents(directory: Path) -> list[Path]:
    """Write a few small documents and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name, title, body in [
        ("network.md", "Network Setup", "Set a static address on the interface. "
         "Use ifconfig or the ip command to configure the address and netmask."),
        ("packages.md", "Package Management", "Add a package to the image with "
         "IMAGE_INSTALL:append. Include recipes in the layer directory."),
        ("booting.md", "Boot Sequence", "The bootloader loads the kernel and "
         "hands control to procnto. Startup scripts run afterwards."),
    ]:
        path = directory / name
        path.write_text(
            f"# {title}\n\n{body}\n\n## Details\n\n" + f"{body} " * 20 + "\n"
        )
        written.append(path)

    return written


def collection_count(chroma_dir: Path, collection: str) -> int:
    """Chunk count for a collection, via the same HTTP API the apps use.

    Reading Chroma's on-disk SQLite directly is brittle -- the schema is an
    implementation detail that differs between Chroma versions -- so this goes
    through the server the same way the application does.
    """
    import chromadb

    client = chromadb.HttpClient(host="127.0.0.1", port=CHROMA_PORT)
    try:
        handle = client.get_collection(name=collection, embedding_function=None)
    except Exception:
        return 0
    try:
        return int(handle.count())
    finally:
        pass
