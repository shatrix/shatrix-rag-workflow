"""A minimal stand-in for the OpenRouter HTTP API.

The end-to-end tests drive the real Streamlit apps in real browser processes, so
they cannot monkeypatch the OpenRouter client in-process the way the unit tests
do. Instead this serves the two endpoints the apps actually call, over the same
wire protocol the ``openai`` SDK speaks:

* ``POST /api/v1/embeddings``
* ``POST /api/v1/chat/completions`` (streaming and non-streaming)

That keeps the tests offline, instant and free, while still exercising the real
HTTP client, the real JSON serialisation and the real streaming parser -- which
is where bugs like the payload-too-large failure actually lived.

Embeddings are lexical: a fixed-width bag of hashed words. Deterministic, and
similar enough that retrieval returns sensible passages for assertions.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EMBED_DIMS = 64
ANSWER_MARKER = "STUB-ANSWER"

_WORD = re.compile(r"[a-z0-9_]+")


def embed_text(text: str, dims: int = EMBED_DIMS) -> list[float]:
    """Deterministic lexical embedding: hashed word counts, L2 normalised."""
    vector = [0.0] * dims
    for word in _WORD.findall(text.lower()):
        index = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % dims
        vector[index] += 1.0
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


class Handler(BaseHTTPRequestHandler):
    # Silence per-request logging; the test output is enough.
    def log_message(self, *args):  # noqa: D102
        pass

    # ── helpers ──────────────────────────────────────────────────────────
    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _first_text(self, messages: list[dict]) -> str:
        for message in reversed(messages or []):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
        return ""

    # ── routes ───────────────────────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802
        if self.path.endswith("/key"):
            self._json(
                {
                    "data": {
                        "label": "e2e-stub",
                        "usage": 0.0,
                        "limit": None,
                        "is_free_tier": True,
                        "rate_limit": {"requests": -1, "interval": "10s"},
                    }
                }
            )
        elif "/embeddings/models" in self.path:
            self._json(
                {
                    "data": [
                        {
                            "id": "stub/embed",
                            "name": "Stub Embed",
                            "context_length": 8192,
                            "pricing": {"prompt": "0", "completion": "0"},
                            "architecture": {"modality": "text->embeddings"},
                        }
                    ]
                }
            )
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.endswith("/embeddings"):
            self._embeddings()
        elif self.path.endswith("/chat/completions"):
            self._chat()
        else:
            self._json({"error": "not found"}, status=404)

    def _embeddings(self) -> None:
        payload = self._read_json()
        raw = payload.get("input", "")
        inputs = [raw] if isinstance(raw, str) else list(raw)
        # Honour a reduced dimension request, as the real API does.
        dims = int(payload.get("dimensions") or EMBED_DIMS)

        self._json(
            {
                "object": "list",
                "model": payload.get("model", "stub/embed"),
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": embed_text(text, dims),
                    }
                    for i, text in enumerate(inputs)
                ],
                "usage": {
                    "prompt_tokens": sum(len(t.split()) for t in inputs),
                    "total_tokens": sum(len(t.split()) for t in inputs),
                },
            }
        )

    def _chat(self) -> None:
        payload = self._read_json()
        messages = payload.get("messages") or []
        prompt = self._first_text(messages)
        wants_stream = bool(payload.get("stream"))

        # Echo a marker plus the leading words of the question, so a test can
        # assert the retrieved context actually reached the model.
        question = ""
        for line in prompt.splitlines():
            if line.startswith("User Question:"):
                question = line.split(":", 1)[1].strip()
                break
        answer = f"{ANSWER_MARKER} {question}".strip()

        if not wants_stream:
            self._json(
                {
                    "id": "stub-1",
                    "object": "chat.completion",
                    "model": payload.get("model", "stub/chat"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": answer},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(answer.split()),
                        "cost": 0.0,
                    },
                }
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def chunk(delta: dict, finish: str | None = None) -> str:
            body = {
                "id": "stub-1",
                "object": "chat.completion.chunk",
                "model": payload.get("model", "stub/chat"),
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(body)}\n\n"

        try:
            # Stream word by word so the client-side reassembly is exercised.
            for index, word in enumerate(answer.split()):
                text = word if index == 0 else " " + word
                self.wfile.write(chunk({"content": text}).encode())
                self.wfile.flush()
            self.wfile.write(chunk({}, "stop").encode())

            usage = {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(answer.split()),
                "cost": 0.0,
            }
            final = {
                "id": "stub-1",
                "object": "chat.completion.chunk",
                "model": payload.get("model", "stub/chat"),
                "choices": [],
                "usage": usage,
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class FakeOpenRouter:
    """Context manager running the stub on an ephemeral port."""

    def __init__(self) -> None:
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        assert self.server is not None
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/api/v1"

    def start(self) -> "FakeOpenRouter":
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def __enter__(self) -> "FakeOpenRouter":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


if __name__ == "__main__":  # manual smoke test
    import urllib.request

    stub = FakeOpenRouter().start()
    try:
        body = json.dumps(
            {"model": "stub/embed", "input": ["hello world"]}
        ).encode()
        request = urllib.request.Request(
            f"{stub.base_url}/embeddings", data=body,
            headers={"Content-Type": "application/json"},
        )
        print("embedding dims:",
              len(json.load(urllib.request.urlopen(request))["data"][0]["embedding"]))
        print("base_url:", stub.base_url)
    finally:
        stub.stop()