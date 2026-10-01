# rag-workflow

A self-hosted retrieval-augmented-generation (RAG) system for your own
documents. Two web interfaces, one vector store, and **no local model at all** —
both the chat model and the embedding model are served remotely through the
[OpenRouter](https://openrouter.ai) API.

- **Admin app** — upload documents, watch them get converted to Markdown,
  chunked, embedded and indexed. See exactly what is in the index and delete
  it again.
- **Query app** — ask questions and get answers grounded in your documents,
  with citations you can check and a reasoning trace you can switch off.

Runs entirely on your own machine, binds to `127.0.0.1` by default, and costs
nothing to run: every model it uses is a free OpenRouter endpoint. See
[models](#models).

---

## Table of contents

- [Why it looks like this](#why-it-looks-like-this)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Run without systemd](#run-without-systemd)
- [Install as background services](#install-as-background-services)
- [Configuration](#configuration)
- [Supported file formats](#supported-file-formats)
- [How indexing works](#how-indexing-works)
- [Changing models](#changing-models)
- [Cost and rate limits](#cost-and-rate-limits)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Development](#development)
- [License](#license)

---

## Why it looks like this

A short rationale, since several choices here are unusual:

**No local model runtime.** OpenRouter serves both `/chat/completions` and
`/embeddings` on one OpenAI-compatible base URL, so the only runtime
dependencies are `streamlit`, `chromadb`, `openai`, `python-dotenv` and
`markitdown`. Nothing is downloaded, there is no GPU requirement, and there is
no multi-gigabyte deep-learning stack to install — a clone, one setup script and
a free API key is the whole provisioning story.

**A Chroma *server*, not embedded mode.** Two processes write to and read from
the same index. Chroma's local persistence mode does not support that safely,
so the two Streamlit apps talk to a `chroma run` server over HTTP. That is the
one structural reason there are three services instead of two.

**Deterministic chunk ids.** A chunk's id is derived from the SHA-256 of its
source file. Re-indexing identical content therefore *overwrites in place*
rather than appending duplicates, so repeatedly re-indexing a document can never
inflate the collection.

**Nothing is blocked on being large.** Uploads are converted by markitdown,
which handles PDF, Word, PowerPoint, Excel, Outlook, HTML, EPUB, notebooks and
archives. The admin page shows you which formats your installation can
actually finish processing, by probing what is installed rather than
hard-coding a list.

---

## Requirements

| Requirement | Notes |
|---|---|
| Linux | `systemd` user services and the shell scripts assume it |
| Python 3.10 – 3.14 | 3.11 or 3.12 recommended; the setup script finds one for you |
| An OpenRouter API key | Free from [openrouter.ai/keys](https://openrouter.ai/keys) |
| ~1 GB disk | The virtualenv, plus your documents and index |
| `systemd` | Only for the background-service option |

No GPU. No Docker. No root.

---

## Quick start

```bash
git clone <your-fork-url> rag-workflow
cd rag-workflow

./scripts/setup.sh          # creates .venv, installs deps, writes .env
./scripts/run-local.sh      # starts all three services in the foreground
```

`setup.sh` asks for your OpenRouter key and saves it to `.env`. If you skip
that, edit `.env` afterwards and set `OPENROUTER_API_KEY`.

Then:

1. Open **http://localhost:8902** — the admin app.
2. Upload some documents and press **Re-scan the documents folder**.
3. Open **http://localhost:8901** — the query app. Ask something.

Ports are deliberately unconventional to reduce the chance of colliding with
something else on your machine:

| Service | Port |
|---|---|
| ChromaDB | 8888 |
| Query app | 8901 |
| Admin app | 8902 |

Verify the installation at any time:

```bash
.venv/bin/python scripts/doctor.py
```

---

## Run without systemd

Useful for development, or if you would rather not install anything.

### Option A — one command

```bash
./scripts/run-local.sh
```

Starts Chroma, waits for it to accept connections, then starts both UIs. Logs
are prefixed per service (`[chroma]`, `[query]`, `[admin]`). `Ctrl-C` stops
everything.

```bash
./scripts/run-local.sh --only query     # just the query app
./scripts/run-local.sh --only admin     # just the admin app
./scripts/run-local.sh --skip-chroma    # Chroma is already running elsewhere
```

### Option B — separate terminals

Use the project virtualenv explicitly. Do not rely on an activated shell: the
services need the same interpreter regardless of what your shell has active.

**Terminal 1 — the vector store**

```bash
cd /path/to/rag-workflow
.venv/bin/chroma run --path data/chroma --host 127.0.0.1 --port 8888
```

Wait for `Frontend server listening on address, addr: 127.0.0.1:8888`.

**Terminal 2 — the query app**

```bash
cd /path/to/rag-workflow
.venv/bin/streamlit run app/query_app.py \
    --server.port 8901 \
    --server.address 127.0.0.1 \
    --server.headless true
```

**Terminal 3 — the admin app**

```bash
cd /path/to/rag-workflow
.venv/bin/streamlit run app/admin_app.py \
    --server.port 8902 \
    --server.address 127.0.0.1 \
    --server.headless true
```

### Option C — tmux

```bash
tmux new -s rag
# inside tmux, run the three commands above with Ctrl-B % to split windows
```

### Order matters

Chroma must be up before the apps are useful. Both apps refuse to render a
query interface without it and say so, but starting it first is smoother.

### Health checks

```bash
curl -s http://127.0.0.1:8888/api/v2/heartbeat   # -> {"nanosecond heartbeat":...}
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8901   # -> 200
.venv/bin/python scripts/doctor.py                # checks everything
```

### Stopping

`Ctrl-C` in each terminal, or:

```bash
pkill -f "[c]hroma run --path data/chroma"
pkill -f "[s]treamlit run app/"
```

---

## Install as background services

For something that starts on login and survives logout.

```bash
./scripts/install-services.sh
```

This renders the templates in `systemd/` into `~/.config/systemd/user/`,
enabling three units: `rag-chroma`, `rag-query`, `rag-admin`. It detects your
user, home directory, uid and project path at runtime, so the repository can
live anywhere and be used by any account.

The script prints the URLs, the exact commands for inspecting and controlling
each unit, and how to remove them.

### Managing them

```bash
systemctl --user status  rag-chroma.service
systemctl --user status  rag-query.service
systemctl --user status  rag-admin.service

systemctl --user restart rag-query.service          # after editing .env
systemctl --user stop    rag-admin.service
systemctl --user --now disable rag-chroma.service

# remove everything (your documents are kept)
./scripts/uninstall-services.sh

# remove everything including the index
./scripts/uninstall-services.sh --purge
```

### Logs

Units log to the journal, not to files. A non-root user cannot write to
`/var/log`, so the units deliberately write nowhere and leave logging to the
journal.

```bash
journalctl --user -u rag-query.service -f
journalctl --user -u rag-admin.service --since "1 hour ago" -p warning
```

### Surviving logout

The install script checks whether `linger` is enabled and tells you how to fix
it if not:

```bash
sudo loginctl enable-linger $USER
```

Without linger, the services stop when you log out.

---

## Configuration

Everything lives in `.env`, created from `.env.example`. **Real environment
variables win over `.env`**, so you can override one value without editing the
file:

```bash
RAG_TOP_K=8 .venv/bin/streamlit run app/query_app.py
```

`.env` is gitignored and `setup.sh` sets its permissions to `600`.

### Models

**Every model this project uses is completely free on OpenRouter** — zero cost
per input and per output token. No paid model is referenced anywhere in the
repository, including in `.env.example`. Using this project costs nothing
beyond your own compute.

#### In use by default

| Role | Model | Context | Vector size | Cost |
|---|---|---|---|---|
| Chat / answers | `nvidia/nemotron-3-ultra-550b-a55b:free` | 1,000,000 tokens | — | free |
| Embeddings | `nvidia/nemotron-3-embed-1b:free` | 32,768 tokens per input | 2048 dims | free |

The chat model is a reasoning model, which is why `RAG_MAX_TOKENS` needs to
leave room for the reasoning trace. Set `RAG_SHOW_REASONING=true` to watch it
think, or leave it off for a cleaner answer.

#### Other free models you can switch to

| Role | Model | Notes |
|---|---|---|
| Chat | `nvidia/nemotron-3.5-lightning:free` | Free alternative. |
| Embeddings | `liquid/lfm-2.5-embedding-350m:free` | 1024 dims, but a **hard 512-token input limit** — keep `RAG_CHUNK_CHARS` at 1500 or below or chunks get truncated. |
| Embeddings (multimodal) | `nvidia/llama-nemotron-embed-vl-1b-v2:free` | Accepts text and images. |

Change a model in `.env`, restart the apps, and — for embeddings — wipe and
re-index, because vector width is fixed per model. See
[changing models](#changing-models).

Free models carry a `:free` suffix and are rate limited per minute and per day.
Check the current catalogue at [openrouter.ai/models](https://openrouter.ai/models),
or confirm your key can reach a model with:

```bash
.venv/bin/python scripts/doctor.py
```

That prints the pricing your key actually gets and exercises both endpoints.

`RAG_VISION_MODEL` is empty by default. The free chat models above are
text-only, so image uploads stay disabled — image conversion would yield only
EXIF metadata. Set it to a model that accepts image input if you need that.

### Retrieval

| Variable | Default | Notes |
|---|---|---|
| `RAG_TOP_K` | `5` | Passages retrieved per question. |
| `RAG_CHUNK_CHARS` | `1800` | Chunk size in **characters** (~450 tokens). |
| `RAG_CHUNK_OVERLAP_CHARS` | `200` | Overlap between adjacent chunks. |
| `RAG_TEMPERATURE` | `0.2` | |
| `RAG_MAX_TOKENS` | `1024` | Must leave room for reasoning models. |
| `RAG_SHOW_REASONING` | `false` | Show the model's reasoning trace. |
| `RAG_SHOW_SOURCES` | `true` | Show retrieved passages. |
| `RAG_SYSTEM_PROMPT` | *(built in)* | Override to change answer style. |

> **Chunk size is clamped per model.** `liquid/lfm-2.5-embedding-350m:free`
> accepts only **512 tokens per input** (~2048 characters). Larger chunks would
> be silently truncated, wrecking retrieval, so the app clamps them and warns.
> If you choose that model, keep `RAG_CHUNK_CHARS` at or below ~1500.

### Services

| Variable | Default |
|---|---|
| `CHROMA_HOST` / `CHROMA_PORT` | `127.0.0.1` / `8888` |
| `CHROMA_COLLECTION` | `rag_docs` |
| `QUERY_APP_PORT` / `ADMIN_APP_PORT` | `8901` / `8902` |
| `APP_BIND_ADDRESS` | `127.0.0.1` |
| `APP_MAX_UPLOAD_MB` | `200` |

### Paths

Relative paths resolve against the repository root, so the project is fully
relocatable.

| Variable | Default | Resolves to |
|---|---|---|
| `RAG_DATA_DIR` | `data` | `<repo>/data` |
| `RAG_DOCS_DIR` | `data/documents` | `<repo>/data/documents` |
| `CHROMA_DIR` | `data/chroma` | `<repo>/data/chroma` |

### Limits

| Variable | Default | Purpose |
|---|---|---|
| `MAX_FILE_MB` | `200` | Per-file upload ceiling. |
| `MAX_FILES_PER_BATCH` | `50` | Files per upload batch. |
| `MAX_ZIP_MEMBERS` | `500` | Entries per archive. |
| `MAX_ZIP_TOTAL_MB` | `500` | Uncompressed archive size. |
| `MAX_ZIP_EXPANSION_RATIO` | `100` | Zip-bomb guard. |
| `EMBED_BATCH_SIZE` | `32` | Texts per embedding request. |
| `EMBED_MAX_RETRIES` | `5` | Retries with exponential backoff. |
| `API_TIMEOUT_SECONDS` | `120` | Per-request timeout. |

---

## Supported file formats

The admin app probes what is installed and only advertises formats that will
complete the whole pipeline. The **Supported formats** tab shows the live
list with the reason anything is unavailable.

| Extensions | Format | Requires |
|---|---|---|
| `.md` `.markdown` `.txt` `.text` `.json` `.jsonl` | Markdown, text, JSON | — |
| `.csv` | CSV | — |
| `.htm` `.html` | HTML | — |
| `.rss` `.atom` `.xml` | RSS / Atom / XML | — |
| `.epub` | EPUB | — |
| `.ipynb` | Jupyter notebook | — |
| `.zip` | Archive | — |
| `.pdf` | PDF | `markitdown[pdf]` |
| `.docx` | Word | `markitdown[docx]` |
| `.pptx` | PowerPoint | `markitdown[pptx]` |
| `.xlsx` | Excel | `markitdown[xlsx]` |
| `.xls` | Excel (legacy) | `markitdown[xls]` |
| `.msg` | Outlook message | `markitdown[outlook]` |
| `.jpg` `.jpeg` `.png` | Image | `RAG_VISION_MODEL` set to a vision model |
| `.mp3` `.wav` `.m4a` `.mp4` | Audio / video | `markitdown[audio-transcription]` (not installed by default) |

### Two limitations worth knowing

**Scanned PDFs will not work.** markitdown's PDF converter reads the *text
layer* only — there is no OCR. An image-only scan produces no text. The app
detects this and reports the document as empty rather than silently indexing
nothing. To add OCR, install the `markitdown-ocr` plugin and point it at an
OpenRouter vision model.

**Images yield EXIF metadata only** unless `RAG_VISION_MODEL` names a
vision-capable model. Free text-only models cannot describe an image, so
image uploads are disabled by default.

### Archives

A `.zip` is expanded by this project rather than by markitdown, because
markitdown's converter concatenates everything into one unattributed blob.
Each member becomes its own document, so sources attribute correctly and
members can be deleted individually. Guards applied:

- members with absolute paths or `..` are rejected
- symlinks are rejected
- more than `MAX_ZIP_MEMBERS` entries is rejected
- more than `MAX_ZIP_TOTAL_MB` uncompressed is rejected
- an expansion ratio above `MAX_ZIP_EXPANSION_RATIO` is rejected as a zip bomb

---

## How indexing works

```
upload → sanitise → save to data/documents → convert to Markdown
       → chunk → embed via OpenRouter → upsert into Chroma
```

1. **Sanitise.** The filename is reduced to a single safe path component:
   separators, `..`, null bytes and control characters removed. The resolved
   path is then verified to be inside `data/documents`.
2. **Convert.** markitdown turns the file into Markdown, preserving headings,
   lists, tables and fenced code.
3. **Chunk.** A Markdown-aware splitter: sections are split at headings,
   paragraphs at blank lines, prose at sentence boundaries. Code fences are
   never cut in half. Each chunk is prefixed with its source and title, so the
   embedding and the LLM both know where it came from.
4. **Embed.** Batched through OpenRouter, with exponential backoff on rate
   limits.
5. **Upsert.** Deterministic ids (`<file-hash>:<index>`) mean re-indexing the
   same content overwrites rather than duplicates.

### Change detection

Before embedding anything, the app compares each file's SHA-256 with the hash
already stored in the collection. Unchanged files are skipped, so re-running a
scan costs no API calls. Files whose content changed have their old chunks
deleted before the new ones are written.

### Managing the index

The admin app lists every indexed document with its chunk count, and offers:

- **Index the uploaded files** — only the batch just uploaded
- **Re-scan the documents folder** — everything supported, skipping unchanged
- **Force re-index everything** — ignores change detection
- **Delete a document from the index** — leaves the source file in place
- **Delete the entire collection** — start over

---

## Changing models

**Changing `RAG_EMBED_MODEL` invalidates the whole index.** Vectors from two
different embedding models are not comparable, and vector width is fixed per
model.

The app detects a mismatch and says so rather than failing at query time. To
switch embedding models:

1. Set the new model in `.env`
2. Restart both apps
3. In the admin app, press **Delete the entire collection and start over**
4. Press **Re-scan the documents folder**

Changing `RAG_LLM_MODEL` needs no re-index — it only affects how answers are
written.

Changing `RAG_CHUNK_CHARS` or `RAG_CHUNK_OVERLAP_CHARS` also warrants a
re-index, since chunk boundaries change.

---

## Cost and rate limits

**This project costs nothing to run.** Every model it uses is a free OpenRouter
endpoint at zero cost per input and output token, so indexing your documents
and asking unlimited questions incurs no charges. OpenRouter reports the actual
cost of each request and the apps surface token counts, which you can use to
watch request volume rather than spend.

The trade-off of the free tier is **rate limiting**, not cost. Free endpoints
are limited per minute *and* per day, so the practical constraint is how fast
you can index:

| Symptom | Mitigation |
|---|---|
| `429` during indexing | `EMBED_MAX_RETRIES=5` with exponential backoff and jitter |
| Slow indexing | Smaller `EMBED_BATCH_SIZE`; content-hash caching means re-runs make no API calls at all |
| Empty or cut-off answers | Raise `RAG_MAX_TOKENS`; free reasoning models can spend the whole budget on reasoning |
| Upstream overload errors | Retried automatically while nothing has been streamed yet |

Once indexing is done, each question makes one embedding call plus one
completion. Lower `RAG_TOP_K` to shorten prompts and reduce request volume.

If you hit the daily limit, wait for it to reset, or index in smaller batches
over a longer period. Nothing in this repository points at a paid model.

---

## Security

**This is a local, single-user tool.** Please read this before exposing it.

- **The admin app has no authentication.** Anyone who can reach its port can
  upload documents, re-index, and delete your entire collection.
- **Both apps bind to `127.0.0.1` by default**, so they are reachable only from
  your own machine.
- If you set `APP_BIND_ADDRESS` to anything else, both interfaces become
  reachable from the network. The admin app shows a red warning on startup when
  you do.

If you need network access, put the apps behind a reverse proxy that
terminates TLS and requires authentication. Do not expose the admin app
directly.

Upload handling is hardened regardless of exposure:

- filenames cannot escape `data/documents` (traversal, absolute paths and null
  bytes are all stripped, and the resolved path is re-verified)
- markitdown is only ever called via `convert_stream()` with a stream we opened
  ourselves — never `convert()` or `convert_uri()`, which would let a crafted
  input reach a remote or `file://` resource
- archive members are filtered for traversal and symlinks, and expansion is
  bounded
- secrets live only in `.env`, which is gitignored and `chmod 600`

---

## Troubleshooting

**`OPENROUTER_API_KEY is not set`** — create `.env` and add your key, or run
`./scripts/setup.sh`.

**`401` / `AuthenticationError`** — the key is wrong or still the placeholder
from `.env.example`. Keys look like `sk-or-v1-...`.

**`402`** — OpenRouter is asking for credits. Every model this project uses is
free, so this normally means the free tier is unavailable for your account;
confirm with `scripts/doctor.py`.

**`429` rate limit** — expected: free endpoints are rate limited per minute and
per day. Wait a moment and try again; the retry logic will usually ride it out.
Index in smaller batches if it keeps recurring.

**`Cannot reach the vector store`** — Chroma is not running. Start it with
`./scripts/run-local.sh`, or check `systemctl --user status rag-chroma.service`.

**Chroma returns `404` on `http://127.0.0.1:8888/`** — that is normal. It is an
API server, not a website. Use `/api/v2/heartbeat`.

**`Address already in use`** — something else holds the port. Change
`CHROMA_PORT` / `QUERY_APP_PORT` / `ADMIN_APP_PORT` in `.env`, then re-run
`./scripts/install-services.sh` so the units pick up the new value.

**Documents index but answers ignore them** — check the relevance scores in
the Sources panel. Consistently low scores across the board suggest a mismatch
between the documents' language and the embedding model.

**"little or no text extracted"** — expected for scanned PDFs and image-only
files. See [supported formats](#supported-file-formats).

**A document indexed twice** — it should not be possible with deterministic
ids. If you see it, run **Delete the entire collection and start over** and
report it.

**Everything is stale after editing `.env`** — restart the apps; the config is
read once at startup:

```bash
systemctl --user restart rag-query.service rag-admin.service
```

**A systemd unit will not start** — read the journal:
`journalctl --user -u rag-chroma.service -n 50`. Note that user units must not
contain `User=` (systemd rejects it there), and must not redirect logs into
`/var/log` (unwritable without root).

**Start over completely** — `rm -rf .venv data .env` and run
`./scripts/setup.sh` again.

---

## Project layout

```
.
├── app/
│   ├── config.py       # .env loading, path resolution, model capabilities
│   ├── formats.py      # which file formats this install can actually do
│   ├── sanitize.py     # filename and path safety
│   ├── documents.py    # markitdown conversion, safe archive expansion
│   ├── chunking.py     # Markdown-aware splitter
│   ├── openrouter.py   # chat streaming, embeddings, retries
│   ├── vectorstore.py  # ChromaDB access
│   ├── indexing.py     # file -> vectors pipeline
│   ├── rag.py          # retrieve, build prompt, stream answer
│   ├── ui.py           # shared Streamlit helpers
│   ├── query_app.py    # the query interface
│   └── admin_app.py    # the admin interface
├── scripts/
│   ├── setup.sh              # venv, deps, .env, verification
│   ├── run-local.sh          # foreground, no systemd
│   ├── install-services.sh   # render and enable systemd user units
│   ├── uninstall-services.sh
│   └── doctor.py             # preflight checks
├── systemd/           # templates, rendered by install-services.sh
├── tests/
├── data/              # your documents and index (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

`systemd/*.service.template` files are **templates, not units**. They contain
placeholders and systemd will reject them if you install one directly. Use
`./scripts/install-services.sh`.

---

## Development

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest              # 170 tests, no network required
.venv/bin/python -m pytest -v
.venv/bin/python scripts/doctor.py
```

The test suite never touches the network or a live vector store: OpenRouter and
Chroma are replaced with in-memory doubles (`tests/conftest.py`).

Run `./scripts/run-local.sh` while editing. Both apps reload on save.

---

## License

MIT — see [LICENSE](LICENSE).

Uses [OpenRouter](https://openrouter.ai) (API access requires your own key),
[ChromaDB](https://www.trychroma.com), [Streamlit](https://streamlit.io) and
[markitdown](https://github.com/microsoft/markitdown) (MIT, Microsoft).