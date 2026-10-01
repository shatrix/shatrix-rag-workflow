#!/usr/bin/env bash
#
# Run everything in the foreground, with no systemd involved.
#
# Useful for development and for checking that a change works before
# installing the background services. Ctrl-C stops all of it.
#
#   ./scripts/run-local.sh                 # chroma + query + admin
#   ./scripts/run-local.sh --only query    # one service
#   ./scripts/run-local.sh --skip-chroma   # chroma is already running
#
# If you prefer separate terminal windows, see the "Run without systemd"
# section of README.md -- this script is just the same three commands with
# colour-coded logs and a single teardown.

set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
PROJECT_ROOT="$(cd -P "$(dirname "$SOURCE")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_PY="$PROJECT_ROOT/.venv/bin/python"
STREAMLIT="$PROJECT_ROOT/.venv/bin/streamlit"
CHROMA="$PROJECT_ROOT/.venv/bin/chroma"

ONLY=""
SKIP_CHROMA=0
PIDS=()

usage() {
    cat <<'USAGE'
Usage: ./scripts/run-local.sh [options]

Options:
  --only <name>     Run just one of: chroma, query, admin
  --skip-chroma     Do not start Chroma (assumes it is already listening)
  -h, --help        Show this help

Without options, starts all three: Chroma, the query UI and the admin UI.

USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="${2:-}"; shift ;;
        --only=*) ONLY="${1#*=}" ;;
        --skip-chroma) SKIP_CHROMA=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; RESET=$'\033[0m'
    C_CHROMA=$'\033[35m'   # magenta
    C_QUERY=$'\033[36m'    # cyan
    C_ADMIN=$'\033[32m'    # green
else
    BOLD=""; DIM=""; RED=""; RESET=""
    C_CHROMA=""; C_QUERY=""; C_ADMIN=""
fi

die() { printf '\n%sERROR:%s %s\n\n' "$RED" "$RESET" "$1" >&2; exit 1; }

# ── preflight ──────────────────────────────────────────────────────────────
[ -x "$VENV_PY" ] || die "No virtual environment found.
  Run ./scripts/setup.sh first."

[ -f "$PROJECT_ROOT/.env" ] || die "No .env found.
  Run ./scripts/setup.sh, or copy .env.example to .env and add your
  OPENROUTER_API_KEY."

if ! "$VENV_PY" -c 'import chromadb, streamlit, openai' 2>/dev/null; then
    die "Dependencies are missing. Run ./scripts/setup.sh."
fi

# Read the configuration without importing streamlit.
eval "$("$VENV_PY" - <<'PY'
import shlex, sys
from app.config import get_settings
s = get_settings()
print(f"CHROMA_DIR={shlex.quote(str(s.chroma_dir))}")
print(f"CHROMA_HOST={shlex.quote(s.chroma_host)}")
print(f"CHROMA_PORT={shlex.quote(str(s.chroma_port))}")
print(f"QUERY_PORT={shlex.quote(str(s.query_port))}")
print(f"ADMIN_PORT={shlex.quote(str(s.admin_port))}")
print(f"BIND_ADDR={shlex.quote(s.bind_address)}")
print(f"MAX_UPLOAD_MB={shlex.quote(str(s.max_upload_mb))}")
PY
)"

mkdir -p "$CHROMA_DIR"

port_busy() {
    # Bash's /dev/tcp needs no external process, so the readiness loop below
    # stays cheap.
    if (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null; then
        exec 3<&- 2>/dev/null || true
        exec 3>&- 2>/dev/null || true
        echo busy
    else
        echo free
    fi
}

# ── teardown ───────────────────────────────────────────────────────────────
cleanup() {
    printf '\n%sShutting down...%s\n' "$BOLD" "$RESET"
    for pid in "${PIDS[@]:-}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            # Give it a moment, then insist.
            for _ in 1 2 3 4 5 6 7 8 9 10; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.2
            done
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    printf '%sDone.%s\n\n' "$DIM" "$RESET"
}
trap cleanup EXIT INT TERM

prefixed() {  # prefixed <colour> <label> <command...>
    local colour="$1" label="$2"; shift 2
    "$@" 2>&1 | sed -u -e "s/^/${colour}[${label}]${RESET} /"
}

# ── chroma ─────────────────────────────────────────────────────────────────
CHROMA_READY=0

start_chroma() {
    if [ "$SKIP_CHROMA" -eq 1 ] || [ "$ONLY" = "query" ] || [ "$ONLY" = "admin" ]; then
        if [ "$(port_busy "$CHROMA_HOST" "$CHROMA_PORT")" = "busy" ]; then
            printf '%sChroma already listening on %s:%s%s\n' \
                "$DIM" "$CHROMA_HOST" "$CHROMA_PORT" "$RESET"
            CHROMA_READY=1
            return 0
        fi
        if [ "$SKIP_CHROMA" -eq 1 ]; then
            die "--skip-chroma was given but nothing is listening on $CHROMA_HOST:$CHROMA_PORT."
        fi
    fi

    if [ "$(port_busy "$CHROMA_HOST" "$CHROMA_PORT")" = "busy" ]; then
        printf '%sPort %s is already in use; assuming Chroma is already running.%s\n' \
            "$YELLOW" "$CHROMA_PORT" "$RESET"
        CHROMA_READY=1
        return 0
    fi

    printf '%sStarting Chroma on %s:%s%s\n' "$C_CHROMA" "$CHROMA_HOST" "$CHROMA_PORT" "$RESET"
    printf '%s  data: %s%s\n' "$DIM" "$CHROMA_DIR" "$RESET"
    prefixed "$C_CHROMA" chroma "$CHROMA" run \
        --path "$CHROMA_DIR" --host "$CHROMA_HOST" --port "$CHROMA_PORT" &
    PIDS+=($!)

    printf '%sWaiting for Chroma to accept connections' "$DIM"
    for _ in $(seq 1 60); do
        if [ "$(port_busy "$CHROMA_HOST" "$CHROMA_PORT")" = "busy" ]; then
            printf ' %sready%s\n' "$BOLD" "$RESET"
            CHROMA_READY=1
            return 0
        fi
        printf '.'
        sleep 0.5
    done
    printf ' %stimed out%s\n' "$RED" "$RESET"
    die "Chroma did not become ready within 30 seconds."
}

# ── streamlit apps ─────────────────────────────────────────────────────────
start_streamlit() {  # start_streamlit <label> <colour> <port> <script>
    local label="$1" colour="$2" port="$3" script="$4"

    if [ "$(port_busy 127.0.0.1 "$port")" = "busy" ]; then
        printf '%sPort %s already in use; skipping %s.%s\n' \
            "$YELLOW" "$port" "$label" "$RESET"
        return 0
    fi

    printf '%sStarting %s UI on port %s%s\n' "$colour" "$label" "$port" "$RESET"
    prefixed "$colour" "$label" "$STREAMLIT" run "$PROJECT_ROOT/$script" \
        --server.port "$port" \
        --server.address "$BIND_ADDR" \
        --server.headless true \
        --server.maxUploadSize "$MAX_UPLOAD_MB" \
        --browser.gatherUsageStats false &
    PIDS+=($!)
}

# ── main ───────────────────────────────────────────────────────────────────
printf '%srag-workflow (foreground)%s %s%s%s\n' \
    "$BOLD" "$RESET" "$DIM" "$PROJECT_ROOT" "$RESET"
printf '%s  bind %s · chroma %s · query %s · admin %s%s\n' \
    "$DIM" "$BIND_ADDR" "$CHROMA_PORT" "$QUERY_PORT" "$ADMIN_PORT" "$RESET"

case "$ONLY" in
    ""|chroma) start_chroma ;;
    query|admin) : ;;
    *) die "--only must be one of: chroma, query, admin" ;;
esac

if [ "$ONLY" = "chroma" ]; then
    printf '\n%sChroma only. Ctrl-C to stop.%s\n' "$DIM" "$RESET"
    wait "${PIDS[@]}" 2>/dev/null || true
    exit 0
fi

# The apps refuse to start usefully without the vector store.
if [ "$CHROMA_READY" -eq 0 ]; then
    if [ "$(port_busy "$CHROMA_HOST" "$CHROMA_PORT")" != "busy" ]; then
        die "Chroma is not running on $CHROMA_HOST:$CHROMA_PORT."
    fi
fi

if [ -z "$ONLY" ] || [ "$ONLY" = "query" ]; then
    start_streamlit "query" "$C_QUERY" "$QUERY_PORT" "app/query_app.py"
fi
if [ -z "$ONLY" ] || [ "$ONLY" = "admin" ]; then
    start_streamlit "admin" "$C_ADMIN" "$ADMIN_PORT" "app/admin_app.py"
fi

HOST_FOR_URL="localhost"
[ "$BIND_ADDR" = "0.0.0.0" ] && HOST_FOR_URL="<this machine's IP>"

cat <<EOF

${BOLD}Ready.${RESET}
  ${BOLD}Admin${RESET}  http://$HOST_FOR_URL:$ADMIN_PORT   (upload and index documents)
  ${BOLD}Query${RESET}  http://$HOST_FOR_URL:$QUERY_PORT   (ask questions)

Start with the admin app, upload some documents and press
"Re-scan the documents folder". Then use the query app.

${DIM}Ctrl-C stops everything. Logs are prefixed per service.
To run these in the background instead:  ./scripts/install-services.sh${RESET}

EOF

wait