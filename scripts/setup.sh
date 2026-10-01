#!/usr/bin/env bash
#
# One-time setup for rag-workflow.
#
# Safe to run more than once: it will not overwrite an existing .env or
# recreate a working virtualenv unnecessarily.
#
#   ./scripts/setup.sh                  # interactive
#   ./scripts/setup.sh --no-key-prompt  # unattended
#   PYTHON=python3.12 ./scripts/setup.sh

set -euo pipefail

# ── locate the project, following symlinks ──────────────────────────────────
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
PROJECT_ROOT="$(cd -P "$(dirname "$SOURCE")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"
PYTHON_BIN="${PYTHON:-}"
PROMPT_FOR_KEY=1

# ── output helpers ─────────────────────────────────────────────────────────
if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

step()  { printf '\n%s==>%s %s%s%s\n' "$BLUE" "$RESET" "$BOLD" "$1" "$RESET"; }
info()  { printf '    %s\n' "$1"; }
ok()    { printf '    %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
warn()  { printf '    %s!%s %s\n' "$YELLOW" "$RESET" "$1"; }
die()   { printf '\n%sERROR:%s %s\n\n' "$RED" "$RESET" "$1" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage: ./scripts/setup.sh [options]

Options:
  --no-key-prompt   Do not ask for an OpenRouter API key.
  -h, --help        Show this help.

Environment:
  PYTHON=...        Interpreter to build the virtualenv with.
                    Auto-detected when unset.

USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --no-key-prompt) PROMPT_FOR_KEY=0 ;;
        -h|--help)       usage; exit 0 ;;
        *)               die "Unknown option: $1  (try --help)" ;;
    esac
    shift
done

printf '%srag-workflow setup%s %s(%s)%s\n' \
    "$BOLD" "$RESET" "$DIM" "$PROJECT_ROOT" "$RESET"

# ── 1. platform ────────────────────────────────────────────────────────────
step "Checking platform"

if [ "$(uname -s)" != "Linux" ]; then
    die "This project targets Linux only (found $(uname -s))."
fi
ok "Linux $(uname -r)"

for tool in curl; do
    command -v "$tool" >/dev/null 2>&1 || warn "$tool not found; some suggestions will be skipped"
done

# ── 2. find a Python interpreter ───────────────────────────────────────────
step "Finding a Python interpreter"

version_ok() {
    # "3 11" is printed for (major, minor), so major must be read first.
    "$1" -c 'import sys; print(sys.version_info[:2])' 2>/dev/null | {
        read -r major minor <<<"$(sed 's/[(,)]//g')"
        [ -n "$major" ] && [ -n "$minor" ] || return 1
        [ "$major" = "3" ] && [ "$minor" -ge 10 ] && [ "$minor" -le 14 ]
    }
}

if [ -n "$PYTHON_BIN" ]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 \
        || die "PYTHON=$PYTHON_BIN not found on PATH"
    version_ok "$PYTHON_BIN" \
        || die "PYTHON=$PYTHON_BIN is not a supported version (need 3.10-3.14)"
    FOUND_PYTHON="$(command -v "$PYTHON_BIN")"
    ok "Using $FOUND_PYTHON ($("$FOUND_PYTHON" --version 2>&1))"
else
    FOUND_PYTHON=""
    # Prefer 3.12, then 3.11, then 3.13/3.10. Wheels for the newest releases
    # tend to lag, so 3.11 is a dependable fallback.
    for candidate in python3.12 python3.11 python3.13 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && version_ok "$candidate"; then
            FOUND_PYTHON="$(command -v "$candidate")"
            ok "Found $FOUND_PYTHON ($("$FOUND_PYTHON" --version 2>&1))"
            break
        fi
    done
fi

if [ -z "$FOUND_PYTHON" ]; then
    die "No suitable Python found. rag-workflow needs Python 3.10-3.14.
  Debian/Ubuntu : sudo apt install python3.12 python3.12-venv
  Fedora        : sudo dnf install python3.12
  Arch          : sudo pacman -S python
  Or point at one explicitly:  PYTHON=/path/to/python3.12 ./scripts/setup.sh"
fi

# ── 3. virtual environment ─────────────────────────────────────────────────
step "Preparing the virtual environment"

if [ -x "$VENV_DIR/bin/python" ]; then
    ok "Reusing $VENV_DIR"
    info "delete it and re-run this script for a clean rebuild"
else
    info "creating $VENV_DIR with $FOUND_PYTHON"
    "$FOUND_PYTHON" -m venv "$VENV_DIR" \
        || die "Failed to create the virtual environment.
  On Debian/Ubuntu the 'python3-venv' package is often missing:
      sudo apt install python3-venv python3.12-venv"
    ok "Created $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

info "upgrading pip, setuptools and wheel"
"$VENV_PYTHON" -m pip install --quiet --upgrade pip setuptools wheel \
    || warn "could not upgrade pip; continuing with the bundled version"

# ── 4. dependencies ────────────────────────────────────────────────────────
step "Installing Python dependencies"

if ! command -v "$VENV_DIR/bin/chroma" >/dev/null 2>&1; then
    info "installing from requirements.txt (this includes chromadb, which"
    info "pulls in onnxruntime and takes a minute)"
fi

"$VENV_PYTHON" -m pip install -r requirements.txt \
    || die "Dependency installation failed. Common causes:
  - no network access, or a proxy is required (export HTTPS_PROXY=...)
  - pip needs --break-system-packages
  - very old pip:  $VENV_PYTHON -m pip install --upgrade pip"

ok "Dependencies installed"

# ── 5. configuration ───────────────────────────────────────────────────────
step "Configuring the environment"

if [ -f "$ENV_FILE" ]; then
    ok "$ENV_FILE already exists, leaving it untouched"
    CREATED_ENV=0
else
    [ -f "$ENV_EXAMPLE" ] || die "$ENV_EXAMPLE is missing; cannot create .env"
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    ok "Created $ENV_FILE from .env.example"
    CREATED_ENV=1
fi

# Create the data folders regardless, so permissions are correct up front.
mkdir -p "$PROJECT_ROOT/data/documents" "$PROJECT_ROOT/data/chroma"

if [ "$CREATED_ENV" -eq 1 ] && [ "$PROMPT_FOR_KEY" -eq 1 ] && [ -t 0 ]; then
    printf '\n    %sGet an API key from %shttps://openrouter.ai/keys%s\n' \
        "$BOLD" "$BLUE" "$RESET"
    printf '    Paste it now, or press Enter to skip and edit .env yourself.\n'
    printf '    %sOPENROUTER_API_KEY%s: ' "$BOLD" "$RESET"
    read -r API_KEY || API_KEY=""
    if [ -n "$API_KEY" ]; then
        # Only replace a line that is present and empty; never clobber a
        # value the operator set deliberately.
        if grep -q '^OPENROUTER_API_KEY=.*[^=[:space:]]' "$ENV_FILE"; then
            warn "OPENROUTER_API_KEY already has a value; keeping it"
        else
            # Escape sed replacement metacharacters in the key.
            ESCAPED=$(printf '%s' "$API_KEY" | sed -e 's/[\/&|]/\\&/g')
            sed -i "s|^OPENROUTER_API_KEY=.*|OPENROUTER_API_KEY=$ESCAPED|" "$ENV_FILE"
            chmod 600 "$ENV_FILE"
            ok "Saved the key to .env (permissions set to 600)"
        fi
    else
        info "skipped -- add OPENROUTER_API_KEY to .env when you are ready"
    fi
elif [ "$CREATED_ENV" -eq 1 ]; then
    info "add OPENROUTER_API_KEY to $ENV_FILE before using the apps"
fi

# .env holds a secret; keep it out of group/other reach.
[ -f "$ENV_FILE" ] && chmod 600 "$ENV_FILE" 2>/dev/null || true

# ── 6. verify ──────────────────────────────────────────────────────────────
step "Verifying the installation"

if "$VENV_PYTHON" scripts/doctor.py; then
    ok "All checks passed"
else
    warn "Some checks did not pass. See the output above."
    info "re-run any time with:  .venv/bin/python scripts/doctor.py"
fi

# ── 7. next steps ──────────────────────────────────────────────────────────
BIND="$("$VENV_PYTHON" - <<'PY' 2>/dev/null || echo "127.0.0.1"
import sys
sys.path.insert(0, ".")
from app.config import get_settings
s = get_settings()
print(f"{s.bind_address}|{s.query_port}|{s.admin_port}")
PY
)"
BIND_ADDR="${BIND%%|*}"; REST="${BIND#*|}"
QUERY_PORT="${REST%%|*}"; ADMIN_PORT="${REST##*|}"

HOST_FOR_URL="localhost"
[ "$BIND_ADDR" = "0.0.0.0" ] && HOST_FOR_URL="<this machine's IP>"

printf '\n%s%sNext steps%s\n' "$BOLD" "$GREEN" "$RESET"
printf '  %s1.%s Open the admin app and upload some documents:\n' "$BOLD" "$RESET"
printf '       %s./scripts/run-local.sh%s\n' "$DIM" "$RESET"
printf '       %sthen browse to  http://%s:%s%s\n' "$DIM" "$HOST_FOR_URL" "$ADMIN_PORT" "$RESET"
printf '\n  %s2.%s Run it in the background instead (systemd user services):\n' "$BOLD" "$RESET"
printf '       %s./scripts/install-services.sh%s\n' "$DIM" "$RESET"
printf '\n  %s3.%s Read the full guide:\n' "$BOLD" "$RESET"
printf '       %sREADME.md%s\n' "$DIM" "$RESET"
printf '\n  %sQuery UI will be at  http://%s:%s%s\n' "$DIM" "$HOST_FOR_URL" "$QUERY_PORT" "$RESET"
printf '\n'