#!/usr/bin/env bash
#
# Install rag-workflow as systemd *user* services.
#
# User services live in ~/.config/systemd/user, run as the invoking user
# without needing root, and can be controlled with `systemctl --user`.
#
#   ./scripts/install-services.sh              # install and start
#   ./scripts/install-services.sh --no-start    # install only
#   ./scripts/install-services.sh --reinstall  # re-render and restart
#
# Remove them with ./scripts/uninstall-services.sh

set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
PROJECT_ROOT="$(cd -P "$(dirname "$SOURCE")/.." && pwd)"

VENV_DIR="$PROJECT_ROOT/.venv"
TEMPLATE_DIR="$PROJECT_ROOT/systemd"
VENV_PY="$VENV_DIR/bin/python"
ENV_FILE="$PROJECT_ROOT/.env"

# Detected dynamically so this works for any user on any machine.
USER_NAME="${USER:-$(id -un)}"
USER_ID="$(id -u)"
USER_HOME="${HOME:-$(getent passwd "$USER_ID" | cut -d: -f6)}"
UNIT_DIR="$USER_HOME/.config/systemd/user"

UNITS=(rag-chroma.service rag-query.service rag-admin.service)
START=1
REINSTALL=0

usage() {
    cat <<'USAGE'
Usage: ./scripts/install-services.sh [options]

Options:
  --no-start     Install the units but do not enable or start them.
  --reinstall    Re-render the units from the templates and restart.
  -h, --help     Show this help.

USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --no-start)  START=0 ;;
        --reinstall) REINSTALL=1; START=1 ;;
        -h|--help)   usage; exit 0 ;;
        *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

step() { printf '\n%s==>%s %s%s%s\n' "$BLUE" "$RESET" "$BOLD" "$1" "$RESET"; }
info() { printf '    %s\n' "$1"; }
ok()   { printf '    %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
warn() { printf '    %s!%s %s\n' "$YELLOW" "$RESET" "$1"; }
die()  { printf '\n%sERROR:%s %s\n\n' "$RED" "$RESET" "$1" >&2; exit 1; }

# ── preflight ──────────────────────────────────────────────────────────────
step "Checking prerequisites"

[ "$(uname -s)" = "Linux" ] || die "systemd user services require Linux."

command -v systemctl >/dev/null 2>&1 || die "systemctl not found."
[ -d /run/systemd/system ] || die "systemd is not the running init system."

[ -x "$VENV_PY" ] || die "No virtual environment at $VENV_DIR.
  Run ./scripts/setup.sh first."

"$VENV_PY" -c 'import chromadb, streamlit, openai' 2>/dev/null \
    || die "Dependencies are missing. Run ./scripts/setup.sh."

if [ ! -f "$ENV_FILE" ]; then
    warn "No .env found -- the services will start but report a configuration"
    warn "error in the UI until you create one. Run ./scripts/setup.sh."
fi

ok "systemd present, virtualenv complete"

# ── read configuration ─────────────────────────────────────────────────────
step "Reading configuration"

eval "$("$VENV_PY" - <<'PY'
import shlex
from app.config import get_settings
s = get_settings()
for key, value in (
    ("CHROMA_DIR", str(s.chroma_dir)),
    ("CHROMA_HOST", s.chroma_host),
    ("CHROMA_PORT", str(s.chroma_port)),
    ("QUERY_PORT", str(s.query_port)),
    ("ADMIN_PORT", str(s.admin_port)),
    ("BIND_ADDRESS", s.bind_address),
    ("MAX_UPLOAD_MB", str(s.max_upload_mb)),
    ("LLM_MODEL", s.llm_model),
    ("EMBED_MODEL", s.embed_model),
):
    print(f"{key}={shlex.quote(value)}")
PY
)"

info "user          $USER_NAME (uid $USER_ID)"
info "project       $PROJECT_ROOT"
info "virtualenv    $VENV_DIR"
info "units go to   $UNIT_DIR"
info "chroma        $CHROMA_HOST:$CHROMA_PORT  (data: $CHROMA_DIR)"
info "bind address  $BIND_ADDRESS"
info "llm model     $LLM_MODEL"
info "embed model   $EMBED_MODEL"

mkdir -p "$CHROMA_DIR" "$UNIT_DIR"

if [ "$BIND_ADDRESS" != "127.0.0.1" ] && [ "$BIND_ADDRESS" != "localhost" ] && [ "$BIND_ADDRESS" != "::1" ]; then
    warn "APP_BIND_ADDRESS is $BIND_ADDRESS, so both UIs will be reachable"
    warn "from other machines. The admin interface has NO authentication."
fi

# ── render templates ───────────────────────────────────────────────────────
step "Rendering systemd units"

render() {  # render <template> <destination>
    local template="$1" destination="$2"
    [ -f "$template" ] || die "Template not found: $template"

    sed \
        -e "s|{{PROJECT_ROOT}}|$PROJECT_ROOT|g" \
        -e "s|{{VENV_DIR}}|$VENV_DIR|g" \
        -e "s|{{CHROMA_DIR}}|$CHROMA_DIR|g" \
        -e "s|{{CHROMA_HOST}}|$CHROMA_HOST|g" \
        -e "s|{{CHROMA_PORT}}|$CHROMA_PORT|g" \
        -e "s|{{QUERY_PORT}}|$QUERY_PORT|g" \
        -e "s|{{ADMIN_PORT}}|$ADMIN_PORT|g" \
        -e "s|{{BIND_ADDRESS}}|$BIND_ADDRESS|g" \
        -e "s|{{MAX_UPLOAD_MB}}|$MAX_UPLOAD_MB|g" \
        "$template" > "$destination"

    # A template that still held placeholders would start with a literal
    # path, so verify every key we substituted is gone. Only the real keys
    # are checked, so prose in a comment cannot trip this.
    if grep -qE '\{\{(PROJECT_ROOT|VENV_DIR|CHROMA_DIR|CHROMA_HOST|CHROMA_PORT|QUERY_PORT|ADMIN_PORT|BIND_ADDRESS|MAX_UPLOAD_MB)\}\}' "$destination"; then
        die "Rendered $destination still contains unreplaced placeholders."
    fi
}

for unit in "${UNITS[@]}"; do
    template="$TEMPLATE_DIR/$unit.template"
    destination="$UNIT_DIR/$unit"

    # Always re-render. Rendering is deterministic, so re-running this script
    # after editing .env must pick up the new ports, paths and bind address --
    # skipping existing files would silently leave stale configuration in place.
    if [ -f "$destination" ]; then
        info "refreshing $unit with current configuration"
    fi

    render "$template" "$destination"
    chmod 644 "$destination"
    ok "$UNIT_DIR/$unit"
done

# ── enable ─────────────────────────────────────────────────────────────────
step "Reloading systemd"

systemctl --user daemon-reload
ok "daemon reloaded"

if [ "$START" -eq 0 ]; then
    warn "--no-start given; units installed but not started."
else
    info "enabling and starting ${UNITS[*]}"
    for unit in "${UNITS[@]}"; do
        systemctl --user enable "$unit" >/dev/null 2>&1 || true
    done

    # Chroma first: the UIs are useless without it, and starting it separately
    # gives it a moment to open its database before they retry.
    systemctl --user restart rag-chroma.service
    sleep 2
    systemctl --user restart rag-query.service rag-admin.service

    ok "services started"
fi

# ── linger ─────────────────────────────────────────────────────────────────
step "Checking that services survive logout"

if loginctl show-user "$USER_NAME" --property=Linger --value 2>/dev/null | grep -qi '^yes$'; then
    ok "linger enabled; services will keep running after you log out"
else
    warn "linger is not enabled, so the services stop when you log out."
    info "Enable it with:  sudo loginctl enable-linger $USER_NAME"
    info "Once enabled they will survive logout and even reboots."
fi

# ── status ─────────────────────────────────────────────────────────────────
step "Status"

for unit in "${UNITS[@]}"; do
    state="$(systemctl --user is-active "$unit" 2>/dev/null || echo inactive)"
    enabled="$(systemctl --user is-enabled "$unit" 2>/dev/null || echo disabled)"
    case "$state" in
        active)   colour="$GREEN" ;;
        inactive) colour="$YELLOW" ;;
        *)        colour="$RED" ;;
    esac
    printf '    %s%-22s%s %-10s %s(%s)%s\n' \
        "$colour" "$unit" "$RESET" "$state" "$DIM" "$enabled" "$RESET"
done

# ── usage ──────────────────────────────────────────────────────────────────
HOST_FOR_URL="localhost"
[ "$BIND_ADDRESS" = "0.0.0.0" ] && HOST_FOR_URL="<this machine's IP>"

cat <<EOF

${BOLD}${GREEN}Installed.${RESET}

${BOLD}Open the interfaces${RESET}
  Admin (upload and index documents)   http://$HOST_FOR_URL:$ADMIN_PORT
  Query (ask questions)                http://$HOST_FOR_URL:$QUERY_PORT

${BOLD}First run${RESET}
  1. Open the admin app and upload some documents.
  2. Press "Re-scan the documents folder".
  3. Open the query app.

${BOLD}Managing the services${RESET}
  ${DIM}systemctl --user status rag-chroma.service${RESET}
  ${DIM}systemctl --user status rag-query.service${RESET}
  ${DIM}systemctl --user status rag-admin.service${RESET}

  ${DIM}systemctl --user restart rag-query.service${RESET}
  ${DIM}systemctl --user stop rag-admin.service${RESET}
  ${DIM}systemctl --user --now disable rag-chroma.service${RESET}

${BOLD}Reading the logs${RESET}
  ${DIM}journalctl --user -u rag-chroma.service -f${RESET}
  ${DIM}journalctl --user -u rag-query.service -f${RESET}
  ${DIM}journalctl --user -u rag-admin.service -f${RESET}
  ${DIM}journalctl --user -u rag-query.service --since "1 hour ago" -p warning${RESET}

${BOLD}After changing .env${RESET}
  ${DIM}systemctl --user restart rag-query.service rag-admin.service${RESET}

${BOLD}Reinstalling after moving the project${RESET}
  ${DIM}./scripts/uninstall-services.sh && ./scripts/install-services.sh${RESET}

${BOLD}Removing everything${RESET}
  ${DIM}./scripts/uninstall-services.sh${RESET}

${BOLD}Prefer to run in the foreground for development?${RESET}
  ${DIM}./scripts/run-local.sh${RESET}

EOF