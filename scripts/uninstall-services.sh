#!/usr/bin/env bash
#
# Remove the rag-workflow systemd user services.
#
#   ./scripts/uninstall-services.sh              # remove units, keep data
#   ./scripts/uninstall-services.sh --purge      # also delete the Chroma index
#   ./scripts/uninstall-services.sh --keep-units # only stop them
#
# Your uploaded documents in data/documents are never deleted.

set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
PROJECT_ROOT="$(cd -P "$(dirname "$SOURCE")/.." && pwd)"
VENV_PY="$PROJECT_ROOT/.venv/bin/python"

USER_ID="$(id -u)"
USER_HOME="${HOME:-$(getent passwd "$USER_ID" | cut -d: -f6)}"
UNIT_DIR="$USER_HOME/.config/systemd/user"

UNITS=(rag-query.service rag-admin.service rag-chroma.service)
PURGE=0
KEEP_UNITS=0

usage() {
    cat <<'USAGE'
Usage: ./scripts/uninstall-services.sh [options]

Options:
  --purge        Also delete the Chroma index (data/chroma). Uploaded
                 documents in data/documents are still kept.
  --keep-units   Stop and disable the services but leave the unit files.
  -h, --help     Show this help.

USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --purge)      PURGE=1 ;;
        --keep-units) KEEP_UNITS=1 ;;
        -h|--help)    usage; exit 0 ;;
        *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'
    GREEN=$'\033[32m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; BLUE=""; RESET=""
fi

step() { printf '\n%s==>%s %s%s%s\n' "$BLUE" "$RESET" "$BOLD" "$1" "$RESET"; }
info() { printf '    %s\n' "$1"; }
ok()   { printf '    %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
warn() { printf '    %s!%s %s\n' "$DIM" "$RESET" "$1"; }

step "Stopping services"

for unit in "${UNITS[@]}"; do
    if systemctl --user is-active --quiet "$unit" 2>/dev/null; then
        systemctl --user stop "$unit" 2>/dev/null || true
        info "stopped $unit"
    fi
    if [ "$KEEP_UNITS" -eq 0 ]; then
        systemctl --user disable "$unit" >/dev/null 2>&1 || true
    fi
done
systemctl --user daemon-reload 2>/dev/null || true
ok "services stopped"

if [ "$KEEP_UNITS" -eq 1 ]; then
    warn "--keep-units given; unit files left in $UNIT_DIR"
else
    step "Removing unit files"
    for unit in "${UNITS[@]}"; do
        target="$UNIT_DIR/$unit"
        if [ -f "$target" ]; then
            rm -f "$target"
            info "removed $target"
        fi
    done
    systemctl --user daemon-reload 2>/dev/null || true
    systemctl --user reset-failed 2>/dev/null || true
    ok "units removed from $UNIT_DIR"
fi

if [ "$PURGE" -eq 1 ]; then
    step "Purging the vector index"
    CHROMA_DIR="$PROJECT_ROOT/data/chroma"
    if [ -d "$CHROMA_DIR" ]; then
        size="$(du -sh "$CHROMA_DIR" 2>/dev/null | cut -f1)"
        rm -rf "${CHROMA_DIR:?}"
        ok "deleted $CHROMA_DIR (was $size)"
    else
        info "no index directory at $CHROMA_DIR"
    fi
    mkdir -p "$CHROMA_DIR"
fi

step "Done"
info "Your uploaded documents are still in $PROJECT_ROOT/data/documents"
info "Reinstall with:  $PROJECT_ROOT/scripts/install-services.sh"
printf '\n'