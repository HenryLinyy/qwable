#!/usr/bin/env bash
# G11: Idempotent installer for the two fusion warmup launchd jobs.
# Symlinks plists from repo to ~/Library/LaunchAgents/ and loads them.
# Run again to update (replaces symlink target — keeps installed plist in sync).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLIST_DIR="${PROJECT_ROOT}/Library/LaunchAgents"
USER_PLIST_DIR="${HOME}/Library/LaunchAgents"

PLISTS=(
    "io.github.henrylinyy.qwable.warmup.quality"
    "io.github.henrylinyy.qwable.warmup.budget"
)

mkdir -p "${USER_PLIST_DIR}"

for label in "${PLISTS[@]}"; do
    src="${PLIST_DIR}/${label}.plist"
    dst="${USER_PLIST_DIR}/${label}.plist"

    echo "[install_warmup_launchd] ${label}"

    # Validate source plist first
    if [ ! -f "${src}" ]; then
        echo "  FAIL: missing source plist at ${src}" >&2
        exit 1
    fi
    plutil -lint "${src}" >/dev/null

    # Unload existing (ignore errors if not loaded)
    launchctl unload "${dst}" 2>/dev/null || true

    # Symlink (so future edits in repo propagate automatically)
    ln -sf "${src}" "${dst}"
    echo "  linked: ${dst}"

    # Load with -w so it persists across reboots
    launchctl load -w "${dst}"
    echo "  loaded"
done

echo ""
echo "[install_warmup_launchd] verify:"
launchctl list | grep qwable-warmup || true

echo ""
echo "[install_warmup_launchd] to trigger immediately (on-demand):"
echo "  launchctl start io.github.henrylinyy.qwable.warmup.quality"
echo "  launchctl start io.github.henrylinyy.qwable.warmup.budget"
echo "  bash ${SCRIPT_DIR}/warmup_now.sh quality"
echo "  bash ${SCRIPT_DIR}/warmup_now.sh budget"
