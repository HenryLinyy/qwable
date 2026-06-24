#!/usr/bin/env bash
# G11: On-demand warmup helper.
# Usage:
#   bash scripts/warmup_now.sh quality   # preload quality preset panel models
#   bash scripts/warmup_now.sh budget    # preload budget preset panel models
#   bash scripts/warmup_now.sh           # defaults to quality
#
# Equivalent to:
#   launchctl start io.github.henrylinyy.qwable.warmup.quality
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRESET="${1:-quality}"
case "$PRESET" in
    quality)
        bash "${SCRIPT_DIR}/warmup_fusion_quality.sh" --quiet
        # Also trigger the launchd job (covers TTL=1h reload):
        launchctl start io.github.henrylinyy.qwable.warmup.quality 2>/dev/null || true
        ;;
    budget)
        bash "${SCRIPT_DIR}/warmup_fusion_budget.sh" --quiet
        launchctl start io.github.henrylinyy.qwable.warmup.budget 2>/dev/null || true
        ;;
    *)
        echo "Usage: $0 [quality|budget]" >&2
        exit 1
        ;;
esac
