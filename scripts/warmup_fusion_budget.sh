#!/usr/bin/env bash
# G11: Warmup helper for budget preset — preload gemma + qwen3.6
# (light, <40GB peak). Same pattern as warmup_fusion_quality.sh.
# G15: Add MLX optimization flags + auto-apply settings.
set -euo pipefail

LMSTUDIO="${LMSTUDIO_CLI:-$HOME/.lmstudio/bin/lms}"
QUIET="${QUIET:-0}"

# G12 micro B: opt-out via env (default true for convenience)
if [ "${QWABLE_WARMUP_ON_BOOT:-true}" != "true" ]; then
    [ "$QUIET" != "1" ] && echo "[warmup_fusion_budget] QWABLE_WARMUP_ON_BOOT!=true, skipping"
    exit 0
fi

log() {
    if [ "$QUIET" != "1" ]; then
        echo "[warmup_fusion_budget] $*"
    fi
}

# G15: MLX optimization flags.
CONTEXT_LENGTH="${FUSION_WARMUP_CONTEXT_LENGTH:-32768}"
PARALLEL="${FUSION_WARMUP_PARALLEL:-2}"
GPU_MODE="${FUSION_WARMUP_GPU_MODE:-max}"
TTL_SECONDS="${FUSION_WARMUP_TTL:-3600}"

APPLY_OPTS="${FUSION_APPLY_MLX_OPTIMIZATIONS:-true}"
if [ "$APPLY_OPTS" = "true" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/..')
try:
    from qwable.mlx_optimizations import apply_recommended_settings
    apply_recommended_settings(dry_run=False)
except Exception as e:
    print(f'[warmup_fusion_budget] settings apply failed: {e}', file=sys.stderr)
" 2>/dev/null || true
fi

log "Pre-flight: lms unload --all"
"$LMSTUDIO" unload --all 2>/dev/null || true

# Budget preset panel order:
#  1. google/gemma-4-26b-a4b-qat (16GB, panelist)
#  2. qwen/qwen3.6-35b-a3b (38GB, panelist + judge)
for model in "google/gemma-4-26b-a4b-qat" "qwen/qwen3.6-35b-a3b"; do
    log "load $model (ctx=$CONTEXT_LENGTH, parallel=$PARALLEL, gpu=$GPU_MODE)"
    "$LMSTUDIO" load "$model" --context-length "$CONTEXT_LENGTH" --parallel "$PARALLEL" --gpu "$GPU_MODE" --ttl "$TTL_SECONDS" >/dev/null 2>&1 || true
    log "probe $model (tiny chat to force compile)"
    curl -fsS "${OLLAMA_BASE_URL:-http://127.0.0.1:1234/v1}/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}" \
        >/dev/null 2>&1 || true
    log "unload $model"
    "$LMSTUDIO" unload --all >/dev/null 2>&1 || true
done

log "done"
