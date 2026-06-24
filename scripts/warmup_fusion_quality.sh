#!/usr/bin/env bash
# G10: Warmup helper for quality preset — preload all 3 panel models in
# sequence (load → unload → next) so the first E2E run doesn't pay the
# full model-load latency on every panel call.
# G11: Add --quiet flag for launchd/cron use (suppress progress chatter).
# G12: Add QWABLE_WARMUP_ON_BOOT opt-out.
# G15: Add MLX optimization flags + auto-apply settings.
set -euo pipefail

LMSTUDIO="${LMSTUDIO_CLI:-$HOME/.lmstudio/bin/lms}"
QUIET=0
for arg in "$@"; do
    case "$arg" in
        --quiet|-q) QUIET=1 ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# G12 micro B: opt-out via env (default true for convenience)
if [ "${QWABLE_WARMUP_ON_BOOT:-true}" != "true" ]; then
    [ "$QUIET" != "1" ] && echo "[warmup_fusion_quality] QWABLE_WARMUP_ON_BOOT!=true, skipping"
    exit 0
fi

log() {
    if [ "$QUIET" != "1" ]; then
        echo "[warmup_fusion_quality] $*"
    fi
}

# G15: MLX optimization flags (read from env with M5 Max defaults).
CONTEXT_LENGTH="${FUSION_WARMUP_CONTEXT_LENGTH:-32768}"
PARALLEL="${FUSION_WARMUP_PARALLEL:-2}"
GPU_MODE="${FUSION_WARMUP_GPU_MODE:-max}"
TTL_SECONDS="${FUSION_WARMUP_TTL:-3600}"

# G15: Apply recommended LM Studio settings (context length, spec decoding flag).
APPLY_OPTS="${FUSION_APPLY_MLX_OPTIMIZATIONS:-true}"
if [ "$APPLY_OPTS" = "true" ] && [ -f "$(dirname "$0")/../pyproject.toml" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/..')
try:
    from qwable.mlx_optimizations import apply_recommended_settings
    apply_recommended_settings(dry_run=False)
except Exception as e:
    print(f'[warmup_fusion_quality] settings apply failed: {e}', file=sys.stderr)
" 2>/dev/null || true
fi

log "Pre-flight: lms unload --all"
"$LMSTUDIO" unload --all 2>/dev/null || true

# Quality preset panel order:
#  1. qwen/qwen3-coder-next (65GB)
#  2. qwen/qwen3.6-35b-a3b (38GB, judge)
#  3. deepseek-r1-distill-qwen-32b (66GB)
#
# We preload each, send a tiny chat to force MLX/LM Studio compilation, then
# unload before the next. The judge (qwen3.6) is left unloaded — runner loads
# it again at judge time.
#
# G15: --parallel and --ttl enable continuous batching + auto-unload after idle.

for model in "qwen/qwen3-coder-next" "qwen/qwen3.6-35b-a3b" "deepseek-r1-distill-qwen-32b"; do
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
