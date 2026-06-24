#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL_VISION_PRO:-qwen/qwen3-vl-30b}"

python3 "${SCRIPT_DIR}/vision_smoke.py" "${MODEL}"
