#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL_VISION_FAST:-google/gemma-4-26b-a4b-qat}"

python3 "${SCRIPT_DIR}/vision_smoke.py" "${MODEL}"
