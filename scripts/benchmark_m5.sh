#!/usr/bin/env bash
set -euo pipefail

QWABLE_BASE_URL="${QWABLE_BASE_URL:-http://127.0.0.1:8088/v1}"
MODEL="${MODEL:-qwable-fast}"

echo "Benchmarking ${MODEL} via ${QWABLE_BASE_URL}/responses"
time curl -fsS "${QWABLE_BASE_URL}/responses" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"input\": \"用一句話回答：Qwable 是什麼？\",
    \"stream\": false
  }" >/tmp/qwable-benchmark-response.json

python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("/tmp/qwable-benchmark-response.json").read_text())
output = data.get("output", [])
text = output[0].get("text", "") if output else ""
print("output_chars=", len(text))
print("usage=", data.get("usage"))
PY
