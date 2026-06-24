#!/usr/bin/env bash
set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:1234/v1}"
MODEL_FORMATTER_MLX="${MODEL_FORMATTER_MLX:-google/gemma-4-26b-a4b-qat}"

payload="$(cat <<JSON
{
  "model": "${MODEL_FORMATTER_MLX}",
  "messages": [{"role": "user", "content": "Reply with exactly OK."}],
  "stream": false,
  "max_tokens": 128,
  "temperature": 0
}
JSON
)"

curl -fsS "${OLLAMA_BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${payload}" | python3 -c 'import sys,json; d=json.load(sys.stdin); text=d["choices"][0]["message"].get("content", "").strip(); print("PASS formatter-mlx:", text) if text else sys.exit("FAIL: empty formatter response")'
