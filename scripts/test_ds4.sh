#!/usr/bin/env bash
set -euo pipefail

DS4_BASE_URL="${DS4_BASE_URL:-http://127.0.0.1:8000/v1}"
DS4_MODEL="${DS4_MODEL:-deepseek-v4-flash}"

payload="$(cat <<JSON
{
  "model": "${DS4_MODEL}",
  "messages": [{"role": "user", "content": "Say OK only."}],
  "max_tokens": 16,
  "stream": false
}
JSON
)"

curl -fsS "${DS4_BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${payload}"
