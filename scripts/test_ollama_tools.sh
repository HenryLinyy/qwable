#!/usr/bin/env bash
set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:1234/v1}"
MODEL_FAST="${MODEL_FAST:-google/gemma-4-26b-a4b-qat}"

payload="$(cat <<JSON
{
  "model": "${MODEL_FAST}",
  "messages": [{"role": "user", "content": "列出目前目錄"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "run_shell",
      "description": "Run a shell command",
      "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"]
      }
    }
  }],
  "tool_choice": "auto",
  "stream": false
}
JSON
)"

curl -fsS "${OLLAMA_BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${payload}" | python3 -c 'import sys,json; d=json.load(sys.stdin); tc=d["choices"][0]["message"].get("tool_calls"); print("PASS tool_calls:", tc) if tc else sys.exit("FAIL: no tool_calls")'
