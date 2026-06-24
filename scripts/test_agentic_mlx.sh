#!/usr/bin/env bash
set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:1234/v1}"
MODEL_AGENTIC_MLX="${MODEL_AGENTIC_MLX:-qwen/qwen3.6-35b-a3b}"

payload="$(cat <<JSON
{
  "model": "${MODEL_AGENTIC_MLX}",
  "messages": [{"role": "user", "content": "Call run_shell with command pwd."}],
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
  "stream": false,
  "max_tokens": 256
}
JSON
)"

curl -fsS "${OLLAMA_BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${payload}" | python3 -c 'import sys,json; d=json.load(sys.stdin); tc=d["choices"][0]["message"].get("tool_calls"); print("PASS agentic-mlx tool_calls:", tc) if tc else sys.exit("FAIL: no tool_calls")'
