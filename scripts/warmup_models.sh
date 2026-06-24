#!/usr/bin/env bash
set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:1234/v1}"
MODELS="${LMSTUDIO_MODELS:-google/gemma-4-26b-a4b-qat qwen/qwen3-coder-next qwen/qwen3.6-35b-a3b qwen/qwen3-vl-30b}"

for model in ${MODELS}; do
  echo "Warming up $model"
  curl -fsS "${OLLAMA_BASE_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$model\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"Say OK only.\"}
      ],
      \"max_tokens\": 16,
      \"stream\": false
    }" > /dev/null
done

echo "Warmup complete."
