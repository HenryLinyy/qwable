#!/usr/bin/env bash
set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:1234/v1}"
MODEL_FAST="${MODEL_FAST:-google/gemma-4-26b-a4b-qat}"
FAST_MAX_TOKENS="${FAST_MAX_TOKENS:-1500}"

curl -fsS "${OLLAMA_BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_FAST}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say OK only.\"}],
    \"max_tokens\": ${FAST_MAX_TOKENS},
    \"stream\": false
  }"
