#!/usr/bin/env bash
set -euo pipefail

# Gate G09: models are managed by LM Studio under ~/.lmstudio/hub/models.
# This helper verifies that the required model ids are present locally; it does
# not call `ollama pull`.
LMS_CLI_PATH="${LMSTUDIO_CLI_PATH:-${LMS_CLI_PATH:-$HOME/.lmstudio/bin/lms}}"
MODELS="${LMSTUDIO_MODELS:-google/gemma-4-26b-a4b-qat qwen/qwen3-coder-next deepseek-r1-distill-qwen-32b qwen/qwen3-vl-30b qwen/qwen3.6-35b-a3b}"

"${LMS_CLI_PATH}" server status >/dev/null
models_json="$(${LMS_CLI_PATH} ls --json)"

for model in ${MODELS}; do
  echo "Checking LM Studio model ${model}"
  python3 -c 'import json,sys; wanted=sys.argv[1]; data=json.loads(sys.stdin.read()); ids={m.get("modelKey") for m in data}; ids.update(m.get("selectedVariant") for m in data if m.get("selectedVariant")); sys.exit(0 if wanted in ids else 1)' \
    "${model}" <<< "${models_json}" || {
      echo "Missing LM Studio model: ${model}" >&2
      echo "Download/select it in LM Studio, then re-run this script." >&2
      exit 1
    }
done

"${LMS_CLI_PATH}" ls
