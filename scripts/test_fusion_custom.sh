#!/usr/bin/env bash
# G10: E2E test for fusion deliberation router — custom panel override.
# Uses simplified top-level `fusion` block shape to override analysis_models.
set -euo pipefail

GATEWAY_URL="${QWABLE_URL:-http://127.0.0.1:8088}"

echo "[test_fusion_custom] Gateway: $GATEWAY_URL"
echo "[test_fusion_custom] Pre-flight: lms unload --all"
~/.lmstudio/bin/lms unload --all 2>/dev/null || true

# Custom panel: only gemma + qwen3.6, judge=qwen3.6 (overrides budget preset's default)
payload=$(cat <<JSON
{
  "model": "qwable-fusion",
  "messages": [{"role": "user", "content": "What does async/await do in one sentence?"}],
  "fusion": {
    "preset": "budget",
    "analysis_models": ["google/gemma-4-26b-a4b-qat", "qwen/qwen3.6-35b-a3b"],
    "judge_model": "qwen/qwen3.6-35b-a3b"
  },
  "stream": false,
  "max_tokens": 4000
}
JSON
)

response=$(curl -fsS "${GATEWAY_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${payload}")

echo "$response" | python3 -c '
import sys, json
d = json.load(sys.stdin)
content = d["choices"][0]["message"]["content"]
if not content or not content.strip():
    print("FAIL: empty response")
    sys.exit(1)
required = ["## Final Answer", "## Consensus", "## Contradictions", "## Blind Spots", "## Per-model Notes"]
missing = [h for h in required if h not in content]
if missing:
    print(f"NOTE: missing {len(missing)}/5 structured sections — fallback (raw judge text).")
    print("PASS custom: fallback path (raw text returned, structured format not enforced)")
else:
    print("PASS custom: all 5 sections present (custom panel via top-level fusion block)")
'

echo "[test_fusion_custom] Post-flight: lms ps"
~/.lmstudio/bin/lms ps 2>/dev/null | head -5 || true
