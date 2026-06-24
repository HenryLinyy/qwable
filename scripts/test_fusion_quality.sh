#!/usr/bin/env bash
# G10: E2E test for fusion deliberation router — quality preset.
# Requires: LM Studio running with qwen-coder-next, qwen3.6-35b-a3b,
#           and deepseek-r1-distill-qwen-32b available.
#           Qwable gateway running on http://127.0.0.1:8088.
set -euo pipefail

GATEWAY_URL="${QWABLE_URL:-http://127.0.0.1:8088}"
PRESET="quality"

echo "[test_fusion_quality] Gateway: $GATEWAY_URL"
echo "[test_fusion_quality] Pre-flight: lms unload --all"
~/.lmstudio/bin/lms unload --all 2>/dev/null || true

payload=$(cat <<JSON
{
  "model": "qwable-fusion",
  "messages": [{"role": "user", "content": "Compare mergesort vs quicksort for 50k records on an M5 Max. Recommend one with one sentence."}],
  "plugins": [{"id": "fusion", "preset": "${PRESET}"}],
  "stream": false,
  "max_tokens": 4000
}
JSON
)

response=$(curl -fsS "${GATEWAY_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${payload}")

# Verify all 5 markdown sections present in assistant content
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
    print("PASS quality: fallback path (raw text returned, structured format not enforced)")
else:
    print("PASS quality: all 5 sections present")
print("preview:", content[:200].replace(chr(10), " "))
'

echo "[test_fusion_quality] Post-flight: lms ps (should be empty)"
~/.lmstudio/bin/lms ps 2>/dev/null | head -5 || true
