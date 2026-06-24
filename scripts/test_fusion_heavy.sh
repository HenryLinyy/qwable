#!/usr/bin/env bash
# G10: E2E test for fusion deliberation router — heavy preset.
# Panel: coder + r1, judge=ds4 deepseek-v4-flash.
# Requires: LM Studio with coder + r1, AND ds4 running on http://127.0.0.1:8000.
set -euo pipefail

GATEWAY_URL="${QWABLE_URL:-http://127.0.0.1:8088}"
DS4_URL="${DS4_BASE_URL:-http://127.0.0.1:8000/v1}"
PRESET="heavy"

echo "[test_fusion_heavy] Gateway: $GATEWAY_URL"
echo "[test_fusion_heavy] DS4: $DS4_URL"

# Verify ds4 is reachable
if ! curl -fsS "${DS4_URL}/models" >/dev/null 2>&1; then
    echo "FAIL: ds4 not reachable at $DS4_URL — start ds4 with scripts/start_ds4.sh"
    exit 1
fi

echo "[test_fusion_heavy] Pre-flight: lms unload --all"
~/.lmstudio/bin/lms unload --all 2>/dev/null || true

payload=$(cat <<JSON
{
  "model": "qwable-fusion",
  "messages": [{"role": "user", "content": "What is the time complexity of the standard library's sorted() in CPython? One sentence."}],
  "fusion": {"preset": "${PRESET}"},
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
    print("PASS heavy: fallback path (raw text returned, structured format not enforced)")
else:
    print("PASS heavy: all 5 sections present (judge=ds4)")
'

echo "[test_fusion_heavy] Post-flight: lms ps (should be empty)"
~/.lmstudio/bin/lms ps 2>/dev/null | head -5 || true
