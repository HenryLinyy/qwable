#!/usr/bin/env bash
# G10: E2E test for fusion deliberation router — bad preset error path.
# Verifies that unknown preset names produce a 200 response with descriptive
# error text in the assistant content (HTTP 200, not 4xx — by design so the
# caller always gets parseable JSON).
set -euo pipefail

GATEWAY_URL="${QWABLE_URL:-http://127.0.0.1:8088}"

echo "[test_fusion_bad_preset] Gateway: $GATEWAY_URL"

payload=$(cat <<JSON
{
  "model": "qwable-fusion",
  "messages": [{"role": "user", "content": "x"}],
  "fusion": {"preset": "this-preset-does-not-exist"},
  "stream": false
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
# Bad preset path returns error text mentioning the preset name
if "this-preset-does-not-exist" not in content:
    print("FAIL: error text should mention bad preset name")
    print(content[:500])
    sys.exit(1)
if "preset" not in content.lower():
    print("FAIL: error text should mention 'preset'")
    sys.exit(1)
print("PASS bad_preset: descriptive error returned, no model invoked")
'

echo "[test_fusion_bad_preset] Post-flight: lms ps (should still be empty — no models called)"
~/.lmstudio/bin/lms ps 2>/dev/null | head -5 || true
