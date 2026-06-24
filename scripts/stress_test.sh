#!/usr/bin/env bash
# Memory-aware sequential stress test for fusion presets + agent workflows.
# Gateway is single-concurrency, so this drives a sustained SEQUENTIAL load and
# checks every response is real (no judge-error / planner-fail / empty output).
# Before each heavy request, if free memory is low it unloads LM Studio models
# so the M5's 128GB isn't blown.
#
#   bash scripts/stress_test.sh

set -uo pipefail
B="${B:-http://127.0.0.1:8088}"
LMS="${LMS:-$HOME/.lmstudio/bin/lms}"
MIN_FREE_PCT="${MIN_FREE_PCT:-12}"
PASS=0; FAIL=0

freepct() { memory_pressure 2>/dev/null | grep -i "free percentage" | grep -oE "[0-9]+" | head -1; }

reclaim_if_low() {
  local f; f="$(freepct)"; f="${f:-100}"
  if [ "$f" -lt "$MIN_FREE_PCT" ]; then
    echo "    [mem ${f}% < ${MIN_FREE_PCT}% -> lms unload --all]"
    "$LMS" unload --all >/dev/null 2>&1 || true
    sleep 3
  fi
}

# run <label> <json-payload> [timeout]
run() {
  local label="$1" payload="$2" to="${3:-720}"
  reclaim_if_low
  local f0 t0 body dt f1
  f0="$(freepct)"; t0=$SECONDS
  body="$(curl -s --max-time "$to" "$B/v1/chat/completions" -H "Content-Type: application/json" -d "$payload")"
  dt=$((SECONDS - t0)); f1="$(freepct)"
  LABEL="$label" DT="$dt" F0="$f0" F1="$f1" python3 -c "
import os,sys,json
lab=os.environ['LABEL']; dt=os.environ['DT']; f0=os.environ['F0']; f1=os.environ['F1']
raw=sys.stdin.read()
try: d=json.loads(raw)
except Exception: print(f'  FAIL {lab} | non-json: {raw[:80]!r}'); sys.exit(1)
if 'error' in d: print(f'  FAIL {lab} | error: {str(d[\"error\"])[:90]}'); sys.exit(1)
c=(d.get('choices',[{}])[0].get('message',{}).get('content') or '')
bad=[x for x in ('judge error','planner_json_parse_failed','produced no output','all fallback candidates failed','malformed') if x in c.lower()]
if c.strip() and not bad:
    print(f'  PASS {lab} | {dt}s | mem {f0}%->{f1}% | {len(c)} chars'); sys.exit(0)
print(f'  FAIL {lab} | {dt}s | bad={bad} | {c[:80]!r}'); sys.exit(1)
" <<<"$body" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
}

fusion() { run "fusion:$1" "{\"model\":\"qwable-fusion\",\"fusion\":{\"preset\":\"$1\"},\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}]}" "${3:-720}"; }
wf()     { run "workflow:$1" "{\"model\":\"qwable-$1\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}]}" 900; }

echo "=== Qwable stress test @ $B (min_free=${MIN_FREE_PCT}%) ==="
echo "baseline mem: $(freepct)% free"

echo ""; echo "── Round 1: fusion presets ──"
fusion budget  "In one sentence, why is the sky blue?"
fusion quality "Give 2 pros and 2 cons of microservices, briefly."
fusion coding  "Write a Python one-liner to flatten a list of lists."
fusion heavy   "In one sentence, what is tail latency and why does it matter?"

echo ""; echo "── Round 2: agent workflows ──"
wf agent       "Write a Python function fib(n) returning the nth Fibonacci number. Keep it short."
wf code-agent  "Implement is_palindrome(s) ignoring case and spaces, with 2 asserts. Concise."
wf review-agent "Review this: def div(a,b): return a/b  — list edge cases and risks."

echo ""; echo "── Round 3: fusion reload stress (repeat) ──"
fusion budget  "Name one tradeoff of caching."
fusion coding  "Write a Python one-liner to reverse a string."

echo ""; echo "=== Summary ==="
printf "  \033[32m%d passed\033[0m, \033[31m%d failed\033[0m | final mem %s%% free\n" "$PASS" "$FAIL" "$(freepct)"
[ "$FAIL" = 0 ] && echo "  ✅ fusion presets + workflows stable under sequential load" || echo "  ⚠️ some requests failed — see above"
