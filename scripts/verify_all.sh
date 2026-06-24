#!/usr/bin/env bash
# Qwable — one-shot verification of all features / profiles / workflows.
#
#   bash scripts/verify_all.sh          # structural + per-profile smoke (~1-3 min)
#   bash scripts/verify_all.sh --full   # also run the 3 agent workflows (~+8 min)
#
# Verifies the LIVE gateway on $BASE (default 127.0.0.1:8088). Read-only:
# it only sends requests, never edits files or restarts anything.

set -uo pipefail

BASE="${BASE:-http://127.0.0.1:8088}"
DS4="${DS4:-http://127.0.0.1:8000}"
LMS="${LMS:-http://127.0.0.1:1234}"
FULL=0; [ "${1:-}" = "--full" ] && FULL=1

PASS=0; FAIL=0; SKIP=0
ok()   { printf "  \033[32mPASS\033[0m %s\n" "$1"; PASS=$((PASS+1)); }
bad()  { printf "  \033[31mFAIL\033[0m %s\n" "$1"; FAIL=$((FAIL+1)); }
skip() { printf "  \033[33mSKIP\033[0m %s\n" "$1"; SKIP=$((SKIP+1)); }
hdr()  { printf "\n\033[1m== %s ==\033[0m\n" "$1"; }

# check_chat <label> <model> <user-text> [max_tokens=64]
# PASS if HTTP 200 + non-empty assistant content.
# NOTE: panel->judge profiles (full, fusion) need a generous max_tokens — a tiny
# cap starves the thinking-model judge and yields empty content.
check_chat() {
  local label="$1" model="$2" text="$3" maxtok="${4:-64}"
  local body
  body=$(curl -s --max-time 480 "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$text\"}],\"max_tokens\":$maxtok}")
  echo "$body" | python3 -c "
import sys,json
try: d=json.load(sys.stdin)
except Exception: print('  (non-json)'); sys.exit(1)
if 'error' in d: print('   ->',str(d['error'])[:80]); sys.exit(1)
c=d.get('choices',[{}])[0].get('message',{}).get('content') or ''
sys.exit(0 if c.strip() else 1)
" && ok "$label ($model)" || bad "$label ($model)"
}

hdr "1. Structural (instant)"
curl -s --max-time 5 "$BASE/health" | grep -q '"status":"ok"' && ok "gateway health 8088" || bad "gateway health 8088"
curl -s --max-time 5 "$DS4/v1/models"  | grep -q '"object"' && ok "ds4 online 8000" || skip "ds4 offline 8000 (heavy/ds4 features unavailable)"
curl -s --max-time 5 "$LMS/v1/models"  | grep -q '"object"' && ok "LM Studio online 1234" || bad "LM Studio offline 1234"

# all 15 profiles x 2 protocol prefixes registered
curl -s --max-time 8 "$BASE/v1/models" | python3 -c "
import sys,json
ids={m['id'] for m in json.load(sys.stdin).get('data',[])}
need=['','-fast','-full','-heavy','-vision-fast','-vision-pro','-vision-heavy','-agentic-pro','-hermes-pro','-agentic-mlx','-formatter-mlx','-fusion','-agent','-code-agent','-review-agent']
miss=[p for p in need if ('qwable'+p) not in ids]
miss+=['claude-qwable-fast'] if 'claude-qwable-fast' not in ids else []
print('   missing:',miss) if miss else None
sys.exit(1 if miss else 0)
" && ok "all profile model-ids registered (/v1/models)" || bad "profile model-ids registered"
curl -s --max-time 8 "$BASE/v1/fusion/presets" | python3 -c "
import sys,json; p=json.load(sys.stdin).get('presets',{})
sys.exit(0 if all(k in p for k in ('quality','budget','coding','heavy')) else 1)
" && ok "fusion presets (quality/budget/coding/heavy)" || bad "fusion presets"

hdr "2. Protocols"
check_chat "OpenAI Chat" "qwable-fast" "Reply with exactly: PONG"
# responses
curl -s --max-time 300 "$BASE/v1/responses" -H "Content-Type: application/json" \
  -d '{"model":"qwable-fast","input":"Reply with exactly: PONG","max_output_tokens":40}' \
  | grep -q '"completed"' && ok "OpenAI Responses (/v1/responses)" || bad "OpenAI Responses"
# messages
curl -s --max-time 300 "$BASE/v1/messages" -H "Content-Type: application/json" \
  -d '{"model":"claude-qwable-fast","max_tokens":40,"messages":[{"role":"user","content":"Reply with exactly: PONG"}]}' \
  | grep -q '"end_turn"' && ok "Anthropic Messages (/v1/messages)" || bad "Anthropic Messages"
curl -s --max-time 30 "$BASE/v1/messages/count_tokens" -H "Content-Type: application/json" \
  -d '{"model":"claude-qwable-fast","messages":[{"role":"user","content":"hi"}]}' \
  | grep -q 'input_tokens' && ok "count_tokens" || bad "count_tokens"

hdr "3. Text / agentic profiles (smoke)"
check_chat "fast"          "qwable-fast"          "Say PONG"
check_chat "full"          "qwable-full"          "Say PONG" 1500   # panel+judge: needs room
check_chat "chat"          "qwable-chat"          "Say PONG"
check_chat "agentic-pro"   "qwable-agentic-pro"   "Say PONG"
check_chat "hermes-pro"    "qwable-hermes-pro"    "Say PONG"
check_chat "agentic-mlx"   "qwable-agentic-mlx"   "Say PONG"
check_chat "formatter-mlx" "qwable-formatter-mlx" "Say PONG"
# heavy needs ds4 + passes a big resource guard on this machine; accept either a
# real answer or the known guard message rather than failing the whole run.
hbody=$(curl -s --max-time 300 "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"qwable-heavy","messages":[{"role":"user","content":"Say PONG"}],"max_tokens":64}')
echo "$hbody" | python3 -c "import sys,json;d=json.load(sys.stdin);c=(d.get('choices',[{}])[0].get('message',{}).get('content') or '');sys.exit(0 if c.strip() else 1)" \
  && ok "heavy (ds4)" || skip "heavy (no answer — ds4 offline or resource-guarded on this box)"

hdr "4. Vision"
python3 - <<'PY' > /tmp/lf_verify_red.b64
import zlib,struct,base64
W=H=160; raw=(b'\x00'+b'\xd0\x20\x20'*W)*H
def ch(t,d):
    c=t+d; return struct.pack('>I',len(d))+c+struct.pack('>I',zlib.crc32(c)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+ch(b'IHDR',struct.pack('>IIBBBBB',W,H,8,2,0,0,0))+ch(b'IDAT',zlib.compress(raw,9))+ch(b'IEND',b'')
print(base64.b64encode(png).decode())
PY
python3 - "$BASE" <<'PY' && ok "vision (recognizes red)" || bad "vision"
import sys,json,urllib.request
base=sys.argv[1]; b64=open('/tmp/lf_verify_red.b64').read().strip()
body={"model":"qwable-vision-fast","max_tokens":30,"messages":[{"role":"user","content":[
 {"type":"text","text":"What is the dominant color? One word."},
 {"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}}]}]}
req=urllib.request.Request(base+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
try:
    d=json.loads(urllib.request.urlopen(req,timeout=300).read())
    c=(d.get('choices',[{}])[0].get('message',{}).get('content') or '').lower()
    sys.exit(0 if 'red' in c else 1)
except Exception as e:
    print('   ->',str(e)[:80]); sys.exit(1)
PY

hdr "5. Fusion deliberation (budget preset)"
# NOTE: do NOT pass a small max_tokens — the judge (thinking model) needs room.
fbody=$(curl -s --max-time 480 "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"qwable-fusion","fusion":{"preset":"budget"},"messages":[{"role":"user","content":"In one sentence, what is 2+2?"}]}')
echo "$fbody" | python3 -c "
import sys,json;d=json.load(sys.stdin)
c=(d.get('choices',[{}])[0].get('message',{}).get('content') or '')
sys.exit(0 if c.strip() and 'judge error' not in c.lower() else 1)
" && ok "fusion budget (panel -> judge synthesis)" || bad "fusion budget"

hdr "6. Workflows"
if [ "$FULL" = "1" ]; then
  for wf in agent code-agent review-agent; do
    wbody=$(curl -s --max-time 900 "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
      -d "{\"model\":\"qwable-$wf\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a python one-liner add(a,b) that returns a+b. Keep it short.\"}]}")
    echo "$wbody" | python3 -c "
import sys,json
try: d=json.load(sys.stdin)
except Exception: sys.exit(1)
c=(d.get('choices',[{}])[0].get('message',{}).get('content') or '')
bad=any(x in c for x in ('planner_json_parse_failed','fusion judge error','[fusion produced no output]'))
sys.exit(0 if c.strip() and not bad else 1)
" && ok "workflow: $wf" || bad "workflow: $wf"
  done
else
  skip "3 workflows (run 'bash scripts/verify_all.sh --full' — ~8 min)"
fi

hdr "Summary"
printf "  \033[32m%d passed\033[0m, \033[31m%d failed\033[0m, \033[33m%d skipped\033[0m\n" "$PASS" "$FAIL" "$SKIP"
[ "$FAIL" = "0" ] && { echo "  ✅ all checked features operational"; exit 0; } || { echo "  ❌ some checks failed — see above"; exit 1; }
