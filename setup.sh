#!/usr/bin/env bash
# Qwable — one-shot setup for a fresh machine (macOS Apple Silicon).
# Creates a venv, installs deps, seeds .env, and checks the external backends.
# It does NOT download models — see SETUP.md for the model list.

set -euo pipefail
cd "$(dirname "$0")"

echo "==> Qwable setup"

# 1. Python
command -v python3 >/dev/null || { echo "ERROR: python3 not found. Install Python 3.11+."; exit 1; }
PYV="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "  python3: $PYV"
case "$PYV" in 3.1[1-9]|3.[2-9]*) : ;; *) echo "  WARNING: Python 3.11+ recommended (found $PYV)";; esac

# 2. venv + deps
if [ ! -d .venv ]; then python3 -m venv .venv; echo "  created .venv"; fi
./.venv/bin/python -m pip install --upgrade pip >/dev/null
./.venv/bin/python -m pip install -r requirements.txt
echo "  dependencies installed"

# 3. .env
if [ ! -f .env ]; then cp .env.example .env; echo "  created .env from .env.example"; else echo "  .env already exists (kept)"; fi

# 4. external backend checks (warnings only — none block setup)
LMS="${LMSTUDIO_CLI_PATH:-$HOME/.lmstudio/bin/lms}"
[ -x "$LMS" ] && echo "  LM Studio CLI: $LMS ✓" || echo "  ⚠ LM Studio CLI not at $LMS — install https://lmstudio.ai then run 'lms bootstrap'"
if curl -s --max-time 2 http://127.0.0.1:1234/v1/models >/dev/null 2>&1; then
  echo "  LM Studio server :1234 ✓"
else
  echo "  ⚠ LM Studio server not on :1234 — run 'lms server start'"
fi
if curl -s --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "  ds4 :8000 ✓"
else
  echo "  ℹ ds4 (:8000) not running — only needed for the 'heavy' preset / long-context (optional)"
fi

cat <<'NEXT'

==> Setup done. Next steps:
  1. Download the models in SETUP.md via LM Studio (lms get ... or the LM Studio UI).
  2. Start the gateway:
       ./.venv/bin/python -m uvicorn qwable.server:app --host 127.0.0.1 --port 8088
  3. Verify everything:
       bash scripts/verify_all.sh
  See SETUP.md for API usage and troubleshooting.
NEXT
