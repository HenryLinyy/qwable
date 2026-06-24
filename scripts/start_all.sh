#!/usr/bin/env bash
set -euo pipefail

# start_all.sh — bring up the full Qwable stack in dependency order:
#   1) LM Studio local model backend  (:1234)   — non-ds4 profiles 都靠它
#   2) ds4 heavy brain                (:8000)   — heavy-agent primary
#   3) Qwable gateway           (:8088)   — 統一入口
#
# 特性:
#   - 冪等:已在執行的服務會略過,不會重複啟動
#   - 依序健康檢查:前一個沒就緒就不啟動下一個 (fail fast)
#   - ds4 / gateway 以背景執行,log 寫到 $LOG_DIR
#
# 常用覆寫 (或直接寫進 .env):
#   LMSTUDIO_CLI_PATH  LM Studio CLI 路徑 (預設 ~/.lmstudio/bin/lms)
#   LMSTUDIO_PORT      LM Studio server port (預設 1234)
#   DS4_BASE_URL       ds4 OpenAI-compatible base (預設 http://127.0.0.1:8000/v1)
#   QWABLE_HOST / QWABLE_PORT   gateway bind (預設 127.0.0.1:8088)
#   LOG_DIR            背景 log 目錄 (預設 <repo>/.run)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# 載入 .env (若存在),讓本腳本沿用使用者的 LMSTUDIO_CLI_PATH / port / DS4_* 設定
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "${PROJECT_ROOT}/.env"
  set +a
fi

LMSTUDIO_CLI_PATH="${LMSTUDIO_CLI_PATH:-$HOME/.lmstudio/bin/lms}"
LMSTUDIO_PORT="${LMSTUDIO_PORT:-1234}"
DS4_BASE_URL="${DS4_BASE_URL:-http://127.0.0.1:8000/v1}"
QWABLE_HOST="${QWABLE_HOST:-127.0.0.1}"
QWABLE_PORT="${QWABLE_PORT:-8088}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/.run}"

mkdir -p "${LOG_DIR}"

# 從 DS4_BASE_URL 推導 ds4 port (取 host 後的 :PORT)
DS4_PORT="$(printf '%s' "${DS4_BASE_URL}" | sed -E 's#^[a-zA-Z]+://[^:/]+:([0-9]+).*$#\1#')"
[[ "${DS4_PORT}" =~ ^[0-9]+$ ]] || DS4_PORT=8000

port_up() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

wait_for_port() { # <port> <label> <timeout_s>
  local port="$1" label="$2" timeout="${3:-30}" i=0
  while (( i < timeout )); do
    if port_up "${port}"; then echo "  ✓ ${label} listening (:${port})"; return 0; fi
    sleep 1; i=$((i + 1))
  done
  echo "  ✗ ${label} 在 ${timeout}s 內未綁定 :${port}" >&2
  return 1
}

wait_for_http() { # <url> <label> <timeout_s>
  local url="$1" label="$2" timeout="${3:-30}" i=0
  while (( i < timeout )); do
    if curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; then echo "  ✓ ${label} health OK"; return 0; fi
    sleep 1; i=$((i + 1))
  done
  echo "  ✗ ${label} health 未通過: ${url}" >&2
  return 1
}

# ds4 等待綁定;若 log 出現單例鎖訊息則提早回傳 2 (不空等)
# 回傳: 0 已綁定 / 1 逾時 / 2 單例鎖未釋放
wait_ds4_bind() { # <timeout_s>
  local timeout="$1" i=0
  while (( i < timeout )); do
    port_up "${DS4_PORT}" && return 0
    grep -q "already running" "${LOG_DIR}/ds4.log" 2>/dev/null && return 2
    sleep 1; i=$((i + 1))
  done
  return 1
}

# ds4 啟動含重試:停掉舊 ds4 後鎖可能還在 drain,短暫等待再試
start_ds4_with_retry() {
  local attempts="${DS4_START_ATTEMPTS:-3}" timeout="${DS4_START_TIMEOUT:-60}" \
        lock_wait="${DS4_LOCK_WAIT:-5}" n=1 rc
  while (( n <= attempts )); do
    echo "  • start_ds4.sh (attempt ${n}/${attempts}, log: ${LOG_DIR}/ds4.log) ..."
    : > "${LOG_DIR}/ds4.log"
    nohup bash "${SCRIPT_DIR}/start_ds4.sh" >"${LOG_DIR}/ds4.log" 2>&1 &
    rc=0; wait_ds4_bind "${timeout}" || rc=$?
    case "${rc}" in
      0) echo "  ✓ ds4 listening (:${DS4_PORT})"; return 0 ;;
      2) echo "    ! ds4 單例鎖未釋放,等 ${lock_wait}s 後重試"; sleep "${lock_wait}" ;;
      *) echo "  ✗ ds4 在 ${timeout}s 內未綁定,請看 ${LOG_DIR}/ds4.log" >&2; return 1 ;;
    esac
    n=$((n + 1))
  done
  echo "  ✗ ds4 在 ${attempts} 次嘗試後仍未啟動 (單例鎖卡住?)" >&2
  return 1
}

echo "=== Qwable stack startup ==="

# ---- 1/3 LM Studio ----------------------------------------------------------
echo "[1/3] LM Studio backend (:${LMSTUDIO_PORT})"
if port_up "${LMSTUDIO_PORT}"; then
  echo "  • 已在執行,略過"
else
  if [[ ! -x "${LMSTUDIO_CLI_PATH}" ]]; then
    echo "  ✗ 找不到可執行的 lms CLI: ${LMSTUDIO_CLI_PATH}" >&2
    echo "    請改開 LM Studio App 啟用 Local Server,或設定 LMSTUDIO_CLI_PATH" >&2
    exit 1
  fi
  echo "  • lms server start ..."
  "${LMSTUDIO_CLI_PATH}" server start
  wait_for_port "${LMSTUDIO_PORT}" "LM Studio" 30
fi

# ---- 2/3 ds4 ----------------------------------------------------------------
echo "[2/3] ds4 heavy brain (:${DS4_PORT})"
if port_up "${DS4_PORT}"; then
  echo "  • 已在執行,略過"
else
  start_ds4_with_retry
fi
wait_for_http "${DS4_BASE_URL}/models" "ds4" 30

# ---- 3/3 gateway ------------------------------------------------------------
echo "[3/3] Qwable gateway (:${QWABLE_PORT})"
if port_up "${QWABLE_PORT}"; then
  echo "  • 已在執行,略過"
else
  echo "  • start_server.sh (log: ${LOG_DIR}/gateway.log) ..."
  nohup bash "${SCRIPT_DIR}/start_server.sh" >"${LOG_DIR}/gateway.log" 2>&1 &
  wait_for_port "${QWABLE_PORT}" "gateway" 60
fi
wait_for_http "http://${QWABLE_HOST}:${QWABLE_PORT}/health" "gateway" 30

# ---- summary ----------------------------------------------------------------
echo
echo "=== 全部就緒 ✅ ==="
printf '  %-10s :%-5s %s\n' "lmstudio" "${LMSTUDIO_PORT}"     "$(port_up "${LMSTUDIO_PORT}" && echo UP || echo DOWN)"
printf '  %-10s :%-5s %s\n' "ds4"      "${DS4_PORT}"          "$(port_up "${DS4_PORT}" && echo UP || echo DOWN)"
printf '  %-10s :%-5s %s\n' "gateway"  "${QWABLE_PORT}" "$(port_up "${QWABLE_PORT}" && echo UP || echo DOWN)"
echo
echo "log: ds4 -> ${LOG_DIR}/ds4.log"
echo "     gateway -> launchd 託管時看 ~/Library/Logs/qwable-gateway.{out,err}.log"
echo "                (手動啟動時看 ${LOG_DIR}/gateway.log)"
echo "停止:bash scripts/stop_all.sh            # 只停 ds4"
echo "     bash scripts/stop_all.sh --gateway   # 連 launchd gateway 一起停"
