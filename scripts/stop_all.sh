#!/usr/bin/env bash
set -euo pipefail

# stop_all.sh — 停止 Qwable stack。
#
# 重要:gateway 與 LM Studio 在這台機器上是 launchd 託管(開機自動起、
# 被 kill 會自動重啟),所以:
#   - ds4 (:8000)        未被 launchd 管 -> 預設依 port kill (會真的停)
#   - gateway (:8088)    launchd KeepAlive -> kill 無效,需 launchctl 才停得了
#   - LM Studio (:1234)  launchd app      -> 預設保留
#
# 用法:
#   bash scripts/stop_all.sh              # 只停 ds4 (gateway/LM Studio 保留)
#   bash scripts/stop_all.sh --gateway    # 連 gateway 一起停 (launchctl bootout)
#   bash scripts/stop_all.sh --lmstudio   # 連 LM Studio server 一起停
#   bash scripts/stop_all.sh --all        # 三個全停

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "${PROJECT_ROOT}/.env"
  set +a
fi

LMSTUDIO_CLI_PATH="${LMSTUDIO_CLI_PATH:-$HOME/.lmstudio/bin/lms}"
LMSTUDIO_PORT="${LMSTUDIO_PORT:-1234}"
DS4_BASE_URL="${DS4_BASE_URL:-http://127.0.0.1:8000/v1}"
QWABLE_PORT="${QWABLE_PORT:-8088}"
GATEWAY_LAUNCHD_LABEL="${GATEWAY_LAUNCHD_LABEL:-io.github.henrylinyy.qwable.gateway}"

DS4_PORT="$(printf '%s' "${DS4_BASE_URL}" | sed -E 's#^[a-zA-Z]+://[^:/]+:([0-9]+).*$#\1#')"
[[ "${DS4_PORT}" =~ ^[0-9]+$ ]] || DS4_PORT=8000

STOP_GATEWAY=0
STOP_LMSTUDIO=0
for arg in "$@"; do
  case "${arg}" in
    --gateway)  STOP_GATEWAY=1 ;;
    --lmstudio) STOP_LMSTUDIO=1 ;;
    --all)      STOP_GATEWAY=1; STOP_LMSTUDIO=1 ;;
    *) echo "未知參數: ${arg}" >&2; exit 2 ;;
  esac
done

launchd_loaded() { launchctl list 2>/dev/null | grep -q "[[:space:]]${1}$"; }

stop_port() { # <port> <label>
  local port="$1" label="$2" pids
  pids="$(lsof -nP -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    echo "  • ${label} (:${port}) 未在執行"
    return 0
  fi
  echo "  • 停止 ${label} (:${port}) PID: ${pids}"
  # shellcheck disable=SC2086
  kill ${pids} 2>/dev/null || true
  for _ in $(seq 1 10); do
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1 || { echo "    ✓ 已停止"; return 0; }
    sleep 1
  done
  echo "    ! 仍在執行,送 SIGKILL"
  # shellcheck disable=SC2086
  kill -9 ${pids} 2>/dev/null || true
}

echo "=== Qwable stack shutdown ==="

# ---- ds4 (未被 launchd 管,kill 會真的停) ----
stop_port "${DS4_PORT}" "ds4"

# ---- gateway (launchd KeepAlive) ----
if [[ "${STOP_GATEWAY}" -eq 1 ]]; then
  if launchd_loaded "${GATEWAY_LAUNCHD_LABEL}"; then
    echo "  • launchctl bootout ${GATEWAY_LAUNCHD_LABEL}"
    launchctl bootout "gui/$(id -u)/${GATEWAY_LAUNCHD_LABEL}" 2>/dev/null || true
    for _ in $(seq 1 10); do
      lsof -nP -iTCP:"${QWABLE_PORT}" -sTCP:LISTEN >/dev/null 2>&1 || { echo "    ✓ gateway 已停"; break; }
      sleep 1
    done
    echo "    ↩ 要恢復: launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/${GATEWAY_LAUNCHD_LABEL}.plist"
  else
    stop_port "${QWABLE_PORT}" "gateway"
  fi
else
  if launchd_loaded "${GATEWAY_LAUNCHD_LABEL}"; then
    echo "  • gateway (:${QWABLE_PORT}) 由 launchd 託管,kill 會被自動重啟 -> 保留 (要停請加 --gateway)"
  else
    echo "  • gateway (:${QWABLE_PORT}) 保留 (要停請加 --gateway)"
  fi
fi

# ---- LM Studio ----
if [[ "${STOP_LMSTUDIO}" -eq 1 ]]; then
  echo "  • 停止 LM Studio server"
  if [[ -x "${LMSTUDIO_CLI_PATH}" ]]; then
    "${LMSTUDIO_CLI_PATH}" server stop || true
  else
    stop_port "${LMSTUDIO_PORT}" "lmstudio"
  fi
else
  echo "  • LM Studio (:${LMSTUDIO_PORT}) 保留 (要停請加 --lmstudio)"
fi

echo "=== done ==="
