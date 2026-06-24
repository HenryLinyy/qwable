#!/usr/bin/env bash
set -euo pipefail

DS4_DIR="${DS4_DIR:-$HOME/Documents/ds4}"
DS4_KV_DIR="${DS4_KV_DIR:-$HOME/Documents/ds4-kv}"
DS4_KV_DISK_SPACE_MB="${DS4_KV_DISK_SPACE_MB:-32768}"

mkdir -p "${DS4_KV_DIR}"
cd "${DS4_DIR}"

exec ./ds4-server \
  --ctx ${DS4_CTX:-100000} \
  --kv-disk-dir "${DS4_KV_DIR}" \
  --kv-disk-space-mb "${DS4_KV_DISK_SPACE_MB}"
