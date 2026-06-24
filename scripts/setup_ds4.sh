#!/usr/bin/env bash
set -euo pipefail

DS4_DIR="${DS4_DIR:-$HOME/Documents/ds4}"
DS4_QUANT="${DS4_QUANT:-q2-imatrix}"

mkdir -p "$(dirname "${DS4_DIR}")"

if [ ! -d "${DS4_DIR}/.git" ]; then
  git clone https://github.com/antirez/ds4 "${DS4_DIR}"
fi

cd "${DS4_DIR}"
./download_model.sh "${DS4_QUANT}"
make
