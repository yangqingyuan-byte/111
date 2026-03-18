#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEVICE="${1:-cuda}"
METRIC="${2:-mae}"

declare -A HORIZONS
HORIZONS[ETTh1]="96 192 336 720"
HORIZONS[ETTh2]="96 192 336 720"
HORIZONS[ETTm1]="96 192 336 720"
HORIZONS[ETTm2]="96 192 336 720"
HORIZONS[ILI]="24 36 48 60"
HORIZONS[exchange]="96 192 336 720"
HORIZONS[Weather]="96 192 336 720"

for dataset in ETTh1 ETTh2 ETTm1 ETTm2 ILI exchange Weather; do
  for pred_len in ${HORIZONS[$dataset]}; do
    echo
    echo ">>> ${dataset} / ${pred_len}"
    bash "${SCRIPT_DIR}/run_chap5_main_compare_plot.sh" "${dataset}" "${pred_len}" "${METRIC}" "${DEVICE}"
  done
done
