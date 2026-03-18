#!/bin/bash
set -euo pipefail

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"

PRESET="${1:-weather_720}"
DEVICE="${2:-cuda}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"

CMD=(
  python /root/0/T3Time/scripts/T3Time_FreTS_FusionExp/generate_chap5_trend_figures.py
  --preset "${PRESET}"
  --device "${DEVICE}"
)

if [ -n "${EPOCHS_OVERRIDE}" ]; then
  CMD+=(--epochs-override "${EPOCHS_OVERRIDE}")
fi

echo "Running preset=${PRESET} device=${DEVICE}"
"${CMD[@]}"
