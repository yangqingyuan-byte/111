#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="${1:-Weather}"
PRED_LEN="${2:-720}"
METRIC="${3:-mae}"
DEVICE="${4:-cuda}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"

case "${DATASET,,}" in
  etth1) DATASET_NORM="ETTh1"; OURS_KEYWORD="T3Time_FreTS_Gated_Qwen_Hyperopt_ETTh1" ;;
  etth2) DATASET_NORM="ETTh2"; OURS_KEYWORD="T3Time_FreTS_Gated_Qwen_Hyperopt_ETTh2" ;;
  ettm1) DATASET_NORM="ETTm1"; OURS_KEYWORD="T3Time_FreTS_Gated_Qwen_Hyperopt_ETTm1" ;;
  ettm2) DATASET_NORM="ETTm2"; OURS_KEYWORD="T3Time_FreTS_Gated_Qwen_Hyperopt_ETTm2" ;;
  ili) DATASET_NORM="ILI"; OURS_KEYWORD="T3Time_FreTS_Gated_Qwen_Hyperopt_ILI" ;;
  weather) DATASET_NORM="Weather"; OURS_KEYWORD="T3Time_FreTS_Gated_Qwen_Hyperopt_Weather" ;;
  exchange|exchange_rate) DATASET_NORM="exchange_rate"; OURS_KEYWORD="T3Time_FreTS_Gated_Qwen_Hyperopt_Exchange" ;;
  *)
    echo "不支持的数据集: ${DATASET}"
    exit 1
    ;;
esac

CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/scripts/T3Time_FreTS_FusionExp/generate_chap5_main_compare_plot.py"
  --dataset "${DATASET_NORM}"
  --pred-len "${PRED_LEN}"
  --baseline-keyword "T3Time"
  --ours-keyword "${OURS_KEYWORD}"
  --metric "${METRIC}"
  --device "${DEVICE}"
)

if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
  CMD+=(--epochs-override "${EPOCHS_OVERRIDE}")
fi

echo "=========================================="
echo "第五章主对比图生成"
echo "Dataset: ${DATASET_NORM}"
echo "Pred Len: ${PRED_LEN}"
echo "Metric: ${METRIC}"
echo "Device: ${DEVICE}"
echo "Ours Keyword: ${OURS_KEYWORD}"
echo "=========================================="

"${CMD[@]}"

