#!/bin/bash
# 一键生成除 ECL 外所有数据集的 GPT-2/original embeddings
# 包括: ETTh1 ETTh2 ETTm1 ETTm2 ILI Weather exchange_rate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASETS=(
  "ETTh1"
  "ETTh2"
  "ETTm1"
  "ETTm2"
  "ILI"
  "Weather"
  "exchange_rate"
)

echo "=========================================="
echo "一键生成 GPT-2/original embeddings"
echo "数据集: ${DATASETS[*]}"
echo "不包含: ECL"
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
  echo
  echo ">>> 开始处理 ${dataset}"
  if [[ "${dataset}" == "ILI" ]]; then
    bash "${PROJECT_ROOT}/scripts/generate_gpt2_embeddings_8gpu.sh" "${dataset}" 36 24
  else
    bash "${PROJECT_ROOT}/scripts/generate_gpt2_embeddings_8gpu.sh" "${dataset}"
  fi
done

echo
echo "=========================================="
echo "✅ 所有非 ECL 数据集的 GPT-2/original embeddings 已生成完成"
echo "=========================================="
