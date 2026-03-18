#!/bin/bash
# 为 ILI 数据集生成适配 seq_len=36 的 qwen3 embeddings
# 用法: bash scripts/generate_qwen3_0.6b_embeddings_ILI_36.sh [gpu_id]

set -euo pipefail

GPU_ID=${1:-0}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH-}"
cd "${PROJECT_ROOT}"

DATA_PATH="ILI"
EMBED_VERSION="qwen3_0.6b_ili36"
INPUT_LEN=36
OUTPUT_LEN=24

echo "=========================================="
echo "生成 ILI 专用 Qwen3-0.6B 嵌入文件"
echo "数据集: ${DATA_PATH}"
echo "GPU: ${GPU_ID}"
echo "seq_len: ${INPUT_LEN}"
echo "base output_len: ${OUTPUT_LEN}"
echo "嵌入版本: ${EMBED_VERSION}"
echo "=========================================="

echo ""
echo "1. 生成 TRAIN 嵌入..."
python storage/store_emb_qwen3_0.6b.py \
    --data_path "${DATA_PATH}" \
    --divide train \
    --input_len "${INPUT_LEN}" \
    --output_len "${OUTPUT_LEN}" \
    --device cuda \
    --batch_size 1 \
    --num_workers 4 \
    --d_model 1024 \
    --l_layers 28 \
    --model_name "Qwen/Qwen3-0.6B" \
    --embed_version "${EMBED_VERSION}"

echo ""
echo "2. 生成 VAL 嵌入..."
python storage/store_emb_qwen3_0.6b.py \
    --data_path "${DATA_PATH}" \
    --divide val \
    --input_len "${INPUT_LEN}" \
    --output_len "${OUTPUT_LEN}" \
    --device cuda \
    --batch_size 1 \
    --num_workers 4 \
    --d_model 1024 \
    --l_layers 28 \
    --model_name "Qwen/Qwen3-0.6B" \
    --embed_version "${EMBED_VERSION}"

echo ""
echo "3. 生成 TEST 嵌入..."
python storage/store_emb_qwen3_0.6b.py \
    --data_path "${DATA_PATH}" \
    --divide test \
    --input_len "${INPUT_LEN}" \
    --output_len "${OUTPUT_LEN}" \
    --device cuda \
    --batch_size 1 \
    --num_workers 4 \
    --d_model 1024 \
    --l_layers 28 \
    --model_name "Qwen/Qwen3-0.6B" \
    --embed_version "${EMBED_VERSION}"

echo ""
echo "=========================================="
echo "✅ ILI 专用嵌入生成完成"
echo "保存路径: ./Embeddings/${DATA_PATH}/${EMBED_VERSION}/{train,val,test}/"
echo "=========================================="
