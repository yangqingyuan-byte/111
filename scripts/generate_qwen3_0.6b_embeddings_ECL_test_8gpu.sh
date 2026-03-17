#!/bin/bash
# 使用 8 张 GPU 并行为 ECL 的 test 集生成 qwen3_0.6b embeddings
# 支持续跑：已存在的 h5 会自动跳过

set -euo pipefail

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH-}"
cd "${PROJECT_ROOT}"

DATA_PATH="ECL"
DIVIDE="test"
EMBED_VERSION="qwen3_0.6b"
NUM_GPUS=8

get_split_len() {
    python - <<'PY'
import importlib.util
from pathlib import Path

mod_path = Path("storage/store_emb_qwen3_0.6b.py")
spec = importlib.util.spec_from_file_location("store_emb_qwen3_0_6b_mod", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
dataset = mod.get_dataset("ECL", "test", 96, 96)
print(len(dataset))
PY
}

launch_shard() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3

    echo "GPU${gpu_id}: ${DIVIDE} [${start_idx}, ${end_idx}]"
    CUDA_VISIBLE_DEVICES=${gpu_id} python storage/store_emb_qwen3_0.6b.py \
        --data_path "${DATA_PATH}" \
        --divide "${DIVIDE}" \
        --input_len 96 \
        --output_len 96 \
        --device cuda \
        --batch_size 1 \
        --num_workers 4 \
        --d_model 1024 \
        --l_layers 28 \
        --model_name "Qwen/Qwen3-0.6B" \
        --embed_version "${EMBED_VERSION}" \
        --start_idx "${start_idx}" \
        --end_idx "${end_idx}" \
        --skip_existing &
}

TEST_LEN=$(get_split_len)

echo "=========================================="
echo "ECL test 八卡并行生成 qwen3_0.6b embeddings"
echo "test: ${TEST_LEN}"
echo "skip_existing: true"
echo "=========================================="

PIDS=()
BASE=$((TEST_LEN / NUM_GPUS))
REM=$((TEST_LEN % NUM_GPUS))
START=0

for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    EXTRA=0
    if [ ${gpu} -lt ${REM} ]; then
        EXTRA=1
    fi
    END=$((START + BASE + EXTRA - 1))
    launch_shard "${gpu}" "${START}" "${END}"
    PIDS+=($!)
    START=$((END + 1))
done

echo "等待所有 test embedding 分片完成..."
for pid in "${PIDS[@]}"; do
    wait "${pid}"
done

echo "=========================================="
echo "✅ ECL test embeddings 生成完成"
echo "保存目录: ./Embeddings/${DATA_PATH}/${EMBED_VERSION}/${DIVIDE}"
echo "=========================================="
