#!/bin/bash
# 使用 8 张 GPU 并行为 ECL 生成 qwen3_0.6b embeddings

set -euo pipefail

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH-}"
cd "${PROJECT_ROOT}"

DATA_PATH="ECL"
EMBED_VERSION="qwen3_0.6b"
NUM_GPUS=8

get_split_len() {
    local split_name=$1
    python - "$split_name" <<'PY'
import sys
import importlib.util
from pathlib import Path

split_name = sys.argv[1]
mod_path = Path("storage/store_emb_qwen3_0.6b.py")
spec = importlib.util.spec_from_file_location("store_emb_qwen3_0_6b_mod", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
dataset = mod.get_dataset("ECL", split_name, 96, 96)
print(len(dataset))
PY
}

launch_shard() {
    local gpu_id=$1
    local split_name=$2
    local start_idx=$3
    local end_idx=$4

    echo "GPU${gpu_id}: ${split_name} [${start_idx}, ${end_idx}]"
    CUDA_VISIBLE_DEVICES=${gpu_id} python storage/store_emb_qwen3_0.6b.py \
        --data_path "${DATA_PATH}" \
        --divide "${split_name}" \
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
        --end_idx "${end_idx}" &
}

TRAIN_LEN=$(get_split_len train)
VAL_LEN=$(get_split_len val)
TEST_LEN=$(get_split_len test)

echo "=========================================="
echo "ECL 八卡并行生成 qwen3_0.6b embeddings"
echo "train: ${TRAIN_LEN}"
echo "val:   ${VAL_LEN}"
echo "test:  ${TEST_LEN}"
echo "=========================================="

PIDS=()

# 6 张卡切 train，1 张卡跑 val，1 张卡跑 test
TRAIN_GPUS=6
BASE=$((TRAIN_LEN / TRAIN_GPUS))
REM=$((TRAIN_LEN % TRAIN_GPUS))
START=0

for gpu in $(seq 0 $((TRAIN_GPUS - 1))); do
    EXTRA=0
    if [ ${gpu} -lt ${REM} ]; then
        EXTRA=1
    fi
    END=$((START + BASE + EXTRA - 1))
    launch_shard "${gpu}" train "${START}" "${END}"
    PIDS+=($!)
    START=$((END + 1))
done

launch_shard 6 val 0 $((VAL_LEN - 1))
PIDS+=($!)

launch_shard 7 test 0 $((TEST_LEN - 1))
PIDS+=($!)

echo "等待所有 embedding 生成任务完成..."
for pid in "${PIDS[@]}"; do
    wait "${pid}"
done

echo "=========================================="
echo "✅ ECL embeddings 生成完成"
echo "保存目录: ./Embeddings/${DATA_PATH}/${EMBED_VERSION}/{train,val,test}"
echo "=========================================="
