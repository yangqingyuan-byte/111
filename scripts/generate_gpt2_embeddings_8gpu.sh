#!/bin/bash
# 用 8 张 GPU 并行为原始 T3Time 生成 GPT-2 embeddings
# 用法:
#   bash scripts/generate_gpt2_embeddings_8gpu.sh Weather
#   bash scripts/generate_gpt2_embeddings_8gpu.sh ILI 36 24

set -euo pipefail

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH-}"
cd "${PROJECT_ROOT}"

DATA_PATH="${1:-Weather}"
INPUT_LEN="${2:-}"
OUTPUT_LEN="${3:-}"
EMBED_VERSION="${EMBED_VERSION:-original}"
NUM_GPUS=8

case "${DATA_PATH}" in
  ILI)
    DEFAULT_INPUT_LEN=36
    DEFAULT_OUTPUT_LEN=24
    ;;
  ETTh1|ETTh2)
    DEFAULT_INPUT_LEN=96
    DEFAULT_OUTPUT_LEN=96
    ;;
  ETTm1|ETTm2)
    DEFAULT_INPUT_LEN=96
    DEFAULT_OUTPUT_LEN=96
    ;;
  Weather|exchange_rate|ECL)
    DEFAULT_INPUT_LEN=96
    DEFAULT_OUTPUT_LEN=96
    ;;
  *)
    DEFAULT_INPUT_LEN=96
    DEFAULT_OUTPUT_LEN=96
    ;;
esac

INPUT_LEN="${INPUT_LEN:-${DEFAULT_INPUT_LEN}}"
OUTPUT_LEN="${OUTPUT_LEN:-${DEFAULT_OUTPUT_LEN}}"

get_split_len() {
    local split_name=$1
    python - "$DATA_PATH" "$split_name" "$INPUT_LEN" "$OUTPUT_LEN" <<'PY'
import sys
import importlib.util
from pathlib import Path

data_path = sys.argv[1]
split_name = sys.argv[2]
input_len = int(sys.argv[3])
output_len = int(sys.argv[4])
mod_path = Path("storage/store_emb.py")
spec = importlib.util.spec_from_file_location("store_emb_mod", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
dataset = mod.get_dataset(data_path, split_name, input_len, output_len)
print(len(dataset))
PY
}

launch_shard() {
    local gpu_id=$1
    local split_name=$2
    local start_idx=$3
    local end_idx=$4

    echo "GPU${gpu_id}: ${split_name} [${start_idx}, ${end_idx}]"
    CUDA_VISIBLE_DEVICES=${gpu_id} python storage/store_emb.py \
        --data_path "${DATA_PATH}" \
        --divide "${split_name}" \
        --input_len "${INPUT_LEN}" \
        --output_len "${OUTPUT_LEN}" \
        --device cuda \
        --batch_size 1 \
        --num_workers 4 \
        --d_model 768 \
        --l_layers 12 \
        --model_name "gpt2" \
        --embed_version "${EMBED_VERSION}" \
        --start_idx "${start_idx}" \
        --end_idx "${end_idx}" \
        --skip_existing &
}

TRAIN_LEN=$(get_split_len train)
VAL_LEN=$(get_split_len val)
TEST_LEN=$(get_split_len test)

echo "=========================================="
echo "GPT-2 embeddings 八卡并行生成"
echo "Dataset: ${DATA_PATH}"
echo "Input Len: ${INPUT_LEN}"
echo "Output Len: ${OUTPUT_LEN}"
echo "Embed Version: ${EMBED_VERSION}"
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
    if [ "${gpu}" -lt "${REM}" ]; then
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
echo "✅ GPT-2 embeddings 生成完成"
echo "保存目录: ./Embeddings/${DATA_PATH}/${EMBED_VERSION}/{train,val,test}"
echo "=========================================="
