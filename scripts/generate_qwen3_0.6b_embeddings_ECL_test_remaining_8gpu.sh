#!/bin/bash
# 使用 8 张 GPU 并行补齐 ECL test 集剩余未生成的 qwen3_0.6b embeddings

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
INDICES_DIR="${PROJECT_ROOT}/Results/emb_logs/ECL_test_remaining_indices"
MONITOR_SCRIPT="${PROJECT_ROOT}/自动监控gpu和发通知dao微信.py"
MONITOR_CONFIG="${PROJECT_ROOT}/.gpu_monitor_config"
mkdir -p "${INDICES_DIR}"

send_wechat_notify() {
    if [ ! -f "${MONITOR_SCRIPT}" ]; then
        echo "⚠️ 未找到微信监控脚本，跳过通知: ${MONITOR_SCRIPT}"
        return 0
    fi

    if [ ! -f "${MONITOR_CONFIG}" ]; then
        echo "⚠️ 未找到监控配置文件，跳过通知: ${MONITOR_CONFIG}"
        return 0
    fi

    echo "📩 启动微信通知监控..."
    if command -v timeout >/dev/null 2>&1; then
        timeout 300 python "${MONITOR_SCRIPT}" --interval 15 --check-count 2 --threshold 0 --config "${MONITOR_CONFIG}" || true
    else
        nohup python "${MONITOR_SCRIPT}" --interval 15 --check-count 2 --threshold 0 --config "${MONITOR_CONFIG}" >/dev/null 2>&1 &
    fi
}

build_missing_indices() {
    python - <<'PY'
import importlib.util
from pathlib import Path

project_root = Path(".")
mod_path = project_root / "storage" / "store_emb_qwen3_0.6b.py"
spec = importlib.util.spec_from_file_location("store_emb_qwen3_0_6b_mod", mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
dataset = mod.get_dataset("ECL", "test", 96, 96)
total_len = len(dataset)

embed_dir = project_root / "Embeddings" / "ECL" / "qwen3_0.6b" / "test"
embed_dir.mkdir(parents=True, exist_ok=True)

missing = []
for idx in range(total_len):
    if not (embed_dir / f"{idx}.h5").exists():
        missing.append(idx)

out_dir = project_root / "Results" / "emb_logs" / "ECL_test_remaining_indices"
out_dir.mkdir(parents=True, exist_ok=True)

num_gpus = 8
base = len(missing) // num_gpus
rem = len(missing) % num_gpus
cursor = 0
for gpu in range(num_gpus):
    extra = 1 if gpu < rem else 0
    part = missing[cursor:cursor + base + extra]
    cursor += len(part)
    with open(out_dir / f"gpu{gpu}.txt", "w", encoding="utf-8") as f:
        for idx in part:
            f.write(f"{idx}\n")

print(total_len)
print(len(missing))
PY
}

launch_gpu() {
    local gpu_id=$1
    local indices_file="${INDICES_DIR}/gpu${gpu_id}.txt"
    local count
    count=$(wc -l < "${indices_file}")
    if [ "${count}" -eq 0 ]; then
        echo "GPU${gpu_id}: 没有剩余索引，跳过"
        return 0
    fi

    echo "GPU${gpu_id}: 剩余索引数 ${count}"
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
        --indices_file "${indices_file}" \
        --skip_existing &
    PIDS+=($!)
}

mapfile -t STATS < <(build_missing_indices)
TOTAL_LEN="${STATS[0]}"
MISSING_COUNT="${STATS[1]}"

echo "=========================================="
echo "ECL test 剩余 embeddings 八卡并行补齐"
echo "test 总样本数: ${TOTAL_LEN}"
echo "剩余缺失数: ${MISSING_COUNT}"
echo "=========================================="

if [ "${MISSING_COUNT}" -eq 0 ]; then
    echo "✅ 没有缺失项，无需继续生成"
    send_wechat_notify
    exit 0
fi

PIDS=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    launch_gpu "${gpu}"
done

echo "等待所有剩余 test embedding 分片完成..."
for pid in "${PIDS[@]}"; do
    wait "${pid}"
done

echo "=========================================="
echo "✅ ECL test 剩余 embeddings 已补齐"
echo "保存目录: ./Embeddings/${DATA_PATH}/${EMBED_VERSION}/${DIVIDE}"
echo "=========================================="
send_wechat_notify
