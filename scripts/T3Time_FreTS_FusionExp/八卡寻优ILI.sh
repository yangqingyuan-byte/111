#!/bin/bash
# T3Time_FreTS_Gated_Qwen ILI 超参数+多种子寻优脚本
# 支持 8 卡自动并行、实时进度/ETA 显示、完成后微信通知

set -uo pipefail

unset __vsc_prompt_cmd_original 2>/dev/null || true

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_NOTIFY=${RUN_NOTIFY:-1}

GPU_ID=${1:-0}
START_IDX=${2:-0}
END_IDX=${3:--1}
PARALLEL=${4:-2}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_Hyperopt/ILI"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

MONITOR_SCRIPT="/root/0/T3Time/自动监控gpu和发通知dao微信.py"
MONITOR_CONFIG="/root/0/T3Time/.gpu_monitor_config"

DATA_PATH="ILI"
NUM_NODES=7
SEQ_LEN=36
PRED_LENS=(24 36 48 60)
E_LAYER=1
D_LAYER=1
EPOCHS=120
PATIENCE=10
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b_ili36"
MODEL_ID="T3Time_FreTS_Gated_Qwen_Hyperopt_ILI"
EMBED_ROOT="/root/0/T3Time/Embeddings/${DATA_PATH}/${EMBED_VERSION}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
PROGRESS_DIR="/root/0/T3Time/Results/progress_ILI_${RUN_TAG}"

CHANNELS=(32 64 128)
DROPOUTS=(0.1 0.3 0.5)
HEADS=(8 16)
LEARNING_RATES=(5e-5 1e-4 2.5e-4 1e-3)
WEIGHT_DECAYS=(1e-4 5e-4 1e-3)
LOSS_FNS=("mse" "smooth_l1")
BATCH_SIZES=(16 32)

SEEDS=(2025 2026)

mkdir -p "${PROGRESS_DIR}"

send_wechat_notify() {
    if [ "${RUN_NOTIFY}" != "1" ]; then
        return 0
    fi

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

check_embeddings_ready() {
    if [ ! -d "${EMBED_ROOT}/train" ] || [ ! -d "${EMBED_ROOT}/val" ] || [ ! -d "${EMBED_ROOT}/test" ]; then
        echo "❌ 缺少 ILI 的 embedding 文件目录: ${EMBED_ROOT}/{train,val,test}"
        echo "请先运行: bash /root/0/T3Time/scripts/generate_qwen3_0.6b_embeddings_ILI_36.sh 0"
        return 1
    fi
    return 0
}

print_progress() {
    local done_count=0
    local now elapsed avg eta
    done_count=$(find "${PROGRESS_DIR}" -maxdepth 1 -name '*.done' 2>/dev/null | wc -l)
    now=$(date +%s)
    elapsed=$((now - SCRIPT_START_TS))

    if [ "${done_count}" -gt 0 ]; then
        avg=$((elapsed / done_count))
        eta=$(((TOTAL_TRACKED - done_count) * avg))
    else
        avg=0
        eta=-1
    fi

    printf '[%s] 进度: %s/%s, 已耗时: %02d:%02d:%02d' \
        "$(date '+%H:%M:%S')" \
        "${done_count}" \
        "${TOTAL_TRACKED}" \
        $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))

    if [ "${eta}" -ge 0 ]; then
        printf ', 预计剩余: %02d:%02d:%02d' $((eta / 3600)) $(((eta % 3600) / 60)) $((eta % 60))
    else
        printf ', 预计剩余: 计算中'
    fi

    printf '\n'
}

D_LLM=1024
param_configs=()
for CHANNEL in "${CHANNELS[@]}"; do
    for DROPOUT_N in "${DROPOUTS[@]}"; do
        for HEAD in "${HEADS[@]}"; do
            if [ $((CHANNEL % HEAD)) -eq 0 ] && [ $((D_LLM % HEAD)) -eq 0 ]; then
                for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
                    for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
                        for LOSS_FN in "${LOSS_FNS[@]}"; do
                            for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                                param_configs+=("${CHANNEL}|${DROPOUT_N}|${HEAD}|${LEARNING_RATE}|${WEIGHT_DECAY}|${LOSS_FN}|${BATCH_SIZE}")
                            done
                        done
                    done
                done
            fi
        done
    done
done

experiments=()
for seed in "${SEEDS[@]}"; do
    for param_config in "${param_configs[@]}"; do
        experiments+=("${param_config}|${seed}")
    done
done

total_exps=${#experiments[@]}
total_param_configs=${#param_configs[@]}
total_seeds=${#SEEDS[@]}
SCRIPT_START_TS=$(date +%s)
TOTAL_TRACKED=${total_exps}

check_embeddings_ready || exit 1

if [ "$#" -eq 0 ]; then
    NUM_GPUS=8
    EXP_PER_GPU=$((total_exps / NUM_GPUS))
    REMAINDER=$((total_exps % NUM_GPUS))

    echo "=========================================="
    echo "检测到未指定参数，启用 ILI 默认八卡自动并行模式"
    echo "预测长度: ${PRED_LENS[@]} (共 ${#PRED_LENS[@]} 个，每个参数组合将依次运行)"
    echo "参数组合数: ${total_param_configs}"
    echo "种子数: ${total_seeds}"
    echo "总实验数: ${total_exps} (${total_param_configs} × ${total_seeds})"
    echo "GPU数量: ${NUM_GPUS}"
    echo "每卡并行数: ${PARALLEL}"
    echo "进度目录: ${PROGRESS_DIR}"
    echo "实验分配:"

    PIDS=()
    start_idx=0

    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        if [ ${gpu} -lt ${REMAINDER} ]; then
            end_idx=$((start_idx + EXP_PER_GPU))
        else
            end_idx=$((start_idx + EXP_PER_GPU - 1))
        fi

        if [ ${gpu} -eq $((NUM_GPUS - 1)) ]; then
            end_idx=$((total_exps - 1))
        fi

        echo "  GPU${gpu}: 实验 [$start_idx, $end_idx] ($((end_idx - start_idx + 1)) 个)"
        RUN_NOTIFY=0 RUN_TAG="${RUN_TAG}" CUDA_VISIBLE_DEVICES=${gpu} bash "$0" ${gpu} ${start_idx} ${end_idx} "${PARALLEL}" &
        PIDS+=($!)

        start_idx=$((end_idx + 1))
    done

    echo "=========================================="

    while true; do
        running_count=$(jobs -r | wc -l)
        print_progress
        if [ "${running_count}" -eq 0 ]; then
            break
        fi
        sleep 30
    done

    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done

    echo "✅ ILI 八卡自动并行超参数+多种子搜索已完成"
    send_wechat_notify
    exit 0
fi

if [ ${END_IDX} -eq -1 ]; then
    END_IDX=$((total_exps - 1))
fi

actual_exps=$((END_IDX - START_IDX + 1))
TOTAL_TRACKED=${actual_exps}

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen ILI 超参数+多种子寻优"
echo "GPU: ${GPU_ID}, 实验范围: [${START_IDX}, ${END_IDX}] / ${total_exps}"
echo "并行数: ${PARALLEL}"
echo "Pred_Len: ${PRED_LENS[@]}"
echo "Seq_Len: ${SEQ_LEN}"
echo "进度目录: ${PROGRESS_DIR}"
echo "当前GPU将运行: ${actual_exps} 个实验"
echo "=========================================="

run_experiment() {
    local exp_idx=$1
    local exp_config=$2
    local gpu_id=$3
    local exp_start_ts exp_end_ts exp_elapsed

    IFS='|' read -r CHANNEL DROPOUT_N HEAD LEARNING_RATE WEIGHT_DECAY LOSS_FN BATCH_SIZE SEED <<< "${exp_config}"

    exp_start_ts=$(date +%s)
    echo "[实验 ${exp_idx}/${total_exps}] GPU${gpu_id} 开始..."
    echo "  Channel: ${CHANNEL}, Dropout: ${DROPOUT_N}, Head: ${HEAD}"
    echo "  LR: ${LEARNING_RATE}, WD: ${WEIGHT_DECAY}, Loss: ${LOSS_FN}, BS: ${BATCH_SIZE}, Seed: ${SEED}"
    echo "  将依次运行预测长度: ${PRED_LENS[@]}"

    for PRED_LEN in "${PRED_LENS[@]}"; do
        log_file="${LOG_DIR}/pred${PRED_LEN}_c${CHANNEL}_d${DROPOUT_N}_h${HEAD}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_loss${LOSS_FN}_bs${BATCH_SIZE}_seed${SEED}.log"
        echo "    -> 开始 Pred_Len=${PRED_LEN} ..."
        python -u /root/0/T3Time/train_frets_gated_qwen.py \
            --data_path "${DATA_PATH}" \
            --batch_size "${BATCH_SIZE}" \
            --seq_len "${SEQ_LEN}" \
            --num_nodes "${NUM_NODES}" \
            --pred_len "${PRED_LEN}" \
            --epochs "${EPOCHS}" \
            --es_patience "${PATIENCE}" \
            --seed "${SEED}" \
            --channel "${CHANNEL}" \
            --learning_rate "${LEARNING_RATE}" \
            --dropout_n "${DROPOUT_N}" \
            --weight_decay "${WEIGHT_DECAY}" \
            --e_layer "${E_LAYER}" \
            --d_layer "${D_LAYER}" \
            --head "${HEAD}" \
            --loss_fn "${LOSS_FN}" \
            --lradj "${LRADJ}" \
            --embed_version "${EMBED_VERSION}" \
            --model_id "${MODEL_ID}" \
            > "${log_file}" 2>&1

        exit_code=$?
        if [ ${exit_code} -eq 0 ]; then
            echo "    ✅ Pred_Len=${PRED_LEN} 完成"
        else
            echo "    ⚠️ Pred_Len=${PRED_LEN} 失败 (退出码: ${exit_code})"
        fi
    done

    exp_end_ts=$(date +%s)
    exp_elapsed=$((exp_end_ts - exp_start_ts))
    touch "${PROGRESS_DIR}/exp_${exp_idx}.done"
    echo "  ✅ GPU${gpu_id} 实验 ${exp_idx} 的所有预测长度已运行完毕 (耗时 ${exp_elapsed}s)"
}

current_idx=${START_IDX}
while [ ${current_idx} -le ${END_IDX} ]; do
    while [ "$(jobs -r | wc -l)" -ge "${PARALLEL}" ]; do
        print_progress
        sleep 10
    done

    exp_config=${experiments[${current_idx}]}
    run_experiment ${current_idx} "${exp_config}" ${GPU_ID} &
    current_idx=$((current_idx + 1))
done

while true; do
    running_jobs=$(jobs -r | wc -l)
    print_progress
    if [ "${running_jobs}" -eq 0 ]; then
        break
    fi
    sleep 30
done

wait

echo "=========================================="
echo "✅ GPU${GPU_ID} 所有 ILI 实验完成！"
echo "结果已追加到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo "=========================================="
send_wechat_notify
