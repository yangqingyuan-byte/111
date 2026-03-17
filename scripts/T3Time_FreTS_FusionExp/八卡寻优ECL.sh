#!/bin/bash
# T3Time_FreTS_Gated_Qwen ECL 超参数+多种子寻优脚本
# 在指定参数空间搜索，并对每个参数组合在多个种子上测试
# 支持多 GPU 并行运行，并在全部结束后触发微信监控通知

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
PARALLEL=${4:-1}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_Hyperopt/ECL"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

MONITOR_SCRIPT="/root/0/T3Time/自动监控gpu和发通知dao微信.py"
MONITOR_CONFIG="/root/0/T3Time/.gpu_monitor_config"

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
        echo "❌ 缺少 ECL 的 embedding 文件目录: ${EMBED_ROOT}/{train,val,test}"
        echo "请先运行: bash /root/0/T3Time/scripts/generate_qwen3_0.6b_embeddings.sh ECL 0"
        return 1
    fi
    return 0
}

DATA_PATH="ECL"
NUM_NODES=321
SEQ_LEN=96
PRED_LENS=(96 192 336 720)
E_LAYER=1
D_LAYER=1
EPOCHS=150
PATIENCE=10
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
MODEL_ID="T3Time_FreTS_Gated_Qwen_Hyperopt_ECL"
EMBED_ROOT="/root/0/T3Time/Embeddings/${DATA_PATH}/${EMBED_VERSION}"

CHANNELS=(64 96 256)
DROPOUTS=(0.1 0.3 0.5 0.7)
HEADS=(8 16)
LEARNING_RATES=(5e-5 7.5e-5 1e-4)
WEIGHT_DECAYS=(1e-4 5e-4 1e-3 2e-3)
LOSS_FNS=("mse" "smooth_l1")
# ECL 节点数高，先保守一些，避免显存压力过大
BATCH_SIZES=(8 16)

SEEDS=()
for seed in $(seq 2025 2026); do
    SEEDS+=(${seed})
done

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

check_embeddings_ready || exit 1

if [ "$#" -eq 0 ]; then
    NUM_GPUS=8
    EXP_PER_GPU=$((total_exps / NUM_GPUS))
    REMAINDER=$((total_exps % NUM_GPUS))

    echo "=========================================="
    echo "检测到未指定参数，启用 ECL 默认八卡自动并行模式"
    echo "预测长度: ${PRED_LENS[@]} (共 ${#PRED_LENS[@]} 个，每个参数组合将依次运行)"
    echo "参数组合数: ${total_param_configs}"
    echo "种子数: ${total_seeds} (${SEEDS[0]}-${SEEDS[-1]})"
    echo "总实验数: ${total_exps} (${total_param_configs} × ${total_seeds})"
    echo "GPU数量: ${NUM_GPUS}"
    echo "每卡并行数: ${PARALLEL}"
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

        RUN_NOTIFY=0 CUDA_VISIBLE_DEVICES=${gpu} bash "$0" ${gpu} ${start_idx} ${end_idx} "${PARALLEL}" &
        PIDS+=($!)

        start_idx=$((end_idx + 1))
    done

    echo "=========================================="
    echo ""

    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done

    echo "✅ ECL 八卡自动并行超参数+多种子搜索已完成"
    send_wechat_notify
    exit 0
fi

if [ ${END_IDX} -eq -1 ]; then
    END_IDX=$((total_exps - 1))
fi

actual_exps=$((END_IDX - START_IDX + 1))

echo "=========================================="
echo "T3Time_FreTS_Gated_Qwen ECL 超参数+多种子寻优"
echo "GPU: ${GPU_ID}, 实验范围: [${START_IDX}, ${END_IDX}] / ${total_exps}"
echo "并行数: ${PARALLEL}"
echo "Pred_Len: ${PRED_LENS[@]}"
echo ""
echo "参数搜索空间:"
echo "  Channel: ${CHANNELS[@]}"
echo "  Dropout: ${DROPOUTS[@]}"
echo "  Head: ${HEADS[@]}"
echo "  Learning Rate: ${LEARNING_RATES[@]}"
echo "  Weight Decay: ${WEIGHT_DECAYS[@]}"
echo "  Loss Function: ${LOSS_FNS[@]}"
echo "  Batch Size: ${BATCH_SIZES[@]}"
echo "预测长度: ${PRED_LENS[@]} (共 ${#PRED_LENS[@]} 个，每个参数组合将依次运行)"
echo "种子范围: ${SEEDS[0]}-${SEEDS[-1]} (共 ${total_seeds} 个)"
echo "参数组合数: ${total_param_configs}"
echo "总实验数: ${total_exps} (${total_param_configs} × ${total_seeds})"
echo "当前GPU将运行: ${actual_exps} 个实验（每个实验将依次运行 ${#PRED_LENS[@]} 个预测长度）"
echo "=========================================="
echo ""

run_experiment() {
    local exp_idx=$1
    local exp_config=$2
    local gpu_id=$3

    IFS='|' read -r CHANNEL DROPOUT_N HEAD LEARNING_RATE WEIGHT_DECAY LOSS_FN BATCH_SIZE SEED <<< "${exp_config}"

    echo "[实验 ${exp_idx}/${total_exps}] GPU${gpu_id} 开始..."
    echo "  Channel: ${CHANNEL}, Dropout: ${DROPOUT_N}, Head: ${HEAD}"
    echo "  LR: ${LEARNING_RATE}, WD: ${WEIGHT_DECAY}, Loss: ${LOSS_FN}, BS: ${BATCH_SIZE}"
    echo "  Seed: ${SEED}"
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

    echo "  ✅ GPU${gpu_id} 实验 ${exp_idx} 的所有预测长度已运行完毕"
}

current_idx=${START_IDX}
running_jobs=0

while [ ${current_idx} -le ${END_IDX} ]; do
    while [ ${running_jobs} -ge ${PARALLEL} ]; do
        sleep 5
        running_jobs=$(jobs -r | wc -l)
    done

    exp_config=${experiments[${current_idx}]}
    run_experiment ${current_idx} "${exp_config}" ${GPU_ID} &

    current_idx=$((current_idx + 1))
    running_jobs=$(jobs -r | wc -l)

    echo "  当前运行中: ${running_jobs}/${PARALLEL} 个实验"
done

echo ""
echo "等待所有实验完成..."
wait

echo "=========================================="
echo "✅ GPU${GPU_ID} 所有 ECL 实验完成！"
echo "=========================================="
echo "结果已追加到: ${RESULT_LOG}"
echo "日志文件保存在: ${LOG_DIR}"
echo ""
send_wechat_notify
