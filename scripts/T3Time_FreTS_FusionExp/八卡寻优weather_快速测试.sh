#!/bin/bash
# T3Time_FreTS_Gated_Qwen Weather 八卡寻优快速测试脚本
# 目的：快速验证 Weather 数据集训练入口、日志落盘和多卡分发流程

set -uo pipefail

unset __vsc_prompt_cmd_original 2>/dev/null || true

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate TimeCMA_Qwen3 2>/dev/null || source activate TimeCMA_Qwen3 2>/dev/null || true

export PYTHONPATH="/root/0/T3Time:${PYTHONPATH-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU_ID=${1:-0}
START_IDX=${2:-0}
END_IDX=${3:--1}
PARALLEL=${4:-1}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_Gated_Qwen_Hyperopt/weather_quick_test"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

DATA_PATH="Weather"
NUM_NODES=21
SEQ_LEN=96
PRED_LENS=(96)
E_LAYER=1
D_LAYER=1
EPOCHS=1
PATIENCE=1
LRADJ="type1"
EMBED_VERSION="qwen3_0.6b"
MODEL_ID="T3Time_FreTS_Gated_Qwen_Hyperopt_Weather_QuickTest"

CHANNELS=(64)
DROPOUTS=(0.1)
HEADS=(8)
LEARNING_RATES=(1e-4)
WEIGHT_DECAYS=(1e-4)
LOSS_FNS=("mse")
BATCH_SIZES=(16)
SEEDS=(2025)

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

if [ "$#" -eq 0 ]; then
    NUM_GPUS=8
    if [ ${total_exps} -lt ${NUM_GPUS} ]; then
        NUM_GPUS=${total_exps}
    fi
    if [ ${NUM_GPUS} -lt 1 ]; then
        echo "没有可运行的实验配置，退出。"
        exit 1
    fi

    EXP_PER_GPU=$((total_exps / NUM_GPUS))
    REMAINDER=$((total_exps % NUM_GPUS))

    echo "=========================================="
    echo "Weather 快速测试：自动并行模式"
    echo "预测长度: ${PRED_LENS[@]}"
    echo "参数组合数: ${total_param_configs}"
    echo "种子数: ${total_seeds}"
    echo "总实验数: ${total_exps}"
    echo "GPU数量: ${NUM_GPUS}"
    echo "每卡并行数: ${PARALLEL}"

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

        echo "  GPU${gpu}: 实验 [${start_idx}, ${end_idx}]"
        CUDA_VISIBLE_DEVICES=${gpu} bash "$0" ${gpu} ${start_idx} ${end_idx} "${PARALLEL}" &
        PIDS+=($!)

        start_idx=$((end_idx + 1))
    done

    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done

    echo "✅ Weather 快速测试已完成"
    exit 0
fi

if [ ${END_IDX} -eq -1 ]; then
    END_IDX=$((total_exps - 1))
fi

actual_exps=$((END_IDX - START_IDX + 1))

echo "=========================================="
echo "Weather 八卡寻优快速测试"
echo "GPU: ${GPU_ID}, 实验范围: [${START_IDX}, ${END_IDX}] / ${total_exps}"
echo "并行数: ${PARALLEL}"
echo "Pred_Len: ${PRED_LENS[@]}"
echo "Epochs: ${EPOCHS}"
echo "当前GPU将运行: ${actual_exps} 个实验"
echo "=========================================="

run_experiment() {
    local exp_idx=$1
    local exp_config=$2
    local gpu_id=$3

    IFS='|' read -r CHANNEL DROPOUT_N HEAD LEARNING_RATE WEIGHT_DECAY LOSS_FN BATCH_SIZE SEED <<< "${exp_config}"

    echo "[实验 ${exp_idx}/${total_exps}] GPU${gpu_id} 开始..."

    for PRED_LEN in "${PRED_LENS[@]}"; do
        log_file="${LOG_DIR}/quick_pred${PRED_LEN}_c${CHANNEL}_d${DROPOUT_N}_h${HEAD}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_loss${LOSS_FN}_bs${BATCH_SIZE}_seed${SEED}.log"

        python /root/0/T3Time/train_frets_gated_qwen.py \
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
            echo "    日志: ${log_file}"
        fi
    done
}

current_idx=${START_IDX}
running_jobs=0

while [ ${current_idx} -le ${END_IDX} ]; do
    while [ ${running_jobs} -ge ${PARALLEL} ]; do
        sleep 2
        running_jobs=$(jobs -r | wc -l)
    done

    exp_config=${experiments[${current_idx}]}
    run_experiment ${current_idx} "${exp_config}" ${GPU_ID} &

    current_idx=$((current_idx + 1))
    running_jobs=$(jobs -r | wc -l)
done

wait

echo "=========================================="
echo "✅ GPU${GPU_ID} Weather 快速测试实验完成"
echo "结果已追加到: ${RESULT_LOG}"
echo "日志目录: ${LOG_DIR}"
echo "=========================================="
