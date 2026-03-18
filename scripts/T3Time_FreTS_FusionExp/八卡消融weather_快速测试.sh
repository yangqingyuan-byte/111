#!/bin/bash
# T3Time_FreTS_FusionExp Weather 八卡消融快速验证脚本
# 用途：验证按 Weather 最佳基线参数做消融时，训练入口、日志落盘和多卡分发流程是否正常

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

GPU_ID=${1:-0}
START_IDX=${2:-0}
END_IDX=${3:--1}
PARALLEL=${4:-1}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_FusionExp_Ablation/weather_quick_test"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

DATA_PATH="Weather"
NUM_NODES=21
PRED_LENS=(96)
EPOCHS=1
MODEL_ID="T3Time_FreTS_FusionExp_Ablation_Weather_QuickTest"
BEST_CONFIG_JSON=${BEST_CONFIG_JSON:-$(ls -t /root/0/T3Time/Results/best_config_exports/Weather_all_models_mse_*.json 2>/dev/null | head -n1)}

QUICK_ABLATION_CONFIGS=(
    "Full|BASE|BASE|BASE|1|1|1|1"
    "w_o_FreTS|BASE|BASE|BASE|0|0|0|1"
    "w_o_Sparsity|BASE|BASE|0.000|1|1|0|1"
    "w_o_ImprovedGate|BASE|BASE|BASE|1|1|1|0"
)

total_exps=${#QUICK_ABLATION_CONFIGS[@]}

get_base_config() {
    local pred_len=$1
    if [ -n "${BEST_CONFIG_JSON}" ] && [ -f "${BEST_CONFIG_JSON}" ]; then
        python - <<'PY' "${BEST_CONFIG_JSON}" "${pred_len}"
import json, sys
path = sys.argv[1]
pred_len = int(sys.argv[2])
data = json.load(open(path, 'r', encoding='utf-8'))
target = None
for item in data:
    if int(item.get('pred_len')) == pred_len:
        target = item
        break
if target is None:
    raise SystemExit(f"未在 {path} 中找到 pred_len={pred_len} 的最佳配置")
params = target['params']
fields = [
    str(params.get('seq_len', 96)),
    str(params.get('channel', 96)),
    str(params.get('batch_size', 32)),
    str(params.get('learning_rate', 1e-4)),
    str(params.get('dropout_n', 0.5)),
    str(params.get('weight_decay', 1e-4)),
    str(params.get('e_layer', 1)),
    str(params.get('d_layer', 1)),
    str(params.get('head', 8)),
    str(params.get('loss_fn', 'smooth_l1')),
    str(params.get('lradj', 'type1')),
    str(params.get('patience', 10)),
    str(params.get('embed_version', 'qwen3_0.6b')),
    str(params.get('frets_scale', 0.018)),
    str(params.get('sparsity_threshold', 0.009)),
    str(params.get('fusion_mode', 'gate')),
    str(params.get('seed', 2026)),
]
print("|".join(fields))
PY
        return
    fi

    case "${pred_len}" in
        96)  echo "96|96|32|0.0001|0.5|0.0001|1|1|8|smooth_l1|type1|10|qwen3_0.6b|0.018|0.009|gate|2026" ;;
        192) echo "96|96|16|0.0001|0.5|0.0001|1|1|8|smooth_l1|type1|10|qwen3_0.6b|0.018|0.009|gate|2026" ;;
        336) echo "96|96|32|5e-05|0.5|0.0001|1|1|16|smooth_l1|type1|10|qwen3_0.6b|0.018|0.009|gate|2026" ;;
        720) echo "96|96|32|7.5e-05|0.5|0.0001|1|1|16|smooth_l1|type1|10|qwen3_0.6b|0.018|0.009|gate|2026" ;;
        *)   echo "96|96|32|0.0001|0.5|0.0001|1|1|8|smooth_l1|type1|10|qwen3_0.6b|0.018|0.009|gate|2026" ;;
    esac
}

get_embed_root() {
    if [ -n "${BEST_CONFIG_JSON}" ] && [ -f "${BEST_CONFIG_JSON}" ]; then
        python - <<'PY' "${BEST_CONFIG_JSON}"
import json, sys
data = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
embed_version = data[0]['params'].get('embed_version', 'qwen3_0.6b')
print(f"/root/0/T3Time/Embeddings/Weather/{embed_version}")
PY
    else
        echo "/root/0/T3Time/Embeddings/Weather/qwen3_0.6b"
    fi
}

EMBED_ROOT="$(get_embed_root)"

if [ ! -d "${EMBED_ROOT}/train" ] || [ ! -d "${EMBED_ROOT}/val" ] || [ ! -d "${EMBED_ROOT}/test" ]; then
    echo "❌ 缺少 Weather 的 embedding 文件目录: ${EMBED_ROOT}/{train,val,test}"
    echo "请先确认 Weather 的 qwen3 embeddings 已生成。"
    exit 1
fi

if [ "$#" -eq 0 ]; then
    NUM_GPUS=8
    if [ ${total_exps} -lt ${NUM_GPUS} ]; then
        NUM_GPUS=${total_exps}
    fi
    if [ ${NUM_GPUS} -lt 1 ]; then
        echo "没有可运行的实验配置，退出。"
        exit 1
    fi

    echo "=========================================="
    echo "Weather 消融快速验证：自动并行模式"
    echo "预测长度: ${PRED_LENS[*]}"
    if [ -n "${BEST_CONFIG_JSON}" ] && [ -f "${BEST_CONFIG_JSON}" ]; then
        echo "最佳配置文件: ${BEST_CONFIG_JSON}"
    else
        echo "最佳配置文件: 未提供，使用内置 fallback 基线参数"
    fi
    echo "实验数: ${total_exps}"
    echo "GPU数量: ${NUM_GPUS}"
    echo "每卡并行数: ${PARALLEL}"
    echo "=========================================="

    PIDS=()
    start_idx=0
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        end_idx=${start_idx}
        echo "  GPU${gpu}: 实验 [${start_idx}, ${end_idx}]"
        CUDA_VISIBLE_DEVICES=${gpu} bash "$0" "${gpu}" "${start_idx}" "${end_idx}" "${PARALLEL}" &
        PIDS+=($!)
        start_idx=$((end_idx + 1))
    done

    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done

    echo "✅ Weather 消融快速验证完成"
    exit 0
fi

if [ ${END_IDX} -eq -1 ]; then
    END_IDX=$((total_exps - 1))
fi

actual_exps=$((END_IDX - START_IDX + 1))

echo "=========================================="
echo "Weather 八卡消融快速验证"
echo "GPU: ${GPU_ID}, 实验范围: [${START_IDX}, ${END_IDX}] / ${total_exps}"
echo "并行数: ${PARALLEL}"
echo "预测长度: ${PRED_LENS[*]}"
echo "Epochs: ${EPOCHS}"
echo "当前GPU将运行: ${actual_exps} 个实验"
echo "=========================================="

run_experiment() {
    local exp_idx=$1
    local exp_config=$2
    local gpu_id=$3

    IFS='|' read -r EXP_NAME FUSION_MODE_OVERRIDE FRETS_SCALE_OVERRIDE SPARSITY_THRESHOLD_OVERRIDE USE_FRETS USE_COMPLEX USE_SPARSITY USE_IMPROVED_GATE <<< "${exp_config}"

    echo "[实验 ${exp_idx}/${total_exps}] GPU${gpu_id} 开始 ${EXP_NAME}"
    for PRED_LEN in "${PRED_LENS[@]}"; do
        IFS='|' read -r SEQ_LEN CHANNEL BATCH_SIZE LEARNING_RATE DROPOUT_N WEIGHT_DECAY E_LAYER D_LAYER HEAD LOSS_FN LRADJ PATIENCE EMBED_VERSION BASE_FRETS_SCALE BASE_SPARSITY_THRESHOLD BASE_FUSION_MODE SEED <<< "$(get_base_config "${PRED_LEN}")"
        FUSION_MODE="${BASE_FUSION_MODE}"
        FRETS_SCALE="${BASE_FRETS_SCALE}"
        SPARSITY_THRESHOLD="${BASE_SPARSITY_THRESHOLD}"
        if [ "${FUSION_MODE_OVERRIDE}" != "BASE" ]; then
            FUSION_MODE="${FUSION_MODE_OVERRIDE}"
        fi
        if [ "${FRETS_SCALE_OVERRIDE}" != "BASE" ]; then
            FRETS_SCALE="${FRETS_SCALE_OVERRIDE}"
        fi
        if [ "${SPARSITY_THRESHOLD_OVERRIDE}" != "BASE" ]; then
            SPARSITY_THRESHOLD="${SPARSITY_THRESHOLD_OVERRIDE}"
        fi

        log_file="${LOG_DIR}/${EXP_NAME}_pred${PRED_LEN}_seed${SEED}.log"
        python -u /root/0/T3Time/train_frets_gated_qwen_fusion_exp.py \
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
            --fusion_mode "${FUSION_MODE}" \
            --frets_scale "${FRETS_SCALE}" \
            --sparsity_threshold "${SPARSITY_THRESHOLD}" \
            --use_frets "${USE_FRETS}" \
            --use_complex "${USE_COMPLEX}" \
            --use_sparsity "${USE_SPARSITY}" \
            --use_improved_gate "${USE_IMPROVED_GATE}" \
            --lradj "${LRADJ}" \
            --embed_version "${EMBED_VERSION}" \
            --model_id "${MODEL_ID}_${EXP_NAME}" \
            > "${log_file}" 2>&1

        exit_code=$?
        if [ ${exit_code} -eq 0 ]; then
            echo "    ✅ ${EXP_NAME} / Pred_Len=${PRED_LEN} 完成"
        else
            echo "    ⚠️ ${EXP_NAME} / Pred_Len=${PRED_LEN} 失败 (退出码: ${exit_code})"
            echo "    日志: ${log_file}"
        fi
    done
}

current_idx=${START_IDX}
while [ ${current_idx} -le ${END_IDX} ]; do
    while [ "$(jobs -r | wc -l)" -ge "${PARALLEL}" ]; do
        sleep 2
    done

    exp_config=${QUICK_ABLATION_CONFIGS[${current_idx}]}
    run_experiment "${current_idx}" "${exp_config}" "${GPU_ID}" &
    current_idx=$((current_idx + 1))
done

wait

echo "=========================================="
echo "✅ GPU${GPU_ID} Weather 消融快速验证完成"
echo "日志目录: ${LOG_DIR}"
echo "结果日志: ${RESULT_LOG}"
echo "=========================================="
