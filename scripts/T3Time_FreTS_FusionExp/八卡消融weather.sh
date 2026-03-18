#!/bin/bash
# T3Time_FreTS_FusionExp Weather 八卡并行消融实验脚本
# 用途：按每个 pred_len 的最佳基线参数做模块消融，并使用多种子结果便于后续筛选论文表格

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

RUN_NOTIFY=${RUN_NOTIFY:-0}

GPU_ID=${1:-0}
START_IDX=${2:-0}
END_IDX=${3:--1}
PARALLEL=${4:-2}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

LOG_DIR="/root/0/T3Time/Results/T3Time_FreTS_FusionExp_Ablation/weather"
RESULT_LOG="/root/0/T3Time/experiment_results.log"
mkdir -p "${LOG_DIR}"

MONITOR_SCRIPT="/root/0/T3Time/自动监控gpu和发通知dao微信.py"
MONITOR_CONFIG="/root/0/T3Time/.gpu_monitor_config"

DATA_PATH="Weather"
NUM_NODES=21
PRED_LENS=(96 192 336 720)
EPOCHS=${EPOCHS:-120}
MODEL_NAME="T3Time_FreTS_Gated_Qwen_FusionExp"
MODEL_ID="T3Time_FreTS_FusionExp_Ablation_Weather"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
PROGRESS_DIR="/root/0/T3Time/Results/progress_ablation_weather_${RUN_TAG}"

BEST_CONFIG_JSON=${BEST_CONFIG_JSON:-$(ls -t /root/0/T3Time/Results/best_config_exports/Weather_all_models_mse_*.json 2>/dev/null | head -n1)}

# 默认多种子，便于后续筛选更稳定或更有代表性的结果
SEEDS_STR="${SEEDS_STR:-2025 2026 2027}"
read -r -a SEEDS <<< "${SEEDS_STR}"

mkdir -p "${PROGRESS_DIR}"

send_wechat_notify() {
    if [ "${RUN_NOTIFY}" != "1" ]; then
        return 0
    fi

    if [ ! -f "${MONITOR_SCRIPT}" ] || [ ! -f "${MONITOR_CONFIG}" ]; then
        return 0
    fi

    if command -v timeout >/dev/null 2>&1; then
        timeout 300 python "${MONITOR_SCRIPT}" --interval 15 --check-count 2 --threshold 0 --config "${MONITOR_CONFIG}" || true
    else
        nohup python "${MONITOR_SCRIPT}" --interval 15 --check-count 2 --threshold 0 --config "${MONITOR_CONFIG}" >/dev/null 2>&1 &
    fi
}

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
]
print("|".join(fields))
PY
        return
    fi

    case "${pred_len}" in
        96)  echo "96|96|32|0.0001|0.5|0.0001|1|1|8|smooth_l1|type1|10|qwen3_0.6b|0.018|0.009|gate" ;;
        192) echo "96|96|16|0.0001|0.5|0.0001|1|1|8|smooth_l1|type1|10|qwen3_0.6b|0.018|0.009|gate" ;;
        336) echo "96|96|32|5e-05|0.5|0.0001|1|1|16|smooth_l1|type1|10|qwen3_0.6b|0.018|0.009|gate" ;;
        720) echo "96|96|32|7.5e-05|0.5|0.0001|1|1|16|smooth_l1|type1|10|qwen3_0.6b|0.018|0.009|gate" ;;
        *)   echo "96|96|32|0.0001|0.5|0.0001|1|1|8|smooth_l1|type1|10|qwen3_0.6b|0.018|0.009|gate" ;;
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

check_embeddings_ready() {
    local embed_root
    embed_root="$(get_embed_root)"
    if [ ! -d "${embed_root}/train" ] || [ ! -d "${embed_root}/val" ] || [ ! -d "${embed_root}/test" ]; then
        echo "❌ 缺少 Weather 的 embedding 文件目录: ${embed_root}/{train,val,test}"
        echo "请先确认 Weather 的 qwen3 embeddings 已生成。"
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

# 格式:
# exp_name|fusion_mode_override|frets_scale_override|sparsity_override|use_frets|use_complex|use_sparsity|use_improved_gate
ABLATION_CONFIGS=(
    "Full|BASE|BASE|BASE|1|1|1|1"
    "w_o_FreTS|BASE|BASE|BASE|0|0|0|1"
    "w_o_Sparsity|BASE|BASE|0.000|1|1|0|1"
    "w_o_ImprovedGate|BASE|BASE|BASE|1|1|1|0"
    "FFT_Complex|BASE|BASE|BASE|0|1|0|1"
    "Fusion_Weighted|weighted|BASE|BASE|1|1|1|1"
    "Fusion_CrossAttn|cross_attn|BASE|BASE|1|1|1|1"
    "Fusion_Hybrid|hybrid|BASE|BASE|1|1|1|1"
)

experiments=()
for seed in "${SEEDS[@]}"; do
    for pred_len in "${PRED_LENS[@]}"; do
        for ablation_cfg in "${ABLATION_CONFIGS[@]}"; do
            experiments+=("${pred_len}|${ablation_cfg}|${seed}")
        done
    done
done

total_exps=${#experiments[@]}
SCRIPT_START_TS=$(date +%s)
TOTAL_TRACKED=${total_exps}

check_embeddings_ready || exit 1

if [ "$#" -eq 0 ]; then
    NUM_GPUS=8
    if [ "${total_exps}" -lt "${NUM_GPUS}" ]; then
        NUM_GPUS=${total_exps}
    fi

    EXP_PER_GPU=$((total_exps / NUM_GPUS))
    REMAINDER=$((total_exps % NUM_GPUS))

    echo "=========================================="
    echo "检测到未指定参数，启用 Weather 八卡消融自动并行模式"
    echo "数据集: ${DATA_PATH}"
    echo "预测长度: ${PRED_LENS[*]}"
    if [ -n "${BEST_CONFIG_JSON}" ] && [ -f "${BEST_CONFIG_JSON}" ]; then
        echo "最佳配置文件: ${BEST_CONFIG_JSON}"
    else
        echo "最佳配置文件: 未提供，使用内置 fallback 基线参数"
    fi
    echo "消融配置数: ${#ABLATION_CONFIGS[@]}"
    echo "种子数: ${#SEEDS[@]} (${SEEDS[*]})"
    echo "总实验数: ${total_exps}"
    echo "GPU数量: ${NUM_GPUS}"
    echo "每卡并行数: ${PARALLEL}"
    echo "进度目录: ${PROGRESS_DIR}"
    echo "=========================================="

    PIDS=()
    start_idx=0

    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        if [ "${gpu}" -lt "${REMAINDER}" ]; then
            end_idx=$((start_idx + EXP_PER_GPU))
        else
            end_idx=$((start_idx + EXP_PER_GPU - 1))
        fi

        if [ "${gpu}" -eq $((NUM_GPUS - 1)) ]; then
            end_idx=$((total_exps - 1))
        fi

        echo "  GPU${gpu}: 实验 [${start_idx}, ${end_idx}]"
        RUN_NOTIFY=0 RUN_TAG="${RUN_TAG}" CUDA_VISIBLE_DEVICES=${gpu} bash "$0" "${gpu}" "${start_idx}" "${end_idx}" "${PARALLEL}" &
        PIDS+=($!)
        start_idx=$((end_idx + 1))
    done

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

    echo "✅ Weather 八卡消融实验全部完成"
    send_wechat_notify
    exit 0
fi

if [ ${END_IDX} -eq -1 ]; then
    END_IDX=$((total_exps - 1))
fi

actual_exps=$((END_IDX - START_IDX + 1))
TOTAL_TRACKED=${actual_exps}

echo "=========================================="
echo "Weather 八卡并行消融实验"
echo "GPU: ${GPU_ID}, 实验范围: [${START_IDX}, ${END_IDX}] / ${total_exps}"
echo "并行数: ${PARALLEL}"
echo "预测长度: ${PRED_LENS[*]}"
echo "当前GPU将运行: ${actual_exps} 个实验"
echo "=========================================="

run_experiment() {
    local exp_idx=$1
    local exp_config=$2
    local gpu_id=$3
    local done_file

    IFS='|' read -r PRED_LEN EXP_NAME FUSION_MODE_OVERRIDE FRETS_SCALE_OVERRIDE SPARSITY_THRESHOLD_OVERRIDE USE_FRETS USE_COMPLEX USE_SPARSITY USE_IMPROVED_GATE SEED <<< "${exp_config}"
    IFS='|' read -r SEQ_LEN CHANNEL BATCH_SIZE LEARNING_RATE DROPOUT_N WEIGHT_DECAY E_LAYER D_LAYER HEAD LOSS_FN LRADJ PATIENCE EMBED_VERSION BASE_FRETS_SCALE BASE_SPARSITY_THRESHOLD BASE_FUSION_MODE <<< "$(get_base_config "${PRED_LEN}")"

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

    echo "[实验 ${exp_idx}/${total_exps}] GPU${gpu_id} 开始 ${EXP_NAME} (pred_len=${PRED_LEN}, seed=${SEED})"
    echo "  base: channel=${CHANNEL}, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}, drop=${DROPOUT_N}, wd=${WEIGHT_DECAY}, head=${HEAD}, loss=${LOSS_FN}"
    echo "  ablation: fusion=${FUSION_MODE}, scale=${FRETS_SCALE}, sparsity=${SPARSITY_THRESHOLD}, use_frets=${USE_FRETS}, use_sparsity=${USE_SPARSITY}, use_improved_gate=${USE_IMPROVED_GATE}"

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

    done_file="${PROGRESS_DIR}/$(printf '%04d' "${exp_idx}")_${EXP_NAME}_pred${PRED_LEN}_seed${SEED}.done"
    touch "${done_file}"
}

current_idx=${START_IDX}
while [ ${current_idx} -le ${END_IDX} ]; do
    while [ "$(jobs -r | wc -l)" -ge "${PARALLEL}" ]; do
        sleep 2
    done

    exp_config=${experiments[${current_idx}]}
    run_experiment "${current_idx}" "${exp_config}" "${GPU_ID}" &
    current_idx=$((current_idx + 1))
done

wait

echo "=========================================="
echo "✅ GPU${GPU_ID} 分配的 Weather 消融实验完成"
echo "结果已追加到: ${RESULT_LOG}"
echo "日志目录: ${LOG_DIR}"
echo "=========================================="
