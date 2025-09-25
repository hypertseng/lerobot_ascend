#!/bin/bash
# =============================================
# train_launcher.sh
# 功能: 根据模型类型或自定义配置启动训练任务
# 支持后台运行，并记录日志和PID
# 可选参数: --nproc, --port, --config, --resume
# =============================================

# -------------------------
# 获取项目根目录
# -------------------------
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo "Project root: $PROJECT_ROOT"

# -------------------------
# 默认参数
# -------------------------
NPROC=8
MASTER_PORT=29500
MODEL_TYPE=""
CUSTOM_CONFIG=""
USE_RESUME=false  # 👈 新增：是否启用 resume

# -------------------------
# 参数解析
# -------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --config)
            CUSTOM_CONFIG="$2"
            shift 2
            ;;
        --resume)  # 👈 新增 resume 选项
            USE_RESUME=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--config <path>] [<model_type>] [--nproc <num>] [--port <port>] [--resume]"
            echo "Available model types: smolvla, pi0, pi0fast (ignored if --config is provided)"
            echo "  --resume : Resume training from last checkpoint (equivalent to --resume=true in train.py)"
            exit 0
            ;;
        *)
            # 第一个非选项参数视为 model_type
            if [ -z "$MODEL_TYPE" ]; then
                MODEL_TYPE="$1"
            else
                echo "Unknown option or too many positional args: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# -------------------------
# 确定配置文件路径
# -------------------------
if [ -n "$CUSTOM_CONFIG" ]; then
    CONFIG_PATH="$PROJECT_ROOT/$CUSTOM_CONFIG"
    echo "Using custom config: $CONFIG_PATH"
elif [ -n "$MODEL_TYPE" ]; then
    declare -A CONFIG_MAP
    CONFIG_MAP=(
        ["smolvla"]="$PROJECT_ROOT/configs/train_smolvla.yaml"
        ["pi0"]="$PROJECT_ROOT/configs/train_pi0.yaml"
        ["pi0fast"]="$PROJECT_ROOT/configs/train_pi0fast.yaml"
    )
    CONFIG_PATH=${CONFIG_MAP[$MODEL_TYPE]}
    if [ -z "$CONFIG_PATH" ]; then
        echo "Unknown model type: $MODEL_TYPE"
        exit 1
    fi
    echo "Using config for model '$MODEL_TYPE': $CONFIG_PATH"
else
    echo "Error: Either --config <path> or <model_type> must be provided."
    exit 1
fi

# -------------------------
# 检查配置文件是否存在
# -------------------------
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config file not found: $CONFIG_PATH"
    exit 1
fi

# -------------------------
# 检查 torchrun 是否存在
# -------------------------
command -v torchrun >/dev/null 2>&1 || {
    echo "torchrun not found, please activate the proper environment."
    exit 1
}

# -------------------------
# 决定日志文件中的“模型名”
# -------------------------
if [ -n "$MODEL_TYPE" ]; then
    LOG_MODEL_NAME="$MODEL_TYPE"
else
    case "$CONFIG_PATH" in
        *pi0fast*) LOG_MODEL_NAME="pi0fast" ;;
        *pi0*)     LOG_MODEL_NAME="pi0" ;;
        *smolvla*) LOG_MODEL_NAME="smolvla" ;;
        *)         LOG_MODEL_NAME="custom" ;;
    esac
fi

# -------------------------
# 自动避免 output_dir 冲突（仅当未启用 resume 时）
# -------------------------
OUTPUT_DIR_ORIG=$(awk '/^[[:space:]]*output_dir:/{gsub(/^[[:space:]]*output_dir:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH")

# 👉 如果是相对路径，加上项目根目录
if [[ -n "$OUTPUT_DIR_ORIG" ]]; then
    if [[ "$OUTPUT_DIR_ORIG" != /* ]]; then
        OUTPUT_DIR_ORIG="$PROJECT_ROOT/$OUTPUT_DIR_ORIG"
    fi
fi

RESUME_IN_CONFIG=$(awk '/^[[:space:]]*resume:/{gsub(/^[[:space:]]*resume:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH" | tr '[:upper:]' '[:lower:]')

# 👇 优先级：命令行 --resume > 配置文件中的 resume
if [[ "$USE_RESUME" == true ]]; then
    RESUME_FINAL=true
elif [[ "$RESUME_IN_CONFIG" == "true" ]]; then
    RESUME_FINAL=true
else
    RESUME_FINAL=false
fi

if [[ "$RESUME_FINAL" != true ]] && [[ -n "$OUTPUT_DIR_ORIG" ]] && [[ -d "$OUTPUT_DIR_ORIG" ]]; then
    OUTPUT_DIR_NEW="${OUTPUT_DIR_ORIG}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR_NEW"

    TEMP_CONFIG=$(mktemp)
    # ⚠️ 注意：写回 CONFIG_PATH 时要保留 PROJECT_ROOT
    sed "s|^\([[:space:]]*\)output_dir:.*|\1output_dir: $OUTPUT_DIR_NEW|" "$CONFIG_PATH" > "$TEMP_CONFIG"
    CONFIG_PATH="$TEMP_CONFIG"

    echo "Auto-adjusted output_dir to avoid conflict: $OUTPUT_DIR_NEW"
fi

# -------------------------
# 输出目录与日志
# -------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/ckpt/train_${LOG_MODEL_NAME}_${TIMESTAMP}.log"

# -------------------------
# NPU 环境变量
# -------------------------
: ${PYTORCH_NPU_ALLOC_CONF:="max_split_size_mb:32,garbage_collection_threshold:0.8"}
export PYTORCH_NPU_ALLOC_CONF
echo "Using PYTORCH_NPU_ALLOC_CONF=$PYTORCH_NPU_ALLOC_CONF"

# -------------------------
# 构建训练命令参数
# -------------------------
TRAIN_ARGS=(
    --config_path="$CONFIG_PATH"
)

# 👇 如果启用了 resume，追加 --resume=True
if [[ "$USE_RESUME" == true ]]; then
    TRAIN_ARGS+=(--resume=true)
    echo "Resume mode enabled."
fi

# -------------------------
# 后台运行训练
# -------------------------
nohup torchrun --nproc_per_node="$NPROC" --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
    -m lerobot.scripts.lerobot_train \
    "${TRAIN_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &

PID=$!

echo "Training started for model: $LOG_MODEL_NAME using config: $CONFIG_PATH (PID: $PID)"
echo "Logs: $LOG_FILE"
# ========== 任务 A: smolvla (2卡) ==========
# ASCEND_RT_VISIBLE_DEVICES=0,1 \
# torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 \
#     /root/zzx/workspace/lerobot-mujoco-tutorial/train_model.py \
#     --config_path=/root/zzx/workspace/lerobot-mujoco-tutorial/smolvla_omy.yaml \
#     --output_dir=/root/zzx/workspace/lerobot-mujoco-tutorial/ckpt/smolvla_train \
#     > smolvla.log 2>&1 &

# ========== 任务 B: pi0 (6卡) ==========
# ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7 \
# torchrun --nproc_per_node=6 --master_addr=127.0.0.1 --master_port=29501 \
#     /root/zzx/workspace/lerobot-mujoco-tutorial/train_model.py \
#     --config_path=/root/zzx/workspace/lerobot-mujoco-tutorial/pi0_omy.yaml \
#     --output_dir=/root/zzx/workspace/lerobot-mujoco-tutorial/ckpt/pi0_train \
#     > pi0.log 2>&1 &