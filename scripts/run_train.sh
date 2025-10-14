#!/bin/bash
# =============================================
# run_train.sh
# 功能: 通用训练脚本，根据模型名自动加载配置文件
# 支持后台运行、resume模式、日志管理
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
USE_RESUME=false

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
        --resume)
            USE_RESUME=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--config <path>] [<model_type>] [--nproc <num>] [--port <port>] [--resume]"
            echo "Example:"
            echo "  ./train_launcher.sh pi0"
            echo "  ./train_launcher.sh smolvla --resume"
            echo "  ./train_launcher.sh --config configs/custom.yaml"
            exit 0
            ;;
        *)
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
    # 自动匹配 configs/train_<MODEL_TYPE>.yaml
    CONFIG_PATH="$PROJECT_ROOT/configs/${MODEL_TYPE}.yaml"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Error: Config file not found for model '$MODEL_TYPE': $CONFIG_PATH"
        echo "Hint: Expected file path is '$PROJECT_ROOT/configs/${MODEL_TYPE}.yaml'"
        exit 1
    fi
    echo "Using config for model '$MODEL_TYPE': $CONFIG_PATH"
else
    echo "Error: Either --config <path> or <model_type> must be provided."
    exit 1
fi

# -------------------------
# 检查 torchrun 是否存在
# -------------------------
if ! command -v torchrun >/dev/null 2>&1; then
    echo "torchrun not found, please activate the proper environment."
    exit 1
fi

# -------------------------
# 检查 output_dir
# -------------------------
OUTPUT_DIR_ORIG=$(awk '/^[[:space:]]*output_dir:/{gsub(/^[[:space:]]*output_dir:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH")

if [[ -n "$OUTPUT_DIR_ORIG" ]]; then
    if [[ "$OUTPUT_DIR_ORIG" != /* ]]; then
        OUTPUT_DIR_ORIG="$PROJECT_ROOT/$OUTPUT_DIR_ORIG"
    fi
fi

RESUME_IN_CONFIG=$(awk '/^[[:space:]]*resume:/{gsub(/^[[:space:]]*resume:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH" | tr '[:upper:]' '[:lower:]')

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
    sed "s|^\([[:space:]]*\)output_dir:.*|\1output_dir: $OUTPUT_DIR_NEW|" "$CONFIG_PATH" > "$TEMP_CONFIG"
    CONFIG_PATH="$TEMP_CONFIG"
    echo "Auto-adjusted output_dir to avoid conflict: $OUTPUT_DIR_NEW"
fi

# -------------------------
# 日志设置
# -------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/ckpt/logs/train_${MODEL_TYPE:-custom}_${TIMESTAMP}.log"
mkdir -p "$PROJECT_ROOT/ckpt/logs"

# -------------------------
# 环境变量
# -------------------------
# export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.8"
# echo "Using PYTORCH_NPU_ALLOC_CONF=$PYTORCH_NPU_ALLOC_CONF"

# -------------------------
# 构建训练命令
# -------------------------
TRAIN_ARGS=(--config_path="$CONFIG_PATH")

if [[ "$USE_RESUME" == true ]]; then
    TRAIN_ARGS+=(--resume=true)
    echo "Resume mode enabled."
fi

# -------------------------
# 启动训练
# -------------------------
nohup torchrun --nproc_per_node="$NPROC" \
    --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
    -m lerobot.scripts.lerobot_train \
    "${TRAIN_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &

PID=$!

echo ""
echo "Training started for model: ${MODEL_TYPE:-custom}"
echo "Config file: $CONFIG_PATH"
echo "Log file: $LOG_FILE"
echo "PID: $PID"
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
#     > pi0.log 2>&1 &为什么我执行 sh scripts/run_train.sh pi05 --resume --config ./ckpt/pi05_libero/checkpoints/last/pretrained_model/train_config.json 训练程序没有执行起来