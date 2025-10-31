#!/bin/bash
# =============================================
# run_train.sh
# 功能: 通用训练脚本，根据模型名自动加载配置文件
# 支持后台运行、resume模式、混合精度模式、日志管理
# =============================================

export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:32"

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
USE_MIXED_PRECISION=false
MIXED_PRECISION_TYPE="bf16"  # 默认使用 bf16

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
        --mix|--mixed|--mixed_precision)
            USE_MIXED_PRECISION=true
            # 检查是否指定了精度类型
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                MIXED_PRECISION_TYPE="$2"
                shift 2
            else
                MIXED_PRECISION_TYPE="fp16"
                shift 1
            fi
            ;;
        -h|--help)
            echo "Usage: $0 [--config <path>] [<model_type>] [--nproc <num>] [--port <port>] [--resume] [--mix [fp16|bf16|fp8]]"
            echo "Example:"
            echo "  ./run_train.sh pi0"
            echo "  ./run_train.sh smolvla --resume"
            echo "  ./run_train.sh smolvla --mix fp16"
            echo "  ./run_train.sh --config configs/custom.yaml --mix bf16"
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
    CONFIG_PATH="$PROJECT_ROOT/configs/${MODEL_TYPE}.yaml"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Error: Config file not found for model '$MODEL_TYPE': $CONFIG_PATH"
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
# 检查 output_dir 与 resume
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
    echo "Auto-adjusted output_dir to avoid conflict: $OUTPUT_DIR_NEW"
fi

# -------------------------
# 日志设置
# -------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/ckpt/logs/train_${MODEL_TYPE:-custom}_${TIMESTAMP}.log"
mkdir -p "$PROJECT_ROOT/ckpt/logs"

# -------------------------
# 构建训练命令参数
# -------------------------
TRAIN_ARGS=(--config_path="$CONFIG_PATH")

if [[ "$USE_RESUME" == true ]]; then
    TRAIN_ARGS+=(--resume=true)
    echo "Resume mode enabled."
fi

# -------------------------
# 构建 accelerate 参数
# -------------------------
ACCELERATE_ARGS=(--multi_gpu --num_processes="$NPROC")

if [[ "$USE_MIXED_PRECISION" == true ]]; then
    echo "Using mixed precision: $MIXED_PRECISION_TYPE"
    ACCELERATE_ARGS+=(--mixed_precision "$MIXED_PRECISION_TYPE")
fi

# -------------------------
# 启动训练
# -------------------------
# nohup accelerate launch "${ACCELERATE_ARGS[@]}" \
RAW_OUTPUT_DIR=$(awk '/^[[:space:]]*output_dir:/{gsub(/^[[:space:]]*output_dir:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH")
RAW_JOB_NAME=$(awk '/^[[:space:]]*job_name:/{gsub(/^[[:space:]]*job_name:[[:space:]]*/, ""); print; exit}' "$CONFIG_PATH")
OUTPUT_DIR_FINAL="${RAW_OUTPUT_DIR}_${TIMESTAMP}"
JOB_NAME_FINAL="${RAW_JOB_NAME}_${TIMESTAMP}"

nohup accelerate launch "${ACCELERATE_ARGS[@]}" \
    $(which lerobot-train) \
    "${TRAIN_ARGS[@]}" \
    --output_dir="$OUTPUT_DIR_FINAL" \
    --job_name="$JOB_NAME_FINAL" \
    > "$LOG_FILE" 2>&1 &
PID=$!

echo ""
echo "============================================="
echo "Training started for model: ${MODEL_TYPE:-custom}"
echo "Config file: $CONFIG_PATH"
echo "Log file: $LOG_FILE"
echo "Mixed precision: ${USE_MIXED_PRECISION:+$MIXED_PRECISION_TYPE}"
echo "Resume: $RESUME_FINAL"
echo "Num processes: $NPROC"
echo "PID: $PID"
echo "============================================="