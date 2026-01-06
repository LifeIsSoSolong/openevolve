#!/usr/bin/env bash
set -euo pipefail

########################
# 参数解析
########################
INPUT_DIR=""
OUTPUT_DIR=""
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

########################
# 参数校验
########################
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$CONFIG_FILE" ]]; then
    echo "用法:"
    echo "  bash start_training.sh \\"
    echo "    --input-dir <INPUT_DIR> \\"
    echo "    --output-dir <OUTPUT_DIR> \\"
    echo "    --config-file <CONFIG_FILE>"
    exit 1
fi

########################
# 路径准备
########################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/run.log"
STATUS_FILE="$OUTPUT_DIR/training.status"
PID_FILE="$OUTPUT_DIR/training.pid"

########################
# 启动训练（后台）
########################
nohup python -u "$SKILL_ROOT/main.py" \
    --config_file "$CONFIG_FILE" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "训练进程已启动，PID=$TRAIN_PID"

########################
# 保存 PID
########################
echo "$TRAIN_PID" > "$PID_FILE"

########################
# 写入状态文件
########################
if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    cat > "$STATUS_FILE" << EOF
{
  "status": "running",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "log_file": "$LOG_FILE",
  "pid": $TRAIN_PID
}
EOF
else
    echo "进程启动失败"
    exit 1
fi

echo "状态文件已写入: $STATUS_FILE"
