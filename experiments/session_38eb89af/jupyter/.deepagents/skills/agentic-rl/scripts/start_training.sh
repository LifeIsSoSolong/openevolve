#!/bin/bash
#
# 在后台启动 RL 训练任务
#
# Usage:
#   start_training.sh --input-dir INPUT_DIR --output-dir OUTPUT_DIR --config-file CONFIG_FILE
#

set -e

# 解析参数
INPUT_DIR=""
OUTPUT_DIR=""
CONFIG_FILE=""
SKILL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
    case $1 in
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
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 验证参数
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 --input-dir INPUT_DIR --output-dir OUTPUT_DIR --config-file CONFIG_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

# 定义日志和状态文件
LOG_FILE="$OUTPUT_DIR/training.log"
PID_FILE="$OUTPUT_DIR/training.pid"
STATUS_FILE="$OUTPUT_DIR/training.status"

# 检查是否已有训练在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "❌ 训练任务已在运行 (PID: $OLD_PID)"
        echo "   请先停止现有任务或等待其完成"
        echo "   使用以下命令检查状态："
        echo "   python $SKILL_ROOT/scripts/check_status.py --output-dir $OUTPUT_DIR"
        exit 1
    else
        echo "⚠️  清理旧的 PID 文件 (进程 $OLD_PID 已不存在)"
        rm -f "$PID_FILE"
    fi
fi

# 创建状态文件
cat > "$STATUS_FILE" << EOF
{
  "status": "starting",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "input_dir": "$INPUT_DIR",
  "output_dir": "$OUTPUT_DIR",
  "config_file": "$CONFIG_FILE",
  "log_file": "$LOG_FILE",
  "pid": null
}
EOF

echo "========================================"
echo "启动 RL 训练任务"
echo "========================================"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "配置文件: $CONFIG_FILE"
echo "日志文件: $LOG_FILE"
echo "状态文件: $STATUS_FILE"
echo "========================================"

# 设置环境变量
export VERL_FILE_LOGGER_PATH="$OUTPUT_DIR"

# 构建训练命令
TRAIN_CMD="cd $SKILL_ROOT && python scripts/main.py \
    --config_file $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --input_dir $INPUT_DIR"

# 在后台启动训练
echo "正在后台启动训练..."
nohup bash -c "$TRAIN_CMD" > "$LOG_FILE" 2>&1 & echo $!
TRAIN_PID=$!

# 保存 PID
echo "$TRAIN_PID" > "$PID_FILE"

# 等待一小段时间确保进程启动
sleep 2

# 验证进程是否成功启动
if ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    # 更新状态文件
    cat > "$STATUS_FILE" << EOF
{
  "status": "running",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "input_dir": "$INPUT_DIR",
  "output_dir": "$OUTPUT_DIR",
  "config_file": "$CONFIG_FILE",
  "log_file": "$LOG_FILE",
  "pid": $TRAIN_PID
}
EOF

    echo ""
    echo "✓ 训练任务已成功启动！"
    echo ""
    echo "进程 ID: $TRAIN_PID"
    echo "日志文件: $LOG_FILE"
    echo ""
    echo "查看实时日志："
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "检查训练状态："
    echo "  python $SKILL_ROOT/scripts/check_status.py --output-dir $OUTPUT_DIR"
    echo ""
    echo "停止训练："
    echo "  python $SKILL_ROOT/scripts/stop_training.py --output-dir $OUTPUT_DIR"
    echo ""
else
    # 更新状态为失败
    cat > "$STATUS_FILE" << EOF
{
  "status": "failed",
  "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "error": "Process failed to start",
  "input_dir": "$INPUT_DIR",
  "output_dir": "$OUTPUT_DIR",
  "config_file": "$CONFIG_FILE",
  "log_file": "$LOG_FILE",
  "pid": null
}
EOF

    echo "❌ 训练启动失败"
    echo "请检查日志文件: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
