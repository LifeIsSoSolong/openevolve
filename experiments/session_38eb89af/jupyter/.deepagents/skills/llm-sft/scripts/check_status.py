#!/usr/bin/env python3
"""
检查训练任务的状态。

Usage:
    check_status.py --output-dir OUTPUT_DIR [--tail N]

Options:
    --output-dir OUTPUT_DIR    训练输出目录
    --tail N                   显示日志最后 N 行（默认：20）
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def read_status_file(output_dir: str) -> dict:
    """读取状态文件"""
    status_file = Path(output_dir) / "training.status"

    if not status_file.exists():
        return None

    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  无法读取状态文件: {e}", file=sys.stderr)
        return None


def check_process_running(pid: int) -> bool:
    """检查进程是否在运行"""
    if pid is None:
        return False

    try:
        # 使用 kill -0 检查进程是否存在（不发送信号）
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_process_info(pid: int) -> dict:
    """获取进程信息（CPU、内存使用）"""
    if not check_process_running(pid):
        return None

    try:
        cmd = ["ps", "-p", str(pid), "-o", "pid,pcpu,pmem,etime,comm"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')

        if len(lines) >= 2:
            # 解析输出
            parts = lines[1].split()
            if len(parts) >= 5:
                return {
                    "pid": int(parts[0]),
                    "cpu_percent": float(parts[1]),
                    "mem_percent": float(parts[2]),
                    "elapsed_time": parts[3],
                    "command": " ".join(parts[4:])
                }
    except Exception as e:
        print(f"⚠️  无法获取进程信息: {e}", file=sys.stderr)

    return None


def get_gpu_usage() -> list:
    """获取 GPU 使用情况"""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpus.append({
                    "index": int(parts[0]),
                    "utilization": int(parts[1]),
                    "mem_used": int(parts[2]),
                    "mem_total": int(parts[3])
                })
        return gpus
    except Exception:
        return []


def tail_log_file(log_file: str, n: int = 20):
    """显示日志文件的最后 N 行"""
    log_path = Path(log_file)

    if not log_path.exists():
        print(f"⚠️  日志文件不存在: {log_file}")
        return

    try:
        cmd = ["tail", f"-{n}", str(log_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        print(f"⚠️  无法读取日志文件: {e}", file=sys.stderr)
        return None


def format_duration(start_time_str: str) -> str:
    """计算运行时长"""
    try:
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        now = datetime.now(start_time.tzinfo)
        duration = now - start_time

        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}小时 {minutes}分钟"
        elif minutes > 0:
            return f"{minutes}分钟 {seconds}秒"
        else:
            return f"{seconds}秒"
    except Exception:
        return "未知"


def print_status(output_dir: str, tail_lines: int = 20):
    """打印训练状态"""
    status = read_status_file(output_dir)

    print("\n" + "=" * 80)
    print("训练任务状态")
    print("=" * 80)

    if status is None:
        print("❌ 未找到训练任务")
        print(f"   输出目录: {output_dir}")
        print(f"   状态文件: {output_dir}/training.status")
        print("\n提示: 该目录可能还没有启动过训练任务")
        print("=" * 80 + "\n")
        return

    # 基本信息
    print(f"\n状态: {status.get('status', 'unknown')}")
    print(f"开始时间: {status.get('start_time', 'unknown')}")

    if 'start_time' in status:
        duration = format_duration(status['start_time'])
        print(f"运行时长: {duration}")

    print(f"日志文件: {status.get('log_file', 'unknown')}")

    # 进程信息
    pid = status.get('pid')
    if pid:
        print(f"\n进程 ID: {pid}")

        if check_process_running(pid):
            print(f"进程状态: ✓ 运行中")

            proc_info = get_process_info(pid)
            if proc_info:
                print(f"CPU 使用: {proc_info['cpu_percent']}%")
                print(f"内存使用: {proc_info['mem_percent']}%")
                print(f"运行时间: {proc_info['elapsed_time']}")
        else:
            print(f"进程状态: ✗ 已停止")
            print(f"\n⚠️  进程已不在运行，但状态文件显示为 '{status.get('status')}'")
            print(f"   可能训练已完成或异常退出，请检查日志")

    # GPU 使用情况
    gpus = get_gpu_usage()
    if gpus:
        print("\n" + "-" * 80)
        print("GPU 使用情况")
        print("-" * 80)
        print(f"{'GPU':<5} {'利用率':<10} {'内存使用':<30}")
        print("-" * 80)
        for gpu in gpus:
            mem_percent = (gpu['mem_used'] / gpu['mem_total'] * 100) if gpu['mem_total'] > 0 else 0
            mem_str = f"{gpu['mem_used']:>6} MB / {gpu['mem_total']:>6} MB ({mem_percent:>5.1f}%)"
            print(f"{gpu['index']:<5} {gpu['utilization']:>3}%{'':<6} {mem_str:<30}")
        print("-" * 80)

    # 日志内容
    log_file = status.get('log_file')
    if log_file and Path(log_file).exists():
        print("\n" + "-" * 80)
        print(f"最近日志（最后 {tail_lines} 行）")
        print("-" * 80)
        log_content = tail_log_file(log_file, tail_lines)
        if log_content:
            print(log_content)
        print("-" * 80)
        print(f"\n查看完整日志: tail -f {log_file}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="检查训练任务状态")
    parser.add_argument("--output-dir", type=str, required=True, help="训练输出目录")
    parser.add_argument("--tail", type=int, default=20, help="显示日志最后 N 行")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(output_dir):
        print(f"❌ 输出目录不存在: {output_dir}")
        sys.exit(1)

    print_status(output_dir, args.tail)


if __name__ == "__main__":
    main()
