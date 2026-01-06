#!/usr/bin/env python3
"""
停止运行中的训练任务。

Usage:
    stop_training.py --output-dir OUTPUT_DIR [--force]

Options:
    --output-dir OUTPUT_DIR    训练输出目录
    --force                    强制终止（使用 SIGKILL）
"""

import os
import sys
import json
import time
import signal
import argparse
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
        print(f"❌ 无法读取状态文件: {e}", file=sys.stderr)
        return None


def update_status_file(output_dir: str, status: str, end_time: str = None):
    """更新状态文件"""
    status_file = Path(output_dir) / "training.status"

    try:
        # 读取现有状态
        if status_file.exists():
            with open(status_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        # 更新状态
        data['status'] = status
        if end_time:
            data['end_time'] = end_time

        # 写回
        with open(status_file, 'w') as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"⚠️  无法更新状态文件: {e}", file=sys.stderr)


def check_process_running(pid: int) -> bool:
    """检查进程是否在运行"""
    if pid is None:
        return False

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_process(pid: int, force: bool = False, timeout: int = 10) -> bool:
    """
    停止进程。

    Args:
        pid: 进程 ID
        force: 是否强制终止
        timeout: 等待进程终止的超时时间（秒）

    Returns:
        True 如果成功停止，False 否则
    """
    if not check_process_running(pid):
        print(f"⚠️  进程 {pid} 已不在运行")
        return True

    try:
        if force:
            # 强制终止
            print(f"强制终止进程 {pid}...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        else:
            # 优雅终止
            print(f"发送终止信号到进程 {pid}...")
            os.kill(pid, signal.SIGTERM)

            # 等待进程终止
            print(f"等待进程终止（最多 {timeout} 秒）...")
            for i in range(timeout):
                time.sleep(1)
                if not check_process_running(pid):
                    print(f"✓ 进程已终止")
                    return True
                print(f"  等待中... ({i + 1}/{timeout})")

            # 超时后强制终止
            if check_process_running(pid):
                print(f"⚠️  超时，强制终止进程...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)

        # 最终检查
        if not check_process_running(pid):
            print(f"✓ 进程 {pid} 已成功停止")
            return True
        else:
            print(f"❌ 无法停止进程 {pid}")
            return False

    except PermissionError:
        print(f"❌ 没有权限终止进程 {pid}")
        return False
    except Exception as e:
        print(f"❌ 停止进程时出错: {e}", file=sys.stderr)
        return False


def stop_training(output_dir: str, force: bool = False):
    """停止训练任务"""
    print("\n" + "=" * 80)
    print("停止训练任务")
    print("=" * 80)

    # 读取状态文件
    status = read_status_file(output_dir)

    if status is None:
        print("❌ 未找到训练任务")
        print(f"   输出目录: {output_dir}")
        print("=" * 80 + "\n")
        sys.exit(1)

    # 获取 PID
    pid = status.get('pid')

    if pid is None:
        print("❌ 状态文件中没有进程 ID")
        print("   无法停止训练任务")
        print("=" * 80 + "\n")
        sys.exit(1)

    print(f"\n训练任务信息：")
    print(f"  进程 ID: {pid}")
    print(f"  开始时间: {status.get('start_time', 'unknown')}")
    print(f"  当前状态: {status.get('status', 'unknown')}")
    print(f"  输出目录: {output_dir}")

    # 检查进程是否在运行
    if not check_process_running(pid):
        print(f"\n⚠️  进程 {pid} 已不在运行")
        print(f"   训练任务可能已完成或异常退出")

        # 清理 PID 文件
        pid_file = Path(output_dir) / "training.pid"
        if pid_file.exists():
            pid_file.unlink()
            print(f"   已清理 PID 文件")

        # 更新状态
        update_status_file(output_dir, "stopped", datetime.utcnow().isoformat() + 'Z')
        print("=" * 80 + "\n")
        sys.exit(0)

    # 停止进程
    print()
    success = stop_process(pid, force)

    if success:
        # 清理 PID 文件
        pid_file = Path(output_dir) / "training.pid"
        if pid_file.exists():
            pid_file.unlink()

        # 更新状态文件
        update_status_file(output_dir, "stopped", datetime.utcnow().isoformat() + 'Z')

        print(f"\n✓ 训练任务已成功停止")
        print(f"   日志文件: {status.get('log_file', 'unknown')}")
    else:
        print(f"\n❌ 无法停止训练任务")
        print(f"   请尝试使用 --force 选项强制终止")
        print(f"   或手动终止进程: kill -9 {pid}")

    print("=" * 80 + "\n")

    sys.exit(0 if success else 1)


def main():
    parser = argparse.ArgumentParser(description="停止训练任务")
    parser.add_argument("--output-dir", type=str, required=True, help="训练输出目录")
    parser.add_argument("--force", action="store_true", help="强制终止进程")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(output_dir):
        print(f"❌ 输出目录不存在: {output_dir}")
        sys.exit(1)

    stop_training(output_dir, args.force)


if __name__ == "__main__":
    main()
