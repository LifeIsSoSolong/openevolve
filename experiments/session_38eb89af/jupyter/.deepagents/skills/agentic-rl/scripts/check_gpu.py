#!/usr/bin/env python3
"""
检查 GPU 可用性和资源情况。

Usage:
    check_gpu.py [--required N]

Options:
    --required N    需要的 GPU 数量（默认：4）
"""

import subprocess
import sys
import argparse
from typing import List, Tuple


def get_gpu_info() -> List[Tuple[int, str, int, int, int]]:
    """
    获取 GPU 信息。

    Returns:
        List of (index, name, memory_used_MB, memory_total_MB, utilization_percent)
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                index = int(parts[0])
                name = parts[1]
                mem_used = int(parts[2])
                mem_total = int(parts[3])
                util = int(parts[4])
                gpu_info.append((index, name, mem_used, mem_total, util))

        return gpu_info

    except subprocess.CalledProcessError as e:
        print(f"❌ 无法获取 GPU 信息: {e}", file=sys.stderr)
        print("   请确保 nvidia-smi 可用且系统有 NVIDIA GPU", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 解析 GPU 信息时出错: {e}", file=sys.stderr)
        sys.exit(1)


def is_gpu_available(mem_used: int, mem_total: int, util: int) -> bool:
    """
    判断 GPU 是否可用。

    标准：利用率 < 20% 且内存使用 < 10%
    """
    mem_usage_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 100
    return util < 20 and mem_usage_percent < 10


def print_gpu_status(gpu_info: List[Tuple[int, str, int, int, int]]):
    """打印 GPU 状态表格"""
    print("\n" + "=" * 100)
    print(f"{'GPU':<5} {'名称':<30} {'内存使用':<25} {'利用率':<10} {'状态':<10}")
    print("=" * 100)

    for index, name, mem_used, mem_total, util in gpu_info:
        mem_usage_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
        mem_str = f"{mem_used:>6} MB / {mem_total:>6} MB ({mem_usage_percent:>5.1f}%)"
        util_str = f"{util:>3}%"

        available = is_gpu_available(mem_used, mem_total, util)
        status = "✓ 可用" if available else "✗ 占用"

        print(f"{index:<5} {name:<30} {mem_str:<25} {util_str:<10} {status:<10}")

    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="检查 GPU 可用性")
    parser.add_argument("--required", type=int, default=4, help="需要的 GPU 数量")
    args = parser.parse_args()

    required_gpus = args.required

    print(f"正在检查 GPU 资源...")
    gpu_info = get_gpu_info()

    if not gpu_info:
        print("❌ 未检测到任何 GPU")
        sys.exit(1)

    total_gpus = len(gpu_info)
    available_gpus = sum(1 for _, _, mem_used, mem_total, util in gpu_info
                         if is_gpu_available(mem_used, mem_total, util))

    print_gpu_status(gpu_info)

    print(f"GPU 总数: {total_gpus}")
    print(f"可用 GPU: {available_gpus}")
    print(f"需要 GPU: {required_gpus}")
    print()

    if available_gpus >= required_gpus:
        print(f"✓ GPU 资源充足！可以启动训练。")
        print(f"  可用 {available_gpus} 个 GPU，满足 {required_gpus} 个 GPU 的需求。")
        sys.exit(0)
    else:
        print(f"❌ GPU 资源不足！无法启动训练。")
        print(f"  需要 {required_gpus} 个 GPU，但只有 {available_gpus} 个可用。")
        print()
        print("建议操作：")
        print(f"  1. 等待其他任务释放 GPU")
        print(f"  2. 修改 config.json 中的 trainer.n_gpus_per_node 为 {available_gpus} 或更小")
        print(f"  3. 终止占用 GPU 的其他进程")
        sys.exit(1)


if __name__ == "__main__":
    main()
