#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


REQUIRED_FILES = [
    "all_results.json",
    "run.log",
    "README.md",
]

CHECKPOINT_PREFIX = "checkpoint-"


def check_output_dir(output_dir: Path) -> int:
    """
    检查 output_dir 中的产出是否完整
    返回：
        0  -> 全部通过
        1  -> 有缺失
    """
    errors = []

    if not output_dir.exists():
        print(f"[ERROR] output-dir 不存在: {output_dir}")
        return 1

    if not output_dir.is_dir():
        print(f"[ERROR] output-dir 不是目录: {output_dir}")
        return 1

    print(f"[INFO] 检查 output-dir: {output_dir}")

    # 1. 检查必须的文件
    for filename in REQUIRED_FILES:
        path = output_dir / filename
        if not path.exists():
            errors.append(f"缺少文件: {filename}")
        elif not path.is_file():
            errors.append(f"不是文件: {filename}")
        else:
            print(f"[OK] 文件存在: {filename}")

    # 2. 检查 checkpoint-* 目录
    checkpoint_dirs = [
        p for p in output_dir.iterdir()
        if p.is_dir() and p.name.startswith(CHECKPOINT_PREFIX)
    ]

    if not checkpoint_dirs:
        errors.append("缺少 checkpoint-* 目录")
    else:
        print(f"[OK] 找到 checkpoint 目录 ({len(checkpoint_dirs)} 个):")
        for p in sorted(checkpoint_dirs):
            print(f"      - {p.name}")

    # 3. 汇总结果
    if errors:
        print("\n[FAILED] 产出检查未通过:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\n[SUCCESS] 产出检查全部通过 ✔")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="检查 output-dir 中的训练 / 运行产出是否完整"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录路径"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    exit_code = check_output_dir(output_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
