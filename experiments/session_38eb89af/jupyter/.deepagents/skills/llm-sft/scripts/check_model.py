#!/usr/bin/env python3
"""
检查模型是否存在，并列出所有可用模型。

Usage:
    check_model.py [--model MODEL_NAME] [--models-dir PATH]

Options:
    --model MODEL_NAME       要检查的模型名称
    --models-dir PATH        模型目录路径（默认：/hpc_data/ktian/models）
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List


def get_available_models(models_dir: str) -> List[str]:
    """
    获取所有可用的模型列表。

    Args:
        models_dir: 模型目录路径

    Returns:
        模型名称列表
    """
    models_path = Path(models_dir)

    if not models_path.exists():
        print(f"❌ 模型目录不存在: {models_dir}", file=sys.stderr)
        return []

    if not models_path.is_dir():
        print(f"❌ 路径不是目录: {models_dir}", file=sys.stderr)
        return []

    # 获取所有子目录作为模型
    models = []
    try:
        for item in models_path.iterdir():
            if item.is_dir():
                models.append(item.name)
    except PermissionError:
        print(f"❌ 没有权限访问目录: {models_dir}", file=sys.stderr)
        return []

    return sorted(models)


def check_model_exists(model_name: str, models_dir: str) -> bool:
    """
    检查模型是否存在。

    Args:
        model_name: 模型名称
        models_dir: 模型目录路径

    Returns:
        True 如果模型存在，否则 False
    """
    model_path = Path(models_dir) / model_name
    return model_path.exists() and model_path.is_dir()


def print_available_models(models: List[str]):
    """打印可用模型列表"""
    if not models:
        print("\n没有找到可用的模型。")
        return

    print("\n" + "=" * 80)
    print("可用模型列表：")
    print("=" * 80)
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="检查模型可用性")
    parser.add_argument("--model", type=str, help="要检查的模型名称")
    parser.add_argument("--models-dir", type=str, default="/hpc_data/ktian/models",
                       help="模型目录路径")
    args = parser.parse_args()

    models_dir = args.models_dir
    model_name = args.model

    print(f"正在扫描模型目录: {models_dir}")
    available_models = get_available_models(models_dir)

    if not available_models:
        print("❌ 未找到任何可用模型")
        sys.exit(1)

    print(f"找到 {len(available_models)} 个可用模型")
    print_available_models(available_models)

    # 如果指定了模型名称，检查是否存在
    if model_name:
        if check_model_exists(model_name, models_dir):
            print(f"✓ 模型存在: {model_name}")
            print(f"  路径: {models_dir}/{model_name}")
            sys.exit(0)
        else:
            print(f"❌ 模型不存在: {model_name}")
            print(f"\n请从上面的可用模型列表中选择一个模型。")
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
