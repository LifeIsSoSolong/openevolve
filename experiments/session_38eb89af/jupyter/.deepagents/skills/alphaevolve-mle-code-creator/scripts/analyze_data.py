#!/usr/bin/env python3
"""
数据探索分析脚本

分析输入目录中的数据文件，输出数据概况和任务类型推断。
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def load_single_file(path: Path) -> pd.DataFrame:
    """根据文件后缀加载数据"""
    suffix = path.suffix.lower()
    
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".feather":
        return pd.read_feather(path)
    elif suffix == ".json":
        return pd.read_json(path)
    elif suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    elif suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def find_data_files(input_dir: Path) -> dict:
    """扫描目录，找出所有数据文件"""
    supported_extensions = {".csv", ".xlsx", ".xls", ".parquet", ".feather", ".json", ".jsonl", ".tsv"}
    
    data_files = []
    for f in input_dir.iterdir():
        if f.is_file() and f.suffix.lower() in supported_extensions:
            data_files.append(f)
    
    # 尝试识别训练/测试文件
    train_file = None
    test_file = None
    other_files = []
    
    train_keywords = ["train", "training", "trn"]
    test_keywords = ["test", "testing", "tst", "val", "valid", "validation", "eval"]
    
    for f in data_files:
        name_lower = f.stem.lower()
        if any(kw in name_lower for kw in train_keywords):
            train_file = f
        elif any(kw in name_lower for kw in test_keywords):
            test_file = f
        else:
            other_files.append(f)
    
    return {
        "train": train_file,
        "test": test_file,
        "other": other_files,
        "all": data_files,
    }


def load_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """加载训练和测试数据，返回 (train_df, test_df, file_info)"""
    file_info = find_data_files(input_dir)
    
    train_df = None
    test_df = None
    
    if file_info["train"]:
        train_df = load_single_file(file_info["train"])
    
    if file_info["test"]:
        test_df = load_single_file(file_info["test"])
    
    # 如果没有明确的 train/test，但有其他文件
    if train_df is None and file_info["other"]:
        # 使用第一个文件作为主数据
        train_df = load_single_file(file_info["other"][0])
        file_info["train"] = file_info["other"][0]
    
    return train_df, test_df, file_info


def infer_target_column(df: pd.DataFrame) -> str:
    """推断目标列"""
    # 常见目标列名
    common_names = ["target", "label", "y", "class", "outcome", "output", 
                    "yield", "price", "value", "churn", "default"]
    
    for name in common_names:
        if name in df.columns:
            return name
        # 不区分大小写
        for col in df.columns:
            if col.lower() == name:
                return col
    
    # 默认返回最后一列
    return df.columns[-1]


def infer_task_type(df: pd.DataFrame, target_col: str) -> str:
    """推断任务类型"""
    target = df[target_col]
    
    # 检查唯一值数量
    n_unique = target.nunique()
    n_samples = len(target)
    
    # 如果是字符串类型，通常是分类
    if target.dtype == object:
        return "classification"
    
    # 如果唯一值很少（相对于样本量），可能是分类
    if n_unique <= 10 and n_unique / n_samples < 0.05:
        return "classification"
    
    # 否则是回归
    return "regression"


def analyze_columns(df: pd.DataFrame, target_col: str) -> dict:
    """分析列信息"""
    columns_info = []
    
    for col in df.columns:
        info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "n_missing": int(df[col].isna().sum()),
            "missing_pct": float(df[col].isna().mean() * 100),
            "is_target": col == target_col,
        }
        
        # 数值列统计
        if pd.api.types.is_numeric_dtype(df[col]):
            info["type"] = "numeric"
            info["min"] = float(df[col].min()) if not df[col].isna().all() else None
            info["max"] = float(df[col].max()) if not df[col].isna().all() else None
            info["mean"] = float(df[col].mean()) if not df[col].isna().all() else None
            info["std"] = float(df[col].std()) if not df[col].isna().all() else None
        else:
            info["type"] = "categorical"
            info["top_values"] = df[col].value_counts().head(5).to_dict()
        
        columns_info.append(info)
    
    return columns_info


def main():
    parser = argparse.ArgumentParser(description="数据探索分析")
    parser.add_argument("--input-dir", required=True, help="输入目录")
    parser.add_argument("--output-json", help="输出 JSON 文件路径")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # 加载数据
    train_df, test_df, file_info = load_data(input_dir)
    
    if train_df is None:
        print("[ERROR] 未找到数据文件")
        print(f"支持的格式: csv, xlsx, xls, parquet, feather, json, jsonl, tsv")
        sys.exit(1)
    
    # 推断目标列和任务类型
    target_col = infer_target_column(train_df)
    task_type = infer_task_type(train_df, target_col)
    
    # 分析结果
    result = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape) if test_df is not None else None,
        "target_column": target_col,
        "task_type": task_type,
        "n_features": len(train_df.columns) - 1,
        "columns": analyze_columns(train_df, target_col),
        "files": {
            "train": str(file_info["train"]) if file_info["train"] else None,
            "test": str(file_info["test"]) if file_info["test"] else None,
            "other": [str(f) for f in file_info["other"]],
        }
    }
    
    # 目标列分析
    target_info = {
        "name": target_col,
        "dtype": str(train_df[target_col].dtype),
        "n_unique": int(train_df[target_col].nunique()),
        "n_missing": int(train_df[target_col].isna().sum()),
    }
    
    if task_type == "classification":
        target_info["class_distribution"] = train_df[target_col].value_counts().to_dict()
    else:
        target_info["statistics"] = {
            "min": float(train_df[target_col].min()),
            "max": float(train_df[target_col].max()),
            "mean": float(train_df[target_col].mean()),
            "std": float(train_df[target_col].std()),
        }
    
    result["target_info"] = target_info
    
    # 输出
    print("=" * 60)
    print("数据分析报告")
    print("=" * 60)
    
    print(f"\n📁 发现的数据文件:")
    if file_info["train"]:
        print(f"   训练数据: {file_info['train'].name}")
    if file_info["test"]:
        print(f"   测试数据: {file_info['test'].name}")
    if file_info["other"]:
        print(f"   其他文件: {[f.name for f in file_info['other']]}")
    
    print(f"\n📊 数据概况:")
    print(f"   训练集: {result['train_shape'][0]} 行 × {result['train_shape'][1]} 列")
    if result["test_shape"]:
        print(f"   测试集: {result['test_shape'][0]} 行 × {result['test_shape'][1]} 列")
    else:
        print(f"   测试集: [未找到，可能需要从训练集划分]")
    print(f"   特征数: {result['n_features']}")
    
    print(f"\n🎯 目标列: {target_col}")
    print(f"   任务类型: {'分类' if task_type == 'classification' else '回归'}")
    
    if task_type == "classification":
        print(f"   类别分布:")
        for cls, cnt in target_info["class_distribution"].items():
            print(f"      {cls}: {cnt} ({cnt/train_df.shape[0]*100:.1f}%)")
    else:
        stats = target_info["statistics"]
        print(f"   范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"   均值: {stats['mean']:.4f}, 标准差: {stats['std']:.4f}")
    
    print(f"\n📋 列信息:")
    n_numeric = sum(1 for c in result["columns"] if c["type"] == "numeric")
    n_categorical = sum(1 for c in result["columns"] if c["type"] == "categorical")
    print(f"   数值列: {n_numeric}, 类别列: {n_categorical}")
    
    # 缺失值
    cols_with_missing = [c for c in result["columns"] if c["n_missing"] > 0]
    if cols_with_missing:
        print(f"\n⚠️ 缺失值:")
        for c in cols_with_missing:
            print(f"   {c['name']}: {c['n_missing']} ({c['missing_pct']:.1f}%)")
    else:
        print(f"\n✅ 无缺失值")
    
    print("\n" + "=" * 60)
    
    # 保存 JSON
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n分析结果已保存至: {args.output_json}")
    
    # 打印机器可读的摘要
    print(f"\n[SUMMARY]")
    print(f"target_column={target_col}")
    print(f"task_type={task_type}")
    print(f"n_features={result['n_features']}")
    print(f"train_file={file_info['train'].name if file_info['train'] else 'None'}")
    print(f"test_file={file_info['test'].name if file_info['test'] else 'None'}")


if __name__ == "__main__":
    main()
