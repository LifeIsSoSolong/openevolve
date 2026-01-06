#!/usr/bin/env python3
"""
检查输入目录完整性并判断任务类型。
支持多种数据格式：csv, xlsx, xls, parquet, feather, json, jsonl, tsv

Usage:
    python check_inputs.py --input-dir /path/to/inputs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# 支持的数据文件格式
TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".feather", ".tsv"}
STRUCTURED_EXTENSIONS = {".json", ".jsonl"}
ALL_DATA_EXTENSIONS = TABULAR_EXTENSIONS | STRUCTURED_EXTENSIONS

# 文件要求定义（agent.py 是核心必须文件，数据文件动态检测）
MLE_CORE_REQUIRED = ["agent.py"]
MLE_OPTIONAL = ["judge.py", "config.json", "task.goal"]

PROMPT_CORE_REQUIRED = [
    "agent.py",
    "generate_press_agent.py",
    "evaluate_press_agent.py",
]
PROMPT_OPTIONAL = ["judge.py", "config.json", "task.goal"]

# 识别训练/测试文件的关键词
TRAIN_KEYWORDS = ["train", "training", "trn"]
TEST_KEYWORDS = ["test", "testing", "tst", "val", "valid", "validation", "eval"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check input directory for AlphaEvolve")
    parser.add_argument("--input-dir", required=True, help="Path to input directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    return parser.parse_args()


def find_data_files(input_dir: Path) -> Dict:
    """扫描目录，找出所有数据文件并尝试识别训练/测试文件"""
    data_files = []
    for f in input_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ALL_DATA_EXTENSIONS:
            data_files.append(f)
    
    train_file = None
    test_file = None
    other_files = []
    
    for f in data_files:
        name_lower = f.stem.lower()
        if any(kw in name_lower for kw in TRAIN_KEYWORDS):
            if train_file is None:
                train_file = f
            else:
                other_files.append(f)
        elif any(kw in name_lower for kw in TEST_KEYWORDS):
            if test_file is None:
                test_file = f
            else:
                other_files.append(f)
        else:
            other_files.append(f)
    
    # 判断数据格式类型
    format_type = None
    if data_files:
        tabular_count = sum(1 for f in data_files if f.suffix.lower() in TABULAR_EXTENSIONS)
        structured_count = sum(1 for f in data_files if f.suffix.lower() in STRUCTURED_EXTENSIONS)
        
        if tabular_count > 0 and structured_count == 0:
            format_type = "tabular"
        elif structured_count > 0 and tabular_count == 0:
            format_type = "structured"
        elif tabular_count > 0 and structured_count > 0:
            format_type = "mixed"
    
    return {
        "train": train_file,
        "test": test_file,
        "other": other_files,
        "all": data_files,
        "format": format_type,
    }


def check_core_files(input_dir: Path, required: List[str], optional: List[str]) -> Dict[str, List[str]]:
    """检查核心文件存在性"""
    result = {"present": [], "missing_required": [], "missing_optional": []}
    
    for f in required:
        if (input_dir / f).exists():
            result["present"].append(f)
        else:
            result["missing_required"].append(f)
    
    for f in optional:
        if (input_dir / f).exists():
            result["present"].append(f)
        else:
            result["missing_optional"].append(f)
    
    return result


def detect_task_type(input_dir: Path) -> Tuple[str, Dict]:
    """检测任务类型"""
    data_info = find_data_files(input_dir)
    
    has_press_agents = (
        (input_dir / "generate_press_agent.py").exists() and
        (input_dir / "evaluate_press_agent.py").exists()
    )
    
    has_jsonl = any(f.suffix.lower() == ".jsonl" for f in data_info["all"])
    
    # Prompt 任务
    if has_press_agents or (has_jsonl and data_info["format"] == "structured"):
        core_check = check_core_files(input_dir, PROMPT_CORE_REQUIRED, PROMPT_OPTIONAL)
        return "prompt", {**core_check, "data_files": data_info}
    
    # MLE 任务
    if data_info["all"] and data_info["format"] in ["tabular", "mixed", None]:
        core_check = check_core_files(input_dir, MLE_CORE_REQUIRED, MLE_OPTIONAL)
        return "mle", {**core_check, "data_files": data_info}
    
    return "unknown", {
        "reason": "No data files found",
        "files_found": [f.name for f in input_dir.iterdir() if f.is_file()],
    }


def validate_data_file(file_path: Path) -> Dict:
    """验证数据文件"""
    result = {"valid": False, "rows": 0, "columns": None, "error": None}
    
    try:
        suffix = file_path.suffix.lower()
        
        if suffix == ".csv":
            import csv
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                rows = sum(1 for _ in reader)
            result.update({"valid": True, "rows": rows, "columns": headers})
                
        elif suffix == ".tsv":
            import csv
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                headers = next(reader, None)
                rows = sum(1 for _ in reader)
            result.update({"valid": True, "rows": rows, "columns": headers})
                
        elif suffix in [".xlsx", ".xls"]:
            import pandas as pd
            df = pd.read_excel(file_path)
            result.update({"valid": True, "rows": len(df), "columns": list(df.columns)})
                
        elif suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(file_path)
            result.update({"valid": True, "rows": len(df), "columns": list(df.columns)})
                
        elif suffix == ".feather":
            import pandas as pd
            df = pd.read_feather(file_path)
            result.update({"valid": True, "rows": len(df), "columns": list(df.columns)})
                
        elif suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                result["valid"] = True
                result["rows"] = len(data)
                if data and isinstance(data[0], dict):
                    result["columns"] = list(data[0].keys())
                    
        elif suffix == ".jsonl":
            rows = 0
            columns = None
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if columns is None and isinstance(obj, dict):
                            columns = list(obj.keys())
                        rows += 1
            result.update({"valid": True, "rows": rows, "columns": columns})
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    
    if not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return 1
    
    task_type, details = detect_task_type(input_dir)
    
    # 验证数据文件
    data_validation = {}
    if "data_files" in details:
        for f in details["data_files"]["all"]:
            data_validation[f.name] = validate_data_file(f)
    
    if args.json:
        if "data_files" in details:
            df = details["data_files"]
            details["data_files"] = {
                "train": str(df["train"]) if df["train"] else None,
                "test": str(df["test"]) if df["test"] else None,
                "other": [str(f) for f in df["other"]],
                "all": [str(f) for f in df["all"]],
                "format": df["format"],
            }
        print(json.dumps({"input_dir": str(input_dir), "task_type": task_type, 
                         "details": details, "data_validation": data_validation}, indent=2))
    else:
        print(f"[INFO] Input directory: {input_dir}")
        print(f"[INFO] Detected task type: {task_type}")
        print()
        
        if task_type in ["mle", "prompt"]:
            if details.get("missing_required"):
                print("[ERROR] Missing required files:")
                for f in details["missing_required"]:
                    print(f"  - {f}")
            else:
                print("[OK] All required core files present")
            
            if details.get("missing_optional"):
                print("[INFO] Missing optional files (can be co-created):")
                for f in details["missing_optional"]:
                    print(f"  - {f}")
            
            print()
            print("[INFO] Data files:")
            df = details["data_files"]
            print(f"  - Train: {df['train'].name if df['train'] else '[未识别]'}")
            print(f"  - Test: {df['test'].name if df['test'] else '[未识别]'}")
            if df["other"]:
                print(f"  - Other: {[f.name for f in df['other']]}")
            print(f"  - Format: {df['format']}")
            
            print()
            print("[INFO] Data file validation:")
            for fname, v in data_validation.items():
                if v["valid"]:
                    col_info = f", {len(v['columns'])} cols" if v["columns"] else ""
                    print(f"  - {fname}: OK ({v['rows']} rows{col_info})")
                else:
                    print(f"  - {fname}: ERROR - {v['error']}")
        else:
            print(f"[WARN] {details.get('reason', 'Unknown')}")
    
    if task_type == "unknown":
        return 2
    if details.get("missing_required"):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
