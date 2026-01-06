#!/usr/bin/env python3
"""
验证 judge.py 能否正确评估 agent.py。

Usage:
    python validate_judge.py --input-dir /path/to/inputs
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate judge.py for AlphaEvolve")
    parser.add_argument("--input-dir", required=True, help="Path to input directory")
    parser.add_argument("--timeout", type=int, default=300, help="Execution timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--skip-run", action="store_true", help="Skip actual execution, only check structure")
    return parser.parse_args()


def check_evaluate_function(judge_path: Path) -> Dict:
    """检查 judge.py 是否包含 evaluate 函数"""
    result = {
        "has_evaluate": False,
        "evaluate_params": [],
        "error": None,
    }
    
    try:
        source = judge_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
                result["has_evaluate"] = True
                result["evaluate_params"] = [arg.arg for arg in node.args.args]
                break
    except Exception as e:
        result["error"] = str(e)
    
    return result


def extract_agent_return_keys(agent_path: Path) -> Optional[Set[str]]:
    try:
        source = agent_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return None

    main_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_func = node
            break
    if main_func is None:
        return None

    keys: Set[str] = set()
    found = False
    for child in ast.walk(main_func):
        if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
            found = True
            for k in child.value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
                elif isinstance(k, ast.Str):
                    keys.add(k.s)
    if not found:
        return None
    return keys


def extract_judge_metric_keys(judge_path: Path) -> Optional[Set[str]]:
    try:
        source = judge_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return None

    keys: Set[str] = set()
    for node in ast.walk(tree):
        # metrics.get("key")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "metrics":
                if node.func.attr == "get" and node.args:
                    key = node.args[0]
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
                    elif isinstance(key, ast.Str):
                        keys.add(key.s)
        # metrics["key"]
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "metrics":
                slice_node = node.slice
                if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                    keys.add(slice_node.value)
                elif isinstance(slice_node, ast.Str):
                    keys.add(slice_node.s)

    return keys if keys else None


def run_judge(input_dir: Path, timeout: int) -> Dict:
    """运行 judge.py 并检查输出"""
    result = {
        "success": False,
        "combined_score": None,
        "metrics": None,
        "stdout": "",
        "stderr": "",
        "error": None,
    }
    
    judge_path = input_dir / "judge.py"
    
    try:
        # 在 input_dir 下运行 judge.py
        env = os.environ.copy()
        env["PYTHONPATH"] = str(input_dir) + ":" + env.get("PYTHONPATH", "")
        
        proc = subprocess.run(
            [sys.executable, str(judge_path)],
            cwd=str(input_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        
        if proc.returncode != 0:
            result["error"] = f"Process exited with code {proc.returncode}"
            return result
        
        # 尝试从输出中解析结果
        # judge.py 的 __main__ 通常会 print 结果
        output = proc.stdout.strip()
        
        # 尝试多种解析方式
        parsed = False
        metrics_parsed = False
        
        # 方式 1：尝试解析为 JSON
        for line in reversed(output.split("\n")):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    data = json.loads(line)
                    if "combined_score" in data:
                        result["combined_score"] = data["combined_score"]
                        result["metrics"] = data
                        parsed = True
                        break
                except json.JSONDecodeError:
                    pass
        
        # 方式 2：尝试解析 EvaluationResult 输出
        if not parsed:
            import re
            # 匹配 combined_score: 0.xxx 或 'combined_score': 0.xxx
            match = re.search(r"['\"]?combined_score['\"]?\s*[:=]\s*([0-9.]+)", output)
            if match:
                result["combined_score"] = float(match.group(1))
                parsed = True

        # 尝试解析 metrics 字典（从打印的 Evaluation results 行中提取）
        if not metrics_parsed:
            import ast as _ast
            for line in reversed(output.split("\n")):
                if "Evaluation results" in line or "Evaluation Result" in line:
                    try:
                        payload = line.split(":", 1)[-1].strip()
                        if payload.startswith("{") and payload.endswith("}"):
                            result["metrics"] = _ast.literal_eval(payload)
                            metrics_parsed = isinstance(result["metrics"], dict)
                            if metrics_parsed and result["combined_score"] is None:
                                result["combined_score"] = result["metrics"].get("combined_score")
                    except Exception:
                        pass
        
        if result["combined_score"] is not None:
            score = result["combined_score"]
            if 0 < score <= 1:
                result["success"] = True
            else:
                result["error"] = f"combined_score {score} is not in range (0, 1]"
        else:
            result["error"] = "Could not find combined_score in output"
            
    except subprocess.TimeoutExpired:
        result["error"] = f"Execution timed out after {timeout} seconds"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def detect_task_type(input_dir: Path) -> str:
    if (input_dir / "train.csv").exists() and (input_dir / "test.csv").exists():
        return "mle"
    if (input_dir / "train.jsonl").exists() and (input_dir / "test.jsonl").exists():
        return "prompt"
    return "unknown"


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    judge_path = input_dir / "judge.py"
    task_type = detect_task_type(input_dir)
    
    if not judge_path.exists():
        print(f"[ERROR] judge.py not found: {judge_path}")
        return 1
    
    result = {
        "input_dir": str(input_dir),
        "judge_path": str(judge_path),
    }
    
    # 检查函数结构
    func_check = check_evaluate_function(judge_path)
    result["structure_check"] = func_check
    
    if not func_check["has_evaluate"]:
        result["valid"] = False
        result["error"] = "evaluate() function not found in judge.py"
        
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"[ERROR] {result['error']}")
        return 2
    
    # 运行验证
    if not args.skip_run:
        print("[INFO] Running judge.py to validate evaluation...")
        run_result = run_judge(input_dir, args.timeout)
        result["run_result"] = run_result
        result["valid"] = run_result["success"]

        if result["valid"] and task_type == "mle":
            agent_keys = extract_agent_return_keys(input_dir / "agent.py")
            judge_keys = extract_judge_metric_keys(judge_path)
            if not agent_keys:
                result["valid"] = False
                result["error"] = "Unable to detect agent.py return metric keys for alignment check"
            elif not judge_keys:
                result["valid"] = False
                result["error"] = "Unable to detect judge.py metric keys used for combined_score"
            else:
                mismatch = set(judge_keys) - set(agent_keys)
                if mismatch:
                    result["valid"] = False
                    result["error"] = (
                        f"judge.py uses metric keys not returned by agent.py: {sorted(mismatch)}"
                    )
        
        if not run_result["success"]:
            result["error"] = run_result["error"]
    else:
        result["valid"] = True
        result["skipped_run"] = True
    
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"[INFO] Validating: {judge_path}")
        print(f"[INFO] Task type: {task_type}")
        print()
        
        # 结构检查结果
        if func_check["has_evaluate"]:
            print(f"[OK] evaluate() function found with params: {func_check['evaluate_params']}")
        
        # 运行结果
        if not args.skip_run:
            run_result = result.get("run_result", {})
            if run_result.get("success"):
                print(f"[OK] judge.py executed successfully")
                print(f"[OK] combined_score: {run_result['combined_score']}")
            else:
                print(f"[ERROR] Execution failed: {run_result.get('error')}")
                if run_result.get("stderr"):
                    print(f"[STDERR]\n{run_result['stderr'][:500]}")
        
        print()
        if result["valid"]:
            print("[OK] judge.py validation passed")
        else:
            print(f"[ERROR] judge.py validation failed: {result.get('error', 'Unknown error')}")
    
    return 0 if result["valid"] else 3


if __name__ == "__main__":
    sys.exit(main())
