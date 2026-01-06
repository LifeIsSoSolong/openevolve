#!/usr/bin/env python3
"""
校验 agent.py 是否符合 AlphaEvolve 规范。

Usage:
    python validate_agent.py --input-dir /path/to/inputs --task-type mle
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


FORBIDDEN_PATH_PATTERNS = [
    "__file__",
    "Path(__file__)",
    "os.path.dirname(__file__)",
    "os.getcwd(",
    "Path.cwd(",
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate agent.py for AlphaEvolve")
    parser.add_argument("--input-dir", required=True, help="Path to input directory")
    parser.add_argument("--task-type", required=True, choices=["mle", "prompt"], help="Task type")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    return parser.parse_args()


def parse_agent_file(file_path: Path) -> Tuple[Optional[ast.Module], Optional[str]]:
    """解析 agent.py 文件"""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        return tree, source
    except SyntaxError as e:
        return None, f"Syntax error: {e}"
    except Exception as e:
        return None, str(e)


def find_functions(tree: ast.Module) -> Dict[str, ast.FunctionDef]:
    """查找所有顶层函数定义"""
    functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
    return functions


def check_function_signature(func: ast.FunctionDef, expected_params: List[str]) -> Dict:
    """检查函数签名"""
    actual_params = [arg.arg for arg in func.args.args]
    
    result = {
        "name": func.name,
        "params": actual_params,
        "has_expected_params": all(p in actual_params for p in expected_params),
        "lineno": func.lineno,
    }
    return result


def check_evolve_block(source: str) -> Dict:
    """检查 EVOLVE-BLOCK 标记"""
    has_start = "# EVOLVE-BLOCK-START" in source or "#EVOLVE-BLOCK-START" in source
    has_end = "# EVOLVE-BLOCK-END" in source or "#EVOLVE-BLOCK-END" in source
    
    return {
        "has_start": has_start,
        "has_end": has_end,
        "valid": has_start and has_end,
    }


def find_forbidden_path_usage(source: str) -> List[str]:
    hits = [pattern for pattern in FORBIDDEN_PATH_PATTERNS if pattern in source]
    return hits


def extract_main_return_keys(tree: ast.Module) -> Optional[Set[str]]:
    """Extract dict keys from return statements inside main()."""
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


def main_uses_root(tree: ast.Module) -> bool:
    """Check whether main() actually references the root parameter."""
    main_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_func = node
            break
    if main_func is None:
        return False
    for child in ast.walk(main_func):
        if isinstance(child, ast.Name) and child.id == "root" and isinstance(child.ctx, ast.Load):
            return True
    return False


def validate_mle_agent(tree: ast.Module, source: str) -> Dict:
    """验证 MLE 任务的 agent.py"""
    result = {
        "valid": False,
        "has_main_root": False,
        "has_evolve_block": False,
        "functions": [],
        "errors": [],
        "suggestions": [],
    }
    
    functions = find_functions(tree)
    result["functions"] = list(functions.keys())
    
    # 检查 main(root) 函数
    if "main" in functions:
        main_check = check_function_signature(functions["main"], ["root"])
        result["has_main_root"] = main_check["has_expected_params"]
        if not result["has_main_root"]:
            result["errors"].append(f"main() function found but missing 'root' parameter. Current params: {main_check['params']}")
            result["suggestions"].append("Add 'root' parameter to main() function")
        elif not main_uses_root(tree):
            result["errors"].append("main(root) does not use 'root' to build data paths")
            result["suggestions"].append("Ensure all data paths are built from root (e.g., Path(root) / 'train.csv')")
    else:
        result["errors"].append("main(root) function not found")
        result["suggestions"].append("Add main(root) function as entry point")
    
    # 检查 EVOLVE-BLOCK
    evolve_check = check_evolve_block(source)
    result["has_evolve_block"] = evolve_check["valid"]
    if not evolve_check["valid"]:
        if not evolve_check["has_start"]:
            result["errors"].append("Missing # EVOLVE-BLOCK-START marker")
        if not evolve_check["has_end"]:
            result["errors"].append("Missing # EVOLVE-BLOCK-END marker")
        result["suggestions"].append("Add EVOLVE-BLOCK markers around the code to be evolved")

    forbidden_hits = find_forbidden_path_usage(source)
    if forbidden_hits:
        result["errors"].append(
            "Forbidden path usage detected: " + ", ".join(forbidden_hits)
        )
        result["suggestions"].append(
            "Remove __file__/cwd-based paths and rebuild data paths using root"
        )

    return_keys = extract_main_return_keys(tree)
    if return_keys is None:
        result["errors"].append("Unable to verify main() return metrics dict (no literal dict return found)")
        result["suggestions"].append("Return a dict literal with metric keys from main(root)")
    elif not return_keys:
        result["errors"].append("main() return dict has no keys")
        result["suggestions"].append("Return at least one metric key from main(root)")
    else:
        result["return_keys"] = sorted(return_keys)
    
    result["valid"] = (
        result["has_main_root"]
        and result["has_evolve_block"]
        and not result["errors"]
    )
    return result


def validate_prompt_agent(tree: ast.Module, source: str) -> Dict:
    """验证 Prompt 任务的 agent.py"""
    result = {
        "valid": False,
        "has_get_prompt": False,
        "has_evolve_block": False,
        "functions": [],
        "errors": [],
        "suggestions": [],
    }
    
    functions = find_functions(tree)
    result["functions"] = list(functions.keys())
    
    # 检查 get_prompt_generate_press() 函数
    if "get_prompt_generate_press" in functions:
        result["has_get_prompt"] = True
    else:
        result["errors"].append("get_prompt_generate_press() function not found")
        result["suggestions"].append("Add get_prompt_generate_press() function that returns (system_prompt, user_prompt_template)")
    
    # 检查 EVOLVE-BLOCK
    evolve_check = check_evolve_block(source)
    result["has_evolve_block"] = evolve_check["valid"]
    if not evolve_check["valid"]:
        if not evolve_check["has_start"]:
            result["errors"].append("Missing # EVOLVE-BLOCK-START marker")
        if not evolve_check["has_end"]:
            result["errors"].append("Missing # EVOLVE-BLOCK-END marker")
        result["suggestions"].append("Add EVOLVE-BLOCK markers around the system_prompt definition")
    
    result["valid"] = result["has_get_prompt"] and result["has_evolve_block"]
    return result


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    agent_path = input_dir / "agent.py"
    
    if not agent_path.exists():
        print(f"[ERROR] agent.py not found: {agent_path}")
        return 1
    
    tree, error_or_source = parse_agent_file(agent_path)
    
    if tree is None:
        print(f"[ERROR] Failed to parse agent.py: {error_or_source}")
        return 2
    
    source = error_or_source
    
    # 根据任务类型验证
    if args.task_type == "mle":
        result = validate_mle_agent(tree, source)
    else:
        result = validate_prompt_agent(tree, source)
    
    result["file_path"] = str(agent_path)
    result["task_type"] = args.task_type
    
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"[INFO] Validating: {agent_path}")
        print(f"[INFO] Task type: {args.task_type}")
        print(f"[INFO] Functions found: {', '.join(result['functions'])}")
        print()
        
        if result["valid"]:
            print("[OK] agent.py is valid for AlphaEvolve")
        else:
            print("[ERROR] agent.py validation failed")
            for err in result["errors"]:
                print(f"  - {err}")
            
            if result["suggestions"]:
                print()
                print("[SUGGESTIONS]")
                for sug in result["suggestions"]:
                    print(f"  - {sug}")
    
    return 0 if result["valid"] else 3


if __name__ == "__main__":
    sys.exit(main())
