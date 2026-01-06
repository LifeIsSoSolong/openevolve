#!/usr/bin/env python3
"""
根据 agent.py 和数据文件自动生成 judge.py。

Usage:
    python generate_judge.py --input-dir /path/to/inputs --task-type mle --output /path/to/judge.py
    python generate_judge.py --input-dir /path/to/inputs --task-type mle --metrics rmse,mape,r2
    python generate_judge.py --input-dir /path/to/inputs --task-type mle --metrics "rmse:lower,r2:higher,custom:lower_pct"
    
指标规格格式:
    - "rmse" - 自动检测属性
    - "rmse:lower" - 越小越好
    - "r2:higher" - 越大越好(范围0-1)
    - "mape:lower_pct" - 越小越好，百分比
    - "r2:higher_r2" - 越大越好，r2类型(范围-1到1)
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_MLE_METRICS = ["rmse", "mape"]

# 已知指标属性
KNOWN_METRICS = {
    # 回归 - 越小越好
    "rmse": ("lower", "越小越好"),
    "mse": ("lower", "越小越好"),
    "mae": ("lower", "越小越好"),
    "mape": ("lower_pct", "越小越好(百分比)"),
    "rrmse": ("lower_pct", "越小越好(百分比)"),
    "smape": ("lower_pct", "越小越好(百分比)"),
    "wape": ("lower_pct", "越小越好(百分比)"),
    "wmape": ("lower_pct", "越小越好(百分比)"),
    
    # 回归 - 越大越好
    "r2": ("higher_r2", "越大越好(范围-1到1)"),
    "r2_score": ("higher_r2", "越大越好(范围-1到1)"),
    "explained_variance": ("higher", "越大越好(范围0-1)"),
    
    # 分类 - 越大越好
    "accuracy": ("higher", "越大越好(范围0-1)"),
    "precision": ("higher", "越大越好(范围0-1)"),
    "recall": ("higher", "越大越好(范围0-1)"),
    "f1": ("higher", "越大越好(范围0-1)"),
    "f1_score": ("higher", "越大越好(范围0-1)"),
    "auc": ("higher", "越大越好(范围0-1)"),
    "roc_auc": ("higher", "越大越好(范围0-1)"),
    "auc_roc": ("higher", "越大越好(范围0-1)"),
    "pr_auc": ("higher", "越大越好(范围0-1)"),
    "ap": ("higher", "越大越好(范围0-1)"),
    "average_precision": ("higher", "越大越好(范围0-1)"),
    "mcc": ("higher_r2", "越大越好(范围-1到1)"),
    "kappa": ("higher_r2", "越大越好(范围-1到1)"),
    
    # 分类 - 越小越好
    "log_loss": ("lower", "越小越好"),
    "logloss": ("lower", "越小越好"),
    "cross_entropy": ("lower", "越小越好"),
    "brier_score": ("lower", "越小越好"),
    "error_rate": ("lower", "越小越好"),
    
    # 排序 - 越大越好
    "ndcg": ("higher", "越大越好(范围0-1)"),
    "mrr": ("higher", "越大越好(范围0-1)"),
    "hit_rate": ("higher", "越大越好(范围0-1)"),
}


def get_metric_type(name: str, hint: Optional[str] = None) -> Tuple[str, str, bool]:
    """
    获取指标类型
    
    Returns:
        (formula_type, description, is_known)
        formula_type: "lower", "lower_pct", "higher", "higher_r2", "unknown"
    """
    if hint:
        hint = hint.lower()
        if hint == "lower":
            return "lower", "越小越好", True
        elif hint == "lower_pct":
            return "lower_pct", "越小越好(百分比)", True
        elif hint == "higher":
            return "higher", "越大越好(范围0-1)", True
        elif hint == "higher_r2":
            return "higher_r2", "越大越好(范围-1到1)", True
    
    name_lower = name.lower()
    
    # 精确匹配
    if name_lower in KNOWN_METRICS:
        t, d = KNOWN_METRICS[name_lower]
        return t, d, True
    
    # 模糊匹配
    for key, (t, d) in KNOWN_METRICS.items():
        if key in name_lower or name_lower.endswith(f"_{key}") or name_lower.startswith(f"{key}_"):
            return t, d, True
    
    return "unknown", "⚠️ 未知指标，请确认", False


def generate_score_code(name: str, formula_type: str) -> str:
    """生成单个指标的分数计算代码"""
    var = f"score_{name}"
    
    if formula_type == "lower":
        return f'    {var} = 1.0 / (1.0 + float(metrics.get("{name}", float("inf"))))'
    elif formula_type == "lower_pct":
        return f'    {var} = 1.0 / (1.0 + float(metrics.get("{name}", float("inf"))) / 100.0)'
    elif formula_type == "higher":
        return f'    {var} = max(0.0, min(1.0, float(metrics.get("{name}", 0))))'
    elif formula_type == "higher_r2":
        return f'    _{name}_v = float(metrics.get("{name}", 0))\n    {var} = max(0.0, min(1.0, (_{name}_v + 1.0) / 2.0))'
    else:
        # 未知指标：生成带注释的占位代码
        return f'''    # ⚠️ TODO: 请确认 "{name}" 的属性并选择正确的公式:
    # 如果越小越好: {var} = 1.0 / (1.0 + float(metrics.get("{name}", float("inf"))))
    # 如果越大越好(0-1): {var} = max(0.0, min(1.0, float(metrics.get("{name}", 0))))
    # 如果越大越好(如r2): {var} = max(0.0, min(1.0, (float(metrics.get("{name}", 0)) + 1.0) / 2.0))
    {var} = 1.0 / (1.0 + float(metrics.get("{name}", float("inf"))))  # 默认越小越好'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate judge.py for AlphaEvolve")
    parser.add_argument("--input-dir", required=True, help="Path to input directory")
    parser.add_argument("--task-type", required=True, choices=["mle", "prompt"], help="Task type")
    parser.add_argument("--output", help="Output path (default: <input-dir>/judge.py)")
    parser.add_argument("--dry-run", action="store_true", help="Print generated code without writing")
    parser.add_argument("--metrics", help='Metrics, e.g. "rmse,mape,r2" or "rmse:lower,r2:higher"')
    return parser.parse_args()


def analyze_agent_return(agent_path: Path) -> Optional[List[str]]:
    """分析 agent.py 返回的 metrics"""
    try:
        source = agent_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        
        all_funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        
        def get_dict_keys(node):
            if isinstance(node, ast.Dict):
                return [k.value if isinstance(k, ast.Constant) else k.s 
                        for k in node.keys if isinstance(k, (ast.Constant, ast.Str))]
            return []
        
        main = all_funcs.get("main")
        if not main:
            return None
        
        for child in ast.walk(main):
            if isinstance(child, ast.Return) and child.value:
                if isinstance(child.value, ast.Dict):
                    keys = get_dict_keys(child.value)
                    if keys:
                        return keys
                elif isinstance(child.value, ast.Name):
                    var = child.value.id
                    for stmt in ast.walk(main):
                        if isinstance(stmt, ast.Assign):
                            for t in stmt.targets:
                                if isinstance(t, ast.Name) and t.id == var:
                                    if isinstance(stmt.value, ast.Dict):
                                        keys = get_dict_keys(stmt.value)
                                        if keys:
                                            return keys
                                    elif isinstance(stmt.value, ast.Call):
                                        fn = stmt.value.func
                                        fname = fn.id if isinstance(fn, ast.Name) else None
                                        if fname and fname in all_funcs:
                                            for s in ast.walk(all_funcs[fname]):
                                                if isinstance(s, ast.Return) and isinstance(s.value, ast.Dict):
                                                    keys = get_dict_keys(s.value)
                                                    if keys:
                                                        return keys
        
        for fname in ["compute_metrics", "calculate_metrics", "eval_metrics", "get_metrics"]:
            if fname in all_funcs:
                for s in ast.walk(all_funcs[fname]):
                    if isinstance(s, ast.Return) and isinstance(s.value, ast.Dict):
                        keys = get_dict_keys(s.value)
                        if keys:
                            return keys
        return None
    except:
        return None


def generate_mle_judge(input_dir: Path, metric_specs: List[str]) -> Tuple[str, List[Dict]]:
    """生成 MLE 任务的 judge.py"""
    
    # 解析指标规格
    metrics_info = []
    for spec in metric_specs:
        if ":" in spec:
            name, hint = spec.split(":", 1)
        else:
            name, hint = spec, None
        ftype, desc, known = get_metric_type(name.strip(), hint)
        metrics_info.append({
            "name": name.strip(),
            "type": ftype,
            "desc": desc,
            "known": known
        })
    
    # 生成分数计算代码
    score_lines = []
    for m in metrics_info:
        score_lines.append(f"    # {m['name']}: {m['desc']}")
        score_lines.append(generate_score_code(m['name'], m['type']))
    
    score_vars = " + ".join([f"score_{m['name']}" for m in metrics_info])
    n = len(metrics_info)
    
    # 生成指标说明
    metric_docs = "\n".join([f"#   - {m['name']}: {m['desc']}" for m in metrics_info])
    
    has_unknown = any(not m['known'] for m in metrics_info)
    warning = ""
    if has_unknown:
        warning = "\n# ⚠️ 警告: 存在未知指标，请检查 calculate_combined_score 函数中的 TODO 注释\n"
    
    template = f'''"""
AlphaEvolve evaluator for MLE task.
Auto-generated by generate_judge.py
{warning}
指标说明:
{metric_docs}
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

from openevolve.evaluation_result import EvaluationResult

ROOT = Path(__file__).resolve().parent


def _load_module(program_path: str):
    """动态加载候选程序模块"""
    module_name = f"candidate_module_{{Path(program_path).stem}}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {{program_path}}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def calculate_combined_score(metrics: Dict[str, Any]) -> float:
    """
    将原始指标转换为 combined_score (0~1)
    
    转换规则:
    - 越小越好: score = 1 / (1 + value)
    - 越小越好(百分比): score = 1 / (1 + value/100)
    - 越大越好(0-1范围): score = value (clamp到0-1)
    - 越大越好(r2类型): score = (value + 1) / 2
    """
{chr(10).join(score_lines)}
    
    combined = ({score_vars}) / {n}
    return max(1e-12, min(1.0, combined))


def evaluate(program_path: str) -> EvaluationResult:
    """AlphaEvolve 评估入口"""
    try:
        module = _load_module(program_path)
        
        if not hasattr(module, "main"):
            return EvaluationResult(
                metrics={{"combined_score": 0.0, "error": "main(root) not found"}}
            )
        
        raw_metrics = module.main(ROOT)
        combined = calculate_combined_score(raw_metrics)
        
        result_metrics = {{"combined_score": combined, **raw_metrics}}
        print(f"✅ Evaluation results: {{result_metrics}}")
        return EvaluationResult(metrics=result_metrics)
        
    except Exception as e:
        print(f"❌ Evaluation error: {{e}}")
        return EvaluationResult(metrics={{"combined_score": 0.0, "error": str(e)}})


if __name__ == "__main__":
    result = evaluate(str(ROOT / "agent.py"))
    print(f"Evaluation Result: {{result}}")
'''
    return template, metrics_info


def generate_prompt_judge(input_dir: Path) -> str:
    """生成 Prompt 任务的 judge.py"""
    return '''"""
AlphaEvolve evaluator for Prompt task.
Auto-generated by generate_judge.py
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from openevolve.evaluation_result import EvaluationResult

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.jsonl"
TEST_PATH = ROOT / "test.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_module(program_path: str):
    module_name = f"candidate_module_{Path(program_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def evaluate(program_path: str) -> EvaluationResult:
    try:
        sys.path.insert(0, str(ROOT))
        from generate_press_agent import generate_press_agent
        from evaluate_press_agent import evaluate_press_agent
        
        module = _load_module(program_path)
        
        if not hasattr(module, "get_prompt_generate_press"):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "get_prompt_generate_press() not found"}
            )
        
        system_prompt, user_prompt_template = module.get_prompt_generate_press()
        log.info("Loaded prompt, length=%d", len(system_prompt))
        
        train_data = _read_jsonl(TRAIN_PATH)
        log.info("Loaded %d training samples", len(train_data))
        
        scores = []
        for idx, sample in enumerate(train_data):
            try:
                user_prompt = user_prompt_template.format(
                    interview_type=sample.get("interview_type", ""),
                    interview_context=sample.get("interview_context", ""),
                )
                generated = generate_press_agent(
                    model_name=os.getenv("GENERATE_MODEL", "gpt-4"),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                result = evaluate_press_agent(
                    model_name=os.getenv("EVALUATE_MODEL", "gpt-4"),
                    generated_press=generated,
                    reference_press=sample.get("ground_truth", ""),
                )
                score = result.get("combined_score", 0.0)
                scores.append(score)
                log.info("Sample %d: score=%.4f", idx, score)
            except Exception as e:
                log.error("Sample %d failed: %s", idx, e)
                scores.append(0.0)
        
        combined = mean(scores) if scores else 0.0
        log.info("Final combined_score: %.4f", combined)
        
        return EvaluationResult(metrics={"combined_score": combined, "num_samples": len(scores)})
        
    except Exception as e:
        log.error("Evaluation error: %s", e)
        return EvaluationResult(metrics={"combined_score": 0.0, "error": str(e)})


if __name__ == "__main__":
    result = evaluate(str(ROOT / "agent.py"))
    print(f"Evaluation Result: {result}")
'''


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output) if args.output else input_dir / "judge.py"
    
    agent_path = input_dir / "agent.py"
    if not agent_path.exists():
        print(f"[ERROR] agent.py not found: {agent_path}")
        return 1
    
    if args.task_type == "mle":
        # 解析指标
        if args.metrics:
            metric_specs = [m.strip() for m in args.metrics.split(",") if m.strip()]
        else:
            # 尝试从 agent.py 自动检测
            detected = analyze_agent_return(agent_path)
            if detected:
                metric_specs = detected
                print(f"[INFO] Detected metrics from agent.py: {metric_specs}")
            else:
                print("[ERROR] Could not detect metrics from agent.py")
                print("[INFO] Please specify metrics with --metrics, e.g.:")
                print('       --metrics "rmse,mape,r2"')
                print('       --metrics "rmse:lower,r2:higher,custom_metric:lower_pct"')
                return 2
        
        code, metrics_info = generate_mle_judge(input_dir, metric_specs)
        
        # 显示指标信息
        print("\n[INFO] 指标配置:")
        has_unknown = False
        for m in metrics_info:
            status = "✓" if m["known"] else "⚠️"
            print(f"  {status} {m['name']}: {m['desc']}")
            if not m["known"]:
                has_unknown = True
        
        if has_unknown:
            print("\n[WARNING] 存在未知指标，请检查生成的代码中的 TODO 注释")
            print("[INFO] 可以使用 --metrics 显式指定指标类型，例如:")
            print('       --metrics "custom_metric:lower" (越小越好)')
            print('       --metrics "custom_metric:higher" (越大越好，范围0-1)')
            print('       --metrics "custom_metric:lower_pct" (越小越好，百分比)')
            print('       --metrics "custom_metric:higher_r2" (越大越好，范围-1到1)')
        
    else:
        code = generate_prompt_judge(input_dir)
    
    if args.dry_run:
        print("\n[DRY RUN] Generated judge.py:")
        print("=" * 60)
        print(code)
        print("=" * 60)
    else:
        output_path.write_text(code, encoding="utf-8")
        print(f"\n[OK] Generated judge.py: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
