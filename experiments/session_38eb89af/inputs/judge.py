"""
AlphaEvolve evaluator for iron ore futures price prediction.

Evaluates the candidate program based on MDA (primary), RMSE and MAPE (secondary).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

from openevolve.evaluation_result import EvaluationResult


ROOT = Path(__file__).resolve().parent  # 真实数据路径，不是临时路径


def _load_module(program_path: str):
    """动态加载候选程序模块"""
    module_name = f"candidate_module_{Path(program_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def calculate_combined_score(metrics: Dict[str, Any]) -> float:
    """
    将原始指标转换为 combined_score (0~1, 且>0)
    
    指标说明：
    - MDA (Mean Directional Accuracy): 方向准确率，0~100，越大越好（主要指标，权重 0.6）
    - RMSE (Root Mean Squared Error): 均方根误差，越小越好（次要指标，权重 0.2）
    - MAPE (Mean Absolute Percentage Error): 平均绝对百分比误差，0~100，越小越好（次要指标，权重 0.2）
    """
    mda = float(metrics.get("mda", 0.0))  # 范围 0~100
    rmse = float(metrics.get("rmse", float("inf")))
    mape = float(metrics.get("mape", float("inf")))
    
    # 转换为 0~1 分数
    # MDA: 直接归一化（百分比 -> 0~1）
    score_mda = mda / 100.0
    
    # RMSE: 使用倒数变换（越小越好）
    score_rmse = 1.0 / (1.0 + rmse)
    
    # MAPE: 使用倒数变换（越小越好，百分比形式）
    score_mape = 1.0 / (1.0 + mape / 100.0)
    
    # 加权组合：MDA 主要（60%），RMSE 和 MAPE 次要（各20%）
    combined = 0.6 * score_mda + 0.2 * score_rmse + 0.2 * score_mape
    
    # 确保在 (0, 1] 范围内
    return max(0.01, min(1.0, combined))


def evaluate(program_path: str) -> EvaluationResult:
    """
    AlphaEvolve 评估入口
    
    Args:
        program_path: 候选程序路径
    
    Returns:
        EvaluationResult with combined_score and metrics
    """
    try:
        # 加载候选程序
        module = _load_module(program_path)
        
        if not hasattr(module, "main"):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "main(root) not found"}
            )
        
        # 运行候选程序，传入数据目录
        metrics_original = module.main(str(ROOT))
        
        if not isinstance(metrics_original, dict):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "main(root) must return dict"}
            )
        
        # 提取指标
        mda = float(metrics_original.get("mda", 0.0))
        rmse = float(metrics_original.get("rmse", float("inf")))
        mape = float(metrics_original.get("mape", float("inf")))
        
        print(f"📊 Original metrics - MDA: {mda:.2f}%, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")
        
        # 计算综合分数
        combined = calculate_combined_score(metrics_original)
        
        print(f"✅ Combined score: {combined:.6f}")
        
        # 返回评估结果
        return EvaluationResult(
            metrics={
                "combined_score": combined,
                "mda": mda,
                "rmse": rmse,
                "mape": mape,
            },
            artifacts={}
        )
    
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"❌ Evaluation failed: {error_msg}")
        traceback.print_exc()
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": error_msg}
        )


if __name__ == "__main__":
    # 本地测试
    program_file = "./agent.py"
    results = evaluate(program_file)
    print(f"\nEvaluation Results: {results}")
