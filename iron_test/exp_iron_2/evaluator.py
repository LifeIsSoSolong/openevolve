"""
Evaluator for iron_test/exp_iron_base.

Loads a candidate module (program_path), calls its train_predict_evaluate()
to run training/validation/prediction, then returns metrics with a combined_score
for OpenEvolve.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict
import numpy as np


def _load_module(program_path: str):
    module_name = f"candidate_module_{Path(program_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 普通的评价函数
def _compute_combined(metrics: Dict[str, Any]) -> float:
    """
    Combine test_mse / test_mae / test_mape / test_da into a single score.
    Lower is better for mse/mae/mape; higher is better for da.
    """
    if "combined_score" in metrics and isinstance(metrics["combined_score"], (int, float)):
        return float(metrics["combined_score"])
    print("metrics from code:\n", metrics)
    mse = metrics.get("test_mse")
    mae = metrics.get("test_mae")
    mape = metrics.get("test_mape")
    da = metrics.get("test_da")

    def inv(x):
        print(type(x))
        try:
            float_x = float(x)
            return 1.0 / (1.0 + float_x)
        except (TypeError, ValueError):
            return None

    parts = []
    for w, val in [(0.3, mse), (0.25, mae), (0.25, mape)]:
        inv_val = inv(val)
        if inv_val is not None:
            parts.append(w * inv_val)
    if isinstance(da, (int, float)) and not (math.isnan(da) if hasattr(math, "isnan") else False):
        parts.append(0.2 * max(0.0, da))
    if parts:
        return float(sum(parts))
    return 0.0

def _compute_combined_harmonic(metrics: Dict[str, Any]) -> float:
    mse = float(metrics.get("test_mse", 1.0))
    da = float(metrics.get("test_da", 0.0))

    # 1. 精度分：我们需要把 MSE (越小越好) 转化为 0~1 (越大越好)
    # 针对你的数据量级，使用缩放后的反比例
    # 当 RMSE = 0.02 时，得分约 0.83；当 RMSE = 0.1 时，得分 0.5
    rmse = math.sqrt(mse)
    score_precision = 1.0 / (1.0 + 10.0 * rmse)

    # 2. 趋势分：直接用 DA
    score_trend = da

    # 3. 调和平均
    # 加上 1e-6 是为了防止除以零
    harmonic_score = (2 * score_precision * score_trend) / (score_precision + score_trend + 1e-6)

    return float(harmonic_score)


def _to_python_scalar(val: Any) -> Any:
    """Convert numpy scalar types to native Python types for JSON safety."""
    if isinstance(val, np.generic):
        return val.item()
    return val


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    OpenEvolve entry point. Runs train_predict_evaluate() from the candidate module.
    """
    cwd = os.getcwd()
    try:
        module_dir = Path(program_path).resolve().parent
        os.chdir(module_dir)

        module = _load_module(program_path)
        if not hasattr(module, "train_predict_evaluate"):
            return {"combined_score": 0.0, "error": "train_predict_evaluate() not found"}

        result = module.train_predict_evaluate()

        # Allow tuple/list of four metrics (test_mse, test_mae, test_mape, test_da)
        if isinstance(result, (tuple, list)) and len(result) == 4:
            result = {
                "test_mse": result[0],
                "test_mae": result[1],
                "test_mape": result[2],
                "test_da": result[3],
            }

        if not isinstance(result, dict):
            return {"combined_score": 0.0, "error": "train_predict_evaluate() must return dict or 4-tuple"}

        combined = _compute_combined_harmonic(result)
        out = {"combined_score": combined}
        out.update({k: (float(v) if isinstance(v, (int, float, np.generic)) else v) for k, v in result.items()})
        # Final pass to convert any numpy scalars to Python scalars for JSON safety
        out = {k: _to_python_scalar(v) for k, v in out.items()}
        return out

    except Exception as e:
        tb = traceback.format_exc()
        return {"combined_score": 0.0, "error": f"{e}\n{tb}"}
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    print(evaluate("./initial_program.py"))
