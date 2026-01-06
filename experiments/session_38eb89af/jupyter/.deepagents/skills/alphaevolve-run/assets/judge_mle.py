"""
Evaluator for bean04.

Loads the candidate program, runs its main to get metrics, and construct combined_score.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from openevolve.evaluation_result import EvaluationResult


ROOT = Path(__file__).resolve().parent  # 这个路径是真实路径不是临时路径


def _load_module(program_path: str):
    module_name = f"candidate_module_{Path(program_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_evaluation_result(metrics_original):

    rmse = float(metrics_original.get("rmse", float("nan")))
    rrmse = float(metrics_original.get("rrmse", float("nan")))
    mape = float(metrics_original.get("mape", float("nan")))
    print(f"✅ Evaluation results - RMSE: {rmse:.6f}, rRMSE: {rrmse:.6f}, MAPE: {mape:.6f}")
    combined = 0.5 * (1.0 / (1.0 + mape)) + 0.5 * (1.0 / (1.0 + rmse))

    metrics = {
        "combined_score": float(combined),
        "rmse": rmse,
        "rrmse": rrmse,
        "mape": mape,
    }
    artifacts = {}

    return EvaluationResult(
        metrics=metrics,
        artifacts=artifacts
    )

def evaluate(program_path: str) -> Dict[str, Any]:
    """
    OpenEvolve entry point.
    Runs the candidate program to get metrics, and construct combined_score.
    """
    try:
        module = _load_module(program_path)
        # Run candidate pipeline
        if hasattr(module, "main"):
            metrics_original = module.main(ROOT)

        metrics_openevolve = get_evaluation_result(metrics_original)
        return metrics_openevolve
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}


if __name__ == "__main__":
    program_file = "./agent.py"
    results = evaluate(program_file)
    print("Evaluation Results:", results)
