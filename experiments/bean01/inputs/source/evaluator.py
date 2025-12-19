"""
Evaluator for bean_test exp1.

Loads the candidate program, runs its training/prediction pipeline to produce
./output/submission.csv (relative to this exp1 folder), then computes MAPE and RMSE
against ./input/test_answer.csv. Returns a metrics dict with combined_score.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent  # 这个路径是真实路径不是临时路径

INPUT_PATH = ROOT / "data" / "test_answer.csv"

# print(f"!!!!!!!!!!!in evaluator.py -> first line. Evaluator path | root path:{ROOT}, input path: {INPUT_PATH}, pred path: {PRED_PATH}")

def mape_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    eps = 1e-6
    mape = np.mean(np.abs((y_pred - y_true) / np.clip(np.abs(y_true), eps, None)))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return float(mape), float(rmse)


def evaluate_predictions(pred_path: Path, truth_path: Path) -> Dict[str, float]:
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"Truth file not found: {truth_path}")

    pred_df = pd.read_csv(pred_path)
    truth_df = pd.read_csv(truth_path)

    required_cols = ["year", "month", "state", "yield"]
    for name, df in [("prediction", pred_df), ("truth", truth_df)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} file missing columns: {missing}")

    merged = truth_df.merge(
        pred_df,
        on=["year", "month", "state"],
        how="left",
        suffixes=("_true", "_pred"),
    )
    merged = merged.dropna(subset=["yield_true", "yield_pred"])
    if merged.empty:
        raise ValueError("No valid rows to evaluate after dropping NaNs.")

    y_true = merged["yield_true"].to_numpy(dtype=float)
    y_pred = merged["yield_pred"].to_numpy(dtype=float)

    mape, rmse = mape_rmse(y_true, y_pred)
    print(f"✅ Evaluation results - MAPE: {mape:.6f}, RMSE: {rmse:.6f}")
    combined = 0.5 * (1.0 / (1.0 + mape)) + 0.5 * (1.0 / (1.0 + rmse))
    return {"combined_score": combined, "mape": mape, "rmse": rmse}


def _load_module(program_path: str):
    module_name = f"candidate_module_{Path(program_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    OpenEvolve entry point.
    Runs the candidate program to generate submission.csv, then scores it.
    """
    # print(f"!!!!!!!!!!!in evaluator.py -> evaluate(), program path: {program_path}")
    try:
        module = _load_module(program_path)
        # Run candidate pipeline
        if hasattr(module, "train_and_predict"):
            out_path = module.train_and_predict(ROOT)

        metrics = evaluate_predictions(out_path, INPUT_PATH)
        return metrics
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}


if __name__ == "__main__":
    program_file = "./initial_program.py"
    results = evaluate(program_file)
    print("Evaluation Results:", results)
