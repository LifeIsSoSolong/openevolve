"""
Quick evaluator for bean_test predictions.

Computes MAPE and RMSE between a submission CSV and the reference answers.
Defaults:
  - answers: ./input/test_answer.csv
  - preds:   ./output/submission.csv

Usage:
  python easy_metric.py
  python easy_metric.py --pred ./output/submission.csv --truth ./input/test_answer.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def mape_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute MAPE and RMSE with small epsilon to avoid div-by-zero."""
    eps = 1e-6
    mape = np.mean(np.abs((y_pred - y_true) / np.clip(np.abs(y_true), eps, None)))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return float(mape), float(rmse)


def evaluate(pred_path: Path, truth_path: Path) -> Tuple[float, float]:
    """Load CSVs, align by key columns, and compute metrics."""
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"Truth file not found: {truth_path}")

    pred_df = pd.read_csv(pred_path)
    truth_df = pd.read_csv(truth_path)

    # Expected columns: year, month, state, yield
    required_cols = ["year", "month", "state", "yield"]
    for name, df in [("prediction", pred_df), ("truth", truth_df)]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} file missing columns: {missing}")

    # Align on keys (year, month, state)
    merged = truth_df.merge(
        pred_df,
        on=["year", "month", "state"],
        how="left",
        suffixes=("_true", "_pred"),
    )
    if merged["yield_pred"].isna().any():
        missing_rows = merged[merged["yield_pred"].isna()][["year", "month", "state"]]
        raise ValueError(f"Predictions missing for some rows:\n{missing_rows.head()}")

    # Drop rows where truth or pred is NaN
    merged = merged.dropna(subset=["yield_true", "yield_pred"])
    if merged.empty:
        raise ValueError("No valid rows to evaluate after dropping NaNs.")

    y_true = merged["yield_true"].to_numpy(dtype=float)
    y_pred = merged["yield_pred"].to_numpy(dtype=float)

    return mape_rmse(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bean_test predictions (MAPE, RMSE)")
    parser.add_argument("--pred", type=Path, default=Path("./output/submission.csv"), help="Path to submission CSV")
    parser.add_argument("--truth", type=Path, default=Path("./input/test_answer.csv"), help="Path to ground truth CSV")
    args = parser.parse_args()

    mape, rmse = evaluate(args.pred, args.truth)
    print(f"MAPE: {mape:.12f}")
    print(f"RMSE: {rmse:.12f}")


if __name__ == "__main__":
    main()
