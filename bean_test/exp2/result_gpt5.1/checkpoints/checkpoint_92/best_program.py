"""
Baseline model for bean_test exp2.

Reads ./input/train.csv and ./input/test.csv, performs simple preprocessing
(state encoding + months_since_crop_start), trains a LightGBM regressor, and
writes predictions to ./output/submission.csv with columns [year, month, state, yield].

The EVOLVE-BLOCK marks the scope that OpenEvolve is allowed to tune.
"""

from __future__ import annotations

import os
from pathlib import Path

# Paths
# PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(r"D:\清华工程博士\C3I\AutoMLAgent\openevolve\bean_test\exp1")
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = INPUT_DIR / "train.csv"
TEST_PATH = INPUT_DIR / "test.csv"

# EVOLVE-BLOCK-START
from typing import Dict, Tuple
import numpy as np
import lightgbm as lgb
import pandas as pd

def encode_state(df: pd.DataFrame, mapping: Dict[str, int] | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Encode state column to integer IDs."""
    df = df.copy()
    if mapping is None:
        states = sorted(df["state"].unique())
        mapping = {s: i for i, s in enumerate(states)}
    df["state_enc"] = df["state"].map(mapping).fillna(-1).astype(int)
    return df, mapping


def months_since_crop_start(df: pd.DataFrame) -> pd.DataFrame:
    """Map month to a simple crop-phase index."""
    df = df.copy()
    df["months_since_crop_start"] = (df["month"] + 2) % 12
    return df


def add_target_stats(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add simple target-encoding style statistics based only on the training data."""
    train = train.copy()
    test = test.copy()

    global_mean = train["yield"].mean()
    state_mean = train.groupby("state")["yield"].mean()
    month_mean = train.groupby("month")["yield"].mean()
    year_mean = train.groupby("year")["yield"].mean()
    state_month_mean = train.groupby(["state", "month"])["yield"].mean().to_dict()
    state_year_mean = train.groupby(["state", "year"])["yield"].mean().to_dict()

    for df in (train, test):
        df["state_yield_mean"] = df["state"].map(state_mean)
        df["month_yield_mean"] = df["month"].map(month_mean)
        df["year_yield_mean"] = df["year"].map(year_mean)

        key_sm = list(zip(df["state"], df["month"]))
        key_sy = list(zip(df["state"], df["year"]))
        df["state_month_yield_mean"] = pd.Series(key_sm, index=df.index).map(state_month_mean)
        df["state_year_yield_mean"] = pd.Series(key_sy, index=df.index).map(state_year_mean)

        cols = [
            "state_yield_mean",
            "month_yield_mean",
            "year_yield_mean",
            "state_month_yield_mean",
            "state_year_yield_mean",
        ]
        df[cols] = df[cols].fillna(global_mean)

    return train, test


def train_and_predict() -> Path:

    # ---------- read ----------
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # ---------- encode & transform ----------
    train, state2idx = encode_state(train)
    test, _ = encode_state(test, mapping=state2idx)
    train = months_since_crop_start(train)
    test = months_since_crop_start(test)
    train, test = add_target_stats(train, test)

    # ---------- feature selection ----------
    numeric_kinds = ("b", "i", "u", "f", "c")
    features = [col for col in train.columns if col != "yield" and train[col].dtype.kind in numeric_kinds]
    target = "yield"

    # ---------- train single LightGBM model ----------
    # Slightly smaller learning rate and more trees + a bit more capacity
    # often improve generalisation on tabular problems like this.
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.025,
        num_leaves=96,
        min_child_samples=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
    )
    # Treat state, month and year as categorical for more flexible splits
    model.fit(train[features], train[target], categorical_feature=["state_enc", "month", "year"])

    # ---------- predict with stacked linear calibration ----------
    train_pred = model.predict(train[features])
    test_pred = model.predict(test[features])

    # baseline from historical state-month mean yield
    train_baseline = train["state_month_yield_mean"].values
    test_baseline = test["state_month_yield_mean"].values

    y_true = train[target].values

    # linear regression with intercept: y ≈ w0 + w1 * model_pred + w2 * baseline
    X = np.column_stack([np.ones_like(train_pred), train_pred, train_baseline])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y_true, rcond=None)
        train_pred_cal = X @ coef
        X_test = np.column_stack([np.ones_like(test_pred), test_pred, test_baseline])
        test_pred = X_test @ coef
    except np.linalg.LinAlgError:
        # fallback to original 1D variance-based calibration
        var = np.var(train_pred)
        if var > 0:
            cov = np.cov(train_pred, y_true, bias=True)[0, 1]
            b = cov / var
            a = y_true.mean() - b * train_pred.mean()
            train_pred_cal = a + b * train_pred
            test_pred = a + b * test_pred
        else:
            # degenerate case: keep uncalibrated predictions
            train_pred_cal = train_pred

    # bias correction by (state, year) using training residuals
    residual = y_true - train_pred_cal
    tmp = train[["state", "year"]].copy()
    tmp["residual"] = residual
    sy_residual = tmp.groupby(["state", "year"])["residual"].mean().to_dict()
    key_sy_test = list(zip(test["state"], test["year"]))
    correction = pd.Series(key_sy_test, index=test.index).map(sy_residual).fillna(0.0).to_numpy()
    test_pred = test_pred + correction

    # clip calibrated predictions (after bias correction) to the observed training range
    y_min, y_max = y_true.min(), y_true.max()
    test_pred = np.clip(test_pred, y_min, y_max)

    # ---------- output ----------
    test_out = test[["year", "month", "state"]].copy()
    test_out["yield"] = test_pred
    out_path = OUTPUT_DIR / "submission.csv"
    test_out.to_csv(out_path, index=False)
    print(f"Model training finished, predictions saved to: {out_path}")
    return out_path

# EVOLVE-BLOCK-END

def main() -> Path:
    out_path = train_and_predict()
    print(f"Model trained; predictions saved to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
