"""
Baseline model for bean yield prediction

Reads ./data/train.csv and ./data/test.csv
writes predictions to ./result/submission.csv with columns [year, month, state, yield].

The EVOLVE-BLOCK marks the scope that OpenEvolve is allowed to tune.
"""

from __future__ import annotations

import os
from pathlib import Path

# construct real path from judge.py's ROOT
def construct_real_path(root):
    PROJECT_ROOT = Path(root)
    INPUT_DIR = PROJECT_ROOT
    OUTPUT_DIR = PROJECT_ROOT.parent / "outputs" / "submission"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_PATH = INPUT_DIR / "train.csv"
    TEST_PATH = INPUT_DIR / "test.csv"

    return TRAIN_PATH, TEST_PATH, OUTPUT_DIR

# EVOLVE-BLOCK-START
from typing import Dict, Tuple, List
import numpy as np
import lightgbm as lgb
import pandas as pd


def encode_state(df: pd.DataFrame, mapping: Dict[str, int] | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    if mapping is None:
        states = sorted(df["state"].astype(str).unique())
        mapping = {s: i for i, s in enumerate(states)}
    df["state_enc"] = df["state"].astype(str).map(mapping).fillna(-1).astype(int)
    return df, mapping


def months_since_crop_start(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def transform(m: int) -> int:
        return m - 10 if m >= 10 else m + 2

    df["months_since_crop_start"] = df["month"].astype(int).apply(transform)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(float)
    df["month_sin"] = np.sin(2.0 * np.pi * m / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * m / 12.0)
    df["month_sq"] = m * m
    # simple interactions that often help tree models
    df["state_x_month"] = df["state_enc"].astype(float) * m
    df["state_x_phase"] = df["state_enc"].astype(float) * df["months_since_crop_start"].astype(float)
    return df


def _target_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
    target: str = "yield",
    alpha: float = 20.0,
) -> Tuple[pd.Series, pd.Series]:
    """Smoothed target encoding computed on full train; mapped onto train/test."""
    global_mean = float(train[target].mean())
    agg = train.groupby(cols, dropna=False)[target].agg(["mean", "count"]).reset_index()
    agg["te"] = (agg["mean"] * agg["count"] + global_mean * alpha) / (agg["count"] + alpha)
    key = cols
    tr = train[key].merge(agg[key + ["te"]], on=key, how="left")["te"].fillna(global_mean)
    te = test[key].merge(agg[key + ["te"]], on=key, how="left")["te"].fillna(global_mean)
    return tr, te


def train_and_predict(root) -> Path:
    # do not change this line
    train_path, test_path, output_path = construct_real_path(root)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # base transforms
    train, state2idx = encode_state(train)
    test, _ = encode_state(test, mapping=state2idx)
    train = months_since_crop_start(train)
    test = months_since_crop_start(test)
    train = add_time_features(train)
    test = add_time_features(test)

    # smoothed target encodings (strong for region + season)
    for cols in (["state_enc"], ["month"], ["year"], ["state_enc", "month"], ["state_enc", "months_since_crop_start"]):
        tr_te, te_te = _target_encode(train, test, cols=cols, target="yield", alpha=30.0)
        name = "te_" + "_".join(cols)
        train[name] = tr_te.astype(float)
        test[name] = te_te.astype(float)

    # feature selection (numeric only; keep engineered TE/time features)
    numeric_kinds = ("b", "i", "u", "f", "c")
    candidate_features = [c for c in train.columns if c != "yield"]
    features = [c for c in candidate_features if train[c].dtype.kind in numeric_kinds]

    X = train[features].copy()
    X_test = test[features].copy()

    # robust missing handling
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X_test = X_test.fillna(med)

    # log-transform helps RMSE+MAPE tradeoff; clip at 0 on inverse
    y = np.log1p(train["yield"].astype(float).clip(lower=0.0))

    # simple time-aware validation for early stopping (hold out last year if possible)
    last_year = int(train["year"].max()) if "year" in train.columns else None
    use_val = last_year is not None and (train["year"] == last_year).sum() >= 50 and (train["year"] != last_year).sum() >= 200

    model = lgb.LGBMRegressor(
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    if use_val:
        tr_idx = train["year"] != last_year
        va_idx = ~tr_idx
        model.fit(
            X.loc[tr_idx],
            y.loc[tr_idx],
            eval_set=[(X.loc[va_idx], y.loc[va_idx])],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
        )
        pred_log = model.predict(X_test, num_iteration=model.best_iteration_)
    else:
        model.fit(X, y)
        pred_log = model.predict(X_test)

    test_pred = np.expm1(pred_log)
    test_pred = np.clip(test_pred, 0.0, None)

    test_out = test.copy()
    test_out["yield"] = test_pred
    test_out = test_out[["year", "month", "state", "yield"]]
    out_path = output_path / "submission.csv"
    test_out.to_csv(out_path, index=False)
    print(f"Model trained. Submission saved to: {out_path}")
    return out_path

# EVOLVE-BLOCK-END

def main() -> Path:
    root = r"D:\清华工程博士\C3I\daguan\agentic-rl\mle-openevolve\experiments\bean03\inputs"
    out_path = train_and_predict(root)
    return out_path


if __name__ == "__main__":
    main()
