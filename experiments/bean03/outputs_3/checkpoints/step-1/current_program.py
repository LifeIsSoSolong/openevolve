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
from sklearn.model_selection import KFold

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


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(int).clip(1, 12)
    ang = 2.0 * np.pi * (m - 1) / 12.0
    df["month_sin"] = np.sin(ang)
    df["month_cos"] = np.cos(ang)
    df["year_centered"] = df["year"].astype(float) - df["year"].astype(float).median()
    return df


def _target_encode_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
    target: str,
    n_splits: int = 5,
    seed: int = 42,
    smooth: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold smoothed target encoding for train; full-train encoding for test (no label leakage)."""
    global_mean = float(train[target].mean())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = pd.Series(index=train.index, dtype=float)
    for tr_idx, val_idx in kf.split(train):
        tr = train.iloc[tr_idx]
        stats = tr.groupby(cols)[target].agg(["mean", "count"])
        enc = (stats["mean"] * stats["count"] + global_mean * smooth) / (stats["count"] + smooth)
        enc = enc.rename("te").reset_index()
        oof.iloc[val_idx] = (
            train.iloc[val_idx][cols].merge(enc, on=cols, how="left")["te"].fillna(global_mean).values
        )

    stats_full = train.groupby(cols)[target].agg(["mean", "count"])
    enc_full = (stats_full["mean"] * stats_full["count"] + global_mean * smooth) / (stats_full["count"] + smooth)
    enc_full = enc_full.rename("te").reset_index()
    te_test = test[cols].merge(enc_full, on=cols, how="left")["te"].fillna(global_mean).values
    return oof.values, te_test


def train_and_predict(root) -> Path:

    # do not change this line
    train_path, test_path, output_path = construct_real_path(root)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # basic transforms
    train, state2idx = encode_state(train)
    test, _ = encode_state(test, mapping=state2idx)
    train = months_since_crop_start(train)
    test = months_since_crop_start(test)
    train = _add_time_features(train)
    test = _add_time_features(test)

    # OOF target encoding to improve generalization
    te_state_tr, te_state_te = _target_encode_oof(train, test, ["state"], "yield", n_splits=5, seed=42, smooth=30.0)
    train["te_state"] = te_state_tr
    test["te_state"] = te_state_te

    te_sm_tr, te_sm_te = _target_encode_oof(
        train, test, ["state", "month"], "yield", n_splits=5, seed=42, smooth=50.0
    )
    train["te_state_month"] = te_sm_tr
    test["te_state_month"] = te_sm_te

    # feature selection (numeric only; exclude raw string)
    numeric_kinds = ("b", "i", "u", "f", "c")
    drop_cols = {"yield", "state"}
    candidate_features = [c for c in train.columns if c not in drop_cols]
    features = [c for c in candidate_features if train[c].dtype.kind in numeric_kinds]

    # log1p target helps relative-error metrics (MAPE-like)
    y = np.log1p(train["yield"].astype(float).clip(lower=0.0))

    # CV bagging + early stopping for stability
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_pred_log = np.zeros(len(test), dtype=float)

    params = dict(
        n_estimators=8000,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        objective="regression",
    )

    X = train[features]
    X_test = test[features]

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
        )
        test_pred_log += model.predict(X_test, num_iteration=model.best_iteration_) / kf.get_n_splits()

    test_pred = np.expm1(test_pred_log)
    test_pred = np.clip(test_pred, 0.0, None)

    test_out = test.copy()
    test_out["yield"] = test_pred
    test_out = test_out[["year", "month", "state", "yield"]]
    out_path = output_path / "submission.csv"
    test_out.to_csv(out_path, index=False)
    return out_path

# EVOLVE-BLOCK-END

def main() -> Path:
    root = r"D:\清华工程博士\C3I\daguan\agentic-rl\mle-openevolve\experiments\bean03\inputs"
    out_path = train_and_predict(root)
    return out_path


if __name__ == "__main__":
    main()
