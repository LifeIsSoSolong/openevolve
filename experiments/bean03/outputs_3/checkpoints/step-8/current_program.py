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
    s = df["state"].astype(str)
    if mapping is None:
        states = sorted(s.unique())
        mapping = {k: i for i, k in enumerate(states)}
    df["state_enc"] = s.map(mapping).fillna(-1).astype(int)
    return df, mapping

def months_since_crop_start(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(int)
    df["months_since_crop_start"] = (m - 10).where(m >= 10, m + 2).astype(int)
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(int)
    ang = 2.0 * np.pi * (m - 1) / 12.0
    df["month_sin"] = np.sin(ang)
    df["month_cos"] = np.cos(ang)
    y = df["year"].astype(int)
    df["year_centered"] = (y - y.min()).astype(int)
    return df

def _smoothed_mean(stats: pd.DataFrame, global_mean: float, smooth: float) -> pd.Series:
    # stats has columns: ["mean","count"]
    return (stats["mean"] * stats["count"] + global_mean * smooth) / (stats["count"] + smooth)

def add_target_encoding_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    cols: List[str],
    n_splits: int = 5,
    smooth: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = train.copy()
    te = test.copy()

    key_name = "_".join(cols)
    te_col = f"te_{key_name}"
    global_mean = float(y.mean())

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = pd.Series(index=tr.index, dtype=float)

    # precompute key for speed/determinism
    tr_key = tr[cols].astype(str).agg("|".join, axis=1)
    te_key = te[cols].astype(str).agg("|".join, axis=1)

    for tr_idx, va_idx in kf.split(tr):
        stats = (
            pd.DataFrame({"key": tr_key.iloc[tr_idx].values, "y": y.iloc[tr_idx].values})
            .groupby("key")["y"]
            .agg(["mean", "count"])
        )
        mapping = _smoothed_mean(stats, global_mean, smooth).to_dict()
        oof.iloc[va_idx] = tr_key.iloc[va_idx].map(mapping).fillna(global_mean).astype(float)

    # test mapping from full train
    stats_full = (
        pd.DataFrame({"key": tr_key.values, "y": y.values})
        .groupby("key")["y"]
        .agg(["mean", "count"])
    )
    mapping_full = _smoothed_mean(stats_full, global_mean, smooth).to_dict()
    te_vals = te_key.map(mapping_full).fillna(global_mean).astype(float)

    tr[te_col] = oof.values
    te[te_col] = te_vals.values
    return tr, te

def train_and_predict(root) -> Path:
    # do not change this line
    train_path, test_path, output_path = construct_real_path(root)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, state2idx = encode_state(train)
    test, _ = encode_state(test, mapping=state2idx)

    train = months_since_crop_start(train)
    test = months_since_crop_start(test)
    train = add_time_features(train)
    test = add_time_features(test)

    y = train["yield"].astype(float)

    # Out-of-fold target encoding to reduce leakage/overfit
    train, test = add_target_encoding_oof(train, test, y, cols=["state"], n_splits=5, smooth=30.0)
    train, test = add_target_encoding_oof(train, test, y, cols=["state", "month"], n_splits=5, smooth=50.0)
    train, test = add_target_encoding_oof(train, test, y, cols=["state", "months_since_crop_start"], n_splits=5, smooth=50.0)

    # numeric-only features (state kept as state_enc + TE)
    numeric_kinds = ("b", "i", "u", "f", "c")
    features = [c for c in train.columns if c != "yield" and train[c].dtype.kind in numeric_kinds]

    # robust missing fill
    med = train[features].median(numeric_only=True)
    X_all = train[features].fillna(med)
    X_test = test[features].fillna(med)

    # log-target to better match MAPE-like behavior; ensure positivity on output
    y_log = np.log1p(y.clip(lower=0.0).values)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_pred_log = np.zeros(len(X_test), dtype=float)

    for tr_idx, va_idx in kf.split(X_all):
        X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
        y_tr, y_va = y_log[tr_idx], y_log[va_idx]

        model = lgb.LGBMRegressor(
            n_estimators=6000,
            learning_rate=0.03,
            num_leaves=127,
            min_child_samples=25,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            objective="regression",
            n_jobs=-1,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
        test_pred_log += model.predict(X_test) / kf.get_n_splits()

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
