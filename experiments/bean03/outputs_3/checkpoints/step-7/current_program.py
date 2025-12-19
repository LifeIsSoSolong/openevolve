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
from typing import Dict, Tuple
import numpy as np
import lightgbm as lgb
import pandas as pd

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


def _add_time_feats(df: pd.DataFrame, year_med: float) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(int).clip(1, 12)
    ang = 2.0 * np.pi * (m - 1) / 12.0
    df["month_sin"] = np.sin(ang)
    df["month_cos"] = np.cos(ang)
    y = df["year"].astype(float)
    df["year_centered"] = y - year_med
    df["state_month"] = df["state_enc"].astype(float) * m.astype(float)
    return df


def train_and_predict(root) -> Path:
    # do not change this line
    train_path, test_path, output_path = construct_real_path(root)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, state2idx = encode_state(train)
    test, _ = encode_state(test, mapping=state2idx)
    train = months_since_crop_start(train)
    test = months_since_crop_start(test)

    year_med = float(np.nanmedian(train["year"].astype(float)))
    train = _add_time_feats(train, year_med)
    test = _add_time_feats(test, year_med)

    # numeric-only features; exclude raw string 'state'
    numeric_kinds = ("b", "i", "u", "f", "c")
    drop_cols = {"yield", "state"}
    features = [c for c in train.columns if c not in drop_cols and train[c].dtype.kind in numeric_kinds]

    # robust missing fill
    med = train[features].median(numeric_only=True)
    X = train[features].fillna(med)
    X_test = test[features].fillna(med)

    # log target helps relative-error metrics (MAPE-like)
    y = np.log1p(train["yield"].astype(float).clip(lower=0.0))

    # validation: hold out most recent year if feasible else deterministic random split
    max_year = train["year"].max()
    val_mask = (train["year"] == max_year).values
    if int(val_mask.sum()) < max(30, int(0.1 * len(train))):
        rng = np.random.RandomState(42)
        val_mask = (rng.rand(len(train)) < 0.2)

    X_tr, X_va = X.loc[~val_mask], X.loc[val_mask]
    y_tr, y_va = y.loc[~val_mask], y.loc[val_mask]

    params = dict(
        n_estimators=8000,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        objective="regression",
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
    )

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
    )
    best_iter = int(getattr(model, "best_iteration_", params["n_estimators"])) or params["n_estimators"]

    params_full = dict(params)
    params_full["n_estimators"] = best_iter
    model_full = lgb.LGBMRegressor(**params_full)
    model_full.fit(X, y)

    test_pred = np.expm1(model_full.predict(X_test))
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
