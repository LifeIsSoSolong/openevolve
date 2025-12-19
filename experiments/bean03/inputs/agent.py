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
    if mapping is None:
        s = sorted(df["state"].astype(str).unique())
        mapping = {k: i for i, k in enumerate(s)}
    df["state_enc"] = df["state"].astype(str).map(mapping).fillna(-1).astype(np.int32)
    return df, mapping

def months_since_crop_start(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(int)
    df["months_since_crop_start"] = np.where(m >= 10, m - 10, m + 2).astype(np.int16)
    return df

def _fe(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, s2i = encode_state(train)
    test, _ = encode_state(test, s2i)
    train = months_since_crop_start(train)
    test = months_since_crop_start(test)

    for df in (train, test):
        ang = 2 * np.pi * (df["month"].astype(float) / 12.0)
        df["month_sin"] = np.sin(ang)
        df["month_cos"] = np.cos(ang)

    y = train["yield"].astype(float)
    g = train.groupby("state", dropna=False)["yield"]
    stat = pd.DataFrame({
        "state": g.mean().index,
        "state_y_mean": g.mean().values,
        "state_y_med": g.median().values,
        "state_y_std": g.std(ddof=0).fillna(0.0).values,
    })
    train = train.merge(stat, on="state", how="left")
    test = test.merge(stat, on="state", how="left")
    for c in ["state_y_mean", "state_y_med", "state_y_std"]:
        fill = float(train[c].median())
        train[c] = train[c].astype(float).fillna(fill)
        test[c] = test[c].astype(float).fillna(fill)

    y_min = float(min(train["year"].min(), test["year"].min()))
    train["year0"] = train["year"].astype(float) - y_min
    test["year0"] = test["year"].astype(float) - y_min
    return train, test

def train_and_predict(root) -> Path:
    train_path, test_path, output_path = construct_real_path(root)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, test = _fe(train, test)

    numk = ("b", "i", "u", "f", "c")
    feats = [c for c in train.columns if c != "yield" and train[c].dtype.kind in numk]
    X, y = train[feats], np.log1p(train["yield"].astype(float).values)
    Xte = test[feats]

    years = train["year"].values
    uy = np.unique(years[~pd.isna(years)])
    if len(uy) >= 2:
        vyear = np.max(uy)
        tr_idx = years != vyear
        va_idx = years == vyear
    else:
        va_idx = train.sample(frac=0.12, random_state=42).index
        tr_idx = ~train.index.isin(va_idx)

    model = lgb.LGBMRegressor(
        n_estimators=10,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.2,
        random_state=42,
        feature_fraction_seed=42,
        bagging_seed=42,
        data_random_seed=42,
        n_jobs=-1,
    )

    callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=False)]
    if np.any(va_idx):
        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], eval_metric="l1", callbacks=callbacks)
    else:
        model.fit(X, y)

    pred = np.expm1(model.predict(Xte))
    pred = np.clip(pred, 0, None)

    out = test.copy()
    out["yield"] = pred
    out = out[["year", "month", "state", "yield"]]
    out_path = output_path / "submission.csv"
    out.to_csv(out_path, index=False)
    return out_path

# EVOLVE-BLOCK-END

def main() -> Path:
    root = r"D:\清华工程博士\C3I\daguan\agentic-rl\mle-openevolve\experiments\bean03\inputs"
    out_path = train_and_predict(root)
    return out_path


if __name__ == "__main__":
    main()
