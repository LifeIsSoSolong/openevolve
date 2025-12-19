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
        states = sorted(df["state"].astype(str).unique())
        mapping = {s: i for i, s in enumerate(states)}
    df["state_enc"] = df["state"].astype(str).map(mapping).fillna(-1).astype(int)
    return df, mapping

def months_since_crop_start(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(int)
    df["months_since_crop_start"] = np.where(m >= 10, m - 10, m + 2).astype(int)
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(float)
    ang = 2.0 * np.pi * (m - 1.0) / 12.0
    df["month_sin"] = np.sin(ang)
    df["month_cos"] = np.cos(ang)
    y = df["year"].astype(float)
    df["year_centered"] = y - y.mean()
    return df

def add_target_encodings(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train.copy(), test.copy()
    gmean = float(tr["yield"].mean())

    def merge_mean(keys, name):
        m = tr.groupby(keys, dropna=False)["yield"].mean().rename(name).reset_index()
        return m

    for keys, name in [
        (["state_enc"], "te_state"),
        (["state_enc", "month"], "te_state_month"),
        (["state_enc", "year"], "te_state_year"),
    ]:
        mdf = merge_mean(keys, name)
        tr = tr.merge(mdf, on=keys, how="left")
        te = te.merge(mdf, on=keys, how="left")
        tr[name] = tr[name].fillna(gmean)
        te[name] = te[name].fillna(gmean)

    return tr, te

def train_and_predict(root) -> Path:
    train_path, test_path, output_path = construct_real_path(root)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, state2idx = encode_state(train)
    test, _ = encode_state(test, mapping=state2idx)

    train = months_since_crop_start(train)
    test = months_since_crop_start(test)
    train = add_time_features(train)
    test = add_time_features(test)

    train, test = add_target_encodings(train, test)

    numeric_kinds = ("b", "i", "u", "f", "c")
    features = [c for c in train.columns if c != "yield" and train[c].dtype.kind in numeric_kinds]

    # time-aware split by year (fallback to random split if only one year)
    years = sorted(train["year"].dropna().unique().tolist())
    if len(years) >= 2:
        n_val = min(2, max(1, len(years) // 4))
        val_years = set(years[-n_val:])
        tr_idx = ~train["year"].isin(val_years)
        va_idx = train["year"].isin(val_years)
    else:
        rs = np.random.RandomState(42)
        mask = rs.rand(len(train)) < 0.8
        tr_idx, va_idx = mask, ~mask

    y_tr = np.log1p(np.clip(train.loc[tr_idx, "yield"].to_numpy(dtype=float), 0, None))
    y_va = np.log1p(np.clip(train.loc[va_idx, "yield"].to_numpy(dtype=float), 0, None))

    model = lgb.LGBMRegressor(
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=127,
        min_data_in_leaf=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train.loc[tr_idx, features],
        y_tr,
        eval_set=[(train.loc[va_idx, features], y_va)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
    )

    pred_log = model.predict(test[features])
    test_pred = np.expm1(pred_log)
    test_pred = np.clip(test_pred, 0, None)

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
