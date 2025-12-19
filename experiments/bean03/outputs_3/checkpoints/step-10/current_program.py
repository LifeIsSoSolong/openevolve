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
from sklearn.model_selection import train_test_split

def encode_state(df: pd.DataFrame, mapping: Dict[str, int] | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    s = df["state"].astype(str)
    if mapping is None:
        keys = sorted(s.unique())
        mapping = {k: i for i, k in enumerate(keys)}
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

def add_loo_target_enc(
    train: pd.DataFrame, test: pd.DataFrame, y: pd.Series, cols: List[str], smooth: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train.copy(), test.copy()
    key = tr[cols].astype(str).agg("|".join, axis=1)
    key_te = te[cols].astype(str).agg("|".join, axis=1)
    global_mean = float(y.mean())

    grp = pd.DataFrame({"k": key.values, "y": y.values}).groupby("k")["y"].agg(["sum", "count"])
    ssum, cnt = grp["sum"].to_dict(), grp["count"].to_dict()

    def tr_enc(k, yi):
        c = cnt.get(k, 0)
        if c <= 1:
            return global_mean
        return (ssum[k] - yi + global_mean * smooth) / (c - 1 + smooth)

    enc_tr = [tr_enc(k, yi) for k, yi in zip(key.values, y.values)]
    enc_te = [(ssum[k] + global_mean * smooth) / (cnt[k] + smooth) if k in cnt else global_mean for k in key_te.values]

    name = "te_" + "_".join(cols)
    tr[name] = np.asarray(enc_tr, float)
    te[name] = np.asarray(enc_te, float)
    return tr, te

def train_and_predict(root) -> Path:
    # do not change this line
    train_path, test_path, output_path = construct_real_path(root)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, state2idx = encode_state(train)
    test, _ = encode_state(test, mapping=state2idx)

    train = add_time_features(months_since_crop_start(train))
    test = add_time_features(months_since_crop_start(test))

    y = train["yield"].astype(float)

    # leakage-safe (LOO) target encodings
    train, test = add_loo_target_enc(train, test, y, cols=["state"], smooth=30.0)
    train, test = add_loo_target_enc(train, test, y, cols=["state", "month"], smooth=50.0)

    numeric_kinds = ("b", "i", "u", "f", "c")
    features = [c for c in train.columns if c != "yield" and train[c].dtype.kind in numeric_kinds]

    med = train[features].median(numeric_only=True)
    X = train[features].fillna(med)
    X_test = test[features].fillna(med)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, random_state=42)

    model = lgb.LGBMRegressor(
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.8,
        random_state=42,
        objective="regression",
        n_jobs=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        categorical_feature=["state_enc"],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
    )

    test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
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
