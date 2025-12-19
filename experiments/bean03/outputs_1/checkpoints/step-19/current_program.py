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
import pandas as pd
import lightgbm as lgb


def encode_state(df: pd.DataFrame, mapping: Dict[str, int] | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    if mapping is None:
        states = sorted(df["state"].astype(str).unique())
        mapping = {s: i for i, s in enumerate(states)}
    df["state_enc"] = df["state"].astype(str).map(mapping).fillna(-1).astype(int)
    return df, mapping


def add_features(df: pd.DataFrame, min_year: int) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(int)
    y = df["year"].astype(int)
    df["months_since_crop_start"] = np.where(m >= 10, m - 10, m + 2).astype(int)
    ang = 2.0 * np.pi * (m - 1) / 12.0
    df["month_sin"] = np.sin(ang).astype(np.float32)
    df["month_cos"] = np.cos(ang).astype(np.float32)
    df["quarter"] = ((m - 1) // 3 + 1).astype(int)
    df["year_rel"] = (y - int(min_year)).astype(np.float32)
    df["state_x_phase"] = (df["state_enc"] * df["months_since_crop_start"]).astype(np.float32)
    return df


def add_loo_te(train: pd.DataFrame, test: pd.DataFrame, keys: List[str], name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-out target encoding for train; mean encoding for test."""
    gmean = float(train["yield"].mean())
    agg = train.groupby(keys, dropna=False)["yield"].agg(_sum="sum", _cnt="count").reset_index()
    tr = train.merge(agg, on=keys, how="left")
    te = test.merge(agg, on=keys, how="left")
    tr[name] = np.where(tr["_cnt"] > 1, (tr["_sum"] - tr["yield"]) / (tr["_cnt"] - 1), gmean)
    te[name] = (te["_sum"] / te["_cnt"]).astype(float)
    tr[name] = tr[name].fillna(gmean).astype(np.float32)
    te[name] = te[name].fillna(gmean).astype(np.float32)
    tr = tr.drop(columns=["_sum", "_cnt"])
    te = te.drop(columns=["_sum", "_cnt"])
    return tr, te


def train_and_predict(root) -> Path:
    train_path, test_path, output_path = construct_real_path(root)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, s2i = encode_state(train)
    test, _ = encode_state(test, mapping=s2i)

    min_year = int(train["year"].min())
    train = add_features(train, min_year=min_year)
    test = add_features(test, min_year=min_year)

    # leakage-controlled encodings
    train, test = add_loo_te(train, test, ["state_enc"], "te_state")
    train, test = add_loo_te(train, test, ["state_enc", "month"], "te_state_month")
    train, test = add_loo_te(train, test, ["state_enc", "months_since_crop_start"], "te_state_phase")

    numeric_kinds = ("b", "i", "u", "f", "c")
    features = [c for c in train.columns if c != "yield" and train[c].dtype.kind in numeric_kinds]

    X = train[features].copy()
    X_test = test[features].copy()
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X_test = X_test.fillna(med)

    y = train["yield"].astype(float).values
    cat_feats = [c for c in ["state_enc", "month", "months_since_crop_start", "quarter"] if c in features]

    params = dict(
        n_estimators=3600,
        learning_rate=0.025,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.7,
        random_state=42,
        n_jobs=-1,
    )

    m1 = lgb.LGBMRegressor(**params)
    m1.fit(X, y, categorical_feature=cat_feats if len(cat_feats) else "auto")

    m2 = lgb.LGBMRegressor(**params)
    m2.fit(X, np.log1p(y), categorical_feature=cat_feats if len(cat_feats) else "auto")

    p1 = m1.predict(X_test)
    p2 = np.expm1(m2.predict(X_test))
    pred = np.clip(0.55 * p1 + 0.45 * p2, 0.0, None)

    test_out = test.copy()
    test_out["yield"] = pred
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
