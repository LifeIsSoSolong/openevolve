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
import pandas as pd
import lightgbm as lgb


def encode_state(df: pd.DataFrame, mapping: Dict[str, int] | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    if mapping is None:
        states = sorted(df["state"].astype(str).unique())
        mapping = {s: i for i, s in enumerate(states)}
    df["state_enc"] = df["state"].astype(str).map(mapping).fillna(-1).astype(int)
    return df, mapping


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(int)
    df["months_since_crop_start"] = np.where(m >= 10, m - 10, m + 2).astype(int)
    ang = 2.0 * np.pi * (m - 1) / 12.0
    df["month_sin"] = np.sin(ang)
    df["month_cos"] = np.cos(ang)
    df["quarter"] = ((m - 1) // 3 + 1).astype(int)
    df["state_x_phase"] = (df["state_enc"] * df["months_since_crop_start"]).astype(float)
    if "year" in df.columns:
        yc = df["year"].astype(float) - 2000.0
        df["year_centered"] = yc
        df["year2"] = yc * yc
    return df


def train_and_predict(root) -> Path:
    train_path, test_path, output_path = construct_real_path(root)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, s2i = encode_state(train)
    test, _ = encode_state(test, mapping=s2i)
    train = add_features(train)
    test = add_features(test)

    # leakage-reduced target encoding (leave-one-out for train, smoothed mean for test)
    y = train["yield"].astype(float)
    global_mean = float(y.mean())
    alpha = 30.0

    def _add_te(cols, name: str):
        g = train.groupby(cols)["yield"]
        s = g.transform("sum")
        c = g.transform("count")
        te_tr = (s - y + alpha * global_mean) / (c - 1.0 + alpha)
        te_tr = te_tr.where(c > 1, global_mean)
        train[name] = te_tr.astype(float)

        agg = train.groupby(cols)["yield"].agg(["sum", "count"]).reset_index()
        agg[name] = (agg["sum"] + alpha * global_mean) / (agg["count"] + alpha)
        test[name] = (
            test[cols].merge(agg[cols + [name]], on=cols, how="left")[name].fillna(global_mean).astype(float)
        )

    _add_te(["state_enc"], "te_state")
    _add_te(["month"], "te_month")
    _add_te(["state_enc", "month"], "te_state_month")

    numeric_kinds = ("b", "i", "u", "f", "c")
    features = [c for c in train.columns if c != "yield" and train[c].dtype.kind in numeric_kinds]

    X = train[features].copy()
    X_test = test[features].copy()

    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X_test = X_test.fillna(med)

    y_log = np.log1p(np.clip(y.to_numpy(), 0.0, None))

    cat_feats = [c for c in ["state_enc", "month", "months_since_crop_start", "quarter"] if c in features]

    params = dict(
        objective="regression",
        n_estimators=6000,
        learning_rate=0.02,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    preds_log = np.zeros(len(X_test), dtype=float)
    for sd in (42, 202):
        model = lgb.LGBMRegressor(**params, random_state=sd)
        model.fit(X, y_log, categorical_feature=cat_feats)
        preds_log += model.predict(X_test)

    pred = np.expm1(preds_log / 2.0)
    # small blend with strong seasonal/location prior
    prior = test["te_state_month"].to_numpy()
    pred = 0.85 * pred + 0.15 * prior
    pred = np.clip(pred, 0.0, None)

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
