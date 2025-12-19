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


def add_features(df: pd.DataFrame, min_year: int) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(int)
    y = df["year"].astype(int)

    # crop-year phase (Oct=0..Sep=11)
    df["months_since_crop_start"] = np.where(m >= 10, m - 10, m + 2).astype(int)

    ang = 2.0 * np.pi * (m - 1) / 12.0
    df["month_sin"] = np.sin(ang).astype(np.float32)
    df["month_cos"] = np.cos(ang).astype(np.float32)
    df["quarter"] = ((m - 1) // 3 + 1).astype(int)

    # year scaling (helps generalization)
    df["year_rel"] = (y - int(min_year)).astype(np.float32)

    # light interactions
    df["state_x_phase"] = (df["state_enc"] * df["months_since_crop_start"]).astype(np.float32)
    df["state_x_year"] = (df["state_enc"].astype(np.float32) * df["year_rel"]).astype(np.float32)
    df["year_x_sin"] = (df["year_rel"] * df["month_sin"]).astype(np.float32)
    df["year_x_cos"] = (df["year_rel"] * df["month_cos"]).astype(np.float32)
    return df


def train_and_predict(root) -> Path:
    train_path, test_path, output_path = construct_real_path(root)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, s2i = encode_state(train)
    test, _ = encode_state(test, mapping=s2i)

    min_year = int(train["year"].min())
    train = add_features(train, min_year=min_year)
    test = add_features(test, min_year=min_year)

    numeric_kinds = ("b", "i", "u", "f", "c")
    features = [c for c in train.columns if c != "yield" and train[c].dtype.kind in numeric_kinds]

    X = train[features].copy()
    X_test = test[features].copy()

    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X_test = X_test.fillna(med)

    y = train["yield"].astype(float).values

    cat_feats = [c for c in ["state_enc", "month", "months_since_crop_start", "quarter"] if c in features]

    model = lgb.LGBMRegressor(
        n_estimators=2800,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, categorical_feature=cat_feats if len(cat_feats) else "auto")

    pred = np.clip(model.predict(X_test), 0.0, None)

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
