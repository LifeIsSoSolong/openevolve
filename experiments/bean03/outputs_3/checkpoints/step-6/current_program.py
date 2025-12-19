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
import lightgbm as lgb
import pandas as pd

def encode_state(df: pd.DataFrame, mapping: Dict[str, int] | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    s = df["state"].astype(str)
    if mapping is None:
        states = sorted(s.unique())
        mapping = {k: i for i, k in enumerate(states)}
    df["state_enc"] = s.map(mapping).fillna(-1).astype(int).astype("category")
    return df, mapping

def months_since_crop_start(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(int)
    df["months_since_crop_start"] = (m - 10).where(m >= 10, m + 2).astype(int)
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

    drop_cols = {"yield", "state"}
    features = [c for c in train.columns if c not in drop_cols]

    model = lgb.LGBMRegressor(
        n_estimators=1800,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=25,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[features], train["yield"].astype(float), categorical_feature=["state_enc"])

    test_pred = model.predict(test[features])
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
