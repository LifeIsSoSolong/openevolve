"""
Baseline model for bean_test exp2.

Reads ./input/train.csv and ./input/test.csv, performs simple preprocessing
(state encoding + months_since_crop_start), trains a LightGBM regressor, and
writes predictions to ./output/submission.csv with columns [year, month, state, yield].

The EVOLVE-BLOCK marks the scope that OpenEvolve is allowed to tune.
"""

from __future__ import annotations

import os
from pathlib import Path

# Paths
# PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(r"D:\清华工程博士\C3I\AutoMLAgent\openevolve\bean_test\exp3")
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = INPUT_DIR / "train.csv"
TEST_PATH = INPUT_DIR / "test.csv"

# EVOLVE-BLOCK-START
from typing import Dict, Tuple
import lightgbm as lgb
import pandas as pd

def encode_state(df: pd.DataFrame, mapping: Dict[str, int] | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Encode state column to integer IDs."""
    df = df.copy()
    if mapping is None:
        states = sorted(df["state"].unique())
        mapping = {s: i for i, s in enumerate(states)}
    df["state_enc"] = df["state"].map(mapping).fillna(-1).astype(int)
    return df, mapping


def months_since_crop_start(df: pd.DataFrame) -> pd.DataFrame:
    """Map month to a simple crop-phase index."""
    df = df.copy()

    def transform(m: int) -> int:
        return m - 10 if m >= 10 else m + 2

    df["months_since_crop_start"] = df["month"].apply(transform)
    return df


def train_and_predict() -> Path:

    # ---------- read ----------
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # ---------- encode & transform ----------
    train, state2idx = encode_state(train)
    test, _ = encode_state(test, mapping=state2idx)
    train = months_since_crop_start(train)
    test = months_since_crop_start(test)

    # ---------- feature selection ----------
    numeric_kinds = ("b", "i", "u", "f", "c")
    candidate_features = [col for col in train.columns if col != "yield"]
    features = [col for col in candidate_features if train[col].dtype.kind in numeric_kinds]
    target = "yield"

    # ---------- train ----------
    
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(train[features], train[target])

    # ---------- predict ----------
    test_pred = model.predict(test[features])

    # ---------- output ----------
    test_out = test.copy()
    test_out["yield"] = test_pred
    test_out = test_out[["year", "month", "state", "yield"]]
    out_path = OUTPUT_DIR / "submission.csv"
    test_out.to_csv(out_path, index=False)
    print(f"✅ 模型训练完成，预测结果已保存至: {out_path}")
    return out_path

# EVOLVE-BLOCK-END

def main() -> Path:
    out_path = train_and_predict()
    print(f"Model trained; predictions saved to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
