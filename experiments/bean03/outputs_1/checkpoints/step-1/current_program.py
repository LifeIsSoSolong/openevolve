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
from typing import Dict, Tuple, List, Optional
import numpy as np
import lightgbm as lgb
import pandas as pd

def encode_state(df: pd.DataFrame, mapping: Dict[str, int] | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    if mapping is None:
        states = sorted(df["state"].unique())
        mapping = {s: i for i, s in enumerate(states)}
    df["state_enc"] = df["state"].map(mapping).fillna(-1).astype(int)
    return df, mapping

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(int)
    # crop-year like index (Oct=0 ... Sep=11)
    df["months_since_crop_start"] = np.where(m >= 10, m - 10, m + 2).astype(int)
    # cyclic month features
    ang = 2.0 * np.pi * (m - 1) / 12.0
    df["month_sin"] = np.sin(ang)
    df["month_cos"] = np.cos(ang)
    df["quarter"] = ((m - 1) // 3 + 1).astype(int)
    return df

def add_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: List[str],
    smooth: float = 20.0,
    name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    key = name or ("te_" + "_".join(cols))
    global_mean = train["yield"].mean()
    agg = train.groupby(cols, dropna=False)["yield"].agg(["mean", "count"]).reset_index()
    agg[key] = (agg["mean"] * agg["count"] + global_mean * smooth) / (agg["count"] + smooth)
    agg = agg[cols + [key]]
    train = train.merge(agg, on=cols, how="left")
    test = test.merge(agg, on=cols, how="left")
    train[key] = train[key].fillna(global_mean)
    test[key] = test[key].fillna(global_mean)
    return train, test

def train_and_predict(root) -> Path:
    train_path, test_path, output_path = construct_real_path(root)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, state2idx = encode_state(train)
    test, _ = encode_state(test, mapping=state2idx)
    train = add_time_features(train)
    test = add_time_features(test)

    # target encodings (train-only statistics)
    train, test = add_target_encoding(train, test, ["state"], smooth=30.0, name="te_state")
    train, test = add_target_encoding(train, test, ["state", "month"], smooth=15.0, name="te_state_month")
    train, test = add_target_encoding(train, test, ["state", "months_since_crop_start"], smooth=15.0, name="te_state_cropm")

    # simple interactions
    train["year2"] = train["year"].astype(float) ** 2
    test["year2"] = test["year"].astype(float) ** 2
    train["state_year"] = train["state_enc"].astype(float) * train["year"].astype(float)
    test["state_year"] = test["state_enc"].astype(float) * test["year"].astype(float)

    numeric_kinds = ("b", "i", "u", "f", "c")
    candidate_features = [c for c in train.columns if c != "yield"]
    features = [c for c in candidate_features if train[c].dtype.kind in numeric_kinds]

    # log-transform target to improve MAPE stability
    y = np.log1p(train["yield"].astype(float).values)

    # time-based validation (last year as validation)
    max_year = int(train["year"].max())
    is_val = train["year"].astype(int).values == max_year
    X_tr, y_tr = train.loc[~is_val, features], y[~is_val]
    X_va, y_va = train.loc[is_val, features], y[is_val]

    model = lgb.LGBMRegressor(
        n_estimators=8000,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        objective="regression",
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)] if len(X_va) else None,
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=250, verbose=False)] if len(X_va) else None,
    )

    pred_log = model.predict(test[features])
    test_pred = np.expm1(pred_log)
    test_pred = np.clip(test_pred, 0, None)

    test_out = test.copy()
    test_out["yield"] = test_pred
    test_out = test_out[["year", "month", "state", "yield"]]
    out_path = output_path / "submission.csv"
    test_out.to_csv(out_path, index=False)
    print(f"✅ 模型训练完成，预测结果已保存至: {out_path}")
    return out_path

# EVOLVE-BLOCK-END

def main() -> Path:
    root = r"D:\清华工程博士\C3I\daguan\agentic-rl\mle-openevolve\experiments\bean03\inputs"
    out_path = train_and_predict(root)
    return out_path


if __name__ == "__main__":
    main()
