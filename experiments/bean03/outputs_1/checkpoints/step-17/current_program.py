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
    df["state_x_year"] = (df["state_enc"].astype(np.float32) * df["year_rel"]).astype(np.float32)
    return df


def add_target_encoding(
    train: pd.DataFrame, test: pd.DataFrame, cols: List[str], name: str, smooth: float = 20.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple smoothed mean encoding computed on train, applied to both train/test."""
    global_mean = float(train["yield"].mean())
    agg = train.groupby(cols, dropna=False)["yield"].agg(["mean", "count"]).reset_index()
    agg[name] = (agg["mean"] * agg["count"] + global_mean * smooth) / (agg["count"] + smooth)
    agg = agg[cols + [name]]
    train = train.merge(agg, on=cols, how="left")
    test = test.merge(agg, on=cols, how="left")
    train[name] = train[name].fillna(global_mean).astype(np.float32)
    test[name] = test[name].fillna(global_mean).astype(np.float32)
    return train, test


def train_and_predict(root) -> Path:
    train_path, test_path, output_path = construct_real_path(root)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, s2i = encode_state(train)
    test, _ = encode_state(test, mapping=s2i)

    min_year = int(train["year"].min())
    train = add_features(train, min_year=min_year)
    test = add_features(test, min_year=min_year)

    # target encodings (often strong with state/month IDs)
    train, test = add_target_encoding(train, test, ["state"], "te_state", smooth=30.0)
    train, test = add_target_encoding(train, test, ["month"], "te_month", smooth=30.0)
    train, test = add_target_encoding(train, test, ["state", "month"], "te_state_month", smooth=15.0)
    train, test = add_target_encoding(train, test, ["state", "months_since_crop_start"], "te_state_cropm", smooth=15.0)

    numeric_kinds = ("b", "i", "u", "f", "c")
    features = [c for c in train.columns if c != "yield" and train[c].dtype.kind in numeric_kinds]

    X = train[features].copy()
    X_test = test[features].copy()
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X_test = X_test.fillna(med)

    y = train["yield"].astype(float).values

    # time-based validation: last year
    years = train["year"].astype(int).values
    max_year = int(years.max())
    is_val = years == max_year
    has_val = bool(is_val.any() and (~is_val).any())

    cat_feats = [c for c in ["state_enc", "month", "months_since_crop_start", "quarter"] if c in features]

    def fit_one(seed: int, use_log: bool):
        yt = np.log1p(y) if use_log else y
        model = lgb.LGBMRegressor(
            n_estimators=12000,
            learning_rate=0.02,
            num_leaves=127,
            min_child_samples=20,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.3,
            random_state=seed,
            n_jobs=-1,
        )
        if has_val:
            X_tr, y_tr = X.loc[~is_val], yt[~is_val]
            X_va, y_va = X.loc[is_val], yt[is_val]
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="l2",
                categorical_feature=cat_feats if len(cat_feats) else "auto",
                callbacks=[lgb.early_stopping(stopping_rounds=400, verbose=False)],
            )
        else:
            model.fit(X, yt, categorical_feature=cat_feats if len(cat_feats) else "auto")
        return model, use_log

    models = [fit_one(42, False), fit_one(52, True)]
    preds = []
    for m, use_log in models:
        it = getattr(m, "best_iteration_", None)
        p = m.predict(X_test, num_iteration=it)
        if use_log:
            p = np.expm1(p)
        preds.append(p)

    pred = np.mean(np.vstack(preds), axis=0)
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
