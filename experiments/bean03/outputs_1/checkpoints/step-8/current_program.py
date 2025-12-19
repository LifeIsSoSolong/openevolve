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
from sklearn.model_selection import KFold


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
    # mild year scaling (helps tree split stability if year is large)
    if "year" in df.columns:
        df["year_centered"] = (df["year"].astype(float) - 2000.0).astype(float)
    return df


def _target_encode_oof(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_cols: List[str],
    target_col: str,
    n_splits: int = 5,
    alpha: float = 20.0,
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series, str]:
    """
    Smoothed mean target encoding with OOF values for train to reduce leakage.
    te = (sum + alpha*global_mean) / (count + alpha)
    """
    global_mean = float(train_df[target_col].mean())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(train_df), dtype=float)

    for tr_idx, val_idx in kf.split(train_df):
        tr = train_df.iloc[tr_idx]
        agg = tr.groupby(group_cols)[target_col].agg(["sum", "count"]).reset_index()
        agg["te"] = (agg["sum"] + alpha * global_mean) / (agg["count"] + alpha)

        val = train_df.iloc[val_idx][group_cols].merge(agg[group_cols + ["te"]], on=group_cols, how="left")["te"]
        oof[val_idx] = val.fillna(global_mean).to_numpy()

    agg_full = train_df.groupby(group_cols)[target_col].agg(["sum", "count"]).reset_index()
    agg_full["te"] = (agg_full["sum"] + alpha * global_mean) / (agg_full["count"] + alpha)
    te_test = test_df[group_cols].merge(agg_full[group_cols + ["te"]], on=group_cols, how="left")["te"].fillna(global_mean)

    name = "te_" + "_".join(group_cols)
    return pd.Series(oof, index=train_df.index, name=name), te_test.rename(name), name


def train_and_predict(root) -> Path:
    train_path, test_path, output_path = construct_real_path(root)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, s2i = encode_state(train)
    test, _ = encode_state(test, mapping=s2i)

    train = add_features(train)
    test = add_features(test)

    # OOF target encodings (additive signal for location/seasonality)
    te_specs = [["state_enc"], ["month"], ["state_enc", "month"]]
    if "year" in train.columns:
        te_specs.append(["state_enc", "year"])
    te_feature_names: List[str] = []
    for cols in te_specs:
        tr_te, te_te, nm = _target_encode_oof(train, test, cols, "yield", n_splits=5, alpha=30.0, seed=42)
        train[nm] = tr_te
        test[nm] = te_te
        te_feature_names.append(nm)

    # numeric features (keep all original numeric cols + engineered)
    numeric_kinds = ("b", "i", "u", "f", "c")
    features = [c for c in train.columns if c != "yield" and train[c].dtype.kind in numeric_kinds]
    # ensure TE features are included even if dtype inference differs
    for nm in te_feature_names:
        if nm not in features:
            features.append(nm)

    X = train[features].copy()
    X_test = test[features].copy()

    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X_test = X_test.fillna(med)

    y = train["yield"].astype(float)
    y_log = np.log1p(np.clip(y.to_numpy(), 0.0, None))

    cat_feats = [c for c in ["state_enc", "month", "months_since_crop_start", "quarter"] if c in features]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=5000,
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
    model.fit(X, y_log, categorical_feature=cat_feats)

    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)
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
