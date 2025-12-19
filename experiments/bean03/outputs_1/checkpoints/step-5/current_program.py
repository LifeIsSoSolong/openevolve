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
    df["state"] = df["state"].astype(str)
    df["state_enc"] = df["state"].map(mapping).fillna(-1).astype(int)
    return df, mapping

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = df["month"].astype(int)
    df["months_since_crop_start"] = np.where(m >= 10, m - 10, m + 2).astype(int)
    ang = 2.0 * np.pi * (m - 1) / 12.0
    df["month_sin"] = np.sin(ang).astype(np.float32)
    df["month_cos"] = np.cos(ang).astype(np.float32)
    df["quarter"] = ((m - 1) // 3 + 1).astype(int)
    return df

def add_past_mean_by_year(
    train: pd.DataFrame, test: pd.DataFrame, keys: List[str], name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leakage-safe per-row mean: for train uses ONLY years < current year; for test uses all train years."""
    global_mean = float(train["yield"].mean())

    yearly = (
        train.groupby(keys + ["year"], dropna=False)["yield"]
        .mean()
        .reset_index(name="_y")
        .sort_values("year")
    )
    # expanding mean per key, shifted so current year is not used
    yearly[name] = yearly.groupby(keys)["_y"].transform(lambda s: s.expanding().mean().shift(1))
    train = train.merge(yearly[keys + ["year", name]], on=keys + ["year"], how="left")
    train[name] = train[name].fillna(global_mean)

    # test: mean over all available years in train (fallback to global mean)
    agg_all = train.groupby(keys, dropna=False)["yield"].mean().reset_index(name=name)
    test = test.merge(agg_all, on=keys, how="left")
    test[name] = test[name].fillna(global_mean)
    return train, test

def train_and_predict(root) -> Path:
    train_path, test_path, output_path = construct_real_path(root)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train, state2idx = encode_state(train)
    test, _ = encode_state(test, mapping=state2idx)
    train = add_time_features(train)
    test = add_time_features(test)

    # leakage-safe historical means
    train, test = add_past_mean_by_year(train, test, ["state"], "pm_state")
    train, test = add_past_mean_by_year(train, test, ["state", "month"], "pm_state_month")
    train, test = add_past_mean_by_year(train, test, ["state", "months_since_crop_start"], "pm_state_cropm")

    # light interactions
    train["state_year"] = train["state_enc"].astype(np.float32) * train["year"].astype(np.float32)
    test["state_year"] = test["state_enc"].astype(np.float32) * test["year"].astype(np.float32)

    numeric_kinds = ("b", "i", "u", "f", "c")
    candidate_features = [c for c in train.columns if c != "yield"]
    features = [c for c in candidate_features if train[c].dtype.kind in numeric_kinds]

    # --------- CV training (GroupKFold by year) + log1p target ----------
    from sklearn.model_selection import GroupKFold

    X = train[features].copy()
    X_te = test[features].copy()

    # robust missing handling
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X_te = X_te.fillna(med)

    y = np.log1p(train["yield"].astype(float).clip(lower=0.0).values)
    groups = train["year"].astype(int).values if "year" in train.columns else np.arange(len(train))

    cat_feats = [c for c in ["state_enc", "month", "quarter", "months_since_crop_start"] if c in features]

    n_groups = len(np.unique(groups))
    n_splits = int(min(5, n_groups)) if n_groups >= 2 else 1

    params = dict(
        n_estimators=20000,
        learning_rate=0.02,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.6,
        random_state=42,
        objective="regression",
        n_jobs=-1,
    )

    if n_splits == 1:
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y, categorical_feature=cat_feats if len(cat_feats) else "auto")
        pred_log = model.predict(X_te)
        test_pred = np.expm1(pred_log)
    else:
        gkf = GroupKFold(n_splits=n_splits)
        pred_log_te = np.zeros(len(X_te), dtype=np.float64)

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups)):
            m = lgb.LGBMRegressor(**{**params, "random_state": 42 + fold * 17})
            m.fit(
                X.iloc[tr_idx],
                y[tr_idx],
                eval_set=[(X.iloc[va_idx], y[va_idx])],
                eval_metric="l2",
                categorical_feature=cat_feats if len(cat_feats) else "auto",
                callbacks=[lgb.early_stopping(stopping_rounds=400, verbose=False)],
            )
            it = getattr(m, "best_iteration_", None)
            pred_log_te += m.predict(X_te, num_iteration=it) / n_splits

        test_pred = np.expm1(pred_log_te)

    test_pred = np.clip(test_pred, 0.0, None)

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
