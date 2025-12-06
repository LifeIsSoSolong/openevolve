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
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

try:
    import xgboost as xgb
except Exception:  # xgboost may not be installed
    xgb = None  # type: ignore[assignment]


def encode_state(df: pd.DataFrame, mapping: Dict[str, int] | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    if mapping is None:
        mapping = {s: i for i, s in enumerate(sorted(df["state"].unique()))}
    df["state_enc"] = df["state"].map(mapping).fillna(-1).astype(int)
    return df, mapping


def months_since_crop_start(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple crop-phase index plus cyclical month features."""
    df = df.copy()
    month = df["month"].astype(float)
    df["months_since_crop_start"] = (month + 2.0) % 12.0
    angle = month * (2.0 * np.pi / 12.0)
    df["month_sin"] = np.sin(angle)
    df["month_cos"] = np.cos(angle)
    return df


def train_and_predict() -> Path:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train, state2idx = encode_state(train)
    test, _ = encode_state(test, mapping=state2idx)
    train = months_since_crop_start(train)
    test = months_since_crop_start(test)

    numeric_kinds = ("b", "i", "u", "f", "c")
    features = [c for c in train.columns if c != "yield" and train[c].dtype.kind in numeric_kinds]

    X_train = train[features].astype(np.float32)
    y = train["yield"].astype(float).to_numpy()
    X_test = test[features].astype(np.float32)

    # For linear models we apply simple median imputation to avoid NaN issues
    X_train_lin_df = train[features].copy()
    feature_medians = X_train_lin_df.median().fillna(0.0)
    X_train_lin = X_train_lin_df.fillna(feature_medians).to_numpy(dtype=np.float32)
    X_test_lin = test[features].copy().fillna(feature_medians).to_numpy(dtype=np.float32)

    if xgb is not None:
        # Gradient boosting model (XGBoost) with CV-based early stopping
        params = dict(
            objective="reg:squarederror",
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="rmse",
        )
        best_n = 1200
        if X_train.shape[0] >= 2:
            dtrain = xgb.DMatrix(X_train, label=y)
            cv_res = xgb.cv(
                params,
                dtrain,
                num_boost_round=2000,
                nfold=min(5, X_train.shape[0]),
                shuffle=True,
                early_stopping_rounds=100,
                verbose_eval=False,
                seed=42,
            )
            if len(cv_res) > 0:
                best_n = int(len(cv_res))
        xgb_model = xgb.XGBRegressor(
            n_estimators=best_n,
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            objective=params["objective"],
            random_state=42,
            n_jobs=-1,
            tree_method=params["tree_method"],
        )
        xgb_model.fit(X_train, y)

        # Simple linear ridge model on median-imputed features
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(X_train_lin, y)

        # Learn optimal convex blending weight on training data (minimize MSE)
        xgb_train_pred = xgb_model.predict(X_train)
        ridge_train_pred = ridge.predict(X_train_lin)
        diff = xgb_train_pred - ridge_train_pred
        denom = float(np.dot(diff, diff))
        if denom > 0.0:
            num = float(np.dot(diff, y - ridge_train_pred))
            w = max(0.0, min(1.0, num / denom))
        else:
            w = 1.0

        xgb_test_pred = xgb_model.predict(X_test)
        ridge_test_pred = ridge.predict(X_test_lin)
        test_pred = w * xgb_test_pred + (1.0 - w) * ridge_test_pred
        model_name = f"XGB+Ridge(w={w:.3f})"
    else:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=1000,
            max_leaf_nodes=63,
            random_state=42,
        )
        model.fit(X_train, y)
        test_pred = model.predict(X_test)
        model_name = "HistGradientBoostingRegressor"

    test_pred = np.maximum(test_pred, 0.0)

    test_out = test.copy()
    test_out["yield"] = test_pred
    test_out = test_out[["year", "month", "state", "yield"]]
    out_path = OUTPUT_DIR / "submission.csv"
    test_out.to_csv(out_path, index=False)
    print(f"模型训练完成，使用模型: {model_name}，预测结果已保存至: {out_path}")
    return out_path

# EVOLVE-BLOCK-END

def main() -> Path:
    out_path = train_and_predict()
    print(f"Model trained; predictions saved to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
