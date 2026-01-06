from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

try:  # pragma: no cover
    from lightgbm import LGBMRegressor
except ModuleNotFoundError:  # pragma: no cover
    LGBMRegressor = None  # type: ignore


LOGGER = logging.getLogger(__name__)


# construct real path from judge.py's ROOT
def construct_real_path(root):
    PROJECT_ROOT = Path(root)
    INPUT_DIR = PROJECT_ROOT
    TRAIN_PATH = INPUT_DIR / "train.csv"
    TEST_PATH = INPUT_DIR / "test.csv"

    return TRAIN_PATH, TEST_PATH


# EVOLVE-BLOCK-START
def encode_state(df: pd.DataFrame, mapping: dict[str, int] | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    if mapping is None:
        mapping = {state: idx for idx, state in enumerate(sorted(df["state"].astype(str).unique()))}
    df["state_enc"] = df["state"].map(mapping).fillna(-1).astype(int)
    return df, mapping


def months_since_crop_start(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["months_since_crop_start"] = df["month"].apply(lambda m: m - 10 if m >= 10 else m + 2)
    return df


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mean_y = float(np.mean(y_true))
    rrmse = float(rmse / mean_y * 100) if mean_y != 0 else float("nan")
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if np.any(mask) else float("nan")
    return {"rmse": rmse, "rrmse": rrmse, "mape": mape}


def main(root) -> tuple[pd.DataFrame, dict[str, float]]:
    # ---------- load ----------
    TRAIN_PATH, TEST_PATH = construct_real_path(root)

    LOGGER.info("Loading train: %s", TRAIN_PATH)
    LOGGER.info("Loading test : %s", TEST_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target = "yield"

    # ---------- preprocess (keep behavior identical) ----------
    train_df, state2idx = encode_state(train_df)
    test_df, _ = encode_state(test_df, mapping=state2idx)

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        df = months_since_crop_start(df)
        for col in ["month_cat", "state_cat", "year_cat"]:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes
        return df

    train_df = _apply(train_df)
    test_df = _apply(test_df)

    features = [c for c in train_df.columns if c not in {target, "state"}]

    # ---------- drop NaN targets in train ----------
    before_drop = len(train_df)
    train_df = train_df.dropna(subset=[target])
    if len(train_df) < before_drop:
        LOGGER.info("Dropped %d rows with missing target from training set", before_drop - len(train_df))

    # ---------- train (same params) ----------
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(train_df[features], train_df[target])

    # ---------- validate (same logic) ----------
    metrics: dict[str, float] = {"rmse": float("nan"), "rrmse": float("nan"), "mape": float("nan")}

    if target in test_df.columns and len(test_df):
        before_drop = len(test_df)
        test_df2 = test_df.dropna(subset=[target])
        if len(test_df2) < before_drop:
            LOGGER.info("Dropped %d rows with missing target from test set", before_drop - len(test_df2))
        if len(test_df2):
            val_pred = model.predict(test_df2[features])
            m = _eval_metrics(test_df2[target].to_numpy(), val_pred)
            metrics.update({"rmse": m["rmse"], "rrmse": m["rrmse"], "mape": m["mape"]})
            LOGGER.info(
                "Test -> RMSE: %.6f | rRMSE: %.6f%% | MAPE: %.6f%%",
                metrics["rmse"],
                metrics["rrmse"],
                metrics["mape"],
            )
        else:
            LOGGER.info("Test set has target column but all targets are NaN; skip metrics")
    else:
        LOGGER.info("Test set has no target column '%s'; skip val metrics", target)

    return metrics
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    root = r"D:\清华工程博士\C3I\AutoMLAgent\openevolve\experiments\bean05\inputs"

    metrics = main(root)
    print(metrics)