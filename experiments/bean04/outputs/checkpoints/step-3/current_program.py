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
    OUTPUT_DIR = PROJECT_ROOT.parent / "outputs" / "submission"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_PATH = INPUT_DIR / "train.csv"
    VAL_PATH = INPUT_DIR / "val.csv"
    TEST_PATH = INPUT_DIR / "test.csv"
    SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

    return TRAIN_PATH, VAL_PATH, TEST_PATH, SUBMISSION_PATH



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
    TRAIN_PATH, VAL_PATH, TEST_PATH, _ = construct_real_path(root)

    LOGGER.info("Loading train: %s", TRAIN_PATH)
    LOGGER.info("Loading val  : %s", VAL_PATH)
    LOGGER.info("Loading test : %s", TEST_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target = "yield"

    # ---------- preprocess ----------
    train_df, state2idx = encode_state(train_df)
    val_df, _ = encode_state(val_df, mapping=state2idx)
    test_df, _ = encode_state(test_df, mapping=state2idx)

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        df = months_since_crop_start(df)
        # cyclical month features (helps trees across year boundary)
        df["m_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["m_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["msc_sin"] = np.sin(2 * np.pi * df["months_since_crop_start"] / 12.0)
        df["msc_cos"] = np.cos(2 * np.pi * df["months_since_crop_start"] / 12.0)
        # mild trend capacity
        if "year" in df.columns:
            df["year2"] = df["year"].astype(float) ** 2
        for col in ["month_cat", "state_cat", "year_cat"]:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes
        return df

    train_df = _apply(train_df)
    val_df = _apply(val_df)
    test_df = _apply(test_df)

    # ---------- drop NaN targets in train ----------
    before_drop = len(train_df)
    train_df = train_df.dropna(subset=[target])
    if len(train_df) < before_drop:
        LOGGER.info("Dropped %d rows with missing target from training set", before_drop - len(train_df))

    # ---------- smoothed target encoding (train-only) ----------
    global_mean = float(train_df[target].mean())
    k = 30.0

    def _add_te(df: pd.DataFrame, keys: list[str], out_col: str) -> pd.DataFrame:
        agg = train_df.groupby(keys, dropna=False)[target].agg(["mean", "count"]).reset_index()
        df = df.merge(agg, on=keys, how="left")
        cnt = df["count"].fillna(0.0)
        mu = df["mean"].fillna(global_mean)
        df[out_col] = (cnt * mu + k * global_mean) / (cnt + k)
        return df.drop(columns=["mean", "count"])

    for keys, out_col in [
        (["state_enc"], "te_state"),
        (["month"], "te_month"),
        (["year"], "te_year"),
        (["state_enc", "month"], "te_state_month"),
        (["state_enc", "year"], "te_state_year"),
    ]:
        train_df = _add_te(train_df, keys, out_col)
        val_df = _add_te(val_df, keys, out_col)
        test_df = _add_te(test_df, keys, out_col)

    features = [c for c in train_df.columns if c not in {target, "state"}]

    # ---------- train (early stopping on val when possible) ----------
    try:  # pragma: no cover
        from lightgbm import early_stopping, log_evaluation
    except Exception:  # pragma: no cover
        early_stopping = None  # type: ignore
        log_evaluation = None  # type: ignore

    model = LGBMRegressor(
        n_estimators=8000,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=25,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=1.5,
        random_state=42,
    )

    X_tr = train_df[features]
    y_tr = train_df[target].to_numpy()

    callbacks = []
    if early_stopping is not None:
        callbacks.append(early_stopping(stopping_rounds=400, verbose=False))
    if log_evaluation is not None:
        callbacks.append(log_evaluation(period=0))

    if target in val_df.columns:
        val_fit = val_df.dropna(subset=[target])
        if len(val_fit):
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(val_fit[features], val_fit[target].to_numpy())],
                eval_metric="rmse",
                callbacks=callbacks if callbacks else None,
            )
        else:
            model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)

    # ---------- validate (same logic) ----------
    metrics: dict[str, float] = {"val_rmse": float("nan"), "val_rrmse": float("nan"), "val_mape": float("nan")}

    if target in val_df.columns and len(val_df):
        before_val_drop = len(val_df)
        val_df2 = val_df.dropna(subset=[target])
        if len(val_df2) < before_val_drop:
            LOGGER.info("Dropped %d rows with missing target from validation set", before_val_drop - len(val_df2))
        if len(val_df2):
            val_pred = np.clip(model.predict(val_df2[features]), 0, None)
            m = _eval_metrics(val_df2[target].to_numpy(), val_pred)
            metrics.update({"val_rmse": m["rmse"], "val_rrmse": m["rrmse"], "val_mape": m["mape"]})
            LOGGER.info(
                "VAL -> RMSE: %.6f | rRMSE: %.6f%% | MAPE: %.6f%%",
                metrics["val_rmse"],
                metrics["val_rrmse"],
                metrics["val_mape"],
            )
        else:
            LOGGER.info("Validation set has target column but all targets are NaN; skip metrics")
    else:
        LOGGER.info("Validation set has no target column '%s'; skip val metrics", target)

    # ---------- inference + submission ----------
    preds = np.clip(model.predict(test_df[features]), 0, None)
    submission = test_df[["year", "month", "state"]].copy()
    submission[target] = preds

    return submission, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    root = r"D:\清华工程博士\C3I\daguan\agentic-rl\mle-openevolve\experiments\bean04\baseline\2months"

    submission, metrics = main(root)

    _, _, _, SUBMISSION_PATH = construct_real_path(root)
    submission.to_csv(SUBMISSION_PATH, index=False)
    LOGGER.info("Saved submission -> %s", SUBMISSION_PATH)
    LOGGER.info("Metrics dict -> %s", metrics)