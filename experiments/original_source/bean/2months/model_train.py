from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
try:  # pragma: no cover
    from lightgbm import LGBMRegressor
except ModuleNotFoundError:  # pragma: no cover
    LGBMRegressor = None  # type: ignore
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data_process" / "brazil_train_eval"
TRAIN_PATH = DATA_DIR / "train_2m.csv"
VAL_PATH = DATA_DIR / "val_2m.csv"
TEST_PATH = DATA_DIR / "test_2m_split.csv"

MODEL_DIR = ROOT_DIR / "model_checkpoints" / "dataagent_two_month"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_EXPORT_PATH = MODEL_DIR / "dataagent_two_month_model_latest.pkl"
SUBMISSION_PATH = MODEL_DIR / "dataagent_two_month_submission.csv"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "dataagent_two_month_feature_importance.csv"
METADATA_PATH = MODEL_DIR / "dataagent_two_month_model_latest.metadata.json"
DATA_PROCESS_PATH = ROOT_DIR / "model" / "DataAgent_model_two_month" / "data_process.py"
DATA_PROCESS_SNAPSHOT_DIR = MODEL_DIR / "data_process_snapshots"


def _relpath(path: Path | str) -> str:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    try:
        return str(path_obj.relative_to(ROOT_DIR))
    except ValueError:
        return str(path_obj)


@dataclass
class TrainingArtifacts:
    model: LGBMRegressor
    features: Sequence[str]
    metrics_summary: dict[str, float]
    fold_metrics: dict[str, list[float]]
    train_rows: int
    val_rows: int
    feature_importance: pd.DataFrame
    submission: pd.DataFrame


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        (path if path.suffix == "" else path.parent).mkdir(parents=True, exist_ok=True)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")
    if not VAL_PATH.exists():
        raise FileNotFoundError(f"Validation data not found: {VAL_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test data not found: {TEST_PATH}")
    LOGGER.info("Loading train dataset from %s", TRAIN_PATH)
    LOGGER.info("Loading validation dataset from %s", VAL_PATH)
    LOGGER.info("Loading test dataset from %s", TEST_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, val_df, test_df


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


def preprocess(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    train_df, state2idx = encode_state(train_df)
    if val_df is not None:
        val_df, _ = encode_state(val_df, mapping=state2idx)
    if test_df is not None:
        test_df, _ = encode_state(test_df, mapping=state2idx)

    def _apply_transforms(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        df = months_since_crop_start(df)
        categorical_cols = [col for col in ["month_cat", "state_cat", "year_cat"] if col in df.columns]
        for col in categorical_cols:
            df[col] = df[col].astype("category").cat.codes
        return df

    train_df = _apply_transforms(train_df)
    val_df = _apply_transforms(val_df)
    test_df = _apply_transforms(test_df)

    if train_df is None:
        raise ValueError("Training dataframe became None after preprocessing.")
    features = [col for col in train_df.columns if col not in {"yield", "state"}]
    return train_df, val_df, test_df, features


def run_cv(train_df: pd.DataFrame, features: Sequence[str], target: str) -> tuple[dict[str, float], dict[str, list[float]]]:
    if LGBMRegressor is None:
        raise ModuleNotFoundError("LightGBM is required for training the DataAgent model. Install 'lightgbm'.")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores: list[float] = []
    rrmse_scores: list[float] = []
    mape_scores: list[float] = []

    X = train_df[features]
    y = train_df[target].values

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        rrmse = float(rmse / np.mean(y_val) * 100)
        mask = y_val != 0
        mape = float(np.mean(np.abs((y_val[mask] - preds[mask]) / y_val[mask])) * 100) if np.any(mask) else float("nan")

        rmse_scores.append(rmse)
        rrmse_scores.append(rrmse)
        mape_scores.append(mape)
        LOGGER.info(
            "Fold %d metrics -> RMSE: %.3f | rRMSE: %.3f%% | MAPE: %s",
            fold,
            rmse,
            rrmse,
            f"{mape:.3f}%" if np.isfinite(mape) else "nan",
        )

    metrics_summary = {
        "cv_rmse_mean": float(np.nanmean(rmse_scores)),
        "cv_rmse_std": float(np.nanstd(rmse_scores)),
        "cv_rrmse_mean": float(np.nanmean(rrmse_scores)),
        "cv_rrmse_std": float(np.nanstd(rrmse_scores)),
        "cv_mape_mean": float(np.nanmean(mape_scores)),
        "cv_mape_std": float(np.nanstd(mape_scores)),
    }
    fold_metrics = {
        "rmse": rmse_scores,
        "rrmse": rrmse_scores,
        "mape": mape_scores,
    }
    return metrics_summary, fold_metrics


def train_final_model(train_df: pd.DataFrame, features: Sequence[str], target: str) -> LGBMRegressor:
    if LGBMRegressor is None:
        raise ModuleNotFoundError("LightGBM is required for training the DataAgent model. Install 'lightgbm'.")
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(train_df[features], train_df[target])
    return model


def generate_submission(model: LGBMRegressor, test_df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    preds = model.predict(test_df[features])
    output = test_df[["year", "month", "state"]].copy()
    output["yield"] = preds
    return output


def capture_data_process_snapshot() -> dict[str, str | bool]:
    if not DATA_PROCESS_PATH.exists():
        LOGGER.warning("Data process script not found at %s", DATA_PROCESS_PATH)
        return {"path": str(DATA_PROCESS_PATH), "exists": False}

    content = DATA_PROCESS_PATH.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = DATA_PROCESS_SNAPSHOT_DIR
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"data_process_{timestamp}.py"
    snapshot_path.write_bytes(content)
    return {
        "path": str(DATA_PROCESS_PATH),
        "exists": True,
        "sha256": sha256,
        "snapshot_path": str(snapshot_path),
        "captured_at": timestamp,
    }


def save_artifacts(
    artifacts: TrainingArtifacts,
    data_process_snapshot: dict[str, str | bool],
    training_duration_sec: float,
) -> dict[str, object]:
    ensure_dirs([MODEL_EXPORT_PATH, SUBMISSION_PATH, FEATURE_IMPORTANCE_PATH, METADATA_PATH])

    joblib.dump(artifacts.model, MODEL_EXPORT_PATH)
    LOGGER.info("Model exported to %s", MODEL_EXPORT_PATH)

    artifacts.feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    LOGGER.info("Feature importance saved to %s", FEATURE_IMPORTANCE_PATH)

    artifacts.submission.to_csv(SUBMISSION_PATH, index=False)
    LOGGER.info("Submission saved to %s", SUBMISSION_PATH)

    feature_importance_values = [float(val) for val in artifacts.model.feature_importances_.tolist()]
    feature_importance_records = [
        {"feature": str(item["feature"]), "importance": float(item["importance"])}
        for item in artifacts.feature_importance.to_dict(orient="records")
    ]

    trained_at = datetime.now(tz=timezone.utc).isoformat()

    metadata = {
        "train_csv": _relpath(TRAIN_PATH),
        "val_csv": _relpath(VAL_PATH),
        "test_csv": _relpath(TEST_PATH),
        "model_path": _relpath(MODEL_EXPORT_PATH),
        "feature_names": list(artifacts.features),
        "metrics": artifacts.metrics_summary,
        "fold_metrics": artifacts.fold_metrics,
        "training_rows": artifacts.train_rows,
        "validation_rows": artifacts.val_rows,
        "training_columns": len(artifacts.features),
        "feature_importance_path": _relpath(FEATURE_IMPORTANCE_PATH),
        "submission_path": _relpath(SUBMISSION_PATH),
        "feature_importance_values": feature_importance_values,
        "feature_importances": feature_importance_records,
        "data_process": data_process_snapshot,
        "note": "DataAgent two-month lead LightGBM",
        "generated_at": trained_at,
        "trained_at": trained_at,
        "training_duration_sec": training_duration_sec,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    LOGGER.info("Metadata saved to %s", METADATA_PATH)
    return metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    start_time = time.perf_counter()

    train_df, val_df, test_df = load_datasets()
    train_df, val_df, test_df, features = preprocess(train_df, val_df, test_df)
    target = "yield"

    # Drop rows without target values to avoid NaNs during CV/training
    before_drop = len(train_df)
    train_df = train_df.dropna(subset=[target])
    if len(train_df) < before_drop:
        LOGGER.info("Dropped %d rows with missing target from training set", before_drop - len(train_df))
    metrics_summary, fold_metrics = run_cv(train_df, features, target)
    LOGGER.info(
        "CV summary -> RMSE: %.3f (std %.3f) | rRMSE: %.3f%% (std %.3f%%) | MAPE: %.3f%% (std %.3f%%)",
        metrics_summary["cv_rmse_mean"],
        metrics_summary["cv_rmse_std"],
        metrics_summary["cv_rrmse_mean"],
        metrics_summary["cv_rrmse_std"],
        metrics_summary["cv_mape_mean"],
        metrics_summary["cv_mape_std"],
    )

    model = train_final_model(train_df, features, target)

    # 验证集可能没有目标列（官方 test_2m.csv 无 yield），有则计算指标，否则跳过
    val_metrics = {"val_rmse": float("nan"), "val_rrmse": float("nan"), "val_mape": float("nan")}
    if target in val_df.columns and len(val_df):
        before_val_drop = len(val_df)
        val_df = val_df.dropna(subset=[target])
        if len(val_df) < before_val_drop:
            LOGGER.info("Dropped %d rows with missing target from validation set", before_val_drop - len(val_df))
        if len(val_df):
            val_preds = model.predict(val_df[features])
            val_rmse = float(np.sqrt(mean_squared_error(val_df[target], val_preds)))
            val_rrmse = float(val_rmse / np.mean(val_df[target]) * 100)
            val_mask = val_df[target] != 0
            val_mape = (
                float(np.mean(np.abs((val_df[target][val_mask] - val_preds[val_mask]) / val_df[target][val_mask])) * 100)
                if np.any(val_mask)
                else float("nan")
            )
            val_metrics.update({"val_rmse": val_rmse, "val_rrmse": val_rrmse, "val_mape": val_mape})
    else:
        LOGGER.info("Validation set has no target column '%s'; skip val metrics", target)
    metrics_summary.update(val_metrics)
    feature_importance = (
        pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
    submission = generate_submission(model, test_df, features)

    artifacts = TrainingArtifacts(
        model=model,
        features=features,
        metrics_summary=metrics_summary,
        fold_metrics=fold_metrics,
        train_rows=len(train_df),
        val_rows=len(val_df),
        feature_importance=feature_importance,
        submission=submission,
    )

    data_process_snapshot = capture_data_process_snapshot()
    for key in ("path", "snapshot_path"):
        value = data_process_snapshot.get(key)
        if value:
            data_process_snapshot[key] = _relpath(value)
    training_duration = time.perf_counter() - start_time
    save_artifacts(artifacts, data_process_snapshot, training_duration_sec=training_duration)
    LOGGER.info("Training complete. Submission saved to %s", SUBMISSION_PATH)


if __name__ == "__main__":
    main()
