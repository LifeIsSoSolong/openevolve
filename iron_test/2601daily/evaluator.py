"""
Evaluator for the iron 2601 daily forecasting task.

Loads the candidate pipeline module, trains a lightweight TimeMixer model
with reduced epochs for speed, then reports validation/test metrics along
with a combined_score (higher is better).
"""

from __future__ import annotations

import importlib.util
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


def _load_module_from_path(module_path: str):
    # Use a unique module name each load to avoid None sys.modules entries during forked eval
    module_name = f"candidate_module_{Path(module_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure the module is registered in sys.modules before execution to keep __module__ valid
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_fast_config(module, data_path: Path):
    """Build a lightweight config for faster evaluation."""
    # Use project_root anchored to the task folder to avoid surprises
    project_root = data_path.parent
    cfg = module.IronDailyConfig(
        project_root=project_root,
        raw_data_override=str(data_path),
        seq_len=48,
        pred_len=12,
        label_len=0,
        batch_size=8,
        learning_rate=1e-2,
        train_epochs=3,
        patience=2,
        down_sampling_layers=2,
        d_model=16,
        d_ff=32,
        dropout=0.1,
        device="cpu",  # force CPU for determinism and portability
    )
    # Ensure checkpoint dir exists under the task folder
    cfg.checkpoint_dir = project_root / "checkpoints" / "standalone_iron_daily"
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _train_and_eval(module, cfg) -> Dict[str, float]:
    """Train briefly and return metrics."""
    # Data preparation
    fused_df = module.fuse_and_align_features(cfg)
    if fused_df is None or len(fused_df) == 0:
        raise ValueError("fuse_and_align_features returned empty data.")

    fe_df = module.run_feature_engineering(fused_df, cfg)
    if fe_df is None or len(fe_df) == 0:
        raise ValueError("run_feature_engineering returned empty data.")
    if len(fe_df) < cfg.seq_len + cfg.pred_len + 5:
        raise ValueError(
            f"Not enough data after feature engineering. rows={len(fe_df)}, "
            f"need>{cfg.seq_len + cfg.pred_len + 5}"
        )

    split_info, feature_cols = module.prepare_custom_style_data(fe_df, cfg)
    loaders = module.make_dataloaders_from_splits(split_info, cfg)
    enc_in = len(feature_cols)

    if any(len(loader.dataset) == 0 for loader in loaders.values()):
        raise ValueError(
            f"Empty dataset split; train={len(loaders['train'].dataset)}, "
            f"val={len(loaders['val'].dataset)}, test={len(loaders['test'].dataset)}"
        )

    model = module.build_model(cfg, enc_in).to(cfg.device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    best_state = None
    best_val = math.inf
    patience_counter = 0

    def _train_one_epoch() -> float:
        model.train()
        running_loss = 0.0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loaders["train"]:
            batch_x = batch_x.to(cfg.device_obj)
            batch_y = batch_y.to(cfg.device_obj)
            batch_x_mark = batch_x_mark.to(cfg.device_obj)
            batch_y_mark = batch_y_mark.to(cfg.device_obj)

            if cfg.down_sampling_layers == 0:
                dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, :cfg.label_len, :], dec_inp], dim=1).to(
                    cfg.device_obj
                )
            else:
                dec_inp = None

            optimizer.zero_grad()
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred_y, true_y = module.extract_target(outputs, batch_y, cfg)
            loss = criterion(pred_y, true_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        steps = max(len(loaders["train"]), 1)
        return running_loss / steps

    for _ in range(cfg.train_epochs):
        _train_one_epoch()
        val_mse, _, _, _ = module.evaluate(model, loaders["val"], cfg, cfg.device_obj)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_mse, val_mae, val_mape, val_da = module.evaluate(
        model, loaders["val"], cfg, cfg.device_obj
    )
    test_mse, test_mae, test_mape, test_da = module.evaluate(
        model, loaders["test"], cfg, cfg.device_obj
    )

    # Higher is better: reward low errors and good direction accuracy
    # combined_score = (
    #     0.4 * (1.0 / (1.0 + test_mse))
    #     + 0.4 * (1.0 / (1.0 + test_mape))
    #     + 0.2 * max(0.0, test_da if not math.isnan(test_da) else 0.0)
    # )
    combined_score = test_da

    return {
        "combined_score": float(combined_score),
        "val_mse": float(val_mse),
        "val_mae": float(val_mae),
        "val_mape": float(val_mape),
        "val_da": float(val_da),
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "test_mape": float(test_mape),
        "test_da": float(test_da),
    }


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Entry point for OpenEvolve. Returns metrics dict with combined_score.
    """
    try:
        # Force a fresh load each time to avoid stale modules
        module = _load_module_from_path(program_path)
        data_path = Path(__file__).resolve().parent / "merged_data.csv"
        if not data_path.exists():
            return {"combined_score": 0.0, "error": f"Data file not found: {data_path}"}
        cfg = _build_fast_config(module, data_path)
        metrics = _train_and_eval(module, cfg)
        return metrics
    except Exception as e:
        tb = traceback.format_exc()
        return {"combined_score": 0.0, "error": f"{e}\n{tb}"}
