"""
AlphaEvolve 兼容的回归任务模板

使用方法:
1. 修改 TARGET_COL 为实际目标列名
2. 根据数据特点调整 EVOLVE-BLOCK 内的代码
3. 运行 python agent.py 验证
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from lightgbm import LGBMRegressor
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as LGBMRegressor

LOGGER = logging.getLogger(__name__)

# 配置
TARGET_COL = "target"  # TODO: 修改为实际目标列名


# EVOLVE-BLOCK-START
def preprocess_features(df: pd.DataFrame, is_train: bool = True,
                        encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    特征预处理
    
    这个函数会被 AlphaEvolve 进化，尝试不同的特征工程策略。
    """
    df = df.copy()
    if encoders is None:
        encoders = {}
    
    # 编码类别特征
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if is_train:
            encoders[col] = {v: i for i, v in enumerate(df[col].unique())}
        if col in encoders:
            df[col] = df[col].map(encoders[col]).fillna(-1).astype(int)
    
    # 填充数值缺失值
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df, encoders


def get_model_params() -> dict:
    """
    模型超参数
    
    这个函数会被 AlphaEvolve 进化，尝试不同的超参数组合。
    """
    return {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    计算评估指标
    
    返回的指标将用于 judge.py 计算 combined_score。
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    # 相对 RMSE
    mean_y = float(np.mean(y_true))
    rrmse = float(rmse / mean_y * 100) if mean_y != 0 else float("nan")
    
    # MAPE
    mask = y_true != 0
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
    
    return {"rmse": rmse, "mae": mae, "rrmse": rrmse, "mape": mape}
# EVOLVE-BLOCK-END


def main(root) -> dict[str, float]:
    """
    AlphaEvolve 入口函数
    
    Args:
        root: 数据目录路径（由 judge.py 传入，指向包含 train.csv/test.csv 的目录）
    
    Returns:
        dict: 评估指标字典（测试集上的指标）
    
    注意:
        - 所有路径必须基于 root 构建
        - 只返回指标 dict，不返回模型等其他对象
        - 指标必须是在测试集上计算的
    """
    data_dir = Path(root)
    
    # 加载数据
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    LOGGER.info(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # 预处理
    train_df, encoders = preprocess_features(train_df, is_train=True)
    test_df, _ = preprocess_features(test_df, is_train=False, encoders=encoders)
    
    # 准备特征和标签
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]
    
    # 训练
    model = LGBMRegressor(**get_model_params())
    model.fit(X_train, y_train)
    
    # 测试集评估（重要：必须是测试集上的指标）
    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test.values, predictions)
    
    LOGGER.info(f"Test metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    # 本地测试时使用当前目录
    result = main(".")
    print(f"Result: {result}")
