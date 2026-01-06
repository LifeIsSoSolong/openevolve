"""
铁矿石期货价格预测 - AlphaEvolve 兼容代码

任务：预测未来 8 周的铁矿石 2601 期货周收盘价
评估指标：MDA（主要）、RMSE、MAPE（次要）
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

LOGGER = logging.getLogger(__name__)

# 配置
TARGET_COL = "value"
PREDICTION_HORIZON = 8  # 预测未来 8 周


# EVOLVE-BLOCK-START
def get_window_size() -> int:
    """
    获取滑动窗口大小（输入的历史周数）
    
    这个参数会被 AlphaEvolve 进化优化。
    """
    return 12  # 默认使用 12 周历史数据


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 date 列提取时间特征
    
    这个函数会被 AlphaEvolve 进化，尝试不同的时间特征组合。
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 基础时间特征
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # 周期性特征（正弦/余弦编码）
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征预处理：处理缺失值
    
    这个函数会被 AlphaEvolve 进化，尝试不同的缺失值处理策略。
    """
    df = df.copy()
    
    # 填充数值列的缺失值（使用中位数）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df


def create_sliding_window_samples(df: pd.DataFrame, window_size: int, 
                                   horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    创建滑动窗口样本
    
    Args:
        df: 时间序列数据（已排序）
        window_size: 输入窗口大小（M 周）
        horizon: 预测窗口大小（8 周）
    
    Returns:
        X: 特征矩阵 (n_samples, window_size * n_features)
        y: 标签矩阵 (n_samples, horizon)
    """
    # 提取时间特征
    df = extract_time_features(df)
    
    # 预处理特征（处理缺失值）
    df = preprocess_features(df)
    
    # 选择特征列（排除 date 和 target）
    feature_cols = [c for c in df.columns if c not in ['date', TARGET_COL]]
    
    X_list = []
    y_list = []
    
    # 滑动窗口
    for i in range(len(df) - window_size - horizon + 1):
        # 输入窗口：window_size 周的数据
        window_data = df.iloc[i:i + window_size]
        
        # 特征：展平为一维向量
        X_features = window_data[feature_cols].values.flatten()
        
        # 标签：未来 horizon 周的 value
        y_targets = df.iloc[i + window_size:i + window_size + horizon][TARGET_COL].values
        
        X_list.append(X_features)
        y_list.append(y_targets)
    
    return np.array(X_list), np.array(y_list)


def build_model():
    """
    构建多输出回归模型
    
    这个函数会被 AlphaEvolve 进化，尝试不同的模型架构和参数。
    """
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    
    base_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )
    
    # 使用 MultiOutputRegressor 包装，预测8个输出
    return MultiOutputRegressor(base_model)


def compute_mda(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算 MDA (Mean Directional Accuracy) - 方向准确率
    
    衡量预测方向（涨/跌）的准确性。
    """
    # 计算真实方向变化
    true_direction = np.diff(y_true, axis=1)  # (n_samples, horizon-1)
    pred_direction = np.diff(y_pred, axis=1)
    
    # 判断方向是否一致
    correct_direction = (true_direction * pred_direction) > 0
    
    # 计算准确率
    mda = float(np.mean(correct_direction) * 100)
    return mda


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    计算评估指标
    
    返回的指标将用于 judge.py 计算 combined_score。
    """
    # MDA（主要指标）
    mda = compute_mda(y_true, y_pred)
    
    # RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # MAPE
    mask = y_true != 0
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
    
    return {"mda": mda, "rmse": rmse, "mape": mape}
# EVOLVE-BLOCK-END


def main(root) -> dict[str, float]:
    """
    AlphaEvolve 入口函数
    
    Args:
        root: 数据目录路径（由 judge.py 传入）
    
    Returns:
        dict: 评估指标字典（测试集上的指标）
    """
    data_dir = Path(root)
    
    # 加载数据
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    # 确保按时间排序
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    train_df = train_df.sort_values('date').reset_index(drop=True)
    test_df = test_df.sort_values('date').reset_index(drop=True)
    
    LOGGER.info(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # 获取窗口大小
    window_size = get_window_size()
    LOGGER.info(f"Window size: {window_size}, Horizon: {PREDICTION_HORIZON}")
    
    # 创建滑动窗口样本
    X_train, y_train = create_sliding_window_samples(
        train_df, window_size, PREDICTION_HORIZON
    )
    X_test, y_test = create_sliding_window_samples(
        test_df, window_size, PREDICTION_HORIZON
    )
    
    LOGGER.info(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}")
    
    # 训练单个多输出模型（预测8周数据）
    model = build_model()
    LOGGER.info(f"Training multi-output model for {PREDICTION_HORIZON} weeks prediction")
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 测试集评估
    metrics = compute_metrics(y_test, y_pred)
    
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
