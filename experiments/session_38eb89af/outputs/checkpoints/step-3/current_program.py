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

LOGGER = logging.getLogger(__name__)

# 配置
TARGET_COL = "value"
PREDICTION_HORIZON = 8  # 预测未来 8 周


# EVOLVE-BLOCK-START
def get_window_size() -> int:
    """滑动窗口大小（历史周数 M）"""
    return 26  # 更长窗口更利于捕捉中期趋势/季节性


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """时间特征 + 历史value衍生特征（仅用到过去，不引入未来）"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # 周期性编码（月度 + 周度）
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["woy_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # value衍生特征（对方向/MDA通常更敏感）
    if TARGET_COL in df.columns:
        v = df[TARGET_COL].astype(float)
        r1 = v.pct_change()
        df["ret1"] = r1
        df["ret2"] = v.pct_change(2)
        df["ma4"] = v.rolling(4).mean()
        df["ma12"] = v.rolling(12).mean()
        df["mom4"] = v - v.shift(4)
        df["vol4"] = r1.rolling(4).std()

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """缺失值：时间序列更稳健的ffill/bfill，中位数兜底"""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].ffill().bfill()
        for c in num_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
    return df


def create_sliding_window_samples(
    df: pd.DataFrame, window_size: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """滑窗样本：返回X, 未来8周level标签, 以及窗口末端last_value（用于diff训练/还原）"""
    df = preprocess_features(extract_time_features(df))

    # 保留value进入特征（方向预测通常关键），仅排除date
    feature_cols = [c for c in df.columns if c != "date"]

    X_list, y_list, last_list = [], [], []
    for i in range(len(df) - window_size - horizon + 1):
        win = df.iloc[i : i + window_size]
        X_list.append(win[feature_cols].values.flatten())

        fut = df.iloc[i + window_size : i + window_size + horizon][TARGET_COL].values
        y_list.append(fut)

        last_list.append(float(df.iloc[i + window_size - 1][TARGET_COL]))

    return np.array(X_list), np.array(y_list), np.array(last_list)


def build_model():
    """单模型多输出：ExtraTrees 往往对方向更敏感且鲁棒"""
    from sklearn.ensemble import ExtraTreesRegressor

    base = ExtraTreesRegressor(
        n_estimators=600,
        random_state=42,
        min_samples_leaf=2,
        max_features=0.7,
        bootstrap=True,
        n_jobs=-1,
    )
    return MultiOutputRegressor(base, n_jobs=-1)


def compute_mda(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MDA：把“走平(0)”也视作方向一致，减少边界噪声惩罚"""
    td = np.diff(y_true, axis=1)
    pd_ = np.diff(y_pred, axis=1)
    return float(np.mean((td * pd_) >= 0) * 100)


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
    X_train, y_train_level, last_train = create_sliding_window_samples(
        train_df, window_size, PREDICTION_HORIZON
    )
    X_test, y_test_level, last_test = create_sliding_window_samples(
        test_df, window_size, PREDICTION_HORIZON
    )

    # 用“未来相对窗口末值的变化量(diff)”训练，通常更利于方向/MDA
    y_train = y_train_level - last_train[:, None]

    LOGGER.info(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}")

    model = build_model()
    LOGGER.info(f"Training multi-output model for {PREDICTION_HORIZON} weeks prediction")
    model.fit(X_train, y_train)

    # 预测diff并还原为level
    y_pred = model.predict(X_test) + last_test[:, None]

    metrics = compute_metrics(y_test_level, y_pred)
    
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
