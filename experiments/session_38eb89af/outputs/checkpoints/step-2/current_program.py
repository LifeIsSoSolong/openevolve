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
    从 date 列提取时间特征（增加周度周期性编码，减少树模型对“跳变整数”的误解）
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # 周期性编码
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["woy_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52.0)
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征预处理：缺失值处理（先时序填充，再用中位数兜底）
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # 对时间序列更友好的填充：ffill/bfill 优先，避免中位数破坏趋势
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
    return df


def create_sliding_window_samples(
    df: pd.DataFrame, window_size: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    创建滑动窗口样本（改为“统计+动量”特征，标签改为相对最后观测值的delta）
    目的：更关注方向/变化（提升MDA），同时减少高维展平带来的噪声/过拟合。
    """
    df = preprocess_features(extract_time_features(df))

    feat_cols = [c for c in df.columns if c not in ["date", TARGET_COL]]

    X_list, y_list, last_list = [], [], []
    for i in range(len(df) - window_size - horizon + 1):
        window = df.iloc[i : i + window_size]
        future = df.iloc[i + window_size : i + window_size + horizon]

        v = window[TARGET_COL].astype(float).to_numpy()
        last = float(v[-1])
        last_list.append(last)

        # 价格序列统计/动量（更贴近“涨跌方向”）
        v_mean = float(np.mean(v))
        v_std = float(np.std(v))
        v_min = float(np.min(v))
        v_max = float(np.max(v))
        mom1 = float(v[-1] - v[-2]) if len(v) >= 2 else 0.0
        mom4 = float(v[-1] - v[-5]) याद if len(v) >= 5 else 0.0

        # 近端均值/波动
        k4 = min(4, len(v))
        k8 = min(8, len(v))
        mean4 = float(np.mean(v[-k4:]))
        mean8 = float(np.mean(v[-k8:]))
        std4 = float(np.std(v[-k4:]))
        std8 = float(np.std(v[-k8:]))

        # 线性趋势斜率
        if len(v) >= 3:
            x = np.arange(len(v), dtype=float)
            slope = float(np.polyfit(x, v, 1)[0])
        else:
            slope = 0.0

        # 使用窗口最后一行的（时间/其他）特征，避免把year/month重复展平
        row_feat = window.iloc[-1][feat_cols].astype(float).to_numpy()

        X = np.concatenate(
            [
                row_feat,
                np.array(
                    [last, v_mean, v_std, v_min, v_max, mom1, mom4, mean4, mean8, std4, std8, slope],
                    dtype=float,
                ),
                v[-k8:],  # 少量最近lags，保留局部形态
            ]
        )

        # 标签用相对最后观测值的delta（预测更关注变化）
        y = future[TARGET_COL].astype(float).to_numpy() - last

        X_list.append(X)
        y_list.append(y)

    return np.asarray(X_list, dtype=float), np.asarray(y_list, dtype=float), np.asarray(last_list, dtype=float)


def build_model():
    """
    构建单个多输出模型：ExtraTrees 原生支持 multi-output 回归。
    通常对小样本、非线性、噪声更稳健，也更容易学到方向性模式。
    """
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=800,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
    )


def compute_mda(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算 MDA：用 sign(diff) 直接比较，避免 diff==0 时被错误计为不正确。
    """
    true_diff = np.diff(y_true, axis=1)
    pred_diff = np.diff(y_pred, axis=1)
    mda = float(np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100.0)
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
    
    # 创建滑动窗口样本（y为delta；last用于还原价格水平）
    X_train, y_train, last_train = create_sliding_window_samples(
        train_df, window_size, PREDICTION_HORIZON
    )
    X_test, y_test, last_test = create_sliding_window_samples(
        test_df, window_size, PREDICTION_HORIZON
    )
    
    LOGGER.info(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}")
    
    # 训练单个多输出模型（预测8周数据）
    model = build_model()
    LOGGER.info(f"Training multi-output model for {PREDICTION_HORIZON} weeks prediction")
    model.fit(X_train, y_train)
    
    # 预测（delta -> level）
    y_pred_delta = model.predict(X_test)
    y_pred = y_pred_delta + last_test.reshape(-1, 1)
    y_true = y_test + last_test.reshape(-1, 1)

    # 测试集评估（在价格水平上评估）
    metrics = compute_metrics(y_true, y_pred)
    
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
