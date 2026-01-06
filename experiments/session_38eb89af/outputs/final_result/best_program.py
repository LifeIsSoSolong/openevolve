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

LOGGER = logging.getLogger(__name__)

# 配置
TARGET_COL = "value"
PREDICTION_HORIZON = 8  # 预测未来 8 周


# EVOLVE-BLOCK-START
def get_window_size() -> int:
    """
    获取滑动窗口大小（输入的历史周数）
    """
    return 26  # 更长窗口更利于捕捉中期趋势/季节性


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 date 列提取时间特征 + 基于历史value的统计特征（不引入未来信息）
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # 周期性编码
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["woy_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # value衍生特征（仅用到历史/当期，不用未来）
    # 用 diff/EMA 替代 pct_change，通常更稳健且对方向更直接
    if TARGET_COL in df.columns:
        v = df[TARGET_COL].astype(float)
        d1 = v.diff()
        df["d1"] = d1
        df["d4"] = v.diff(4)

        df["ma4"] = v.rolling(4, min_periods=1).mean()
        df["ma12"] = v.rolling(12, min_periods=1).mean()
        df["ema8"] = v.ewm(span=8, adjust=False).mean()

        vol4 = d1.rolling(4, min_periods=1).std()
        df["vol4"] = vol4

        # 均线偏离（标准化），对“超买/超卖->回归/延续”的方向模式更敏感
        df["dev_ma4"] = (v - df["ma4"]) / (vol4 + 1e-6)

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征预处理：时间序列更稳健的缺失值处理
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    # 先用时间序列常用的前向/后向填充，再用中位数兜底
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].ffill().bfill()
        for c in num_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
    return df


def create_sliding_window_samples(
    df: pd.DataFrame, window_size: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    创建滑动窗口样本（低维统计特征 + 少量近端lags），标签改为未来8周“逐周增量”。

    目的：
    - 避免 window_size * n_features 的高维展平导致噪声/过拟合（尤其小样本）
    - 用 increments 让模型直接学习“每周涨跌”，通常更利于提升 MDA
    """
    df = preprocess_features(extract_time_features(df))
    feat_cols = [c for c in df.columns if c not in ["date", TARGET_COL]]

    X_list, y_list, last_list = [], [], []
    n = len(df)
    for i in range(n - window_size - horizon + 1):
        window = df.iloc[i : i + window_size]
        future = df.iloc[i + window_size : i + window_size + horizon]

        v = window[TARGET_COL].astype(float).to_numpy()
        last = float(v[-1])
        last_list.append(last)

        # 基础统计/动量（更贴近方向）
        v_mean = float(np.mean(v))
        v_std = float(np.std(v))
        v_min = float(np.min(v))
        v_max = float(np.max(v))
        mom1 = float(v[-1] - v[-2]) if len(v) >= 2 else 0.0
        mom4 = float(v[-1] - v[-5]) if len(v) >= 5 else 0.0

        k4 = min(4, len(v))
        k8 = min(8, len(v))
        mean4 = float(np.mean(v[-k4:]))
        mean8 = float(np.mean(v[-k8:]))
        std4 = float(np.std(v[-k4:]))
        std8 = float(np.std(v[-k8:]))

        # 斜率/近端增量序列/方向特征（更贴近 MDA）
        r = np.diff(v) if len(v) >= 2 else np.array([0.0])
        kr = min(8, len(r))
        r8 = r[-kr:]

        if len(v) >= 3:
            x = np.arange(len(v), dtype=float)
            slope = float(np.polyfit(x, v, 1)[0])
            r_mean = float(np.mean(r8))
            r_std = float(np.std(r8))
            up_ratio = float(np.mean(r8 > 0))

            # 最近连续同方向变化的“趋势强度”（正=连涨，负=连跌）
            streak = 0.0
            for t in r8[::-1]:
                if t > 0:
                    streak = streak + 1 if streak >= 0 else 1
                elif t < 0:
                    streak = streak - 1 if streak <= 0 else -1
                else:
                    break
        else:
            slope, r_mean, r_std, up_ratio, streak = 0.0, 0.0, 0.0, 0.0, 0.0

        # 使用窗口最后一行的外生/时间/派生特征（不做展平）
        row_feat = window.iloc[-1][feat_cols].astype(float).to_numpy()

        X = np.concatenate(
            [
                row_feat,
                np.array(
                    [
                        last, v_mean, v_std, v_min, v_max,
                        mom1, mom4, mean4, mean8, std4, std8,
                        slope, r_mean, r_std, up_ratio, streak,
                    ],
                    dtype=float,
                ),
                v[-k8:],  # 少量近端lags保留形态
                r8,       # 近端增量序列（方向信息更强）
            ]
        )

        # 标签改为：未来8周相对“最后观测价”的delta（更贴近历史最佳方案）
        future_v = future[TARGET_COL].astype(float).to_numpy()
        y = future_v - last

        X_list.append(X)
        y_list.append(y)

    return np.asarray(X_list, float), np.asarray(y_list, float), np.asarray(last_list, float)


def build_model():
    """
    构建单个多输出模型：ExtraTreesRegressor 原生支持 multi-output 回归。
    相比 MultiOutputRegressor 包装，更简洁且常更稳健。
    """
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=1400,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features=0.6,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )


def compute_mda(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MDA：把“走平(0)”也视作方向一致，减少边界噪声带来的惩罚
    """
    td = np.diff(y_true, axis=1)
    pd = np.diff(y_pred, axis=1)
    correct = (td * pd) >= 0
    return float(np.mean(correct) * 100)


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
    
    # 创建滑动窗口样本（y 为未来8周相对last的 delta；last 用于还原 level）
    X_train, y_train_delta, last_train = create_sliding_window_samples(
        train_df, window_size, PREDICTION_HORIZON
    )
    X_test, y_test_delta, last_test = create_sliding_window_samples(
        test_df, window_size, PREDICTION_HORIZON
    )

    LOGGER.info(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}")

    # 轻度clip训练标签（仅用训练集分位数，抑制极端波动，利于RMSE/MAPE且通常不伤MDA）
    clip = float(np.nanpercentile(np.abs(y_train_delta), 97))
    if np.isfinite(clip) and clip > 0:
        y_train_delta = np.clip(y_train_delta, -clip, clip)

    # 单个多输出模型（预测8周 delta）
    model = build_model()
    LOGGER.info(f"Training multi-output model for {PREDICTION_HORIZON} weeks prediction")
    model.fit(X_train, y_train_delta)

    # delta -> level（直接加回 last）
    y_pred_delta = model.predict(X_test)
    y_pred = last_test[:, None] + y_pred_delta
    y_true = last_test[:, None] + y_test_delta

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
