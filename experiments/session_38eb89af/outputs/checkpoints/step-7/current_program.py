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
    
    这个参数会被 AlphaEvolve 进化优化。
    """
    return 26  # 更长窗口更利于捕捉中期趋势/季节性，通常提升方向稳定性


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
    """缺失值：先ffill/bfill，再用中位数兜底（更短、更快）"""
    df = df.copy()
    num = df.select_dtypes(include=[np.number]).columns
    if len(num) > 0:
        df[num] = df[num].ffill().bfill()
        df[num] = df[num].fillna(df[num].median())
    return df


def create_sliding_window_samples(
    df: pd.DataFrame, window_size: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    滑窗样本：更偏“方向”的特征，并把标签改成未来逐周增量(inc)。
    经验上：直接学 inc 往往比学 level/delta 更利于 MDA（方向一致性）。
    """
    df = preprocess_features(extract_time_features(df))
    feat_cols = [c for c in df.columns if c not in ["date", TARGET_COL]]

    X_list, y_list, last_list = [], [], []
    for i in range(len(df) - window_size - horizon + 1):
        w = df.iloc[i : i + window_size]
        f = df.iloc[i + window_size : i + window_size + horizon]

        v = w[TARGET_COL].astype(float).to_numpy()
        last = float(v[-1])
        last_list.append(last)

        k8 = min(8, len(v))
        k4 = min(4, len(v))

        v_mean, v_std = float(v.mean()), float(v.std())
        mean4, mean8 = float(v[-k4:].mean()), float(v[-k8:].mean())
        std4, std8 = float(v[-k4:].std()), float(v[-k8:].std())

        mom1 = float(v[-1] - v[-2]) if len(v) >= 2 else 0.0
        mom4 = float(v[-1] - v[-5]) if len(v) >= 5 else 0.0

        # 近端“收益/方向”特征（对MDA更直接）
        r = np.diff(v) if len(v) >= 3 else np.array([0.0])
        kr = min(8, len(r))
        r8 = r[-kr:]
        r_mean, r_std = float(r8.mean()), float(r8.std())
        up_ratio = float(np.mean(r8 > 0))
        streak = 0.0
        for t in r8[::-1]:
            if t > 0:
                streak = streak + 1 if streak >= 0 else 1
            elif t < 0:
                streak = streak - 1 if streak <= 0 else -1
            else:
                break

        slope = float(np.polyfit(np.arange(len(v), dtype=float), v, 1)[0]) if len(v) >= 3 else 0.0

        row_feat = w.iloc[-1][feat_cols].astype(float).to_numpy()
        X = np.concatenate(
            [
                row_feat,
                np.array(
                    [last, v_mean, v_std, mean4, mean8, std4, std8, mom1, mom4, slope, r_mean, r_std, up_ratio, streak],
                    dtype=float,
                ),
                v[-k8:],   # 仅保留少量近端形态
                r8,        # 近端增量序列（方向信息更强）
            ]
        )

        future_v = f[TARGET_COL].astype(float).to_numpy()
        y = np.diff(np.concatenate([[last], future_v]))  # 8个逐周inc

        X_list.append(X)
        y_list.append(y)

    return np.asarray(X_list, float), np.asarray(y_list, float), np.asarray(last_list, float)


def build_model():
    """
    单个多输出模型：ExtraTrees（对噪声/非线性鲁棒）。
    略加深度约束 + 更高max_features，常能提升方向稳定性并降低过拟合。
    """
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=900,
        max_depth=16,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features=0.75,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )


def compute_mda(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算 MDA：用 sign(diff) 直接比较，避免 diff==0 时被错误计为不正确。
    """
    td = np.diff(y_true, axis=1)
    pd_ = np.diff(y_pred, axis=1)
    return float(np.mean((td * pd_) >= 0) * 100.0)


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
    # 轻度clip训练标签：用训练集自身分位数抑制极端波动（常有利于RMSE/MAPE且不明显伤MDA）
    clip = float(np.nanpercentile(np.abs(y_train), 97))
    if np.isfinite(clip) and clip > 0:
        y_train = np.clip(y_train, -clip, clip)

    model = build_model()
    LOGGER.info(f"Training multi-output model for {PREDICTION_HORIZON} weeks prediction")
    model.fit(X_train, y_train)
    
    # 预测（weekly inc -> level via cumsum）
    y_pred_inc = model.predict(X_test)
    y_pred = last_test.reshape(-1, 1) + np.cumsum(y_pred_inc, axis=1)
    y_true = last_test.reshape(-1, 1) + np.cumsum(y_test, axis=1)

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
