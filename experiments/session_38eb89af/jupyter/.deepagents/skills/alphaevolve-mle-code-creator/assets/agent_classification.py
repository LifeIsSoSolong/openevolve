"""
AlphaEvolve 兼容的分类任务模板

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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMClassifier
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as LGBMClassifier

LOGGER = logging.getLogger(__name__)

# 配置
TARGET_COL = "target"  # TODO: 修改为实际目标列名


# EVOLVE-BLOCK-START
def preprocess_features(df: pd.DataFrame, target_col: str, is_train: bool = True,
                        encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """
    特征预处理
    
    这个函数会被 AlphaEvolve 进化，尝试不同的特征工程策略。
    """
    df = df.copy()
    if encoders is None:
        encoders = {}
    
    # 编码类别特征（排除目标列）
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col == target_col:
            continue
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
        "class_weight": "balanced",  # 处理类别不平衡
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict[str, float]:
    """
    计算分类指标
    
    返回的指标将用于 judge.py 计算 combined_score。
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
    
    # AUC（仅二分类）
    if y_prob is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except:
            pass  # 多分类或其他情况下可能失败
    
    return metrics
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
    
    # 编码目标列
    le_target = LabelEncoder()
    train_df[TARGET_COL] = le_target.fit_transform(train_df[TARGET_COL].astype(str))
    test_df[TARGET_COL] = le_target.transform(test_df[TARGET_COL].astype(str))
    
    n_classes = len(le_target.classes_)
    LOGGER.info(f"Number of classes: {n_classes}")
    
    # 预处理
    train_df, encoders = preprocess_features(train_df, TARGET_COL, is_train=True)
    test_df, _ = preprocess_features(test_df, TARGET_COL, is_train=False, encoders=encoders)
    
    # 准备特征和标签
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]
    
    # 训练
    model = LGBMClassifier(**get_model_params())
    model.fit(X_train, y_train)
    
    # 预测
    predictions = model.predict(X_test)
    
    # 获取概率（二分类时用于 AUC）
    probabilities = None
    if n_classes == 2:
        probabilities = model.predict_proba(X_test)[:, 1]
    
    # 测试集评估（重要：必须是测试集上的指标）
    metrics = compute_metrics(y_test.values, predictions, probabilities)
    
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
