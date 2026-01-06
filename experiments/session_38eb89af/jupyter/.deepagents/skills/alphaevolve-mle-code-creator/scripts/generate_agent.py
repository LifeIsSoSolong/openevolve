#!/usr/bin/env python3
"""
生成符合 AlphaEvolve 规范的 agent.py

根据数据分析结果和任务类型，生成可进化的机器学习代码。
"""

import argparse
import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd


def get_regression_template(target_col: str, feature_cols: list, 
                            has_categorical: bool, has_datetime: bool) -> str:
    """回归任务模板"""
    
    categorical_code = ""
    if has_categorical:
        categorical_code = '''
    # 编码类别特征
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if is_train:
            encoders[col] = {v: i for i, v in enumerate(df[col].unique())}
        if col in encoders:
            df[col] = df[col].map(encoders[col]).fillna(-1).astype(int)
'''

    datetime_code = ""
    if has_datetime:
        datetime_code = '''
    # 时间特征
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df = df.drop(columns=[col])
'''

    template = f'''"""
AlphaEvolve 兼容的机器学习代码
任务类型: 回归
目标列: {target_col}
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


# EVOLVE-BLOCK-START
def preprocess_features(df: pd.DataFrame, is_train: bool = True,
                        encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """特征预处理（可被进化）"""
    df = df.copy()
    if encoders is None:
        encoders = {{}}
{categorical_code}{datetime_code}
    # 填充数值缺失值
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df, encoders


def get_model_params() -> dict:
    """模型超参数（可被进化）"""
    return {{
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
    }}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算评估指标（可被进化）"""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    # rRMSE (相对 RMSE)
    mean_y = float(np.mean(y_true))
    rrmse = float(rmse / mean_y * 100) if mean_y != 0 else float("nan")
    
    # MAPE
    mask = y_true != 0
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
    
    return {{"rmse": rmse, "mae": mae, "rrmse": rrmse, "mape": mape}}
# EVOLVE-BLOCK-END


def main(root) -> dict[str, float]:
    """
    AlphaEvolve 入口函数
    
    Args:
        root: 数据目录路径（由 judge.py 传入）
    
    Returns:
        dict: 评估指标（测试集上的）
    """
    data_dir = Path(root)
    
    # 加载数据
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    LOGGER.info(f"Train: {{train_df.shape}}, Test: {{test_df.shape}}")
    
    target_col = "{target_col}"
    
    # 预处理
    train_df, encoders = preprocess_features(train_df, is_train=True)
    test_df, _ = preprocess_features(test_df, is_train=False, encoders=encoders)
    
    # 准备特征和标签
    feature_cols = [c for c in train_df.columns if c != target_col]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # 训练
    model = LGBMRegressor(**get_model_params())
    model.fit(X_train, y_train)
    
    # 测试集评估
    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test.values, predictions)
    
    LOGGER.info(f"Test metrics: {{metrics}}")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    result = main(".")
    print(result)
'''
    return template


def get_classification_template(target_col: str, feature_cols: list,
                                 has_categorical: bool, is_binary: bool) -> str:
    """分类任务模板"""
    
    categorical_code = ""
    if has_categorical:
        categorical_code = '''
    # 编码类别特征
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col == target_col:
            continue
        if is_train:
            encoders[col] = {v: i for i, v in enumerate(df[col].unique())}
        if col in encoders:
            df[col] = df[col].map(encoders[col]).fillna(-1).astype(int)
'''

    if is_binary:
        metrics_code = '''
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict[str, float]:
    """计算分类指标（可被进化）"""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
    }
    
    if y_prob is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except:
            metrics["auc"] = 0.5
    
    return metrics'''
        predict_code = '''
    # 预测
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # 评估
    metrics = compute_metrics(y_test.values, predictions, probabilities)'''
    else:
        metrics_code = '''
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算分类指标（可被进化）"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }'''
        predict_code = '''
    # 预测
    predictions = model.predict(X_test)
    
    # 评估
    metrics = compute_metrics(y_test.values, predictions)'''

    template = f'''"""
AlphaEvolve 兼容的机器学习代码
任务类型: {'二分类' if is_binary else '多分类'}
目标列: {target_col}
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score{"" if not is_binary else ", roc_auc_score"}
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMClassifier
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as LGBMClassifier

LOGGER = logging.getLogger(__name__)


# EVOLVE-BLOCK-START
def preprocess_features(df: pd.DataFrame, target_col: str, is_train: bool = True,
                        encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """特征预处理（可被进化）"""
    df = df.copy()
    if encoders is None:
        encoders = {{}}
{categorical_code}
    # 填充数值缺失值
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df, encoders


def get_model_params() -> dict:
    """模型超参数（可被进化）"""
    return {{
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "class_weight": "balanced",
        "random_state": 42,
    }}


{metrics_code}
# EVOLVE-BLOCK-END


def main(root) -> dict[str, float]:
    """
    AlphaEvolve 入口函数
    
    Args:
        root: 数据目录路径（由 judge.py 传入）
    
    Returns:
        dict: 评估指标（测试集上的）
    """
    data_dir = Path(root)
    
    # 加载数据
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    LOGGER.info(f"Train: {{train_df.shape}}, Test: {{test_df.shape}}")
    
    target_col = "{target_col}"
    
    # 编码目标列
    le_target = LabelEncoder()
    train_df[target_col] = le_target.fit_transform(train_df[target_col].astype(str))
    test_df[target_col] = le_target.transform(test_df[target_col].astype(str))
    
    # 预处理
    train_df, encoders = preprocess_features(train_df, target_col, is_train=True)
    test_df, _ = preprocess_features(test_df, target_col, is_train=False, encoders=encoders)
    
    # 准备特征和标签
    feature_cols = [c for c in train_df.columns if c != target_col]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # 训练
    model = LGBMClassifier(**get_model_params())
    model.fit(X_train, y_train)
{predict_code}
    
    LOGGER.info(f"Test metrics: {{metrics}}")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    result = main(".")
    print(result)
'''
    return template


def analyze_data(input_dir: Path) -> dict:
    """分析数据"""
    train_path = input_dir / "train.csv"
    if not train_path.exists():
        train_path = input_dir / "train.xlsx"
    
    if train_path.suffix == ".csv":
        train_df = pd.read_csv(train_path)
    else:
        train_df = pd.read_excel(train_path)
    
    # 推断目标列
    common_names = ["target", "label", "y", "class", "outcome", "yield", "price", "churn"]
    target_col = None
    for name in common_names:
        if name in train_df.columns:
            target_col = name
            break
        for col in train_df.columns:
            if col.lower() == name:
                target_col = col
                break
        if target_col:
            break
    
    if target_col is None:
        target_col = train_df.columns[-1]
    
    # 推断任务类型
    target = train_df[target_col]
    n_unique = target.nunique()
    
    if target.dtype == object or (n_unique <= 10 and n_unique / len(target) < 0.05):
        task_type = "classification"
        is_binary = n_unique == 2
    else:
        task_type = "regression"
        is_binary = False
    
    # 检查特征类型
    feature_cols = [c for c in train_df.columns if c != target_col]
    has_categorical = any(train_df[col].dtype == object for col in feature_cols)
    has_datetime = any(pd.api.types.is_datetime64_any_dtype(train_df[col]) for col in feature_cols)
    
    return {
        "target_col": target_col,
        "task_type": task_type,
        "is_binary": is_binary,
        "feature_cols": feature_cols,
        "has_categorical": has_categorical,
        "has_datetime": has_datetime,
    }


def main():
    parser = argparse.ArgumentParser(description="生成 AlphaEvolve 兼容的 agent.py")
    parser.add_argument("--input-dir", required=True, help="输入目录")
    parser.add_argument("--task-type", choices=["regression", "classification"], help="任务类型")
    parser.add_argument("--target-col", help="目标列名")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不写入文件")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # 分析数据
    analysis = analyze_data(input_dir)
    
    # 覆盖用户指定的参数
    if args.task_type:
        analysis["task_type"] = args.task_type
        analysis["is_binary"] = args.task_type == "classification" and analysis.get("is_binary", True)
    if args.target_col:
        analysis["target_col"] = args.target_col
    
    # 生成代码
    if analysis["task_type"] == "regression":
        code = get_regression_template(
            target_col=analysis["target_col"],
            feature_cols=analysis["feature_cols"],
            has_categorical=analysis["has_categorical"],
            has_datetime=analysis["has_datetime"],
        )
    else:
        code = get_classification_template(
            target_col=analysis["target_col"],
            feature_cols=analysis["feature_cols"],
            has_categorical=analysis["has_categorical"],
            is_binary=analysis["is_binary"],
        )
    
    if args.dry_run:
        print("=" * 60)
        print("预览生成的 agent.py")
        print("=" * 60)
        print(code)
        print("=" * 60)
        print(f"\n[INFO] 任务类型: {analysis['task_type']}")
        print(f"[INFO] 目标列: {analysis['target_col']}")
        print(f"[INFO] 特征数: {len(analysis['feature_cols'])}")
    else:
        output_path = input_dir / "agent.py"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"✅ agent.py 已生成: {output_path}")
        print(f"   任务类型: {analysis['task_type']}")
        print(f"   目标列: {analysis['target_col']}")


if __name__ == "__main__":
    main()
