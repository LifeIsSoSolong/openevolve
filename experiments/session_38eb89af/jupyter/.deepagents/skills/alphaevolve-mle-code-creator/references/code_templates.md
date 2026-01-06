# 代码模板参考

本文档提供常见任务类型的 agent.py 代码模板，生成代码时应参考这些模板的**结构**。

**重要**：
- 模板中使用 `train.csv`/`test.csv` 仅作为占位符示意
- 实际生成代码时，必须替换为用户的**真实数据文件名**
- 支持多种格式：csv, xlsx, parquet, json 等，使用对应的 `pd.read_*()` 方法

## 回归任务模板

### 基础 LightGBM 回归

```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

LOGGER = logging.getLogger(__name__)


# EVOLVE-BLOCK-START
def preprocess_features(df: pd.DataFrame, is_train: bool = True, 
                        encoders: dict = None) -> tuple[pd.DataFrame, dict]:
    """特征预处理"""
    df = df.copy()
    if encoders is None:
        encoders = {}
    
    # 处理类别特征
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if is_train:
            encoders[col] = {v: i for i, v in enumerate(df[col].unique())}
        if col in encoders:
            df[col] = df[col].map(encoders[col]).fillna(-1).astype(int)
    
    # 填充缺失值
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    return df, encoders


def get_model_params() -> dict:
    """模型超参数"""
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
    """计算评估指标"""
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
    
    return {"rmse": rmse, "mae": mae, "rrmse": rrmse, "mape": mape}
# EVOLVE-BLOCK-END


def main(root) -> dict[str, float]:
    """AlphaEvolve 入口"""
    data_dir = Path(root)
    
    # 加载数据
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    LOGGER.info(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # 目标列（根据实际数据调整）
    target_col = "target"
    
    # 预处理
    train_df, encoders = preprocess_features(train_df, is_train=True)
    test_df, _ = preprocess_features(test_df, is_train=False, encoders=encoders)
    
    # 准备特征
    feature_cols = [c for c in train_df.columns if c != target_col]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # 训练
    model = LGBMRegressor(**get_model_params())
    model.fit(X_train, y_train)
    
    # 测试集预测和评估
    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test.values, predictions)
    
    LOGGER.info(f"Test metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = main(".")
    print(result)
```

### 基础 XGBoost 回归

```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

try:
    import xgboost as xgb
except ImportError:
    xgb = None

LOGGER = logging.getLogger(__name__)


# EVOLVE-BLOCK-START
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程"""
    df = df.copy()
    
    # 数值特征的对数变换（处理偏斜）
    for col in df.select_dtypes(include=[np.number]).columns:
        if (df[col] > 0).all():
            df[f"{col}_log"] = np.log1p(df[col])
    
    return df


def get_xgb_params() -> dict:
    """XGBoost 参数"""
    return {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "tree_method": "hist",
    }
# EVOLVE-BLOCK-END


def main(root) -> dict[str, float]:
    data_dir = Path(root)
    
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    target_col = "target"
    
    # 特征工程
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    feature_cols = [c for c in train_df.columns if c != target_col]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # 训练
    model = xgb.XGBRegressor(**get_xgb_params())
    model.fit(X_train, y_train)
    
    # 评估
    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    
    return {"rmse": rmse}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(main("."))
```

## 分类任务模板

### 二分类（LightGBM）

```python
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
    LGBMClassifier = None

LOGGER = logging.getLogger(__name__)


# EVOLVE-BLOCK-START
def encode_categorical(df: pd.DataFrame, encoders: dict = None, 
                       is_train: bool = True) -> tuple[pd.DataFrame, dict]:
    """编码类别特征"""
    df = df.copy()
    if encoders is None:
        encoders = {}
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        elif col in encoders:
            le = encoders[col]
            # 处理未知类别
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
    
    return df, encoders


def get_classifier_params() -> dict:
    """分类器参数"""
    return {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "class_weight": "balanced",  # 处理类别不平衡
        "random_state": 42,
        "verbosity": -1,
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_prob: np.ndarray = None) -> dict[str, float]:
    """计算分类指标"""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary")),
    }
    
    if y_prob is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except:
            metrics["auc"] = 0.5
    
    return metrics
# EVOLVE-BLOCK-END


def main(root) -> dict[str, float]:
    """AlphaEvolve 入口"""
    data_dir = Path(root)
    
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    LOGGER.info(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    target_col = "target"
    
    # 编码
    train_df, encoders = encode_categorical(train_df, is_train=True)
    test_df, _ = encode_categorical(test_df, encoders=encoders, is_train=False)
    
    # 准备数据
    feature_cols = [c for c in train_df.columns if c != target_col]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # 训练
    model = LGBMClassifier(**get_classifier_params())
    model.fit(X_train, y_train)
    
    # 预测
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # 评估
    metrics = compute_classification_metrics(y_test.values, predictions, probabilities)
    
    LOGGER.info(f"Test metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(main("."))
```

### 多分类

```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

LOGGER = logging.getLogger(__name__)


# EVOLVE-BLOCK-START
def preprocess(df: pd.DataFrame, label_encoder: LabelEncoder = None,
               cat_encoders: dict = None, is_train: bool = True):
    """数据预处理"""
    df = df.copy()
    
    if cat_encoders is None:
        cat_encoders = {}
    
    # 编码类别特征
    for col in df.select_dtypes(include=['object']).columns:
        if col == "target":
            continue
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            cat_encoders[col] = le
        elif col in cat_encoders:
            df[col] = cat_encoders[col].transform(df[col].astype(str))
    
    return df, cat_encoders
# EVOLVE-BLOCK-END


def main(root) -> dict[str, float]:
    data_dir = Path(root)
    
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    target_col = "target"
    
    # 编码目标
    le_target = LabelEncoder()
    train_df[target_col] = le_target.fit_transform(train_df[target_col])
    test_df[target_col] = le_target.transform(test_df[target_col])
    
    # 预处理
    train_df, encoders = preprocess(train_df, is_train=True)
    test_df, _ = preprocess(test_df, cat_encoders=encoders, is_train=False)
    
    feature_cols = [c for c in train_df.columns if c != target_col]
    
    # 训练
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )
    model.fit(train_df[feature_cols], train_df[target_col])
    
    # 评估
    predictions = model.predict(test_df[feature_cols])
    
    accuracy = float(accuracy_score(test_df[target_col], predictions))
    f1_macro = float(f1_score(test_df[target_col], predictions, average="macro"))
    f1_weighted = float(f1_score(test_df[target_col], predictions, average="weighted"))
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(main("."))
```

## 时序数据模板

```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

LOGGER = logging.getLogger(__name__)


# EVOLVE-BLOCK-START
def create_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """创建时间特征"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["quarter"] = df[date_col].dt.quarter
    
    # 周期性编码
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df


def create_lag_features(df: pd.DataFrame, target_col: str, 
                        lags: list[int] = [1, 7, 14]) -> pd.DataFrame:
    """创建滞后特征"""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df
# EVOLVE-BLOCK-END


def main(root) -> dict[str, float]:
    data_dir = Path(root)
    
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    target_col = "target"
    date_col = "date"
    
    # 时间特征
    train_df = create_time_features(train_df, date_col)
    test_df = create_time_features(test_df, date_col)
    
    # 移除日期列
    train_df = train_df.drop(columns=[date_col])
    test_df = test_df.drop(columns=[date_col])
    
    feature_cols = [c for c in train_df.columns if c != target_col]
    
    # 填充缺失值
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    # 简单模型
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(train_df[feature_cols], train_df[target_col])
    
    predictions = model.predict(test_df[feature_cols])
    rmse = float(np.sqrt(mean_squared_error(test_df[target_col], predictions)))
    
    return {"rmse": rmse}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(main("."))
```

## 代码生成注意事项

1. **根据数据特点选择模板**：
   - 数值目标 → 回归模板
   - 类别目标（2类）→ 二分类模板
   - 类别目标（多类）→ 多分类模板
   - 有日期列 → 时序模板

2. **调整目标列名**：模板中的 `target_col = "target"` 需要根据实际数据修改

3. **调整特征处理**：根据数据的实际特征类型调整预处理逻辑

4. **EVOLVE-BLOCK 位置**：
   - 特征工程函数
   - 模型参数函数
   - 评估指标函数
   - 核心训练逻辑

5. **保持固定**：
   - main(root) 函数签名
   - 数据加载方式
   - 返回值格式
