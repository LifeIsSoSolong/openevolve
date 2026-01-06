# agent.py 规范说明（AlphaEvolve 兼容）

## 核心规范

生成的 `agent.py` 必须满足以下所有要求，才能被 AlphaEvolve 正确进化。

**重要**：示例代码中使用 `train.csv`/`test.csv` 仅作为示意，实际生成代码时应使用用户的**真实数据文件名**（如 `training_data.parquet`, `validation.xlsx` 等）。

### 1. main(root) 入口函数

**必须**实现 `main(root)` 函数作为入口：

```python
def main(root):
    """
    AlphaEvolve 入口函数
    
    Args:
        root: 数据目录的绝对路径（由 judge.py 传入）
              包含训练和测试数据文件
    
    Returns:
        dict: 仅包含评估指标，如 {"rmse": 0.5, "accuracy": 0.85}
    """
    data_dir = Path(root)
    # 使用实际的数据文件名
    train_df = pd.read_csv(data_dir / "实际训练文件名.csv")
    test_df = pd.read_parquet(data_dir / "实际测试文件名.parquet")
    # ... 训练和评估逻辑
    return {"metric_name": metric_value}
```

**重要**：
- 参数名必须是 `root`
- root 是 judge.py 所在目录的绝对路径
- 所有数据文件路径必须基于 root 构建

### 2. EVOLVE-BLOCK 标记

AlphaEvolve 只会修改标记内的代码：

```python
# EVOLVE-BLOCK-START
def some_function():
    """这里的代码会被进化"""
    pass
# EVOLVE-BLOCK-END
```

**规则**：
- 标记必须成对出现
- 目前建议使用单个 EVOLVE-BLOCK（多个 EVOLVE-BLOCK 支持有限）
- 标记外的代码（如数据加载、结果返回）不会被修改
- 核心算法逻辑应放在 EVOLVE-BLOCK 内

**EVOLVE-BLOCK 范围选择指南**：

| 应该放入 EVOLVE-BLOCK | 不应该放入 EVOLVE-BLOCK |
|----------------------|------------------------|
| 特征工程函数 | 数据加载代码 |
| 模型超参数配置 | main(root) 函数签名 |
| 模型构建/选择 | 测试集的 transform 调用 |
| 预处理逻辑 | 结果返回语句 |
| 指标计算方式 | 文件路径构建 |

**典型 EVOLVE-BLOCK 结构**：

```python
# EVOLVE-BLOCK-START
def preprocess_features(df, is_train=True, encoders=None):
    """特征预处理 - 会被进化"""
    # ...

def get_model_params():
    """模型超参数 - 会被进化"""
    return {"n_estimators": 100, "max_depth": 6}

def build_model():
    """模型构建 - 会被进化"""
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**get_model_params())
# EVOLVE-BLOCK-END

def main(root):
    # 数据加载 - 不进化
    train_df = pd.read_csv(Path(root) / "train.csv")
    test_df = pd.read_csv(Path(root) / "test.csv")
    
    # 调用进化块内的函数
    train_df, encoders = preprocess_features(train_df, is_train=True)
    test_df, _ = preprocess_features(test_df, is_train=False, encoders=encoders)
    
    # 训练和评估 - 不进化
    model = build_model()
    model.fit(X_train, y_train)
    ...
```

**安全警告**：
- ⚠️ 不要将测试集的 fit 操作放入 EVOLVE-BLOCK，可能导致进化产生数据泄露代码
- ⚠️ 如果用户要求扩大 EVOLVE-BLOCK 范围包含测试集处理，需要警告风险

### 3. 路径构建规则

**正确做法**：
```python
from pathlib import Path

def main(root):
    data_dir = Path(root)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    model_path = data_dir / "model.pkl"
```

**禁止做法**：
```python
# ❌ 错误：使用相对路径
train_df = pd.read_csv("train.csv")

# ❌ 错误：使用 __file__
root = Path(__file__).parent
train_df = pd.read_csv(root / "train.csv")

# ❌ 错误：使用 os.getcwd()
import os
train_df = pd.read_csv(os.path.join(os.getcwd(), "train.csv"))

# ❌ 错误：使用 Path.cwd()
train_df = pd.read_csv(Path.cwd() / "train.csv")

# ❌ 错误：硬编码路径
train_df = pd.read_csv("/home/user/data/train.csv")
```

**原因**：AlphaEvolve 会将程序复制到临时目录执行，只有 root 参数指向真实数据位置。

### 4. 返回值要求

**必须返回**：包含评估指标的 dict

```python
def main(root):
    # ... 训练逻辑
    
    # 在测试集上评估
    predictions = model.predict(X_test)
    
    # 计算指标（必须是测试集上的）
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    # 返回指标 dict
    return {
        "rmse": rmse,
        "mape": mape
    }
```

**要求**：
- 只返回 dict，不返回模型、DataFrame 等其他对象
- 返回值指标必须是在**测试集**上计算的
- 返回值不能使用训练集或训练集划分的验证集的指标
- 指标 key 必须与 judge.py 中使用的一致

**常用指标命名**：
- 回归：`rmse`, `mse`, `mae`, `mape`, `rrmse`, `r2`
- 分类：`accuracy`, `precision`, `recall`, `f1`, `auc`, `log_loss`

### 5. 数据泄露防范（重要）

**数据泄露**是指测试集信息在训练过程中被错误使用，会导致评估指标虚高、模型实际表现差。

**禁止的做法**：

```python
# ❌ 错误：测试集参与 fit
scaler.fit(pd.concat([train_df, test_df]))  # 测试集信息泄露！

# ❌ 错误：先合并再 fit_transform
all_data = pd.concat([train_df, test_df])
all_data = scaler.fit_transform(all_data)  # 测试集信息泄露！

# ❌ 错误：使用全局统计量
mean_val = pd.concat([train_df, test_df])['col'].mean()  # 包含测试集！
train_df['col'] = train_df['col'].fillna(mean_val)

# ❌ 错误：在 EVOLVE-BLOCK 内对测试集 fit
# EVOLVE-BLOCK-START
def process(train_df, test_df):
    encoder.fit(test_df)  # 进化可能产生这种错误代码！
# EVOLVE-BLOCK-END
```

**正确的做法**：

```python
# ✅ 正确：只在训练集上 fit
scaler.fit(train_df[feature_cols])
train_scaled = scaler.transform(train_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])  # 测试集只 transform

# ✅ 正确：统计量只从训练集计算
mean_val = train_df['col'].mean()  # 只用训练集！
train_df['col'] = train_df['col'].fillna(mean_val)
test_df['col'] = test_df['col'].fillna(mean_val)  # 应用相同的值

# ✅ 正确：编码器只在训练集上 fit
encoder.fit(train_df[cat_cols])
train_df[cat_cols] = encoder.transform(train_df[cat_cols])
test_df[cat_cols] = encoder.transform(test_df[cat_cols])
```

**EVOLVE-BLOCK 内的安全规范**：
- EVOLVE-BLOCK 内的函数应该只接收训练数据进行 fit
- 测试数据的处理应该在 EVOLVE-BLOCK 外，或只调用 transform
- 返回的 encoder/scaler 对象用于后续处理测试集

### 6. 代码结构模板

```python
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

LOGGER = logging.getLogger(__name__)

# EVOLVE-BLOCK-START
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据预处理（会被进化）"""
    df = df.copy()
    # 特征工程
    return df

def build_model():
    """模型构建（会被进化）"""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, random_state=42)

def train_and_evaluate(train_df, test_df, target_col):
    """训练和评估（会被进化）"""
    # 预处理
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    
    # 准备特征和标签
    feature_cols = [c for c in train_df.columns if c != target_col]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # 训练
    model = build_model()
    model.fit(X_train, y_train)
    
    # 预测和评估（测试集）
    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    
    return {"rmse": rmse}
# EVOLVE-BLOCK-END

def main(root):
    """AlphaEvolve 入口"""
    data_dir = Path(root)
    
    # 加载数据
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    LOGGER.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # 训练和评估
    metrics = train_and_evaluate(train_df, test_df, target_col="target")
    
    LOGGER.info(f"Metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = main(".")
    print(result)
```

## 常见错误

### 错误 1：返回值包含非指标内容
```python
# ❌ 错误
return model, {"rmse": 0.5}

# ✅ 正确
return {"rmse": 0.5}
```

### 错误 2：使用验证集指标而非测试集
```python
# ❌ 错误：在训练集划分验证集评估
X_train, X_val, y_train, y_val = train_test_split(X, y)
model.fit(X_train, y_train)
val_rmse = ...  # 这是验证集指标
return {"rmse": val_rmse}

# ✅ 正确：在独立测试集评估
model.fit(X_train, y_train)
test_rmse = ...  # 这是测试集指标
return {"rmse": test_rmse}
```

### 错误 3：EVOLVE-BLOCK 位置不当
```python
# ❌ 错误：把数据加载放入 EVOLVE-BLOCK
# EVOLVE-BLOCK-START
def main(root):
    train_df = pd.read_csv(Path(root) / "train.csv")  # 数据加载不应被进化
# EVOLVE-BLOCK-END

# ✅ 正确：只把核心算法放入
def main(root):
    train_df = pd.read_csv(Path(root) / "train.csv")  # 固定逻辑
    # EVOLVE-BLOCK-START
    features = engineer_features(train_df)  # 可进化逻辑
    # EVOLVE-BLOCK-END
```
