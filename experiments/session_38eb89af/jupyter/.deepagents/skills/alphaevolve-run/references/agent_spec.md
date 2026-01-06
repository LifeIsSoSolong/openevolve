# agent.py 规范说明

## 概述

agent.py 是 AlphaEvolve 进化的目标程序。AlphaEvolve 会在每轮迭代中修改 `EVOLVE-BLOCK` 内的代码，生成新版本并评估效果。

## 任务类型规范

共创时可参考模板（只借用结构，不改用户核心逻辑）：
- `assets/agent_mle.py`
- `assets/agent_prompt.py`

### MLE 任务

**必须满足**：
1. 包含 `main(root)` 函数，接收 `root` 参数（数据目录的绝对路径，由judge.py在调用main(root)时传入）
2. main(root) 函数必须返回 **仅包含指标的 dict**，指标 key 可以自定义，但必须与 judge.py 计算 combined_score 时使用的指标一致

**原因**：AlphaEvolve 每轮进化生成的 program 会被放置在临时目录，如果使用program文件的相对路径，会导致无法找到数据。但是judge.py所在的路径是不变的，所以需要通过 `root` 参数把judge.py所在目录传给program，方便program找到原始数据。

**数据读取修复要求**：
- 结合输入目录中的数据和judge.py来修改agent.py
- agent.py中的所有路径，都必须基于 `root` 来拼接构建，不能依赖agent.py所在的工作目录或硬编码绝对路径。
- 若用户代码里使用了 `"train.csv"` / `"test.csv"` 等裸文件名，需改为 `Path(root) / "train.csv"`。
- 若核心逻辑函数内部自行读取数据，应在函数参数中显式传入 `root` 或 `train_path/test_path`。
- 修复时只改“路径构建/参数传递”，不改用户核心算法。
- 务必使用`root`来拼接工作和加载数据的路径，这个root其实就是judge.py所在的目录，judge把该目录传给agent.py，方便它找到数据
- **禁止** 使用 `__file__` / `Path(__file__)` / `os.getcwd()` / `Path.cwd()` 来确定数据目录
 - 如需自动修复路径，优先使用：`scripts/fix_agent_paths.py`

**返回值要求**：
- 只返回指标 dict，不返回模型、特征、日志等其它内容
- 指标 key 需与 judge.py 使用的 key 完全一致（例如返回 `val_rmse`，judge 也必须用 `val_rmse`）
- 建议优先使用通用指标名（如 `rmse`/`rrmse`/`mape`），但不强制
- 要求返回的指标必须是在测试集上的指标，不能是在训练集/训练集中拆分的验证集的指标 （强制使用测试集指标）

**规范结构**：

```python
from pathlib import Path

# EVOLVE-BLOCK-START
def some_logic():
    """这里是会被进化的代码"""
    pass
# EVOLVE-BLOCK-END

def main(root):
    """
    主入口函数
    
    Args:
        root: 数据目录路径（包含训练和测试数据文件）
    
    Returns:
        dict: 仅包含评估指标，如 {"rmse": 0.5, "rrmse": 12.0, "mape": 10.0}
    """
    data_dir = Path(root)
    # 根据实际数据文件名加载（支持多种格式）
    train_path = data_dir / "train.csv"  # 或其他格式如 train.parquet
    test_path = data_dir / "test.csv"    # 或其他格式如 test.parquet
    
    # 调用进化块中的逻辑
    # ...
    
    return {"rmse": rmse, "rrmse": rrmse, "mape": mape}

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    metrics = main(".")
    print(metrics)
```

### Prompt 任务

**必须满足**：
1. 包含 `get_prompt_generate_press()` 函数
2. 函数返回 `(system_prompt, user_prompt_template)` 元组

**规范结构**：

```python
# user_prompt_template 由评估器填充，不要修改
user_prompt_template = """
## 稿件类型：
{interview_type}

## 采访资料：
{interview_context}
"""

def get_prompt_generate_press():
    # EVOLVE-BLOCK-START
    system_prompt = """
    你是一名专业的新闻稿撰写人员...
    """
    # EVOLVE-BLOCK-END
    return system_prompt, user_prompt_template

if __name__ == "__main__":
    print(get_prompt_generate_press())
```

## EVOLVE-BLOCK 标记

AlphaEvolve 只会修改 `# EVOLVE-BLOCK-START` 和 `# EVOLVE-BLOCK-END` 之间的代码。

**规则**：
- 标记必须成对出现
- 可以有多个 EVOLVE-BLOCK
- 标记外的代码不会被修改

## 自动包装示例

### 情况 1：MLE 任务缺少 main(root)

**原始代码**（用户上传）：

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_model(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # ... 训练逻辑
    return {"rmse": 0.5, "rrmse": 12.0, "mape": 8.0}
```

**包装后**：

```python
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# EVOLVE-BLOCK-START
def train_model(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # ... 训练逻辑
    return {"rmse": 0.5, "rrmse": 12.0, "mape": 8.0}
# EVOLVE-BLOCK-END

def main(root):
    """AlphaEvolve 入口函数"""
    data_dir = Path(root)
    return train_model(
        str(data_dir / "train.csv"),
        str(data_dir / "test.csv")
    )

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    metrics = main(".")
    print(metrics)
```

### 情况 2：缺少 EVOLVE-BLOCK 标记

如果用户代码没有标记，需要识别核心逻辑并添加标记。通常将主要的可优化函数包裹在 EVOLVE-BLOCK 中。

## 校验脚本使用

```bash
python validate_agent.py --input-dir /path/to/inputs --task-type mle
```

输出示例：

```
[OK] Found main(root) function
[OK] EVOLVE-BLOCK markers found
[OK] agent.py is valid for MLE task
```

或：

```
[ERROR] Missing main(root) function
[SUGGESTION] Wrap existing logic and add main(root) entry point
```
