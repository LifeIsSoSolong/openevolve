# judge.py 规范说明

## 概述

judge.py 是 AlphaEvolve 的评估器，负责对每轮进化生成的 program 进行评分。AlphaEvolve 会调用 `evaluate(program_path)` 函数获取评估结果。

## 核心要求

1. **必须实现** `evaluate(program_path: str)` 函数
2. **必须返回** 包含 `combined_score` 的结果（0~1，且验证要求 >0，越大越好）
3. combined_score 必须基于 agent.py 返回的指标计算（必须使用测试集/完全不参与训练过程的独立验证集上的指标），**使用的指标 key 必须与 agent.py 返回的 key 对齐**
4. 推荐返回 `EvaluationResult` 对象（来自 openevolve）

共创时可参考模板（只借用结构，不改用户核心逻辑）：
- `assets/judge_mle.py`
- `assets/judge_prompt.py`

## 返回值规范

### 方式 1：返回 EvaluationResult（推荐）

```python
from openevolve.evaluation_result import EvaluationResult

def evaluate(program_path: str):
    # ... 评估逻辑
    return EvaluationResult(
        metrics={
            "combined_score": 0.85,  # 必须，(0, 1]
            "rmse": 0.5,             # 可选，其他指标
            "accuracy": 0.9,         # 可选
        },
        artifacts={
            "predictions": "...",    # 可选，附加产物
        }
    )
```

### 方式 2：返回字典

```python
def evaluate(program_path: str):
    # ... 评估逻辑
    return {
        "combined_score": 0.85,  # 必须
        "error": None,           # 可选，错误信息
    }
```

## combined_score 计算

### MLE 任务示例

将原始指标转换为 0~1 分数：

```python
def calculate_combined_score(metrics: dict) -> float:
    """
    将原始指标转换为 combined_score
    
    常用转换方式：
    - 误差类指标（越小越好）：1 / (1 + error)
    - 准确率类指标（越大越好）：直接使用或归一化
    """
    rmse = metrics.get("rmse", float("inf"))
    rrmse = metrics.get("rrmse", float("inf"))
    mape = metrics.get("mape", float("inf"))
    
    # 示例：加权组合
    score_rmse = 1.0 / (1.0 + rmse)
    score_rrmse = 1.0 / (1.0 + rrmse / 100.0)
    score_mape = 1.0 / (1.0 + mape / 100.0)
    
    combined = (score_rmse + score_rrmse + score_mape) / 3.0
    return max(0.0, min(1.0, combined))
```
务必以 agent.py 返回的指标为原始指标计算 combined_score，不能使用常数或与指标无关的值。

### Prompt 任务示例

通常由 LLM 评估，将评分归一化：

```python
def calculate_combined_score(llm_score: float) -> float:
    """
    LLM 评分通常是 0-10，转换为 0-1
    """
    return max(0.0, min(1.0, llm_score / 10.0))
```

## MLE 任务 judge.py 模板

```python
"""
AlphaEvolve evaluator for MLE task.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

from openevolve.evaluation_result import EvaluationResult

ROOT = Path(__file__).resolve().parent


def _load_module(program_path: str):
    """动态加载候选程序模块"""
    module_name = f"candidate_module_{Path(program_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def calculate_combined_score(metrics: Dict[str, Any]) -> float:
    """将原始指标转换为 combined_score (0~1, 且>0)"""
    # TODO: 根据实际指标调整计算方式
    rmse = float(metrics.get("rmse", float("inf")))
    mape = float(metrics.get("mape", float("inf")))
    
    score_rmse = 1.0 / (1.0 + rmse)
    score_mape = 1.0 / (1.0 + mape / 100.0)  # MAPE 通常是百分比
    
    combined = 0.5 * score_rmse + 0.5 * score_mape
    return max(0.0, min(1.0, combined))


def evaluate(program_path: str) -> EvaluationResult:
    """
    AlphaEvolve 评估入口
    
    Args:
        program_path: 候选程序路径
    
    Returns:
        EvaluationResult with combined_score
    """
    try:
        module = _load_module(program_path)
        
        if not hasattr(module, "main"):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "main(root) not found"}
            )
        
        # 调用候选程序的 main 函数，传入数据目录
        raw_metrics = module.main(ROOT)
        
        # 计算 combined_score
        combined = calculate_combined_score(raw_metrics)
        
        metrics = {
            "combined_score": combined,
            **raw_metrics,
        }
        
        return EvaluationResult(metrics=metrics)
        
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": str(e)}
        )


if __name__ == "__main__":
    # 本地测试
    result = evaluate(str(ROOT / "agent.py"))
    print(f"Evaluation Result: {result}")
```

## Prompt 任务 judge.py 模板

```python
"""
AlphaEvolve evaluator for Prompt task.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from openevolve.evaluation_result import EvaluationResult

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.jsonl"
TEST_PATH = ROOT / "test.jsonl"


def _load_module(program_path: str):
    """动态加载候选程序模块"""
    module_name = f"candidate_module_{Path(program_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def evaluate(program_path: str) -> EvaluationResult:
    """
    AlphaEvolve 评估入口
    """
    try:
        # 导入生成和评估模块（用户提供）
        from generate_press_agent import generate_press_agent
        from evaluate_press_agent import evaluate_press_agent
        
        # 加载候选程序
        module = _load_module(program_path)
        
        if not hasattr(module, "get_prompt_generate_press"):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "get_prompt_generate_press() not found"}
            )
        
        system_prompt, user_prompt_template = module.get_prompt_generate_press()
        
        # 读取训练数据
        train_data = _read_jsonl(TRAIN_PATH)
        
        # 评估每个样本
        scores = []
        for sample in train_data:
            user_prompt = user_prompt_template.format(**sample)
            
            # 生成内容
            generated = generate_press_agent(
                model_name="...",  # TODO: 配置模型
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            
            # 评估生成内容
            result = evaluate_press_agent(
                model_name="...",  # TODO: 配置模型
                generated_press=generated,
                reference_press=sample.get("ground_truth", ""),
            )
            
            scores.append(result.get("combined_score", 0.0))
        
        combined = sum(scores) / len(scores) if scores else 0.0
        
        return EvaluationResult(
            metrics={"combined_score": combined}
        )
        
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": str(e)}
        )


if __name__ == "__main__":
    result = evaluate(str(ROOT / "agent.py"))
    print(f"Evaluation Result: {result}")
```

## 自动生成 judge.py

使用脚本自动生成：

```bash
python generate_judge.py --input-dir /path/to/inputs --task-type mle
```

脚本会：
1. 分析 agent.py 的返回值结构
2. 读取数据文件确定字段
3. 生成适配的 judge.py

## 验证 judge.py

```bash
python validate_judge.py --input-dir /path/to/inputs
```

验证步骤：
1. 检查 `evaluate()` 函数存在
2. 运行 `python judge.py`
3. 检查返回值包含 `combined_score`
4. 检查 `combined_score` 在 (0, 1] 范围内
5. 确认 combined_score 使用的指标 key 与 agent.py 返回的指标 key 一致
