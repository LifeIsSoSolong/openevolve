# Agentic RL 训练示例

本文档提供详细的代码示例和使用场景说明。

## 数据格式示例

### train.jsonl / test.jsonl 格式

每行是一个独立的 JSON 对象：

```json
{
  "id": "uuid-1",
  "messages": [
    {
      "role": "user",
      "content": "问题文本"
    }
  ],
  "ground_truth": "期望答案"
}
```

**完整示例：**

```json
{"id": "math-001", "messages": [{"role": "user", "content": "计算 25 * 4"}], "ground_truth": "100"}
{"id": "math-002", "messages": [{"role": "user", "content": "求解方程 2x + 5 = 15"}], "ground_truth": "x = 5"}
{"id": "math-003", "messages": [{"role": "user", "content": "计算 √144"}], "ground_truth": "12"}
```

### agent.json 格式

```json
{
  "model_name": "Qwen2.5-7B-Instruct"
}
```

或使用 agent.jsonl：

```jsonl
{"model_name": "Qwen2.5-7B-Instruct"}
```

### config.json 示例

最小配置示例：

```json
{
  "algorithm": {
    "adv_estimator": "grpo",
    "use_kl_in_reward": false
  },
  "data": {
    "train_batch_size": 16,
    "max_prompt_length": 2048,
    "max_response_length": 2048
  },
  "actor_rollout_ref": {
    "rollout": {
      "n": 2,
      "gpu_memory_utilization": 0.6
    },
    "actor": {
      "optim": {
        "lr": 1e-6
      }
    }
  },
  "trainer": {
    "n_gpus_per_node": 4,
    "total_epochs": 5,
    "test_freq": 10,
    "save_freq": 30
  }
}
```

### task.goal 示例

```
You are a helpful assistant. Please solve the following problem:

{prompt}

Provide your answer directly without explanation.
```

### judge.py 示例

```python
def compute_reward(response: str, ground_truth: str, metadata: dict) -> float:
    """
    计算奖励分数。

    Args:
        response: 模型的响应
        ground_truth: 标准答案
        metadata: 额外的元数据

    Returns:
        奖励分数 (0.0 到 1.0)
    """
    # 简单的完全匹配
    if response.strip().lower() == ground_truth.strip().lower():
        return 1.0

    # 部分匹配
    if ground_truth.lower() in response.lower():
        return 0.5

    return 0.0
```

**更复杂的奖励函数示例：**

```python
import re
from difflib import SequenceMatcher

def compute_reward(response: str, ground_truth: str, metadata: dict) -> float:
    """数学问题的奖励函数"""

    # 提取数字答案
    def extract_number(text):
        match = re.search(r'-?\d+\.?\d*', text)
        return float(match.group()) if match else None

    response_num = extract_number(response)
    truth_num = extract_number(ground_truth)

    # 数值匹配
    if response_num is not None and truth_num is not None:
        if abs(response_num - truth_num) < 1e-6:
            return 1.0
        elif abs(response_num - truth_num) < 0.1:
            return 0.8

    # 文本相似度
    similarity = SequenceMatcher(None,
                                response.lower().strip(),
                                ground_truth.lower().strip()).ratio()

    return similarity
```

## 手动数据验证示例

### Python 脚本验证

```python
import json
from pathlib import Path

def validate_jsonl(file_path):
    """验证 JSONL 文件格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                # 检查必需字段
                assert 'messages' in data, f"行 {i}: 缺少 'messages' 字段"
                assert 'ground_truth' in data, f"行 {i}: 缺少 'ground_truth' 字段"

                # 检查 messages 格式
                assert isinstance(data['messages'], list), f"行 {i}: 'messages' 必须是列表"
                assert len(data['messages']) > 0, f"行 {i}: 'messages' 不能为空"

                for msg in data['messages']:
                    assert 'role' in msg, f"行 {i}: 消息缺少 'role' 字段"
                    assert 'content' in msg, f"行 {i}: 消息缺少 'content' 字段"

                print(f"✓ 行 {i}: 有效")

            except json.JSONDecodeError as e:
                print(f"✗ 行 {i}: JSON 解析错误 - {e}")
            except AssertionError as e:
                print(f"✗ {e}")

# 使用示例
validate_jsonl("train.jsonl")
validate_jsonl("test.jsonl")
```

### 配置参数检查

```python
import json

with open("config.json") as f:
    config = json.load(f)

# 提取关键参数
trainer = config.get("trainer", {})
data = config.get("data", {})
actor = config.get("actor_rollout_ref", {}).get("actor", {})

print("训练配置：")
print(f"  总轮数: {trainer.get('total_epochs', 5)}")
print(f"  批大小: {data.get('train_batch_size', 16)}")
print(f"  学习率: {actor.get('optim', {}).get('lr', 1e-6)}")
print(f"  GPU数量: {trainer.get('n_gpus_per_node', 4)}")
print(f"  测试频率: {trainer.get('test_freq', 10)} 步")
print(f"  保存频率: {trainer.get('save_freq', 30)} 步")
```

## 使用场景

### 场景 1：完整的训练流程

**用户请求：**
```
用户："在我的数学数据集上运行 RL 训练"
```

**执行流程：**
1. 检查 `$EVO_INPUT_DIR` 是否设置
2. 列出目录中的文件，验证所有必需文件存在
3. 运行 `validate_data.py` 验证格式
4. 读取 `config.json` 并向用户确认关键参数
5. 检查 GPU 可用性
6. 执行训练命令
7. 监控训练进度

### 场景 2：用户需要准备数据

**用户请求：**
```
用户："我想训练一个智能体"
```

**执行流程：**
1. 检查 `$EVO_INPUT_DIR`（未设置或为空）
2. 告知用户需要准备的文件和格式
3. 提供数据格式示例
4. 等待用户上传文件
5. 文件准备好后继续验证流程

### 场景 3：自定义训练参数

**用户请求：**
```
用户："训练 10 个 epoch，批大小改为 32，学习率用 5e-6"
```

**执行流程：**
1. 读取现有的 `config.json`
2. 更新参数：
   - `trainer.total_epochs = 10`
   - `data.train_batch_size = 32`
   - `actor_rollout_ref.actor.optim.lr = 5e-6`
3. 保存修改后的配置
4. 向用户确认修改
5. 继续训练流程

### 场景 4：从检查点恢复训练

**用户请求：**
```
用户："从上次的检查点继续训练"
```

**执行流程：**
1. 检查 `$EVO_OUTPUT_DIR/checkpoints/` 目录
2. 列出可用的检查点
3. 让用户选择检查点或自动选择最新的
4. 修改配置以从检查点加载
5. 继续训练

## 常见配置调整示例

### 减少内存使用

适用于 GPU 内存不足的情况：

```json
{
  "data": {
    "train_batch_size": 8
  },
  "actor_rollout_ref": {
    "rollout": {
      "gpu_memory_utilization": 0.4,
      "log_prob_micro_batch_size_per_gpu": 2
    },
    "actor": {
      "ppo_micro_batch_size_per_gpu": 2,
      "fsdp_config": {
        "param_offload": true,
        "optimizer_offload": true
      }
    }
  }
}
```

### 加快训练速度

适用于快速实验：

```json
{
  "data": {
    "train_batch_size": 32
  },
  "trainer": {
    "test_freq": 50,
    "save_freq": 100,
    "total_epochs": 3
  }
}
```

### 提高训练精度

适用于追求最佳性能：

```json
{
  "trainer": {
    "total_epochs": 20,
    "test_freq": 5
  },
  "actor_rollout_ref": {
    "rollout": {
      "n": 4
    },
    "actor": {
      "optim": {
        "lr": 5e-7
      }
    }
  }
}
```

## 后台任务管理

### 启动后台训练

训练任务**必须**在后台运行，使用启动脚本：

```bash
# 设置环境变量
export EVO_INPUT_DIR="/path/to/input"
export EVO_OUTPUT_DIR="/path/to/output"

# 后台启动训练
bash <SKILL_ROOT>/scripts/start_training.sh \
    --input-dir "$EVO_INPUT_DIR" \
    --output-dir "$EVO_OUTPUT_DIR" \
    --config-file "$EVO_INPUT_DIR/config.json"
```

**成功启动后输出：**
```
========================================
启动 RL 训练任务
========================================
输入目录: /path/to/input
输出目录: /path/to/output
配置文件: /path/to/input/config.json
日志文件: /path/to/output/training.log
状态文件: /path/to/output/training.status
========================================

✓ 训练任务已成功启动！

进程 ID: 12345
日志文件: /path/to/output/training.log

查看实时日志：
  tail -f /path/to/output/training.log

检查训练状态：
  python <SKILL_ROOT>/scripts/check_status.py --output-dir /path/to/output

停止训练：
  python <SKILL_ROOT>/scripts/stop_training.py --output-dir /path/to/output
```

### 查看训练状态

使用状态检查脚本：

```bash
python <SKILL_ROOT>/scripts/check_status.py --output-dir "$EVO_OUTPUT_DIR"
```

**输出示例：**
```
================================================================================
训练任务状态
================================================================================

状态: running
开始时间: 2025-01-15T10:30:00Z
运行时长: 2小时 15分钟

输入目录: /path/to/input
输出目录: /path/to/output
配置文件: /path/to/input/config.json
日志文件: /path/to/output/training.log

进程 ID: 12345
进程状态: ✓ 运行中
CPU 使用: 350.2%
内存使用: 12.5%
运行时间: 2:15:30

--------------------------------------------------------------------------------
GPU 使用情况
--------------------------------------------------------------------------------
GPU   利用率      内存使用
--------------------------------------------------------------------------------
0      98%        22450 MB / 24576 MB ( 91.4%)
1      97%        22380 MB / 24576 MB ( 91.1%)
2      96%        22310 MB / 24576 MB ( 90.8%)
3      98%        22490 MB / 24576 MB ( 91.5%)
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
最近日志（最后 20 行）
--------------------------------------------------------------------------------
[2025-01-15 12:45:23] Epoch 3/5, Step 150
[2025-01-15 12:45:23] Average reward: 0.75
[2025-01-15 12:45:23] Policy loss: 0.023
[2025-01-15 12:45:23] Value loss: 0.015
...
--------------------------------------------------------------------------------

查看完整日志: tail -f /path/to/output/training.log
================================================================================
```

### 查看实时日志

```bash
# 实时跟踪日志（推荐）
tail -f $EVO_OUTPUT_DIR/training.log

# 查看最后 100 行
tail -n 100 $EVO_OUTPUT_DIR/training.log

# 搜索特定内容
grep "reward" $EVO_OUTPUT_DIR/training.log | tail -20
grep "loss" $EVO_OUTPUT_DIR/training.log | tail -20
grep "checkpoint" $EVO_OUTPUT_DIR/training.log
```

### 停止训练

**优雅停止（推荐）：**

```bash
python <SKILL_ROOT>/scripts/stop_training.py --output-dir "$EVO_OUTPUT_DIR"
```

**输出示例：**
```
================================================================================
停止训练任务
================================================================================

训练任务信息：
  进程 ID: 12345
  开始时间: 2025-01-15T10:30:00Z
  当前状态: running
  输出目录: /path/to/output

发送终止信号到进程 12345...
等待进程终止（最多 10 秒）...
  等待中... (1/10)
  等待中... (2/10)
✓ 进程已终止

✓ 训练任务已成功停止
   日志文件: /path/to/output/training.log
   检查点目录: /path/to/output/checkpoints/
================================================================================
```

**强制停止（进程无响应时）：**

```bash
python <SKILL_ROOT>/scripts/stop_training.py --output-dir "$EVO_OUTPUT_DIR" --force
```

### 检查检查点

```bash
# 列出所有检查点
ls -lh $EVO_OUTPUT_DIR/checkpoints/

# 查看最新检查点
ls -t $EVO_OUTPUT_DIR/checkpoints/ | head -1

# 检查点目录结构
tree $EVO_OUTPUT_DIR/checkpoints/global_step100/
```

### 查看训练状态文件

```bash
# 查看 JSON 格式的状态
cat $EVO_OUTPUT_DIR/training.status | python -m json.tool
```

**状态文件示例：**

```json
{
  "status": "running",
  "start_time": "2025-01-15T10:30:00Z",
  "input_dir": "/path/to/input",
  "output_dir": "/path/to/output",
  "config_file": "/path/to/input/config.json",
  "log_file": "/path/to/output/training.log",
  "pid": 12345
}
```

## 监控和调试

### GPU 使用监控

```bash
# 实时监控 GPU 使用
watch -n 2 nvidia-smi

# 查看训练进程的 GPU 使用
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# 查看特定进程的 GPU 使用
nvidia-smi | grep 12345
```

### 进程监控

```bash
# 查看进程信息
ps aux | grep "scripts/main.py"

# 查看进程树
pstree -p 12345

# 查看进程资源使用
top -p 12345
```

### 常见问题排查

**问题 1：训练启动失败**

```bash
# 检查日志
tail -n 50 $EVO_OUTPUT_DIR/training.log

# 检查状态文件
cat $EVO_OUTPUT_DIR/training.status
```

**问题 2：GPU 内存不足**

```bash
# 查看 GPU 内存使用
nvidia-smi

# 修改配置减小批大小
# 编辑 config.json，降低 train_batch_size
```

**问题 3：训练进程僵死**

```bash
# 检查进程是否响应
ps -p $(cat $EVO_OUTPUT_DIR/training.pid)

# 强制停止
python <SKILL_ROOT>/scripts/stop_training.py --output-dir "$EVO_OUTPUT_DIR" --force
```

**问题 4：无法启动新训练（已有任务在运行）**

```bash
# 检查现有任务状态
python <SKILL_ROOT>/scripts/check_status.py --output-dir "$EVO_OUTPUT_DIR"

# 如果旧任务已完成但未清理，手动清理 PID 文件
rm $EVO_OUTPUT_DIR/training.pid
```
