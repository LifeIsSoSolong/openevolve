---
name: agentic-rl
description: Train reinforcement learning agents using AgentLightning/VERL with PPO. Validates data, configures training, executes training runs. Use for agent training, RL training, PPO, reinforcement learning, VERL.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# 智能体强化学习训练

使用 AgentLightning 框架运行强化学习训练任务，自动进行数据验证、资源检查和配置。

## 概述

本技能支持使用 VERL/PPO 在自定义数据集上训练智能体。执行前会验证输入数据、检查 GPU 可用性、确认训练参数，确保安全启动训练。

## 必需的输入文件

训练需要在 `EVO_INPUT_DIR` 中准备以下文件：

**数据文件：**
- `train.jsonl` - 训练数据集（JSON Lines 格式）
- `test.jsonl` - 评估/测试数据集

**配置文件：**
- `config.json` - 训练配置（详见 [config_schema.md](references/config_schema.md)）
- `agent.json` / `agent.jsonl` - 智能体元数据，包含 `model_name` 字段
- `task.goal` - 智能体的提示模板
- `judge.py` - 奖励函数，实现 `compute_reward(response, ground_truth, metadata)` 函数

**数据格式和详细示例请参见：** [examples.md](examples.md)

## 工作流程

**CRITICAL：必须严格按照顺序执行，特别是 GPU 检查（步骤 3）是启动训练的前置条件。**

### 步骤 1：检查输入目录和文件

检查 `$EVO_INPUT_DIR` 是否设置：
```bash
echo $EVO_INPUT_DIR
ls -la $EVO_INPUT_DIR
```

验证所有必需文件存在：`train.jsonl`、`test.jsonl`、`config.json`、`agent.json`、`task.goal`、`judge.py`

**如果缺失：** 告知用户需要准备的文件（参见 [examples.md](examples.md) 的格式说明）。

### 步骤 2：验证数据格式

运行验证脚本：
```bash
python <SKILL_ROOT>/scripts/validate_data.py $EVO_INPUT_DIR
```

检查输出，确保所有文件格式正确。**如果验证失败：** 报告具体错误并要求用户修复。

### 步骤 3：检查 GPU 可用性 ⚠️ **【必须执行】**

**这是最关键的步骤！必须在启动训练前检查 GPU 资源。**

首先从 `config.json` 读取所需 GPU 数量：

```python
import json
with open(f"{input_dir}/config.json") as f:
    config = json.load(f)
    required_gpus = config.get("trainer", {}).get("n_gpus_per_node", 4)
```

然后使用 GPU 检查脚本验证资源：

```bash
python <SKILL_ROOT>/scripts/check_gpu.py --required $required_gpus
```

**脚本会自动：**
1. 获取所有 GPU 的状态（内存使用、利用率）
2. 识别可用 GPU（利用率 < 20% 且内存使用 < 10%）
3. 比较可用 GPU 与需求
4. 以表格形式显示 GPU 状态

**【关键判断】根据脚本退出码决定：**
- **退出码 0** → GPU 充足，继续执行
- **退出码 1** → GPU 不足，**停止执行**，告知用户：
  - 需要等待 GPU 释放
  - 或修改 `config.json` 中的 `trainer.n_gpus_per_node`
  - 或终止占用 GPU 的其他进程

**⛔ 如果 GPU 不足，绝对不能继续后续步骤！**

### 步骤 4：验证和选择模型

读取配置的模型名称：

```python
import json
with open(f"{input_dir}/agent.json") as f:
    agent_data = json.load(f)
    configured_model = agent_data.get("model_name")
```

使用模型检查脚本验证：

```bash
python <SKILL_ROOT>/scripts/check_model.py --model "$configured_model"
```

**脚本会自动：**
1. 列出所有可用模型
2. 检查指定模型是否存在
3. 显示模型路径

**根据结果处理：**

- **模型存在（退出码 0）**：继续下一步
- **模型不存在（退出码 1）**：
  1. 脚本已显示所有可用模型列表
  2. 使用 **AskUserQuestion** 让用户从可用模型中选择
  3. 更新 `agent.json` 中的 `model_name` 字段：
     ```python
     agent_data["model_name"] = selected_model
     with open(f"{input_dir}/agent.json", "w") as f:
         json.dump(agent_data, f, indent=2)
     ```
  4. 重新运行检查脚本验证

### 步骤 5：审查和确认训练配置

从 `config.json` 提取并显示关键参数：

```python
trainer = config.get("trainer", {})
data = config.get("data", {})
actor = config.get("actor_rollout_ref", {}).get("actor", {})

print("训练配置：")
print(f"  模型: {configured_model}")
print(f"  训练轮数: {trainer.get('total_epochs', 5)}")
print(f"  批大小: {data.get('train_batch_size', 16)}")
print(f"  学习率: {actor.get('optim', {}).get('lr', 1e-6)}")
print(f"  GPU 数量: {trainer.get('n_gpus_per_node', 4)}")
print(f"  评估频率: 每 {trainer.get('test_freq', 10)} 步")
print(f"  保存频率: 每 {trainer.get('save_freq', 30)} 步")
```

**询问用户确认：** "以上配置正确吗？是否开始训练？"

### 步骤 6：准备输出目录

```python
output_dir = os.getenv("EVO_OUTPUT_DIR") or f"{input_dir}/../outputs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
```

### 步骤 7：后台启动训练 ⚠️ **【必须后台执行】**

**仅在所有检查通过后才执行！训练任务必须在后台运行。**

使用启动脚本在后台执行训练：

```bash
bash <SKILL_ROOT>/scripts/start_training.sh \
    --input-dir "$EVO_INPUT_DIR" \
    --output-dir "$EVO_OUTPUT_DIR" \
    --config-file "$EVO_INPUT_DIR/config.json"
```

**脚本会自动：**
1. 检查是否已有训练在运行（防止重复启动）
2. 创建必要的输出目录
3. 使用 `nohup` 在后台启动训练
4. 将所有输出重定向到 `$EVO_OUTPUT_DIR/training.log`
5. 保存进程 ID 到 `$EVO_OUTPUT_DIR/training.pid`
6. 创建状态文件 `$EVO_OUTPUT_DIR/training.status`

**输出文件：**
- `training.log` - 训练日志（stdout + stderr）
- `training.pid` - 进程 ID
- `training.status` - 训练状态（JSON 格式）
- `checkpoints/` - 模型检查点目录

**启动成功后，告知用户：**
- 训练已在后台启动
- 进程 ID
- 日志文件位置
- 如何查看状态和日志

### 步骤 8：监控训练状态

训练启动后，提供以下监控方式：

#### 8.1 查看训练状态

使用状态检查脚本：

```bash
python <SKILL_ROOT>/scripts/check_status.py --output-dir "$EVO_OUTPUT_DIR"
```

**脚本会显示：**
- 训练状态（运行中/已停止/失败）
- 进程信息（PID、CPU、内存使用）
- 运行时长
- GPU 使用情况
- 最近的日志输出（默认最后 20 行）

#### 8.2 查看实时日志

```bash
# 查看完整日志最后 50 行
tail -n 50 $EVO_OUTPUT_DIR/training.log

# 实时跟踪日志
tail -f $EVO_OUTPUT_DIR/training.log

# 搜索特定内容
grep "reward" $EVO_OUTPUT_DIR/training.log | tail -20
```

#### 8.3 监控 GPU 使用

```bash
# 实时监控 GPU
watch -n 2 nvidia-smi

# 查看训练进程的 GPU 使用
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### 步骤 9：管理训练任务（可选）

#### 9.1 停止训练

如果需要停止训练：

```bash
python <SKILL_ROOT>/scripts/stop_training.py --output-dir "$EVO_OUTPUT_DIR"
```

优雅停止（推荐）：发送 SIGTERM 信号，等待进程清理资源。

强制停止（进程无响应时）：

```bash
python <SKILL_ROOT>/scripts/stop_training.py --output-dir "$EVO_OUTPUT_DIR" --force
```

#### 9.2 重新启动训练

如果训练中断，可以从检查点恢复（需要在 config.json 中配置检查点路径）。

#### 9.3 检查训练结果

训练完成后，检查输出目录：

```bash
# 查看所有检查点
ls -lh $EVO_OUTPUT_DIR/checkpoints/

# 查看最新检查点
ls -t $EVO_OUTPUT_DIR/checkpoints/ | head -1
```

## 输出结构

训练期间和完成后，`EVO_OUTPUT_DIR` 包含：
```
EVO_OUTPUT_DIR/
├── training.log            # 训练日志（stdout + stderr）
├── training.pid            # 进程 ID（训练运行时）
├── training.status         # 训练状态（JSON 格式）
├── checkpoints/            # 模型检查点目录
│   ├── global_step100/
│   ├── global_step200/
│   └── ...
└── metrics.jsonl           # 训练指标（如果配置）
```

**文件说明：**
- `training.log` - 所有训练输出，包括进度、奖励、损失等
- `training.pid` - 当前训练进程的 PID（训练完成后会被删除）
- `training.status` - JSON 格式的状态文件，包含：
  - `status`: "starting" | "running" | "stopped" | "failed"
  - `start_time`: 开始时间（ISO 8601 格式）
  - `end_time`: 结束时间（如果已完成）
  - `pid`: 进程 ID
  - `input_dir`, `output_dir`, `config_file`: 路径信息

## 错误处理

**启动前检查（强制）：**
1. **GPU 不足** → 停止执行，告知用户等待或调整配置
2. **模型不存在** → 展示可用模型，让用户选择
3. **数据格式错误** → 显示错误详情，要求修复后重试
4. **文件缺失** → 列出缺失文件，提供格式说明

**训练中错误：**
- **OOM（内存溢出）** → 建议减小 `train_batch_size` 或启用参数卸载
- **训练崩溃** → 检查 `$EVO_OUTPUT_DIR/training.log`
- **CUDA 错误** → 重新检查 GPU 状态和配置

## 常见配置调整

详细配置选项和示例见 [config_schema.md](references/config_schema.md) 和 [examples.md](examples.md)。

**快速调整：**
- 减少内存：降低 `train_batch_size`，启用 `fsdp_config.param_offload`
- 加快训练：增大 `train_batch_size`，减少 `test_freq`
- 提高精度：增加 `total_epochs`，降低学习率

## 重要提醒

⚠️ **必须严格执行的检查：**
1. **GPU 可用性检查**（步骤 3） - **绝对不能跳过**
2. **模型路径验证**（步骤 4） - 必须存在
3. **用户确认训练配置**（步骤 5） - 获得明确同意
4. **后台执行训练**（步骤 7） - **必须使用后台执行**

⚠️ **安全启动条件：**
- ✓ 可用 GPU ≥ 配置要求的 GPU 数量
- ✓ 模型路径存在于 `/hpc_data/ktian/models/`
- ✓ 所有数据文件格式正确
- ✓ 用户明确确认启动训练
- ✓ 没有其他训练任务正在运行（检查 PID 文件）

⚠️ **后台执行要求：**
- **必须使用** `start_training.sh` 脚本在后台启动
- **不要**直接运行 `python scripts/main.py`（会阻塞会话）
- 训练启动后，立即告知用户：
  - 训练已在后台运行
  - 如何查看状态：`check_status.py`
  - 如何查看日志：`tail -f training.log`
  - 如何停止训练：`stop_training.py`
