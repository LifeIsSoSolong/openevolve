---
name: llm-sft
description: 使用LLaMA-Factory来对基础的大模型进行指令微调（SFT），以提升模型的效果。它的功能是通过大量“输入–标准输出”示例，直接教会模型在特定任务上按你期望的方式作答，本质是对已有能力的定向强化和风格固化；它最适用于答案明确、评价标准稳定、输出格式或规则清晰的场景，如结构化输出、信息抽取、规则化改写、领域翻译或把现有人工/规则系统蒸馏成模型，而不适合目标模糊、策略性强或长时序决策的任务。
---

# llm-sft 运行

## 执行流程（严格按顺序）

### 0）告知用户skill的执行流程
    - 读取环境变量
    - 检查输入目录文件
    - 验证数据格式
    - GPU可用性
    - 验证和选择模型
    - 审查和确认训练配置
    - 审查和确认训练配置
    - 检查输出目录是否存在
    - 通过nohup后台启动训练任务
    - 训练状态查询监控（可选）
    - 管理训练任务（可选）

### 1) 读取环境变量

先从环境变量读取 EVO_INPUT_DIR 和 EVO_OUTPUT_DIR, 务必要先读取环境变量，此步不可跳过。

```bash
echo $EVO_INPUT_DIR
echo $$EVO_OUTPUT_DIR
```
读取到后，将其作为 input_dir 与 output_dir 使用。
INPUT_DIR=EVO_INPUT_DIR
OUTPUT_DIR=EVO_OUTPUT_DIR

### 2) 使用 ls 检查输入目录文件

使用 `ls` 检查 $EVO_INPUT_DIR 是否包含任务所需的所有文件。

```bash
ls -la $EVO_INPUT_DIR
```

根据任务类型检查所需文件是否存在：
  - agent.json
  - config.json
  - ds_z3_config.json
  - train.jsonl

若 $EVO_INPUT_DIR 文件缺失或者目录缺失，明确列出缺失项并告知需要上传哪些文件，用户上传后你再去 $EVO_INPUT_DIR 查找检查文件（直到所需文件都存在，再进行下一步）


### 3) 验证数据格式

运行验证脚本：
```bash
python <SKILL_ROOT>/scripts/validate_data.py $EVO_INPUT_DIR
```

检查输出，确保所有文件格式正确。**如果验证失败：** 报告具体错误并要求用户修复。


### 4) GPU可用性

首先从 `config.json` 读取所需 GPU 数量：

```python
import json
with open(f"{input_dir}/config.json") as f:
    config = json.load(f)
    required_gpus = config.get("n_gpus_per_node", 4)
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


### 5) 验证和选择模型

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


### 6) 审查和确认训练配置

审查和确认训练配置

从 `config.json` 提取并显示关键参数：

```python
stage = config.get("stage", {})
template = config.get("template", {})
save_strategy = config.get("save_strategy", {})
overwrite_output_dir = config.get("overwrite_output_dir",True)
save_only_model = config.get("save_only_model",True)
per_device_train_batch_size = config.get("per_device_train_batch_size",4)
learning_rate = config.get("learning_rate",1.0e-5)
num_train_epochs = config.get("num_train_epochs", 1.0)
n_gpus_per_node = config.get("n_gpus_per_node",4)


print("训练配置：")
print(f"  模型: {configured_model}")
print(f"  训练模版: {template}")
print(f"  保存频率: {save_strategy}")
print(f"  覆盖输出: {overwrite_output_dir}")
print(f"  是否只保留模型: {save_only_model}")
print(f"  每卡上batch样本数：{per_device_train_batch_size}")
print(f"  学习率：{learning_rate}")
print(f"  训练轮数：{num_train_epochs}")
print(f"  训练GPU数量：{n_gpus_per_node}")
```

**询问用户确认：** "以上配置正确吗？是否开始训练？"


### 7) 检查输出目录是否存在

检查环境变量`EVO_OUTPUT_DIR`目录（即输出目录`<OUTPUT_DIR>`）是否存在，若不存在，你进行创建，并告知用户
```bash
mkdir -p "<OUTPUT_DIR>"

```


### 8) 通过nohup后台启动训练任务

仅在所有检查通过后才执行！训练任务必须在后台运行。
使用启动脚本在后台执行训练：

```bash
bash <SKILL_ROOT>/scripts/start_training.sh \
    --config-file "$EVO_INPUT_DIR/config.json" \
    --input-dir "$EVO_INPUT_DIR" \
    --output-dir "$EVO_OUTPUT_DIR" \
```

**脚本会自动：**
1. 使用 `nohup` 在后台启动训练
2. 将所有输出重定向到 `$EVO_OUTPUT_DIR/run.log`
3. 保存进程 ID 到 `$EVO_OUTPUT_DIR/training.pid`
4. 创建状态文件 `$EVO_OUTPUT_DIR/training.status`

**输出文件：**
- `run.log` - 训练日志（stdout + stderr）
- `training.pid` - 进程 ID
- `training.status` - 训练状态（JSON 格式）

**启动成功后，告知用户：**
- 训练已在后台启动
- 进程 ID
- 日志文件位置
- 如何查看状态和日志


### 9) 训练状态查询监控（可选）

训练启动后，提供以下监控方式：

#### 9.1 查看训练状态

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

#### 9.2 查看实时日志

```bash
# 查看完整日志最后 50 行
tail -n 50 $EVO_OUTPUT_DIR/run.log

# 实时跟踪日志
tail -f $EVO_OUTPUT_DIR/run.log

# 搜索特定内容
grep "reward" $EVO_OUTPUT_DIR/run.log | tail -20
```

#### 9.3 监控 GPU 使用

```bash
# 实时监控 GPU
watch -n 2 nvidia-smi

# 查看训练进程的 GPU 使用
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
``` 

### 10) 管理训练任务（可选）

#### 10.1 停止训练

如果需要停止训练：

```bash
python <SKILL_ROOT>/scripts/stop_training.py --output-dir "$EVO_OUTPUT_DIR"
```

优雅停止（推荐）：发送 SIGTERM 信号，等待进程清理资源。

强制停止（进程无响应时）：

```bash
python <SKILL_ROOT>/scripts/stop_training.py --output-dir "$EVO_OUTPUT_DIR" --force
```

#### 10.2 检查训练结果

```bash
python "<SKILL_ROOT>/scripts/check_outputs.py" --output-dir "$EVO_OUTPUT_DIR"
```

##### 产出检查说明

脚本默认检查：
- all_results.json
- run.log
- checkpoint-* (目录)
- README.md

部分文件是迭代过程中逐步生成的；缺失时默认为仅告知。
