## 启动命令
- 算法优化脚本只需关注输入和输出，/inputs 和 /outputs 作为绝对路径。严格固定传入参数，训练参数从 config.json 中获取，其他数据可从 /inputs 读取（如MLE数据）  。FastAPI 管理 Task ID。
```shell
# Don't make the script guess where data is. Tell it explicitly.
python main.py \
    --config_file /inputs/config.json \
    --train_file /inputs/train.jsonl \
    --eval_file /inputs/test.jsonl \
    --reward_def /inputs/reward_definition.py \
    --input_dir /inputs \
    --output_dir /outputs
```

## 输入数据
- 使用 OpenAI 格式的 messages，其中 reward_meta 字段可以根据任务定制化。
```json
// Row 1: Simple QA
{"id": "uuid-1", "messages": [{"role": "user", "content": "Fix this bug..."}], "ground_truth": "<code>..."}
// Row 2: Contextual RL / Few-shot
{"id": "uuid-2", "messages": [{"role": "system", "content": "You are Linux."}, {"role": "user", "content": "Hi"}], "reward_meta": {"constraints": ["no_sudo"]}}
```

- reward_definition.py 强制格式严格统一的接口。运行时作为模块导入标准函数。
```python
# The file MUST implement this function signature.
# If it crashes, the worker catches the exception and kills the job.
def compute_reward(prediction: str, ground_truth: str, metadata: dict) -> float:
    # Your logic here
    return 1.0
```

- 统一的输入文件结构，将用户上传的 zip 解压到 `/hpc_data/daguan_data/rl_backend/${task_id}/inputs`
```shell
/inputs/
├── train.jsonl
├── test.jsonl
├── agent.py
├── others
```

## 配置文件
- 根据优化任务定义，最多包含2层（便于前端可视化）。
```json
{
    "algorithm": "ppo",  // or "dspy_mipro", "reflexion"
    "model": {
        "base_model": "llama-3-8b",
        "adapter_path": "optional/lora/path",
        "initial_template": "You are a helpful assistant..."
    },
    "training": {
        "learning_rate": 1e-5,
        "batch_size": 64,
        "max_steps": 1000,
        "gradient_accumulation": 2
    },
    "environment": {
        "max_turns": 5,
        "stop_tokens": ["EOS"]
    }
}
```

## 输出数据
- 统一的输出文件结构，可根据任务调整细节，均可被后端读取解析。
- 保存位置 `/hpc_data/daguan_data/rl_backend/${task_id}/outputs`

```shell
/outputs/
├── events.jsonl        <-- metrics log (Append Only!)
├── status.json         <-- Heartbeat & current state (Overwritten atomically)
├── stdout.log          <-- FastAPI 启动任务时自动写入
├── checkpoints/
│   ├── step-100/
│   │   ├── pytorch.bin
│   │   └── prompt_state.json  <-- For prompt tuning, save the discrete text
│   └── step-200/
└── final_result/
    └── best_model.json
```

- events.jsonl 每行保存一个记录，用于后端读取解析。
```json
{"step": 1, "type": "train", "loss": 2.3, "reward": 0.1, "timestamp": 1715000000}
{"step": 5, "type": "eval", "accuracy": 0.85, "timestamp": 1715000100, "extra": {"prompt": "You are superman"}}
{"step": 10, "type": "system", "gpu_util": 0.95, "memory": "12GB"}
```

- status.json 保存最新的训练状态，仅用于后端读取。
```json
{
    "state": "running", // running, failed, completed
    "current_step": 50,
    "total_steps": 100,
    "last_update": 1715000200,
    "error": null,
    "extra": {
            "final_prompt": "You are a superman.",
            "final_score": 99.99
        }
}
```
