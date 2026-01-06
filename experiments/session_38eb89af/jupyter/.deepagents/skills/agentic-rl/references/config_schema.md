# 配置说明

本文档描述了 agentic-rl 训练的配置结构。

## 完整配置示例

```json
{
  "algorithm": {
    "adv_estimator": "grpo",
    "use_kl_in_reward": false
  },
  "agentlightning": {
    "port": 9999
  },
  "data": {
    "train_batch_size": 16,
    "max_prompt_length": 2048,
    "max_response_length": 2048,
    "truncation": "error"
  },
  "actor_rollout_ref": {
    "rollout": {
      "tensor_model_parallel_size": 1,
      "n": 2,
      "log_prob_micro_batch_size_per_gpu": 4,
      "multi_turn": {
        "format": "hermes"
      },
      "name": "vllm",
      "gpu_memory_utilization": 0.6,
      "engine_kwargs": {
        "vllm": {
          "enable_auto_tool_choice": false
        }
      }
    },
    "actor": {
      "ppo_mini_batch_size": 16,
      "ppo_micro_batch_size_per_gpu": 4,
      "optim": {
        "lr": 1e-6
      },
      "use_kl_loss": false,
      "kl_loss_coef": 0.0,
      "entropy_coeff": 0,
      "clip_ratio_low": 0.2,
      "clip_ratio_high": 0.3,
      "fsdp_config": {
        "param_offload": true,
        "optimizer_offload": true
      }
    },
    "ref": {
      "log_prob_micro_batch_size_per_gpu": 8,
      "fsdp_config": {
        "param_offload": true
      }
    },
    "model": {
      "use_remove_padding": true,
      "enable_gradient_checkpointing": true
    }
  },
  "trainer": {
    "n_gpus_per_node": 4,
    "val_before_train": true,
    "critic_warmup": 0,
    "logger": ["file"],
    "project_name": "AgentLightning",
    "experiment_name": "Agent_RL",
    "nnodes": 1,
    "test_freq": 10,
    "save_freq": 30,
    "total_epochs": 5
  }
}
```

## 配置部分

### algorithm

控制 RL 算法行为。

- **adv_estimator**（字符串）：优势估计器类型。选项：`"grpo"`、`"gae"`
- **use_kl_in_reward**（布尔值）：是否在奖励计算中包含 KL 散度

### agentlightning

AgentLightning 框架设置。

- **port**（整数）：AgentLightning 服务器端口

### data

数据加载和处理配置。

- **train_batch_size**（整数）：每个训练批次的样本数
- **max_prompt_length**（整数）：提示的最大 token 长度
- **max_response_length**（整数）：响应的最大 token 长度
- **truncation**（字符串）：截断策略。选项：`"error"`、`"left"`、`"right"`

### actor_rollout_ref

核心训练组件配置。

#### actor_rollout_ref.rollout

Rollout 生成设置。

- **tensor_model_parallel_size**（整数）：张量并行度
- **n**（整数）：每个提示的 rollout 样本数
- **log_prob_micro_batch_size_per_gpu**（整数）：对数概率计算的微批次大小
- **multi_turn.format**（字符串）：多轮对话格式。选项：`"hermes"`
- **name**（字符串）：推理引擎。选项：`"vllm"`
- **gpu_memory_utilization**（浮点数）：GPU 内存利用率（0-1）

#### actor_rollout_ref.actor

Actor（策略）网络设置。

- **ppo_mini_batch_size**（整数）：PPO 小批次大小
- **ppo_micro_batch_size_per_gpu**（整数）：每个 GPU 的 PPO 微批次大小
- **optim.lr**（浮点数）：学习率（例如 `1e-6`）
- **use_kl_loss**（布尔值）：是否使用 KL 损失
- **kl_loss_coef**（浮点数）：KL 损失系数
- **entropy_coeff**（浮点数）：熵奖励系数
- **clip_ratio_low**（浮点数）：PPO 裁剪比率下界
- **clip_ratio_high**（浮点数）：PPO 裁剪比率上界
- **fsdp_config**：全分片数据并行配置
  - **param_offload**（布尔值）：将参数卸载到 CPU
  - **optimizer_offload**（布尔值）：将优化器状态卸载到 CPU

#### actor_rollout_ref.ref

参考模型设置（用于 KL 惩罚）。

- **log_prob_micro_batch_size_per_gpu**（整数）：参考模型的微批次大小
- **fsdp_config.param_offload**（布尔值）：卸载参考模型参数

#### actor_rollout_ref.model

模型优化设置。

- **use_remove_padding**（布尔值）：移除填充以提高效率
- **enable_gradient_checkpointing**（布尔值）：启用梯度检查点以节省内存

### trainer

训练循环配置。

- **n_gpus_per_node**（整数）：每个节点的 GPU 数量
- **val_before_train**（布尔值）：在第一次训练步骤之前运行验证
- **critic_warmup**（整数）：critic 的预热步数（如果使用价值网络）
- **logger**（列表）：日志记录器类型。选项：`["console"]`、`["file"]`、`["console", "file"]`
- **project_name**（字符串）：日志记录的项目名称
- **experiment_name**（字符串）：日志记录的实验名称
- **nnodes**（整数）：分布式训练的节点数
- **test_freq**（整数）：每 N 步运行一次评估
- **save_freq**（整数）：每 N 步保存一次检查点
- **total_epochs**（整数）：训练的总轮数

## 常见修改

### 增加训练时长

修改 `trainer.total_epochs`：
```json
"trainer": {
  "total_epochs": 10
}
```

### 调整批大小

修改 `data.train_batch_size`：
```json
"data": {
  "train_batch_size": 32
}
```

### 更改学习率

修改 `actor_rollout_ref.actor.optim.lr`：
```json
"actor_rollout_ref": {
  "actor": {
    "optim": {
      "lr": 5e-6
    }
  }
}
```

### 更频繁地保存

修改 `trainer.save_freq` 和 `trainer.test_freq`：
```json
"trainer": {
  "test_freq": 5,
  "save_freq": 10
}
```

### 减少内存使用

启用卸载并减小批大小：
```json
"data": {
  "train_batch_size": 8
},
"actor_rollout_ref": {
  "rollout": {
    "gpu_memory_utilization": 0.5
  },
  "actor": {
    "ppo_micro_batch_size_per_gpu": 2,
    "fsdp_config": {
      "param_offload": true,
      "optimizer_offload": true
    }
  }
}
```

## 注意事项

- 模型路径会从 `agent.json` 自动设置（`/hpc_data/ktian/models/{model_name}`）
- 数据文件路径会从 `--input_dir` 参数自动设置
- 输出路径会从 `--output_dir` 参数自动配置
- 日志路径通过 `VERL_FILE_LOGGER_PATH` 环境变量设置
