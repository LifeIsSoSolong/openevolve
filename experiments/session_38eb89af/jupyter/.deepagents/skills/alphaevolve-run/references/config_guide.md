# config.json 配置指南

## 概述

config.json 是 AlphaEvolve 的运行配置文件，控制进化过程的各项参数。本指南说明各字段含义和调整建议。

## 配置结构

```json
{
    "algorithm": "alpha_evolve",
    "task_type": "mle|prompt",
    "max_iterations": 5,
    "checkpoint_interval": 1,
    "log_level": "INFO",
    "random_seed": 42,
    "llm": { ... },
    "database": { ... },
    "evaluator": { ... },
    "evolution_trace": { ... }
}
```

## 字段详解

### 基础配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `algorithm` | string | "alpha_evolve" | 算法类型，固定值 |
| `task_type` | string | - | 任务类型：`mle` 或 `prompt` |
| `max_iterations` | int | 5 | 最大进化轮数 |
| `checkpoint_interval` | int | 1 | 检查点保存间隔 |
| `log_level` | string | "INFO" | 日志级别：DEBUG/INFO/WARNING/ERROR |
| `random_seed` | int | 42 | 随机种子，用于复现 |

**调整建议**：
- `max_iterations`：初次测试用 3-5 轮，正式运行可设 10-20 轮
- `checkpoint_interval`：保持为 1，方便回溯

### LLM 配置

```json
"llm": {
    "primary_model": "gpt-5.2",
    "primary_model_weight": 0.8,
    "secondary_model": "gpt-5.2",
    "secondary_model_weight": 0.2,
    "temperature": 0.5,
    "max_tokens": 60000,
    "timeout": 300
}
```

| 字段 | 说明 |
|------|------|
| `primary_model` | 主模型名称 |
| `primary_model_weight` | 主模型采样权重 |
| `secondary_model` | 辅助模型名称 |
| `secondary_model_weight` | 辅助模型采样权重 |
| `temperature` | 生成温度，越高越随机 |
| `max_tokens` | 最大生成 token 数 |
| `timeout` | API 调用超时（秒） |

**调整建议**：
- `temperature`：0.3-0.7，太低缺乏多样性，太高可能生成无效代码
- `timeout`：复杂任务可增加到 600

### Database 配置

```json
"database": {
    "population_size": 16,
    "archive_size": 8,
    "num_islands": 2,
    "elite_selection_ratio": 0.25,
    "exploitation_ratio": 0.6
}
```

| 字段 | 说明 |
|------|------|
| `population_size` | 种群大小 |
| `archive_size` | 精英存档大小 |
| `num_islands` | 岛屿数量（并行进化分支） |
| `elite_selection_ratio` | 精英选择比例 |
| `exploitation_ratio` | 利用（vs 探索）比例 |

**调整建议**：
- MLE 任务：`population_size` 可设 16-32
- Prompt 任务：API 调用多，`population_size` 可设 8-16
- `exploitation_ratio`：0.6 平衡探索与利用

### Evaluator 配置

```json
"evaluator": {
    "timeout": 3000,
    "parallel_evaluations": 1
}
```

| 字段 | 说明 |
|------|------|
| `timeout` | 单次评估超时（秒） |
| `parallel_evaluations` | 并行评估数量 |

**调整建议**：
- MLE 任务：`timeout` 设 300-600 通常够用
- Prompt 任务：需要多次 API 调用，`timeout` 设 3000-30000
- `parallel_evaluations`：受限于 API 并发，通常设 1

### Evolution Trace 配置

```json
"evolution_trace": {
    "enabled": true,
    "format": "jsonl",
    "include_code": true,
    "include_prompts": true,
    "output_path": null,
    "buffer_size": 1,
    "compress": false
}
```

| 字段 | 说明 |
|------|------|
| `enabled` | 是否启用进化追踪 |
| `format` | 输出格式 |
| `include_code` | 是否记录代码 |
| `include_prompts` | 是否记录 prompt |
| `output_path` | 自定义输出路径（null 使用默认） |
| `buffer_size` | 写入缓冲大小 |
| `compress` | 是否压缩 |

**调整建议**：
- 保持 `enabled: true` 以便分析进化过程
- Prompt 任务可设 `include_prompts: false` 减少日志量

## 任务类型差异

### MLE 任务默认配置

```json
{
    "task_type": "mle",
    "database": {
        "population_size": 16,
        "archive_size": 8
    },
    "evaluator": {
        "timeout": 3000
    }
}
```

### Prompt 任务默认配置

```json
{
    "task_type": "prompt",
    "database": {
        "population_size": 12,
        "archive_size": 6
    },
    "evaluator": {
        "timeout": 30000
    },
    "evolution_trace": {
        "include_prompts": false
    }
}
```

## 共创流程

### 已有 config.json

1. 读取用户 config 和对应模板
2. 按 key 对比，汇报三类结果：
   - **超出模板的 key**：提醒删除
   - **模板有但用户没有的 key**：推荐使用模板默认值
   - **双方都有的 key**：对比 value，确认使用哪个
3. 确认每项差异后生成最终 config

### 没有 config.json

1. 展示对应任务类型的模板
2. 逐项询问是否需要调整（只允许调整模板内的 key）
3. 用户确认后写入文件

## 注意事项

1. **config.json 的 key 只能是模板 key 的子集**
2. 不允许添加模板之外的 key
3. 修改前务必向用户确认
4. 运行前完整展示最终配置
