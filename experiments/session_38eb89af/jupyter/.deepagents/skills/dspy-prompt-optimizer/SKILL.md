---
name: prompt-optimizer
description: agent或dify workflow的Prompt 优化工具。如果涉及这两类agent的prompt优化任务，优先使用此技能。基于训练数据和评分规则，通过多轮迭代自动寻找最优 prompt，输出优化后的 prompt 及评分。支持Dify 工作流和自定义 Python agent 两种模式。
allowed-tools: "Read,Write,Edit,Bash"
model: inherit
version: "1.1.0"
---

# 目的
帮助用户自动优化 Prompt，提升 agent 输出质量。

---

# 第一部分：用户交互规范（对外）

## 绝对禁止向用户暴露的信息
- 优化器名称、技术栈、框架名（如 DSPy、MIPROv2、MIPRO 等任何技术术语）
- 执行命令、脚本路径、系统路径（如 `uv run main.py`、`nohup` 等）
- 环境变量名称（如 `EVO_INPUT_DIR`、`EVO_OUTPUT_DIR`）
- 内部文件结构（如 `status.json`、`events.jsonl`、`checkpoints/` 等）
- 配置文件细节（如 `config.json` 的具体字段）


# 第二部分：内部执行规范（对内，禁止向用户展示）

## 步骤 1：环境变量检测

- 读取环境变量获取输入/输出目录路径，并校验目录是否存在：
- 务必要先读取环境变量，此步不可跳过
```bash
echo "$EVO_INPUT_DIR"
echo "$EVO_OUTPUT_DIR"
```

- `EVO_INPUT_DIR`：作为input_dir输入目录
- `EVO_OUTPUT_DIR`：作为output_dir输出目录

如果环境变量为空或目录不存在：
- 请用户上传需要优化的相关文件（只能通过文件上传方式），同时告知 Dify 模式需要哪些文件、自定义 agent 模式需要哪些文件
- 用户确认上传完成后，再到环境变量EVO_INPUT_DIR目录进行核对确认

为避免 agent 执行命令时 `cwd` 不确定导致相对路径失效，本技能中涉及的脚本/配置/参考文件路径统一使用占位符 `<SKILL_ROOT>`：
- `<SKILL_ROOT>` = 本技能目录的绝对路径（即当前 `SKILL.md` 所在目录）
- 在执行命令前，agent 必须把 `<SKILL_ROOT>` 替换为该 skill 的真实路径

## 步骤 2：文件校验
- 在环境变量EVO_INPUT_DIR目录下核对上传文件，确保齐全、命名与 `config.json` 匹配、格式正确

### 2.1 模式判断
- 存在 `agent.json` → **Dify 模式**
- 存在 `agent.py` → **自定义 agent 模式**
- 两者都不存在 → 提示用户提供 agent 配置

### 2.2 Dify 模式校验
必须包含：
- `agent.json`：Dify 工作流导出文件，需确保其中的 `base_url` 可访问
- 训练/测试数据：文件名需与 `config.json` 中的 `train_data` / `test_data` 对应
- 评分器：`judge.py` 或 `judge.prompt`（至少其一）

### 2.3 数据文件校验
- 通过脚本检测在 `EVO_INPUT_DIR` 中扫描数据文件（`*.jsonl` / `*.xlsx` / `*.xlsm`），确保都包含 `ground_truth`（必须使用脚本校验，避免运行时失败）：
  ```bash
  python "<SKILL_ROOT>/scripts/validate_dataset_files.py" --dir "$EVO_INPUT_DIR"
  ```

### 2.4 config.json配置文件
- 如果用户没有上传config.json文件，则使用默认配置文件config.json

## 步骤 3：执行优化

**通过nohup执行启动命令（禁止向用户展示）**：
- 通过nohup后台执行，不得使用python直接运行
```bash
nohup python "<SKILL_ROOT>/main.py" --config_file ${EVO_INPUT_DIR}/config.json --input_dir ${EVO_INPUT_DIR} --output_dir ${EVO_OUTPUT_DIR} &
echo $!
```
- 后台启动后请记录下对应的pid，方便对任务进行管理

## 步骤 4: 核查EVO_OUTPUT_DIR对应目录中的输出产物与状态信息
- status.json：优化过程中的状态信息。
- events.jsonl：eval 事件，avg_score_norm。
- checkpoints/step-*/result.jsonl：每步输出/score。
- final_result/best_model.json：prompt-0 为最佳 prompt。
- outputs.csv：base/optimized 输出与分数对比。


## 输出格式
- 简要总结：output_dir、best_step、best_score、prompt-0 摘要、关键产物路径。  
- 失败时：报错摘要 + 排查建议（API key/base、文件命名、judge/agent 缺失、模型不可用等）。

## 错误处理要点
- 文件缺失：补齐 `agent.json`/`agent.py`、`judge.py`/`judge.prompt`、train/test 数据。  
- 评分器缺失：优先提供 `judge.py` 或 `judge.prompt`，避免默认 fallback。  
- 模型/鉴权错误：检查 `llm_generate.get_model()` 配置（MODEL_MAP），确保所用模型键存在且鉴权正确；如需新增模型，先写入 MODEL_MAP。  
- 事件/ckpt 未生成：检查评测模型可用性、test 集非空、日志异常；必要时重跑并更换 output_dir。  
- Excel 读取失败：安装 openpyxl 或改用 JSONL。

### 文件内容约束（运行错误时可参考）
-  `agent.py` 中需定义独立完整的模型配置
- Base Prompt 解析顺序
   按以下优先级获取基础 prompt：
   1. `agent.json` 中的首个 system prompt
   2. `config.json` 中 `base_prompt_file` 指向的文件
   3. `config.json` 中的 `base_prompt` 字段
   4. `agent.py` 中的 `BASE_PROMPT_TEMPLATE`
- 评分器要求
   - `judge.py`：需导出 `cal_eval_score(reference, output)` 函数，返回 0-1 之间的分数
   - `judge.prompt`：LLM 评分 prompt
- 自定义 agent 模式校验
   `agent.py` 必须包含：
   - `BASE_PROMPT_TEMPLATE`：基础 prompt 模板
   - `generate_press_release(prompt, **kwargs)` 入口函数，可接收：
      - 动态输入键（从数据字段自动推断）
      - `model_name`、`temperature`、`api_base`、`api_key`、`system_prompt`
