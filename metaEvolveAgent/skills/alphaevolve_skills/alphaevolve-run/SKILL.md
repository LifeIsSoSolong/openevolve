---
name: alphaevolve-run
description: 使用 OpenEvolve（alphaevolve）main.py 执行 MLE 代码进化或 prompt 进化，并进行输入检查与产出检查。适用于用户要求启动 MLE 推理进化、prompt 推理进化、大豆产量预测代码进化、新闻稿 prompt 进化，或任何使用 --config_file/--input_dir/--output_dir 的 alphaevolve/openevolve 运行任务。
---

# AlphaEvolve 运行

## 快速开始

以下命令均为占位示例，请根据实际运行环境替换路径。

1) 输入目录优先策略：
   - 用户显式给出 input_dir 时直接使用。
   - 未给出时，先运行 `scripts/check_inputs.py`，它会在 `SKILL_ROOT` 下查找 `inputs/`。
     - 若找到多个或找不到，提示用户指定 `--input-dir`。

2) 运行进化（直接调用 main.py）：

```bash
python "<REPO_ROOT>/main.py" --config_file "<INPUT_DIR>/config.json" --input_dir "<INPUT_DIR>" --output_dir "<OUTPUT_DIR>"
```

3) 运行结束后检查产出：

```bash
python "<SKILL_ROOT>/scripts/check_outputs.py" --output-dir "<OUTPUT_DIR>"
```

## 输入要求（仅检查存在性）

- MLE 代码进化：
  - agent.py
  - judge.py
  - config.json
  - task.goal
  - train.csv
  - test.csv

- prompt 进化：
  - agent.py
  - judge.py
  - config.json
  - task.goal
  - train.jsonl
  - test.jsonl
  - generate_press_agent.py
  - evaluate_press_agent.py

若用户只说“启动推理进化任务”但未指定类型，先询问要进化哪一类任务。

## 产出检查

脚本默认检查：
- events.jsonl
- status.json
- config_evolve_merged.yaml
- logs/（至少有一个 .log）
- checkpoints/（包含 step-*）

部分文件是迭代过程中逐步生成的；缺失时默认为警告，除非启用严格模式。

## 脚本参数

scripts/check_inputs.py：
- --input-dir <path>（不传则在 SKILL_ROOT 下搜索 inputs/）
- --task-type auto|mle|prompt|generic
- --require <file>（可重复）

scripts/check_outputs.py：
- --output-dir <path>
- --strict（缺失必需产出时直接报错）

## 添加新任务类型

编辑 scripts/check_inputs.py，扩展 TASK_REQUIREMENTS 列表即可。
