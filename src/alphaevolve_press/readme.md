# AlphaEvolve Press01

## 1. 准备
- 将 `inputs/source/` 下内容上传到 `/***/inputs/`（包含 `config_evolve.yaml`、`initial_program.py`、`evaluator.py` 等）。
- 将 `config.json` 上传到同级 `/***/inputs/`。

## 2. 运行
```bash
python main.py \
    --config_file "/abs/path/to/inputs/config.json" \
    --input_dir "/abs/path/to/inputs" \
    --output_dir "/abs/path/to/outputs"
```

## 3. 输出目录结构
```
outputs/
├─ events.jsonl          # 逐步评测日志（追加）
├─ status.json           # 最新状态（覆盖写）
├─ checkpoints/
│  ├─ step-1/
│  ├─ step-2/
│  └─ ...
├─ logs/
│  └─ openevolve_*.log
└─ final_result/
   ├─ best_program.py
   └─ best_program_info.json
```
- `events.jsonl` 每步包含训练集指标，若启用测试集也会追加 `test_*` 指标。
- `checkpoints/step-*` 保留每轮快照；`final_result/` 为最终最优。

## 4. 配置
- `inputs/config.json`：前端/后端传入的覆盖配置（例如 LLM、timeout 等）。
- `inputs/source/config_evolve.yaml`：默认配置模板，未在 `config.json` 覆盖的字段沿用这里的值。

## 5. 依赖
- Python 环境需包含运行 OpenEvolve 所需依赖，以及可访问的 LLM 接口（`api_base`/`api_key`）。 
