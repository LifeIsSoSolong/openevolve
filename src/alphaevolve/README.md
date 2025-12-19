# AlphaEvolve

- 1.upload source/ to /***/inputs/, such as /hpc_data/2T/kaikai/agentic-rl/deploy/bean01/inputs/

- 2.upload config.json to /***/inputs/, such as /hpc_data/2T/kaikai/agentic-rl/deploy/bean01/inputs/config.json

- 3.run the following command

```shell
python main.py \
    --config_file "/hpc_data/zhangkaiyan/agentic-rl/src/alphaevolve/config.json" \
    --input_dir "/hpc_data/zhangkaiyan/agentic-rl/src/alphaevolve/inputs" \
    --output_dir "/hpc_data/zhangkaiyan/agentic-rl/src/alphaevolve/outputs"
```

## outputs/
``` markdown
outputs/
├── events.jsonl             # 逐轮评估日志（append-only）
├── status.json             # 当前状态（step/running/failed/completed 等，覆盖写）
├── checkpoints/
│   ├── step-1/
│   │   ├── best_program.py
│   │   ├── best_program_info.json # 最佳程序的指标信息
│   │   └── ...              # 数据库快照等
│   ├── step-2/
│   │   └── ...
│   └── ...
├── logs/
│   └── openevolve_*.log
├── final_result/
│   ├── best_program.py
│   └── best_program_info.json
└── evolution_trace.jsonl    # 可选，启用 evolution_trace 时生成,可以包含每一轮的父子关系、child code、prompt、llm输出等详细内容
```