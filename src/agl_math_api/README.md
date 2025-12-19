## Agent-Lightning-MATH

使用定制化的 tracking.py 替换 verl 库的日志文件，位于 /root/miniconda3/lib/python3.10/site-packages/verl/utils/tracking.py  



```
export VERL_FILE_LOGGER_PATH=/hpc_data/ktian/agent-lighting/examples/math_api/outputs

python main.py \
    --config_file /hpc_data/ktian/agent-lighting/examples/math_api/inputs/config.json \
    --train_file /hpc_data/ktian/agent-lighting/examples/math_api/inputs/train.jsonl \
    --eval_file /hpc_data/ktian/agent-lighting/examples/math_api/inputs/test.jsonl \
    --reward_def /hpc_data/ktian/agent-lighting/examples/math_api/inputs/reward_definition.py \
    --output_dir /hpc_data/ktian/agent-lighting/examples/math_api/outputs
```