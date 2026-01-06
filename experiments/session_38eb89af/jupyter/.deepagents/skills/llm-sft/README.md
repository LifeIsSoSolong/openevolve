

建议的引导指令：使用LLaMA-Factory来对基础的大模型进行指令微调，提升最终效果。

算法执行流程：
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


环境依赖：需要安装LLaMA-Factory（https://github.com/hiyouga/LLaMA-Factory），使得llamafactory-cli在系统PATH中，推荐使用https://github.com/hiyouga/LLaMA-Factory/releases/tag/v0.9.3或者examples/llm-sft/llm-sft-outputs/LLaMA-Factory.tar.gz。

进入项目根目录执行
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics,deepspeed]"
```

为了保留日志events.jsonl,status.jsonl, 需要对源代码进行覆盖

在项目根目录下执行
```bash
cp ./examples/llm-sft/llm-sft-outputs/llama_code_new/finetuning_args.py LLaMA-Factory/src/llamafactory/hparams/finetuning_args.py 
cp ./examples/llm-sft/llm-sft-outputs/llama_code_new/parser.py LLaMA-Factory/src/llamafactory/hparams/parser.py 
cp ./examples/llm-sft/llm-sft-outputs/llama_code_new/callbacks.py LLaMA-Factory/src/llamafactory/train/callbacks.py 
cp ./examples/llm-sft/llm-sft-outputs/llama_code_new/tuner.py LLaMA-Factory/src/llamafactory/train/tuner.py 
```
