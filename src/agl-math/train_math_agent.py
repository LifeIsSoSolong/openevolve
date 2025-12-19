# Copyright (c) Microsoft. All rights reserved.

"""训练数学问题求解智能体

使用 Agent-lightning 框架训练一个单轮智能体，用于求解数学问题。
智能体接收数学问题，使用 LLM 生成答案，然后通过 accuracy_reward 评估正确性。

Usage:
    python train_math_agent.py fast    # 快速训练用于测试
    python train_math_agent.py qwen    # 使用 Qwen 模型训练
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from math_agent import LitMathAgent

import agentlightning as agl

import types
from pathlib import Path
import json
from pprint import pprint
from tqdm import tqdm
from agentlightning.verl.daemon import AgentModeDaemon
from verl.utils.tracking import Tracking
import verl
from agentlightning.adapter import TraceToTripletBase
from agentlightning.verl.trainer import _timer
from agentlightning.verl.trainer import AgentLightningTrainer
from omegaconf import OmegaConf

import argparse
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "agentlightning": {
        "port": 9999,  # AgentLightningServer 控制平面端口，可以自定义以避免冲突
    },
    "data": {
        "train_files": "data/test2.jsonl",
        "val_files": "data/test2.jsonl",  # 训练和验证使用同一个数据
        "train_batch_size": 16,
        "max_prompt_length": 2048,
        "max_response_length": 2048,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 2,
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.6,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": False,
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 16,
            "ppo_micro_batch_size_per_gpu": 4,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 8,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "/hpc_data/ktian/models/Qwen3-4B-Instruct-2507",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 4,  # 使用 4 个 GPU
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "file"],
        "project_name": "AgentLightning",
        "experiment_name": "math",
        "nnodes": 1,
        "test_freq": 2,
        "save_freq": 100000,
        "total_epochs": 5,
    },
}


def config_train_fast() -> Dict[str, Any]:
    """快速训练配置，用于 CI 测试。"""

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_NAME = f"math_{timestamp}"

    PROJECT_NAME = "AgentLightningCI"

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"project_name={PROJECT_NAME}\n")
            f.write(f"run_name={EXPERIMENT_NAME}\n")

    print("Set environment variables:")
    print(f"PROJECT_NAME={PROJECT_NAME}")
    print(f"EXPERIMENT_NAME={EXPERIMENT_NAME}")

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.6
    config["actor_rollout_ref"]["model"]["path"] = "/hpc_data/ktian/models/Qwen3-4B-Instruct-2507"
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["experiment_name"] = EXPERIMENT_NAME
    config["trainer"]["project_name"] = PROJECT_NAME
    config["trainer"]["test_freq"] = 1
    config["trainer"]["n_gpus_per_node"] = 1  # 快速模式使用 1 个 GPU
    return config


def config_train_qwen() -> Dict[str, Any]:
    """使用 Qwen-2.5-Coder-1.5B-Instruct 模型的训练配置。"""

    config = deepcopy(RL_TRAINING_CONFIG)
    return config


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载数据。

    Args:
        file_path: JSONL 文件路径

    Returns:
        数据列表，每个元素包含 'problem' 和 'answer' 字段
    """
    data: List[Dict[str, Any]] = []
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                # 确保包含 problem 和 answer 字段
                if "problem" not in item or "answer" not in item:
                    raise ValueError(f"数据项缺少 'problem' 或 'answer' 字段: {item}")
                data.append(
                    {"problem": item["problem"], "answer": item["answer"]})

    return data


def train(config: Dict[str, Any], active_agent: Optional[str]) -> None:
    """训练数学问题求解智能体。"""

    agent = LitMathAgent()
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=10, algorithm=algorithm, adapter={
                          "agent_match": active_agent})
    print("Adapter agent match acknowledged:",
          trainer.adapter.agent_match)  # type: ignore

    # 加载数据
    train_data = load_jsonl(config["data"]["train_files"])
    val_data = load_jsonl(config["data"]["val_files"])

    print(f"加载训练数据: {len(train_data)} 条")
    print(f"加载验证数据: {len(val_data)} 条")

    trainer.fit(agent, train_dataset=train_data,
                val_dataset=val_data)  # type: ignore


def main() -> None:
    """主函数，解析参数并运行训练。"""
    parser = argparse.ArgumentParser(
        description="训练数学问题求解智能体，支持不同的模型配置"
    )

    parser.add_argument(
        "config",
        choices=["fast", "qwen"],
        help="训练配置: 'fast' (CI 测试), 'qwen' (Qwen-2.5-Coder-1.5B-Instruct)",
    )

    parser.add_argument(
        "--active-agent",
        type=str,
        help="覆盖智能体名称（默认：根据配置自动生成）",
    )

    parser.add_argument(
        "--port",
        type=int,
        help="AgentLightningServer 控制平面端口（默认：9999，用于避免多任务端口冲突）",
    )

    args = parser.parse_args()

    # 获取相应的配置
    config_functions = {
        "fast": config_train_fast,
        "qwen": config_train_qwen,
    }
    config = config_functions[args.config]()

    # 如果指定了端口，覆盖默认端口
    if args.port is not None:
        config["agentlightning"]["port"] = args.port
        print(f"使用自定义端口: {args.port}")

    # 设置 active agent
    active_agent = args.active_agent

    print(f"使用 '{args.config}' 配置开始训练...")
    print(f"Active agent: {active_agent}")
    print(f"GPU 数量: {config['trainer']['n_gpus_per_node']}")
    print(f"AgentLightningServer 端口: {config['agentlightning']['port']}")

    train(config, active_agent)


if __name__ == "__main__":
    main()
