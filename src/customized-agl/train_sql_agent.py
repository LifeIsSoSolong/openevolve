# Copyright (c) Microsoft. All rights reserved.

"""Train an SQL agent on the Spider dataset using Agent-lightning.

This module provides a training script for SQL agents using different model configurations.
The script supports three different training configurations:

1. 'fast' - A lightweight configuration optimized for CI testing with reduced epochs
2. 'qwen' - Standard configuration using Qwen-2.5-Coder-1.5B-Instruct model
3. 'llama' - Configuration using LLaMA-3.2-1B-Instruct model with JSON formatting

Usage:
    python train_sql_agent.py fast    # Fast training for CI/testing
    python train_sql_agent.py qwen    # Standard Qwen model training
    python train_sql_agent.py llama   # LLaMA model training

The script uses reinforcement learning with VERL framework
to train agents on the Spider dataset for text-to-SQL generation tasks.
"""

from __future__ import annotations
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
from sql_agent import LitSQLAgent

import agentlightning as agl

RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "data/train_spider.parquet",
        "val_files": "data/test_dev_500.parquet",
        "train_batch_size": 32,
        "max_prompt_length": 4096,
        "max_response_length": 2048,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 4,
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.8,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 32,
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
            "path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 4,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "tensorboard"],
        "project_name": "AgentLightning",
        "experiment_name": "spider2",
        "nnodes": 1,
        "test_freq": 32,
        "total_epochs": 2,
    },
}

log_path = Path("events.jsonl")


def wrapper_fit(self):
    logger = Tracking(
        project_name=self.config.trainer.project_name,
        experiment_name=self.config.trainer.experiment_name,
        default_backend=self.config.trainer.logger,
        config=OmegaConf.to_container(self.config, resolve=True),
    )

    self.global_steps = 0

    # load checkpoint before doing anything
    self._load_checkpoint()

    assert self.async_rollout_mode, "If agent mode is enabled, async server must be enabled"
    if self.adapter is not None and not isinstance(self.adapter, TraceToTripletBase):
        raise ValueError(
            "Adapter must be a TraceToTripletBase for currently VERL implementation.")
    verl_version = verl.__version__
    if verl_version == "0.5.0":
        # Note (Zhiyuan): To avoid further patch into vllm async server, using the same sentence to get the naming here.
        # However, it is possible that verl updates the naming and causes incompatibility.
        # Reference: https://github.com/volcengine/verl/blob/5b5e09d9cc20625e436d01f69d9cc739ff681c54/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L217
        model = "/".join(self.config.actor_rollout_ref.model.path.split("/")
                         [-2:])
    else:
        # For other versions (e.g., 0.6.0), we use the full path to the model.
        model = self.config.actor_rollout_ref.model.path
    self.agent_mode_daemon = AgentModeDaemon(
        self.config.agentlightning.port,
        self.config.actor_rollout_ref.rollout.n,
        train_information={
            "model": model,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        },
        tokenizer=self.tokenizer,
        mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
        pad_token_id=self.tokenizer.pad_token_id,
        mode="v1" if self.store is not None else "v0",
        store=self.store,
        llm_proxy=self.llm_proxy,
        adapter=self.adapter,
    )
    self.agent_mode_daemon.start()

    # perform validation before training
    # currently, we only support validation using the reward_function.
    if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        val_metrics = self._validate()
        assert val_metrics, f"{val_metrics=}"
        pprint(f"Initial validation metrics: {val_metrics}")
        logger.log(data=val_metrics, step=self.global_steps)
        if self.config.trainer.get("val_only", False):
            return

    # add tqdm
    progress_bar = tqdm(total=self.total_training_steps,
                        initial=self.global_steps, desc="Training Progress")

    # we start from step 1
    self.global_steps += 1
    last_val_metrics = None

    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            # train step
            metrics = self._train_step(batch_dict)

            # validate
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with _timer("validate", timing_raw):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                with _timer("save_checkpoint", timing_raw):
                    self._save_checkpoint()

            # step metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
            )

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()

                # This exit logic is to ensure a robust CI.
                pprint(f"Flush the logger...")
                del logger  # Make sure the loggers are flushed and closed properly
                pprint(f"Training finished at step {self.global_steps}.")
                return

            progress_bar.update(1)
            self.global_steps += 1


AgentLightningTrainer.fit = types.MethodType(
    wrapper_fit, AgentLightningTrainer)


def config_train_fast() -> Dict[str, Any]:
    """A fast training run for CI testing purposes."""

    # `EXPERIMENT_NAME="spider_$(date +%Y%m%d%H%M%S)"`
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    EXPERIMENT_NAME = f"spider_{timestamp}"

    # `PROJECT_NAME=AgentLightningCI`
    PROJECT_NAME = "AgentLightningCI"

    # Simulate writing to $GITHUB_OUTPUT if it’s set
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
    config["actor_rollout_ref"]["model"]["path"] = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    config["data"]["val_files"] = "data/test_dev.parquet"
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["experiment_name"] = EXPERIMENT_NAME
    config["trainer"]["project_name"] = PROJECT_NAME
    config["trainer"]["test_freq"] = 1
    return config


def config_train_qwen() -> Dict[str, Any]:
    """A configuration for training with Qwen-2.5B."""

    config = deepcopy(RL_TRAINING_CONFIG)
    return config


def config_train_llama() -> Dict[str, Any]:
    """A configuration for training with LLaMA-3.2-1B-Instruct.

    You will need a `HF_TOKEN` set to run with this config.
    """

    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["rollout"]["multi_turn"]["format"] = "llama3_json"
    config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["tool_call_parser"] = "llama3_json"
    config["actor_rollout_ref"]["model"]["path"] = "/hpc_data/zhangkaiyan/hf-models/meta-llama/Llama-3.2-3B-Instruct"
    return config


def train(config: Dict[str, Any], active_agent: Optional[str]) -> None:
    """Train the SQL agent with the given configuration."""

    agent = LitSQLAgent()
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=10, algorithm=algorithm, adapter={
                          "agent_match": active_agent})
    print("Adapter agent match acknowledged:",
          trainer.adapter.agent_match)  # type: ignore

    train_data = pd.read_parquet(config["data"]["train_files"]).to_dict(
        orient="records")  # type: ignore
    val_data = pd.read_parquet(config["data"]["val_files"]).to_dict(
        orient="records")  # type: ignore
    trainer.fit(agent, train_dataset=train_data,
                val_dataset=val_data)  # type: ignore


def main() -> None:
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train an SQL agent on the Spider dataset using different model configurations"
    )

    parser.add_argument(
        "config",
        choices=["fast", "qwen", "llama"],
        help="Training configuration: 'fast' (CI testing), 'qwen' (Qwen-2.5-Coder-1.5B), 'llama' (LLaMA-3.2-3B)",
    )

    parser.add_argument(
        "--active-agent", type=str, help="Override the active agent name (default: auto-generated based on config)"
    )

    args = parser.parse_args()

    # Get the appropriate configuration
    config_functions = {"fast": config_train_fast,
                        "qwen": config_train_qwen, "llama": config_train_llama}

    config = config_functions[args.config]()

    # Set active agent - use provided value or default based on config choice
    active_agent = args.active_agent

    print(f"Starting training with '{args.config}' configuration...")
    print(f"Active agent: {active_agent}")

    train(config, active_agent)


if __name__ == "__main__":
    main()
