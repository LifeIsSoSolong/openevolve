"""
Math-solving example using LinusPromptOptimizer, training on datasets/train.json

- Agent contract: run(task: str, prompt: str) -> str
- Agent here calls OpenAI to solve math prompts under the provided prompt
- Evaluator compares the agent's output to ground-truth answers and emits English feedback

Reference: https://github.com/Arize-ai/prompt-learning/
"""
from __future__ import annotations
import argparse

import srsly
import os
from typing import Dict, List, Tuple
import logging

from linus_optimizer import LinusPromptOptimizer
from openai import OpenAI
import concurrent.futures
from math_verify import parse, verify

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class MathAgent:
    """Agent that delegates to OpenAI with the given prompt and task."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, base_url: str = None) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        base_url = base_url or os.getenv("OPENAI_API_BASE", "")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def run(self, task: str, prompt: str) -> str:
        # Fill template variable {task} if present
        formatted = prompt.replace("{task}", task)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted}],
        )
        return (resp.choices[0].message.content or "").strip()


def math_evaluator(task: str, response: str, expected: str) -> str:
    ret_score = 0.0

    def _compute():
        """Helper function to compute score with parsing and verification."""
        parsed_output = parse(response, parsing_timeout=None)
        parsed_ground_truth = parse(expected, parsing_timeout=None)
        return verify(parsed_output, parsed_ground_truth, timeout_seconds=None)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_compute)
            ret_score = future.result(timeout=10)  # 10 seconds timeout
    except Exception:
        logger.debug("Exception during evaluation", exc_info=True)
        ret_score = 0.0

    if ret_score:
        return "correct"
    return f"incorrect: expected '{expected}', got '{response}'."


def main(args, run_logger: RunLogger) -> None:
    # Configuration
    config = srsly.read_json(args.config_file)
    train_file = os.path.join(args.input_dir, "train.jsonl")
    test_file = os.path.join(args.input_dir, "test.jsonl")

    os.environ["OPENAI_API_KEY"] = config["environments"]["OPENAI_API_KEY"]
    os.environ["OPENAI_API_BASE"] = config["environments"]["OPENAI_API_BASE"]

    baseline_prompt = config["agent"]["initial_template"]
    agent_model = config["agent"]["agent_model"]
    optimizer_model = config["agent"]["optimizer_model"]

    train_size = config["training"]["train_size"]
    test_size = config["training"]["test_size"]

    loops = config["training"]["loops"]
    max_workers = config["training"]["max_workers"]

    run_logger.update_status(
        state="running",
        current_step=0,
        total_steps=loops,
        error=None,
    )

    # Loading Dataset
    train_samples = list(srsly.read_jsonl(train_file))[:train_size]
    train_data: List[Dict[str, str]] = [
        {"task": ex["messages"][0]["content"], "ground_truth": ex["ground_truth"]} for ex in train_samples
    ]

    test_samples = list(srsly.read_jsonl(test_file))[:test_size]
    test_data: List[Dict[str, str]] = [
        {"task": ex["messages"][0]["content"], "ground_truth": ex["ground_truth"]} for ex in test_samples
    ]

    agent = MathAgent(model=agent_model)
    optimizer = LinusPromptOptimizer(model=optimizer_model)

    # Test before optimization
    logger.warning("Evaluating baseline prompt...")
    baseline_acc, _ = LinusPromptOptimizer.fill_outputs_with_agent(
        agent=agent,
        baseline_prompt=baseline_prompt,
        dataset=test_data,
        task_field="task",
        output_column="output",
        max_workers=max_workers,
        metric_func=math_evaluator
    )
    logger.warning(f"Baseline Prompt Accuracy: {baseline_acc:.2%}")
    run_logger.log_event(
        {
            "step": 0,
            "type": "eval",
            "accuracy": baseline_acc,
            "prompt": baseline_prompt
        }
    )

    logger.warning("Optimizing prompt...")
    improved, final_score = optimizer.optimize_with_agent(
        baseline_prompt=baseline_prompt,
        agent=agent,
        dataset=train_data,
        test_dataset=test_data,
        task_field="task",
        output_column="output",
        feedback_fn=math_evaluator,
        loops=loops,
        context_size_tokens=128000,
        max_workers=max_workers,
        run_logger=run_logger
    )

    print("\n=== Baseline Prompt ===\n")
    print(baseline_prompt)
    print("\n=== Improved Prompt ===\n")
    print(improved)

    run_logger.update_status(
        state="completed",
        current_step=loops,
        total_steps=loops,
        error=None,
        extra={
            "final_prompt": improved,
            "final_score": final_score
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--reward_def", type=str, default=None)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    from run_logger import RunLogger
    run_logger = RunLogger(args.output_dir)

    try:
        main(args, run_logger)
    except Exception as e:
        run_logger.update_status(
            state="failed",
            current_step=-1,
            total_steps=-1,
            error=str(e),
        )
        raise
