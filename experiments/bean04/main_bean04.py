#!/usr/bin/env python
"""
Wrapper entrypoint for running OpenEvolve with JSON config overrides.

Expected invocation (from FastAPI/back-end):
    python main.py --config_file /abs/path/inputs/config.json \
                   --input_dir  /abs/path/inputs/ \
                   --output_dir /abs/path/outputs/

This script will:
1) Load the base YAML config from <input_dir>/config_evolve.yaml.
2) Load overrides from the provided JSON config_file.
3) Merge them (JSON wins), write the merged YAML to <output_dir>/config_evolve_merged.yaml.
4) Apply environment variables specified in JSON (under "environments").
5) Run OpenEvolve on <input_dir>/{agent.py, judge.py} and store results in output_dir.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
import types
import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from openevolve import OpenEvolve
from openevolve.config import Config


# 把openevolve需要用到的配置文件（默认的config_evolve.yaml）内容写死在这里; 前端传入的config.json会覆盖部分配置项
BASE_CONFIG = {
    "max_iterations": 10,
    "checkpoint_interval": 1,
    "log_level": "DEBUG",
    "random_seed": 42,
    "llm": {
        "primary_model": "gpt-5.2",
        "primary_model_weight": 0.8,
        "secondary_model": "gpt-5.2",
        "secondary_model_weight": 0.2,
        # "api_base": "https://newapi.frontis.top/v1",
        # "api_key": "sk-JegueK6Qy16ZJ6zjxfoKBMwQ32kt56LRZgyTPlapD7cGw0QN",
        "api_base": "http://35.164.11.19:3887/v1",
        "api_key": "sk-DOnUAnR18NW7Yp3OCp9sfWTgyFPOTURpGovP9EFIrTNEzozV",
        "temperature": 0.5,
        "max_tokens": 60000,
        "timeout": 300,
        "reasoning_effort": "medium",
    },
    "prompt": {
        "system_message": (
            "You are optimizing machine learning code for predicting Brazilian soybean yield. Goal: minimize MAPE/RMSE on the held-out test set.\n\n"
            "Hard constraints:\n"
            "- Only modify code inside the EVOLVE-BLOCK; do not touch any other code, function signatures, paths, or helpers.\n"
            "- Keep the script runnable standalone: read train and test data, write submission.csv, the format of submission.csv must be same with test_answer.csv.\n"
            " do not drop all features, you can combine them to generate new features for better performance.\n"
            "- You can use any other standard Python libraries for better model.\n\n"
            "Freedom:\n"
            "- Inside EVOLVE-BLOCK you may change model type (LightGBM/CatBoost/XGBoost/linear/NN/heuristics/FFN), features, and hyperparameters to improve accuracy.\n\n"
            "Output format:\n"
            "- Respond ONLY with valid SEARCH/REPLACE diffs for the EVOLVE-BLOCK. If you cannot propose a valid diff, return an empty diff.\n"
        )
    },
    "database": {
        "population_size": 16,
        "archive_size": 8,
        "num_islands": 2,
        "elite_selection_ratio": 0.25,
        "exploitation_ratio": 0.6,
    },
    "evaluator": {
        "timeout": 300,
        "parallel_evaluations": 1,
        "cascade_evaluation": False,
    },
    "diff_based_evolution": True,
    "max_code_length": 60000,
    "evolution_trace": {
        "enabled": True,
        "format": "jsonl",
        "include_code": True,
        "include_prompts": True,
        "output_path": None,
        "buffer_size": 1,
        "compress": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenEvolve with JSON overrides.")
    parser.add_argument("--config_file", required=True, help="Path to JSON config file.")
    parser.add_argument("--input_dir", required=True, help="Path to input directory (contains config_evolve.yaml, agent.py, judge.py).")
    parser.add_argument("--output_dir", required=True, help="Path to output directory.")
    return parser.parse_args()


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge overrides into base (overrides win)."""
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def load_config_sources(input_dir: Path, json_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load base config (embedded) and JSON overrides, then merge."""
    base_cfg = copy.deepcopy(BASE_CONFIG)

    # 若存在 task.goal，则用其中内容覆盖 system_message
    task_goal_path = input_dir / "task.goal"
    if task_goal_path.exists():
        try:
            goal_text = task_goal_path.read_text(encoding="utf-8").strip()
            if goal_text:
                base_cfg.setdefault("prompt", {})["system_message"] = goal_text
                preview = (goal_text[:200] + "...") if len(goal_text) > 200 else goal_text
                print(f"[main_bean] Loaded system_message from task.goal: {preview}")
        except Exception as exc:
            print(f"Warning: failed to read task.goal: {exc}")

    with open(json_path, "r", encoding="utf-8") as f:
        json_cfg = json.load(f) or {}

    # Remove non-config keys that are meta info
    meta_keys = {"algorithm", "task_type"}
    meta_keys.update("prompt")
    cfg_overrides = {k: v for k, v in json_cfg.items() if k not in meta_keys and k != "environments"}

    merged = deep_merge(base_cfg, cfg_overrides)
    return merged, json_cfg.get("environments", {})


def apply_environments(envs: Dict[str, Any]) -> None:
    """Set environment variables from JSON (strings only)."""
    for k, v in envs.items():
        if v is not None:
            os.environ[str(k)] = str(v)


async def run_openevolve(config_dict: Dict[str, Any], input_dir: Path, output_dir: Path) -> int:
    """Instantiate and run OpenEvolve with merged config."""
    initial_program = input_dir / "agent.py"
    evaluator_file = input_dir / "judge.py"

    if not initial_program.exists():
        raise FileNotFoundError(f"Initial program not found: {initial_program}")
    if not evaluator_file.exists():
        raise FileNotFoundError(f"Evaluator file not found: {evaluator_file}")

    # Build Config object
    config_obj = Config.from_dict(config_dict)

    # Respect log level override if provided
    if getattr(config_obj, "log_level", None):
        import logging

        logging.getLogger().setLevel(getattr(logging, str(config_obj.log_level).upper(), logging.INFO))

    # Prepare custom event/status outputs (non-intrusive monkey patch)
    events_path = output_dir / "events.jsonl"
    status_path = output_dir / "status.json"
    total_steps = config_obj.max_iterations

    def write_status(state: str, step: int, error: str | None = None) -> None:
        payload = {
            "state": state,
            "current_step": step,
            "total_steps": total_steps,
            "last_update": int(time.time()),
            "error": error,
        }
        tmp = status_path.with_suffix(status_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(status_path)

    openevolve = OpenEvolve(
        initial_program_path=str(initial_program),
        evaluation_file=str(evaluator_file),
        config=config_obj,
        output_dir=str(output_dir),
    )

    # Monkey-patch database.add to append events.jsonl and update status.json
    orig_add = openevolve.database.add
    expected_step = 0
    pending_events: Dict[int, Dict[str, Any]] = {}

    def flush_event(step: int, event: Dict[str, Any]) -> None:
        """Write one event and status."""
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        write_status("running", step, error=None)

    def add_wrapper(self, program, iteration=None, target_island=None):
        res = orig_add(program, iteration=iteration, target_island=target_island)

        step = (
            iteration
            if iteration is not None
            else getattr(program, "iteration_found", None)
            if getattr(program, "iteration_found", None) is not None
            else self.last_iteration
        )

        best = self.get(self.best_program_id) if self.best_program_id else self.get_best_program()
        best_metrics = best.metrics if best else {}
        metrics = program.metrics or {}

        event: Dict[str, Any] = {
            "step": step,
            "type": "initial_eval" if step == 0 else "evolve_eval",
            "timestamp": int(time.time()),
            "error": metrics.get("error"),
        }
        # 当前程序的所有指标直接展开
        for k, v in metrics.items():
            event[k] = v
        # 最优程序的指标加 best_ 前缀
        for k, v in best_metrics.items():
            event[f"best_{k}"] = v
        nonlocal expected_step, pending_events
        try:
            # 如果发现缺口（某些轮次没有事件，如无效 diff），补写占位事件
            if step > expected_step:
                for missing in range(expected_step, step):
                    placeholder = {
                        "step": missing,
                        "type": "initial_eval" if missing == 0 else "evolve_eval",
                        "timestamp": int(time.time()),
                        "error": "missing event (likely no valid diff / skipped iteration)",
                        "combined_score": 0.0,
                    }
                    for k, v in best_metrics.items():
                        placeholder[f"best_{k}"] = v
                    flush_event(missing, placeholder)
                expected_step = step

            # Buffer to preserve order: only write when step == expected_step
            pending_events[step] = event
            while expected_step in pending_events:
                flush_event(expected_step, pending_events.pop(expected_step))
                expected_step += 1
        except Exception as log_exc:
            print(f"Warning: failed to write events/status: {log_exc}")

        return res

    openevolve.database.add = types.MethodType(add_wrapper, openevolve.database)

    # Monkey-patch checkpoint naming: checkpoint_* -> step-*
    orig_save_checkpoint = openevolve._save_checkpoint

    def save_checkpoint_wrapper(self, iteration: int) -> None:
        orig_save_checkpoint(iteration)
        ckpt_root = Path(self.output_dir) / "checkpoints"
        src = ckpt_root / f"checkpoint_{iteration}"
        dst = ckpt_root / f"step-{iteration}"
        if src.exists():
            try:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
                # 保存当前轮的程序快照（区分于 best_program）
                candidates = [
                    p for p in self.database.programs.values() if p.iteration_found == iteration
                ]
                current = None
                if candidates:
                    candidates.sort(
                        key=lambda p: (
                            p.metrics.get("combined_score", float("-inf")),
                            p.timestamp,
                        ),
                        reverse=True,
                    )
                    current = candidates[0]
                if current:
                    cur_path = dst / f"current_program{self.file_extension}"
                    cur_path.write_text(current.code, encoding="utf-8")
            except Exception as move_exc:
                print(f"Warning: failed to rename {src} -> {dst}: {move_exc}")

    openevolve._save_checkpoint = types.MethodType(save_checkpoint_wrapper, openevolve)

    # Initial heartbeat
    write_status("running", 0, error=None)

    best_program = None
    try:
        best_program = await openevolve.run(
            iterations=None,  # use config.max_iterations
            target_score=getattr(config_obj, "target_score", None),
            checkpoint_path=None,
        )
        # Ensure final_result directory naming
        best_dir = output_dir / "best"
        final_dir = output_dir / "final_result"
        if best_dir.exists() and not final_dir.exists():
            try:
                shutil.move(str(best_dir), str(final_dir))
            except Exception as move_exc:
                print(f"Warning: failed to move best -> final_result: {move_exc}")
        # Rename checkpoints/checkpoint_* -> checkpoints/step-*
        ckpt_root = output_dir / "checkpoints"
        if ckpt_root.exists():
            for name in os.listdir(ckpt_root):
                src = ckpt_root / name
                if src.is_dir() and name.startswith("checkpoint_"):
                    step_suffix = name.split("_", 1)[-1]
                    dst = ckpt_root / f"step-{step_suffix}"
                    try:
                        shutil.move(str(src), str(dst))
                    except Exception as move_exc:
                        print(f"Warning: failed to rename {src} -> {dst}: {move_exc}")
        write_status("completed", openevolve.database.last_iteration, error=None)
    except Exception as exc:
        write_status("failed", openevolve.database.last_iteration, error=str(exc))
        raise

    if best_program is None:
        return 1
    return 0


async def main_async() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    json_path = Path(args.config_file).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    merged_cfg, envs = load_config_sources(input_dir, json_path)
    apply_environments(envs)

    merged_yaml_path = output_dir / "config_evolve_merged.yaml"
    with open(merged_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged_cfg, f, sort_keys=False, allow_unicode=True)

    return await run_openevolve(merged_cfg, input_dir, output_dir)


def main() -> int:
    try:
        return asyncio.run(main_async())
    except Exception as exc:
        print(f"Error running OpenEvolve: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
