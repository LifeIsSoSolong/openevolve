#!/usr/bin/env python
"""
Wrapper entrypoint for running OpenEvolve on press01 with JSON config overrides.

Invocation:
    python main_press.py --config_file /abs/path/inputs/config.json \
                         --input_dir  /abs/path/inputs/ \
                         --output_dir /abs/path/outputs/
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

# 内置默认配置（原 config_evolve.yaml 内容），前端传入的 config.json 会覆盖对应字段
BASE_CONFIG = {
    "max_iterations": 3,
    "checkpoint_interval": 1,
    "log_level": "INFO",
    "random_seed": 42,
    "llm": {
        "primary_model": "gpt-5.2",
        "primary_model_weight": 0.8,
        "secondary_model": "gpt-5.2",
        "secondary_model_weight": 0.2,
        "api_base": "https://newapi.frontis.top/v1",
        "api_key": "sk-JegueK6Qy16ZJ6zjxfoKBMwQ32kt56LRZgyTPlapD7cGw0QN",
        "temperature": 0.5,
        "max_tokens": 60000,
        "timeout": 300,
        "reasoning_effort": "medium",
    },
    "prompt": {
        "system_message": (
            "你是一名提示词优化专家，目标是改写系统提示词，使生成的新闻稿更贴近黄继妍风格。\n\n"
            "任务对象：\n"
            "- 代码里有 get_prompt_generate_press()，返回 system_prompt 与 user_prompt_template。\n"
            "- user_prompt_template 已固定，负责填入采访类型与采访素材；严禁在 system_prompt 中再次放入这些占位符或移动数据位置。\n"
            "- 只优化 system_prompt，让 generate_press_agent 生成的稿件更像参考稿（train.jsonl）。\n\n"
            "硬性约束：\n"
            "- 仅修改 EVOLVE-BLOCK 内内容。\n"
            "- 语言保持中文，避免幻觉，尊重采访素材，禁止虚构政策/数据/人名。\n"
            "- 输出直接为 SEARCH/REPLACE diff；若无法提供有效修改，返回空 diff。\n\n"
        )
    },
    "database": {
        "population_size": 12,
        "archive_size": 6,
        "num_islands": 2,
        "elite_selection_ratio": 0.25,
        "exploitation_ratio": 0.6,
    },
    "evaluator": {
        "timeout": 30000,
        "parallel_evaluations": 1,
        "cascade_evaluation": False,
    },
    "diff_based_evolution": True,
    "max_code_length": 40000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenEvolve (press02) with JSON overrides.")
    parser.add_argument("--config_file", required=True, help="Path to JSON config file.")
    parser.add_argument("--input_dir", required=True, help="Path to input directory (contains agent.py, judge.py, etc.).")
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
    """Load embedded base config and JSON overrides, then merge."""
    base_cfg = copy.deepcopy(BASE_CONFIG)

    # 若存在 task.goal，则用其中内容覆盖 system_message（优先级高于内置配置）
    task_goal_path = input_dir / "task.goal"
    if task_goal_path.exists():
        try:
            goal_text = task_goal_path.read_text(encoding="utf-8").strip()
            if goal_text:
                base_cfg.setdefault("prompt", {})["system_message"] = goal_text
                preview = (goal_text[:200] + "...") if len(goal_text) > 200 else goal_text
                print(f"[main_press02] Loaded system_message from task.goal: {preview}")
        except Exception as exc:
            print(f"Warning: failed to read task.goal: {exc}")

    with open(json_path, "r", encoding="utf-8") as f:
        json_cfg = json.load(f) or {}

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

    config_obj = Config.from_dict(config_dict)

    if getattr(config_obj, "log_level", None):
        import logging

        logging.getLogger().setLevel(getattr(logging, str(config_obj.log_level).upper(), logging.INFO))

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
    # 让 evaluator 可感知输出目录，便于写 submission_* 文件
    os.environ["OPENEVOLVE_OUTPUT_DIR"] = str(output_dir)

    # Monkey-patch database.add to append events.jsonl and update status.json
    orig_add = openevolve.database.add
    expected_step = 0
    pending_events: Dict[int, Dict[str, Any]] = {}

    def flush_event(step: int, event: Dict[str, Any]) -> None:
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
                    # 取本轮 combined_score 最高的，若缺失则按时间最新
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
                # 将 submission_* 挪到对应 step 目录，便于按轮查看
                for name in ("submission_train.jsonl", "submission_test.jsonl"):
                    src_sub = Path(self.output_dir) / name
                    if src_sub.exists():
                        try:
                            shutil.move(str(src_sub), str(dst / name))
                        except Exception as move_sub_exc:
                            print(f"Warning: failed to move {name} -> {dst}: {move_sub_exc}")
            except Exception as move_exc:
                print(f"Warning: failed to rename {src} -> {dst}: {move_exc}")

    openevolve._save_checkpoint = types.MethodType(save_checkpoint_wrapper, openevolve)

    write_status("running", 0, error=None)

    best_program = None
    try:
        best_program = await openevolve.run(
            iterations=None,
            target_score=getattr(config_obj, "target_score", None),
            checkpoint_path=None,
        )
        # Rename best -> final_result
        best_dir = output_dir / "best"
        final_dir = output_dir / "final_result"
        if best_dir.exists() and not final_dir.exists():
            try:
                shutil.move(str(best_dir), str(final_dir))
            except Exception as move_exc:
                print(f"Warning: failed to move best -> final_result: {move_exc}")
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
