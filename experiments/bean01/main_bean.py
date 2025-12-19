#!/usr/bin/env python
"""
Wrapper entrypoint for running OpenEvolve with JSON config overrides.

Expected invocation (from FastAPI/back-end):
    python main.py --config_file /abs/path/inputs/config.json \
                   --input_dir  /abs/path/inputs/ \
                   --output_dir /abs/path/outputs/

This script will:
1) Load the base YAML config from <input_dir>/source/config_evolve.yaml.
2) Load overrides from the provided JSON config_file.
3) Merge them (JSON wins), write the merged YAML to <output_dir>/config_evolve_merged.yaml.
4) Apply environment variables specified in JSON (under "environments").
5) Run OpenEvolve on <input_dir>/source/{initial_program.py, evaluator.py} and store results in output_dir.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
import types
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from openevolve import OpenEvolve
from openevolve.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenEvolve with JSON overrides.")
    parser.add_argument("--config_file", required=True, help="Path to JSON config file.")
    parser.add_argument("--input_dir", required=True, help="Path to input directory (contains source/).")
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
    """Load base YAML and JSON overrides, then merge."""
    base_yaml_path = input_dir / "source" / "config_evolve.yaml"
    if not base_yaml_path.exists():
        raise FileNotFoundError(f"Base YAML config not found: {base_yaml_path}")

    with open(base_yaml_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    with open(json_path, "r", encoding="utf-8") as f:
        json_cfg = json.load(f) or {}

    # Remove non-config keys that are meta info
    meta_keys = {"algorithm", "task_type"}
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
    source_dir = input_dir / "source"
    initial_program = source_dir / "initial_program.py"
    evaluator_file = source_dir / "evaluator.py"

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

        event = {
            "step": step,
            "type": "initial_eval" if step == 0 else "evolve_eval",
            "score": metrics.get("combined_score"),
            "mape": metrics.get("mape"),
            "rmse": metrics.get("rmse"),
            "best_score": best_metrics.get("combined_score"),
            "best_mape": best_metrics.get("mape"),
            "best_rmse": best_metrics.get("rmse"),
            "timestamp": int(time.time()),
        }
        nonlocal expected_step, pending_events
        try:
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
