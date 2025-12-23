#!/usr/bin/env python
"""
Validate input directory contents for OpenEvolve tasks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


COMMON_REQUIRED = [
    "agent.py",
    "judge.py",
    "config.json",
    "task.goal",
]

TASK_REQUIREMENTS = {
    "mle": COMMON_REQUIRED
    + [
        "train.csv",
        "test.csv",
    ],
    "prompt": COMMON_REQUIRED
    + [
        "train.jsonl",
        "test.jsonl",
        "generate_press_agent.py",
        "evaluate_press_agent.py",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check required input files before running OpenEvolve."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Path to input directory. If omitted, search for inputs/ under the skill root.",
    )
    parser.add_argument(
        "--task-type",
        choices=["auto", "mle", "prompt", "generic"],
        default="auto",
        help="Task type for input validation. auto detects by required files.",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="Extra required files relative to input-dir. Can be repeated.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def has_all(input_dir: Path, required: Iterable[str]) -> bool:
    return all((input_dir / name).exists() for name in required)


def detect_task_type(input_dir: Path) -> str:
    if has_all(input_dir, TASK_REQUIREMENTS["prompt"]):
        return "prompt"
    if has_all(input_dir, TASK_REQUIREMENTS["mle"]):
        return "mle"
    if has_all(input_dir, COMMON_REQUIRED):
        return "generic"
    return "unknown"


def resolve_task_type(requested: str, input_dir: Path) -> str:
    if requested != "auto":
        return requested
    return detect_task_type(input_dir)


def missing_files(input_dir: Path, required: Iterable[str]) -> List[str]:
    return [name for name in required if not (input_dir / name).exists()]


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_inputs_under(root: Path) -> List[Path]:
    return [path for path in root.rglob("inputs") if path.is_dir()]


def pick_input_dir(input_dir: Optional[str]) -> Tuple[Optional[Path], List[Path]]:
    if input_dir:
        return normalize_path(Path(input_dir)), []

    root = skill_root()
    candidates = find_inputs_under(root)
    if len(candidates) == 1:
        return candidates[0].resolve(), candidates
    return None, candidates


def main() -> int:
    args = parse_args()

    input_dir, candidates = pick_input_dir(args.input_dir)
    if input_dir is None:
        if not candidates:
            print("[error] No inputs/ directory found under skill root.")
        else:
            print("[error] Multiple inputs/ directories found. Please pass --input-dir.")
            for path in candidates:
                print(f"  - {path}")
        return 2

    if not input_dir.is_dir():
        print(f"[error] input_dir does not exist: {input_dir}")
        return 2

    task_type = resolve_task_type(args.task_type, input_dir)
    if task_type == "unknown":
        print("[error] Unable to detect task type from input files.")
        return 2

    required = list(COMMON_REQUIRED) if task_type == "generic" else list(TASK_REQUIREMENTS[task_type])
    required.extend(args.require)

    missing = missing_files(input_dir, required)
    if missing:
        print("[error] Missing required input files:")
        for name in missing:
            print(f"  - {name}")
        return 2

    print("[ok] input_dir:", input_dir)
    print("[ok] task_type:", task_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
