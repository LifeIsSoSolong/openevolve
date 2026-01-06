#!/usr/bin/env python
"""
Validate output directory contents for OpenEvolve runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


OUTPUT_REQUIRED_FILES = [
    "events.jsonl",
    "status.json",
    "config_evolve_merged.yaml",
]

OUTPUT_REQUIRED_DIRS = [
    "logs",
    "checkpoints",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check required output files after OpenEvolve finishes."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if required outputs are missing.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def check_outputs(output_dir: Path) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    warnings: List[str] = []

    for name in OUTPUT_REQUIRED_FILES:
        if not (output_dir / name).is_file():
            missing.append(name)

    for name in OUTPUT_REQUIRED_DIRS:
        if not (output_dir / name).is_dir():
            missing.append(f"{name}/")

    logs_dir = output_dir / "logs"
    if logs_dir.is_dir() and not list(logs_dir.glob("*.log")):
        warnings.append("logs/ has no .log files yet")

    ckpt_dir = output_dir / "checkpoints"
    if ckpt_dir.is_dir() and not list(ckpt_dir.glob("step-*")):
        warnings.append("checkpoints/ has no step-* entries yet")

    evo_trace = output_dir / "evolution_trace.jsonl"
    if not evo_trace.is_file():
        warnings.append("evolution_trace.jsonl not found (may be disabled)")

    return missing, warnings


def main() -> int:
    args = parse_args()
    output_dir = normalize_path(Path(args.output_dir))
    if not output_dir.is_dir():
        print(f"[error] output_dir does not exist: {output_dir}")
        return 2

    missing, warnings = check_outputs(output_dir)
    if missing:
        print("[error] Missing required outputs:")
        for name in missing:
            print(f"  - {name}")
        if args.strict:
            return 3

    if warnings:
        print("[warn] Output checks:")
        for msg in warnings:
            print(f"  - {msg}")

    print("[ok] output_dir:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
