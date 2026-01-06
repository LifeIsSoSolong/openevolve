#!/usr/bin/env python3
"""Patch agent.py to rebuild data paths from main(root).

This script applies a minimal, deterministic patch:
- Injects root-based path assignments inside main(root)
- Leaves core logic untouched
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List


PATH_VARS = ["ROOT_DIR", "DATA_DIR", "TRAIN_PATH", "VAL_PATH", "TEST_PATH"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix agent.py data paths to use main(root)")
    parser.add_argument("--input-dir", required=True, help="Path to input directory")
    parser.add_argument("--task-type", required=True, choices=["mle", "prompt"], help="Task type")
    parser.add_argument("--write", action="store_true", help="Write changes back to agent.py")
    parser.add_argument("--output", help="Write patched content to file instead of agent.py")
    return parser.parse_args()


def find_main_line(lines: List[str]) -> int | None:
    for idx, line in enumerate(lines):
        if re.match(r"^\s*def\s+main\s*\(", line):
            return idx
    return None


def has_path_import(lines: List[str]) -> bool:
    return any("from pathlib import Path" in line for line in lines)


def insert_import(lines: List[str]) -> List[str]:
    if has_path_import(lines):
        return lines
    # Insert after __future__ import if present, else after shebang or at top
    insert_idx = 0
    if lines and lines[0].startswith("#!"):
        insert_idx = 1
    for idx, line in enumerate(lines):
        if line.startswith("from __future__ import"):
            insert_idx = idx + 1
    lines.insert(insert_idx, "from pathlib import Path")
    return lines


def detect_declared_path_vars(source: str) -> List[str]:
    declared = []
    for name in PATH_VARS:
        if re.search(rf"^\s*{name}\s*=", source, flags=re.M):
            declared.append(name)
    return declared


def build_patch_lines(indent: str, declared: List[str]) -> List[str]:
    lines = []
    lines.append(f"{indent}root_path = Path(root)")
    for name in declared:
        if name == "ROOT_DIR":
            lines.append(f"{indent}ROOT_DIR = root_path")
        elif name == "DATA_DIR":
            lines.append(f"{indent}DATA_DIR = root_path")
        elif name == "TRAIN_PATH":
            lines.append(f"{indent}TRAIN_PATH = root_path / \"train.csv\"")
        elif name == "VAL_PATH":
            lines.append(f"{indent}VAL_PATH = root_path / \"test.csv\"")
        elif name == "TEST_PATH":
            lines.append(f"{indent}TEST_PATH = root_path / \"test.csv\"")
    return lines


def main() -> int:
    args = parse_args()
    if args.task_type != "mle":
        print("[ERROR] fix_agent_paths.py only supports MLE tasks")
        return 2

    input_dir = Path(args.input_dir).resolve()
    agent_path = input_dir / "agent.py"
    if not agent_path.exists():
        print(f"[ERROR] agent.py not found: {agent_path}")
        return 2

    source = agent_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    main_idx = find_main_line(lines)
    if main_idx is None:
        print("[ERROR] main(root) not found in agent.py")
        return 2

    lines = insert_import(lines)
    declared = detect_declared_path_vars("\n".join(lines))

    # Determine insertion point (after docstring if present)
    insert_at = main_idx + 1
    indent = re.match(r"^(\s*)", lines[main_idx]).group(1) + "    "

    if insert_at < len(lines) and lines[insert_at].strip().startswith(("\"\"\"", "'''")):
        quote = lines[insert_at].strip()[:3]
        insert_at += 1
        while insert_at < len(lines):
            if lines[insert_at].strip().endswith(quote):
                insert_at += 1
                break
            insert_at += 1

    patch_lines = build_patch_lines(indent, declared)
    if not patch_lines:
        print("[WARN] No known path variables detected; nothing to patch")
        return 1

    lines[insert_at:insert_at] = patch_lines
    patched = "\n".join(lines) + "\n"

    if args.output:
        Path(args.output).write_text(patched, encoding="utf-8")
        print(f"[OK] Wrote patched agent.py to {args.output}")
        return 0

    if not args.write:
        print(patched)
        return 0

    backup_path = agent_path.with_suffix(".py.bak")
    backup_path.write_text(source, encoding="utf-8")
    agent_path.write_text(patched, encoding="utf-8")
    print(f"[OK] Patched agent.py (backup: {backup_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
