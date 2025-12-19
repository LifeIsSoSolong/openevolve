#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert a JSON array file into JSONL (one JSON object per line).

Usage:
    python json2jsonl.py --input data.json --output data.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_json_to_jsonl(input_path: Path, output_path: Path) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array of objects")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSON array to JSONL")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    args = parser.parse_args()

    convert_json_to_jsonl(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
