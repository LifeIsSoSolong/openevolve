#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate press submissions for a given prompt program and test set.

Example:
    python run_press_submission.py \
      --prompt_program_path "D:\\清华工程博士\\C3I\\daguan\\agentic-rl\\mle-openevolve\\experiments\\press01\\inputs\\source\\initial_program.py" \
      --test_data_path "D:\\清华工程博士\\C3I\\daguan\\agentic-rl\\mle-openevolve\\experiments\\press01\\inputs\\source\\data\\test.jsonl" \
      --output_path "submission.jsonl"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time

# 固定引用已有的 agent/evaluator 配置
generate_press_agent_path = (
    r"D:\清华工程博士\C3I\daguan\agentic-rl\mle-openevolve\experiments\press01\inputs\source\generate_press_agent.py"
)
evaluator_path = (
    r"D:\清华工程博士\C3I\daguan\agentic-rl\mle-openevolve\experiments\press01\inputs\source\evaluator.py"
)


def _load_module(path: Path, attr_hint: str | None = None):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def get_prompt_generate_press_fromprogram(prompt_program_path: Path) -> str:
    mod = _load_module(prompt_program_path)
    if hasattr(mod, "get_prompt_generate_press"):
        return mod.get_prompt_generate_press()
    elif hasattr(mod, "prompt_generate_press"):
        return mod.prompt_generate_press
    else:
        raise AttributeError("prompt_generate_press not found in program")


def _load_test_data(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _process_one(
    idx: int,
    sample: Dict[str, Any],
    prompt_base: str,
    generate_press_agent,
    model_name: str,
    api_base: str | None,
    api_key: str | None,
    temperature: float,
    throttle: float,
) -> Tuple[int, Dict[str, Any]]:
    interview_type = sample.get("interview_type", "")
    interview_context = sample.get("interview_context", "")
    prompt_final = prompt_base.format(
        interview_type=interview_type,
        interview_context=interview_context,
    )
    if throttle and throttle > 0:
        time.sleep(throttle)
    generated_press = generate_press_agent(
        model_name=model_name,
        prompt_generate_press_final=prompt_final,
        interview_context=interview_context,
        interview_type=interview_type,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
    )
    rec = dict(sample)
    rec["generated_press"] = generated_press
    return idx, rec


def generate_press(
    prompt_program_path: Path,
    test_data_path: Path,
    output_path: Path,
    workers: int = 1,
    throttle_seconds: float = 0.0,
) -> None:
    # 加载 prompt
    prompt_base = get_prompt_generate_press_fromprogram(prompt_program_path)

    # 加载测试数据
    records = _load_test_data(test_data_path)

    # 加载 generate_press_agent 和 evaluator 中的接口配置
    agent_mod = _load_module(Path(generate_press_agent_path))
    evaluator_mod = _load_module(Path(evaluator_path))

    generate_press_agent = getattr(agent_mod, "generate_press_agent")
    api_base = getattr(evaluator_mod, "API_BASE_GENERATE_PRESS", None)
    api_key = getattr(evaluator_mod, "API_KEY_GENERATE_PRESS", None)
    model_name = getattr(evaluator_mod, "MODEL_NAME_GENERATE_PRESS", "gpt-5.1")
    temperature = getattr(evaluator_mod, "TEMPERATURE_GENERATE_PRESS", 0.0)

    outputs: List[Dict[str, Any]] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                _process_one,
                idx,
                sample,
                prompt_base,
                generate_press_agent,
                model_name,
                api_base,
                api_key,
                temperature,
                throttle_seconds,
            )
            for idx, sample in enumerate(records)
        ]
        for future in as_completed(futures):
            idx, rec = future.result()
            outputs[idx] = rec

    with open(output_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run PRESS submission for news generation.")
    parser.add_argument("--prompt_program_path", type=str, required=True, help="Path to the prompt program file.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated news output.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrency for generation (thread pool size).")
    parser.add_argument("--throttle_seconds", type=float, default=1.0, help="Optional sleep seconds before each generation call (per task).")
    args = parser.parse_args()

    prompt_program_path = Path(args.prompt_program_path).resolve()
    test_data_path = Path(args.test_data_path).resolve()
    output_path = Path(args.output_path).resolve()

    generate_press(
        prompt_program_path,
        test_data_path,
        output_path,
        workers=args.workers,
        throttle_seconds=args.throttle_seconds,
    )
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
