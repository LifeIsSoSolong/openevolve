"""
OpenEvolve evaluator for press02.

For each candidate program:
1) Load prompt_generate_press() from the candidate module.
2) Load train.jsonl (for evolution) and test.jsonl (display only).
3) For each sample, format prompt, call generate_press_agent (OpenAI), then
   evaluate_press_agent (OpenAI) to get scores.
4) Return averaged train metrics (with combined_score). Print test metrics.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple

from openevolve.evaluation_result import EvaluationResult


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
TRAIN_PATH = DATA_DIR / "train.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"

# basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# LLM agent settings (override here as needed)
API_BASE_GENERATE_PRESS = "https://api.moonshot.cn/v1"
API_KEY_GENERATE_PRESS = "sk-fFFLxJCzM9bkVgZ5uYLOlZdjC4d80tllQEgNM4Ins96q4izB"
MODEL_NAME_GENERATE_PRESS = "kimi-k2-turbo-preview"
TEMPERATURE_GENERATE_PRESS = 0.0

# API_BASE_EVALUATE_PRESS = "https://api.deepseek.com/v1"
# API_KEY_EVALUATE_PRESS = "sk-353a88a777bd4c598f17b2923677e100"
# MODEL_NAME_EVALUATE_PRESS = "deepseek-reasoner"
API_BASE_EVALUATE_PRESS = "https://newapi.frontis.top/v1"
API_KEY_EVALUATE_PRESS = "sk-JegueK6Qy16ZJ6zjxfoKBMwQ32kt56LRZgyTPlapD7cGw0QN"
MODEL_NAME_EVALUATE_PRESS = "gpt-5.2"
TEMPERATURE_EVALUATE_PRESS = 0.0
# 并发度（每轮内部同时处理的样本并发数量）
SAMPLE_CONCURRENCY = 10


def _load_module(program_path: str):
    module_name = f"candidate_module_{Path(program_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items




def _ground_truth_stats(ground_truths: List[str]) -> Dict[str, Any]:
    def stats(values: List[int]) -> Dict[str, float]:
        if not values:
            return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": float(mean(values)),
            "median": float(median(values)),
        }

    lengths = [len(t) for t in ground_truths]
    paragraphs = [len([line for line in t.splitlines() if line.strip()]) for t in ground_truths]
    sentences = [sum(t.count(p) for p in "\u3002\uFF1F\uFF01") for t in ground_truths]
    digits = [sum(ch.isdigit() for ch in t) for t in ground_truths]

    return {
        "samples": len(ground_truths),
        "length_chars": stats(lengths),
        "paragraphs": stats(paragraphs),
        "sentences": stats(sentences),
        "digits": stats(digits),
    }


def _evaluate_split(
    dataset: List[Dict[str, Any]],
    system_prompt: str,
    user_prompt_template: str,
    split_name: str,
    output_dir: Path | None = None,
) -> Dict[str, float]:
    from generate_press_agent import generate_press_agent
    from evaluate_press_agent import evaluate_press_agent

    def _process_one(idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        interview_type = sample.get("interview_type", "")
        interview_context = sample.get("interview_context", "")
        groundtruth = sample.get("ground_truth", "")

        user_prompt = user_prompt_template.format(
            interview_type=interview_type,
            interview_context=interview_context,
        )

        log.info("Sample %s: type=%s, interview_len=%s", idx, interview_type, len(interview_context))
        generated_press = ""
        try:
            generated_press = generate_press_agent(
                model_name=MODEL_NAME_GENERATE_PRESS,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_base=API_BASE_GENERATE_PRESS,
                api_key=API_KEY_GENERATE_PRESS,
                temperature=TEMPERATURE_GENERATE_PRESS,
            )
            log.info("Sample %s: generated_press_len=%s", idx, len(generated_press))
        except Exception as exc:
            log.error("Sample %s: generate_press failed: %s", idx, exc)
            return {"combined_score": 0.0, "error": f"generate_press_failed: {exc}"}

        try:
            metrics = evaluate_press_agent(
                model_name=MODEL_NAME_EVALUATE_PRESS,
                generated_press=generated_press,
                reference_press=groundtruth,
                api_base=API_BASE_EVALUATE_PRESS,
                api_key=API_KEY_EVALUATE_PRESS,
                temperature=TEMPERATURE_EVALUATE_PRESS,
            )
        except Exception as exc:
            log.error("Sample %s: evaluate_press failed: %s", idx, exc)
            metrics = {"combined_score": 0.0, "error": str(exc)}

        log.info("Sample %s metrics: %s", idx, metrics)
        # 追加 submission_{split}.jsonl
        if output_dir is not None:
            rec = dict(sample)
            rec.update(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "generated_press": generated_press,
                    "metrics": metrics,
                }
            )
            sub_path = output_dir / f"submission_{split_name}.jsonl"
            sub_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sub_path, "a", encoding="utf-8") as wf:
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return metrics

    # 动态收集所有数值型指标做均值，其余跳过
    numeric_acc: Dict[str, List[float]] = {}

    with ThreadPoolExecutor(max_workers=SAMPLE_CONCURRENCY) as executor:
        future_to_idx = {executor.submit(_process_one, idx, sample): idx for idx, sample in enumerate(dataset)}
        for future in as_completed(future_to_idx):
            metrics = future.result()
            # 确保 combined_score 存在
            if "combined_score" not in metrics:
                metrics["combined_score"] = 0.0
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    numeric_acc.setdefault(k, []).append(float(v))

    def avg(lst: List[float]) -> float:
        return mean(lst) if lst else 0.0

    aggregated = {k: avg(vs) for k, vs in numeric_acc.items()}
    if "combined_score" not in aggregated:
        aggregated["combined_score"] = 0.0
    return aggregated


def evaluate(program_path: str) -> Any:
    """
    Entry point for OpenEvolve.
    Returns train metrics (with combined_score) for evolution guidance.
    """
    try:
        if not TRAIN_PATH.exists():
            return {"combined_score": 0.0, "error": f"train data missing: {TRAIN_PATH}"}

        train_data = _read_jsonl(TRAIN_PATH)
        test_data = _read_jsonl(TEST_PATH) if TEST_PATH.exists() else []

        ground_truths = [item.get("ground_truth", "") for item in train_data]
        ground_truth_stats = _ground_truth_stats(ground_truths)
        sample_index = random.randrange(len(train_data)) if train_data else None
        ground_truth_sample = {}
        if sample_index is not None:
            ground_truth_sample = {
                "index": sample_index,
                "ground_truth": train_data[sample_index].get("ground_truth", ""),
            }
        log.info("Loaded train=%d test=%d", len(train_data), len(test_data))

        # get systemprompt from program
        module = _load_module(program_path)
        prompt_fn = None
        if hasattr(module, "get_prompt_generate_press"):
            prompt_fn = module.get_prompt_generate_press
        else:
            return {"combined_score": 0.0, "error": "prompt_generate_press() not found in program"}

        system_prompt, user_prompt_template = prompt_fn()
        log.info("Get prompt from program, length=%d", len(system_prompt))

        output_dir_env = os.getenv("OPENEVOLVE_OUTPUT_DIR")
        out_dir = Path(output_dir_env) if output_dir_env else None

        train_metrics = _evaluate_split(train_data, system_prompt, user_prompt_template, "train", out_dir)
        test_metrics = _evaluate_split(test_data, system_prompt, user_prompt_template, "test", out_dir) if test_data else {}

        if test_metrics:
            print(f"Test metrics (display only): {test_metrics}")
            # 将测试集指标写回返回结果，便于上层记录
            train_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

        artifacts = {}
        if ground_truth_sample:
            artifacts["train_groundtruth_sample"] = json.dumps(
                ground_truth_sample, ensure_ascii=False, indent=2
            )
        artifacts["train_groundtruth_stats"] = json.dumps(
            ground_truth_stats, ensure_ascii=False, indent=2
        )

        return EvaluationResult(metrics=train_metrics, artifacts=artifacts)
    except Exception as exc:
        return {"combined_score": 0.0, "error": str(exc)}


if __name__ == "__main__":
    prog = ROOT / "agent.py"
    res = evaluate(str(prog))
    print(f"Train metrics: {res}")
