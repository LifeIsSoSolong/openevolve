"""
OpenEvolve evaluator for press01.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
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
API_BASE_GENERATE_PRESS = "https://api.deepseek.com/v1"
API_KEY_GENERATE_PRESS = "sk-353a88a777bd4c598f17b2923677e100"
MODEL_NAME_GENERATE_PRESS = "deepseek-chat"
TEMPERATURE_GENERATE_PRESS = 0.0

API_BASE_EVALUATE_PRESS = "https://api.deepseek.com/v1"
API_KEY_EVALUATE_PRESS = "sk-353a88a777bd4c598f17b2923677e100"
MODEL_NAME_EVALUATE_PRESS = "deepseek-reasoner"
TEMPERATURE_EVALUATE_PRESS = 0.0
# 并发度（每轮样本内部的并发数量）
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


def _evaluate_split(
    dataset: List[Dict[str, Any]],
    prompt_base: str,
) -> Dict[str, float]:
    from generate_press_agent import generate_press_agent
    from evaluate_press_agent import evaluate_press_agent

    def _process_one(idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        interview_type = sample.get("interview_type", "")
        interview_context = sample.get("interview_context", "")
        groundtruth = sample.get("ground_truth", "")

        prompt_final = prompt_base.format(
            interview_type=interview_type,
            interview_context=interview_context,
        )

        log.info("Sample %s: type=%s, interview_len=%s", idx, interview_type, len(interview_context))
        try:
            generated_press = generate_press_agent(
                model_name=MODEL_NAME_GENERATE_PRESS,
                prompt_generate_press_final=prompt_final,
                interview_context=interview_context,
                interview_type=interview_type,
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
                prompt_generate_press_base=prompt_base,
                api_base=API_BASE_EVALUATE_PRESS,
                api_key=API_KEY_EVALUATE_PRESS,
                temperature=TEMPERATURE_EVALUATE_PRESS,
            )
        except Exception as exc:
            log.error("Sample %s: evaluate_press failed: %s", idx, exc)
            metrics = {"combined_score": 0.0, "error": str(exc)}

        log.info(
            "Sample %s metrics: %s",
            idx,
            {k: metrics.get(k) for k in ["combined_score", "overall", "style", "structure", "content", "evidence", "facts_format", "error"]},
        )
        return metrics

    combined, overall, style, structure, content_score, evidence, facts = ([] for _ in range(7))

    with ThreadPoolExecutor(max_workers=SAMPLE_CONCURRENCY) as executor:
        future_to_idx = {executor.submit(_process_one, idx, sample): idx for idx, sample in enumerate(dataset)}
        for future in as_completed(future_to_idx):
            metrics = future.result()
            combined.append(metrics.get("combined_score", 0.0))
            overall.append(metrics.get("overall", 0.0))
            style.append(metrics.get("style", 0.0))
            structure.append(metrics.get("structure", 0.0))
            content_score.append(metrics.get("content", 0.0))
            evidence.append(metrics.get("evidence", 0.0))
            facts.append(metrics.get("facts_format", 0.0))

    def avg(lst: List[float]) -> float:
        return mean(lst) if lst else 0.0

    return {
        "combined_score": avg(combined),
        "overall": avg(overall),
        "style": avg(style),
        "structure": avg(structure),
        "content": avg(content_score),
        "evidence": avg(evidence),
        "facts_format": avg(facts),
    }


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Entry point for OpenEvolve.
    Returns train metrics (with combined_score) for evolution guidance.
    """
    try:
        if not TRAIN_PATH.exists():
            return {"combined_score": 0.0, "error": f"train data missing: {TRAIN_PATH}"}

        train_data = _read_jsonl(TRAIN_PATH)
        test_data = _read_jsonl(TEST_PATH) if TEST_PATH.exists() else []
        log.info("Loaded train=%d test=%d", len(train_data), len(test_data))

        module = _load_module(program_path)
        prompt_fn = None
        if hasattr(module, "get_prompt_generate_press"):
            prompt_fn = module.get_prompt_generate_press
        else:
            return {"combined_score": 0.0, "error": "prompt_generate_press() not found in program"}

        prompt_base = prompt_fn()
        log.info("Get prompt template from program, length=%d", len(prompt_base))

        train_metrics = _evaluate_split(train_data, prompt_base)
        test_metrics = _evaluate_split(test_data, prompt_base) if test_data else {}

        if test_metrics:
            print(f"Test metrics (display only): {test_metrics}")
            # 将测试集指标写回返回结果，便于上层记录
            train_metrics.update(
                {
                    "test_combined_score": test_metrics.get("combined_score", 0.0),
                    "test_overall": test_metrics.get("overall", 0.0),
                    "test_style": test_metrics.get("style", 0.0),
                    "test_structure": test_metrics.get("structure", 0.0),
                    "test_content": test_metrics.get("content", 0.0),
                    "test_evidence": test_metrics.get("evidence", 0.0),
                    "test_facts_format": test_metrics.get("facts_format", 0.0),
                }
            )

        return train_metrics
    except Exception as exc:
        return {"combined_score": 0.0, "error": str(exc)}


if __name__ == "__main__":
    prog = ROOT / "initial_program.py"
    res = evaluate(str(prog))
    print(f"Train metrics: {res}")
