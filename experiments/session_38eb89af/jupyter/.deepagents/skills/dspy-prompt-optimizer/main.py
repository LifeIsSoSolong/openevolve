"""
标准化的 DSPy Prompt 优化脚本（单一配置文件版）：
- 启动：python src/main.py --config_file config.json --input_dir ./data --output_dir ./outputs
- config.json 合并任务/超参，输入输出路径仅填文件名，通过 input_dir/output_dir 拼接。
- 所有中间/最终结果写入指定 output_dir。
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import re
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union
from loguru import logger

import dspy
import pandas as pd
from dspy.teleprompt import MIPROv2
from dspy.teleprompt.gepa.gepa import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from llm_generate import llm_generate_thread_async  # type: ignore
from extra_agent import fetch_dify
from dspy.utils.callback import BaseCallback

# 运行时模型选择（在 main 中赋值）
RUNTIME_JUDGE_MODEL = "claude"
RUNTIME_JUDGE_PROMPT = ""
JUDGE_MODE = "llm"  # llm | python
PY_SCORER_FN: Any = None

# 运行中用于事件/状态的全局信息
OUTPUT_ROOT: Path | None = None
TESTSET_SIZE: int = 0
AGENT_INPUT_KEYS: list[str] = []
TOTAL_STEPS = 0
USE_EVAL_CALLBACK = False  # 通过 Evaluate 回调直接落盘
EVAL_LOGGER_INSTANCE: Any = None

# 路径与默认常量
ROOT_DIR = Path(__file__).resolve().parents[1]


class EvalRunLogger(BaseCallback):
    """通过 Evaluate 回调捕获一次完整评测的结果并落盘。"""

    def __init__(self, task_dir: Path, total_steps: int, testset_size: int, agent_input_keys: list[str]):
        self.task_dir = task_dir
        self.total_steps = total_steps
        self.testset_size = max(1, testset_size)
        self.agent_input_keys = agent_input_keys
        self.call_info: Dict[str, Dict[str, Any]] = {}
        self.step_counter = 0
        self.best_score = 0.0

    def on_evaluate_start(self, call_id, instance, inputs):
        devset = inputs.get("devset") or []
        meta = inputs.get("callback_metadata") or {}
        # 标记一次完整评测：devset 覆盖了验证集，或 meta 显式标记 eval_full
        is_full = len(devset) >= self.testset_size or meta.get("metric_key") == "eval_full"
        self.call_info[call_id] = {
            "is_full": is_full,
            "devset_size": len(devset),
        }

    def on_evaluate_end(self, call_id, outputs, exception=None):
        info = self.call_info.pop(call_id, None)
        if not info or not info.get("is_full") or outputs is None or exception is not None:
            return
        try:
            results = getattr(outputs, "results", []) or []
            if not results:
                return
            self.step_counter += 1
            step_id = self.step_counter

            ckpt_dir = ensure_dir(self.task_dir / "checkpoints" / f"step-{step_id}")
            result_path = ckpt_dir / "result.jsonl"

            score_sum = 0.0
            prompt_seen = None
            with result_path.open("w", encoding="utf-8") as f:
                for example, prediction, score in results:
                    score_val = float(score) if score is not None else 0.0
                    score_sum += score_val
                    prompt_used = getattr(prediction, "prompt_used", None)
                    if prompt_used and prompt_seen is None:
                        prompt_seen = prompt_used
                        
                    rec = {k: example.get(k) for k in self.agent_input_keys}
                    rec.update(
                        {
                            "ground_truth": getattr(example, "ground_truth", ""),
                            "output": getattr(prediction, "output", ""),
                            "score": score_val,
                        }
                    )
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            avg_score = score_sum / max(1, len(results))
            avg_score_norm = avg_score if avg_score <= 1.0 else avg_score / 100.0
            self.best_score = max(self.best_score, avg_score_norm)

            log_event(
                self.task_dir,
                step=step_id,
                event={
                    "type": "eval",
                    "step": step_id,
                    "extra": {"prompt": prompt_seen},
                    "avg_score_norm": avg_score_norm,
                    "current_best_score": self.best_score,
                    "count": len(results),
                },
            )
            update_status(
                self.task_dir,
                state="running",
                current_step=step_id,
                total_steps=self.total_steps,
                extra={"prompt": prompt_seen},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"EvalRunLogger failed: {e}")

# -------------------------- 工具函数 -------------------------- #
def build_input_key(data: Dict[str, Any]) -> str:
    """对输入字段做稳定哈希，避免大文本直接拼接导致不稳定。"""
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_json_config(path: Path, name: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{name} 不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{name} 解析失败: {e}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(log_path: Path):
    ensure_dir(log_path.parent)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)
    # 同步 loguru 到同一日志/控制台
    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove()
        loguru_logger.add(log_path, enqueue=True)
        loguru_logger.add(sys.stdout)
    except Exception:
        pass


def log_event(task_dir: Path, step: int, event: Dict[str, Any]):
    event_rec = {"step": step, "timestamp": int(time.time())}
    event_rec.update(event)
    events_path = task_dir / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event_rec, ensure_ascii=False) + "\n")


def update_status(task_dir: Path, state: str, current_step: int, total_steps: int, error: str | None = None, extra: Dict[str, Any] | None = None):
    status_path = task_dir / "status.json"
    status = {
        "state": state,
        "current_step": current_step,
        "total_steps": total_steps,
        "last_update": int(time.time()),
        "error": error,
    }
    if extra:
        status["extra"] = extra
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(task_dir: Path, step: int, prompt_state: Dict[str, Any]):
    ckpt_dir = ensure_dir(task_dir / "checkpoints" / f"step-{step}")
    (ckpt_dir / "prompt_state.json").write_text(
        json.dumps(prompt_state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_final_result(task_dir: Path, optimized_template: str, base_template: str, metadata: Dict[str, Any],final_model:Any):
    final_dir = ensure_dir(task_dir / "final_result")
    payload = {
        "prompt-0": optimized_template,
        "base_prompt": base_template,
        "metadata": metadata,
    }
    (final_dir / "best_model.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path = final_dir / "best_model"
    # out_path.mkdir(parents=True, exist_ok=True)
    # final_model.save(str(out_path),save_program=True)
    logger.info(f"save model to {out_path}")


def resolve_path(p: str, base: Path | None = None) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    if base:
        return (base / path).resolve()
    return (ROOT_DIR / path).resolve()


def load_extra_agent(module_path: Path, fn_name: str):
    spec = importlib.util.spec_from_file_location("extra_agent_dynamic", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, fn_name):
        raise AttributeError(f"模块 {module_path} 不包含函数 {fn_name}")
    return module, getattr(module, fn_name)


# -------------------------- DSPy 相关 -------------------------- #
def _normalize_model_key(name: str) -> str:
    """去掉提供商前缀，并对常见别名做一次归一化。"""
    if name.startswith("kimi"):
        return "kimi"
    return name

        
def configure_lm(cfg: Dict[str, Any]):
    """
    根据 config 中的模型配置初始化 DSPy 默认 LM。
    优先使用 optimizer_model_name，并结合 llm_generate.get_model() 中的配置获取 api/key。
    """
    from llm_generate import get_model

    raw_name = (
        cfg.get("optimizer_model_name")
        or cfg.get("dspy_model_name")
        or cfg.get("model_name")
        or "openai/deepseek-chat"
    )
    key = _normalize_model_key(raw_name)
    model_map = get_model()
    entry = model_map.get(key, {})

    model_name = entry.get("model_name", raw_name)
    api_key = entry.get("api_key") or cfg.get("optimizer_model_api_key") or os.getenv("OPENAI_API_KEY")
    api_base = entry.get("api") or cfg.get("optimizer_model_api_base") or os.getenv("OPENAI_API_BASE")
    temperature = float(cfg.get("optimizer_temperature", 0.3))

    if not api_key:
        raise RuntimeError("未提供可用的 API key（optimizer_model_api_key 或 OPENAI_API_KEY）")
    if not api_base:
        raise RuntimeError("未提供可用的 API base（optimizer_model_api_base 或 OPENAI_API_BASE）")

    lm = dspy.LM(
        model="openai/"+model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
    )
    dspy.configure(lm=lm)


def build_reflection_lm(params: Dict[str, Any]) -> dspy.LM:
    model = params.get("reflection_model", "openai/deepseek-chat")
    api_key = params.get("reflection_api_key", "sk-353a88a777bd4c598f17b2923677e100")
    api_base = params.get("reflection_api_base", "https://api.deepseek.com/v1")
    temperature = params.get("reflection_temperature", 1.0)
    max_tokens = params.get("reflection_max_tokens", 32000)
    return dspy.LM(
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _load_jsonl_dataset(path: Path, base_template: str):
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件: {path}")
    records = []
    global AGENT_INPUT_KEYS
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败 {path}: {e}") from e
            # 动态识别 agent 输入：除 ground_truth 外的所有字段
            if not AGENT_INPUT_KEYS:
                AGENT_INPUT_KEYS = [k for k in obj.keys() if k != "ground_truth"]
            inputs = {k: obj.get(k, "") for k in AGENT_INPUT_KEYS}
            example = dspy.Example(
                ground_truth=str(obj.get("ground_truth", "")),
                split="train",
                **inputs,
            ).with_inputs(*AGENT_INPUT_KEYS)
            records.append(example)
    if not records:
        raise ValueError(f"{path} 数据为空")
    return records


def load_datasets(
    train_path: Path,
    base_template: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    test_path: Path | None = None,
):
    global AGENT_INPUT_KEYS
    if train_path.suffix.lower() == ".jsonl" or (test_path and test_path.suffix.lower() == ".jsonl"):
        trainset = _load_jsonl_dataset(train_path, base_template)
        if not test_path:
            raise ValueError("使用 JSONL 模式时，需同时提供 test_data（test.jsonl）")
        testset = _load_jsonl_dataset(test_path, base_template)
        # 标记 split
        for ex in trainset:
            setattr(ex, "split", "train")
        for ex in testset:
            setattr(ex, "split", "test")
        if AGENT_INPUT_KEYS:
            logger.info(f"检测到动态 Agent 输入字段: {AGENT_INPUT_KEYS}")
        return trainset, testset

    if not train_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {train_path}")
    try:
        df = pd.read_excel(train_path)
    except ImportError as e:
        raise RuntimeError("读取 Excel 需要安装 openpyxl：pip install openpyxl") from e

    df = df.copy()
    df["ground_truth"] = df.get("ground_truth", "")
    if not AGENT_INPUT_KEYS:
        AGENT_INPUT_KEYS = [c for c in df.columns if c != "ground_truth"]
    logger.info(f"检测到动态 Agent 输入字段: {AGENT_INPUT_KEYS}")

    row_buffer = []
    for _, row in df.iterrows():
        inputs = {k: str(row.get(k, "")) for k in AGENT_INPUT_KEYS}
        row_buffer.append({"inputs": inputs, "ground_truth": str(row.get("ground_truth", ""))})

    if not row_buffer:
        raise ValueError("数据为空，无法构建 train/test。")

    import random

    rng = random.Random(seed)
    rng.shuffle(row_buffer)
    split_idx = max(1, int(len(row_buffer) * train_ratio))
    split_idx = min(split_idx, len(row_buffer) - 1)
    train_rows, test_rows = row_buffer[:split_idx], row_buffer[split_idx:]

    trainset, testset = [], []
    for item in train_rows:
        ex = dspy.Example(
            ground_truth=item["ground_truth"],
            split="train",
            **item["inputs"],
        ).with_inputs(*AGENT_INPUT_KEYS)
        trainset.append(ex)
    for item in test_rows:
        ex = dspy.Example(
            ground_truth=item["ground_truth"],
            split="test",
            **item["inputs"],
        ).with_inputs(*AGENT_INPUT_KEYS)
        testset.append(ex)

    return trainset, testset

class PressPromptSig(dspy.Signature):
    base_template: str = dspy.InputField(desc="当前使用的提示")
    # optimization_goal: str = dspy.InputField(desc="对 base_template 进行改写时的全局优化目标描述。")
    # ground_truth: str = dspy.InputField(desc="通过优化后的提示，期望生成的目标内容。引导模板优化方向。")
    prompt_template: str = dspy.OutputField(
        desc="优化后的提示。"
    )

class PressAgentWrapper(dspy.Module):
    def __init__(
        self,
        agent_fn: Callable[..., str],
        base_template: str,
        optimization_goal: str,
        model_name: str = "gpt5",
        temperature: float = 0.3,
    ):
        super().__init__()
        self.prompt_generator = dspy.ChainOfThought(PressPromptSig)
        self.agent_fn = agent_fn
        self.model_name = model_name
        self.temperature = temperature
        self.base_template = base_template
        self.optimization_goal = optimization_goal

    def forward(self, **kwargs):
        prompt_out = self.prompt_generator(
            base_template= self.base_template,
            # optimization_goal=self.optimization_goal
        )
        optimized_template = prompt_out.prompt_template

        try:
            start_time = time.time()
            result = self.agent_fn(
                prompt=optimized_template,
                **{k: kwargs.get(k) for k in AGENT_INPUT_KEYS},  # 通过动态传参数来传递参数，适应不同的agent
            )
            logger.info(f"extra agent_fn latency：: {time.time() - start_time:.2f} seconds")
            if isinstance(result, dict):
                code = result.get("code", 500)
                if code != 200:
                    err_msg = result.get("msg") or f"agent 调用失败，code={code}"
                    logger.info(err_msg)
                output = result.get("output") or ""
            else:
                output = str(result)
        except Exception as e:  # noqa: BLE001
            output = ""
            logger.info(str(e))

        return dspy.Prediction(output=output, prompt_used=optimized_template)


def _parse_score(text: str) -> float:
    if not text:
        return 0.0
    m = re.search(r"最终得分[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10", text)
    if not m:
        m = re.search(r"score[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10", text, re.IGNORECASE)
    if m:
        try:
            return max(0.0, min(10.0, float(m.group(1))))
        except Exception:
            pass
    nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", text)
    if nums:
        try:
            return max(0.0, min(10.0, float(nums[-1])))
        except Exception:
            return 0.0
    return 0.0


async def _llm_score_async(output: str, reference: str, model_name: str, temperature: float = 0.2) -> Tuple[float, str]:
    if not reference:
        return 0.0, ""

    judge_prompt = (RUNTIME_JUDGE_PROMPT or "").format(reference=reference, output=output)
    messages = judge_prompt
    text = await llm_generate_thread_async(
        text=messages,
        model_name=model_name,
        temperature=temperature,
    )
    logger.info(f"评分结果：{_parse_score(text)}")
    return _parse_score(text), text


def llm_score(output: str, reference: str, model_name: str, temperature: float = 0.2) -> Tuple[float, str]:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_llm_score_async(output, reference, model_name, temperature))
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def judge_score(reference: str, output: str) -> Tuple[float, str]:
    """根据 judge 类型选择 LLM 或 Python scorer。统一返回 raw_score (0-10) 和反馈文本。"""
    if JUDGE_MODE == "python" and PY_SCORER_FN:
        try:
            score_raw = float(PY_SCORER_FN(reference, output))
            score_raw = max(0.0, min(1.0, score_raw)) * 10.0
            return score_raw, ""
        except Exception as e:  # noqa: BLE001
            logger.warning(f"python judge 评分失败: {e}")
            return 0.0, str(e)

    raw_score, feedback_text = llm_score(output, reference, model_name=RUNTIME_JUDGE_MODEL, temperature=0)
    return raw_score, feedback_text


def _score_and_handle(example: dspy.Example, pred: Union[dspy.Prediction, dict, None]) -> Tuple[float, str]:
    """计算得分并统一处理事件/写盘，返回 (score_norm, feedback_text)。"""
    reference = getattr(example, "ground_truth", "") or ""
    # output = getattr(pred, "output", "") or ""
    if pred is None:
        output = ""
    elif isinstance(pred, dict):
        # 主要兼容 dict["output"]，可按需增加兜底字段
        output = pred.get("output") or ""
    else:
        # dspy.Prediction 或其他带 .output 属性的对象
        output = getattr(pred, "output", "") or ""
    if not reference or not output:
        return 0.0, ""

    raw_score, feedback_text = judge_score(reference, output)
    score_norm = max(0.0, min(1.0, raw_score / 10.0))
    if score_norm == 0:
        logger.warning(f"零分输出，生成内容：{output}")
    return score_norm, feedback_text

def metric(example: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    score_norm, _ = _score_and_handle(example, pred)
    return score_norm

def press_metric_with_feedback(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
    score_norm, feedback_text = _score_and_handle(gold, pred)
    return ScoreWithFeedback(score=score_norm, feedback=feedback_text)

# -------------------------- 主流程 -------------------------- #
def main():
    parser = argparse.ArgumentParser(description="标准化 DSPy Prompt 优化任务运行器（单配置版）")
    parser.add_argument("--config_file", required=True, help="合并任务+超参的 JSON 配置文件")
    parser.add_argument("--input_dir", required=True, help="输入文件所在目录（配置中仅写文件名）")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    args = parser.parse_args()


    cfg = load_json_config(Path(args.config_file),"config")
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    global OUTPUT_ROOT
    OUTPUT_ROOT = output_root

    task_dir = ensure_dir(output_root)
    log_path = task_dir / "stdout.log"
    setup_logging(log_path)

    total_steps = 11
    global TOTAL_STEPS
    TOTAL_STEPS = total_steps

    # 解析任务配置
    extra_agent_path = resolve_path(cfg.get("extra_agent_path", "agent.py"), base=input_root)
    extra_agent_fn_name = cfg.get("extra_agent_fn", "generate_press_release")
    train_data_path = resolve_path(cfg.get("train_data", "huangjiyan_press_rev_interview.xlsx"), base=input_root)
    test_data_path_val = cfg.get("test_data")
    test_data_path = resolve_path(test_data_path_val, base=input_root) if test_data_path_val else None
    base_prompt = cfg.get("base_prompt")
    base_prompt_file = cfg.get("base_prompt_file")
    optimization_goal = cfg.get(
        "optimization_goal",
        "聚焦 base_template 的核心任务，确保忠实采访资料并提升文风/结构相似度。",
    )
    model_name_total = _normalize_model_key(cfg.get("optimizer_model_name") or cfg.get("dspy_model_name") or cfg.get("model_name", "deepseek"))
    judge_model_name = cfg.get("judge_model_name", "claude")
    # 评测提示从 external_evaluate_path 读取，若不存在则使用默认
    # 自动从 input_dir 下寻找 judge.py / judge.prompt（优先 .py）
    judge_prompt_text = None
    judge_py_path = input_root / "judge.py"
    judge_prompt_path = input_root / "judge.prompt"
    if judge_py_path.exists():
        spec = importlib.util.spec_from_file_location("judge_module", judge_py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 judge 模块: {judge_py_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "cal_eval_score"):
            raise AttributeError(f"judge 模块缺少 cal_eval_score: {judge_py_path}")
        global PY_SCORER_FN, JUDGE_MODE
        PY_SCORER_FN = getattr(module, "cal_eval_score")
        JUDGE_MODE = "python"
        logger.info(f"使用 Python judge: {judge_py_path}")
    elif judge_prompt_path.exists():
        judge_prompt_text = judge_prompt_path.read_text(encoding="utf-8")
        logger.info(f"使用 judge prompt: {judge_prompt_path}")
    if not judge_prompt_text and JUDGE_MODE != "python":
        judge_prompt_text = "你是一名严苛的资深新闻编辑，负责评估“模型生成的新闻稿”相对于“专家参考标准稿”的仿写质量。你的任务是：严格区分出【高质量仿写】与【低质量仿写】，评分结果将用于模型强化学习，请务必保持评分有区分度。\n\n【输出格式要求】\n逐条分析，最后一行输出“最终得分: X/10”，只允许 0-10 之间数字。\n\n参考标准稿：\n{reference}\n\n模型生成的新闻稿：\n{output}\n"
    global RUNTIME_JUDGE_MODEL, RUNTIME_JUDGE_PROMPT
    RUNTIME_JUDGE_MODEL = judge_model_name
    RUNTIME_JUDGE_PROMPT = judge_prompt_text or ""

    # 加载 extra_agent
    # 根据inputs文件夹中的文件进行判定，如果inputs里面有agent.py就是非dify模式，如果有agent.json就是dify模式，dify模式优先
    agent_json_path = input_root / "agent.json"
    agent_py_path = input_root / "agent.py"
    dify_mode = agent_json_path.exists()
    if dify_mode:
        logger.info("enter dify mode")
        workflow_path = agent_json_path
        module_base_prompt = fetch_dify.extract_system_prompt_text(fetch_dify.load_workflow(workflow_path))
        def agent_fn(prompt: str, **agent_inputs):
            base_host = os.getenv("DIFY_BASE_URL") or "http://10.200.2.52"
            base_url = base_host.rstrip("/")
            if not base_url.endswith("/api/v1/agents/tasks/workflow"):
                base_url = f"{base_url}/api/v1/agents/tasks/workflow"
            return fetch_dify.run_dify_with_prompt(
                workflow_path=workflow_path,
                prompt_text=prompt,
                input_vars=agent_inputs,
                base_url=base_url,
            )
    else:
        if not agent_py_path.exists():
            raise FileNotFoundError(f"未找到 agent.py: {agent_py_path}")
        logger.info("enter extra agent mode")
        agent_module, agent_fn_raw = load_extra_agent(agent_py_path, extra_agent_fn_name)
        module_base_prompt = getattr(agent_module, "BASE_PROMPT_TEMPLATE", "")

        def agent_fn(prompt: str, **agent_inputs):
            return agent_fn_raw(
                prompt=prompt,
                model_name=agent_inputs.get("model_name", model_name_total),
                temperature=agent_inputs.get("temperature", generator_temperature),
                **{k: v for k, v in agent_inputs.items() if k not in {"model_name", "temperature", "prompt"}},
            )

    if base_prompt_file:
        bpf = resolve_path(base_prompt_file, base=input_root)
        if not bpf.exists():
            raise FileNotFoundError(f"base_prompt_file 不存在: {bpf}")
        base_template = bpf.read_text(encoding="utf-8")
    else:
        base_template = base_prompt or module_base_prompt
    if not base_template:
        raise ValueError("未提供 base_prompt，且模块中不存在 BASE_PROMPT_TEMPLATE")

    configure_lm(cfg)

    train_ratio = float(cfg.get("train_ratio", 0.8))
    seed = int(cfg.get("seed", 42))
    generator_temperature = float(cfg.get("generator_temperature", 0.3))
    gepa_params = cfg.get("gepa", {})
    mipro_params = cfg.get("mipro", {})
    optimizer_name = str(cfg.get("optimizer", "mipro")).lower()
    num_trials = int(cfg.get("num_trials", 0))

    trainset, testset = load_datasets(
        train_data_path,
        base_template=base_template,
        train_ratio=train_ratio,
        seed=seed,
        test_path=test_data_path,
    )
    logging.info(f"加载数据完成：train={len(trainset)}, test={len(testset)}")
    global TESTSET_SIZE
    TESTSET_SIZE = len(testset)
    global USE_EVAL_CALLBACK
    USE_EVAL_CALLBACK = True
    global EVAL_LOGGER_INSTANCE

    program = PressAgentWrapper(
        agent_fn=agent_fn,
        base_template=base_template,
        optimization_goal=optimization_goal,
        model_name=model_name_total,
        temperature=generator_temperature,
    )

    if optimizer_name in ("miprov2", "mipro"):
        teleprompter = MIPROv2(
            metric=metric,
            auto=None,
            num_candidates=mipro_params.get("num_candidates"),
            num_threads=mipro_params.get("num_threads"),
            max_errors=mipro_params.get("max_errors"),
            max_bootstrapped_demos=mipro_params.get("max_bootstrapped_demos", 4),
            max_labeled_demos=mipro_params.get("max_labeled_demos", 4),
            verbose=bool(mipro_params.get("verbose", False)),
            track_stats=bool(mipro_params.get("track_stats", True)),
            # log_dir=str(task_dir / "mipro_logs"),
            metric_threshold=mipro_params.get("metric_threshold"),
            seed=seed
            
        )
        num_trials = mipro_params.get("num_trials", 0)
        TOTAL_STEPS = num_trials if num_trials > 0 else TOTAL_STEPS
        logging.info("开始优化程序（MIPROv2）")
    elif optimizer_name in ("gepa", "default"):
        teleprompter = GEPA(
            metric=press_metric_with_feedback,
            auto=gepa_params.get("auto", "light"),
            reflection_lm=build_reflection_lm(gepa_params),
            track_stats=gepa_params.get("track_stats", True),
            track_best_outputs=gepa_params.get("track_best_outputs", True),
            add_format_failure_as_feedback=gepa_params.get("add_format_failure_as_feedback", False),
            use_merge=gepa_params.get("use_merge", True),
            num_threads=gepa_params.get("num_threads", 4),
            seed=seed,
            log_dir=str(task_dir / "gepa_logs"),
        )
        logging.info("开始优化程序（GEPA）")
    else:
        raise ValueError(f"不支持的 optimizer 配置: {optimizer_name}")
    # 配置回调日志
    EVAL_LOGGER_INSTANCE = EvalRunLogger(task_dir, TOTAL_STEPS, TESTSET_SIZE, AGENT_INPUT_KEYS)
    dspy.settings.configure(callbacks=[EVAL_LOGGER_INSTANCE])
    update_status(task_dir, state="running", current_step=0, total_steps=TOTAL_STEPS)
    # 启动训练任务
    if num_trials:
        print (f"使用 num_trials={num_trials} 进行优化...")
        optimized_program = teleprompter.compile(student=program, trainset=trainset, valset=testset, num_trials=num_trials-1,minibatch=False,provide_traceback=True)
    else:
        print (f"使用 num_trials={num_trials} 进行优化...")
        optimized_program = teleprompter.compile(student=program, trainset=trainset, valset=testset)
    # 从 events.jsonl 中选取得分最高的 step，并提取对应 prompt
    best_prompt_template = base_template
    best_step = None
    events_path = task_dir / "events.jsonl"
    if events_path.exists():
        best_score = -1.0
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "eval":
                continue
            score_val = obj.get("score_norm") or obj.get("avg_score_norm")
            if score_val is None:
                continue
            if score_val > best_score:
                best_score = score_val
                best_step = obj.get("step")
                extra = obj.get("extra") or {}
                best_prompt_template = extra.get("prompt")
    optimized_template = best_prompt_template

    # 从最佳 step 的 checkpoint 读取评测结果，便于对齐输出
    optimized_records = {}
    if best_step is not None:
        result_path = task_dir / "checkpoints" / f"step-{best_step}" / "result.jsonl"
        if result_path.exists():
            for line in result_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key_payload = {k: rec.get(k, "") for k in AGENT_INPUT_KEYS}
                key = rec.get("input_key") or build_input_key(key_payload)
                optimized_records[key] = rec

    # 评估并写出对比结果
    results_path = task_dir / "outputs.csv"
    rows = []
    for split_name, dataset in (("test", testset),):
        for idx, ex in enumerate(dataset, start=1):
            input_payload = {k: getattr(ex, k, "") for k in AGENT_INPUT_KEYS}
            base_result = agent_fn(
                prompt=base_template,
                model_name=program.model_name,
                temperature=program.temperature,
                **input_payload,
            )
            base_output = base_result.get("output") if isinstance(base_result, dict) else str(base_result)
            # 优先使用 checkpoint 中的输出/得分
            opt_key = build_input_key(input_payload)
            opt_record = optimized_records.get(opt_key, {})
            optimized_output = opt_record.get("output","") 
            reference = getattr(ex, "ground_truth", "") or ""
            if reference:
                base_score_val, base_feedback = _score_and_handle(ex, base_result)
                # optimized_score
                optimized_score_val = opt_record.get("score",0)
                optimized_feedback = ""
            else:
                base_score_val, optimized_score_val = 0.0, 0.0
                base_feedback = optimized_feedback = ""

            rows.append(
                {
                    "split": split_name,
                    "id": idx,
                    "ground_truth": reference,
                    "base_prompt": base_template,
                    "optimized_prompt": optimized_template,
                    "base_output": base_output,
                    "optimized_output": optimized_output,
                    "base_score": base_score_val,
                    "optimized_score": optimized_score_val,
                    **{k: input_payload.get(k, "") for k in AGENT_INPUT_KEYS},
                }
            )

    pd.DataFrame(rows).to_csv(results_path, index=False, encoding="utf-8")

    save_final_result(
        task_dir,
        optimized_template=optimized_template,
        base_template=base_template,
        metadata={
            "model_name": model_name_total,
            "judge_model_name": judge_model_name,
            "train_size": len(trainset),
            "test_size": len(testset),
            "timestamp": int(time.time()),
            "best_prompt_template": best_prompt_template,
        },
        final_model=optimized_program
    )

    final_step = EVAL_LOGGER_INSTANCE.step_counter if USE_EVAL_CALLBACK and EVAL_LOGGER_INSTANCE else 0
    update_status(task_dir, state="completed", current_step=final_step, total_steps=TOTAL_STEPS)
    logging.info(f"任务完成，结果已保存到 {task_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"任务失败: {exc}", file=sys.stderr)
        try:
            # 尽量写入已有的输出目录，保持 current_step/total_steps 信息
            fallback_output = OUTPUT_ROOT or Path("outputs").resolve()
            task_dir = ensure_dir(fallback_output)
            fallback_step = 0
            try:
                if USE_EVAL_CALLBACK and EVAL_LOGGER_INSTANCE:
                    fallback_step = EVAL_LOGGER_INSTANCE.step_counter
            except Exception:
                fallback_step = 0
            update_status(
                task_dir,
                state="failed",
                current_step=fallback_step,
                total_steps=TOTAL_STEPS if TOTAL_STEPS else 0,
                error=str(exc),
            )
        except Exception:
            pass
        raise
