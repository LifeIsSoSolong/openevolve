"""
evaluate_press_agent: call OpenAI  to judge generated press vs reference.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from openai import OpenAI


JUDGE_PROMPT = """
你是一名严谨的新闻稿评审，任务是评估模型生成的新闻稿相对于参考标准稿，在以下维度上的相似度与质量表现。评分结果将用于后续大模型训练，请务必保持打分有区分度，避免大多数样本集中在 8~10 分。\n\n【评估维度】\n1. 结构与叙事逻辑（0-10）：\n- 叙事逻辑或结构是否与参考稿一致或更优？\n- 是否具有清晰的导语→主体故事→政策/数据支撑→总结提升结构？\n- 段落之间是否有自然的过渡与递进，而不是简单堆砌？\n\n2. 文风与写作特点（0-10）：\n- 语言是否凝练、有新闻感，兼顾严肃性与可读性？\n- 是否像黄继妍的稿件那样，将宏观政策、产业脉络与微观故事自然结合？\n- 是否有画面感和细节描写，而非空洞口号和模板化语言？\n\n3. 数据与政策使用（0-10）：\n- 是否恰当地引用了权威数据或具体指标来量化成效（例如产值、增速、就业、投资等）？\n- 是否自然嵌入政策名称、行动计划等，而不是机械罗列？\n- 和参考稿相比，数据与政策的使用方式是否接近？\n\n4. 整体相似度与可读性（0-10）：\n- 综合考虑内容选取、叙事节奏、语气风格，整体读起来是否像同一位作者写的？\n- 是否在保持事实前提下，有适当的提炼与总结，而不是简单照抄参考稿结构？\n\n【打分要求】\n- 每个维度给出一个 0-10 的整数分（0 表示完全不符合，5 表示一般，8 表示很好，9-10 仅用于极为接近参考标准稿的情况）。\n- 最后给出一个综合“最终得分: X/10”，X 为 0-10 的整数，综合以上维度，不是简单平均，可以适度提高结构与文风的权重。\n- 请先按维度逐条简要分析（每个维度 1-2 句），再给出各维度分数，最后一行单独输出：\n最终得分: X/10\n不要输出其他形式的总分。\n\n【参考标准稿】：\n{reference}\n\n【模型生成的新闻稿】：\n{article}
"""


def evaluate_press_agent(
    model_name: str,
    generated_press: str,
    reference_press: str,
    api_base: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 5120,
    temperature: float = 0.0,
    **_: Any,
) -> Dict[str, Any]:
    """
    Judge generated_press vs reference_press using OpenAI.

    Returns:
        dict with numeric metrics and judge outputs for downstream artifacts.
    """
    client = OpenAI(
        base_url=api_base or os.getenv("OPENAI_API_BASE"),
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )
    # print("[EVALUATE] Using model:", model_name)
    # print("[EVALUATE] Reference Press:\n", reference_press)
    # print("[EVALUATE] Generated Press:\n", generated_press)
    user_content = JUDGE_PROMPT.format(reference=reference_press, article=generated_press)
    # print("[EVALUATE] Prompt of evaluate_press_agent:\n", user_content)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            # {"role": "system", "content": "你是严谨的新闻稿评审，严格按提示输出 JSON。"},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = resp.choices[0].message.content.strip()
    print("[EVALUATE] Response of evaluate_press_agent:\n", content)

    # 新打分格式：只关心最后一行 “最终得分: X/10”
    score_match = re.search(r"最终得分\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10", content)
    if not score_match:
        return {
            "metrics": {"combined_score": 0.0, "error": "final score not found"},
            "judge_output": content,
            "judge_summary": _summarize_low_reason(content, {}, 0.0),
            "dimension_scores": {},
        }

    raw_score = float(score_match.group(1))
    combined = max(0.0, min(10.0, raw_score)) / 10.0

    scores = _extract_dimension_scores(content)
    return {
        "metrics": {"combined_score": combined},
        "judge_output": content,
        "judge_summary": _summarize_low_reason(content, scores, combined),
        "dimension_scores": scores,
    }


_DIMENSION_PATTERNS = {
    "structure": [r"结构与叙事逻辑", r"结构与叙事"],
    "style": [r"文风与写作特点", r"文风与写作", r"文风"],
    "data_policy": [r"数据与政策使用", r"数据与政策"],
    "overall_similarity": [r"整体相似度与可读性", r"整体相似度", r"可读性"],
}


def _extract_dimension_scores(content: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for key, patterns in _DIMENSION_PATTERNS.items():
        found = None
        for line in lines:
            for label in patterns:
                if label in line:
                    match = re.search(
                        label + r"[^0-9]{0,10}([0-9]+(?:\.[0-9]+)?)", line
                    )
                    if match:
                        found = float(match.group(1))
                        break
            if found is not None:
                break
        if found is not None:
            scores[key] = found
    return scores


def _summarize_low_reason(
    content: str,
    dimension_scores: Dict[str, float],
    combined_score: float,
    max_chars: int = 400,
) -> str:
    parts = []
    if dimension_scores:
        low_dims = sorted(dimension_scores.items(), key=lambda item: item[1])
        low_dims = low_dims[:2]
        dims_str = ", ".join(f"{name}={score:g}" for name, score in low_dims)
        parts.append(f"lowest_dimensions: {dims_str}")
    parts.append(f"final_score_10pt: {combined_score * 10:.1f}")
    excerpt = _extract_analysis_excerpt(content, max_chars=max_chars)
    if excerpt:
        parts.append(f"judge_excerpt: {excerpt}")
    return " | ".join(parts)


def _extract_analysis_excerpt(content: str, max_chars: int) -> str:
    lines = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if "最终得分" in line:
            continue
        if re.search(r"/\s*10", line) and len(line) <= 40:
            continue
        lines.append(line)
        if len(" ".join(lines)) >= max_chars or len(lines) >= 3:
            break
    return _truncate_text(" ".join(lines), max_chars)


def _truncate_text(text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[truncated]"
