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
你是一名严谨的新闻稿评审，任务是评价“模型生成的新闻稿”与“专家参考标准稿”的一致性，给出分数和理由。

一共6个打分维度，如下：
1. 语言与文风的一致性  
看用词、语气、节奏，是否贴近参考稿的专家风格。

2. 结构与叙事的一致性  
看整体结构、段落安排、信息展开顺序是否与参考稿的叙事路径一致，是否同样条理清晰、无明显堆砌或重复。

3. 内容要点与主题的一致性  
看是否与参考稿的核心主题和关键要点一致，有无明显要点缺失或偏题内容。

4. 细节支撑方式的一致性（案例/现场/数据/政策）  
用具体案例、现场描写、权威数据、政策名称与解读来支撑主题看是否像参考稿一致，而不是泛泛而谈。

5. 事实与格式习惯的一致性  
看主要事实、数据方向、因果关系是否与参考稿保持一致，有无严重错误或虚构；标点、数字单位、时间写法、引用习惯是否大体规范并接近参考稿。若本项为 0 分，则总分不得高于 4 分。

6. 综合评分  
综合上述各项，给出一个整体评分，不能高于各项均值。

每个维度各 0~10 打分，并输出 JSON：
{{
  "style": 0-10,           # 语言与风格一致性
  "structure": 0-10,       # 结构与叙事一致性
  "content": 0-10,         # 主题/要点覆盖
  "evidence": 0-10,        # 细节/案例/数据支撑
  "facts_format": 0-10,    # 事实与格式习惯一致性
  "overall": 0-10,         # 综合分，不能高于上述维度均值
  "comments": "简要问题与改进建议"
}}

请严格输出该格式的 JSON，无额外文本。

【专家参考标准稿】
{reference}

【模型生成的新闻稿】
{article}
"""




def evaluate_press_agent(
    model_name: str,
    generated_press: str,
    reference_press: str,
    prompt_generate_press_base: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 5120,
    temperature: float = 0.0,
    **_: Any,
) -> Dict[str, float]:
    """
    Judge generated_press vs reference_press using OpenAI.

    Returns:
        dict with per-dimension scores and combined_score (overall/10).
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
            {"role": "system", "content": "你是严谨的新闻稿评审，严格按提示输出 JSON。"},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = resp.choices[0].message.content.strip()
    # print("[EVALUATE] Response of evaluate_press_agent:\n", content)

    def _extract_json(text: str) -> Dict[str, Any]:
        """Try best-effort extraction of JSON payload from the judge response."""
        candidates = [text.strip()]
        # remove markdown fences like ```json ... ```
        if text.strip().startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\\s*", "", text.strip())
            cleaned = re.sub(r"```\\s*$", "", cleaned).strip()
            candidates.append(cleaned)

        for cand in candidates:
            try:
                return json.loads(cand)
            except Exception:
                pass

        # fallback: first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            fragment = text[start : end + 1]
            try:
                return json.loads(fragment)
            except Exception:
                pass

        # fallback: regex parse numeric lines like `style: 8`
        num_pattern = re.compile(
            r'"?(style|structure|content|evidence|facts_format|overall)"?\\s*[:=]\\s*([0-9]+(?:\\.[0-9]+)?)',
            re.IGNORECASE,
        )
        matches = num_pattern.findall(text)
        if matches:
            parsed: Dict[str, Any] = {}
            for key, val in matches:
                parsed[key.lower()] = float(val)
            return parsed

        raise ValueError(f"invalid judge output: {text[:200]}...")

    def _safe_float(val: Any) -> float:
        try:
            return float(val)
        except Exception:
            return 0.0

    try:
        data = _extract_json(content)
    except Exception as exc:
        return {"combined_score": 0.0, "error": str(exc)}

    style = _safe_float(data.get("style", 0.0))
    structure = _safe_float(data.get("structure", 0.0))
    content_score = _safe_float(data.get("content", 0.0))
    evidence = _safe_float(data.get("evidence", 0.0))
    facts_format = _safe_float(data.get("facts_format", 0.0))
    overall = _safe_float(data.get("overall", 0.0))

    combined = max(0.0, min(10.0, overall)) / 10.0

    return {
        "style": style,
        "structure": structure,
        "content": content_score,
        "evidence": evidence,
        "facts_format": facts_format,
        "overall": overall,
        "combined_score": combined,
    }
