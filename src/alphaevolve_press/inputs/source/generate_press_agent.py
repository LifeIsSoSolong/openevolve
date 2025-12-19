"""
generate_press_agent: call OpenAI to produce a press draft.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


def generate_press_agent(
    model_name: str,
    prompt_generate_press_final: str,
    interview_context: str,
    interview_type: str,
    api_base: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.1,
    **_: Any,
) -> str:
    """
    Generate a press draft using OpenAI chat completion.

    Args:
        model_name: e.g., "gpt-5.1"
        prompt_generate_press_final: formatted prompt with placeholders filled
        interview_context: source interview text
        interview_type: category/type for context
        api_base/api_key: optional overrides; default to env OPENAI_API_BASE / OPENAI_API_KEY
    """
    client = OpenAI(
        base_url=api_base or os.getenv("OPENAI_API_BASE"),
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )

    system_message = (
        "你是一名资深新闻撰稿人，擅长将采访资料整理为客观、结构化的中文新闻稿。"
        "写作时需确保事实准确、逻辑清晰，避免臆测和无依据的夸大。"
    )
    user_message = (
        f"{prompt_generate_press_final}\n\n"
        f"【采访类型】{interview_type}\n"
        f"【采访稿】{interview_context}\n"
        "请直接输出完整新闻稿。"
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
