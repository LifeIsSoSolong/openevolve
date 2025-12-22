"""
generate_press_agent: call OpenAI to produce a press draft.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


def generate_press_agent(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    api_base: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    **_: Any,
) -> str:
    """
    Generate a press draft using OpenAI chat completion.

    Args:
        model_name: e.g., "gpt-5.1"
        system_prompt: system prompt for the chat completion
        user_prompt: user prompt for the chat completion
        api_base/api_key: optional overrides; default to env OPENAI_API_BASE / OPENAI_API_KEY
    """
    client = OpenAI(
        base_url=api_base or os.getenv("OPENAI_API_BASE"),
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
