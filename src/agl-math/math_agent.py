# Copyright (c) Microsoft. All rights reserved.

"""数学问题求解智能体，使用 Agent-lightning 框架进行训练。

这是一个单轮智能体，接收数学问题并生成答案。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, cast

from eval import accuracy_reward

import agentlightning as agl

agl.configure_logger()
logger = logging.getLogger(__name__)


class LitMathAgent(agl.LitAgent[Dict[str, Any]]):
    """数学问题求解智能体。

    这是一个单轮智能体，接收数学问题（problem）并生成答案。
    使用 accuracy_reward 函数评估答案的正确性。
    """

    def __init__(
        self,
        trained_agents: Optional[str] = None,
        val_temperature: Optional[float] = None,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        """执行单轮数学问题求解。

        Args:
            task: 包含 'problem' (输入问题) 和 'answer' (正确答案) 的字典
            resources: 资源字典，包含 'main_llm' (LLM 资源)
            rollout: Rollout 元数据

        Returns:
            奖励分数 (0.0 或 1.0)，如果出错则返回 None
        """
        problem = task.get("problem", "")
        ground_truth = task.get("answer", "")

        if not problem or not ground_truth:
            logger.error(f"[Rollout {rollout.rollout_id}] Missing problem or answer in task")
            return None

        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        # 构建提示词
        prompt = f"Solve the following math problem. Provide your answer in LaTeX format, and box your final answer using \\boxed{{}}.\n\nProblem: {problem}\n\nSolution:"

        rollout_id = rollout.rollout_id
        logger.info(f"[Rollout {rollout_id}] Problem: {problem[:100]}...")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")

        try:
            # 调用 LLM 生成答案
            # 注意：这里使用的是 VERL 的 LLM proxy，它会路由到正在训练的模型
            # llm.get_base_url() 返回的 endpoint 指向 VERL 的 proxy，该 proxy 提供 OpenAI 兼容的 API
            # 但实际调用的是配置中指定的训练模型（通过 vLLM 服务）
            from langchain.chat_models import init_chat_model

            temperature = (
                self.val_temperature
                if self.val_temperature is not None and rollout.mode == "validation"
                else llm.sampling_parameters.get("temperature", 0.7)
            )

            # 获取 VERL proxy 的 endpoint，它会路由到正在训练的模型
            endpoint = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
            # 使用 OpenAI 兼容的 API 格式（因为 VERL proxy 提供 OpenAI 兼容接口）
            chat_model = init_chat_model(
                llm.model,  # 逻辑模型名称，VERL proxy 会路由到实际的训练模型
                model_provider="openai",  # OpenAI 兼容的 API 格式
                openai_api_base=endpoint,  # VERL proxy 的 endpoint
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=temperature,
                max_retries=0,
                max_tokens=2048,
            )

            from langchain_core.messages import HumanMessage

            response = chat_model.invoke([HumanMessage(content=prompt)])
            generated_answer = response.content if hasattr(response, "content") else str(response)

            logger.info(f"[Rollout {rollout_id}] Generated Answer: {generated_answer[:200]}...")

            # 使用 accuracy_reward 评估答案
            reward = accuracy_reward(generated_answer, ground_truth)
            logger.info(f"[Rollout {rollout_id}] Reward: {reward}")

            return reward

        except Exception as e:
            logger.exception(f"[Rollout {rollout_id}] Error during rollout: {e}")
            return None

