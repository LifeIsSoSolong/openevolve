"""
新闻稿 Agent（标准化版）

功能：
- 提供 generate_press_release 作为统一入口，供 dspy_rl/press_gepa 等脚本调用。
- 支持命令行直接生成新闻稿，便于单测与联调。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List

# 允许脚本直接运行时找到项目根目录下的 src 包
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm_generate import llm_generate_thread_async


SYSTEM_ROLE = (
    "你是一名资深新闻撰稿人，擅长将采访资料整理为客观、结构化的中文新闻稿。"
    "写作时需确保事实准确、逻辑清晰，避免臆测和无依据的夸大。"
)

DEFAULT_MODEL_NAME = "gpt5"

# 初始 prompt 模板：优化器将在此基础上叠加/改写
BASE_PROMPT_TEMPLATE = """## 角色:
你是《江西日报》经济部副主任黄继妍，擅长“以小见大”进行新闻报道，即以政策为导向，将宏观政策与微观故事结合，通过单一企业/村庄的变革，折射全省产业升级脉络，同时引用权威数据量化成效，语言凝练生动，兼顾严肃性与可读性。
 
## 具体任务
请根据提供的采访资料和稿件类型，撰写一篇新闻报道。

## 稿件类型：
{interview_type}

## 采访资料：
{interview_context}

## 稿件要求:
1.内容真实，主要事实、数据必须基于用户提供的信息，符合新闻伦理，ai不得自行编造或推测未提供的关键信息；
2.表达自然，没有ai味儿，避免过于通用、缺乏细节的空话、套话，适合在社交平台特别是微信公众号上进行传播；
3.按照标准稿件格式输出内容，采用连贯叙事结构，将零散的内容点融入完整段落，避免使用视觉分隔符号；

## 工作流程：
1. 对用户提供的采访资料进行整理分析，提炼出重点、要点，能吸引读者阅读兴趣的点；
2. 根据提炼出的要点，生成新闻报道，报道需围绕大众密切关注的话题展开；
3. 确保内容准确、结构清晰、语言流畅，符合新闻写作规范，具体要求如下：
- 结构上：标题精练有力，可结合使用主副标题，突出核心主题；导语场景化，以具体案例或现象引入，快速吸引读者注意；主体围绕“问题-对策-成效”展开，环环相扣；结尾升华，展望未来。
- 内容支撑：深度剖析典型企业/地区案例，聚焦代表性主题，通过细节展现成效；同时精准引用关键数据，增强论证可信度；此外引用权威声音，强化政策解读深度。
- 语言风格：兼具专业性与生动性，将专业术语通俗化，辅以比喻降低理解门槛；细节描写增强感染力。
- 输出要短小精悍，整体不超过100字
"""


async def _call_llm(messages: List[Dict[str, str]], model_name: str = DEFAULT_MODEL_NAME, temperature: float = 0.3) -> str:
    """
    调用统一的异步 LLM 接口。

    Args:
        messages: OpenAI 风格的 message 列表。
        model_name: 选择的模型名称，需在 llm_generate.get_model() 中存在。
        temperature: 采样温度。
    """
    return await llm_generate_thread_async(text=messages, model_name=model_name, temperature=temperature)


def _run_sync(coro):
    """在独立事件循环中运行协程，避免复用已关闭的 loop。"""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def generate_press_release(
    prompt: str | None,
    interview_material: str,
    interview_type: str,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.3,
) -> str:
    """
    基于采访资料生成新闻稿。

    Args:
        prompt: 用户的高阶写作要求（如报道角度、受众、篇幅等）。
        interview_material: 采访整理出的原始资料文本。
        interview_type: 采访的类型（如“视频采访”、“文字采访”等）。
        model_name: 使用的模型（默认 gpt5，可切换到 llm_generate.get_model 中已配置的模型）。
        temperature: 采样温度，默认 0.3（更偏稳健）。

    Returns:
        str: 生成的完整新闻稿。
    """
    base_prompt = prompt or BASE_PROMPT_TEMPLATE
    messages = base_prompt.format(interview_type=interview_type, interview_context=interview_material)
    # 兼容 llm_generate_thread_async 的消息格式
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return _run_sync(_call_llm(messages, model_name=model_name, temperature=temperature))


def get_base_prompt() -> str:
    """提供给外部调用的基础模板获取函数。"""
    return BASE_PROMPT_TEMPLATE


def main():
    parser = argparse.ArgumentParser(description="基于采访资料生成新闻稿（命令行）")
    parser.add_argument("--interview", type=str, help="采访资料文本，或文件路径（.txt/.md）", required=False)
    parser.add_argument("--interview-file", type=str, help="采访资料文件路径，优先级高于 --interview", required=False)
    parser.add_argument("--type", type=str, default="消息", help="采访类型，如“消息”/“深度”")
    parser.add_argument("--prompt-file", type=str, help="自定义 prompt 模板文件路径（可选）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="模型名称，需在 llm_generate.get_model 中存在")
    parser.add_argument("--temperature", type=float, default=0.3, help="采样温度")
    args = parser.parse_args()

    # 读取采访资料
    if args.interview_file:
        path = Path(args.interview_file)
        if not path.exists():
            raise FileNotFoundError(f"采访资料文件不存在: {path}")
        interview_material = path.read_text(encoding="utf-8")
    elif args.interview:
        # 若传入路径字符串也兼容
        p = Path(args.interview)
        if p.exists():
            interview_material = p.read_text(encoding="utf-8")
        else:
            interview_material = args.interview
    else:
        raise ValueError("必须提供 --interview 或 --interview-file")

    # 读取自定义 prompt
    if args.prompt_file:
        pf = Path(args.prompt_file)
        if not pf.exists():
            raise FileNotFoundError(f"prompt 文件不存在: {pf}")
        prompt_tpl = pf.read_text(encoding="utf-8")
    else:
        prompt_tpl = BASE_PROMPT_TEMPLATE

    article = generate_press_release(
        prompt=prompt_tpl,
        interview_material=interview_material,
        interview_type=args.type,
        model_name=args.model,
        temperature=args.temperature,
    )
    print(article)


if __name__ == '__main__':
    main()
