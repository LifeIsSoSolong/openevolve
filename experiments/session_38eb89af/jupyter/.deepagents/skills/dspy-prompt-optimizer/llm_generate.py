# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 16:50
# @Author  : likaixiang
# @FileName: llm_generate.py
# @Software: frontis
import asyncio
import json
import os
import time
import traceback

import httpx
from loguru import logger
from openai import OpenAI, AsyncOpenAI
import requests


def get_model():
    llm_model_config = {}

    MODEL_MAP = {

        "qwen2.5-72b":{
                        "model_name": "Qwen2.5-72b-dev-server",
                        "stop_tokens": ["<|endoftext|>", "<|im_end|>", "<|im_start|>"],
                        "api":  "http://10.200.4.8:30620/v1/chat/completions",
                        "api_key":"frontis_7db74fcab3c64798",
                    },
        "gpt5": {
            "model_name": "gpt-5.1",
            "api": "https://newapi2.frontis.top/v1",
            "api_key": "sk-fZSYQDKy7cdhkyMzHmYOVHjZJRFCH0LXPMr8v15i8IQ6ZYrl",
        },
        "qwen3-220b": {
            "model_name": "Qwen3-235B-A22B",
            "api": "http://10.200.4.8:30122/v1/chat/completions",
            "api_key": "frontis_cb02e00fa53440d8",
        },
        "deepseek": {
            "api_key": "sk-353a88a777bd4c598f17b2923677e100",
            "model_name": "deepseek-chat",  # deepseek-reasoner
            "api": "https://api.deepseek.com/v1",

        },
        "deepseek-reasoner": {
            "api_key": "sk-353a88a777bd4c598f17b2923677e100",
            "model_name": "deepseek-reasoner",  # deepseek-reasoner
            "api": "https://api.deepseek.com/v1",

        },
        "qwen-max":{
            "model_name":"qwen-max",
            "stop_tokens":["<|endoftext|>", "<|im_end|>", "<|im_start|>"],
            "api":"https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            "api_call_weights":[1.0],
            "api_key":"sk-928bbd1501ca44e7ada962ff09d51d68",
            "timeout":600,
            "retry_times":3
        },
        "kimi": {
            "model_name": "kimi-k2-turbo-preview",
            "api": "https://api.moonshot.cn/v1",
            "api_key": "sk-fFFLxJCzM9bkVgZ5uYLOlZdjC4d80tllQEgNM4Ins96q4izB",

        },
        "gpt-4o": {
            "model_name": "gpt-4o",
            "api": "https://api3.apifans.com/v1",
            "api_key": "sk-5d2ZcXHg0SSTEB8V6005Dd8f74E14f1e95Ae3c0e205b3f2a",
            "retry_times": 3
        },
        "claude": {
            "model_name": "claude-sonnet-4-20250514",
            "api": "https://api3.apifans.com/v1",
            "api_key": "sk-5d2ZcXHg0SSTEB8V6005Dd8f74E14f1e95Ae3c0e205b3f2a",
        },
        "claude-sonnet": {
            "model_name": "claude-sonnet-4-5-20250929",
            "api": "https://newapi2.frontis.top/v1",
            "api_key": "sk-fZSYQDKy7cdhkyMzHmYOVHjZJRFCH0LXPMr8v15i8IQ6ZYrl",
        },

    }
    return MODEL_MAP

def get_text_emb(text):
    """
    获取文本的emb向量
    """
    api_url = "http://10.200.4.8:30292/api/v1/embeddings"
    emb_model = "bge-m3"  # 1024 维度
    params = {"input": text, "model": emb_model}

    try:
        resp = requests.post(api_url, json=params, timeout=30)
        resp.raise_for_status()
        try:
            result = resp.json()
        except ValueError:
            logger.error(f"call emb failed, response is not JSON. status={resp.status_code}, text={resp.text[:500]}")
            raise RuntimeError(f"call emb failed: non-JSON response, status={resp.status_code}")

        if "data" not in result or not result["data"]:
            logger.error(f"call emb failed, missing data field. resp={result}")
            raise RuntimeError("call emb failed: missing embedding data")

        embedding = result["data"][0].get("embedding")
        if embedding is None:
            logger.error(f"call emb failed, embedding missing. resp={result}")
            raise RuntimeError("call emb failed: embedding missing")

        return embedding
    except Exception:
        error_msg = traceback.format_exc()
        logger.error(f"call emb failed! {error_msg}")
        raise RuntimeError(f"call emb failed! {error_msg}")
    

async def llm_openai_style_async(text,model_name,temperature=0.3,top_p=0.7,response_format={"type": "text"}):
    MODEL_MAP = get_model()
    if not isinstance(text, list):
        message = [{"role": "user", "content": f"{text}"}]
    else:
        message = text
    async with AsyncOpenAI(
        api_key=MODEL_MAP[model_name]["api_key"],
        organization=None,
        base_url=MODEL_MAP[model_name]["api"].replace("/chat/completions",""),
        timeout=200,
        max_retries=1,
    ) as client:
        chat_completion = await client.chat.completions.create(
            messages=message,
            model=MODEL_MAP[model_name]["model_name"],
            temperature=temperature,
            # stop=MODEL_MAP[model_name]["stop_tokens"],
            stream=False,
        )

        # text_res = chat_completion.choices[0].message.reasoning_content
        text_res = chat_completion.choices[0].message.content
        return text_res

def retry_on_failure(retries=3):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            model_name = kwargs.pop('model_name',"")
            model_names = [model_name,"kimi","claude"]  # 待重试的model，可能是自己网络波动，也可能是对方服务
            for attempt in range(retries):
                try:
                    # 使用每次传入的不同 model_name
                    if model_names[attempt]:
                        kwargs['model_name'] = model_names[attempt]
                    res = await func(*args, **kwargs)
                    if res:
                        return res
                    else:
                        raise  # 如果是最后一次重试，抛出异常
                except Exception as e:
                    logger.info(f"Attempt {attempt + 1} failed with model {model_names[attempt]}: {e}")
                    if attempt == retries - 1:
                        raise  # 如果是最后一次重试，抛出异常
        return wrapper
    return decorator

@retry_on_failure(retries=3)
async def llm_generate_thread_async(text: str, model_name="qwen2.5-72b", temperature=0.5, top_p=0.7, response_format={"type": "text"}):
    start_time = time.time()
    result = await llm_openai_style_async(text=text, model_name=model_name, temperature=temperature, top_p=top_p, response_format=response_format)
    logger.info(f"model {model_name} latency：{time.time()-start_time}")
    # logger.info(f"model {model_name} output：{result}")
    return result

if __name__ == '__main__':
    prompt = """# 角色
你是一名专业的建筑施工计划工程师，擅长从文本描述中精确提取施工区段和任务，并构建它们之间的逻辑关系。

# 任务
根据提供的“建筑施工信息摘要”和“分区分段及流水描述文本”，执行以下操作：
1.  **识别施工任务**：从“分区分段及流水描述文本”中，精确识别出所有独立的可作为“当前任务”和“前序任务”的施工分段或作业单元。任务命名应简洁明了，能唯一标识该区段/作业。
2.  **确定逻辑关系**：分析文本中关于施工顺序、依赖条件（如“...完成后方可开始...”，“...开始后即可进行...”，“...需同时完成...”等）的描述，为识别出的任务之间建立逻辑关系。
3.  **分配关系类型**：根据依赖描述，将关系类型指定为FS（完成-开始）、SS（开始-开始）或FF（完成-完成）。
4.  **设置时间间隔**：间隔的天数。
5.  **输出格式**：将结果严格组织成JSON数组格式，每个元素包含“前序任务”、“当前任务”、“关系类型”和“时间间隔”字段。确保输出内容仅为JSON，不包含任何解释性文字或标记。

## 分区分段及流水描述文本：
机电工程共划分为地下室、地上两个施工区域，其各施工区根据作业特点水平、竖向、机房进行平行独立施工，水平施工区按层划分施工段，自下而上流水施工；竖向施工区每三层一个施工段，自下而上流水施工；地下机房施工区内按类按数量流水施工。

# 输出要求
严格按照以下JSON格式输出，且只输出JSON内容：
[{"前序任务":"","当前任务":"","关系类型":"","时间间隔":0}]"""
    # asyncio.run(llm_generate_thread_async("你是谁",model_name="r1-32b"))
    # asyncio.run(llm_generate_thread_async(text="你是谁",model_name="gpt5"))
    asyncio.run(llm_generate_thread_async(text="你是谁",model_name="claude-sonnet"))
    # print(llm_generate_thread(text="你是谁"))
    # print(llm_generate_thread(text="你叫什么名字"))
