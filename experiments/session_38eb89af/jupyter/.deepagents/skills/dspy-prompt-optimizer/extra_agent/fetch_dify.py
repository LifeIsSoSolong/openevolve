"""
工具：读取/更新 Dify 工作流 JSON，并可调用 Dify 接口执行推理。

核心功能：
- extract_prompt_template: 从 Dify workflow JSON 中提取 LLM 节点的 prompt_template（列表形式或拼接文本）。
- update_prompt_template: 将优化后的 prompt 写回 workflow JSON（默认替换第一个 system 文本）。
- call_dify_workflow: 按文档示例调用 Dify agent workflow 接口。

默认输入样例：data/temp/dify_input.json
默认接口文档：data/temp/fetch_dify_doc.md（POST https://.../agents/tasks/workflow）
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Any
from loguru import logger
import threading
import requests
# 线程锁
_metric_lock = threading.Lock()

DEFAULT_INPUT = Path("data/temp/dify_input.json")
DEFAULT_DIFY_BASE_URL = "http://10.200.2.52"


def _get_dify_base_url(explicit: str | None = None) -> str:
    base = (explicit or os.getenv("DIFY_BASE_URL") or DEFAULT_DIFY_BASE_URL).rstrip("/")
    if base.endswith("/api/v1/agents/tasks/workflow"):
        return base
    return f"{base}/api/v1/agents/tasks/workflow"


def load_workflow(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_prompt_template(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    从 workflow JSON 中提取第一个 LLM 节点的 prompt_template。
    返回原始 message 列表（包含 role/text）。
    """
    nodes = data.get("graph", {}).get("nodes", [])
    for node in nodes:
        nd = node.get("data", {})
        if nd.get("type") == "llm" and "prompt_template" in nd:
            return nd.get("prompt_template", [])
    raise ValueError("未找到 LLM 节点或 prompt_template")


def extract_system_prompt_text(data: Dict[str, Any]) -> str:
    """
    提取第一个 LLM 节点中 role=system/user 的文本，按顺序拼接（部分 workflow 将提示拆成 system 与 user）。
    """
    tmpl = extract_prompt_template(data)
    parts: List[str] = []
    for msg in tmpl:
        # if msg.get("role") in {"system", "user"}:
        if msg.get("role") in {"system"}:
            text = msg.get("text", "")
            if text:
                parts.append(text)
    if parts:
        return "\n".join(parts)
    return tmpl[0].get("text", "") if tmpl else ""


def prompt_as_text(prompt_template: List[Dict[str, str]]) -> str:
    """将 prompt_template 列表转成单一文本，便于优化。"""
    parts = []
    for msg in prompt_template:
        role = msg.get("role", "")
        text = msg.get("text", "")
        if role:
            parts.append(f"[{role.upper()}]")
        parts.append(text)
    return "\n".join(parts)


def update_prompt_template(data: Dict[str, Any], new_prompt: str) -> Dict[str, Any]:
    """
    用 new_prompt 替换第一个 LLM 节点的 prompt_template 中的第一个 message 文本（通常是 system）。
    返回修改后的数据（原地修改并返回引用）。
    """
    nodes = data.get("graph", {}).get("nodes", [])
    for node in nodes:
        nd = node.get("data", {})
        if nd.get("type") == "llm" and "prompt_template" in nd:
            pt = nd.get("prompt_template", [])
            if not pt:
                pt.append({"id": "updated", "role": "system", "text": new_prompt})
            else:
                # 只替换第一条文本
                pt[0]["text"] = new_prompt
            nd["prompt_template"] = pt
            return data
    raise ValueError("未找到 LLM 节点或 prompt_template 以更新")


def call_dify_workflow(
    payload: Dict[str, Any],
    base_url: str | None = None,
) -> Dict[str, Any]:
    """
    调用 Dify workflow 接口（新版：先上传任务，再轮询结果）。
    返回 {code, msg, output}，code 非 200 表示失败。
    """
    base_url = _get_dify_base_url(base_url)
    url = base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    post_resp = requests.post(url, headers=headers, json=payload, timeout=120)
    try:
        post_resp.raise_for_status()
    except Exception as e:
        try:
            err_obj = post_resp.json()
        except Exception:
            err_obj = {"text": post_resp.text}
        return {"code": post_resp.status_code, "msg": f"调用失败: {e}", "detail": err_obj, "output": ""}

    try:
        post_obj = post_resp.json()
    except Exception:
        post_obj = None

    if not isinstance(post_obj, dict):
        return {"code": 500, "msg": f"返回结果不是有效的 JSON 对象: {post_obj}", "output": ""}
    try:
        run_id = post_obj.get("data", {}).get("workflow_run_record_id")
        logger.info(f"获得草稿workflow_run_record_id：{run_id}")
    except Exception:
        logger.error(f"返回结果缺少 workflow_run_record_id，无法轮询.detail: {post_obj}")
        return {"code": 500, "msg": "返回结果缺少 workflow_run_record_id，无法轮询", "detail": post_obj, "output": ""}
        

    # 轮询接口写死为官方 GET 结果地址，直接填入 run_id
    run_url = f"{base_url}/runs/{run_id}"
    poll_interval = 3.0
    max_polls = 40
    last_obj = None
    get_resp = None
    for _ in range(max_polls):
        try:
            get_resp = requests.get(run_url, headers=headers, timeout=120)
            get_resp.raise_for_status()
            last_obj = get_resp.json()
            data = last_obj.get("data", {}) if isinstance(last_obj, dict) else {}
            resp_payload = data.get("response_payload") or {}
            run_status = data.get("run_status") or resp_payload.get("data", {}).get("status")
            outputs = (
                resp_payload.get("data", {}).get("outputs")
                or resp_payload.get("outputs")
                or data.get("outputs")
            )
        except Exception as e:  # noqa: PERF203
            continue
       
        result_text = None
        if isinstance(outputs, dict):
            result_text = outputs.get("result") or outputs.get("text")

        if result_text is not None:
            return {"code": 200, "msg": "ok", "output": result_text}

        time.sleep(poll_interval)
    logger.info(f"轮询失败:  detail: {get_resp}")
    return {"code": 408, "msg": "轮询超时，未获取到结果", "output": ""}


def update_prompt_and_run(
    workflow_path: Path,
    new_prompt: str,
    base_url: str | None = None,
) -> Dict[str, Any]:
    """
    便利函数：读取 workflow -> 替换 prompt -> 调用 Dify（不落盘）。

    Args:
        workflow_path: 输入 workflow JSON 路径
        new_prompt: 新的 prompt 文本
        base_url: Dify base url

    Returns:
        dict: Dify 返回结果（或 status/text）
    """
    payload = load_workflow(workflow_path)
    payload = update_prompt_template(payload, new_prompt)
    return call_dify_workflow(payload, base_url=base_url)

with _metric_lock:
    def run_dify_with_prompt(
        workflow_path: Path,
        prompt_text: str,
        input_vars: Dict[str, Any],
        base_url: str | None = None,
    ) -> Dict[str, Any]:
        """
        用新的 prompt 文本与动态用户输入调用 Dify：
        - 替换 system prompt 为 prompt_text
        - 构造 user prompt，将输入变量按段落拼接
        - 调用 Dify workflow 并返回结果（含 result）
        """
        payload = load_workflow(workflow_path)
        payload = update_prompt_template(payload, prompt_text)

        user_text_parts = []
        for k, v in input_vars.items():
            user_text_parts.append(f"## {k}：\n{v}")
        user_text = "\n\n".join(user_text_parts)

        tmpl = extract_prompt_template(payload)
        for msg in tmpl:
            if msg.get("role") == "user":
                msg["text"] = user_text
                break
        nodes = payload.get("graph", {}).get("nodes", [])
        for node in nodes:
            nd = node.get("data", {})
            if nd.get("type") == "llm" and "prompt_template" in nd:
                nd["prompt_template"] = tmpl
                break

        return call_dify_workflow(payload, base_url=base_url)


def main():
    parser = argparse.ArgumentParser(description="Dify workflow prompt 提取/更新/调用工具")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Dify workflow JSON 路径")
    parser.add_argument("--export-prompt", action="store_true", help="仅打印提取到的 prompt 文本")
    parser.add_argument("--new-prompt-file", type=str, help="替换的 prompt 文本文件路径")
    parser.add_argument("--call", action="store_true", help="调用 Dify workflow 接口")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    payload = load_workflow(Path(args.input))

    if args.export_prompt:
        tmpl = extract_prompt_template(payload)
        print(prompt_as_text(tmpl))
        return

    if args.new_prompt_file:
        new_prompt = Path(args.new_prompt_file).read_text(encoding="utf-8")
        payload = update_prompt_template(payload, new_prompt)

    if args.call:
        result = call_dify_workflow(payload, base_url=args.base_url)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
