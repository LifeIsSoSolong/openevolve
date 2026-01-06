"""大观训练任务的便捷工具函数。

特性：
- 统一读取环境变量：
  - EVO_API_BASE（默认 https://evo.frontis.top/api/v1/agents）
  - EVO_TOKEN（必填，作为 Bearer Token）
- 封装常用接口：
  - list_models() / list_algorithms()
  - upload_inputs() 分片上传压缩包/文件
  - create_task() 提交训练任务
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests

CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from agent_api_file_uploader import AgentApiFileUploader, UploadResult  # noqa: E402

DEFAULT_API_BASE = "https://evo.frontis.top/api/v1/agents"
DEFAULT_META_BASE = "http://10.200.4.4:30022"


def _get_base_url(base_url: Optional[str] = None) -> str:
    """获取主要任务/上传 API 基础地址并裁剪尾部斜线。"""
    base = (base_url or os.getenv("EVO_API_BASE") or DEFAULT_API_BASE).rstrip("/")
    if not base:
        msg = "API 基础地址不能为空，请设置 EVO_API_BASE"
        raise ValueError(msg)
    return base


def _get_token(token: Optional[str] = None) -> str:
    """获取 Token（优先参数，其次环境变量 EVO_TOKEN）。"""
    resolved = (token or os.getenv("EVO_TOKEN") or "").strip()
    if not resolved:
        msg = "缺少 Token，请设置环境变量 EVO_TOKEN"
        raise ValueError(msg)
    return resolved


def _get_meta_base_url(base_url: Optional[str] = None) -> str:
    """获取元数据（模型/算法列表）基础地址，默认为内网网关。"""
    base = (base_url or os.getenv("EVO_META_BASE") or DEFAULT_META_BASE).rstrip("/")
    if not base:
        msg = "元数据地址不能为空，请设置 EVO_META_BASE"
        raise ValueError(msg)
    return base


def _headers(token: Optional[str] = None) -> Dict[str, str]:
    """构造通用请求头。"""
    return {
        "Authorization": f"Bearer {_get_token(token)}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def list_models(*, base_url: Optional[str] = None, token: Optional[str] = None) -> Dict[str, Any]:
    """获取模型列表：GET /meta/models。"""
    base = _get_meta_base_url(base_url)
    resp = requests.get(f"{base}/meta/models", timeout=15)
    resp.raise_for_status()
    return resp.json()


def list_algorithms(
    *, base_url: Optional[str] = None, token: Optional[str] = None
) -> Dict[str, Any]:
    """获取算法列表：GET /meta/algorithms。"""
    base = _get_meta_base_url(base_url)
    resp = requests.get(f"{base}/meta/algorithms", timeout=15)
    resp.raise_for_status()
    return resp.json()


def upload_inputs(
    archive_path: str,
    *,
    task_id: Optional[str] = None,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    chunk_size: int = 5 * 1024 * 1024,
) -> UploadResult:
    """分片上传数据/代码压缩包，返回 UploadResult（含 task_id）。"""
    uploader = AgentApiFileUploader(
        base_url=_get_base_url(base_url),
        token=_get_token(token),
        chunk_size=chunk_size,
    )
    try:
        return uploader.upload_file(archive_path, task_id=task_id, print_task_id=True)
    finally:
        uploader.close()


def create_task(
    payload: Dict[str, Any],
    *,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """提交任务：POST /tasks，返回 JSON。

    Args:
        payload: 已构造好的任务配置字典
        base_url: API 基础地址
        token: Bearer Token
        timeout: 请求超时（秒）
    """
    base = _get_base_url(base_url)
    resp = requests.post(f"{base}/tasks", json=payload, headers=_headers(token), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


__all__ = [
    "list_models",
    "list_algorithms",
    "upload_inputs",
    "create_task"
]
