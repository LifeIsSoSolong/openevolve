"""基于智能体 API 的分片上传/合并工具（默认读取环境变量）。

默认配置
- 基础 URL：EVO_API_BASE（默认 https://evo.frontis.top/api/v1/agents）
- Token：EVO_TOKEN（必填）

核心能力
- `get_task_id()`：GET tasks/id
- `upload_file()`：分片上传 + 合并，返回 UploadResult（含 task_id）

示例（读取环境变量）：
    u = AgentApiFileUploader.from_env()
    task_id = u.get_task_id()
    u.upload_file("inputs.zip", task_id=task_id, print_task_id=True)
    u.close()
"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

DEFAULT_BASE_URL = os.getenv("EVO_API_BASE", "https://evo.frontis.top/api/v1/agents")
DEFAULT_TOKEN = os.getenv("EVO_TOKEN")

# ---------- 响应解析模块开始 ----------
class AgentApiUploaderError(RuntimeError):
    """上传工具错误。"""


def _extract_success_data(resp: httpx.Response) -> Dict[str, Any]:
    """解析项目统一响应格式：{status,message,data}，并返回 data。"""
    try:
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise AgentApiUploaderError(f"响应解析失败：HTTP {resp.status_code}") from exc

    if resp.status_code != 200 or not isinstance(payload, dict):
        raise AgentApiUploaderError(f"请求失败：HTTP {resp.status_code}")

    status = payload.get("status")
    if status != 200:
        message = payload.get("message") or "请求失败"
        raise AgentApiUploaderError(str(message))

    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def _guess_content_type(filename: str) -> str:
    """尽量不引入额外依赖，这里只做最保守的类型。"""
    _ = filename
    return "application/octet-stream"

# ---------- 响应解析模块结束 ----------


# ---------- 上传配置模块开始 ----------
@dataclass(frozen=True)
class UploadResult:
    """上传结果（含 task_id、file_id 及 merge 返回 data）。"""

    task_id: str
    file_id: str
    filename: str
    chunk_total: int
    merge_data: Dict[str, Any]
# ---------- 上传配置模块结束 ----------


# ---------- 上传工具类模块开始 ----------
class AgentApiFileUploader:
    """智能体 API 文件上传工具类。"""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 60.0,
        chunk_size: int = 5 * 1024 * 1024,
    ) -> None:
        self._base_url = (base_url or DEFAULT_BASE_URL or "").rstrip("/")
        if not self._base_url:
            raise ValueError("base_url 不能为空（可通过 EVO_API_BASE 环境变量传入）")

        # Token 默认来自环境变量
        self._token = (token or DEFAULT_TOKEN or "").strip() or None
        if not self._token:
            raise ValueError("Token 不能为空，请设置 EVO_TOKEN 环境变量")

        self._chunk_size = max(1, int(chunk_size))
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout, follow_redirects=True)

    def close(self) -> None:
        """关闭底层 HTTP 连接池。"""
        self._client.close()

    def __enter__(self) -> "AgentApiFileUploader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    @classmethod
    def from_env(cls) -> "AgentApiFileUploader":
        """使用环境变量创建实例。"""
        return cls(base_url=DEFAULT_BASE_URL, token=DEFAULT_TOKEN)

    # ---------- 任务 ID 获取模块开始 ----------
    def get_task_id(self) -> str:
        """获取新的 task_id（GET /api/v1/agents/tasks/id）。"""
        resp = self._client.get("tasks/id", headers=self._headers())
        data = _extract_success_data(resp)
        task_id = str(data.get("task_id") or "").strip()
        if not task_id:
            raise AgentApiUploaderError("获取 task_id 失败：响应缺少 task_id")
        return task_id

    def generate_task_id(self) -> str:
        """get_task_id 的别名，便于外部服务语义化调用。"""
        return self.get_task_id()
    # ---------- 任务 ID 获取模块结束 ----------

    # ---------- 文件上传模块开始 ----------
    def upload_file(
        self,
        file_path: str,
        *,
        task_id: Optional[str] = None,
        filename: Optional[str] = None,
        print_task_id: bool = False,
    ) -> UploadResult:
        """上传本地文件并合并。

        参数：
        - file_path：本地文件路径
        - task_id：可传入指定 task_id；不传则自动调用 get_task_id() 获取
        - filename：合并后的文件名，不传则取本地文件名
        - print_task_id：上传完成后是否打印 task_id

        返回：
        - UploadResult（包含 task_id）
        """
        normalized_path = os.path.abspath(os.path.expanduser(str(file_path)))
        if not os.path.exists(normalized_path) or not os.path.isfile(normalized_path):
            raise AgentApiUploaderError(f"文件不存在：{normalized_path}")

        target_task_id = (task_id or "").strip() or self.get_task_id()
        target_filename = (filename or "").strip() or os.path.basename(normalized_path)
        if not target_filename:
            raise AgentApiUploaderError("filename 不能为空")

        file_size = os.path.getsize(normalized_path)
        chunk_total = max(1, int(math.ceil(file_size / self._chunk_size)) if file_size else 1)
        file_id = str(uuid.uuid4())
        content_type = _guess_content_type(target_filename)

        with open(normalized_path, "rb") as fp:
            for chunk_index in range(chunk_total):
                chunk_bytes = fp.read(self._chunk_size) if file_size else b""

                # 与前端一致：multipart/form-data 字段名必须对应后端 Form 参数名
                data = {
                    "file_id": file_id,
                    "chunk_index": str(chunk_index),
                    "chunk_total": str(chunk_total),
                    "task_id": target_task_id,
                }
                files = {
                    "chunk_file": (target_filename, chunk_bytes, content_type),
                }
                resp = self._client.post(
                    "tasks/files/chunk",
                    data=data,
                    files=files,
                    headers=self._headers(),
                )
                _ = _extract_success_data(resp)

        merge_payload = {
            "file_id": file_id,
            "filename": target_filename,
            "chunk_total": chunk_total,
            "task_id": target_task_id,
        }
        merge_resp = self._client.post(
            "tasks/files/merge",
            json=merge_payload,
            headers=self._headers(),
        )
        merge_data = _extract_success_data(merge_resp)
        result = UploadResult(
            task_id=target_task_id,
            file_id=file_id,
            filename=target_filename,
            chunk_total=chunk_total,
            merge_data=merge_data,
        )
        if print_task_id:
            print(result.task_id)
        return result

    def upload_file_return_task_id(
        self,
        file_path: str,
        *,
        task_id: Optional[str] = None,
        filename: Optional[str] = None,
        print_task_id: bool = True,
    ) -> str:
        """上传文件并返回 task_id（常用于只关心 task_id 的场景）。"""
        result = self.upload_file(
            file_path,
            task_id=task_id,
            filename=filename,
            print_task_id=print_task_id,
        )
        return result.task_id
    # ---------- 文件上传模块结束 ----------

# ---------- 上传工具类模块结束 ----------


__all__ = ["AgentApiFileUploader", "UploadResult", "AgentApiUploaderError"]
