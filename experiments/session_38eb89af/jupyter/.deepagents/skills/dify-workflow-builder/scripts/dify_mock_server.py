#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests
from pathlib import Path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787

# API paths (match fetch_dify_result.py and data/dify_draft_api.md)
IMPORT_PATH = "/api/v1/agents/tasks/workflow/import"
RUN_PATH = "/api/v1/agents/tasks/workflow/run"
RUNS_PATH_PREFIX = "/api/v1/agents/tasks/workflow/runs"
MODELS_PATH = "/api/v1/agents/models/llm"
DATASETS_PATH = "/api/v1/agents/datasets"

LLM_MODE = "auto"  # auto|llm|offline
DEFAULT_LLM_API_BASE = "https://newapi2.frontis.top/v1"
DEFAULT_LLM_API_KEY = "sk-fZSYQDKy7cdhkyMzHmYOVHjZJRFCH0LXPMr8v15i8IQ6ZYrl"
DEFAULT_LLM_MODEL = "gpt-5-mini"


def _now_iso() -> str:
    # Keep it simple; consistent with example format.
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _now_epoch() -> int:
    return int(time.time())


def _ms_id() -> str:
    return str(int(time.time() * 1000))


def _setup_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("dify_mock_server")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length else b""
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _write_json(handler: BaseHTTPRequestHandler, status_code: int, obj: Any) -> None:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _llm_enabled() -> bool:
    if LLM_MODE == "offline":
        return False
    if LLM_MODE == "llm":
        return True
    return bool(DEFAULT_LLM_API_BASE and DEFAULT_LLM_API_KEY and DEFAULT_LLM_MODEL)


def _llm_generate_result(*, workflow: Dict[str, Any], inputs: Dict[str, Any], logger: logging.Logger) -> str:
    """
    Optional: call an OpenAI-compatible /chat/completions endpoint.
    Falls back to offline generation if any error occurs.
    """
    api_base = DEFAULT_LLM_API_BASE
    api_key = DEFAULT_LLM_API_KEY
    model = DEFAULT_LLM_MODEL

    if not (api_base and api_key and model):
        raise RuntimeError("Missing mock LLM config (DEFAULT_LLM_API_BASE/KEY/MODEL)")

    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = (
        "你正在模拟 Dify Workflow 的运行输出。"
        "请根据给定的 workflow 定义与 inputs，生成一个合理的最终输出 result 文本。"
        "只输出最终结果文本，不要额外解释。\n\n"
        f"workflow.name={((workflow.get('app') or {}).get('name') or '')}\n"
        f"workflow.mode={((workflow.get('app') or {}).get('mode') or '')}\n"
        f"inputs={json.dumps(inputs, ensure_ascii=False)}\n"
    )

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a deterministic workflow runner simulator."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        obj = resp.json()
        content = (((obj.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        if not content:
            raise RuntimeError(f"Empty content from LLM response: {obj}")
        return content
    except Exception as exc:  # noqa: PERF203
        logger.warning(f"LLM generation failed; falling back to offline result. error={exc}")
        raise


def _offline_generate_result(*, workflow: Dict[str, Any], inputs: Dict[str, Any]) -> str:
    name = (workflow.get("app") or {}).get("name") or "workflow"
    lines = [f"[MOCK DIFY] {name} 运行完成", ""]
    for k in sorted(inputs.keys()):
        v = inputs[k]
        if isinstance(v, str) and len(v) > 300:
            v = v[:300] + "…"
        lines.append(f"- {k}: {v!r}")
    lines.append("")
    lines.append("（此为挡板服务输出；真实 Dify 恢复后可切回真实接口。）")
    return "\n".join(lines)


def _load_config_json(name: str) -> Dict[str, Any]:
    base_dir = Path(__file__).resolve().parents[1] / "config"
    path = base_dir / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _mock_models_response() -> Dict[str, Any]:
    cfg = _load_config_json("models.json")
    models = cfg.get("models") if isinstance(cfg, dict) else []
    providers: Dict[str, list] = {}
    for m in models or []:
        provider = m.get("provider")
        name = m.get("name")
        if not provider or not name:
            continue
        providers.setdefault(provider, []).append(
            {
                "model": name,
                "label": {"zh_Hans": name, "en_US": name},
                "model_properties": {"mode": m.get("mode") or "chat"},
            }
        )

    data = [{"provider": p, "models": lst} for p, lst in providers.items()]
    return {"status": 200, "message": "success", "data": {"data": data, "has_more": False}}


def _mock_datasets_response() -> Dict[str, Any]:
    cfg = _load_config_json("knowledge_bases.json")
    kbs = cfg.get("knowledge_bases") if isinstance(cfg, dict) else []
    data = []
    for kb in kbs or []:
        data.append(
            {
                "id": kb.get("id") or kb.get("dataset_id"),
                "name": kb.get("name") or "",
                "description": kb.get("description") or "",
                "provider": kb.get("provider") or "",
                "retrieval_model_dict": kb.get("retrieval_model_dict"),
                "external_retrieval_model": kb.get("external_retrieval_model"),
            }
        )
    return {"status": 200, "message": "success", "data": {"data": data, "has_more": False}}


@dataclass
class ImportedWorkflow:
    app_id: str
    mode: str
    workflow: Dict[str, Any]
    created_at: str = field(default_factory=_now_iso)


@dataclass
class RunRecord:
    record_id: str
    task_id: str
    app_id: str
    workflow_snapshot: Dict[str, Any]
    inputs: Dict[str, Any]
    run_status: str = "pending"  # pending/running/succeeded/failed
    error_message: Optional[str] = None
    response_payload: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)


class MockState:
    def __init__(self, logger: logging.Logger) -> None:
        self._lock = threading.Lock()
        self.logger = logger
        self.imported: Dict[str, ImportedWorkflow] = {}
        self.runs: Dict[str, RunRecord] = {}

    def import_workflow(self, payload: Dict[str, Any]) -> ImportedWorkflow:
        mode = str(payload.get("mode") or "workflow")
        yaml_content = payload.get("yaml_content") or ""

        try:
            workflow = json.loads(yaml_content) if isinstance(yaml_content, str) else {}
        except Exception:
            workflow = {}

        app_id = str((workflow.get("app") or {}).get("app_id") or uuid.uuid4())
        imported = ImportedWorkflow(app_id=app_id, mode=mode, workflow=workflow)
        with self._lock:
            self.imported[app_id] = imported
        return imported

    def create_run(self, *, task_id: str, app_id: str, inputs: Dict[str, Any]) -> RunRecord:
        with self._lock:
            imported = self.imported.get(app_id)
        if not imported:
            # Still allow run, but snapshot is empty to keep behavior tolerant.
            workflow = {}
        else:
            workflow = imported.workflow

        record_id = _ms_id()
        rec = RunRecord(
            record_id=record_id,
            task_id=task_id,
            app_id=app_id,
            workflow_snapshot=workflow,
            inputs=inputs,
            run_status="pending",
        )
        with self._lock:
            self.runs[record_id] = rec
        return rec

    def get_run(self, record_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self.runs.get(record_id)

    def _finalize_run(self, record_id: str) -> None:
        rec = self.get_run(record_id)
        if not rec:
            return
        # Mark running
        with self._lock:
            rec.run_status = "running"
            rec.updated_at = _now_iso()

        workflow = rec.workflow_snapshot or {}
        try:
            if _llm_enabled():
                result_text = _llm_generate_result(workflow=workflow, inputs=rec.inputs, logger=self.logger)
            else:
                result_text = _offline_generate_result(workflow=workflow, inputs=rec.inputs)
            response_payload = {
                "event": "workflow_finished",
                "task_id": str(uuid.uuid4()),
                "workflow_run_id": str(uuid.uuid4()),
                "data": {
                    "id": str(uuid.uuid4()),
                    "error": None,
                    "files": [],
                    "status": "succeeded",
                    "outputs": {"result": result_text},
                    "created_at": _now_epoch(),
                    "finished_at": _now_epoch(),
                    "total_steps": 1,
                    "workflow_id": str(uuid.uuid4()),
                    "elapsed_time": 0.5,
                    "total_tokens": 0,
                    "exceptions_count": 0,
                },
            }
            with self._lock:
                rec.run_status = "succeeded"
                rec.error_message = None
                rec.response_payload = response_payload
                rec.updated_at = _now_iso()
        except Exception as exc:  # noqa: PERF203
            with self._lock:
                rec.run_status = "failed"
                rec.error_message = str(exc)
                rec.response_payload = {
                    "event": "workflow_finished",
                    "data": {"status": "failed", "error": str(exc), "outputs": {}},
                }
                rec.updated_at = _now_iso()

    def schedule_finalize(self, record_id: str, delay_seconds: float = 1.0) -> None:
        t = threading.Timer(delay_seconds, self._finalize_run, args=(record_id,))
        t.daemon = True
        t.start()


def _match_path(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


class MockHandler(BaseHTTPRequestHandler):
    server_version = "MockDify/0.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Silence BaseHTTPRequestHandler default logging; use our logger instead.
        return

    @property
    def state(self) -> MockState:
        return getattr(self.server, "state")  # type: ignore[attr-defined]

    @property
    def logger(self) -> logging.Logger:
        return getattr(self.server, "logger")  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        self.logger.info(f"GET {path}")

        if path == MODELS_PATH:
            _write_json(self, 200, _mock_models_response())
            return

        if path == DATASETS_PATH:
            _write_json(self, 200, _mock_datasets_response())
            return

        if path == "/health":
            _write_json(self, 200, {"status": 200, "message": "ok"})
            return

        if _match_path(path, RUNS_PATH_PREFIX):
            parts = path.rstrip("/").split("/")
            record_id = parts[-1] if parts else ""
            rec = self.state.get_run(record_id)
            if not rec:
                _write_json(self, 404, {"status": 404, "message": "not found", "data": None})
                return
            data = {
                "id": rec.record_id,
                "task_id": rec.task_id,
                "dify_app_id": rec.app_id,
                "workflow_snapshot": rec.workflow_snapshot,
                "response_payload": rec.response_payload,
                "run_status": rec.run_status,
                "error_message": rec.error_message,
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
            }
            _write_json(self, 200, {"status": 200, "message": "success", "data": data})
            return

        _write_json(self, 404, {"status": 404, "message": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        self.logger.info(f"POST {path}")

        try:
            body = _read_json_body(self)
        except Exception as exc:
            _write_json(self, 400, {"status": 400, "message": f"invalid json: {exc}", "data": None})
            return

        if path == IMPORT_PATH:
            imported = self.state.import_workflow(body)
            self.logger.info(f"IMPORT app_id={imported.app_id} mode={imported.mode}")
            resp = {
                "status": 0,
                "message": "success",
                "data": {
                    "id": str(uuid.uuid4()),
                    "status": "success",
                    "app_id": imported.app_id,
                    "app_mode": imported.mode,
                    "current_dsl_version": "0.0.0",
                    "imported_dsl_version": "0.0.0",
                    "error": None,
                },
            }
            _write_json(self, 200, resp)
            return

        if path == RUN_PATH:
            task_id = ""
            app_id = ""
            if "task_id" in query:
                task_id = (query.get("task_id") or [""])[0]
            if "app_id" in query:
                app_id = (query.get("app_id") or [""])[0]

            if isinstance(body.get("task_id"), str) and not task_id:
                task_id = body.get("task_id") or ""
            if isinstance(body.get("app_id"), str) and not app_id:
                app_id = body.get("app_id") or ""

            inputs = body.get("inputs") if isinstance(body.get("inputs"), dict) else {}
            if not app_id:
                # For tolerance: if missing, generate one.
                app_id = str(uuid.uuid4())
            if not task_id:
                task_id = _ms_id()

            rec = self.state.create_run(task_id=task_id, app_id=app_id, inputs=inputs)
            self.logger.info(f"RUN record_id={rec.record_id} app_id={app_id} inputs_keys={list(inputs.keys())}")
            self.state.schedule_finalize(rec.record_id, delay_seconds=1.0)
            _write_json(self, 200, {"status": 200, "message": "success", "data": {"workflow_run_record_id": rec.record_id}})
            return

        _write_json(self, 404, {"status": 404, "message": "not found"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock Evo/Dify draft workflow API server.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--llm-mode", choices=["auto", "llm", "offline"], default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    global LLM_MODE
    LLM_MODE = args.llm_mode

    logger = _setup_logging(args.verbose)
    server = ThreadingHTTPServer((args.host, args.port), MockHandler)
    server.logger = logger  # type: ignore[attr-defined]
    server.state = MockState(logger)  # type: ignore[attr-defined]

    logger.info(f"Listening on http://{args.host}:{args.port}")
    logger.info("Endpoints:")
    logger.info(f"  POST {IMPORT_PATH}")
    logger.info(f"  POST {RUN_PATH}")
    logger.info(f"  GET  {RUNS_PATH_PREFIX}" + "/{record_id}")
    logger.info(f"LLM mode: {LLM_MODE} ({'enabled' if _llm_enabled() else 'disabled'})")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
