#!/usr/bin/env python3
"""
Fetch Dify workflow result via Evo Dify Draft APIs.

Implements an end-to-end flow:
1) Upload/import the workflow DSL (from a local workflow JSON file content).
2) Run the imported draft workflow with dynamic inputs (parsed from start node vars).
3) Poll run record until succeeded/failed (max 5 minutes; every 5 seconds).

API reference: data/dify_draft_api.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests


DEFAULT_BASE_URL = "http://10.200.2.52"
IMPORT_PATH = "/api/v1/agents/tasks/workflow/import"
RUN_PATH = "/api/v1/agents/tasks/workflow/run"
RUNS_PATH_PREFIX = "/api/v1/agents/tasks/workflow/runs"

POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 5 * 60
HTTP_TIMEOUT_SECONDS = 30


class DifyDraftError(RuntimeError):
    pass


def _get_token() -> str:
    return os.getenv("DIFY_TOKEN") or os.getenv("EVO_TOKEN") or os.getenv("FRONTIS_TOKEN") or ""


def _require_output_dir() -> Path:
    out = os.getenv("EVO_OUTPUT_DIR")
    if not out:
        raise DifyDraftError("Missing required env var EVO_OUTPUT_DIR")
    out_dir = Path(out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _resolve_base_url(cli_base_url: Optional[str]) -> str:
    return (cli_base_url or os.getenv("DIFY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _setup_logging(out_dir: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("fetch_dify_result")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG if verbose else logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        fh = logging.FileHandler(out_dir / "fetch_dify_result.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_kv_pairs(pairs: Iterable[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise DifyDraftError(f"Invalid --set value (expected key=value): {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise DifyDraftError(f"Invalid --set value (empty key): {item!r}")

        value = raw_value.strip()
        if value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
            continue
        if value.lower() in {"null", "none"}:
            result[key] = None
            continue
        try:
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                result[key] = int(value)
                continue
            result[key] = float(value)
            continue
        except ValueError:
            pass

        result[key] = value
    return result


def _extract_start_variables(workflow_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = ((workflow_obj.get("workflow") or {}).get("graph") or {}).get("nodes") or []
    for node in nodes:
        data = node.get("data") or {}
        if data.get("type") == "start":
            return list(data.get("variables") or [])
    return []


def _normalize_start_var_schema(start_vars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize and filter Start.variables entries into a stable schema:
    - variable (str) is required
    - required/type/label/options/default/max_length are optional
    """
    schema: List[Dict[str, Any]] = []
    for v in start_vars:
        if not isinstance(v, dict):
            continue
        var_name = v.get("variable")
        if not isinstance(var_name, str) or not var_name.strip():
            continue
        schema.append(
            {
                "variable": var_name.strip(),
                "label": v.get("label") if isinstance(v.get("label"), str) else "",
                "type": v.get("type") if isinstance(v.get("type"), str) else "",
                "required": bool(v.get("required")) if "required" in v else False,
                "options": v.get("options") if isinstance(v.get("options"), list) else [],
                "default": v.get("default"),
                "max_length": v.get("max_length"),
            }
        )
    # Deduplicate by variable name (first wins) while preserving order
    seen: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for item in schema:
        name = item["variable"]
        if name in seen:
            continue
        seen.add(name)
        deduped.append(item)
    return deduped


def _extract_allowed_select_values(options: List[Any]) -> List[str]:
    values: List[str] = []
    for opt in options:
        if isinstance(opt, str) and opt.strip():
            values.append(opt.strip())
            continue
        if isinstance(opt, dict):
            for key in ("value", "label", "name", "id"):
                v = opt.get(key)
                if isinstance(v, str) and v.strip():
                    values.append(v.strip())
                    break
    # dedupe preserve order
    seen: Set[str] = set()
    out: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def render_inputs_markdown_template(schema: List[Dict[str, Any]]) -> str:
    """
    Render a user-fillable Markdown template for workflow inputs.
    Intended for the agent to show users in the chat window.
    """
    if not schema:
        return "（未从 Start 节点 variables 中解析到输入参数；该 workflow 可能无入参或未按 DSL 填写 variables。）"

    lines: List[str] = []
    lines.append("请按下列参数填写运行输入（建议直接回复一个 JSON 对象，key 为参数名）。")
    lines.append("")
    # Provide a JSON skeleton with label text as placeholders so users know each variable's purpose.
    if schema:
        skeleton = {s["variable"]: (s.get("label") or "请填写") for s in schema}
        lines.append("建议直接按以下 JSON 骨架填写（value 为每个参数的用途说明，可直接替换成你的真实内容）：")
        lines.append("```json")
        lines.append(json.dumps(skeleton, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    for item in schema:
        name = item.get("variable", "")
        label = item.get("label", "") or ""
        var_type = item.get("type", "") or ""
        required = bool(item.get("required"))
        max_length = item.get("max_length")
        default = item.get("default")
        options = item.get("options") or []

        title = f"## {name}"
        if label:
            title += f"：{label}"
        lines.append(title)
        meta_parts = []
        if required:
            meta_parts.append("必填")
        else:
            meta_parts.append("可选")
        if var_type:
            meta_parts.append(f"type={var_type}")
        if isinstance(max_length, int) and max_length > 0:
            meta_parts.append(f"max_length={max_length}")
        if default not in (None, ""):
            meta_parts.append(f"default={default!r}")
        allowed = _extract_allowed_select_values(options) if var_type == "select" else []
        if allowed:
            meta_parts.append(f"options={allowed}")
        if meta_parts:
            lines.append(f"- " + "；".join(meta_parts))
        if label:
            lines.append(f"- 说明：{label}")
            lines.append(f"- 建议填写：{label}")
        if var_type in {"paragraph"}:
            lines.append("- 填写建议：输入较长文本，可多段粘贴。")
        elif var_type in {"text-input", "text"}:
            lines.append("- 填写建议：输入简短文本。")
        elif var_type == "number":
            lines.append("- 填写建议：输入数字（可为整数或小数）。")
        elif var_type == "select" and allowed:
            lines.append("- 填写建议：从 options 中选择一个值。")
        elif var_type == "file":
            lines.append("- 填写建议：仅支持已上传文件变量（如需文件，请改用可运行的文件上传流程）。")
        elif var_type == "file-list":
            lines.append("- 填写建议：仅支持已上传文件列表变量（如需文件，请改用可运行的文件上传流程）。")
        lines.append("")
        lines.append("请填写：")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def validate_workflow_inputs(
    *,
    schema: List[Dict[str, Any]],
    inputs: Dict[str, Any],
    strict: bool,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate `inputs` against Start.variables schema.

    Returns:
        (ok, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    by_name: Dict[str, Dict[str, Any]] = {s["variable"]: s for s in schema if isinstance(s.get("variable"), str)}
    required = [name for name, s in by_name.items() if s.get("required")]

    missing = [k for k in required if k not in inputs]
    if missing:
        errors.append(f"Missing required inputs: {missing}")

    unknown = [k for k in inputs.keys() if k not in by_name]
    if unknown:
        warnings.append(f"Extra inputs provided (not in start.variables): {unknown}")

    for name, value in inputs.items():
        if name not in by_name:
            continue
        spec = by_name[name]
        var_type = (spec.get("type") or "").strip()

        if spec.get("required"):
            if value is None:
                errors.append(f"Input {name!r} is required but value is null")
            elif isinstance(value, str) and not value.strip():
                errors.append(f"Input {name!r} is required but value is empty string")

        if var_type == "number":
            if isinstance(value, (int, float)):
                pass
            elif isinstance(value, str):
                try:
                    float(value.strip())
                except ValueError:
                    errors.append(f"Input {name!r} expects number, got {value!r}")
            else:
                errors.append(f"Input {name!r} expects number, got {type(value).__name__}")

        if var_type == "select":
            allowed = _extract_allowed_select_values(spec.get("options") or [])
            if allowed and value is not None:
                v = value.strip() if isinstance(value, str) else str(value)
                if v not in allowed:
                    errors.append(f"Input {name!r} not in options {allowed}: got {v!r}")

        max_length = spec.get("max_length")
        if isinstance(max_length, int) and max_length > 0 and isinstance(value, str):
            if len(value) > max_length:
                warnings.append(f"Input {name!r} length {len(value)} exceeds max_length={max_length}")

    ok = not errors
    if not strict and errors:
        # In non-strict mode, downgrade to warnings so agent can proceed if desired.
        warnings.extend(errors)
        errors = []
        ok = True

    return ok, errors, warnings


@dataclass(frozen=True)
class ImportResult:
    app_id: str
    raw: Dict[str, Any]


@dataclass(frozen=True)
class RunResult:
    record_id: str
    raw: Dict[str, Any]


def upload_or_update_workflow(
    *,
    workflow_json_path: Path,
    base_url: str,
    token: str,
    input_dir: Optional[str],
    session: Optional[requests.Session],
    logger: logging.Logger,
) -> ImportResult:
    """
    Upload/import a workflow DSL into Dify draft system.

    Returns:
        ImportResult(app_id=..., raw=response_json)
    """
    sess = session or requests.Session()

    workflow_obj = _read_json(workflow_json_path)
    mode = ((workflow_obj.get("app") or {}).get("mode") or "workflow")

    url = f"{base_url}{IMPORT_PATH}"
    params: Dict[str, str] = {}
    if input_dir:
        params["input_dir"] = input_dir

    payload = {
        "mode": "yaml-content",
        "yaml_content": json.dumps(workflow_obj, ensure_ascii=False),
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    logger.info(f"Importing workflow DSL: mode={mode}, file={workflow_json_path}")
    logger.debug(f"POST {url} params={params}")
    resp = sess.post(url, headers=headers, params=params, json=payload, timeout=HTTP_TIMEOUT_SECONDS)
    logger.info(f"Import response: http={resp.status_code}")

    try:
        obj = resp.json()
    except ValueError as exc:
        raise DifyDraftError(f"Import returned non-JSON response: http={resp.status_code}, text={resp.text[:2000]!r}") from exc

    if resp.status_code >= 400:
        raise DifyDraftError(f"Import failed: http={resp.status_code}, body={obj}")

    data = (obj or {}).get("data") or {}
    app_id = data.get("app_id") or data.get("appId") or data.get("dify_app_id")
    if not app_id:
        raise DifyDraftError(f"Import response missing app_id: {obj}")

    logger.info(f"Imported/updated workflow app_id={app_id}")
    return ImportResult(app_id=str(app_id), raw=obj)


def run_workflow_and_get_result(
    *,
    base_url: str,
    token: str,
    app_id: str,
    task_id: Optional[str],
    inputs: Dict[str, Any],
    session: Optional[requests.Session],
    logger: logging.Logger,
    poll_timeout_seconds: int = POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: int = POLL_INTERVAL_SECONDS,
) -> Dict[str, Any]:
    """
    Run the draft workflow, then poll its run record until completion.

    Returns:
        The final run record JSON (GET /runs/{record_id} response object).
    """
    sess = session or requests.Session()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    run_url = f"{base_url}{RUN_PATH}"
    run_params: Dict[str, str] = {"app_id": app_id}
    if task_id:
        run_params["task_id"] = task_id

    run_payload = {"task_id": task_id or "", "inputs": inputs, "files": []}
    logger.info(f"Running workflow: app_id={app_id} task_id={task_id or '(empty)'} inputs_keys={sorted(inputs.keys())}")
    logger.debug(f"POST {run_url} params={run_params}")
    run_resp = sess.post(
        run_url, headers=headers, params=run_params, json=run_payload, timeout=HTTP_TIMEOUT_SECONDS
    )
    logger.info(f"Run response: http={run_resp.status_code}")

    try:
        run_obj = run_resp.json()
    except ValueError as exc:
        raise DifyDraftError(f"Run returned non-JSON response: http={run_resp.status_code}, text={run_resp.text[:2000]!r}") from exc

    if run_resp.status_code >= 400:
        raise DifyDraftError(f"Run failed: http={run_resp.status_code}, body={run_obj}")

    record_id = ((run_obj or {}).get("data") or {}).get("workflow_run_record_id")
    if not record_id:
        raise DifyDraftError(f"Run response missing workflow_run_record_id: {run_obj}")

    record_id = str(record_id)
    logger.info(f"workflow_run_record_id={record_id}")

    poll_url = f"{base_url}{RUNS_PATH_PREFIX}/{record_id}"
    deadline = time.time() + poll_timeout_seconds
    last_status: Optional[str] = None
    attempt = 0

    while True:
        attempt += 1
        resp = sess.get(poll_url, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
        logger.debug(f"GET {poll_url} -> http={resp.status_code} attempt={attempt}")

        try:
            obj = resp.json()
        except ValueError:
            obj = {"_non_json_text": resp.text[:2000]}

        if resp.status_code >= 400:
            raise DifyDraftError(f"Poll failed: http={resp.status_code}, body={obj}")

        data = (obj or {}).get("data") or {}
        run_status = data.get("run_status") or ((data.get("response_payload") or {}).get("data") or {}).get("status")
        if isinstance(run_status, str) and run_status != last_status:
            last_status = run_status
            logger.info(f"run_status={run_status} (attempt={attempt})")

        if run_status in {"succeeded", "failed"}:
            return obj

        if time.time() >= deadline:
            raise DifyDraftError(f"Polling timed out after {poll_timeout_seconds}s (record_id={record_id}). Last status={run_status!r}")

        time.sleep(poll_interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload/import a Dify workflow JSON and run it until completion.")
    parser.add_argument("--workflow-json", required=True, help="Path to generated Dify workflow JSON file")
    parser.add_argument("--base-url", default=None, help=f"API base url (default: {DEFAULT_BASE_URL} or env DIFY_BASE_URL)")
    parser.add_argument("--task-id", default=None, help="task_id (optional; required when app_id is not provided by import in your env)")
    parser.add_argument("--input-dir", default=None, help="Query param input_dir for import API (optional)")
    parser.add_argument("--inputs-json", default=None, help="JSON file containing workflow inputs dict")
    parser.add_argument("--set", action="append", default=[], help="Set input key=value (repeatable)")
    parser.add_argument("--strict-inputs", action="store_true", help="Fail on input validation errors (recommended)")
    parser.add_argument("--print-input-template", action="store_true", help="Print Start.variables markdown template and exit")
    parser.add_argument("--write-input-template", action="store_true", help="Write input template markdown to $EVO_OUTPUT_DIR/workflow_input_template.md")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs and exit (no import/run)")
    parser.add_argument("--print-outputs", action="store_true", help="Print workflow outputs to stdout after run")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    args = parser.parse_args()

    out_dir = _require_output_dir()
    base_url = _resolve_base_url(args.base_url)
    logger = _setup_logging(out_dir, verbose=args.verbose)

    token = _get_token()

    workflow_path = Path(args.workflow_json).expanduser()
    if not workflow_path.exists():
        raise DifyDraftError(f"workflow json not found: {workflow_path}")

    workflow_obj = _read_json(workflow_path)
    start_vars = _extract_start_variables(workflow_obj)
    schema = _normalize_start_var_schema(start_vars)
    logger.info(f"Detected start.variables: {[s['variable'] for s in schema]}")

    if args.print_input_template or args.write_input_template:
        md = render_inputs_markdown_template(schema)
        if args.print_input_template:
            print(md)
        if args.write_input_template:
            path = out_dir / "workflow_input_template.md"
            path.write_text(md, encoding="utf-8")
            logger.info(f"Wrote workflow_input_template.md: {path}")
        return 0

    inputs: Dict[str, Any] = {}
    if args.inputs_json:
        inputs_path = Path(args.inputs_json).expanduser()
        inputs = _read_json(inputs_path)
        if not isinstance(inputs, dict):
            raise DifyDraftError(f"--inputs-json must contain an object/dict, got: {type(inputs).__name__}")
    inputs.update(_parse_kv_pairs(args.set))

    ok, errors, warnings = validate_workflow_inputs(schema=schema, inputs=inputs, strict=args.strict_inputs)
    for w in warnings:
        logger.warning(w)
    if errors:
        for e in errors:
            logger.error(e)
        return 1

    # Persist request inputs for traceability
    _write_json(out_dir / "workflow_inputs.json", inputs)
    _write_json(out_dir / "inputs.json", inputs)

    if args.validate_only:
        logger.info("validate-only: inputs are acceptable; skipping import/run.")
        return 0

    sess = requests.Session()
    import_res = upload_or_update_workflow(
        workflow_json_path=workflow_path,
        base_url=base_url,
        token=token,
        input_dir=args.input_dir or str(out_dir),
        session=sess,
        logger=logger,
    )
    _write_json(out_dir / "import_response.json", import_res.raw)

    final_obj = run_workflow_and_get_result(
        base_url=base_url,
        token=token,
        app_id=import_res.app_id,
        task_id=args.task_id,
        inputs=inputs,
        session=sess,
        logger=logger,
    )
    _write_json(out_dir / "run_final_record.json", final_obj)

    payload = ((final_obj.get("data") or {}).get("response_payload") or {}).get("data") or {}
    outputs = payload.get("outputs") or {}
    if isinstance(outputs, dict):
        result_text = outputs.get("result") or outputs.get("text")
        if isinstance(result_text, str) and result_text.strip():
            (out_dir / "workflow_result.txt").write_text(result_text, encoding="utf-8")
            logger.info(f"Wrote workflow_result.txt ({len(result_text)} chars)")
    else:
        result_text = None

    if args.print_outputs:
        if isinstance(outputs, dict) and outputs:
            print(json.dumps(outputs, ensure_ascii=False, indent=2))
        elif isinstance(result_text, str) and result_text.strip():
            print(result_text)
        else:
            print("{}")

    logger.info(f"Done. Artifacts in: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except DifyDraftError as exc:
        print(f"❌ {exc}")
        code = 1
    except Exception as exc:
        print(f"❌ Unexpected error: {exc}")
        code = 1
    raise SystemExit(code)
