#!/usr/bin/env python3
"""
Sync Dify models and knowledge bases into local config files.

Requirements:
- No CLI args; configuration is centralized in URLS below.
- Timeout is fixed at 5 seconds.
- No retries.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

DEFAULT_BASE_URL = "http://10.200.2.52"


def _get_base_url() -> str:
    return (os.getenv("DIFY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _build_urls(base_url: str) -> Dict[str, str]:
    return {
        "models": f"{base_url}/api/v1/agents/models/llm",
        "datasets": f"{base_url}/api/v1/agents/datasets",
    }

TIMEOUT_SECONDS = 30


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_token() -> str:
    return os.getenv("DIFY_TOKEN") or os.getenv("EVO_TOKEN") or os.getenv("FRONTIS_TOKEN") or ""


def _get_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _request_json(url: str, headers: Dict[str, str], params: Dict[str, Any] = None) -> Dict[str, Any]:
    resp = requests.get(url, headers=headers, params=params or {}, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def _parse_models(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    data = (payload or {}).get("data", {}).get("data", []) or []
    for provider_block in data:
        provider_name = provider_block.get("provider")
        for m in provider_block.get("models", []) or []:
            model_name = m.get("model")
            label = (m.get("label") or {}).get("zh_Hans") or (m.get("label") or {}).get("en_US") or model_name
            mode = (m.get("model_properties") or {}).get("mode") or "chat"
            if not provider_name or not model_name:
                continue
            models.append(
                {
                    "provider": provider_name,
                    "name": model_name,
                    "mode": mode,
                    "default_completion_params": {"temperature": 0.7},
                }
            )
    return models


def _parse_datasets(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    datasets: List[Dict[str, Any]] = []
    data = (payload or {}).get("data", {}).get("data", []) or []
    for d in data:
        datasets.append(
            {
                "id": d.get("id"),
                "name": d.get("name"),
                "description": d.get("description"),
                "provider": d.get("provider"),
                # "data_source_type": d.get("data_source_type"),
                # "indexing_technique": d.get("indexing_technique"),
                # "embedding_model": d.get("embedding_model"),
                # "embedding_model_provider": d.get("embedding_model_provider"),
                "retrieval_model_dict": d.get("retrieval_model_dict"),
                "external_retrieval_model": d.get("external_retrieval_model"),
            }
        )
    return datasets


def _fetch_all_datasets(headers: Dict[str, str], datasets_url: str) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    page = 1
    while True:
        payload = _request_json(
            datasets_url,
            headers=headers,
            params={"page": page, "limit": 100},
        )
        all_items.extend(_parse_datasets(payload))
        data = (payload or {}).get("data", {}) or {}
        if not data.get("has_more"):
            break
        page += 1
    return all_items


def _write_json(path: Path, content: Dict[str, Any]) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    urls = _build_urls(_get_base_url())
    token = _get_token()
    # if not token:
    #     print("❌ Missing token. Set DIFY_TOKEN (or EVO_TOKEN / FRONTIS_TOKEN).")
    #     return 1

    headers = _get_headers(token)
    try:
        models_payload = _request_json(urls["models"], headers=headers)
        models = _parse_models(models_payload)
        datasets = _fetch_all_datasets(headers, urls["datasets"])
    except requests.RequestException as exc:
        print(f"❌ Request failed: {exc}")
        return 1
    except ValueError as exc:
        print(f"❌ Invalid JSON response: {exc}")
        return 1

    base_dir = Path(__file__).resolve().parents[1] / "config"
    models_path = base_dir / "models.json"
    kb_path = base_dir / "knowledge_bases.json"

    _write_json(
        models_path,
        {
            "schema_version": "0.1",
            "last_updated": _now_iso(),
            "source": {"type": "dify_api", "url": urls["models"]},
            "models": models,
        },
    )
    _write_json(
        kb_path,
        {
            "schema_version": "0.1",
            "last_updated": _now_iso(),
            "source": {"type": "dify_api", "url": urls["datasets"]},
            "knowledge_bases": datasets,
        },
    )

    print(f"✅ Synced {len(models)} models -> {models_path}")
    print(f"✅ Synced {len(datasets)} knowledge bases -> {kb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
