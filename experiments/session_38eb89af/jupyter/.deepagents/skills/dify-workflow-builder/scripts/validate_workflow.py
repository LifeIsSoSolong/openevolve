#!/usr/bin/env python3
"""
Dify Workflow DSL validator (YAML/JSON).

Goal:
- Catch common schema mistakes that lead to Dify import/runtime failures.
- Align key rules with Dify backend (Graph validation + NodeData pydantic schemas).

This validator is intentionally "practical":
- Some fields are optional in backend but recommended for UX; those become warnings.
- Config-driven constraints (models/tools/knowledge bases) only become strict when the
  corresponding config file exists and is non-empty.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _load_workflow_dsl(path: Path) -> Dict[str, Any]:
    """
    Load workflow DSL from JSON or YAML.

    Dify export is YAML, but JSON is valid YAML 1.2 subset and is often used by skills.
    """
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        return data

    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ValueError("File is not valid JSON; PyYAML is required to parse YAML.") from exc

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("DSL content must be a mapping at top-level.")
    return loaded


class WorkflowValidator:
    """Workflow DSL validator."""

    def __init__(self, workflow: Dict[str, Any]):
        self.workflow = workflow
        self.errors: List[str] = []
        self.warnings: List[str] = []

        self._models_config = self._load_models_config()
        self._tools_config = self._load_tools_config()
        self._knowledge_bases_config = self._load_knowledge_bases_config()

        self.allowed_models = self._models_config["allowed_models"]
        self.allowed_tools = self._tools_config["allowed_tools"]
        self.allowed_dataset_ids = self._knowledge_bases_config["allowed_dataset_ids"]

        graph = (workflow.get("workflow") or {}).get("graph") or {}
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or []

        self.nodes: Dict[str, Dict[str, Any]] = {n["id"]: n for n in nodes if isinstance(n, dict) and "id" in n}
        self.edges: List[Dict[str, Any]] = [e for e in edges if isinstance(e, dict)]
        self.parent_ids: Dict[str, str] = {}
        self.children: Dict[str, List[str]] = defaultdict(list)
        for node_id, node in self.nodes.items():
            parent_id = node.get("parentId")
            if isinstance(parent_id, str) and parent_id:
                self.parent_ids[node_id] = parent_id
                self.children[parent_id].append(node_id)

        app = workflow.get("app") or {}
        self.mode = app.get("mode") or "workflow"

        self.adj: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adj: Dict[str, List[str]] = defaultdict(list)
        for edge in self.edges:
            src, tgt = edge.get("source"), edge.get("target")
            if isinstance(src, str) and isinstance(tgt, str) and src and tgt:
                self.adj[src].append(tgt)
                self.reverse_adj[tgt].append(src)

    def _load_models_config(self) -> Dict[str, Any]:
        config_path = Path(__file__).resolve().parents[1] / "config" / "models.json"
        if not config_path.exists():
            return {"present": False, "count": 0, "allowed_models": set()}
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {"present": True, "count": 0, "allowed_models": set(), "invalid": True}

        allowed: Set[Tuple[str, str]] = set()
        for model in data.get("models", []) or []:
            provider = model.get("provider")
            name = model.get("name")
            if provider and name:
                allowed.add((provider, name))
        return {"present": True, "count": len(allowed), "allowed_models": allowed}

    def _load_tools_config(self) -> Dict[str, Any]:
        config_path = Path(__file__).resolve().parents[1] / "config" / "tools.json"
        if not config_path.exists():
            return {"present": False, "count": 0, "allowed_tools": set()}
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {"present": True, "count": 0, "allowed_tools": set(), "invalid": True}

        allowed: Set[Tuple[str, str]] = set()
        for tool in data.get("tools", []) or []:
            provider_name = tool.get("provider_name")
            tool_name = tool.get("tool_name")
            if provider_name and tool_name:
                allowed.add((provider_name, tool_name))
        return {"present": True, "count": len(allowed), "allowed_tools": allowed}

    def _load_knowledge_bases_config(self) -> Dict[str, Any]:
        config_path = Path(__file__).resolve().parents[1] / "config" / "knowledge_bases.json"
        if not config_path.exists():
            return {"present": False, "count": 0, "allowed_dataset_ids": set()}
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {"present": True, "count": 0, "allowed_dataset_ids": set(), "invalid": True}

        allowed: Set[str] = set()
        for kb in data.get("knowledge_bases", []) or []:
            dataset_id = kb.get("dataset_id") or kb.get("id")
            if dataset_id:
                allowed.add(str(dataset_id))
        return {"present": True, "count": len(allowed), "allowed_dataset_ids": allowed}

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        self._check_top_level()
        self._check_edge_endpoints()
        self._check_entry_nodes()
        self._check_start_variables_schema()
        self._check_terminal_nodes()
        self._check_dag()
        self._check_reachability()
        self._check_unique_ids()
        self._check_variable_references()
        self._check_branch_completeness()
        self._check_iteration_constraints()

        self._check_document_extractor_schema()
        self._check_list_operator_schema()
        self._check_llm_schema()
        self._check_if_else_schema()
        self._check_template_transform_schema()
        self._check_code_schema()
        self._check_tools_schema()
        self._check_knowledge_retrieval_schema()

        return len(self.errors) == 0, self.errors, self.warnings

    def _check_top_level(self) -> None:
        kind = self.workflow.get("kind")
        if kind is None:
            self.warnings.append("W030: 建议在顶层显式提供 kind: 'app'")
        if kind is not None and kind != "app":
            self.errors.append(f"E100: kind 必须为 'app'（当前: {kind!r}）")

        version = self.workflow.get("version")
        if version is None:
            self.warnings.append("W031: 建议在顶层显式提供 version（Dify App DSL 版本）")
        if version is not None and not isinstance(version, str):
            self.errors.append(f"E101: version 必须为字符串（当前类型: {type(version).__name__}）")

        if not isinstance(self.workflow.get("app"), dict):
            self.errors.append("E102: 缺少顶层 app 对象")
        if not isinstance(self.workflow.get("workflow"), dict):
            self.errors.append("E103: 缺少顶层 workflow 对象")

        graph = (self.workflow.get("workflow") or {}).get("graph")
        if not isinstance(graph, dict):
            self.errors.append("E104: workflow.graph 必须为对象")

    def _check_start_variables_schema(self) -> None:
        """
        Align with backend StartNodeData.variables (VariableEntity).

        Key points:
        - variables is a list of VariableEntity
        - VariableEntity.options is Sequence[str] (NOT [{label,value}, ...])
        """
        allowed_types = {
            "text-input",
            "select",
            "paragraph",
            "number",
            "external_data_tool",
            "file",
            "file-list",
            "checkbox",
            "json_object",
        }

        for node_id, node in self.nodes.items():
            data = node.get("data") or {}
            if data.get("type") != "start":
                continue

            variables = data.get("variables")
            if variables is None:
                continue
            if not isinstance(variables, list):
                self.errors.append(f"E070: Start节点 {node_id} 的 variables 必须是数组")
                continue

            seen_names: Set[str] = set()
            for i, var in enumerate(variables):
                if not isinstance(var, dict):
                    self.errors.append(f"E071: Start节点 {node_id} 的 variables[{i}] 必须是对象")
                    continue

                name = var.get("variable")
                label = var.get("label")
                vtype = var.get("type")

                if not isinstance(name, str) or not name:
                    self.errors.append(f"E072: Start节点 {node_id} 的 variables[{i}] 缺少 variable（非空字符串）")
                else:
                    if name in seen_names:
                        self.errors.append(f"E073: Start节点 {node_id} 存在重复的输入变量名: {name}")
                    seen_names.add(name)

                if not isinstance(label, str) or not label:
                    self.errors.append(f"E074: Start节点 {node_id} 的 variables[{i}] 缺少 label（非空字符串）")

                if not isinstance(vtype, str) or vtype not in allowed_types:
                    self.errors.append(
                        f"E075: Start节点 {node_id} 的 variables[{i}].type 无效: {vtype!r}（允许: {sorted(allowed_types)}）"
                    )

                if "required" in var and not isinstance(var.get("required"), bool):
                    self.errors.append(f"E076: Start节点 {node_id} 的 variables[{i}].required 必须是 boolean")

                max_length = var.get("max_length")
                if max_length is not None and not isinstance(max_length, int):
                    self.errors.append(f"E077: Start节点 {node_id} 的 variables[{i}].max_length 必须是整数或 null")

                options = var.get("options")
                if options is None:
                    options = []

                if not isinstance(options, list):
                    self.errors.append(f"E078: Start节点 {node_id} 的 variables[{i}].options 必须是数组")
                    options = []

                if any(isinstance(opt, dict) for opt in options):
                    self.errors.append(
                        f"E079: Start节点 {node_id} 的 variables[{i}].options 必须是 string 数组；不支持 {{label,value}} 对象数组"
                    )
                    continue
                if any(not isinstance(opt, str) for opt in options):
                    self.errors.append(f"E080: Start节点 {node_id} 的 variables[{i}].options 只能包含字符串")
                    continue

                if vtype == "select":
                    if not options:
                        self.errors.append(f"E081: Start节点 {node_id} 的 select 变量 {name!r} 必须提供非空 options")
                    default = var.get("default")
                    if default is not None and isinstance(default, str) and options and default not in options:
                        self.errors.append(
                            f"E082: Start节点 {node_id} 的 select 变量 {name!r} default={default!r} 不在 options 中"
                        )

                # Validate default value type for common variable types.
                default_value = var.get("default", None)
                if vtype == "file":
                    # In Dify backend, file variable is expected to be a File object at runtime,
                    # and default should be None (optional).
                    if default_value not in (None, ""):
                        self.errors.append(
                            f"E083: Start节点 {node_id} 的 file 变量 {name!r} default 必须为 null（不要用字符串/其他类型）"
                        )
                    if default_value == "":
                        self.errors.append(
                            f"E084: Start节点 {node_id} 的 file 变量 {name!r} default 不能为 \"\"；请改为 null"
                        )

                elif vtype == "file-list":
                    # Default for file-list should be [] (or None).
                    if default_value == "":
                        self.errors.append(
                            f"E085: Start节点 {node_id} 的 file-list 变量 {name!r} default 不能为 \"\"；请改为 []"
                        )
                    elif default_value is not None and not isinstance(default_value, list):
                        self.errors.append(
                            f"E086: Start节点 {node_id} 的 file-list 变量 {name!r} default 必须为 [] 或 null"
                        )

                elif vtype == "number":
                    if default_value is not None and not isinstance(default_value, (int, float)):
                        self.errors.append(
                            f"E087: Start节点 {node_id} 的 number 变量 {name!r} default 必须为数字或 null"
                        )

                elif vtype == "checkbox":
                    if default_value is not None and not isinstance(default_value, bool):
                        self.errors.append(
                            f"E088: Start节点 {node_id} 的 checkbox 变量 {name!r} default 必须为 boolean 或 null"
                        )

    def _check_edge_endpoints(self) -> None:
        for i, edge in enumerate(self.edges):
            source = edge.get("source")
            target = edge.get("target")
            if not isinstance(source, str) or not isinstance(target, str) or not source or not target:
                self.errors.append(f"E110: 第{i}条边缺少有效的 source/target")
                continue
            if source not in self.nodes:
                self.errors.append(f"E111: 边引用了不存在的 source 节点: {source}")
            if target not in self.nodes:
                self.errors.append(f"E112: 边引用了不存在的 target 节点: {target}")

    def _check_entry_nodes(self) -> None:
        start_nodes = [n for n in self.nodes.values() if (n.get("data") or {}).get("type") == "start"]
        trigger_nodes = [
            n
            for n in self.nodes.values()
            if isinstance((n.get("data") or {}).get("type"), str)
            and ((n.get("data") or {}).get("type") or "").startswith("trigger-")
        ]

        # Backend rule: start and trigger nodes cannot coexist (but either side alone is allowed).
        if start_nodes and trigger_nodes:
            self.errors.append("E001: Start节点与Trigger节点不能共存（请二选一）")

        if not start_nodes and not trigger_nodes:
            self.errors.append("E002: 缺少入口节点：需要 Start 或至少一个 Trigger 节点")

        if len(start_nodes) > 1:
            self.errors.append(f"E003: 存在{len(start_nodes)}个Start节点，应该只有1个")

        if len(start_nodes) == 1:
            start_id = start_nodes[0]["id"]
            if start_id in self.reverse_adj:
                self.errors.append("E004: Start节点不应有入边")

    def _check_terminal_nodes(self) -> None:
        end_nodes = [n for n in self.nodes.values() if (n.get("data") or {}).get("type") == "end"]
        answer_nodes = [n for n in self.nodes.values() if (n.get("data") or {}).get("type") == "answer"]

        if self.mode == "workflow":
            if not end_nodes:
                self.errors.append("E005: Workflow类型必须有End节点")
        elif self.mode == "chat":
            if not answer_nodes:
                self.errors.append("E006: Chatflow类型必须有Answer节点")

        for node in end_nodes:
            node_id = node.get("id")
            if node_id in self.adj and self.adj[node_id]:
                self.errors.append(f"E007: End节点 {node_id} 不应有出边")

    def _check_dag(self) -> None:
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            for neighbor in self.adj.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited and dfs(node_id):
                self.errors.append("E008: 检测到环路，Workflow必须是DAG")
                return

    def _check_reachability(self) -> None:
        start_nodes = [n["id"] for n in self.nodes.values() if (n.get("data") or {}).get("type") == "start"]
        trigger_nodes = [
            n["id"]
            for n in self.nodes.values()
            if isinstance((n.get("data") or {}).get("type"), str)
            and ((n.get("data") or {}).get("type") or "").startswith("trigger-")
        ]

        entry_ids = start_nodes or trigger_nodes
        if not entry_ids:
            return

        reachable: Set[str] = set()
        stack = list(entry_ids)
        while stack:
            node_id = stack.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            stack.extend(self.adj.get(node_id, []))

        # Treat container children (e.g. iteration/loop internal nodes) as reachable if their parent is reachable.
        # This matches how Dify models subgraphs via parentId rather than explicit edges from parent.
        def mark_children(parent_id: str) -> None:
            for child_id in self.children.get(parent_id, []):
                if child_id in reachable:
                    continue
                reachable.add(child_id)
                mark_children(child_id)

        for rid in list(reachable):
            mark_children(rid)

        orphans = set(self.nodes.keys()) - reachable
        if orphans:
            self.errors.append(f"E009: 以下节点无法从入口节点到达: {sorted(orphans)}")

        terminal_nodes = [
            n["id"] for n in self.nodes.values() if (n.get("data") or {}).get("type") in ("end", "answer")
        ]
        if not terminal_nodes:
            return

        can_reach_terminal: Set[str] = set()
        stack = list(terminal_nodes)
        while stack:
            node_id = stack.pop()
            if node_id in can_reach_terminal:
                continue
            can_reach_terminal.add(node_id)
            stack.extend(self.reverse_adj.get(node_id, []))

        dead_ends = reachable - can_reach_terminal
        if dead_ends:
            self.warnings.append(f"W002: 以下节点无法到达终止节点: {sorted(dead_ends)}")

    def _check_unique_ids(self) -> None:
        node_ids = list(self.nodes.keys())
        if len(node_ids) != len(set(node_ids)):
            self.errors.append("E010: 存在重复的节点ID")

        edge_ids: List[str] = []
        missing_edge_id = 0
        for edge in self.edges:
            edge_id = edge.get("id")
            if not edge_id:
                missing_edge_id += 1
                edge_id = f"{edge.get('source')}-{edge.get('sourceHandle','source')}-{edge.get('target')}-{edge.get('targetHandle','target')}"
            edge_ids.append(str(edge_id))

        if missing_edge_id:
            self.warnings.append(f"W003: 存在{missing_edge_id}条边缺少 id（建议补齐以便编辑与排错）")

        if len(edge_ids) != len(set(edge_ids)):
            self.errors.append("E011: 存在重复的边ID（或缺少 id 且自动生成后发生冲突）")

    def _check_variable_references(self) -> None:
        import re

        var_pattern = re.compile(r"\{\{#([^.]+)\.([^#]+)#\}\}")
        # Node output contracts (derived from backend runtime behavior in this repo).
        # Keep this minimal and focused on high-confidence keys to avoid false positives.
        allowed_outputs_by_type: Dict[str, Set[str]] = {
            "llm": {"text", "usage", "reasoning_content"},
            # KnowledgeRetrievalNode outputs only 'result' in this repo version.
            "knowledge-retrieval": {"result"},
            "http-request": {"status_code", "body", "headers"},
            "template-transform": {"output"},
            "document-extractor": {"text"},
            "list-operator": {"result", "first_record", "last_record"},
            "variable-aggregator": {"output"},
            "iteration": {"output", "item", "index"},
        }

        def get_upstream(node_id: str) -> Set[str]:
            upstream: Set[str] = set()
            stack = list(self.reverse_adj.get(node_id, []))
            while stack:
                n = stack.pop()
                if n in upstream:
                    continue
                upstream.add(n)
                stack.extend(self.reverse_adj.get(n, []))
            # For subgraph nodes (iteration/loop children), allow referencing parent container variables.
            parent_id = self.parent_ids.get(node_id)
            while parent_id:
                upstream.add(parent_id)
                parent_id = self.parent_ids.get(parent_id)
            return upstream

        for node_id, node in self.nodes.items():
            data = node.get("data") or {}
            data_str = json.dumps(data, ensure_ascii=False)

            upstream = get_upstream(node_id)
            upstream.update({"sys", "env"})

            for match in var_pattern.finditer(data_str):
                ref_node = match.group(1)
                ref_var_full = match.group(2)
                ref_var = ref_var_full.split(".", 1)[0]

                if ref_node not in upstream and ref_node not in self.nodes and ref_node not in ("sys", "env"):
                    self.errors.append(f"E012: 节点 {node_id} 引用了不存在的节点变量 {match.group(0)}")
                elif ref_node in self.nodes and ref_node not in upstream:
                    self.errors.append(f"E013: 节点 {node_id} 引用了下游节点 {ref_node} 的变量")
                elif ref_node in self.nodes:
                    ref_node_type = (self.nodes[ref_node].get("data") or {}).get("type")
                    if isinstance(ref_node_type, str):
                        if ref_node_type == "code":
                            outputs = ((self.nodes[ref_node].get("data") or {}).get("outputs")) or {}
                            if isinstance(outputs, dict) and ref_var not in outputs:
                                self.errors.append(
                                    f"E120: 节点 {node_id} 引用了 Code节点 {ref_node} 不存在的输出 '{ref_var_full}'"
                                )
                        else:
                            allowed = allowed_outputs_by_type.get(ref_node_type)
                            if allowed and ref_var not in allowed:
                                self.errors.append(
                                    f"E121: 节点 {node_id} 引用了 {ref_node_type} 节点 {ref_node} 不存在的输出 '{ref_var_full}'（允许: {sorted(allowed)}）"
                                )

    def _check_branch_completeness(self) -> None:
        for node_id, node in self.nodes.items():
            node_type = (node.get("data") or {}).get("type")

            if node_type == "if-else":
                out_handles = {edge.get("sourceHandle") for edge in self.edges if edge.get("source") == node_id}
                if "true" not in out_handles:
                    self.errors.append(f"E014: IF/ELSE节点 {node_id} 缺少 true 分支（sourceHandle='true'）")
                if "false" not in out_handles:
                    self.errors.append(f"E015: IF/ELSE节点 {node_id} 缺少 false 分支（sourceHandle='false'）")

            elif node_type == "question-classifier":
                classes = (node.get("data") or {}).get("classes") or []
                if not isinstance(classes, list) or not classes:
                    continue
                class_ids = {c.get("id") for c in classes if isinstance(c, dict) and c.get("id")}
                out_handles = {edge.get("sourceHandle") for edge in self.edges if edge.get("source") == node_id}
                missing = {cid for cid in class_ids if cid not in out_handles}
                if missing:
                    self.errors.append(
                        f"E016: Question Classifier节点 {node_id} 以下分类缺少出边（sourceHandle=class.id）: {sorted(missing)}"
                    )

    def _check_iteration_constraints(self) -> None:
        allowed_error_modes = {"terminated", "continue-on-error", "remove-abnormal-output"}

        for node_id, node in self.nodes.items():
            data = node.get("data") or {}
            if data.get("type") != "iteration":
                continue

            iterator_selector = data.get("iterator_selector")
            if not isinstance(iterator_selector, list) or len(iterator_selector) < 2:
                self.errors.append(f"E017: Iteration节点 {node_id} 缺少 iterator_selector（应为 [node_id, var]）")

            error_handle_mode = data.get("error_handle_mode")
            if error_handle_mode is not None and error_handle_mode not in allowed_error_modes:
                self.errors.append(
                    f"E018: Iteration节点 {node_id} 的 error_handle_mode 无效: {error_handle_mode!r}（允许: {sorted(allowed_error_modes)}）"
                )

    def _check_document_extractor_schema(self) -> None:
        """
        Align with backend DocumentExtractorNodeData:
        - type must be 'document-extractor'
        - variable_selector must be Sequence[str]
        """
        for node_id, node in self.nodes.items():
            data = node.get("data") or {}
            node_type = data.get("type")

            # common mistake: 'doc-extractor' (not a valid NodeType)
            if node_type == "doc-extractor":
                self.errors.append(
                    f"E090: 节点 {node_id} 使用了无效类型 'doc-extractor'；正确类型应为 'document-extractor'"
                )
                continue

            if node_type != "document-extractor":
                continue

            variable_selector = data.get("variable_selector")
            if not isinstance(variable_selector, list) or len(variable_selector) < 2:
                self.errors.append(
                    f"E091: Document Extractor节点 {node_id} 缺少 variable_selector（应为 [node_id, var]）"
                )

            if "is_array_file" in data:
                self.warnings.append(
                    f"W090: Document Extractor节点 {node_id} 包含未知字段 is_array_file（后端 schema 不需要）"
                )

    def _check_list_operator_schema(self) -> None:
        """
        Align with backend ListOperatorNodeData:
        - variable: Sequence[str]
        - filter_by/order_by/limit are required objects
        """
        for node_id, node in self.nodes.items():
            data = node.get("data") or {}
            if data.get("type") != "list-operator":
                continue

            # common mistake: use 'filter' instead of 'filter_by'
            if "filter" in data and "filter_by" not in data:
                self.errors.append(
                    f"E092: List Operator节点 {node_id} 使用了字段 filter；应改为 filter_by（结构见后端 ListOperatorNodeData）"
                )

            variable = data.get("variable")
            if not isinstance(variable, list) or len(variable) < 2:
                self.errors.append(f"E093: List Operator节点 {node_id} 缺少 variable（应为 [node_id, var]）")

            for required_key in ("filter_by", "order_by", "limit"):
                if required_key not in data or not isinstance(data.get(required_key), dict):
                    self.errors.append(f"E094: List Operator节点 {node_id} 缺少 {required_key} 对象")

            extract_by = data.get("extract_by")
            if extract_by is not None and not isinstance(extract_by, dict):
                self.errors.append(f"E095: List Operator节点 {node_id} 的 extract_by 必须是对象")

    def _check_llm_schema(self) -> None:
        allowed_roles = {"system", "user", "assistant"}

        for node_id, node in self.nodes.items():
            data = node.get("data") or {}
            if data.get("type") != "llm":
                continue

            # Dify runtime expects LLM config inside data.model
            if "temperature" in data or "max_tokens" in data:
                self.errors.append(
                    f"E020: LLM节点 {node_id} 使用了顶层 temperature/max_tokens；请将其放入 data.model.completion_params"
                )

            model = data.get("model")
            if not isinstance(model, dict):
                self.errors.append(f"E021: LLM节点 {node_id} 缺少 model 或 model 不是对象")
                continue

            for key in ("provider", "name", "mode"):
                if not model.get(key):
                    self.errors.append(f"E022: LLM节点 {node_id} 的 model 缺少必填字段: {key}")

            provider = model.get("provider")
            name = model.get("name")
            if self._models_config.get("present") and self._models_config.get("count", 0) == 0:
                self.errors.append("E023: config/models.json 中 models 为空；无法为 LLM 选择可用模型（请先同步配置）")
            elif self.allowed_models and provider and name and (provider, name) not in self.allowed_models:
                self.errors.append(
                    f"E024: LLM节点 {node_id} 使用了未在 config/models.json 声明的模型: provider={provider}, name={name}"
                )

            completion_params = model.get("completion_params")
            if completion_params is None:
                self.warnings.append(f"W010: LLM节点 {node_id} 未显式提供 model.completion_params（将使用Dify默认值）")
            elif not isinstance(completion_params, dict):
                self.errors.append(f"E025: LLM节点 {node_id} 的 model.completion_params 必须是对象")

            prompt_template = data.get("prompt_template")
            if not isinstance(prompt_template, list) or not prompt_template:
                self.errors.append(f"E026: LLM节点 {node_id} 缺少 prompt_template 或为空")
                continue

            for i, item in enumerate(prompt_template):
                if not isinstance(item, dict):
                    self.errors.append(f"E027: LLM节点 {node_id} 的 prompt_template[{i}] 不是对象")
                    continue
                role = item.get("role")
                if role not in allowed_roles:
                    self.errors.append(
                        f"E028: LLM节点 {node_id} 的 prompt_template[{i}] role无效: {role}（允许: {sorted(allowed_roles)}）"
                    )
                if "text" not in item or not isinstance(item.get("text"), str):
                    self.errors.append(f"E029: LLM节点 {node_id} 的 prompt_template[{i}] 缺少 text 或 text不是字符串")
                if not item.get("id"):
                    self.warnings.append(
                        f"W011: LLM节点 {node_id} 的 prompt_template[{i}] 缺少 id（不影响运行，但可能影响前端编辑体验）"
                    )

            # Backend schema requires context
            context = data.get("context")
            if not isinstance(context, dict):
                self.errors.append(f"E030: LLM节点 {node_id} 缺少 context（必须为对象，至少包含 enabled）")
            elif "enabled" not in context:
                self.errors.append(f"E031: LLM节点 {node_id} 的 context 缺少 enabled 字段")

            for optional_key in ("vision", "memory"):
                if optional_key not in data:
                    self.warnings.append(f"W001: LLM节点 {node_id} 建议补充 {optional_key} 字段（可保持 enabled=false）")

    def _check_if_else_schema(self) -> None:
        for node_id, node in self.nodes.items():
            data = node.get("data") or {}
            if data.get("type") != "if-else":
                continue

            cases = data.get("cases")
            if cases is None:
                self.errors.append(f"E032: IF/ELSE节点 {node_id} 缺少 cases（后端 schema 推荐使用 cases）")
                legacy_conditions = data.get("conditions")
                if isinstance(legacy_conditions, list) and legacy_conditions and isinstance(legacy_conditions[0], dict):
                    if "case_id" not in legacy_conditions[0] and "conditions" in legacy_conditions[0]:
                        self.warnings.append(
                            f"W040: IF/ELSE节点 {node_id} 使用了旧版 data.conditions 结构（分组条件），建议迁移为 data.cases"
                        )
                continue
            if not isinstance(cases, list) or not cases:
                self.errors.append(f"E033: IF/ELSE节点 {node_id} 的 cases 必须是非空数组")
                continue

            for i, case in enumerate(cases):
                if not isinstance(case, dict):
                    self.errors.append(f"E034: IF/ELSE节点 {node_id} 的 cases[{i}] 不是对象")
                    continue
                if not case.get("case_id"):
                    self.errors.append(f"E035: IF/ELSE节点 {node_id} 的 cases[{i}] 缺少 case_id")
                if case.get("logical_operator") not in ("and", "or"):
                    self.errors.append(
                        f"E036: IF/ELSE节点 {node_id} 的 cases[{i}].logical_operator 必须为 'and'/'or'"
                    )
                conditions = case.get("conditions")
                if not isinstance(conditions, list):
                    self.errors.append(f"E037: IF/ELSE节点 {node_id} 的 cases[{i}].conditions 必须是数组")
                    continue
                for j, cond in enumerate(conditions):
                    if not isinstance(cond, dict):
                        self.errors.append(f"E038: IF/ELSE节点 {node_id} 的 cases[{i}].conditions[{j}] 不是对象")
                        continue
                    variable_selector = cond.get("variable_selector")
                    if not isinstance(variable_selector, list) or len(variable_selector) < 2:
                        self.errors.append(
                            f"E039: IF/ELSE节点 {node_id} 的条件缺少 variable_selector（应为 [node_id, var]）"
                        )
                    if not cond.get("comparison_operator"):
                        self.errors.append(f"E040: IF/ELSE节点 {node_id} 的条件缺少 comparison_operator")

    def _check_template_transform_schema(self) -> None:
        for node_id, node in self.nodes.items():
            data = node.get("data") or {}
            node_type = data.get("type")
            if node_type == "template":
                self.errors.append(f"E041: Template节点 {node_id} 使用了过时类型 'template'；请使用 'template-transform'")
            if node_type != "template-transform":
                continue
            if not isinstance(data.get("template"), str):
                self.errors.append(f"E042: TemplateTransform节点 {node_id} 缺少 template 字符串")
            if not isinstance(data.get("variables"), list):
                self.errors.append(f"E043: TemplateTransform节点 {node_id} 缺少 variables 数组")

    def _check_code_schema(self) -> None:
        for node_id, node in self.nodes.items():
            data = node.get("data") or {}
            if data.get("type") != "code":
                continue
            outputs = data.get("outputs")
            if not isinstance(outputs, dict) or not outputs:
                self.errors.append(f"E044: Code节点 {node_id} 的 outputs 必须是非空对象")
                continue
            for out_key, out_value in outputs.items():
                if not isinstance(out_key, str) or not out_key:
                    self.errors.append(f"E045: Code节点 {node_id} 的 outputs key 必须为非空字符串")
                    continue
                if not isinstance(out_value, dict) or "type" not in out_value:
                    self.errors.append(f"E046: Code节点 {node_id} 的 outputs['{out_key}'] 必须包含 type")

    def _check_tools_schema(self) -> None:
        for node_id, node in self.nodes.items():
            data = node.get("data") or {}
            if data.get("type") != "agent":
                continue

            agent_parameters = data.get("agent_parameters") or {}
            tools_param = agent_parameters.get("tools") or {}
            if not isinstance(tools_param, dict) or tools_param.get("type") != "constant":
                continue

            tools = tools_param.get("value")
            if tools is None:
                tools = []
            if not isinstance(tools, list):
                self.errors.append(f"E050: Agent节点 {node_id} 的 agent_parameters.tools.value 必须是数组")
                continue

            if tools and self._tools_config.get("present") and self._tools_config.get("count", 0) == 0:
                self.errors.append("E051: config/tools.json 中 tools 为空；无法使用 Agent 工具（请先补齐配置）")
                continue

            for i, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    self.errors.append(f"E052: Agent节点 {node_id} 的 tools[{i}] 不是对象")
                    continue
                provider_name = tool.get("provider_name")
                tool_name = tool.get("tool_name")
                if not provider_name or not tool_name:
                    self.errors.append(f"E053: Agent节点 {node_id} 的 tools[{i}] 缺少 provider_name/tool_name")
                    continue
                if self.allowed_tools and (provider_name, tool_name) not in self.allowed_tools:
                    self.errors.append(
                        f"E054: Agent节点 {node_id} 使用了未在 config/tools.json 声明的工具: provider_name={provider_name}, tool_name={tool_name}"
                    )

    def _check_knowledge_retrieval_schema(self) -> None:
        for node_id, node in self.nodes.items():
            data = node.get("data") or {}
            if data.get("type") != "knowledge-retrieval":
                continue

            dataset_ids = data.get("dataset_ids")
            if dataset_ids is None:
                dataset_ids = []
            if not isinstance(dataset_ids, list):
                self.errors.append(f"E060: Knowledge Retrieval节点 {node_id} 的 dataset_ids 必须是数组")
                continue

            if dataset_ids and self._knowledge_bases_config.get("present") and self._knowledge_bases_config.get("count", 0) == 0:
                self.errors.append(
                    "E061: config/knowledge_bases.json 中 knowledge_bases 为空；无法为 knowledge-retrieval 选择 dataset_ids（请先同步配置）"
                )
                continue

            for ds in dataset_ids:
                ds_id = str(ds)
                if ds_id.startswith("__") and ds_id.endswith("__"):
                    self.warnings.append(
                        f"W020: Knowledge Retrieval节点 {node_id} 使用了占位 dataset_id: {ds_id}（请替换为实际知识库ID）"
                    )
                    continue
                if self.allowed_dataset_ids and ds_id not in self.allowed_dataset_ids:
                    self.errors.append(
                        f"E062: Knowledge Retrieval节点 {node_id} 使用了未在 config/knowledge_bases.json 声明的 dataset_id: {ds_id}"
                    )


def validate_workflow_file(filepath: str) -> Tuple[bool, List[str], List[str]]:
    workflow = _load_workflow_dsl(Path(filepath))
    validator = WorkflowValidator(workflow)
    return validator.validate()


def main() -> int:
    parser = argparse.ArgumentParser(description="验证Dify Workflow DSL（JSON/YAML）")
    parser.add_argument("file", help="Workflow DSL文件路径（.json/.yml/.yaml）")
    parser.add_argument("--strict", action="store_true", help="严格模式（warnings 也视为失败）")
    args = parser.parse_args()

    try:
        is_valid, errors, warnings = validate_workflow_file(args.file)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {args.file}")
        return 1
    except ValueError as exc:
        print(f"❌ 解析失败: {exc}")
        return 1

    if errors:
        print("❌ 验证失败:")
        for err in errors:
            print(f"   [ERROR] {err}")

    if warnings:
        print("⚠️  警告:")
        for warn in warnings:
            print(f"   [WARN] {warn}")

    if is_valid and not (args.strict and warnings):
        print("✅ 验证通过")
        return 0

    print("❌ 验证未通过")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
