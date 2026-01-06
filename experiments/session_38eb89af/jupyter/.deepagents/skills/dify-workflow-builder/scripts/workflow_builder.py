#!/usr/bin/env python3
"""
Dify Workflow Builder - 核心构建模块
提供节点创建、边连接、完整Workflow组装的功能
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class NodeType(Enum):
    START = "start"
    END = "end"
    ANSWER = "answer"
    LLM = "llm"
    KNOWLEDGE_RETRIEVAL = "knowledge-retrieval"
    QUESTION_CLASSIFIER = "question-classifier"
    IF_ELSE = "if-else"
    CODE = "code"
    TEMPLATE = "template"
    ITERATION = "iteration"
    HTTP_REQUEST = "http-request"
    AGENT = "agent"
    PARAMETER_EXTRACTOR = "parameter-extractor"
    VARIABLE_AGGREGATOR = "variable-aggregator"

@dataclass
class Position:
    x: int = 80
    y: int = 282

class IDGenerator:
    """ID生成器，确保唯一性"""
    _last_ts = 0
    
    @classmethod
    def generate(cls) -> str:
        ts = int(time.time() * 1000)
        if ts <= cls._last_ts:
            ts = cls._last_ts + 1
        cls._last_ts = ts
        return str(ts)

class NodeBuilder:
    """节点构建器"""
    
    @staticmethod
    def _base_node(node_type: str, title: str, pos: Position) -> Dict:
        return {
            "id": IDGenerator.generate(),
            "type": "custom",
            "position": {"x": pos.x, "y": pos.y},
            "positionAbsolute": {"x": pos.x, "y": pos.y},
            "width": 243,
            "height": 88,
            "sourcePosition": "right",
            "targetPosition": "left",
            "selected": False,
            "data": {
                "type": node_type,
                "title": title,
                "desc": ""
            }
        }
    
    @classmethod
    def start(cls, variables: List[Dict] = None, pos: Position = None) -> Dict:
        pos = pos or Position(80, 282)
        node = cls._base_node("start", "开始", pos)
        node["data"]["variables"] = variables or [
            {"label": "输入", "variable": "query", "type": "paragraph", "required": True, "max_length": 2000}
        ]
        return node
    
    @classmethod
    def end(cls, outputs: List[Dict] = None, pos: Position = None) -> Dict:
        pos = pos or Position(680, 282)
        node = cls._base_node("end", "结束", pos)
        node["data"]["outputs"] = outputs or []
        return node
    
    @classmethod
    def answer(cls, answer_text: str = "", pos: Position = None) -> Dict:
        pos = pos or Position(680, 282)
        node = cls._base_node("answer", "回复", pos)
        node["data"]["answer"] = answer_text
        return node
    
    @classmethod
    def llm(cls, title: str = "LLM", 
            system_prompt: str = "你是一个有帮助的助手。",
            user_prompt: str = "",
            model_provider: str = "langgenius/moonshot/moonshot",
            model_name: str = "kimi-k2-turbo-preview",
            temperature: float = 0,
            max_tokens: Optional[int] = None,
            context_enabled: bool = False,
            context_selector: List[str] = None,
            memory_enabled: bool = False,
            memory_size: int = 10,
            pos: Position = None) -> Dict:
        pos = pos or Position(380, 282)
        node = cls._base_node("llm", title, pos)
        node["data"].update({
            "model": {
                "provider": model_provider, "name": model_name, "mode": "chat",
                "completion_params": {
                    "temperature": temperature,
                    **({} if max_tokens is None else {"max_tokens": max_tokens}),
                    "top_p": 1,
                },
            },
            "prompt_template": [
                {"id": IDGenerator.generate(), "role": "system", "text": system_prompt},
                {"id": IDGenerator.generate(), "role": "user", "text": user_prompt}
            ],
            "context": {"enabled": context_enabled, "variable_selector": context_selector or []},
            "memory": {"enabled": memory_enabled, "role_prefix": {"assistant": "", "user": ""}, 
                      "window": {"enabled": memory_enabled, "size": memory_size}},
            "vision": {"enabled": False}
        })
        return node
    
    @classmethod
    def knowledge_retrieval(cls, title: str = "知识检索",
                           query_selector: List[str] = None,
                           dataset_ids: List[str] = None,
                           top_k: int = 5, score_threshold: float = 0.5,
                           pos: Position = None) -> Dict:
        pos = pos or Position(380, 282)
        node = cls._base_node("knowledge-retrieval", title, pos)
        node["data"].update({
            "query_variable_selector": query_selector or ["start", "query"],
            "dataset_ids": dataset_ids or [],
            "retrieval_mode": "multiple",
            "multiple_retrieval_config": {"top_k": top_k, "score_threshold": score_threshold, "score_threshold_enabled": True}
        })
        return node
    
    @classmethod
    def question_classifier(cls, title: str = "问题分类",
                           query_selector: List[str] = None,
                           classes: List[Dict] = None,
                           model_provider: str = "langgenius/moonshot/moonshot",
                           model_name: str = "kimi-k2-turbo-preview",
                           pos: Position = None) -> Dict:
        pos = pos or Position(380, 282)
        node = cls._base_node("question-classifier", title, pos)
        node["data"].update({
            "query_variable_selector": query_selector or ["start", "query"],
            "model": {"provider": model_provider, "name": model_name, "mode": "chat", "completion_params": {"temperature": 0}},
            "classes": classes or [{"id": "class_1", "name": "分类1"}, {"id": "class_2", "name": "分类2"}],
            "instruction": ""
        })
        return node
    
    @classmethod
    def if_else(cls, title: str = "条件判断", conditions: List[Dict] = None, pos: Position = None) -> Dict:
        pos = pos or Position(380, 282)
        node = cls._base_node("if-else", title, pos)
        node["data"]["conditions"] = conditions or [{"id": IDGenerator.generate(), "logical_operator": "and", "conditions": []}]
        return node
    
    @classmethod
    def code(cls, title: str = "代码处理", language: str = "python3",
            code: str = "def main():\n    return {}", variables: List[Dict] = None,
            outputs: List[Dict] = None, pos: Position = None) -> Dict:
        pos = pos or Position(380, 282)
        node = cls._base_node("code", title, pos)
        node["data"].update({
            "code_language": language, "code": code,
            "variables": variables or [],
            "outputs": outputs or [{"variable": "result", "type": "string"}]
        })
        return node
    
    @classmethod
    def template(cls, title: str = "模板转换", template_text: str = "",
                variables: List[Dict] = None, pos: Position = None) -> Dict:
        pos = pos or Position(380, 282)
        node = cls._base_node("template", title, pos)
        node["data"].update({"template": template_text, "variables": variables or []})
        return node
    
    @classmethod
    def iteration(cls, title: str = "迭代处理", iterator_selector: List[str] = None,
                 output_selector: List[str] = None, is_parallel: bool = True,
                 parallel_nums: int = 5, error_mode: str = "continue-on-error",
                 pos: Position = None) -> Dict:
        pos = pos or Position(380, 282)
        node = cls._base_node("iteration", title, pos)
        node["data"].update({
            "iterator_selector": iterator_selector or [], "output_selector": output_selector or [],
            "output_type": "array[string]", "is_parallel": is_parallel,
            "parallel_nums": parallel_nums, "error_handle_mode": error_mode
        })
        return node
    
    @classmethod
    def http_request(cls, title: str = "HTTP请求", method: str = "GET", url: str = "",
                    headers: str = "{}", body_type: str = "none", body_data: str = "",
                    pos: Position = None) -> Dict:
        pos = pos or Position(380, 282)
        node = cls._base_node("http-request", title, pos)
        node["data"].update({
            "method": method, "url": url, "headers": headers, "params": "",
            "body": {"type": body_type, "data": body_data},
            "timeout": {"connect": 10, "read": 60, "write": 10}
        })
        return node
    
    @classmethod
    def agent(cls, title: str = "智能体", strategy: str = "ReAct",
             model_provider: str = "langgenius/moonshot/moonshot", model_name: str = "kimi-k2-turbo-preview",
             instruction: str = "", query_ref: str = "", tools: List[Dict] = None,
             max_iterations: int = 10, pos: Position = None) -> Dict:
        pos = pos or Position(380, 282)
        node = cls._base_node("agent", title, pos)
        node["data"].update({
            "agent_strategy_provider_name": "langgenius/agent/agent",
            "agent_strategy_name": strategy,
            "agent_parameters": {
                "model": {
                    "type": "constant",
                    "value": {
                        "provider": model_provider,
                        "name": model_name,
                        "mode": "chat",
                        "completion_params": {"temperature": 0},
                    },
                },
                "tools": {"type": "constant", "value": tools or []},
                "instruction": {"type": "constant", "value": instruction},
                "query": {"type": "variable", "value": query_ref},
                "max_iterations": {"type": "constant", "value": max_iterations}
            }
        })
        return node
    
    @classmethod
    def parameter_extractor(cls, title: str = "参数提取", query_selector: List[str] = None,
                           parameters: List[Dict] = None, instruction: str = "",
                           pos: Position = None) -> Dict:
        pos = pos or Position(380, 282)
        node = cls._base_node("parameter-extractor", title, pos)
        node["data"].update({
            "query": query_selector or ["start", "query"],
            "model": {
                "provider": "langgenius/moonshot/moonshot",
                "name": "kimi-k2-turbo-preview",
                "mode": "chat",
                "completion_params": {"temperature": 0},
            },
            "parameters": parameters or [],
            "instruction": instruction
        })
        return node
    
    @classmethod
    def variable_aggregator(cls, title: str = "变量聚合", variables: List[List[str]] = None,
                           output_type: str = "string", pos: Position = None) -> Dict:
        pos = pos or Position(380, 282)
        node = cls._base_node("variable-aggregator", title, pos)
        node["data"].update({"variables": variables or [], "output_type": output_type})
        return node


class EdgeBuilder:
    @staticmethod
    def create(source_id: str, target_id: str, source_handle: str = "source",
               source_type: str = "", target_type: str = "") -> Dict:
        return {
            "id": f"{source_id}-{source_handle}-{target_id}-target",
            "source": source_id, "sourceHandle": source_handle,
            "target": target_id, "targetHandle": "target",
            "type": "custom", "zIndex": 0,
            "data": {"isInIteration": False, "sourceType": source_type, "targetType": target_type}
        }


class WorkflowBuilder:
    """Workflow构建器"""
    
    def __init__(self, name: str, mode: str = "workflow", description: str = ""):
        self.name = name
        self.mode = mode
        self.description = description
        self.nodes: List[Dict] = []
        self.edges: List[Dict] = []
        self._node_map: Dict[str, Dict] = {}
    
    def add_node(self, node: Dict, alias: str = None) -> str:
        self.nodes.append(node)
        self._node_map[alias or node["id"]] = node
        return node["id"]
    
    def add_edge(self, source: str, target: str, source_handle: str = "source") -> None:
        src_node = self._node_map.get(source, {"id": source})
        tgt_node = self._node_map.get(target, {"id": target})
        src_id, tgt_id = src_node.get("id", source), tgt_node.get("id", target)
        src_type = src_node.get("data", {}).get("type", "")
        tgt_type = tgt_node.get("data", {}).get("type", "")
        self.edges.append(EdgeBuilder.create(src_id, tgt_id, source_handle, src_type, tgt_type))
    
    def get_node_id(self, alias: str) -> str:
        node = self._node_map.get(alias)
        return node["id"] if node else alias
    
    def var_ref(self, node_alias: str, var_name: str) -> str:
        return f"{{{{#{self.get_node_id(node_alias)}.{var_name}#}}}}"
    
    def var_selector(self, node_alias: str, var_name: str) -> List[str]:
        return [self.get_node_id(node_alias), var_name]
    
    def build(self) -> Dict:
        return {
            "app": {"description": self.description, "icon": "🤖", "icon_background": "#FFEAD5",
                   "mode": self.mode, "name": self.name, "use_icon_as_answer_icon": False},
            "dependencies": [], "kind": "app", "version": "0.1.5",
            "workflow": {
                "conversation_variables": [], "environment_variables": [],
                "features": {"file_upload": {"enabled": False}, "retriever_resource": {"enabled": True}},
                "graph": {"edges": self.edges, "nodes": self.nodes, "viewport": {"x": 0, "y": 0, "zoom": 1.0}}
            }
        }
    
    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.build(), f, ensure_ascii=False, indent=2)
        print(f"✅ Workflow已保存: {filepath}")


# ==================== 快速构建函数 ====================

def build_simple_qa(name: str = "简单问答", system_prompt: str = "你是一个有帮助的助手。") -> Dict:
    wb = WorkflowBuilder(name, "workflow")
    start = NodeBuilder.start(pos=Position(80, 282))
    wb.add_node(start, "start")
    llm = NodeBuilder.llm(title="AI回答", system_prompt=system_prompt,
                         user_prompt=wb.var_ref("start", "query"), pos=Position(380, 282))
    wb.add_node(llm, "llm")
    end = NodeBuilder.end(outputs=[{"variable": "answer", "value_selector": wb.var_selector("llm", "text")}],
                         pos=Position(680, 282))
    wb.add_node(end, "end")
    wb.add_edge("start", "llm")
    wb.add_edge("llm", "end")
    return wb.build()


def build_rag_qa(name: str = "知识库问答", dataset_ids: List[str] = None) -> Dict:
    wb = WorkflowBuilder(name, "workflow", "基于知识库的问答系统")
    start = NodeBuilder.start(pos=Position(80, 282))
    wb.add_node(start, "start")
    kr = NodeBuilder.knowledge_retrieval(query_selector=wb.var_selector("start", "query"),
                                        dataset_ids=dataset_ids or ["__DATASET_ID__"], pos=Position(380, 282))
    wb.add_node(kr, "kr")
    sys_prompt = f"基于以下参考资料回答问题。如无相关信息请如实告知。\n\n参考资料：\n{wb.var_ref('kr', 'context')}"
    llm = NodeBuilder.llm(title="生成回答", system_prompt=sys_prompt, user_prompt=wb.var_ref("start", "query"),
                         context_enabled=True, context_selector=wb.var_selector("kr", "result"), pos=Position(680, 282))
    wb.add_node(llm, "llm")
    end = NodeBuilder.end(outputs=[{"variable": "answer", "value_selector": wb.var_selector("llm", "text")}],
                         pos=Position(980, 282))
    wb.add_node(end, "end")
    wb.add_edge("start", "kr")
    wb.add_edge("kr", "llm")
    wb.add_edge("llm", "end")
    return wb.build()


def build_intent_classifier(name: str = "意图分类",
                           classes: List[Tuple[str, str, str]] = None) -> Dict:
    wb = WorkflowBuilder(name, "workflow", "根据意图分类处理")
    if classes is None:
        classes = [("tech", "技术问题", "你是技术专家。"), ("biz", "业务问题", "你是业务顾问。"), ("other", "其他", "你是通用助手。")]
    
    start = NodeBuilder.start(pos=Position(80, 282))
    wb.add_node(start, "start")
    
    classifier_classes = [{"id": c[0], "name": c[1], "description": ""} for c in classes]
    classifier = NodeBuilder.question_classifier(query_selector=wb.var_selector("start", "query"),
                                                 classes=classifier_classes, pos=Position(300, 282))
    wb.add_node(classifier, "classifier")
    wb.add_edge("start", "classifier")
    
    llm_nodes = []
    for i, (class_id, class_name, sys_prompt) in enumerate(classes):
        llm = NodeBuilder.llm(title=class_name, system_prompt=sys_prompt,
                             user_prompt=wb.var_ref("start", "query"), pos=Position(550, 150 + i * 150))
        wb.add_node(llm, f"llm_{class_id}")
        wb.add_edge("classifier", f"llm_{class_id}", source_handle=class_id)
        llm_nodes.append(f"llm_{class_id}")
    
    agg = NodeBuilder.variable_aggregator(variables=[wb.var_selector(n, "text") for n in llm_nodes], pos=Position(800, 282))
    wb.add_node(agg, "agg")
    for n in llm_nodes:
        wb.add_edge(n, "agg")
    
    end = NodeBuilder.end(outputs=[{"variable": "answer", "value_selector": wb.var_selector("agg", "output")}], pos=Position(1050, 282))
    wb.add_node(end, "end")
    wb.add_edge("agg", "end")
    return wb.build()


def build_chatflow(name: str = "智能对话", system_prompt: str = "你是一个友好的对话助手。", memory_size: int = 20) -> Dict:
    wb = WorkflowBuilder(name, "chat", "支持多轮对话")
    start = NodeBuilder.start(variables=[], pos=Position(80, 282))
    wb.add_node(start, "start")
    llm = NodeBuilder.llm(title="对话", system_prompt=system_prompt, user_prompt="{{#sys.query#}}",
                         memory_enabled=True, memory_size=memory_size, pos=Position(380, 282))
    wb.add_node(llm, "llm")
    answer = NodeBuilder.answer(answer_text=wb.var_ref("llm", "text"), pos=Position(680, 282))
    wb.add_node(answer, "answer")
    wb.add_edge("start", "llm")
    wb.add_edge("llm", "answer")
    return wb.build()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dify Workflow 快速构建")
    parser.add_argument("--template", choices=["simple", "rag", "classifier", "chat"], required=True)
    parser.add_argument("--name", default="My Workflow")
    parser.add_argument("--output", default="workflow.json")
    parser.add_argument("--dataset-ids", nargs="+")
    args = parser.parse_args()
    
    if args.template == "simple":
        workflow = build_simple_qa(args.name)
    elif args.template == "rag":
        workflow = build_rag_qa(args.name, args.dataset_ids)
    elif args.template == "classifier":
        workflow = build_intent_classifier(args.name)
    elif args.template == "chat":
        workflow = build_chatflow(args.name)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(workflow, f, ensure_ascii=False, indent=2)
    print(f"✅ Workflow已创建: {args.output}")
