# Dify Workflow DSL 完整规范

## 1. 顶层结构（YAML/JSON）

Dify 的 App DSL 在后端以 **YAML** 为主（导出即为 YAML），但 **JSON 也是合法的 YAML 1.2 子集**，因此 skill 常以 JSON 输出同构结构。

```json
{
  "version": "0.5.0",
  "kind": "app",
  "app": {
    "description": "",
    "icon": "🤖",
    "icon_background": "#FFEAD5",
    "mode": "workflow",
    "name": "应用名称",
    "use_icon_as_answer_icon": false
  },
  "dependencies": [],
  "workflow": {
    "conversation_variables": [],
    "environment_variables": [],
    "features": {
      "file_upload": { "enabled": false },
      "retriever_resource": { "enabled": true }
    },
    "graph": {
      "nodes": [],
      "edges": [],
      "viewport": { "x": 0, "y": 0, "zoom": 1.0 }
    }
  }
}
```

## 2. 节点通用结构

```json
{
  "id": "start_1",
  "type": "custom",
  "position": { "x": 80, "y": 282 },
  "positionAbsolute": { "x": 80, "y": 282 },
  "width": 244,
  "height": 90,
  "sourcePosition": "right",
  "targetPosition": "left",
  "selected": false,
  "data": {
    "type": "start",
    "title": "节点标题",
    "desc": ""
  }
}

```

**ID生成规则**:
- 后端只要求“全局唯一字符串”，既可以用语义化ID（`start_1`/`llm_main`），也可以用时间戳。
- 若使用时间戳，建议毫秒级字符串（如 `"1755502773326"`），避免重复。

**位置计算**: 
- 起始X=80，每个节点水平间距约300
- Y坐标通常保持一致（约282），分支时上下偏移150
**节点类型**:
- 节点的“业务类型”在 `node.data.type`（如 `llm` / `knowledge-retrieval` / `if-else`）。
- `node.type` 通常固定为 `"custom"`（画布节点类型），少数注释节点会用 `"custom-note"`。
---

## 3. 完整节点Schema

### 3.1 Start 节点

```json
{
  "data": {
    "type": "start",
    "title": "开始",
    "desc": "",
    "variables": [
      {
        "label": "用户输入",
        "variable": "query",
        "type": "text-input",
        "required": true,
        "max_length": 2000,
        "options": [],
        "default": ""
      }
    ]
  }
}
```

**variables.type可选值**:
- `text-input`: 单行文本
- `paragraph`: 多行文本
- `select`: 下拉选择（需配置options）
- `number`: 数字
- `file`: 单文件
- `file-list`: 多文件

> 说明：Dify 也支持纯 Trigger 驱动的工作流（如 `trigger-webhook` / `trigger-schedule` 等），此时可以没有 Start 节点；但 Start 与 Trigger 节点不能共存（后端会拒绝）。

### 3.2 End 节点

```json
{
  "data": {
    "type": "end",
    "title": "结束",
    "desc": "",
    "outputs": [
      {
        "variable": "result",
        "value_selector": ["llm_node_id", "text"]
      }
    ]
  }
}
```

### 3.3 Answer 节点 (仅Chatflow)

```json
{
  "data": {
    "type": "answer",
    "title": "回答",
    "desc": "",
    "answer": "{{#llm_node_id.text#}}"
  }
}
```

### 3.4 LLM 节点

可用模型以 `../config/models.json` 为准。

```json
{
  "data": {
    "type": "llm",
    "title": "LLM",
    "desc": "",
    "model": {
      "provider": "langgenius/moonshot/moonshot",
      "name": "kimi-k2-turbo-preview",
      "mode": "chat",
      "completion_params": {
        "temperature": 0
      }
    },
    "prompt_template": [
      {
        "id": "system_prompt_id",
        "role": "system",
        "text": "你是一个专业的助手。"
      },
      {
        "id": "user_prompt_id", 
        "role": "user",
        "text": "{{#start.query#}}"
      }
    ],
    "context": {
      "enabled": false,
      "variable_selector": []
    },
    "memory": {
      "enabled": false,
      "role_prefix": { "assistant": "", "user": "" },
      "window": { "enabled": false, "size": 10 }
    },
    "vision": {
      "enabled": false,
      "configs": { "variable_selector": [] }
    }
  }
}
```

**运行时必填字段**（对齐后端 `LLMNodeData`，缺失会导致导入/运行时报错）:
- `data.model.provider` / `data.model.name` / `data.model.mode`
- `data.prompt_template[]` 每项必须包含 `role` / `text`
- `data.context.enabled`（`context` 对象在后端 schema 中为必填）

**推荐字段**（不影响运行，但建议显式写出，减少默认行为差异）:
- `data.model.completion_params`（如 `temperature` / `max_tokens` / `top_p`）
- `data.vision` / `data.memory`
- `prompt_template[].id`（不影响运行，但可能影响前端编辑体验）

**常见错误（禁止写法）**:
- 不要把 `temperature` / `max_tokens` 写在 `data` 顶层；必须写到 `data.model.completion_params` 内
  - 说明：后端不要求 `prompt_template[].id`，但某些前端编辑场景可能更依赖它


### 3.5 Knowledge Retrieval 节点

可用知识库（`dataset_ids`）以 `../config/knowledge_bases.json` 为准。

```json
{
  "data": {
    "type": "knowledge-retrieval",
    "title": "知识检索",
    "desc": "",
    "query_variable_selector": ["start", "query"],
    "dataset_ids": ["dataset_uuid_here"],
    "retrieval_mode": "multiple",
    "single_retrieval_config": {
      "model": {
        "provider": "openai",
        "name": "gpt-4o-mini",
        "mode": "chat",
        "completion_params": { "temperature": 0 }
      }
    },
    "multiple_retrieval_config": {
      "top_k": 5,
      "score_threshold": 0.5,
      "score_threshold_enabled": true,
      "reranking_enable": false,
      "reranking_model": {
        "provider": "",
        "model": ""
      }
    }
  }
}
```

**输出变量**: 
- `result`: array[object] - 检索结果列表
- `context`: string - 拼接后的上下文文本

### 3.6 Question Classifier 节点

```json
{
  "data": {
    "type": "question-classifier",
    "title": "问题分类",
    "desc": "",
    "query_variable_selector": ["start", "query"],
    "model": {
      "provider": "openai",
      "name": "gpt-4o-mini",
      "mode": "chat",
      "completion_params": { "temperature": 0 }
    },
    "classes": [
      {
        "id": "tech_class",
        "name": "技术问题",
        "description": "关于技术、编程、系统的问题"
      },
      {
        "id": "biz_class", 
        "name": "业务问题",
        "description": "关于业务流程、产品功能的问题"
      },
      {
        "id": "other_class",
        "name": "其他",
        "description": "无法归类的问题"
      }
    ],
    "instruction": ""
  }
}
```

**重要**: 每个class的id将作为出边的sourceHandle

### 3.7 IF/ELSE 节点

```json
{
  "id": "if_1",
  "type": "custom",
  "position": { "x": 380, "y": 282 },
  "data": {
    "type": "if-else",
    "title": "条件判断",
    "desc": "",
    "cases": [
      {
        "case_id": "true",
        "logical_operator": "and",
        "conditions": [
          {
            "variable_selector": ["llm_1", "text"],
            "comparison_operator": "contains",
            "value": "成功"
          }
        ]
      }
    ]
  }
}

```

**comparison_operator可选值**:
- 字符串/数组: `contains`, `not contains`, `start with`, `end with`, `is`, `is not`, `empty`, `not empty`, `in`, `not in`, `all of`
- 数值: `=`, `≠`, `>`, `<`, `≥`, `≤`
- 空值: `null`, `not null`
- 文件: `exists`, `not exists`

### 3.8 Code 节点

```json
{
  "data": {
    "type": "code",
    "title": "代码处理",
    "desc": "",
    "code_language": "python3",
    "code": "def main(input_text: str) -> dict:\n    result = input_text.upper()\n    return {\"output\": result}",
    "variables": [
      { "variable": "input_text", "value_selector": ["start_1", "query"] }
    ],
    "outputs": {
      "output": { "type": "string", "children": null }
    }
  }
}
```

**outputs[*].type可选值**: `string`, `number`, `boolean`, `object`, `array[string]`, `array[number]`, `array[object]`, `array[boolean]`

### 3.9 Template Transform 节点

```json
{
  "data": {
    "type": "template-transform",
    "title": "模板转换",
    "desc": "",
    "template": "处理结果:\n{{ result }}\n\n来源: {{ source }}",
    "variables": [
      { "variable": "result", "value_selector": ["llm", "text"] },
      { "variable": "source", "value_selector": ["kr", "context"] }
    ]
  }
}
```

**输出**: `output` (string)

### 3.10 Iteration 节点

```json
{
  "data": {
    "type": "iteration",
    "title": "迭代处理",
    "desc": "",
    "iterator_selector": ["extractor_1", "items"],
    "output_selector": ["inner_llm_1", "text"],
    "is_parallel": true,
    "parallel_nums": 5,
    "error_handle_mode": "continue-on-error",
    "flatten_output": true
  }
}
```

**error_handle_mode**:
- `terminated`: 遇错停止
- `continue-on-error`: 跳过错误项，输出null
- `remove-abnormal-output`: 跳过错误项，不输出

**内置变量**: `{{#iteration_node_id.item#}}`, `{{#iteration_node_id.index#}}`

**禁止在内部使用**: Answer, Variable Assigner, Tool

### 3.11 HTTP Request 节点

```json
{
  "data": {
    "type": "http-request",
    "title": "HTTP请求",
    "desc": "",
    "method": "POST",
    "url": "https://api.example.com/v1/chat",
    "headers": "{\n  \"Content-Type\": \"application/json\",\n  \"Authorization\": \"Bearer {{#env.API_KEY#}}\"\n}",
    "params": "",
    "body": {
      "type": "json",
      "data": "{\n  \"query\": \"{{#start.query#}}\"\n}"
    },
    "authorization": {
      "type": "no-auth",
      "config": null
    },
    "timeout": {
      "connect": 10,
      "read": 60,
      "write": 10
    },
    "mask_authorization_header": true
  }
}
```

**method**: `GET`, `POST`, `PUT`, `PATCH`, `DELETE`

**body.type**: `none`, `form-data`, `x-www-form-urlencoded`, `raw-text`, `json`, `binary`

**输出**: `status_code`, `body`, `headers`

### 3.12 Agent 节点

```json
{
  "data": {
    "type": "agent",
    "title": "智能体",
    "desc": "",
    "agent_strategy_provider_name": "langgenius/agent/agent",
    "agent_strategy_name": "ReAct",
    "agent_strategy_label": "ReAct",
    "agent_parameters": {
      "model": {
        "type": "constant",
        "value": {
          "provider": "openai",
          "name": "gpt-4o",
          "mode": "chat",
          "completion_params": { "temperature": 0 }
        }
      },
      "tools": {
        "type": "constant",
        "value": [
          {
            "provider_name": "websearch",
            "tool_name": "web_search",
            "enabled": true,
            "parameters": {}
          }
        ]
      },
      "instruction": {
        "type": "constant",
        "value": "你是一个搜索助手，帮助用户查找信息。"
      },
      "query": {
        "type": "variable",
        "value": "{{#start.query#}}"
      },
      "max_iterations": {
        "type": "constant",
        "value": 5
      }
    },
    "memory": {
      "enabled": false,
      "window": { "enabled": false, "size": 10 }
    }
  }
}
```

可用工具清单以 `../config/tools.json` 为准。

**常用策略**: `ReAct`, `function_calling`

### 3.13 Parameter Extractor 节点

```json
{
  "data": {
    "type": "parameter-extractor",
    "title": "参数提取",
    "desc": "",
    "query": ["start", "query"],
    "model": {
      "provider": "openai",
      "name": "gpt-4o-mini",
      "mode": "chat",
      "completion_params": { "temperature": 0 }
    },
    "parameters": [
      {
        "name": "order_id",
        "type": "string",
        "description": "订单编号，格式如ORD-12345",
        "required": true
      },
      {
        "name": "items",
        "type": "array[string]",
        "description": "商品名称列表",
        "required": false
      }
    ],
    "instruction": "从用户输入中提取订单信息",
    "reasoning_mode": "function_call"
  }
}
```

### 3.14 Variable Aggregator 节点

```json
{
  "data": {
    "type": "variable-aggregator",
    "title": "变量聚合",
    "desc": "",
    "variables": [
      ["branch1_llm", "text"],
      ["branch2_llm", "text"]
    ],
    "output_type": "string",
    "advanced_settings": {
      "group_enabled": false
    }
  }
}
```

**约束**: 只能聚合相同类型的变量

### 3.15 List Operator 节点

```json
{
  "data": {
    "type": "list-operator",
    "title": "列表操作",
    "desc": "",
    "variable": ["node_id", "array_var"],
    "filter_by": {
      "enabled": true,
      "conditions": [
        {
          "key": "type",
          "comparison_operator": "is",
          "value": "document"
        }
      ]
    },
    "order_by": {
      "enabled": true,
      "key": "created_at",
      "value": "desc"
    },
    "limit": {
      "enabled": true,
      "size": 10
    },
    "extract_by": {
      "enabled": false,
      "serial": "1"
    }
  }
}
```

### 3.16 Document Extractor 节点

```json
{
  "data": {
    "type": "document-extractor",
    "title": "文档提取",
    "desc": "",
    "variable_selector": ["start_1", "ref_file"]
  }
}
```

---

## 4. 边(Edge)结构

```json
{
  "id": "source_id-sourceHandle-target_id-targetHandle",
  "source": "source_node_id",
  "sourceHandle": "source",
  "target": "target_node_id",
  "targetHandle": "target",
  "type": "custom",
  "zIndex": 0,
  "data": {
    "isInIteration": false,
    "sourceType": "start",
    "targetType": "llm"
  }
}
```

**sourceHandle规则**:
| 源节点类型 | sourceHandle值 |
|------------|----------------|
| 普通节点 | `"source"` |
| IF/ELSE (条件满足) | `"true"` |
| IF/ELSE (条件不满足) | `"false"` |
| Question Classifier | 对应class的id |
| 错误处理分支 | `"fail-branch"` |

---

## 5. 硬约束清单

| 编号 | 约束 | 验证方法 |
|------|------|----------|
| HC-1 | 有且仅有1个Start节点 | 统计type=start的节点数 |
| HC-2 | Start节点无入边 | 检查无edge.target=start_id |
| HC-3 | End节点无出边 | 检查无edge.source=end_id |
| HC-4 | Workflow必须有End节点 | mode=workflow时检查 |
| HC-5 | Chatflow必须有Answer节点 | mode=chat时检查 |
| HC-6 | 图是DAG（无环） | 拓扑排序检测 |
| HC-7 | 所有节点从Start可达 | BFS/DFS遍历 |
| HC-8 | 所有节点可达End/Answer | 反向遍历 |
| HC-9 | 节点ID唯一 | Set检测 |
| HC-10 | 边ID唯一 | Set检测 |
| HC-11 | 变量引用有效 | 引用的node_id存在且在上游 |
| HC-12 | IF/ELSE有true和false两条出边 | 统计出边handle |
| HC-13 | Question Classifier每个class有出边 | 对比classes和出边 |
| HC-14 | Iteration输入是数组 | 检查上游输出类型 |
| HC-15 | Iteration内无禁止节点 | 检查内部节点类型 |

---

## 6. 变量系统

### 引用语法
```
{{#node_id.variable_name#}}
{{#sys.query#}}
{{#sys.files#}}
{{#sys.user_id#}}
{{#sys.conversation_id#}}
{{#env.VARIABLE_NAME#}}
```

### 系统变量
| 变量 | 类型 | 说明 | 适用 |
|------|------|------|------|
| sys.query | string | 用户输入 | Chatflow |
| sys.files | array[file] | 上传文件 | 全部 |
| sys.user_id | string | 用户ID | 全部 |
| sys.conversation_id | string | 会话ID | Chatflow |
| sys.dialogue_count | number | 对话轮次 | Chatflow |

### 节点输出类型速查
| 节点类型 | 输出变量 | 类型 |
|----------|----------|------|
| llm | text | string |
| llm | usage | object |
| knowledge-retrieval | result | array[object] |
| knowledge-retrieval | context | string |
| code | 自定义 | 自定义 |
| template | output | string |
| http-request | status_code | number |
| http-request | body | string/object |
| http-request | headers | object |
| parameter-extractor | 各参数名 | 对应类型 |
| iteration | output | array[...] |
| list-operator | result | array |
| list-operator | first_record | any |
| list-operator | last_record | any |
| document-extractor | text | string |
| agent | text | string |

---

## 7. 错误代码

错误码以 `scripts/validate_workflow.py` 的输出为准（会随 Dify 后端 schema 演进而更新）；本文不再维护一份静态对照表，避免过期误导。
