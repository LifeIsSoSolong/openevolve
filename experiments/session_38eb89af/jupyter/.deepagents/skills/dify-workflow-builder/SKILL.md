---
name: dify-workflow-builder
description: "自动构建高质量Dify Workflow并进行效果测试与迭代的专业技能。当用户需要：(1)根据需求自动生成Dify Workflow JSON，(2)构建复杂的多步骤AI工作流，(3)设计RAG、Agent、条件分支、迭代等高级流程。"
---

# Dify Workflow Builder

自动构建高质量的Dify Workflow JSON，并引导用户直接在本技能中进行效果测试与迭代

## 环境配置

为避免把“可用模型/工具/知识库”硬编码在技能文本里，本技能将它们拆分为独立配置文件。

为避免 agent 执行命令时 `cwd` 不确定导致相对路径失效，本技能中涉及的脚本/配置/参考文件路径统一使用占位符 `<SKILL_ROOT>`：

- `<SKILL_ROOT>` = 本技能目录的绝对路径（即当前 `SKILL.md` 所在目录）
- 在执行命令前，agent 必须把 `<SKILL_ROOT>` 替换为该 skill 的真实路径
- 可用模型清单：`<SKILL_ROOT>/config/models.json`
- 可用工具清单：`<SKILL_ROOT>/config/tools.json`
- 可用知识库清单：`<SKILL_ROOT>/config/knowledge_bases.json`（用于填充 `knowledge-retrieval.dataset_ids`）

### 输出目录（强制）

本技能的**所有输出文本**（最终 Workflow JSON、校验结果摘要等）必须写入环境变量 `EVO_OUTPUT_DIR` 指定的目录。

在开始构建/输出前，agent **必须先读取并确认**该环境变量（并确保目录存在）：
```bash
echo "$EVO_OUTPUT_DIR"
test -n "$EVO_OUTPUT_DIR" || { echo "❌ Missing EVO_OUTPUT_DIR"; exit 1; }
mkdir -p "$EVO_OUTPUT_DIR"
```

必须先同步dify的最新信息到配置文件：
```bash
python "<SKILL_ROOT>/scripts/sync_dify_config.py"
```

使用要求：
- 设计/生成 Workflow 前，先查看上述配置文件，必须从清单中选择 `model.provider/name/mode`、tools、dataset ids。
- 若配置为空，则证明当前没有可用的对应的资源。

## 强制约束（避免Dify导入/运行报错，必须要完成的内容，不能跳过）

在输出最终 Workflow JSON 前，必须确保所有 `llm` 节点满足：
- `data.model` 必须包含 `provider/name/mode/completion_params`，且 `provider+name` 必须来自 `<SKILL_ROOT>/config/models.json`
- 禁止使用 `data.temperature` / `data.max_tokens` 顶层字段（必须放进 `data.model.completion_params`）
- `data.prompt_template[]` 每一项必须包含 `role/text`（`id` 强烈建议提供，用于提升前端编辑体验）
- `agent` 节点的 `agent_parameters.tools` 只能使用 `<SKILL_ROOT>/config/tools.json` 中声明的工具
- `knowledge-retrieval.dataset_ids` 只能使用 `<SKILL_ROOT>/config/knowledge_bases.json` 中声明的知识库ID

构建完成后必须运行校验（规则在 `scripts/validate_workflow.py`）：
`python "<SKILL_ROOT>/scripts/validate_workflow.py" <your_workflow.json>`

构建并校验完成后必须通过已定义的流程【Phase 5】引导用户进行效果测试与迭代优化

## 核心工作流程

### Phase 1: 需求分析

**理解任务本质:**
1. 识别核心目标（问答、生成、分析、自动化等）
2. 确定输入输出（用户提供什么，期望得到什么）
3. 识别复杂度（单步/多步、是否需要条件分支/迭代）
4. 确定类型：`workflow`（批处理）或 `chat`（对话）
5. 设计节点拓扑结构

**关键问题清单:**
- 是否需要知识库检索（RAG）？
- 是否需要条件判断分流？
- 是否需要迭代处理数组数据？
- 是否需要调用外部API或工具？
- 是否需要Agent自主决策？

### Phase 2: 架构设计

**选择合适的模式:**

| 场景 | 推荐模式 |
|------|----------|
| 简单问答 | Start → LLM → End |
| 知识库问答 | Start → Knowledge Retrieval → LLM(context) → End |
| 意图分流 | Start → Question Classifier → [分支LLM] → Aggregator → End |
| 条件处理 | Start → LLM → IF/ELSE → [分支处理] → Aggregator → End |
| 批量处理 | Start → Parameter Extractor → Iteration[LLM] → Template → End |
| 复杂推理 | Start → Agent(tools) → End |
| 多步生成 | Start → LLM1 → LLM2 → LLM3 → End |

**设计原则:**
1. **最小节点原则**: 能用一个节点完成的不用两个
2. **并行优先**: 无依赖的任务尽量并行
3. **错误处理**: 关键节点配置fail-branch
4. **变量复用**: 避免重复的知识库检索或LLM调用

### Phase 3: 构建Workflow

**MANDATORY**: 构建前必须阅读 [DSL规范](references/DSL_SPEC.md)（路径：`<SKILL_ROOT>/references/DSL_SPEC.md`）

**构建步骤:**

1. **参考规范与模板**
   - 查阅 `references/DSL_SPEC.md` 获取节点Schema（`<SKILL_ROOT>/references/DSL_SPEC.md`）
   - 查阅 `templates/patterns.json` 获取相似模式的完整示例（`<SKILL_ROOT>/templates/patterns.json`）

2. **按拓扑顺序添加节点**
   - 从Start节点开始
   - 每添加一个节点，立即添加其入边
   - 确保变量引用指向已存在的上游节点

3. **配置节点参数**
   - 参考 `templates/` 中的标准配置（`<SKILL_ROOT>/templates/`）
   - LLM节点：选择合适的模型和参数
   - 条件节点：定义清晰的判断逻辑

4. **验证合法性**
   - 运行 `<SKILL_ROOT>/scripts/validate_workflow.py` 检查
   - 确保满足所有硬约束

### Phase 4: 优化与交付

**优化检查清单:**
- [ ] 无冗余节点
- [ ] LLM调用最小化
- [ ] 可并行的任务已并行
- [ ] 节点标题清晰描述功能
- [ ] 关键路径有错误处理

**交付产物**：将 Workflow JSON 写入 `$EVO_OUTPUT_DIR/` 路径下。

交付完成后进入 Phase 5：引导用户进行效果测试与迭代优化

### Phase 5: 效果测试与迭代（必须执行，不得跳过）

JSON 输出后，**必须**完成以下步骤；

1. **通过脚本生成输入参数模板**
执行命令获取该 Workflow 的输入参数 schema：
```bash
   python "<SKILL_ROOT>/scripts/fetch_dify_result.py" \
     --workflow-json "$EVO_OUTPUT_DIR/<your_workflow>.json" \
     --print-input-template
```
2. **向用户展示json参数填写模板，引导其填写提交参数**

3. **等待用户输入后通过脚本执行测试**

收到用户填写的 JSON 后，将用户输入保存为 inputs.json 然后通过接口脚本执行运行验证（具体流程详见 `references/RUN_VALIDATION_STRATEGY.md`，路径：`<SKILL_ROOT>/references/RUN_VALIDATION_STRATEGY.md`）。

4. **向用户展示运行结果，询问是否需要进行workflow的迭代优化**
展示运行结果的文件路径即可

5. **对用户给出的结果反馈或改进建议，进行认真分析思考，对workflow json进行优化，并再次进行效果测试**
---

## 快速参考

### 节点类型速查

| 类型 | 用途 | 关键配置 |
|------|------|----------|
| `start` | 入口，定义用户输入 | `variables[]` |
| `end` | Workflow出口 | `outputs[]` |
| `answer` | Chatflow响应 | `answer` (支持变量) |
| `llm` | LLM调用 | `model`, `prompt_template` |
| `knowledge-retrieval` | RAG检索 | `dataset_ids`, `query_variable_selector` |
| `question-classifier` | 意图分类 | `classes[]`, 每个class一条出边 |
| `if-else` | 条件分支 | `cases[]`, 必须true/false两条出边 |
| `code` | Python/JS执行 | `code`, `variables`, `outputs` |
| `iteration` | 数组迭代 | `iterator_selector`(必须是数组) |
| `template-transform` | 模板渲染 | `template`, `variables` |
| `list-operator` | 列表过滤/排序/截取 | `variable`, `filter_by`, `order_by`, `limit` |
| `document-extractor` | 文件提取文本 | `variable_selector`(指向 file/file-list) |
| `http-request` | API调用 | `method`, `url`, `body` |
| `agent` | 自主决策 | `agent_strategy_name`, `tools` |
| `parameter-extractor` | 结构化提取 | `parameters[]` |

### 变量引用语法

```
{{#node_id.variable#}}     // 节点输出
{{#sys.query#}}            // 系统变量(Chatflow)
{{#env.API_KEY#}}          // 环境变量
```

### 硬约束速记

```
✓ 有且仅有1个Start节点
✓ Start无入边，End无出边
✓ Workflow必须有End，Chatflow必须有Answer
✓ 禁止环路（DAG）
✓ 变量只能引用上游节点
✓ IF/ELSE必须有true和false两条出边
✓ Iteration输入必须是数组类型
```

### ID生成规则

```
- 使用语义化ID：`start_1`, `llm_main`, `kr_products`
- 保持全局唯一
```

---

## 资源文件

| 文件 | 用途 | 何时加载 |
|------|------|----------|
| [`<SKILL_ROOT>/references/DSL_SPEC.md`](references/DSL_SPEC.md) | 完整DSL规范 | 构建前必读 |
| [`<SKILL_ROOT>/templates/patterns.json`](templates/patterns.json) | 常用模式模板 | 设计阶段参考 |
| `<SKILL_ROOT>/scripts/validate_workflow.py` | 验证合法性 | 构建完成后 |

---

## 示例：构建RAG问答Workflow

**需求**: 用户输入问题，从知识库检索相关内容，LLM生成回答

**分析**:
- 类型: `workflow`
- 模式: Start → Knowledge Retrieval → LLM → End
- 节点数: 4

**构建流程**:

1. 阅读 `references/DSL_SPEC.md` 了解节点结构（`<SKILL_ROOT>/references/DSL_SPEC.md`）
2. 参考 `templates/patterns.json` 中的 `rag_qa` 模式（`<SKILL_ROOT>/templates/patterns.json`）
3. 直接编写完整JSON

**验证**:
```bash
python "<SKILL_ROOT>/scripts/validate_workflow.py" "$EVO_OUTPUT_DIR/rag_workflow.json"
```

**输出JSON结构要点**:
```json
{
  "app": { "mode": "workflow", "name": "RAG问答" },
  "workflow": {
    "graph": {
      "nodes": [
        { "id": "start", "data": { "type": "start", "variables": [...] } },
        { "id": "kr", "data": { "type": "knowledge-retrieval", ... } },
        { "id": "llm", "data": { "type": "llm", "context": { "enabled": true, ... } } },
        { "id": "end", "data": { "type": "end", "outputs": [...] } }
      ],
      "edges": [
        { "source": "start", "target": "kr" },
        { "source": "kr", "target": "llm" },
        { "source": "llm", "target": "end" }
      ]
    }
  }
}
```
