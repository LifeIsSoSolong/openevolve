# 运行验证（E2E）交互策略：生成 Workflow → 收集入参 → 运行 → 反馈迭代

在 Workflow JSON 生成后，引导用户进行效果测试与迭代的完整流程。


## 1) 参数获取（自动 + 对话补齐）

### 1.1 从 workflow JSON 自动提取参数 schema

执行以下命令（将 `<your_workflow>.json` 替换为实际生成的 JSON 文件名）：

```bash
python "<SKILL_ROOT>/scripts/fetch_dify_result.py" \
  --workflow-json "$EVO_OUTPUT_DIR/<your_workflow>.json" \
  --print-input-template
```

### 1.2 给用户展示模板并收集值

优先建议用户用“JSON 对象”一次性填写（key 必须与模板中的参数名一致），例如：

```json
{
  "param1": "...",
  "param2": "...",
  "constraints": ""
}
```

若用户不方便一次性给 JSON，则按模板逐项追问必填项，直到满足校验。

## 2) 输入校验（脚本能力，必须）

在真正运行前，先把用户提供的 inputs 落盘到 `$EVO_OUTPUT_DIR/inputs.json`，并做校验（不触发导入/运行）：

```bash
python "<SKILL_ROOT>/scripts/fetch_dify_result.py" \
  --workflow-json "$EVO_OUTPUT_DIR/<your_workflow>.json" \
  --inputs-json "$EVO_OUTPUT_DIR/inputs.json" \
  --validate-only \
  --strict-inputs
```

校验失败时：不要运行；把缺失/错误项总结给用户并继续补齐。

## 3) 运行与轮询（必须）

校验通过后，通过nohup后台执行导入草稿 + 运行 + 轮询（5 分钟、每 5 秒一次）：

```bash
nohup python "<SKILL_ROOT>/scripts/fetch_dify_result.py" \
  --workflow-json "$EVO_OUTPUT_DIR/<your_workflow>.json" \
  --inputs-json "$EVO_OUTPUT_DIR/inputs.json" \
  --strict-inputs &
echo $!
```

输出产物均在 `$EVO_OUTPUT_DIR/`，包括：
- `fetch_dify_result.log`
- `import_response.json`
- `run_final_record.json`
- `workflow_result.txt`（若 outputs 中存在 result/text）

## 4) 面向用户的回执（用于迭代反馈）

输出要点：
- `run_status`: succeeded / failed / timeout
- `result` 摘要（必要时截断，并说明完整文本已保存）
- 下一步问题：让用户指出需要改进的地方

## 5) 执行注意事项

- 不要因“导入前置”而跳过模板输出：参数模板来自本地 workflow JSON。
- 校验失败时必须追问补齐，禁止直接运行。
