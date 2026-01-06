---
name: alphaevolve-mle-code-creator
description: |
  根据用户数据和任务描述，生成MLE代码，且该代码符合 AlphaEvolve 进化规范（agent.py）。
  触发场景：创建MLE code、创建MLE代码、创建mle代码、生成MLE代码、生成mle代码、创建 AlphaEvolve 兼容的机器学习代码、为数据集生成可进化的 ML 模型代码、准备 MLE 进化任务的初始代码、把现有 ML 代码转换为 AlphaEvolve 格式。
  本技能与 alphaevolve-run 联动，生成的代码可直接用于 AlphaEvolve 进化。
  支持任意格式数据（csv, xlsx, parquet, json 等），支持单数据集自动划分或已划分的训练/测试集。
---

# AlphaEvolve MLE Code Creator

**Version: 0.1.1**

基于用户数据和任务描述，生成符合 AlphaEvolve 进化规范的 `agent.py`。

## 核心原则

1. **灵活适应用户数据** — 不限制文件名和格式，Claude 动态分析用户上传的数据
2. **交互式共创** — 与用户确认目标列、任务类型、评估指标
3. **严格遵守 AlphaEvolve 规范** — 生成的代码必须满足 `alphaevolve-run` 的要求

## 执行流程

```
1. 环境与数据检查 → 2. 数据分析与确认 → 3. 任务描述共创 → 4. 代码生成 → 5. 验证与准备
```

---

## 步骤 1：环境与数据检查

### 1.1 检查环境变量
```bash
echo "EVO_INPUT_DIR=$EVO_INPUT_DIR"
echo "EVO_OUTPUT_DIR=$EVO_OUTPUT_DIR"
```

- ✅ 两个变量都有值 → 继续
- ❌ 任一变量为空 → **停止执行**，告知用户需要先设置环境变量

### 1.2 明确 SKILL_ROOT
找到当前 SKILL.md 所在路径，记录为 `$SKILL_ROOT`

### 1.3 扫描输入目录
```bash
ls -la "$EVO_INPUT_DIR"
```

**支持的数据格式**：
- 表格数据：`.csv`, `.xlsx`, `.xls`, `.parquet`, `.feather`
- 结构化数据：`.json`, `.jsonl`
- 压缩文件：`.zip`, `.gz`（需先解压）

**数据组织方式**（都支持）：
- **单文件**：一个数据文件，需要自动划分训练/测试集
- **已划分**：已有 train/test 文件（任意命名，如 `train.csv`, `training_data.xlsx`, `data_train.parquet` 等）
- **目录结构**：`train/` 和 `test/` 子目录

### 1.4 向用户确认数据
列出发现的数据文件，询问用户：
- 哪个是训练数据？哪个是测试数据？
- 如果只有一个文件，是否需要自动划分（默认 80/20）？
- 是否有需要排除的文件？

---

## 步骤 2：数据分析与确认

### 2.1 加载并分析数据
根据文件格式动态加载数据，分析：
- 数据形状（行数、列数）
- 列名、数据类型
- 缺失值情况
- 数值列统计信息
- 类别列的唯一值分布

### 2.2 推断任务信息
- **目标列**：尝试识别（常见名称：target, label, y, class, outcome 或最后一列）
- **任务类型**：根据目标列类型推断（连续值→回归，离散值→分类）
- **评估指标**：根据任务类型推荐

### 2.3 与用户确认
展示分析结果，**必须确认**：
1. **目标列**：推断是否正确？
2. **任务类型**：回归还是分类？
3. **评估指标**：使用哪些指标？（如 RMSE、AUC、F1 等）
4. **特殊列处理**：ID 列、时间列、需要排除的列？

---

## 步骤 3：任务描述共创

### 3.1 检查是否已有描述
```bash
cat "$EVO_INPUT_DIR/description.md" 2>/dev/null || echo "[NOT EXISTS]"
```

### 3.2 共创流程

**若不存在 description.md**：
1. 阅读 `references/description_guide.md` 了解编写指南
2. 向用户询问：
   - **任务背景**：在做什么业务/研究？
   - **优化目标**：最关心哪个指标？
   - **约束条件**：运行时间、内存、不能用的方法？
   - **改进方向**（可选）：想尝试什么？
3. 生成 description.md 并展示
4. 用户确认后写入

**若已存在**：展示内容，询问是否需要补充

---

## 步骤 4：代码生成

### 4.1 生成前准备
**必须阅读**规范文档：
- `references/agent_spec.md` — agent.py 的硬性要求
- `references/code_templates.md` — 代码模板参考

### 4.2 数据文件确认

确认用户的数据文件：
- **训练数据**：确认哪个文件是训练数据
- **测试数据**：确认哪个文件是测试数据，如果没有则需要从训练集划分

**注意**：`alphaevolve-run` 已支持多种数据格式（csv, xlsx, parquet 等）和灵活的文件命名，无需强制重命名为 `train.csv`/`test.csv`。

### 4.3 生成 agent.py
根据确认的信息，生成符合规范的代码。

**必须满足的规范**（详见 `references/agent_spec.md`）：

1. **main(root) 入口函数**：
   ```python
   def main(root) -> dict:
       data_dir = Path(root)
       # 所有路径基于 root 构建，使用实际的数据文件名
       train_df = pd.read_csv(data_dir / "training_data.csv")  # 根据实际文件名
       test_df = pd.read_parquet(data_dir / "test_set.parquet")  # 支持多种格式
       # ... 训练和评估
       return {"rmse": rmse_value, "mape": mape_value}
   ```

2. **EVOLVE-BLOCK 标记**：包裹可进化的核心逻辑
   ```python
   # EVOLVE-BLOCK-START
   def feature_engineering(df):
       # 这部分代码会被 AlphaEvolve 进化
       pass
   # EVOLVE-BLOCK-END
   ```

3. **路径规则**：
   - ✅ `Path(root) / "实际文件名.csv"` 或其他格式
   - ❌ 禁止 `__file__`, `os.getcwd()`, `Path.cwd()`, 硬编码路径

4. **返回值**：
   - 只返回 dict，包含测试集上的评估指标
   - 指标 key 需与后续 judge.py 一致

### 4.4 EVOLVE-BLOCK 范围确认（重要）

生成代码后，**必须**与用户确认 EVOLVE-BLOCK 的范围。

**展示格式**：
```
📦 EVOLVE-BLOCK 范围确认

当前 EVOLVE-BLOCK 包含以下函数/代码段：
┌─────────────────────────────────────────────────────────┐
│ # EVOLVE-BLOCK-START                                    │
│                                                         │
│ 1. preprocess_features()    - 特征预处理                │
│ 2. get_model_params()       - 模型超参数                │
│ 3. compute_metrics()        - 指标计算                  │
│                                                         │
│ # EVOLVE-BLOCK-END                                      │
└─────────────────────────────────────────────────────────┘

💡 EVOLVE-BLOCK 内的代码会被 AlphaEvolve 进化优化
💡 EVOLVE-BLOCK 外的代码保持不变（如数据加载、main 函数结构）

请确认或提出修改意见：
1. ✅ 确认当前范围
2. ➕ 需要添加更多代码到 EVOLVE-BLOCK
3. ➖ 需要从 EVOLVE-BLOCK 移除某些代码
4. 🔄 需要调整 EVOLVE-BLOCK 的位置
```

**用户可能的修改意见**：

| 用户意见 | 处理方式 |
|----------|----------|
| "把模型选择也加进去" | 将模型构建函数移入 EVOLVE-BLOCK |
| "特征工程不要进化" | 将 preprocess_features 移出 EVOLVE-BLOCK |
| "只进化超参数" | 缩小范围，只保留 get_model_params |
| "把整个训练流程都进化" | 扩大范围，但提醒注意数据泄露风险 |
| "分成多个 EVOLVE-BLOCK" | 说明目前只支持单个 EVOLVE-BLOCK |

**安全提醒**：
- 如果用户要求将测试集处理逻辑放入 EVOLVE-BLOCK，**必须警告**可能导致数据泄露
- 建议将数据加载、测试集 transform 保持在 EVOLVE-BLOCK 外

### 4.5 最终确认并写入

用户确认 EVOLVE-BLOCK 范围后：
1. 展示完整代码
2. 解释代码结构和返回的指标
3. 等待用户最终确认
4. 写入 `$EVO_INPUT_DIR/agent.py`

---

## 步骤 5：验证与准备

### 5.1 语法检查
```bash
python -m py_compile "$EVO_INPUT_DIR/agent.py"
```

### 5.2 数据泄露检测（重要）
```bash
python "$SKILL_ROOT/scripts/check_data_leakage.py" --file "$EVO_INPUT_DIR/agent.py"
```

**必须确保**：
- ❌ 测试集不能参与训练（fit/train 方法只在训练集上调用）
- ❌ 不能先合并 train/test 再做 fit_transform
- ❌ 统计量（mean/std/min/max）必须只从训练集计算
- ❌ EVOLVE-BLOCK 内不能有测试集的训练操作

**如果检测到泄露**：
1. 向用户说明问题
2. 修复代码（确保正确的 fit -> transform 顺序）
3. 重新检测直到通过

### 5.3 运行测试
```bash
cd "$EVO_INPUT_DIR" && python -c "
from agent import main
result = main('.')
print('返回结果:', result)
assert isinstance(result, dict), '必须返回 dict'
assert len(result) > 0, '必须包含指标'
for k, v in result.items():
    assert isinstance(v, (int, float)), f'指标 {k} 必须是数值'
print('✅ 验证通过')
"
```

### 5.4 规范校验（可选）
```bash
python /mnt/skills/user/alphaevolve-run/scripts/validate_agent.py \
    --input-dir "$EVO_INPUT_DIR" --task-type mle
```

### 5.5 完成提示
```
✅ agent.py 生成成功！

已通过检测：
- 语法检查 ✓
- 数据泄露检测 ✓
- 运行测试 ✓

已生成/准备的文件：
- agent.py（符合 AlphaEvolve 规范，无数据泄露）
- 数据文件（用户原始格式）
- description.md（任务描述）

下一步：使用 alphaevolve-run 技能启动进化
```

---

## 资源说明

### references/
| 文档 | 用途 |
|------|------|
| `agent_spec.md` | agent.py 规范要求（**生成代码前必读**） |
| `description_guide.md` | 任务描述共创指南 |
| `code_templates.md` | 不同任务类型的代码模板 |

### assets/
| 文件 | 用途 |
|------|------|
| `agent_regression.py` | 回归任务参考模板 |
| `agent_classification.py` | 分类任务参考模板 |

### scripts/（辅助脚本，非必须）
| 脚本 | 用途 |
|------|------|
| `analyze_data.py` | 数据分析（Claude 也可直接用 Python 分析） |
| `generate_agent.py` | 代码生成（Claude 也可直接生成） |

---

## 与 alphaevolve-run 联动

```
[用户任意数据：any_train.parquet, validation.xlsx, ...] 
    ↓
[alphaevolve-mle-code-creator]
    - 分析数据（自动识别格式）
    - 共创任务描述  
    - 生成 agent.py（使用实际文件名）
    ↓
[alphaevolve-run]
    - 自动识别数据文件（支持多种格式和命名）
    - 校验 agent.py
    - 生成 judge.py, task.goal, config.json
    - 启动进化
    ↓
[优化后的 best_program.py]
```

## 注意事项

1. **数据格式灵活**：支持 csv, xlsx, parquet, json 等多种格式，根据实际文件动态处理
2. **文件名灵活**：无需重命名，alphaevolve-run 会自动识别含有 train/test/validation 等关键词的文件
3. **交互确认**：目标列、任务类型、评估指标必须与用户确认，不要自作主张
4. **代码规范**：生成的代码必须严格遵守 agent_spec.md 中的规范，使用实际的数据文件名
