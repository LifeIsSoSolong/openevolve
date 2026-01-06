---
name: alphaevolve-run
description: 使用 AlphaEvolve/OpenEvolve/alphaevolve-run 执行MLE代码进化或 prompt 进化任务。支持 MLE 代码进化（如机器学习模型优化）和 Prompt 进化（如提示词优化）。触发场景：启动 alphaevolve 进化、运行 openevolve、MLE 代码进化、prompt 进化、使用 alphaevolve 运行任务。本 skill 提供完整的输入校验、文件共创、运行管理和结果检查功能。最终启动进化时按照步骤4的bash命令来运行（AlphaEvolve/OpenEvolve已经封装好）
---

# AlphaEvolve/OpenEvolve 运行

**Version: 0.6.4**

## 执行流程概览

```
1. 环境检查 → 2. 输入校验 → 3. 文件共创（含 judge 验证） → 4. 启动进化 → 5. 结果检查
```

**严格按顺序执行，每步完成后才能进入下一步。**
**步骤1、2、5自动执行，仅向用户展示结果，步骤3需要和用户交互共创完成，步骤4需要在最终启动前向用户确认**
---

## 步骤 1：环境检查

### 执行明确环境变量
```bash
echo "EVO_INPUT_DIR=$EVO_INPUT_DIR"
echo "EVO_OUTPUT_DIR=$EVO_OUTPUT_DIR"
```

- 只允许在EVO_INPUT_DIR和EVO_OUTPUT_DIR中修改和生成文件, 不允许在其他目录保存/修改内容。

### 判断环境变量
- ✅ 两个变量都有值 → 记录为 `INPUT_DIR` 和 `OUTPUT_DIR`，继续步骤 2
- ❌ 任一变量为空 → **停止执行**，告知用户需要先设置环境变量

---

### 明确SKILL_ROOT
- `SKILL_ROOT`：当前SKILL.md所在的路径，需要你查找该路径，并写入变量`$SKILL_ROOT`

## 步骤 2：输入校验

### 执行
主动查看输入目录并运行校验：

```bash
# 查看输入目录文件列表
ls -la "$EVO_INPUT_DIR"

# 运行校验脚本
python "$SKILL_ROOT/scripts/check_inputs.py" --input-dir "$EVO_INPUT_DIR"
```

### 判断任务类型
根据脚本输出的 `task_type`：
- `mle` → MLE 代码进化任务
- `prompt` → Prompt 进化任务
- `unknown` → 主动查看文件内容判断，或询问用户

### 必须文件

| MLE 任务 | Prompt 任务 |
|----------|-------------|
| agent.py | agent.py |
| 训练数据文件* | train.jsonl |
| 测试数据文件* | test.jsonl |
| - | generate_press_agent.py |
| - | evaluate_press_agent.py |

**\*MLE 任务数据文件说明**：
- 支持格式：`.csv`, `.xlsx`, `.xls`, `.parquet`, `.feather`, `.json`, `.jsonl`, `.tsv`
- 文件名灵活：不强制要求 `train.csv`/`test.csv`，脚本会自动识别含有 `train`/`test` 等关键词的文件
- 单数据集：如果只有一个数据文件，需要与用户确认是否划分为训练/测试集
- 最终确认：向用户确认哪个文件是训练数据、哪个是测试数据

### 处理
- ✅ 必须文件齐全 → 继续步骤 3
- ⚠️ 数据文件未明确识别 → 向用户确认训练/测试数据文件
- ❌ 核心文件缺失（agent.py）→ 列出缺失文件，告知用户需要上传，**等待用户确认后重新执行 `ls` 检查**

---

## 步骤 3：文件共创

按顺序处理：`agent.py 规范` → `task.goal` → `judge.py（生成+验证）` → `config.json`
每一步都要按要求进行，明确说明需要用户确认的地方及时交互，不要自作主张。
**judge.py 验证通过后才能进入 config.json 共创。**

### 3.1 agent.py 规范校验

#### 执行
```bash
# 查看 agent.py 内容
cat "$EVO_INPUT_DIR/agent.py"

# 运行规范校验
python "$SKILL_ROOT/scripts/validate_agent.py" --input-dir "$EVO_INPUT_DIR" --task-type "$TASK_TYPE"
```

#### 规范要求
- **所有任务都要满足**：agent.py 中的数据路径必须使用 `main(root)` 的 `root` 来拼接构建。禁止使用 `__file__` / `Path(__file__)` / `os.getcwd()` / `Path.cwd()` 获取目录（agent.py 会被移动到临时路径）。
- **MLE 任务**：
  - 必须有 `main(root)` 函数 + EVOLVE-BLOCK 标记
  - `main(root)` **只返回指标 dict**，指标 key 可根据原始agent.py来构造，但必须与 judge.py 计算 combined_score 时使用的指标一致
  - 返回的指标必须是在测试集上的指标，不能是在训练集/训练集中拆分的验证集的指标
- **Prompt 任务**：必须有 `get_prompt_generate_press()` 函数 + EVOLVE-BLOCK 标记
#### 处理
- ✅ 校验通过 → 继续 3.2
- ❌ 校验失败 → 
  1. 阅读 `references/agent_spec.md` 了解规范，必要时参考 `assets/agent_mle.py` 或 `assets/agent_prompt.py`
  2. 若存在路径问题（未使用 root / 使用 `__file__` 等），优先执行：
     ```bash
     python "$SKILL_ROOT/scripts/fix_agent_paths.py" --input-dir "$EVO_INPUT_DIR" --task-type "$TASK_TYPE" --write
     ```
  3. 基于已读取的 agent.py 内容，生成符合规范的包装代码
  4. 展示修改方案给用户确认
  5. 用户确认后写入文件
  6. 重新运行校验确认通过

### 3.2 task.goal 共创

#### 执行
```bash
# 检查是否存在
ls "$EVO_INPUT_DIR/task.goal" 2>/dev/null && cat "$EVO_INPUT_DIR/task.goal" || echo "[NOT EXISTS]"
```

#### 处理
**若不存在**：
1. 阅读 `references/task_goal_guide.md` 了解编写指南
2. 询问用户：任务描述、优化目标（可选：数据特点、约束条件）
3. 生成 task.goal 内容并展示
4. 用户确认后写入 `$EVO_INPUT_DIR/task.goal`

**若已存在**：
1. 展示当前内容（已通过 cat 读取）
2. 询问用户是否需要优化
3. 如需优化，提出建议，用户确认后更新

### 3.3 judge.py 共创

#### 执行
```bash
# 检查是否存在
ls "$EVO_INPUT_DIR/judge.py" 2>/dev/null && echo "[EXISTS]" || echo "[NOT EXISTS]"
```

#### 处理
**若不存在**：
1. 阅读 `references/judge_spec.md` 了解规范，必要时参考 `assets/judge_mle.py` 或 `assets/judge_prompt.py`
2. **确认指标属性**（重要）：
   - 向用户确认每个指标的属性：
     - 是"越大越好"还是"越小越好"
     - 是否是百分比指标（值域 0-100）
   - 已知指标会自动识别，未知指标需要用户确认
   - 指标类型参考：
     | 类型 | 说明 | 示例 |
     |------|------|------|
     | `lower` | 越小越好 | rmse, mse, mae, log_loss |
     | `lower_pct` | 越小越好，百分比 | mape, rrmse, smape |
     | `higher` | 越大越好，范围0-1 | accuracy, f1, auc, precision |
     | `higher_r2` | 越大越好，范围-1到1 | r2, mcc, kappa |
3. 运行生成脚本预览：
   ```bash
   python "$SKILL_ROOT/scripts/generate_judge.py" --input-dir "$EVO_INPUT_DIR" --task-type "$TASK_TYPE" --dry-run
   ```
   - 若检测到未知指标，脚本会显示警告
   - 可以显式指定指标类型：`--metrics "rmse:lower,custom:higher,pct_error:lower_pct"`
4. **检查生成的代码**：
   - 确认 `calculate_combined_score` 函数中每个指标的转换公式正确
   - 如有 TODO 注释，**必须**根据指标含义修改为正确的公式
5. 展示生成的代码给用户确认
6. 用户确认后正式生成：
   ```bash
   python "$SKILL_ROOT/scripts/generate_judge.py" --input-dir "$EVO_INPUT_DIR" --task-type "$TASK_TYPE"
   ```
7. 生成后立即进行运行验证：
   ```bash
   python "$SKILL_ROOT/scripts/validate_judge.py" --input-dir "$EVO_INPUT_DIR" --timeout 300
   ```
8. 验证失败则修复并重新验证，**验证通过才进入 3.4**。

**若已存在**：
1. 直接进行运行验证：
   ```bash
   python "$SKILL_ROOT/scripts/validate_judge.py" --input-dir "$EVO_INPUT_DIR" --timeout 300
   ```
2. 验证失败则需要修改agent.py和judge.py来修复并重新验证，**验证通过才进入 3.4**。

**注意**：
- Prompt 任务会真实调用 LLM API，需要告知用户可能产生费用
- 超时时间可根据任务复杂度调整
- MLE 任务必须确保 combined_score 计算正确（指标方向、范围都对）
- **对于未知指标，应该根据指标名称和上下文推断其属性，并与用户确认**

### 3.4 config.json 共创

#### 执行
```bash
# 读取模板
cat "$SKILL_ROOT/assets/config_$TASK_TYPE.json"

# 检查用户 config 是否存在
ls "$EVO_INPUT_DIR/config.json" 2>/dev/null && cat "$EVO_INPUT_DIR/config.json" || echo "[NOT EXISTS]"
```

#### 处理
**若不存在**：
1. 阅读 `references/config_guide.md` 了解各字段含义
2. 展示模板内容，重点说明关键参数：
   - `max_iterations`：进化轮数（默认 5）
   - `evaluator.timeout`：评估超时（MLE 默认 3000s，Prompt 默认 30000s）
   - `evolution_trace.enable`：此功能打开可生成详细的进化过程，用于生成进化树
3. 询问用户是否需要调整参数
4. 生成最终 config 并展示config中的完整内容
5. 待用户确认后写入 `$EVO_INPUT_DIR/config.json`

**若已存在**：
1. 对比用户 config 和模板，汇报差异：
   - 超出模板范围的 key → 必须删除
   - 模板有但用户没有的 key → 推荐使用默认值
   - 双方都有的 key → 确认使用哪个值
2. 展示最终配置
3. 用户确认后覆盖写入

**重要**：最终 config 的 key 只能是模板 key 的子集。

**下一步**：`$EVO_INPUT_DIR/config.json`写入后，执行步骤4启动进化(按既定命令，不要自己瞎写)

---

## 步骤 4：启动进化

### 启动前的准备工作
```bash
# 创建输出目录
mkdir -p "$EVO_OUTPUT_DIR"

# 展示最终配置
cat "$EVO_INPUT_DIR/config.json"
```

### 启动前向用户确认
展示配置后询问用户："配置确认无误，是否启动进化任务？"，必须要用户确认才能启动

### 进化进程启动
严格按照如下命令，使用nohup在后台启动进化进程（AlphaEvolve/OpenEvolve已经封装在了$SKILL_ROOT/scripts/main.py中），直接使用如下bash，不要自己造：

```bash
nohup python "$SKILL_ROOT/scripts/main.py" \
    --config_file "$EVO_INPUT_DIR/config.json" \
    --input_dir "$EVO_INPUT_DIR" \
    --output_dir "$EVO_OUTPUT_DIR" \
    > "$EVO_OUTPUT_DIR/run.log" 2>&1 &
echo "PID: $!"
```

### 反馈
- 进程 PID
- 日志路径：`$EVO_OUTPUT_DIR/run.log`
- 查看命令：`tail -f $EVO_OUTPUT_DIR/run.log`

---

## 步骤 5：结果检查

### 产出检查
```bash
python "$SKILL_ROOT/scripts/check_outputs.py" --output-dir "$EVO_OUTPUT_DIR"
```

### 状态查询
用户查询运行状态时主动执行：

```bash
# 查看状态
cat "$EVO_OUTPUT_DIR/status.json"

# 查看最新事件
tail -20 "$EVO_OUTPUT_DIR/events.jsonl"

# 查看日志
tail -50 "$EVO_OUTPUT_DIR/run.log"
```

汇报：当前状态、进度、最优 combined_score、提升百分比

### 查看最优结果
进化完成后：
```bash
ls "$EVO_OUTPUT_DIR/final_result/"
cat "$EVO_OUTPUT_DIR/final_result/best_program.py"
```

### 进化结果验证（重要）
进化完成后，**必须**检测最优程序是否存在数据泄露：

```bash
python "$SKILL_ROOT/scripts/check_data_leakage.py" \
    --file "$EVO_OUTPUT_DIR/final_result/best_program.py"
```

**检测项目**：
- 测试集是否参与了训练（fit/train 调用）
- 是否先合并 train/test 再做预处理
- 统计量是否错误地使用了测试集数据
- EVOLVE-BLOCK 内是否有测试集训练操作

**如果检测到泄露**：
1. 向用户报告发现的泄露问题
2. 说明进化过程可能产生了不安全的代码
3. 建议：
   - 修改 task.goal，明确禁止测试集参与训练
   - 调整 EVOLVE-BLOCK 范围，将测试集处理移到块外
   - 重新运行进化

**检测通过后**：
```
✅ 进化完成！

最优程序：$EVO_OUTPUT_DIR/final_result/best_program.py
数据泄露检测：通过 ✓
最终 combined_score：[分数]
相比初始提升：[百分比]
```

---

## 资源说明

### scripts/
| 脚本 | 功能 |
|------|------|
| `main.py` | OpenEvolve 运行入口 |
| `check_inputs.py` | 输入目录校验 |
| `check_outputs.py` | 输出目录校验 |
| `validate_agent.py` | agent.py 规范校验 |
| `fix_agent_paths.py` | 自动修复 agent.py 数据路径（强制使用 judge.py传给agent.py的root） |
| `validate_judge.py` | judge.py 运行验证 |
| `generate_judge.py` | 自动生成 judge.py |
| `check_data_leakage.py` | 数据泄露检测（进化完成后必须运行） |

### references/
| 文档 | 阅读时机 |
|------|----------|
| `agent_spec.md` | 步骤 3.1 需要修复 agent.py 时 |
| `judge_spec.md` | 步骤 3.3 需要生成 judge.py 时 |
| `task_goal_guide.md` | 步骤 3.2 需要共创 task.goal 时 |
| `config_guide.md` | 步骤 3.4 需要解释配置时 |

### assets/
| 文件 | 用途 |
|------|------|
| `agent_mle.py` | MLE 任务 agent.py 参考模板 |
| `agent_prompt.py` | Prompt 任务 agent.py 参考模板 |
| `config_mle.json` | MLE 任务配置模板 |
| `config_prompt.json` | Prompt 任务配置模板 |
| `judge_mle.py` | MLE 任务 judge.py 参考模板 |
| `judge_prompt.py` | Prompt 任务 judge.py 参考模板 |
