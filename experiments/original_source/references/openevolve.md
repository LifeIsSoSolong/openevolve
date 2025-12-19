# 对 OpenEvolve 的核心机制总结（便于后续改造/对齐 GEPA 思路）

## 1. Artifacts：现状与数据流

### 1.1 Artifacts 是什么
OpenEvolve 里的 `artifacts` 本质上是“评测过程产生的附加信息侧信道”，用于把**错误/日志/诊断上下文**带入下一轮 prompt，帮助 LLM 做 debug/定位问题。它和 `metrics` 分离：
- `metrics`：数值指标（用于选择/进化排序）
- `artifacts`：文本或二进制内容（用于给 LLM 提供上下文）

对应的数据结构是 `openevolve/evaluation_result.py` 的 `EvaluationResult(metrics, artifacts)`。

### 1.2 Artifacts 从哪里来（Evaluator 端）
核心入口：`openevolve/evaluator.py:132` `Evaluator.evaluate_program()`。

Artifacts 来源目前主要有四类：
1) **超时 artifacts**
   - 当评测超时，`Evaluator` 会写入类似：`timeout=true`、`timeout_duration`、`failure_stage`、`error_type=timeout` 等（见 `openevolve/evaluator.py:173`、`openevolve/evaluator.py:252`）。
2) **异常 artifacts**
   - 当评测函数抛异常，写入：`stderr`（异常字符串）、`traceback`（Python traceback）、`failure_stage`、`attempt` 等（见 `openevolve/evaluator.py:267`）。
3) **级联评测（cascade） artifacts**
   - 若启用 cascade（多阶段评测），stage1/2/3 失败会带 stage 级别的 stderr/traceback/timeout，以及 `_create_cascade_error_context()` 的上下文信息（`openevolve/evaluator.py:380`、`openevolve/evaluator.py:644`）。
4) **评测函数主动返回 artifacts**
   - 只有当你的 evaluator（例如 `judge.py`）返回 `EvaluationResult(metrics=..., artifacts=...)`，成功运行也能产生 artifacts；如果只是返回 dict，则没有 artifacts 侧信道（`openevolve/evaluator.py:298`）。

补充：artifacts 是否启用还受环境变量影响：`ENABLE_ARTIFACTS`（`openevolve/evaluator.py:150`，默认 `"true"`）。

### 1.3 Artifacts 如何存储（Database 端）
并行模式下：worker 把 artifacts 放在 `SerializableResult.artifacts` 返回主进程，主进程调用：
- `database.store_artifacts(child_id, artifacts)`（`openevolve/process_parallel.py:504`）

`store_artifacts`（`openevolve/database.py:2315`）会把 artifacts 按大小分流：
- 小 artifacts：序列化进 `Program.artifacts_json`
- 大 artifacts：写到磁盘目录 `Program.artifact_dir`

### 1.4 Artifacts 怎么进入下一轮 prompt
Prompt 构造时，如果 `PromptConfig.include_artifacts=true` 且拿到了 `program_artifacts`，则插入到 user prompt 的 `{artifacts}` 段：
- `openevolve/prompt/sampler.py:124` → `openevolve/prompt/sampler.py:565` `_render_artifacts()`
- 默认模板 `openevolve/prompts/defaults/diff_user.txt` 中 `{artifacts}` 出现在 “Current Program Information” 后

渲染特性：
- 每个 artifact 变成一个 `### key` + 代码块
- 超过 `max_artifact_bytes` 会截断（默认 20KB，`openevolve/config.py:252`）
- 可选简单脱敏 `artifact_security_filter`（`openevolve/prompt/sampler.py:620`）

### 1.5 并行模式的一个“隐藏限制”（很重要）
`ProcessParallelController._create_database_snapshot()` 为了控制进程间传输成本，**只把前 100 个 program 的 artifacts 带给 worker**：
- `openevolve/process_parallel.py:384`

结果是：即使数据库里历史 program 有 artifacts，worker 也可能拿不到 parent 的 artifacts（尤其当 program 数量很大时）。

### 1.6 当前现状观察：Artifacts 经常“为空”
以我们 bean03 的一次大规模 test run（`experiments/bean03/test_results/258975194553516032`）为例，检查所有 checkpoint 下的 `programs/*.json`：
- 没有任何 program 写入 `artifacts_json`/`artifact_dir`

这通常意味着：
- evaluator 大部分时间“正常运行”且不超时/不异常（所以 Evaluator 不会自动生成 artifacts）；并且
- `judge.evaluate()` 仅返回 dict（metrics），没有主动返回 `EvaluationResult(..., artifacts=...)`；同时
- 系统不会自动 capture 成功执行时的 stdout/stderr（除非你在 evaluator 里显式把它们作为 artifacts 返回）。

因此目前 artifacts 更偏向“失败诊断”而不是 GEPA 所强调的 `μ_f`（成功/失败都能提供富文本反馈）。

## 2. Build Prompt：除了 artifacts 还包含什么（现状）

OpenEvolve 的 prompt 由两部分构成：`system` 与 `user`（`openevolve/prompt/sampler.py:51`）。并行模式下 worker 会在 `_run_iteration_worker()` 里调用 `PromptSampler.build_prompt(...)`（`openevolve/process_parallel.py:170`）。

### 2.1 system message 从哪里来
`PromptSampler.build_prompt()` 的 system message 选择逻辑（`openevolve/prompt/sampler.py:101`）：
- 若 `set_templates(system_template=...)` 设置过 override，则用 override 模板；
- 否则用 `config.prompt.system_message`：默认是字符串 `"system_message"`，会映射到模板文件 `openevolve/prompts/defaults/system_message.txt`。

### 2.2 user message 的模板与主要组成块
默认 diff 模式的 user 模板是 `openevolve/prompts/defaults/diff_user.txt`，其主要占位符有：
- `{fitness_score}`：当前 parent 的适应度（用于进化选择），通过 `get_fitness_score()` 计算（`openevolve/utils/metrics_utils.py:69`，优先用 `combined_score`）。
- `{feature_coords}`：展示 feature dimensions 在 metrics 里的取值（`openevolve/utils/metrics_utils.py:117`）。注意：如果 evaluator 不返回与 feature_dimensions 同名的指标，会显示 “No feature coordinates”；这不影响数据库内部仍使用 built-in `complexity/diversity` 做 MAP-Elites 分箱（`openevolve/database.py:820`）。
- `{improvement_areas}`：自动生成的“本轮关注点”摘要（`openevolve/prompt/sampler.py:170`），主要来源：
  - 当前 fitness vs `previous_programs[-1]` fitness 的变化（improved/declined/stable）；
  - 代码过长提醒（默认阈值 500 chars）。
- `{evolution_history}`：历史段落（`openevolve/prompt/sampler.py:228`）由三块拼装：
  - Previous Attempts（最多 3 个）
  - Top Performing Programs + Diverse Programs（由 `num_top_programs/num_diverse_programs` 控制）
  - Inspiration Programs（来自 `inspirations`）
- `{current_program}`：当前 parent 的完整代码。
- `{language}`：代码块语言标注。
- `{feature_dimensions}`：告诉 LLM 系统维护多样性的维度名列表。
- `{artifacts}`：上一节说的执行/诊断产物（可为空）。

补充：`build_prompt()` 内部也会计算 `metrics_str = _format_metrics(program_metrics)`（`openevolve/prompt/sampler.py:111`），但默认模板 `diff_user.txt` 并没有 `{metrics}` 占位符，因此很多场景下“算了但不展示”；自定义模板可启用它。

### 2.3 这些历史程序（previous/top/diverse/inspirations）从哪里来（并行模式）
worker 会基于 parent 所在岛屿收集上下文（`openevolve/process_parallel.py:150`）：
- `island_programs`：从快照里取该岛所有程序，按 fitness 排序；
- `previous_programs`：取 top `num_top_programs`（用于 “Previous Attempts”）；
- `top_programs`：取 top `num_top_programs + num_diverse_programs`（用于 “Top” + “Diverse”）；
- `inspirations`：由主进程在提交任务时 `database.sample_from_island(..., num_inspirations=prompt.num_top_programs)` 抽样得到（`openevolve/process_parallel.py:722`、`openevolve/database.py:389`）。

这意味着 prompt 里除了 artifacts，还系统性包含：
- 当前程序的 fitness/指标提示（用于指导改进方向）
- 历史尝试与高分样例（用于 resurfacing past ideas）
- 灵感样例（用于探索多样化策略）
