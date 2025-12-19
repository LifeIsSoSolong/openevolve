# paper name 
GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning
# paper link
https://arxiv.org/abs/2507.19457

## 论文核心解读（中文）

### 一句话总结
GEPA 认为：对很多下游任务/Agent 系统，RL（如 GRPO）只用“最终标量 reward”学习信号太稀疏、样本效率低；而 LLM 系统的 rollout 本身是可读的语言轨迹（推理、工具调用、工具输出、评测日志/报错），可以让 LLM 用自然语言做诊断与归因（reflection），从而用很少 rollouts 学到高层规则并更新 prompt。GEPA 用“反思 + 遗传式搜索 + Pareto 前沿采样”实现这一点，并宣称在多任务上用更少 rollouts 达到更高分。

### 论文想解决的问题（动机）
- 在真实系统里 rollouts 很贵（会调用检索/执行/编译等工具，或者模型本身很贵）。
- GRPO/Policy Gradient 把一条 rollout 压成一个标量 reward，丢掉了大量过程信息（例如报错、trace、工具输出），导致需要成千上万条 rollouts 才学到有效策略。
- 语言轨迹是 LLM 能理解的学习介质：如果把过程信息喂给 LLM，让它反思总结规则，再去改 prompt，可能比梯度更“信息密集”。

### GEPA 的三根支柱
1) Genetic（遗传式演化）
   - 维护一个候选池 P（每个候选是“整个系统所有模块 prompts 的一个版本”）。
   - 迭代地产生新候选（主要是 mutation，也支持 merge/crossover）。
2) Reflection（反思式 prompt mutation）
   - 每轮选一个候选系统、再选一个模块（module）来更新。
   - 在一个 minibatch 上跑 rollouts，收集“系统轨迹 + 评测轨迹 + 文本反馈”，用 LLM 反思并提出该模块 prompt 的更新。
   - 反思的目的不是写感想，而是做“隐式 credit assignment”：把最终成败归因到模块 prompt 的某些要素/缺失，并给出可执行的新指令。
3) Pareto（按实例的 Pareto 前沿选父代）
   - 不是只选“平均分最高”的单一候选继续改（容易陷入局部最优）。
   - 维护“按训练实例逐点最优”的候选集合，并在非支配（non-dominated）的 Pareto 前沿里随机抽候选进行演化，偏向在更多实例上表现最强的候选。

### 核心算法流程（论文 Algorithm 1/2 的直译版）
设：
- 系统为 Φ（包含多个 LLM 模块 prompts），训练数据集 D_train。
- 评价函数为 μ：给定系统输出与标注，返回得分（越大越好）。
- 反馈函数为 μ_f：返回 (score, feedback_text)，feedback_text 可包含评测过程产生的关键文本轨迹。
- 总 rollout 预算为 B；minibatch 大小 b；Pareto 集大小 n_pareto。

Algorithm 1（GEPA 主循环）大意：
1. 把 D_train 切成两份：D_feedback（用于迭代更新）和 D_pareto（用于维护 Pareto 前沿；大小 n_pareto）。
2. 初始化候选池 P 只有一个候选：原始系统 Φ（seed prompts）。
3. 对每个 D_pareto 实例 i，计算该候选在 i 上的分数，形成“候选×实例”的分数矩阵 S。
4. 循环直到 budget B 用完：
   - 用 Algorithm 2（Pareto-based selection）从候选池选一个候选 Φ_k。
   - 选择一个目标模块 j（论文中建议 round-robin，保证每个模块都能得到优化机会）。
   - 从 D_feedback 抽一个 minibatch M（大小 b），在 M 上执行系统并用 μ_f 收集 score + 文本反馈 + 轨迹（含模块级轨迹）。
   - 用 LLM 进行 UPDATE_PROMPT：根据反馈/轨迹反思，得到模块 j 的新 prompt π'_j。
   - 得到新候选 Φ'：复制 Φ_k，但把模块 j prompt 替换为 π'_j。
   - 在 minibatch M 上比较改动前后平均分 σ vs σ'；如果 σ' 提升，则把 Φ' 加入候选池，并在 D_pareto 上把 S_{Φ'} 补全（为 Pareto 选择提供按实例向量）。
5. 返回在 D_pareto 上平均分最高的候选。

Algorithm 2（Pareto-based candidate selection）直觉：
- 对每个训练实例 i，找“当前候选池在 i 上的最高分”，把达到该最高分的候选集合并起来。
- 剪枝：删除被其他候选严格支配（所有实例不差且至少一处更好）的候选，得到 Pareto 前沿。
- 在前沿里抽样父代：更偏好“在更多实例上属于最优集合”的候选（出现频率高）。

### Reflection（反思）机制的关键点（Section 3.2）
- 把 rollout 的“自然语言轨迹”当作可学习信号：推理链、工具调用、工具输出、以及系统内部每个模块的输入/输出。
- 进一步强调：评测函数 μ 在执行过程中也会产生大量诊断文本（例如编译错误、运行日志、profiling 结果），这些在被压缩成标量 reward 前很有价值。
- 因此提出 μ_f：在返回 score 的同时返回 feedback_text（甚至能提供 module-level feedback，例如 multi-hop 每一跳的反馈）。
- LLM 基于这些轨迹做归因与改写：本质是“用语言做 credit assignment + targeted prompt update”。

### 论文声称的效果（从摘要提炼）
- GEPA 相比 GRPO：平均提升约 10%，最高到 20%，同时可用更少 rollouts（最高宣称 35× 更省）。
- 也超过 MIPROv2 等 prompt optimizer，并展示了在代码优化场景作为 inference-time 搜索策略的潜力。

### 与 AlphaEvolve / OpenEvolve 的关系（对照理解）
- GEPA 的“候选”是“多模块 prompts 的一组参数”；AlphaEvolve/OpenEvolve 的“候选”通常是“代码程序本身”。
- 但 GEPA 的两条核心信号在 OpenEvolve 里几乎都有对应物：
  - 系统执行轨迹：LLM 回复、diff、代码版本、历史尝试；
  - 评测轨迹：运行输出/报错/超时等 artifacts（如果 evaluator 把它们保留下来）。
- 因此把 GEPA 的 μ_f 思想迁移到 OpenEvolve 的一个直接方式是：让 evaluator 返回/存储更丰富的 artifacts（stderr、traceback、关键中间指标、数据切分信息等），并在每轮提示词里加入结构化 reflection 和记忆检索。

### 可借鉴的思路
- μ_f（反馈函数）思想：让 evaluator 除了返回 combined_score 外，还系统化返回 feedback_text/traces（stderr/traceback/关键中间指标/数据切分信息），并把它作为 artifacts 进入 prompt（OpenEvolve 已有 EvaluationResult.artifacts 管道，可直接用 evaluator.py + sampler.py 串起来）。
- 显式 Reflection 回路：每轮评测后新增一次“反思生成”（结构化 JSON：原因→归因→可执行下一步），写入 Program.metadata 或 artifacts；下一轮 prompt 检索并注入“最近反思/相似反思”（对应 GEPA 的“用语言做 credit assignment + targeted update”）。