## 当前 harness 相比之前版本的优化：简洁版

### 分析对象

- **代码范围**：`binderloop/`
- **运行结果**：`outputs/sc2rbd_closed_loop_llm_10r_v13/`
- **当前 HEAD**：`8cad92de3749f30055c8be123be82de42865ae63`
- **说明**：当前环境不能直接执行 `git diff`，因此本文基于当前源码、历史引用读取结果、已有改进分析文档和 `10r_v13` 实际输出进行等效 diff 分析。

---

## 一句话结论

当前 harness 已经从旧版的**线性候选筛选/简单主动学习流程**，升级为一个具备**质量回滚、策略级主动学习、fragment 模板记忆、自动长度选择、配置契约、物理 guardrail、checkpoint/retry 和完整审计链路**的闭环 agent 系统。

---

## 核心优化概览

| 优化方向 | 之前版本 | 当前版本 |
| --- | --- | --- |
| 闭环推进 | 基本从上一轮线性继续 | 基于 reward 决定 `advance` / `rollback` / `stop` |
| 质量信号 | 偏单个 best candidate | 混合 best iPTM、top-k median iPTM、success count |
| 失败处理 | 执行失败容易污染质量判断 | 执行/配置失败从 reward 历史中排除 |
| 主动学习 | 简单 exploitation/exploration | 多策略 arm：hotspot、foldability、pose、clash、module repair/exploit |
| fragment 复用 | 局部片段难以跨轮执行化 | `FragmentTemplateMiningAgent` 生成跨轮模板库 |
| 模板门控 | 易受全局 iPTM 或路径问题影响 | 默认用 inter-chain PAE gate，并检查模板源是否可挂载 |
| 长度选择 | 固定长度或 LLM 直接建议 | `BinderLengthPolicyAgent` 基于结构质量自动推荐长度 |
| 配置更新 | 隐式覆盖，难审计 | 多来源 merge report，记录来源、覆盖、拒绝和 clamp |
| LLM 安全 | 可能输出不可执行或过激参数 | 白名单配置契约 + physical guardrail |
| 运行鲁棒性 | 中断恢复和重复提交风险较高 | checkpoint、attempt ledger、模块验证、retry |

---

## 关键优化点

### 1. 质量回滚机制

当前新增 `RollbackController`，每轮写出 `rollback_decision.json`。

它解决的问题是：旧版一旦某轮退化，下一轮仍可能继续沿退化分支搜索。

当前机制会记录：

- 本轮 best iPTM；
- top-k median iPTM；
- pass candidate 数量；
- 当前策略 arm；
- 是否执行失败；
- 相对历史最佳轮的 reward drop。

在 `10r_v13` 中：

- 第 8 轮是当前最佳轮：`reward=0.852759`，`best_iptm=0.58092`，`median_iptm=0.54069`，`success_count=3`；
- 第 9 轮明显退化：`reward=0.169498`，相对第 8 轮下降约 `80%`；
- 系统已识别该退化，但因仍在 `patience=2` 的第一轮内，所以暂时继续观察。

---

### 2. 策略级主动学习增强

当前 `StrategyLevelActiveLearner` 不只是选择 top candidates，而是根据失败模式生成不同策略臂：

- `exploit_reliable_seed`
- `diversity_explore`
- `hotspot_repair`
- `foldability_repair`
- `interface_pose_repair`
- `clash_repair`
- `module_exploitation`
- `module_repair`

此外，回滚后可以阻断失败 arm，避免重复走同一条退化路径。

这使 harness 的学习对象从“哪个候选好”升级为“哪类设计策略有效”。

---

### 3. fragment template mining

当前新增/强化 `FragmentTemplateMiningAgent`，把高质量局部结构片段转成 `fragment_templates.json`。

主要改进：

- 默认用 `min_design_to_target_pae <= 10.0` 作为 preserve template gate；
- 保留 fragment 的热点接触、接触类型、质量分、序列和 CA 坐标；
- 形成跨轮 template library；
- 只有 orchestrator 内部可注入真正可执行的 `binder_template`，LLM 不能直接写入；
- 检查模板源是否可挂载，避免旧版本常见的 Taiji package 深层路径导致空轮。

在 `round_08/fragment_templates.json` 中，多个 preserve fragment 的 `interchain_pae` 位于 `3.56–9.36` 区间，并包含 `E:153`、`E:157`、`E:162`、`E:168`、`E:173` 等 hotspot contact 证据。

---

### 4. 自动 binder 长度策略

当前新增 `BinderLengthPolicyAgent`，每轮输出 `binder_length_recommendation.json`。

它会根据结构质量判断下一轮长度应：

- 变短：当 chain break、foldability failure 多；
- 变长：当 interface weak/tiny 但折叠还可以；
- 聚焦：当某个长度 bucket 明显更好；
- 保持：当没有明显信号。

同时，它不会突破用户冻结的 `binder_length_range`。

在 `round_09/next_round_config_merge_report.json` 中，policy 尝试将长度范围改到 `[90]`，但 hard constraint 将其拒绝，保留用户冻结范围，说明当前版本既能学习长度，又能守住用户约束。

---

### 5. 配置契约与物理 guardrail

当前新增/强化 `config_parameter_contract.py`。

核心作用：

- 只允许 agent 修改白名单字段；
- LLM 不能直接注入内部执行字段，例如 `binder_template`；
- 对关键数值参数加硬边界和每轮变化限制。

典型边界：

- `alpha`: `0.001–0.05`
- `exploration_ratio`: `0.20–0.60`
- `noise_scale`: `0.6–0.9`
- `step_scale`: `0.6–1.0`
- `hotspot_weight`: `0.5–3.0`

在 `round_09/next_round_config_merge_report.json` 中：

- `hotspot_weight=5.4` 被 clamp 到 `3.0`；
- `noise_scale=0.8` 被 clamp 到 `0.75`；
- `num_designs=30` 因预算冻结被还原为 `80`；
- `binder_length_range=[90]` 因长度范围冻结被拒绝。

这显著降低了 LLM 参数漂移导致整轮崩溃的风险。

---

### 6. 可审计的多来源配置合并

当前下一轮配置来自多个模块：

1. `InputConfigurationAgent`
2. `BinderLengthPolicyAgent`
3. `ActiveLearningPolicyAgent`
4. `FragmentTemplateMiningAgent`

最终合并结果写入 `next_round_config_merge_report.json`，其中记录：

- 每个 key 的来源；
- 哪个来源覆盖了哪个来源；
- 哪些 key 被 hard constraint 拒绝；
- 哪些数值被 physical guardrail clamp；
- 最终 `applied_update`。

这让每轮配置变化具备完整 provenance，便于复盘失败原因。

---

### 7. checkpoint、retry 与模块验证

当前 orchestrator 对每个模块输出做验证，并为每轮写出：

- `round_checkpoint.json`
- `execution_attempts.json`
- `execution_records.json`
- `ingestions.json`
- `evaluation_summary.json`
- `structure_evaluation.json`
- `fragment_templates.json`
- `binder_length_recommendation.json`
- `binder_quality_analysis.json`
- `diagnostic_report.json`
- `next_round_config_merge_report.json`
- `next_jobs.json`

这使长程闭环更适合真实 Taiji/GPU 运行：可恢复、可审计、能避免重复提交未完成任务。

---

## `10r_v13` 效果摘要

来自 `iteration_metrics_stats.json`：

| 轮次 | best iPTM | mean iPTM | best min PAE | 说明 |
| --- | ---: | ---: | ---: | --- |
| round 0 | `0.26736` | `0.10334` | `11.0086` | 初始状态 |
| round 3 | `0.57057` | `0.16219` | `3.57591` | 明显爬坡 |
| round 4 | `0.58544` | `0.14815` | `3.41004` | best iPTM 最高 |
| round 8 | `0.58092` | `0.15516` | `3.56364` | reward 最佳，`success_count=3` |
| round 9 | `0.19215` | `0.09903` | `15.34992` | 明显退化，被识别为 80% drop |

### 正面效果

- best iPTM 从 round 0 的 `0.26736` 提升到 round 4 的 `0.58544`；
- best min design-to-target PAE 从 `11.0086` 改善到 `3.41004`；
- 第 8 轮形成综合最佳 reward，并产生 `3` 个 pass candidate；
- 第 9 轮退化被系统明确识别，没有被静默当作正常主线；
- 每轮产物链完整，支持复盘和恢复。

### 主要不足

- 当前实际策略仍偏 `exploit_reliable_seed`，多 arm 机制还需要更强制的多样化调度；
- 第 9 轮退化后尚未进入第二个 regression，因此还未真正触发 rollback；
- fragment 模板能识别局部热点接触，但能否稳定转化为后续 pass 仍需继续验证；
- 对不同策略 arm 的贡献还缺少消融统计。

---

## 总结

当前版本 harness 的最大价值不是单个指标提升，而是建立了一个更可靠的闭环控制面：

- **知道什么时候退化**；
- **知道为什么退化**；
- **知道哪些配置改动来自哪里**；
- **能限制危险参数漂移**；
- **能把结构片段转成跨轮记忆**；
- **能用 checkpoint/retry 支撑长程运行**。

因此，相比之前版本，当前 harness 已经从实验脚本形态进入了可审计、可恢复、可扩展的策略级 binder design harness 阶段。