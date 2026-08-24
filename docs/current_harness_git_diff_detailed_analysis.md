## 当前 harness 相比之前版本的功能优化：git diff 视角详细分析

### 分析范围与依据

- **当前代码范围**：`binderloop/`。
- **当前运行结果范围**：`outputs/sc2rbd_closed_loop_llm_10r_v13/`。
- **当前 HEAD**：`8cad92de3749f30055c8be123be82de42865ae63`。
- **对照方式**：当前执行环境无法直接运行 `git diff` 命令；本文采用等效 diff 证据进行分析：
  - 当前源码结构与旧版分析记录对照；
  - 历史提交中 `binderloop/active_learning/strategy.py`、`binderloop/orchestration/orchestrator.py`、`binderloop/config.py` 读取返回 `object not found`，说明当前闭环核心路径相对旧版本属于新增或大规模重组后的能力；
  - 对照 `docs/sc2rbd_closed_loop_llm_30r_v3_harness_improvement_analysis.md` 中记录的旧版本 5 轮/30 轮行为；
  - 结合 `outputs/sc2rbd_closed_loop_llm_10r_v13/` 中每轮真实产物，包括 `rollback_decision.json`、`fragment_templates.json`、`binder_length_recommendation.json`、`next_round_config_merge_report.json`、`iteration_metrics_stats.json`。

---

## 一、总体变化：从线性脚本升级为质量感知的闭环 agent harness

旧版本的主线更接近一次性或线性主动学习流程：生成 jobs、运行模型、收集结果、基于 top candidates 继续下一轮。当前版本的主线已经转移到 `BinderDesignOrchestrator`，形成了完整的多阶段闭环：

```text
DesignJob
  -> execution / retry / checkpoint
  -> ResultIngestionAgent
  -> EvaluationAgent
  -> StructureEvaluationAgent
  -> FragmentTemplateMiningAgent
  -> BinderLengthPolicyAgent
  -> RollbackController
  -> BinderQualityAnalysisAgent
  -> HypothesisAgent
  -> DiagnosticCoachAgent
  -> InputConfigurationAgent
  -> ActiveLearningPolicyAgent
  -> config merge / guardrail
  -> StrategyLevelActiveLearner
  -> next DesignJob
```

这意味着当前 harness 的优化目标不再只是“找出当前轮最好的候选”，而是进一步学习：

- 哪些 **策略臂** 更有效；
- 哪些 **片段/模板** 可以跨轮复用；
- 哪些 **参数变更** 应被采纳、限幅或拒绝；
- 哪些轮次属于真实质量退化，哪些属于基础设施/配置失败；
- 如何在失败后回滚到最佳分支并避免重复同一失败路径。

---

## 二、详细优化点

### 1. 新增质量感知回滚：避免退化分支被线性继承

#### diff 级变化

当前版本新增了 `binderloop/active_learning/rollback.py`，核心对象包括：

- `RoundOutcome`：记录每轮 `best_iptm`、`median_iptm`、`success_count`、`arm_signature`、是否为执行失败等；
- `RollbackDecision`：输出 `advance`、`rollback` 或 `stop`；
- `RollbackController`：追踪历史最优轮，判断当前轮是否相对最佳轮显著退化；
- `round_reward()`：不再只依赖单个 best iPTM，而是混合 top-k median iPTM、best iPTM 和 pass 数。

旧版问题是线性推进：如果第 N 轮之后明显塌陷，第 N+1 轮仍继续沿退化父本搜索。当前版本通过 reward 历史显式判定退化分支。

#### 机制优化

当前 reward 逻辑为：

```text
reward = median_weight * median_iptm
       + (1 - median_weight) * best_iptm
       + success_weight * success_count
```

这比旧式 best-only 更稳健，因为单个 lucky candidate 不会完全支配下一轮方向。

此外，当前版本把执行/配置失败从质量回滚中剥离：如果某轮因为 `boltzgen_config_error`、`missing_ceph_mount_secret`、资源排队等原因没有产生候选，它不会污染 reward 历史，也不会被误认为设计质量退化。

#### v13 运行证据

`outputs/sc2rbd_closed_loop_llm_10r_v13/round_08/rollback_decision.json` 显示第 8 轮是当前最佳轮：

- `best_iptm=0.58092`
- `median_iptm=0.54069`
- `success_count=3`
- `reward=0.852759`
- `decision.action=advance`

`round_09/rollback_decision.json` 显示第 9 轮明显回落：

- `best_iptm=0.19215`
- `median_iptm=0.15979`
- `success_count=0`
- `current_reward=0.169498`
- 相对第 8 轮 `relative_drop=0.801236`
- 当前决策为 `advance`，原因是 `consecutive_regressions=1`，仍在 `patience=2` 内。

这说明当前版本已经具备“识别质量退化但不过早回滚”的控制能力，避免了旧版本中单轮波动导致频繁 rollback 的问题。

---

### 2. 主动学习策略从单一 exploitation 扩展为多策略 arm

#### diff 级变化

当前版本新增/重构了 `StrategyLevelActiveLearner`，能够基于结构标签、假设、质量分析和回滚阻断信息组合策略 arm：

- `exploit_reliable_seed`
- `diversity_explore`
- `hotspot_repair`
- `foldability_repair`
- `interface_pose_repair`
- `clash_repair`
- `module_exploitation`
- `module_repair`
- `forced_branch_switch_explore`

旧版本更像按照 top candidates 顺序继续 exploitation；当前版本则把失败类型转成可执行的搜索臂。

#### 机制优化

当前策略有几项关键增强：

- **exploit/explore 分流**：根据 `exploration_ratio` 选择部分父本做 exploitation，剩余父本做 exploration；
- **回滚后阻断失败臂**：当回滚决策携带 `blocked_arm_signature` 时，下一轮会过滤同名 arm，防止重复同一失败分支；
- **模板半条件化**：如果 `binder_template` 被启用，只有约一半父本使用 template conditioning，另一半自动去掉 `binder_template`，避免一个坏模板杀死整轮；
- **结构标签驱动修复**：例如 hotspot miss 触发 `hotspot_repair`，clash risk 触发 `clash_repair`。

#### 实际意义

这使 harness 从“候选级排序器”升级成“策略级主动学习器”。它优化的不再只是某个设计分数，而是下一轮应该采取的搜索策略。

---

### 3. 新增 fragment template mining：把局部结构质量转成跨轮可复用模板

#### diff 级变化

当前版本新增/强化了 `FragmentTemplateMiningAgent`，将 `StructureEvaluationAgent` 识别出的高质量/低质量片段转成结构化 `FragmentTemplate`：

- `template_id`
- `source_structure_file`
- `binder_residue_span`
- `target_contact_residues`
- `hotspot_contacts`
- `contact_types`
- `quality_score`
- `interchain_pae`
- `binder_sequence`
- `ca_coordinates`

并写入每轮的 `fragment_templates.json`。

#### 机制优化

当前版本最重要的改动是：**preserve template 不再默认依赖全局 iPTM gate，而是默认使用 inter-chain PAE gate**。

具体规则：

- 默认 `fragment_template_gate=interchain_pae`；
- 当结构的 `min_design_to_target_pae <= 10.0` 时，才允许作为 preserve 模板；
- `iptm` gate 保留为 legacy 选项，但默认关闭；
- 只允许 `FragmentTemplateMiningAgent` 产生内部可执行的 `binder_template`，LLM 输出不能直接注入 `binder_template`。

这个改动解决了两个旧问题：

1. **全局 iPTM 对局部界面质量不敏感**：硬靶标上全局 iPTM 可能偏低，但局部界面 PAE 已经可信；
2. **模板来源不安全**：旧式路径可能指向 Taiji package 内部中间产物，下一轮无法重新挂载，导致空轮或配置失败。

当前版本通过 `_is_mountable_source()` 检查模板源是否真实存在、是否不在不可重挂载路径中，避免 `missing_ceph_mount_secret` / `boltzgen_config_error` 类失败。

#### v13 运行证据

`round_08/fragment_templates.json` 中出现大量 `reuse_mode=preserve` 的高质量 fragment，例如：

- 质量分 `quality_score=0.939`；
- 多个模板具有 `contacts_target_hotspot` 证据；
- 多个模板的 `interchain_pae` 在 `3.56–9.36` 区间，满足默认 `<=10.0` 的 preserve gate；
- 热点接触包括 `E:153`、`E:157`、`E:162`、`E:168`、`E:173`。

这说明 v13 中 fragment mining 不只是生成报告，而是在真实运行中形成了可复用局部结构记忆。

---

### 4. 新增 binder 长度自动策略：从固定长度枚举变成结构质量驱动的长度选择

#### diff 级变化

当前版本新增 `BinderLengthPolicyAgent`，每轮输出 `binder_length_recommendation.json` 并作为 `binder_length_policy` 来源参与下一轮配置合并。

旧版本通常依赖固定 `binder_lengths` 或 LLM 直接提出长度；当前版本会读取结构评价中的：

- chain break / reliability；
- weak or tiny interface；
- interface residue count；
- clash density；
- inter-chain PAE；
- 每个长度 bucket 的质量分。

#### 机制优化

长度策略遵循可审计规则：

- foldability failure 高：向短长度移动；
- weak/tiny interface 高且 folding 可接受：向长长度移动；
- 某个长度 bucket 明显更优：聚焦该长度附近；
- 无明显信号：保持当前范围；
- 永远被 `binder_length_range` 和全局 `[30, 180]` 限制。

这让 binder length 不再只是静态搜索维度，而成为可学习、可受约束的策略变量。

#### v13 运行证据

`round_09/next_round_config_merge_report.json` 显示：

- 输入配置希望锁定 `binder_length_range=[80,80]`；
- policy proposal 曾尝试改为 `[90]`；
- hard constraint guardrail 记录：`binder_length_range` 因 `freeze_binder_length_range` 被保留为 `[80]`；
- 说明当前版本既允许策略建议长度变化，也能尊重用户冻结边界。

---

### 5. 新增配置契约与物理 guardrail：防止 LLM 参数漂移破坏搜索

#### diff 级变化

当前版本新增/强化 `config_parameter_contract.py`，定义：

- `ADJUSTABLE_CONFIG_PARAMETERS`：LLM/agent 可以调整的白名单字段；
- `INTERNAL_ONLY_CONFIG_PARAMETERS`：仅 orchestrator 内部可注入的字段，如 `binder_template`；
- `PARAM_BOUNDS`：关键数值参数硬边界；
- `clamp_config_with_inertia()`：对每轮参数变化施加硬边界和步长惯性。

旧版本中 LLM 可能直接生成不可执行字段或过激参数，导致某轮整体塌陷。当前版本把“建议”与“可执行配置”之间加入契约层。

#### 关键保护

当前 hard bounds 包括：

| 参数 | 当前限制 |
| --- | --- |
| `alpha` | `0.001–0.05`，且每轮最多 3 倍变化 |
| `exploration_ratio` | `0.20–0.60`，每轮最多绝对变化 `0.15` |
| `noise_scale` | `0.6–0.9`，每轮最多绝对变化 `0.15` |
| `step_scale` | `0.6–1.0`，每轮最多绝对变化 `0.2` |
| `hotspot_weight` | `0.5–3.0` |
| `refolding_rmsd_threshold` | `1.0–4.0` |

#### v13 运行证据

`round_09/next_round_config_merge_report.json` 明确记录了 guardrail 生效：

- `hotspot_weight` 从 `5.4` 被 clamp 到 `3.0`，原因是 `above max 3.0`；
- `noise_scale` 从 `0.8` 被 clamp 到 `0.75`，原因是 `step capped to +0.15 from current 0.6`；
- `num_designs=30` 被 hard constraint 还原为 `80`，原因是 round budget frozen；
- `binder_length_range=[90]` 被 hard constraint 拒绝，保留用户冻结范围。

这说明当前版本不再盲信 LLM 或策略 agent 的输出，而是把所有配置更新变成可审计、可限幅、可追责的 merge 决策。

---

### 6. 多来源配置合并从隐式覆盖变成可审计 provenance

#### diff 级变化

当前 `BinderDesignOrchestrator._merge_next_round_updates()` 将以下来源统一合并：

1. `input_configuration`
2. `binder_length_policy`
3. `policy_proposal`
4. `fragment_template_mining`

每个 key 的来源、覆盖关系、被忽略字段、hard constraint freeze、physical clamp 都记录在 `next_round_config_merge_report.json`。

#### v13 运行证据

`round_09/next_round_config_merge_report.json` 中可以看到：

- `policy_proposal` 覆盖了 `input_configuration` 的 `hotspot_weight`、`diffusion_batch_size`、`inverse_fold_num_sequences`、`additional_filters`、`binder_chain`、`binder_lengths` 等；
- `fragment_template_mining` 又把 `run_filtering` 从 `false` 覆盖回 `true`，并强化 `clash_filter`、`module_guided_repair`；
- `applied_sources` 清楚标出最终每个参数来自哪个模块。

这种 provenance 对调试非常重要：当某轮失败时，可以准确知道失败配置来自 LLM、policy、length policy 还是 fragment mining。

---

### 7. 模块输出验证、checkpoint、resume/retry 保护增强

#### diff 级变化

当前 orchestrator 引入 `_run_validated_module()`，对每个模块输出做 schema/存在性验证：

- `execution_records.json`
- `ingestions.json`
- `evaluation_summary.json`
- `structure_evaluation.json`
- `fragment_templates.json`
- `binder_length_recommendation.json`
- `binder_quality_analysis.json`
- `hypotheses.json`
- `diagnostic_report.json`
- `next_round_input_configuration.json`
- `next_round_parameter_proposal.json`
- `next_round_config_merge_report.json`
- `next_jobs.json`

每轮写 `round_checkpoint.json`，执行阶段写 `execution_attempts.json`。

#### 优化价值

这使当前 harness 具备：

- 单模块输出不合法时可重试；
- 运行中断后可从 completed checkpoint 恢复；
- 未完成 attempt 会阻止重复提交，避免 Taiji 任务重复消耗资源；
- 每轮完整 artifact 被记录，便于复盘。

`outputs/sc2rbd_closed_loop_llm_10r_v13/` 的每轮目录均包含上述核心产物，说明该能力已经在真实运行中落地。

---

### 8. 资源与多 GPU 执行约束更安全

#### diff 级变化

当前 orchestrator 在 `_enforce_round_cap()` 中对多 GPU 场景做特殊处理：

- 当 `host_gpu_num > 1` 时，`max_parallel` 被限制为 `1`；
- 多 GPU BoltzGen job 在单个 Taiji 任务内做 GPU shard，而不是并行提交多个 8-GPU 任务；
- 每轮 `num_designs`、`num_designs_per_round`、`max_binders_per_round` 被统一 cap。

这避免了旧版本可能出现的资源重复申请、预算超限或多个 Taiji job 抢占同一 GPU 组的问题。

---

## 三、v13 指标层面的效果

### 关键轮次表现

来自 `outputs/sc2rbd_closed_loop_llm_10r_v13/iteration_metrics_stats.json`：

| 轮次 | best design-to-target iPTM | mean design-to-target iPTM | best min design-to-target PAE | 说明 |
| --- | ---: | ---: | ---: | --- |
| round 0 | `0.26736` | `0.10334` | `11.0086` | 初始轮，界面信号较弱 |
| round 3 | `0.57057` | `0.16219` | `3.57591` | 明显爬坡，局部界面 PAE 大幅改善 |
| round 4 | `0.58544` | `0.14815` | `3.41004` | best iPTM 达到全局最高附近 |
| round 8 | `0.58092` | `0.15516` | `3.56364` | reward 最佳，且有 `success_count=3` |
| round 9 | `0.19215` | `0.09903` | `15.34992` | 明显退化，被 rollback controller 标记为 80% drop |

### 主要收益

- **早期爬坡明显**：best iPTM 从 round 0 的 `0.26736` 提升到 round 4 的 `0.58544`。
- **局部界面 PAE 明显改善**：best min design-to-target PAE 从 round 0 的 `11.0086` 降到 round 4 的 `3.41004`。
- **reward 更稳健**：第 8 轮不是单纯 best iPTM 最高，但因为 `median_iptm=0.54069` 且 `success_count=3`，被识别为综合最优轮。
- **退化可被识别**：第 9 轮 iPTM 和 PAE 双重退化，系统明确记录相对第 8 轮 reward 下降约 `80%`。
- **策略执行可审计**：每轮都生成 rollback、template、length、config merge、diagnostic 等产物。

### 仍然存在的问题

- **退化后尚未完成回滚动作**：第 9 轮只是第一轮 regression，尚在 `patience=2` 内；如果第 10 轮继续退化才会触发 rollback。
- **策略仍偏 exploitation**：第 8、9 轮 `arm_signature` 都是 `exploit_reliable_seed`，多 arm 机制具备，但实际运行中仍可能被可靠种子利用主导。
- **模板路径安全与模板有效性仍需继续验证**：当前已经有 `_is_mountable_source()` 防止不可挂载模板成为 `binder_template`，但 `fragment_templates.json` 仍会记录来自 Taiji package 深层目录的分析模板；这些可以用于分析，但不能直接作为可执行模板。
- **热点约束仍是核心难点**：虽然 fragment 中记录到多个 hotspot contacts，但是否稳定转化为下一轮全局 pass 仍未完全解决。

---

## 四、与之前版本相比的功能优化总结

| 维度 | 之前版本 | 当前版本 |
| --- | --- | --- |
| 闭环推进 | 线性从上一轮继续 | 基于 reward 历史做 advance / rollback / stop |
| reward | 偏 best candidate | 混合 top-k median、best iPTM、success_count |
| 失败处理 | 容易把执行失败混作质量失败 | 执行/配置失败从 reward 中排除 |
| 主动学习 | 主要 exploitation / 简单探索 | 多 arm：hotspot、foldability、pose、clash、module repair/exploit |
| 模板复用 | 局部片段难以安全执行化 | interchain PAE gated fragment template + mountable source check |
| 长度策略 | 固定或 LLM 建议 | 结构质量驱动的 `BinderLengthPolicyAgent` |
| 配置更新 | 隐式覆盖，易漂移 | 白名单契约 + hard constraint + physical guardrail + provenance |
| LLM 输出 | 可能产生不可执行字段或过激参数 | `supported_config_changes` 过滤并 clamp |
| 运行鲁棒性 | 中断/重复提交风险较高 | checkpoint、attempt ledger、模块输出验证和 retry |
| 审计产物 | 局部产物 | 每轮完整 artifact chain 和 merge report |

---

## 五、结论

当前 harness 相比之前版本的核心优化可以概括为：**从线性候选筛选脚本，升级为带质量回滚、策略级主动学习、片段模板记忆、自动长度策略、配置契约、物理 guardrail 和完整审计链路的闭环设计系统**。

从 `sc2rbd_closed_loop_llm_10r_v13` 的实际结果看，这些改动已经产生明确效果：系统在 round 3/4/8 找到显著优于初始轮的设计分支，best iPTM 从 `0.26736` 提升到 `0.58544`，best min design-to-target PAE 从 `11.0086` 改善到 `3.41004`，并在 round 8 形成 reward 最佳轮。更重要的是，round 9 的退化没有被静默继承，而是被明确记录为相对 best round 的 `80%` reward drop。

下一步最值得继续优化的是：在触发 rollback 后强制更多非 exploitation arm、进一步降低模板路径不可执行风险、增强 hotspot engagement 的稳定转化，并对不同策略 arm 做消融统计，以判断哪些机制真正贡献了 round 8 的成功。