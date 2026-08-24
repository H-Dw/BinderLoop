# binderloop agents 与 harness 改进分析

## 1. 当前 harness 使用了哪些 agents

当前项目有两条相关路径：一条是较早的简易 `run_pipeline` 路径，另一条是当前更完整的闭环 `BinderDesignOrchestrator` 路径。真正承担闭环 agent 编排的是 `binderloop/orchestration/orchestrator.py`，入口通常来自 `scripts/run_closed_loop_orchestrator.py`。

### 1.1 闭环主流程中的 agents

- `StrategyLevelActiveLearner`
  - 位置：`binderloop/active_learning/strategy.py`
  - 作用：生成初始 `DesignJob`，并在每轮结束后基于 top candidates、结构标签、hypotheses、quality guidance 和 policy update 生成下一轮 jobs。
  - 它不是以 `Agent` 命名，但在闭环中承担策略层 active learning 调度角色。

- `ResultIngestionAgent`
  - 位置：`binderloop/agents/result_ingestion_agent.py`
  - 作用：扫描 BoltzGen 输出目录，收集 metrics CSV、final designs、intermediate dirs、log tail 和 run-level issues。
  - 输出：`IngestedBoltzGenRun`，随后写入每轮 `ingestions.json`。

- `EvaluationAgent`
  - 位置：`binderloop/agents/evaluation_agent.py`
  - 作用：把 BoltzGen metrics 映射成统一指标，计算 weighted score，并用 failure taxonomy 标记 `hotspot_miss`、`folding_failure`、`binding_pose_failure` 等。
  - 输出：`EvaluationSummary`，包含 top candidates、failed examples、tag counts 和 observations。

- `StructureEvaluationAgent`
  - 位置：`binderloop/agents/structure_evaluation_agent.py`
  - 作用：读取 PDB/CIF/mmCIF，提取 binder-target 接触、热点覆盖、clash、chain break、局部 fragment quality。
  - 输出：`StructureBatchEvaluation`，包括 aggregate tags、reliable seed fraction、每个结构的 high/low quality fragments。

- `BinderQualityAnalysisAgent`
  - 位置：`binderloop/agents/binder_quality_analysis_agent.py`
  - 作用：把 metric 与 coordinate-level fragment evidence 转成质量分析。配置 LLM 时走 LLM，否则走确定性 fallback。
  - 输出：`BinderQualityAnalysis`，包括 `high_quality_modules`、`low_quality_modules`、`causal_factors`、`next_round_guidance`。

- `HypothesisAgent`
  - 位置：`binderloop/agents/hypothesis_agent.py`
  - 作用：提出失败假设与下一轮干预建议。配置 LLM 时走 LLM，否则根据 failure tags 和 structural tags 走规则 fallback。
  - 输出：`HypothesisSet`，其中 hypothesis 可携带经过白名单过滤的 `config_parameter_changes`。

- `DiagnosticCoachAgent`
  - 位置：`binderloop/agents/diagnostic_coach_agent.py`
  - 作用：综合执行状态、metrics summary、evaluation、structural analysis、历史 job 记录和当前 config，做 pipeline health 与 corrective actions。
  - 输出：`DiagnosticReport`，包括 root causes、metric interpretation、corrective actions、monitoring recommendations、pipeline health。

- `InputConfigurationAgent`
  - 位置：`binderloop/agents/input_configuration_agent.py`
  - 作用：根据 diagnostic、evaluation、structure、quality、hypotheses、memory 与 constraints 生成下一轮可执行配置。
  - 输出：`InputConfiguration`，其中 `recommended_config` 会通过 `supported_config_changes` 过滤。

- `ActiveLearningPolicyAgent`
  - 位置：`binderloop/agents/active_learning_policy_agent.py`
  - 作用：将 evaluation tags、structural tags、quality modules、diagnostic actions、hypotheses 和 memory 转成下一轮 BoltzGen 参数更新。
  - 输出：`NextRoundParameterProposal`，随后写入 `next_round_parameter_proposal.json` 和 `next_round_config.yaml`。

### 1.2 执行相关 agents

这些 agent 主要在 `scripts/run_closed_loop_orchestrator.py` 的 executor 中使用，而不是在 `BinderDesignOrchestrator.__init__` 内部直接实例化。

- `DesignSpecAgent`
  - 位置：`binderloop/agents/design_spec_agent.py`
  - 作用：把 `DesignJob + params` 翻译成自包含 BoltzGen project package、design spec、run script、expected outputs 和 manifest。
  - 关键点：支持多 GPU shard，把 `num_designs` 拆到每个 GPU 的子进程中运行。

- `TaijiExecutionAgent`
  - 位置：`binderloop/agents/taiji_execution_agent.py`
  - 作用：生成 Taiji simple/full config，合并模板与资源选项，提交 `taiji_client start`。
  - 输出：`TaijiSubmitSpec` 和 `TaijiSubmissionRecord`。

- `RunMonitorAgent`
  - 位置：`binderloop/agents/run_monitor_agent.py`
  - 作用：单次查询 Taiji instance detail/logs，推断状态，检查 expected outputs，给出 failure hints。
  - 注意：它本身不做循环 poll，循环由 `_wait_for_taiji_completion` 承担。

### 1.3 简易 pipeline 路径中的 agents

- `DesignParameterAgent`
  - 位置：`binderloop/agents/design_parameter_agent.py`
  - 作用：把紧凑 YAML 配置扩展为 BoltzGen 参数计划。
  - 当前主要用于较早的测试脚本和简易路径，闭环 orchestrator 目前直接用 `_base_params()` 与后续 policy/input config 更新。

- `run_pipeline`
  - 位置：`binderloop/pipeline.py`
  - 特征：直接用 `StrategyLevelActiveLearner` 生成初始 jobs，再用 model adapter 构造命令并逐个执行。它没有闭环分析、memory、message bus、diagnostic 或 LLM agent。

## 2. agents 之间的信息交互与上下文关联

### 2.1 主数据流

每一轮的核心顺序是：

1. `ExperimentMemoryStore.load()` 读取跨轮 memory。
2. `StrategyLevelActiveLearner.initial_jobs()` 或上一轮 `propose_next()` 给出当前 round 的 `DesignJob`。
3. `_run_jobs()` 调用 executor，executor 内部可能使用 `DesignSpecAgent`、`TaijiExecutionAgent`、`RunMonitorAgent`。
4. `ResultIngestionAgent` 从每个 job 的 output/log 中提取 candidates 和 artifacts。
5. `EvaluationAgent` 对 candidates 评分、排序、打 failure tags。
6. `StructureEvaluationAgent` 对 final design structures 做 coordinate-level 分析。
7. orchestrator 组装一个统一 `context`：
   - `round_id`
   - `evaluation`
   - `structural_analysis`
   - `memory`
   - `target_analysis`
   - `current_config`
   - `constraints`
   - 当前 round 的 `messages`
8. `BinderQualityAnalysisAgent` 基于这个 context 输出质量分析。
9. `HypothesisAgent` 在加入 `quality_analysis` 后提出 failure hypotheses。
10. `DiagnosticCoachAgent` 使用 monitor snapshot、metrics summary、evaluation、structure、history、config 做诊断。
11. `InputConfigurationAgent` 生成下一轮 recommended config。
12. `ActiveLearningPolicyAgent` 汇总 evaluation、structure、hypotheses、quality、diagnostic、memory，生成最终 params update。
13. `_apply_next_round_update()` 把 input config 与 policy proposal 合并进 `HarnessConfig`。
14. `StrategyLevelActiveLearner.propose_next()` 生成下一轮 jobs。
15. 当前轮所有 artifacts、decisions、messages 回写到 `ExperimentMemoryStore`。

### 2.2 MessageBus 的设置

- `MessageBus` 是 append-only JSONL：默认文件为输出目录下的 `agent_messages.jsonl`。
- `AgentMessage` 支持 `sender`、`recipient`、`message_type`、`round_id`、`job_id`、`correlation_id`、`parent_id`、`confidence`、`requires_response`、`artifacts`。
- 当前实际用途偏轻量：
  - round started status。
  - job execution/retry status，recipient 常写成 `RunMonitorAgent` 或 `all`。
  - diagnostic summary。
  - iteration plot status。
- 当前并不是事件驱动的 agent-to-agent 对话；大多数 agent 不订阅消息，也不直接消费某个 sender 的输出。消息主要由 orchestrator 查询后塞进 LLM/analysis context。

### 2.3 Memory 的设置

- `ExperimentMemoryStore` 持久化到 `outputs/.../memory/experiment_memory.json`。
- `RoundRecord` 保存 jobs、ingestion、evaluation、structural_analysis、quality_analysis、hypotheses、decisions、retry_events、artifacts。
- `summarize_for_agent()` 默认提供最近 5 轮和最近 50 条消息。
- 这让 LLM 与规则 agent 能看到短期历史，但还不是完整的实验数据库，也没有 candidate-level lineage、parameter-to-outcome 的结构化因果表。

### 2.4 上下文关联是否合理

整体上，对简易闭环 harness 来说是合理的：

- 每个 agent 都返回 dataclass/JSON，可审计、可落盘、可恢复。
- LLM agent 都有 deterministic fallback，测试与离线运行不会被 LLM 可用性卡住。
- LLM 产生的 config 变更会经过 `config_parameter_contract.py` 的白名单过滤，避免把不可执行建议直接写入配置。
- context 中同时包含 metric、structure、memory、target、constraints，足够支撑一轮“观察 -> 假设 -> 诊断 -> 更新”的基本闭环。

但当前上下文设计仍有明显简化：

- 统一大 context 由 orchestrator 手工拼接，缺少版本化 schema 与字段级 provenance。
- `AgentMessage.parent_id/correlation_id` 设计了但基本未形成链式调用关系，难以追踪“某条假设来自哪个结构片段/哪个失败样本”。
- `MessageBus.publish()` 没有显式锁；`_run_jobs()` 中多个 worker 可能并发写入同一 JSONL，虽然单行追加通常可用，但不是严格的跨平台并发日志协议。
- Quality、Hypothesis、Diagnostic、InputConfiguration、Policy 之间有较多职责重叠，尤其是都可能提出参数变更；最终合并策略是“后写覆盖 + 白名单过滤”，不是显式仲裁或多目标优化。
- `DiagnosticCoachAgent` prompt 中含有 IL-17A 相关硬编码知识，这对特定任务有帮助，但对通用 binder harness 可能造成 target-specific leakage。

## 3. 并行执行逻辑是否合理

### 3.1 当前并行设置

- `BinderDesignOrchestrator.__init__` 中：
  - `requested_parallel = max_parallel or cfg.runtime.max_parallel or 1`
  - `self.max_parallel = min(requested_parallel, cfg.resource.max_parallel_jobs)`
  - 如果 `cfg.resource.host_gpu_num > 1`，强制 `self.max_parallel = 1`
- `_run_jobs()` 使用 `concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel)` 并行执行 jobs。
- 每个 job 内部有 attempt ledger：
  - `execution_attempts.json` 记录 started/finished/terminal attempt。
  - 如果发现上次 started 但未结束，会拒绝提交新 job，避免重复占用远程资源。
  - `max_retries` 表示每个 job 最大 submit attempts，包含首次提交。
- Taiji backend 中，如果 `--submit` 且没有 `--no-wait-taiji`，每个 worker 会在 `_wait_for_taiji_completion()` 内轮询直到终态或 timeout。
- 多 GPU 时，job-level 并行被关闭，而 `DesignSpecAgent` 生成的 shell 会在单个 Taiji task 内按 GPU shard 并行跑多个 BoltzGen 子进程。

### 3.2 合理之处

- 对 Taiji/GPU 资源来说，`host_gpu_num > 1` 时只提交一个 job 是保守且合理的，避免多个 Taiji task 都申请整组 GPU。
- attempt ledger 能防止恢复时重复提交未完成任务，这是远程队列场景中非常关键的安全保护。
- job 级并行与 GPU shard 并行被分开处理，避免“外层并行 + 内层占满 GPU”的过度并发。
- `max_binders_per_round` 会在 `_enforce_round_cap()`、`_base_params()`、`_apply_next_round_update()` 多处 clamp，能减少 LLM 或配置误放大采样预算的风险。

### 3.3 不足

- 并行粒度只覆盖 job execution；ingestion、evaluation、structure analysis、quality/hypothesis/diagnostic/policy 都是串行。
- `EvaluationAgent` 和 `StructureEvaluationAgent` 在 ingestion 后彼此相对独立，可以并行；Quality、Hypothesis、Diagnostic 的部分输入也可拆成 DAG 并行。
- Taiji 轮询发生在 worker 内，会长期占住线程；如果后续需要高并发远程任务，最好改为异步 job state machine。
- `ThreadPoolExecutor.map()` 会按输入顺序返回，失败/慢任务可能延迟全轮进入分析；可以增加 partial result ingestion 与 early triage。
- 没有全局资源调度器，例如每个 job 的 GPU/CPU/内存需求、Taiji quota、queue 状态、cache 命中等都未纳入调度决策。

## 4. 热门 harness 项目的实施手段对比

### 4.1 LLM/agent eval harness 的共性

- EleutherAI `lm-evaluation-harness`
  - 特点：任务注册、统一模型 backend、标准化 few-shot/metrics、结果可复现。
  - 对本项目启发：需要把 target/task、scorer、backend、metrics 变成插件化 registry，而不是散落在 YAML 与 agent 规则里。

- OpenAI Evals / Promptfoo / DeepEval 类框架
  - 特点：YAML/配置驱动 eval、grader 可组合、适合 CI gating 与 prompt/model 对比。
  - 对本项目启发：每轮 binder design 可以定义 pass/fail gates、回归测试、参数 sweep 与自动报告，而不仅是单次实验输出。

- Inspect AI
  - 特点：`Dataset -> Solver -> Scorer` 分层，支持 tool-use agent、多 agent、sandbox、model-graded scoring、日志查看与执行限制。
  - 对本项目启发：可将 binder harness 重构为 `TargetDataset -> DesignSolver/Backend -> StructureScorer/ExperimentalProxyScorer`，并把每个 agent 的输入输出作为可复用 solver/scorer 组件。

- SWE-bench harness
  - 特点：Docker 隔离环境、per-instance 并行、cache level、timeout、日志与最终 report。
  - 对本项目启发：binder harness 也需要更强的环境封装、artifact cache、失败日志规范和 resource-aware parallel execution。

### 4.2 蛋白 binder 设计 harness / pipeline 的共性

- RFdiffusion + ProteinMPNN + AF2/Boltz verifier 是常见多阶段范式：
  - backbone generation。
  - sequence design。
  - structure prediction / refolding / complex validation。
  - metric filtering and ranking。
- ProteinDJ、nf-binder-design 等 pipeline 强调：
  - Nextflow/容器化/HPC 可扩展。
  - 多工具 backend：RFdiffusion、BindCraft、BoltzGen、ProteinMPNN/FAMPNN、AF2、Boltz-2。
  - GPU 与 CPU 并行、批量筛选、HTML/TSV 报告。
- ProtDBench 类 benchmark 强调：
  - 固定 target suite。
  - 统一 verifier protocol。
  - 成功率、throughput、sequence diversity、structural consistency 等多指标。
  - 对 filtering threshold 和 verifier choice 的敏感性分析。

## 5. 当前简易版 harness 的主要不足

### 5.1 Benchmark 与实验设计层

- 缺少 target/task registry：目前更像单任务闭环实验，不像可以横向比较多个 targets、backends、strategies 的 benchmark harness。
- 缺少固定 splits、baseline configs、golden expected outputs 与回归测试集。
- 缺少多目标评估：当前 score 偏 weighted single score，尚未系统纳入 novelty、diversity、developability、liability、throughput、cost、uncertainty。
- 缺少统计置信度：没有 bootstrap、重复 seed 方差、ablation 或 significance analysis。

### 5.2 执行与可复现层

- 本地/远程环境依赖较多，尚未达到 SWE-bench/Inspect 那种强隔离、可缓存、可重建的 sandbox/container harness。
- Taiji 远程提交已有 packaging，但 artifact lineage、环境指纹、镜像/模型版本、checkpoint hash、配置 hash 还可以更系统。
- 暂未支持 partial result streaming：必须等 job 结束后才 ingestion 和 analysis，无法在 folding/analysis 中途发现明显失败并早停/修复。

### 5.3 数据模型与 agent 协作层

- MessageBus 目前偏审计日志，不是 agent coordination substrate。
- 缺少事件 schema、订阅机制、锁、幂等处理和 agent output validation registry。
- 多个 agent 都能提出配置变更，但缺少显式仲裁：
  - 哪个证据优先？
  - 冲突参数如何解决？
  - 风险/收益如何打分？
  - 是否保留 exploration budget？
- Memory 以 round 为主，candidate/module/fragment lineage 不够强；很难回答“这个片段来自哪个 parent、哪个 seed、哪个 arm，是否跨轮复现”。

### 5.4 结构与生物物理评估层

- 当前 `structure_features.py` 已有接触、热点、clash、chain break、局部 fragment quality，但还比较轻量。
- 缺少更完整的 interface metrics：
  - buried surface area / SASA。
  - shape complementarity。
  - interface PAE / ipAE。
  - side-chain rotamer quality。
  - unsatisfied polar atoms。
  - electrostatics / charge complementarity。
  - oligomeric state / target conformer compatibility。
- 缺少 verifier ensemble：
  - Boltz-2、AF2-multimer、Chai、ESMFold 等交叉验证。
  - 同一候选在不同 verifier 下的一致性与不确定性。
- 缺少 sequence-level risk agent：
  - aggregation-prone motifs、glycosylation/protease/deamidation/liability、cysteine/Met/Trp 风险、低复杂度、免疫原性 proxy。

### 5.5 模块片段利用层

- 当前已经能识别 `high_quality_fragments` / `low_quality_fragments`，并通过 `BinderQualityAnalysisAgent` 生成 `exploit_fragment_modules` / `avoid_fragment_modules`。
- `ActiveLearningPolicyAgent` 会消费这些字段，并开启 `module_guided_exploitation` 或 `module_guided_repair` arms。
- 但目前这些 module 更像“策略提示 ID”，还不是可直接驱动生成模型的结构模板：
  - 缺少 fragment 坐标、局部 frame、二级结构、接触图、target anchor、sequence motif 的标准化保存。
  - `DesignSpecAgent`/BoltzGen command 目前没有明确把 fragment template 翻译成可执行 conditional generation 输入。
  - 没有跨候选聚类，无法判断某个片段是偶然高分还是可复用 motif。

## 6. 是否应加入更多结构视角评测 agent

应该加入，而且这是 binder 设计闭环最值得扩展的方向。原因是 binder 设计的失败常常不是单个全局指标能解释的，而是局部结构、界面几何、热点覆盖、序列设计性和目标 patch 选择共同决定。

建议新增以下 agent：

- `TargetPatchAnalysisAgent`
  - 分析 target 表面 patch、热点可达性、凹凸性、电荷/疏水分布、跨链界面、可设计 anchor。
  - 输出可作为 `target_include`、`target_binding_types`、hotspot subsets、patch ensembles 的依据。

- `InterfacePhysicsAgent`
  - 计算 buried surface area、shape complementarity、interface H-bond/salt bridge、unsatisfied polar、clash/packing、charge complementarity。
  - 输出更可靠的 interface pass/fail 和 repair reason。

- `BinderTopologyAgent`
  - 分析 binder secondary structure、Rg、end-to-end、contact order、chain break、helix bundle/loop/sheet topology、foldability risk。
  - 输出 `binder_structure_prior`、`secondary_structure`、length adjustment、foldability repair 建议。

- `FragmentTemplateMiningAgent`
  - 从 high-quality fragments 中抽取可复用模板：
    - binder residue span。
    - sequence 与 secondary structure。
    - backbone coordinates。
    - target-contact residue set。
    - hotspot mapping。
    - local coordinate frame / anchor transform。
    - fragment quality evidence。
  - 对跨候选 fragments 聚类，筛出稳定 motif，而不是只看单个候选。

- `CrossRoundLineageAgent`
  - 建立 `candidate -> parent job -> seed -> arm -> params -> metrics -> fragments` 的 lineage。
  - 估计哪些参数/arms 真正提升了结构质量，哪些只是相关。

- `VerifierEnsembleAgent`
  - 调度 AF2/Boltz/Chai 等 verifier，对候选做 ensemble consistency。
  - 输出 uncertainty 与 verifier disagreement，避免单模型打分过拟合。

- `DiversityAndNoveltyAgent`
  - 计算 sequence clustering、structure clustering、interface contact-map clustering。
  - 控制 exploitation 不要过早塌缩。

- `ManufacturabilityAgent`
  - 对序列 liability、表达/溶解性 proxy、低复杂度、免疫原性 proxy 做早期筛查。
  - 防止只优化结构指标而生成实验不可行 binder。

## 7. target、binder 模块片段能否用于下一轮优化

可以，而且当前代码已经有部分雏形，但还没有走到“直接模板化 conditional generation”的程度。

### 7.1 目前已经具备的基础

- `analyze_binder_structure()` 会按 sliding window 给 binder fragment 打分。
- 每个 fragment 有：
  - residue span。
  - interface contact count。
  - hotspot contact count。
  - clash count。
  - hbond/salt/hydrophobic contact。
  - hydrophobic/polar fraction。
  - local chain break。
  - quality score、label、reasons、suggested action。
- `BinderQualityAnalysisAgent` fallback 会把 high fragments 转成 `high_quality_modules`，并生成 `exploit_high_quality_fragments` guidance。
- `ActiveLearningPolicyAgent` 会把 high modules 写入 `exploit_fragment_modules`，把 low modules 写入 `avoid_fragment_modules`。
- `StrategyLevelActiveLearner` 会在发现 high/low modules 后加入 `module_exploitation` / `module_repair` strategy arms。

### 7.2 现在还不能直接做到的部分

- `module_id` 目前通常是 `structure_file:fragment_id`，不是独立可移植模板。
- 没有把 fragment 的坐标、局部 frame、target anchor、序列/二级结构约束保存成可复用 artifact。
- 没有把高质量 fragment 转成 BoltzGen/RFdiffusion/ProteinMPNN 可以直接消费的 conditional input。
- 没有验证该 fragment 在不同候选、不同 seed、不同 verifier 中是否稳定。
- 没有对 fragment exploitation 设置风险控制，例如防止过度固定导致 diversity collapse 或 off-target hydrophobic patch。

### 7.3 建议的模板化 conditional generation 设计

可以新增一个标准 `FragmentTemplate` artifact，建议字段包括：

- `template_id`
- `source_structure_file`
- `source_candidate_id`
- `source_round_id`
- `binder_chain`
- `binder_residue_span`
- `binder_sequence`
- `secondary_structure`
- `backbone_atoms_or_ca_trace`
- `target_contact_residues`
- `hotspot_contacts`
- `contact_map`
- `local_frame`
- `quality_score`
- `evidence`
- `reuse_mode`
  - `preserve`
  - `perturb`
  - `graft`
  - `avoid`
- `compatible_target_patch`
- `risk_flags`

随后让 `FragmentTemplateMiningAgent` 产出：

- `fragment_templates.json`
- `template_clusters.json`
- `template_to_config_updates.json`

再由 `InputConfigurationAgent` / `ActiveLearningPolicyAgent` 消费这些 artifact，把它们转成两类下一轮优化：

- 参数级优化：
  - 调整 `hotspots`、`prioritize_hotspots`、`target_include`、`target_binding_types`。
  - 调整 `binder_lengths`、`length_delta_hint`、`secondary_structure`、`binder_structure_prior`。
  - 调整 `run_filtering`、`clash_filter`、`additional_filters`。

- 模板级 conditional generation：
  - 把高质量 motif 作为 binder 片段模板进行 preserve/perturb。
  - 把 target patch 与 binder fragment 的 contact map 转成约束。
  - 把低质量 motif 作为 avoid constraints。
  - 对同一 target patch 生成多个 template-conditioned arms，保留 exploration arm 防止过拟合。

如果当前 BoltzGen backend 无法直接消费 fragment template，也可以先用 soft mode：

- 在 design spec 中写入 residue/secondary-structure constraints。
- 在 post-generation filter 中优先保留 contact-map 相似、热点覆盖一致、clash 更低的候选。
- 在 active learner 中把 template cluster 当作 arm，而不是直接固定结构。

## 8. 改进路线建议

### 8.1 P0：让当前闭环更可靠

- 为 MessageBus 增加写锁或改成 SQLite/JSONL locked writer。
- 为 agent outputs 增加 JSON Schema 与 schema version。
- 明确参数变更仲裁顺序：diagnostic、quality、hypothesis、input config、policy 的冲突需要可解释 merge report。
- 去除或外置 target-specific prompt 知识，例如 IL-17A 规则应放入 target profile，而不是通用 diagnostic prompt。
- 在每轮 summary 中加入 config diff、evidence-to-action trace。

### 8.2 P1：补齐 benchmark harness 能力

- 引入 `TargetDataset` / `TaskRegistry` / `ScorerRegistry` / `BackendRegistry`。
- 固定 baseline configs 与 target suites，支持批量运行和横向比较。
- 增加 structured reports：success rate、top-k、diversity、cost、throughput、failure taxonomy、verifier consistency。
- 增加 seed repeat、bootstrap confidence intervals、ablation runs。
- 为 local/Taiji 后端记录环境指纹、checkpoint hash、command hash、artifact manifest。

### 8.3 P2：增强结构 agent 与模板化生成

- 新增 `TargetPatchAnalysisAgent`、`InterfacePhysicsAgent`、`BinderTopologyAgent`、`FragmentTemplateMiningAgent`、`CrossRoundLineageAgent`。
- 把 `high_quality_fragments` 升级成 `FragmentTemplate` artifact。
- 让 `DesignSpecAgent` 或 backend adapter 支持 template-conditioned config：
  - hard constraints：固定/半固定 motif、residue constraints。
  - soft constraints：contact-map similarity、hotspot-prioritized sampling、template-aware filtering。
- 对 fragment templates 做跨轮聚类与验证，只 exploitation 高置信模板。
- 每轮保留固定比例 exploration arms，防止模板过拟合。

### 8.4 P3：从简易 active learning 升级到多目标优化

- 用 multi-armed bandit 或 Bayesian optimization 管理 strategy arms。
- 将 objective 拆成多目标：
  - interface confidence。
  - hotspot coverage。
  - foldability。
  - diversity/novelty。
  - developability。
  - compute cost。
- 对每个参数变更记录 evidence、expected signal、observed signal，形成可学习的 policy memory。

## 9. 总体结论

- 当前 `binderloop` 已经不是单纯脚本，而是一个轻量闭环 agent harness：有执行、ingestion、evaluation、structure analysis、quality analysis、hypothesis、diagnostic、input configuration、policy update 和 memory。
- 对当前阶段来说，agent 拆分和串行编排是可理解的；它优先保证可审计、可恢复、可控预算，而不是追求复杂多 agent 协作。
- 最大短板是：MessageBus 还不是协作系统，memory 还不是结构化实验数据库，结构 fragment 还没有变成可执行模板，benchmark/reproducibility 能力也还偏弱。
- 对 binder 设计而言，下一步最有价值的方向不是单纯扩大采样范围，而是把 target patch、binder topology、interface physics、fragment template 和 cross-round lineage 结构化，并让高质量片段直接进入下一轮 conditional generation 或 template-aware filtering。

## 10. 参考资料

- EleutherAI `lm-evaluation-harness`: https://github.com/EleutherAI/lm-evaluation-harness
- Inspect AI: https://inspect.ai-safety-institute.org.uk/index.html
- SWE-bench harness reference: https://www.swebench.com/SWE-bench/reference/harness/
- OpenProtein RFdiffusion binder design walkthrough: https://dev.docs.openprotein.ai/walkthroughs/Protein_protein_binder_design_with_RFdiffusion.html
- ProteinDJ modular protein design pipeline: https://pmc.ncbi.nlm.nih.gov/articles/PMC12820799/
- nf-binder-design Nextflow pipeline: https://github.com/Australian-Protein-Design-Initiative/nf-binder-design
- ProtDBench unified binder design benchmark: https://arxiv.org/html/2605.04118v1
