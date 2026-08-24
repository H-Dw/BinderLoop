# Binder Harness 中 BoltzGen binder 生成参数的控制面与 Agent 协作

> 核对基线：当前仓库代码（2026-07-16），不是旧 README。本文以 `scripts/run_closed_loop_orchestrator.py` 启动的 `BinderDesignOrchestrator` 生产闭环为主；`pipeline.py` 和旧脚本只在明确标注时讨论。

## 1. 范围与分类原则

本文把“参数”按**最终消费者**分成四类，避免把调度或评分参数误称为模型参数。

1. **直接生成参数**：改变 BoltzGen CLI、design spec 或 structure-redesign mask，因而直接改变生成。包括协议、采样、inverse folding、checkpoint、目标 BINDING/crop、binder 长度、模板固定区等。
2. **harness 间接物化参数**：本身不是 BoltzGen CLI 参数，但被 harness 翻译成上述直接输入。例如 `auxiliary_hotspots`、`epitope_crop_mode`、`binding_site_policy`、`template_conditioned_fraction`、`binder_lengths` 集合和模板 provenance。
3. **资源/编排参数**：控制轮次、分支、预算分配、GPU/host shard、Taiji、本地/远端分析、重试和日志。它们影响吞吐与实验覆盖，但不是 BoltzGen 模型参数。
4. **筛选/评估/元数据参数**：`additional_filters`、结构 clash selection、评分阈值、memory、self-improvement、lineage、deprecated audit 字段等；它们影响保留、排名、证据或审计，不改变已生成 backbone。`config_overrides` 是开放通道，只有其中实际传到某 BoltzGen step 的 token 才属于直接模型/任务配置。

契约按 native CLI、adapter-translated、runtime-resource、internal-metadata、deprecated-metadata 分类，见 `binderloop/agents/config_parameter_contract.py:88-142`。

## 2. 正式 owner YAML 可配置面

`load_config` 只接受 `schema_version: 1` 和 `owner`；owner 正式段为：

- `task_hard_constraints`（必需）：target 路径、可选 BoltzGen input、chain、hotspots、`binder_length_range/step`、`num_designs`、include/binding types/groups，以及三个 freeze 开关。
- `boltzgen_design_native`：`protocol`、`diffusion_batch_size`、`design_checkpoints`、`steps`、`checkpoint_dir`、`cache`、`moldir`、`use_kernels`、`num_workers`、`reuse`、`silence`、heartbeat、`auto_binder_length`、`epitope_crop_mode`、`binder_chain`。
- `boltzgen_inverse_fold_and_validation`：`inverse_fold_num_sequences`、`inverse_fold_avoid`、三个 checkpoint、`skip_inverse_folding`、`only_inverse_fold`。
- `boltzgen_filtering_ranking`：`alpha`、`filter_biased`、`refolding_rmsd_threshold`、`metrics_override`、`size_buckets`、`config_overrides`。
- `sampler_bounds`：只允许 `noise_scale/step_scale/alpha` 的 `{min,max}`，且必须落在物理绝对包络内。
- `filtering_budget`：显式 `budget`，以及固定为 true 的 `run_filtering`、失败分析保留和用户 `additional_filters`。
- `harness_template_policy`：模板启用、gate/阈值、Top-K、conditioned fraction、proximity、库与固定区/对齐/PAE/失败策略参数。
- 相关闭环段：`active_learning_and_rollback`、`runtime_resources.{runtime,resource}`、`llm_context_learning.{memory,self_improvement,quality_collaboration}`；另有 `harness_search_space` 和只影响证据的 `harness_selection_and_evidence`。

字段白名单与装配见 `binderloop/config.py:174-228`；模板默认值和边界见 `binderloop/config.py:128-154`；active-learning、runtime、resource 默认值见 `binderloop/config.py:40-94`。

## 3. 参数族、调控方式与消费者

### 3.1 Target、hotspot 与 crop

- `target_structure_path/target_chain_id`：用户硬约束；进入 `DesignJob`，最终写到 spec 的 target file。默认冻结，Agent 不能替换。
- `hotspots`、`target_binding_types`、`target_include`、`structure_groups`：用户定义优先。Adapter 把 hotspot 转为 `binding_types`，include 缺省为整个 target chain；见 `binderloop/models/boltzgen_adapter.py:188-218`。
- `auxiliary_hotspots`：公共 Agent 可提议，但 orchestrator 只保留与用户 hotspot 同链且序号距离不超过 15 的最多 3 个；再物化为 expanded BINDING，而不是改写用户 hotspots（`binderloop/orchestration/orchestrator.py:3413-3443,3460-3537`）。
- `epitope_crop_mode`：`disabled/off/none/auto/hotspot_focus/engaged_focus/union`。默认 `disabled`；若用户没有内部 `allow_agent_epitope_crop`，禁用即硬约束，Agent 不能开启。开启后只有 `FragmentTemplateMiningAgent` 可产生 internal-only `target_include/target_binding_types/structure_groups`（`binderloop/orchestration/orchestrator.py:3609-3665`）。
- `harness_selection_and_evidence` 的 contact/clash/window 参数只影响分析与证据；`weighted_hotspot_conditioning` 当前 checkpoint 不支持并直接报错（`binderloop/config.py:217-219`）。

### 3.2 Binder length

- 用户入口为 `binder_length_range`（必需）和 `binder_length_step`（默认 10）；展开包含端点（`binderloop/config.py:231-240`）。
- `auto_binder_length` 控制 `BinderLengthPolicyAgent` 是否依据 chain break、可靠性、interface、clash、PAE 推荐离散 `binder_lengths`。其全局物理包络是 30–180，但生产路径还会夹到用户 range（`binderloop/agents/binder_length_policy_agent.py:11-16,51-55,152-178`）。
- `binder_lengths` 是公共 delta 中的特殊 harness-owned 字段，**专属 owner 是 `binder_length_policy`**；其他来源不能覆盖已由它给出的不同值，冲突仲裁可最终覆盖（`binderloop/orchestration/orchestrator.py:3952-3985`）。
- 最终每个长度一份 design spec，`protein.sequence` 为整数长度；GPU fan-out 使用 per-length specs（`binderloop/agents/design_spec_agent.py:226-262,728-743`）。模板分支还做 motif-safe length transform，失败则拒绝该模板分支（`binderloop/active_learning/strategy.py:212-263`）。

### 3.3 Protocol 与 diffusion/sampling

- `protocol` 默认 `protein-anything`，合法值还包括 peptide、protein-small_molecule、nanobody；用户 YAML 所有，Agent 公共 delta 不可改。
- 公共可调：`diffusion_batch_size`（正整数）、`step_scale`、`noise_scale`、`alpha`。绝对默认/包络：step `0.8/[0.6,1.0]`、noise `0.7/[0.6,0.9]`、alpha `0.001/[0.001,0.05]`；每轮惯性分别 0.2、0.15、alpha 3 倍（`binderloop/agents/config_parameter_contract.py:23-29`）。owner `sampler_bounds` 只能收窄。
- `sampler_explore` arm 本身是 intent；后续物化时，仅在值缺失时把 noise/step 设为有效上界（`binderloop/orchestration/orchestrator.py:3256-3276`）。
- CLI 消费这些字段；多 GPU shard 会把 `diffusion_batch_size` 派生为每 shard batch（`binderloop/models/boltzgen_renderer.py:18-24`，`binderloop/agents/design_spec_agent.py:242-257,804-818`）。
- 重要语义：**不传 `step_scale/noise_scale` 与显式传 harness baseline 0.8/0.7 不保证等价**；前者让上游 BoltzGen 自己采用版本默认，后者明确覆盖。审计必须区分 absent 与 explicit。

### 3.4 Inverse folding、folding 与 checkpoints

- 用户可配 `inverse_fold_num_sequences`（候选上界为 `num_designs × inverse_fold_num_sequences`）、`inverse_fold_avoid`（公共 Agent 可调）、`skip_inverse_folding`、`only_inverse_fold`、`refolding_rmsd_threshold`。
- checkpoint：`design_checkpoints`、`inverse_fold_checkpoint`、`folding_checkpoint`、`affinity_checkpoint`；缺省由 adapter 指向 `<boltzgen_root>/checkpoints` 下固定文件；cache/moldir 也被本地物化（`binderloop/models/boltzgen_adapter.py:99-167`）。
- `checkpoint_dir` 是 owner YAML 接受字段，但不在 renderer CLI，也不在参数契约完整键集中；`with_default_local_artifacts` 会在仍存在于 params 时读取它，而 pre-submit full-job validation会依契约剔除未知字段。生产脚本另通过 CLI `--checkpoint-dir` 构造 `DesignSpecAgent`。因此 YAML `checkpoint_dir` 存在**合同断链风险**，不应假定会生效。
- `steps` 是用户字段，但 renderer 本身不渲染；`DesignSpecAgent._ensure_complete_pipeline` 最后补上 `--steps`。本地默认 GPU steps，不含 analysis/filtering 时另写本地 analysis script；Taiji 强制远端 full steps 并确保 filtering（`binderloop/agents/design_spec_agent.py:579-607`）。

### 3.5 Filtering、ranking 与开放 overrides

- `filter_biased` 是公共 Agent 可调的 lowercase choice `true|false`；Python bool 会被规范化。`alpha` 同时作为 CLI 的 diversity/adherence 参数。
- `metrics_override`、`size_buckets`、`additional_filters`、`refolding_rmsd_threshold` 用户拥有。`additional_filters` 是 metrics CSV 硬筛选，不是 Agent delta；明确丢弃伪造的 `designfolding_iptm`，支持形如 `designfolding-filter_rmsd<2.5`（`binderloop/agents/model_input_spec.py:204-282`）。
- `run_filtering` 在闭环固定 true，用户配置 false 会在 load 时失败，验证与 DesignSpecAgent 也再次强制，以保证 ranked metrics（`binderloop/config.py:210-212`；`binderloop/agents/model_input_spec.py:490-509`）。
- `config_overrides` 是公共 Agent 通道，归一为 `[[step,key=value,...]]`，step 必须属于七个 BoltzGen steps，仅剔除已确认会崩溃的少数 key。其可用 setting 来自上游 Hydra task，故**参数空间不可封闭枚举**；验证是保守 denylist，不是完整 allowlist（`binderloop/agents/model_input_spec.py:609-717`）。

### 3.6 Fragment template redesign

用户必须用 `harness_template_policy.enabled` 显式开启；Agent 不能自行开启。默认 gate=`interchain_pae`、阈值 10 Å、Top-K=1、conditioned fraction=0.5、proximity=8 Å、最大固定比例 0.5、至少 8 个可设计残基（`binderloop/config.py:128-147`）。

`FragmentTemplateMiningAgent` 从 structure evaluation 中挖 preserve/avoid fragment，要求 PAE/provenance、质量、可打包 source、当前 target patch 对齐与 RMSD；只有成功门控、对齐并 stage 的 preserve fragment 才生成 internal-only `binder_template(s)`（`binderloop/agents/fragment_template_mining_agent.py:102-205,413-482,485-624`）。Adapter 将模板写成 structure-redesign design spec，并产生 inverse-fold redesign mask 的 `restrictions.not_design.within_proximity`（`binderloop/models/boltzgen_adapter.py:58-96,231-281`）。

`template_conditioned_fraction` 只是 allocation intent，范围 `[0,0.8]`；无有效模板时 execution resolver 删除它。有效模板必须配 template-free control；无效模板份额按 `reject_and_rematerialize` 重新分配，而非静默退化（`binderloop/execution_governance.py:106-185,368-377`）。

### 3.7 预算、分支与 GPU shard（编排，不是模型参数）

- `num_designs` 是用户轮预算且默认 freeze；`max_binders_per_round` 为 cap。`filtering_budget.budget` 是 BoltzGen CLI 的计算预算，**必须显式提供**；resolver 禁止 adapter fallback，并在多 GPU 时提升到 `max(99999,candidate_upper_bound)`（`binderloop/execution_governance.py:309-366`）。
- `branch_width/promote_top_branches/branch_budget_policy/min_designs_per_branch` 控制 arm/controlled comparison。闭 catalog 包含 baseline、binding-site、crop、sampler、clash selection、template exploit（`binderloop/active_learning/strategy.py:32-41,88-130`）。
- `devices/host_count/taiji_multi_host_mode` 决定 GPU/host 分片。总预算先按长度分，再拆到 worker 并做 largest-first 平衡，最终必须守恒（`binderloop/agents/design_spec_agent.py:627-726`）。
- `exploration_ratio` 是 active-learning 内部参数，不是模型参数；公共 LLM delta 不可直接改。`seed` 仅用于 learner 的本地选择 RNG/DesignJob 兼容字段，BoltzGen 无 seed 控制，因此不会生成 length×seed 作业（`binderloop/active_learning/strategy.py:59-70`）。

### 3.8 资源与日志（不是模型参数）

`backend`、host/GPU 数、GPUName、parallel jobs、Taiji template/options、timeout、runtime roots/output/python、`analysis_location`、reuse、GPU sharding toggles控制执行。`silence` 只控制详细日志是否镜像屏幕；heartbeat 默认 360 秒且夹在 1–360 秒。最终脚本验证 target、checkpoint、cache、moldir，记录 command/spec/parameter plan/redesign mask（`binderloop/agents/design_spec_agent.py:763-775,965-1031,1182-1238`）。资源字段被 hard freeze；仅资源失败重试可降级。

## 4. 公共 Agent delta 白名单与所有权

当前公共白名单**精确为**：

`diffusion_batch_size`, `step_scale`, `noise_scale`, `alpha`, `inverse_fold_avoid`, `filter_biased`, `config_overrides`, `auxiliary_hotspots`, `epitope_crop_mode`, `template_conditioned_fraction`, `binder_lengths`。

前十项来自 `ADJUSTABLE_CONFIG_PARAMETERS`，`binder_lengths` 单独并入 `PUBLIC_AGENT_CONFIG_KEYS`（`binderloop/agents/config_parameter_contract.py:35-46,75-86`）。

- **用户拥有**：`additional_filters`、fragment enable/Top-K，以及完整 job 中的 task、协议、steps、budget、资源、inverse-fold 数和 RMSD threshold 等；可以静态配置并被 full-job validator 保留，但 LLM/policy 不得编辑。
- **internal-only**：`binder_template(s)`、template proximity、harness crop payload、`exploration_ratio`、length hints/avoid list 等，只接受可信 orchestrator/fragment-mining 来源（`binderloop/agents/config_parameter_contract.py:59-80`）。
- deprecated hotspot/clash/module flags只迁移到 `deprecated_strategy_audit`，无执行消费者。

## 5. Agent 角色与实际接线

### 直接提案层

- `InputConfigurationAgent`：综合诊断、结构、质量、假设、memory 和 constraints，输出公共 delta。
- `BinderLengthPolicyAgent`：`binder_lengths` 专属 owner。
- `ActiveLearningPolicyAgent`：规则式公共 delta，并吸收 quality/hypothesis/diagnostic 建议。
- `FragmentTemplateMiningAgent`：唯一可信模板/crop internal payload 来源。
- `StrategyConflictResolutionAgent`：只处理检测到的软参数族冲突，可 choose/blend/hold/revert/controlled compare；不能越过硬约束。

### 证据/建议层

`ResultIngestionAgent -> EvaluationAgent + StructureEvaluationAgent -> BinderQualityAnalysisAgent`，必要时由 `BinderQualityCollaborationAgent` 多专家复核；之后 `HypothesisAgent -> DiagnosticCoachAgent`。Memory retrieval/compression、SelfImprovementSkillAgent、RollbackController 提供跨轮证据、规则和回退，但不会绕过公共 contract。代码时序见 `binderloop/orchestration/orchestrator.py:725-805,831-1039,1050-1186`。

### 治理/执行层

- `BinderDesignOrchestrator`：中央 merge、ownership、hard freeze、物理/惯性 clamp、pressure conflict、arms、预算与 job 物化。
- `ConfigValidationAgent`：**不在 orchestrator 类内部的每个分析模块后调用**；但生产入口为 local/Taiji executor 构造它，并在每次 pre-submit 验证完整 job，Taiji 失败后再做针对性修复（`scripts/run_closed_loop_orchestrator.py:237-355,452-464`）。若 backend=`dry_run`，executor 为 `None`，不会走这条 pre-submit 验证。
- `DesignSpecAgent`、`BoltzGenAdapter/renderer`：resolve、spec/mask、CLI、shard、脚本和审计产物。
- `TaijiExecutionAgent`：提交；RunMonitorAgent 监控。

`DesignParameterAgent` **未被当前生产 `BinderDesignOrchestrator` 导入或实例化**；它只出现在独立 legacy/demo/test scripts，如 `scripts/run_coached_pipeline_v4.py`、`scripts/run_il17a_full_pipeline*.py`、`scripts/test_boltzgen_taiji_agents.py`。不能因类仍存在就把其 heuristic defaults 当作生产闭环默认。

## 6. 协作时序、merge 与冲突仲裁

生产闭环可概括为：

1. result ingest → candidate evaluation / structure evaluation；
2. quality（单 Agent 或 collaboration）→ hypothesis → diagnostic；
3. InputConfiguration、BinderLengthPolicy、FragmentTemplateMining 并基于相同证据提出 delta；ActiveLearningPolicy 再汇总规则建议；
4. 中央 preview merge，顺序固定为 `input_configuration -> binder_length_policy -> policy_proposal -> fragment_template_mining`；同 key 后者覆盖前者，但 `binder_lengths` owner 冲突除外；
5. 检测 soft conflicts；StrategyConflictResolution 输出时作为最后来源追加，再做最终 merge；
6. hard freeze → normalized families（当前不互斥）→ physical/inertia clamp → pressure-conflict resolver → 必要时再次 clamp → 写 live config；详见 `binderloop/orchestration/orchestrator.py:3915-4071`；
7. active learner 排序 arms、构造分支和 controlled comparisons；物化 BINDING/crop/sampler/template；执行 round budget resolver、长度 guardrail、GPU shard；
8. ConfigValidation（非 dry-run executor）→ execution resolver → DesignSpecAgent/adapter/renderer → local/Taiji execute；
9. 新结果反馈 memory/self-improvement/template ledger；回归触发 rollback。

Pressure conflict 在核心指标回归时禁止继续加 hotspot/crop/template pressure，并且在 strategy job 物化后再次执行，防止 arm 绕过中央 merge（`binderloop/orchestration/orchestrator.py:3286-3304,3784-4037`）。

回滚不是“回到类似参数”：当前轮提案全部变 audit-only，随后恢复 best-round 的完整 nested `boltzgen_config` 和 target/length/AL 状态，克隆 durable logical jobs；模板还校验 execution identity，预算和长度若不完全兼容则报错，确保 **exact replay**（`binderloop/orchestration/orchestrator.py:1251-1280,1486-1496,5286-5388`）。

## 7. 完整链路与最终审计产物

输入 YAML 只是初始 owner intent。有效值依次经过：YAML load/物化 → Agent delta → merge provenance → freeze/clamp/conflict → arm/budget/template/length materialization → full-job validation → execution resolver → artifact defaults → steps/shards/render。

每个 package 的关键事实源：

- `configs/boltzgen_parameter_plan.yaml`：DesignSpecAgent 解析后的完整参数；
- `configs/effective_execution_plan.json`：resolved params、lineage、applicability、candidate upper bound、digest、CLI/spec/shard parity 和 consumer receipts；
- `configs/boltzgen_parameter_consumption.json`：逐字段落点（CLI/design_spec/harness_transform/allocation/runtime/metadata/rejected）；
- `configs/boltzgen_design_spec.yaml` 与 per-length specs；
- `configs/boltzgen_redesign_mask.yaml`（模板分支）；
- `configs/cluster_shard_plan.json`（分片时）；
- `scripts/run_boltzgen_full.sh`、`boltzgen_run_manifest.json` 和其中最终 shell/CLI。

生成与审计写入点见 `binderloop/agents/design_spec_agent.py:335-440,481-508`。**最终有效值不能只看输入 YAML，也不能只看 next-round delta；应优先看 effective execution plan，再交叉核对 consumption、spec/mask、shard plan 和最终 CLI。**

## 8. 已确认的实现注意事项与不一致

1. `DesignParameterAgent` 是遗留独立路径，不是生产 orchestrator 必经。
2. omission 与 explicit baseline 不等价，特别是 `step_scale/noise_scale`。
3. `config_overrides` 的 Hydra setting 空间不可封闭；当前只验证 step/shape 并拒绝少数已知崩溃 key。
4. execution resolver 显式要求 `budget`；缺失即失败，不能依赖旧 adapter/DesignParameterAgent fallback。
5. `steps` 由 `DesignSpecAgent` 最后补到 CLI，而不是 renderer 的常规字段循环。
6. owner YAML 的 `checkpoint_dir` 可能在 full-job contract validation 时被剔除，形成合同断链；生产可靠入口是 runner CLI `--checkpoint-dir` 或显式 checkpoint 文件字段。
7. BoltzGen `seed` 非可调；不做 length×seed fan-out。
8. deprecated hotspot/clash/module 参数仅 audit，无生成消费者。
9. `run_filtering` 闭环固定 true；false 在正式 YAML load 阶段即被拒绝。
10. `ActiveLearningPolicyAgent` 中 diversity-collapse 临时提议可到 0.1，但中央物理 guardrail 最终夹到 alpha≤0.05；因此应看 merge report/effective plan 而非原始 proposal（`binderloop/agents/active_learning_policy_agent.py:112-115`）。
11. `DesignSpecAgent` 的 artifact default、resolver budget floor、per-shard num_designs/batch 都会改变“最终值”；这是预期治理，不是模型自行调参。

## 9. 快速判读清单

- 问“模型实际吃了什么”：看 effective plan + final CLI + design spec/mask。
- 问“谁改了它”：看 `next_round_config_merge_report.json` 的 `decisions/applied_sources` 和 effective plan lineage。
- 问“为什么 YAML 值不同”：依次查 hard freeze、physical clamp、pressure conflict、validation tombstone、resolver、shard derivation。
- 问“Agent 能不能改”：先对照 11-key 公共白名单；用户拥有字段只可静态配置，internal-only 只可信来源可写。
- 问“筛选是否改变生成”：`additional_filters`/selection 通常只改变保留与证据；只有进入 CLI sampling/inverse-fold/spec/mask 的字段才直接改变生成。
