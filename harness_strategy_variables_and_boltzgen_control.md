# Harness 可调整策略变量与 BoltzGen 控制链审计

> 范围：当前正式任务 YAML/CLI、闭环 orchestrator、BoltzGen adapter/DesignSpecAgent、主动学习、模板/长度策略、评价与回滚、LLM/记忆/自改进、资源调度及 v22 历史产物。基准日期：2026-07-15。

## 1. 影响等级与结论

- **D0 直接生成控制**：最终进入 `boltzgen run` CLI、design spec、redesign mask、实际输入文件，或直接决定长度与 `num_designs`。
- **D1 条件直接/输出控制**：进入 inverse folding、folding、analysis、filtering/ranking，或仅在模板/crop/特定 step 启用时生效。
- **I 间接闭环控制**：决定 parent、arm、分支预算、模板资格、回滚、LLM proposal 或下一轮参数。
- **N 无当前生成作用**：日志/路径/审计字段，或虽已定义但主闭环没有消费者。

真正控制 BoltzGen 的入口只有三类：CLI 参数、design spec/redesign mask、任务 materialization。LLM prompt、arm 名称、评分和记忆配置必须翻译到这些入口才有效。

明确无可执行生成语义的旧字段：`hotspot_weight`、`prioritize_hotspots`、`clash_filter`、`module_guided_repair`、`module_guided_exploitation`、`exploit_fragment_modules`。当前 validator 将其移入 `deprecated_strategy_audit`。

需谨慎的 typed intent：`selection_policy`、`sampler_policy`、`target_context_policy`。它们已被记录，但部分没有生产执行翻译器，不能仅凭字段存在认定已生效。

## 2. 最终调用链

```text
用户任务 YAML / 部分原生 BoltzGen YAML
 -> load_config() 与硬约束派生
 -> BinderDesignOrchestrator._base_params()
 -> 结构分析、LLM、policy、模板、长度、回滚提案
 -> 白名单清洗 + merge + freeze + clamp + pressure resolution
 -> StrategyLevelActiveLearner/controlled comparison 物化 DesignJob
 -> round budget 与长度 guardrail
 -> ConfigValidationAgent.validate_full_job_config()
 -> DesignSpecAgent
    -> boltzgen_design_spec[_lenN].yaml
    -> 可选 boltzgen_redesign_mask.yaml
    -> cluster_shard_plan.json
    -> boltzgen run CLI 与 run_boltzgen_full.sh
 -> local 或 Taiji 执行
```

关键实现：`binder_harness/config.py:387-582`、`binder_harness/agents/config_parameter_contract.py:23-140`、`binder_harness/orchestration/orchestrator.py:2717-2829,3294-3371`、`binder_harness/active_learning/strategy.py:63-298`、`binder_harness/agents/design_spec_agent.py:68-354,437-529`、`binder_harness/models/boltzgen_adapter.py:168-251`。

## 3. 模块 A：用户任务硬约束（`task.*`）

| 变量 | 默认/来源 | 作用 | 影响 |
|---|---|---|---|
| `task_name` | `binder_task` | manifest/Taiji/memory 标识 | N |
| `target_structure_path` | 必填 | 打包并写入 target file entity | D0 |
| `boltzgen_input_path` | `None` | 部分导入原生 YAML 的 target/include/binding/groups/hotspots | D0（翻译后） |
| `target_chain_id` | `A` | target include/binding 默认链 | D0 |
| `hotspots` | `[]` | 转成 `target_binding_types.chain.binding` | D0 |
| `binder_length_range` | 必填 | binder 长度硬外边界 | D0 |
| `binder_length_step` | `10` | 离散长度网格 | D0 |
| `max_binders_per_round` | 必填 | 轮级 backbone cap，派生各 job `--num_designs` | D0 |
| `target_include` | `[]` | target 裁剪范围 | D0 |
| `target_binding_types` | `[]` | BINDING/NOT_BINDING 条件 | D0 |
| `structure_groups` | `None` | target/template 坐标分组 | D0 |
| `notes` | `None` | 报告/LLM 上下文 | I/N |
| `freeze_target_definition` | `true` | 锁定 target/hotspots/crop | I（强硬锁） |
| `freeze_binder_length_range` | `true` | 禁止扩大长度边界 | I（强硬锁） |
| `freeze_round_budget` | `true` | 禁止 agent 改写样本数 | D0 |

`boltzgen_input_path` 不是完整透传。`_merge_boltzgen_input()` 只抽取有限字段；例如原 spec 的 binder `id` 不会被读取，闭环 adapter 会回退到默认 binder chain `B`。

## 4. 模块 B：BoltzGen diffusion/生成 CLI

| 变量 | 默认/范围 | 实际消费 | 影响 |
|---|---|---|---|
| `protocol` | `protein-anything`; 另有 peptide/small-molecule/nanobody | `--protocol` | D0 |
| `num_designs` | 正式闭环由 round cap 派生 | `--num_designs` | D0 |
| `diffusion_batch_size` | 未统一；常用 1 | `--diffusion_batch_size` | D0 |
| `step_scale` | 原生 schedule；contract 建议 0.8，范围 0.6–1.0 | `--step_scale` | D0 |
| `noise_scale` | 原生 schedule；contract 建议 0.7，范围 0.6–0.9 | `--noise_scale` | D0 |
| `design_checkpoints` | diverse + adherence | `--design_checkpoints` | D0 |
| `steps` | 默认完整流水线 | `--steps` | D0 |
| `skip_inverse_folding` | false | CLI flag | D0 |
| `only_inverse_fold` | false | CLI flag | D0 |
| `reuse` | false | CLI flag | D1 |

`diffusion_batch_size` 历史偏差：v22 旧 shard 脚本曾把配置值 1 覆盖为 shard 的 `num_designs`（9/17）；当前源码已使用独立 `SHARD_BATCH` 修复。历史审计必须看最终 shell shard command，而非只看 parameter plan。

## 5. 模块 C：inverse folding、folding、filtering/ranking

| 变量 | 默认/候选 | 作用 | 影响 |
|---|---|---|---|
| `inverse_fold_num_sequences` | 原生 1，常用 1/2 | 每 backbone 序列变体数 | D1 |
| `inverse_fold_avoid` | protein 常空；peptide/nanobody 常避 C | 限制逆折叠氨基酸 | D1 |
| `inverse_fold_checkpoint` | `boltzgen1_ifold.ckpt` | inverse-fold 权重 | D1 |
| `folding_checkpoint` | `boltz2_conf_final.ckpt` | folding/refolding 权重 | D1 |
| `affinity_checkpoint` | `boltz2_aff.ckpt` | affinity step | D1 |
| `budget` | 原生 30；adapter fallback 10 | filtering 最终集合大小 | D1 |
| `alpha` | 常用 0.001；安全范围 0.001–0.05 | quality/diversity selection | D1 |
| `filter_biased` | `true/false` | 组成偏置过滤 | D1 |
| `refolding_rmsd_threshold` | 常用 2.0/2.5 Å | RMSD filter | D1 |
| `additional_filters` | 用户静态拥有 | `--additional_filters` 硬过滤 | D1 |
| `metrics_override` | `None` | filtering ranking 权重 | D1 |
| `size_buckets` | `None` | 最终集合尺寸区间约束 | D1 |
| `config_overrides` | `[]` | `--config <step> key=value` | D0/D1 |
| `run_filtering` | 实际固定 true | 强制包含 filtering | D1，但不可调 |
| `keep_unfiltered_for_failure_analysis` | 常用 true | 保留失败分析材料 | I/N |

`additional_filters` 还会被 harness ingest 后再次应用于分析 cohort；它能改变正例、模板池与后续策略，但不改变初始 diffusion。`config_overrides` 是最宽泛的底层入口，validator 会拒绝非法 step、无 `=` token 及已知崩溃 key。

## 6. 模块 D：BoltzGen design spec 翻译参数

| 变量 | 默认/来源 | 写入位置 | 影响 |
|---|---|---|---|
| `binder_chain` | `B` | designed protein/template file `id` | D0 |
| `binder_sequence` | `str(job.binder_length)` | protein `sequence` | D0 |
| `target_chain` | job chain | target include/binding | D0 |
| `target_res_index` | 无 | target `include` fallback | D0 |
| `target_include` / `include` | 用户/裁剪 | file `include` | D0 |
| `target_binding_types` | 物化结果 | file `binding_types` | D0 |
| `not_binding` | 无 | `binding_types.chain.not_binding` | D0 |
| `binder_binding_types` | 无 | designed protein | D0 |
| `residue_constraints` | 无 | designed protein | D0 |
| `cyclic` | 无 | designed protein | D0 |
| `constraints` | 无 | spec 顶层 | D0 |
| `total_len` | 无 | 顶层 total_len constraint | D0 |
| `structure_groups` | 用户/translator | target file | D0 |
| `binder_structure_groups` | 模板派生 | template file | D0 |

## 7. 模块 E：target/binding translator

| 变量 | 默认/来源 | 作用 | 影响 |
|---|---|---|---|
| `auxiliary_hotspots` | LLM 可提小集合 | 转成 expanded BINDING residues | D0（翻译后） |
| `negative_binding_residues` | 静态/内部 | 转成 NOT_BINDING | D0（翻译后） |
| `binding_site_policy` | arm intent | `primary`/`primary_expanded`/`primary_negative` | D0（翻译后） |
| `epitope_crop_mode` | `disabled` | 生成 target include/binding crop | D0（启用时） |
| `allow_agent_epitope_crop` | false，隐藏静态项 | 解锁 agent 启用 crop | I→D0 |
| `target_context_policy` | `focus` intent | 期望改变 target context | N/I（当前缺完整翻译） |
| `output_binder_chain_hint` | `A` | 输出结构链检测 | I/N |

primary/expanded/negative 已统一物化并记录 accepted/rejected/effective provenance。BoltzGen 当前只看到二值 `BINDING/NOT_BINDING`，不会消费 per-residue float hotspot weight。

## 8. 模块 F：fragment template / structure redesign

| 变量 | 默认/来源 | 作用 | 影响 |
|---|---|---|---|
| `fragment_templates_enabled` | false，用户静态开关 | 是否允许 executable template branch | I→D0 |
| `fragment_template_gate` | `interchain_pae` | preserve 模板资格指标；可选旧 `iptm` | I→D0 |
| `fragment_interchain_pae_max` | 10.0 Å | PAE 资格阈值 | I→D0 |
| `fragment_template_min_quality` | 隐含 0.70 | executable template 最低质量 | I→D0 |
| `fragment_template_top_k` | 1 | 最大模板分支数 | I→D0 |
| `template_conditioned_fraction` | 建议 0.5，范围 0–0.8 | 模板分支与 template-free control 的预算比例 | D0（分配） |
| `binder_template` | internal-only | template file entity + redesign mask | D0 |
| `binder_templates` | internal-only Top-K | 为每个模板创建独立 job | D0 |
| `binder_template_proximity` | 8.0 Å | redesign mask `within_proximity` | D0 |
| `max_templates` | 硬编码 20 | 每轮 mined template 上限 | I |
| `library_size` | 硬编码 30 | 跨轮模板库规模 | I |
| fixed fraction 上限 | 硬编码 0.5 | 模板固定残基比例 gate | I→D0 |
| minimum designable residues | 硬编码 8 | 模板可设计自由度 gate | I→D0 |

真实模板控制链：

```text
上一轮结构
 -> PAE/iPTM + quality gate
 -> stable template_artifacts staging
 -> binder_template(s)
 -> template_exploit eligibility
 -> 每模板独立 DesignJob + template-free control
 -> template file entity / not_design / design_insertions
 -> boltzgen_redesign_mask.yaml
 -> --config inverse_folding data.cfg.design_mask_override=...
```

当前已实现：目录无关的 packagability、content-digest staging、有效模板才生成 `template_exploit`、Top-K、template-free control、exact motif residue groups、长度变长 insertion、requested/staged/applied/drop audit、历史 library fallback。

仍未闭环：无模板时 `template_conditioned_fraction` 没有明确 `not_applicable` 状态；package 失效仍可能降级为 template-free 而非拒绝重物化；安全 shortening、模板 blacklist/衰减、target-patch alignment、生产态 motif retention 归因未完成。`motif_retention_metrics()` 有定义但没有生产调用点。

## 9. 模块 G：动态 binder length

| 变量/常量 | 默认 | 作用 | 影响 |
|---|---|---|---|
| `auto_binder_length` | orchestrator 默认 true | 启用结构质量驱动的下一轮长度推荐 | I→D0 |
| `binder_lengths` | 从 range 派生/策略提出 | 每个长度生成独立 spec 并分片 | D0 |
| `GLOBAL_MIN_LENGTH` / `GLOBAL_MAX_LENGTH` | 30 / 180 | 无用户范围时的物理包络 | D0 |
| `max_lengths` | 4 | 每轮推荐长度数量 | D0 |
| `min_support_fraction` | 0.15 | 长度 bucket 证据门槛 | I |
| fold failure threshold | 0.4 | 达阈值向短长度移动 | I→D0 |
| weak interface threshold | 0.4 | 达阈值向长长度移动 | I→D0 |
| weak interface residue count | `<6` | 结构弱界面判据 | I |
| reliability threshold | `<0.5` | foldability failure | I |
| clash density threshold | `>0.15` | 长度策略 clash 判据 | I |

`binder_lengths` 会被 `_enforce_binder_length_range()` snap 回用户允许网格，因此可以在范围内收窄，但不能越过 `task.binder_length_range`。

## 10. 模块 H：typed strategy arm catalog

| Arm | 物化动作 | BoltzGen 影响 |
|---|---|---|
| `baseline_hold` | 保持 resolved parent | I |
| `site_primary_condition` | `binding_site_policy=primary` | D0（翻译后） |
| `site_expanded_condition` | primary + expanded BINDING | D0（翻译后） |
| `site_negative_exclusion` | primary BINDING + negative NOT_BINDING | D0（翻译后） |
| `target_context_focus` | 写 `target_context_policy=focus` | 当前 N/I；依赖 crop translator |
| `sampler_explore` | 写 `sampler_policy=explore` | 当前 N/I；没有自动转成 noise/step 参数 |
| `clash_select` | 写 heavy-atom clash `selection_policy` | 当前 N/I；metrics 已有但最终 gate/rank 消费证据不足 |
| `template_exploit` | structure-redesign template branch | D0 |

Arm 优先级和触发条件目前是代码常量，例如 primary 90、clash 85、expanded 80、template 75、target context 70、sampler 50、baseline 0。它们影响 arm 排序，但不是 YAML 可配置项。

## 11. 模块 I：主动学习、分支与回滚

| 变量 | 默认 | 作用 | 影响/状态 |
|---|---|---|---|
| `max_rounds` | 5 | 总迭代上限；CLI 可覆盖 | I |
| `max_retries` | 3 | 每 job 最大 attempt 数（含首次） | I |
| `strategy` | `successive_halving` | 声明策略名 | N：主闭环未发现消费者 |
| `top_k` | 8 | parent/evidence 选择 | I |
| `exploration_ratio` | dataclass 0.30；contract baseline 0.35 | exploit/explore parent 比例 | I；默认不一致 |
| `branch_width` | 1 | 同轮策略分支数量 | D0（改变预算拆分） |
| `promote_top_branches` | 1 | 声明晋级数 | N：未发现消费者 |
| `branch_budget_policy` | `equal` | 声明预算策略 | N/弱：实际按 weight 或硬编码均分 |
| `min_designs_per_branch` | 1 | controlled comparison 最低预算 | I |
| `enable_strategy_skills` | false | strategy skills 是否参与闭合 arm 排序 | I |
| `enable_exploitation_arms` | false | 是否允许 template exploitation | I→D0 |
| `min_current_positives_for_exploit` | 2 | template arm 正例门槛 | I |
| `prior_positive_decay_after_zero_rounds` | 2 | 历史正例衰减 | I |
| `near_miss_top_k` | 4 | prompt 近失例数 | I |
| `near_miss_min_confidence` | 0.30 | 近失例最低置信度 | I |
| `near_miss_weight` | 0.25 | 近失证据权重 | I |
| `enable_backtracking` | true | 允许 replay best | I |
| `regression_tolerance` | 0.25 | legacy reward 相对下降阈值 | I |
| `rollback_patience` | 2 | 连续非最佳轮耐心 | I |
| rollback metric tolerances | `(0.01,0.10,0.02,0.50,0.25)` | 新式 RoundRankKey 显著下降判断 | I，硬编码 |
| `min_round_for_rollback` | 1 | 最早回滚轮 | I，未暴露 |
| `stop_after_regressions` | 0 | 0 表示禁用 early stop | I，未暴露 |

`round_budget_weight` 和 `round_budget_allocation` 是 materialization/executor 字段。前者直接改变每个 branch 获得的 `num_designs`；后者是最终分配记录。

## 12. 模块 J：评价、成功门与结构分析

### 12.1 主成功阈值

| 常量 | 当前值 | 作用 | 影响 |
|---|---|---|---|
| `SUCCESS_IPTM_MIN` | 0.50 | strict-positive iPTM 门 | I（强） |
| `SUCCESS_PAE_MAX_ANGSTROM` | 10.0 | interaction PAE 门 | I（强） |
| `SUCCESS_PTM_MIN` | 0.70 | design pTM 门 | I（强） |
| `SUCCESS_RMSD_MAX_ANGSTROM` | 2.5 | refold RMSD 门 | I（强） |

它们不进入 BoltzGen，但决定 strict positive、parent ranking、template eligibility 和 rollback，是高影响的间接策略常量。当前未暴露给 HarnessConfig。

### 12.2 结构分析常量

- `contact_cutoff=5.0 Å`
- `clash_cutoff=2.0 Å`
- `clash_density_max=0.02`
- `fragment_window=8`
- `fragment_stride=4`
- `auto_detect_chains=true`

这些控制 contact、heavy-atom clash、fragment mining 与后续策略证据，属于 I。当前不是 YAML/CLI 变量。

### 12.3 `scoring.*` 当前无效

`ScoringWeights` 定义并可从 YAML 加载：`interface_confidence`、`hotspot_contact`、`binder_plddt`、`clash_penalty`、`diversity`、`sequence_designability`。但主闭环没有读取 `cfg.scoring`；实际排序使用 canonical `core_rank_key`。因此全部标记为 N（已定义但未消费）。

## 13. 模块 K：LLM 可调面、memory、自改进与质量协作

### 13.1 LLM 允许提出的 executable delta

公共合同只允许：

- `diffusion_batch_size`
- `step_scale`
- `noise_scale`
- `alpha`
- `inverse_fold_avoid`
- `filter_biased`
- `config_overrides`
- `auxiliary_hotspots`
- `epitope_crop_mode`
- `template_conditioned_fraction`
- `binder_lengths`

它们仍需经过类型正规化、hard-constraint freeze、物理 bounds/inertia、pressure conflict 和 full-job validation。LLM 不能直接控制 budget、round cap、资源、target 定义、模板 provenance、`additional_filters`。

### 13.2 Endpoint 变量

`enabled`、`default_model`、`base_url`、`model`、`provider`、`thinking`、`thinking_budget_tokens`、`max_prompt_bytes`、`context_window_tokens`、`max_output_tokens`、`timeout_seconds`、`max_retries`、`retry_backoff_seconds`、`request_lock_path`、`default_headers`、`extra_body` 均为 I/N：它们改变 proposal/context/可靠性，不直接进入 BoltzGen。

### 13.3 MemorySpec

`enabled`、`index_items`、`retrieval`、`semantic_rerank`、`compression`、`apply_prompt_budget`、`retrieval_candidate_limit=24`、`retrieval_top_k=8`、`mmr_lambda=0.7`、`max_active_items=24`、`compression_batch_size=6`、`max_summary_chars=1200`、`prompt_max_bytes=750000`：全部 I/N。

### 13.4 SelfImprovementSpec

`enabled`、`skill_path`、`max_active_rules=6`、`max_rules=48`、`promotion_min_support=2`、`retirement_contradictions=2`、`reward_improvement_threshold=0.01`、`strong_improvement_threshold=0.05`、`semantic_candidate_limit=8`、`semantic_confidence_threshold=0.72`、`conflict_resolution_enabled=true`、`prompt_max_bytes=24000`、`recent_round_window=5`：全部 I。

### 13.5 QualityCollaborationSpec

包括启用开关、revisions、performance/recovery thresholds、连续多 agent 轮限制、PAE/hotspot degradation ratio、confidence/high-impact gates、timeouts、cooldown、token budgets、API call cap。全部 I/N。

重要覆盖：`max_revisions` 虽可配置，orchestrator 当前强制最大为 1。

## 14. 模块 L：资源、运行时与主 CLI

### 14.1 ResourceSpec

| 变量 | 默认 | 作用 | 影响 |
|---|---|---|---|
| `backend` | local | local/taiji/dry_run | N/I；决定是否执行 |
| `host_num` | 1 | host 数与分片 | I |
| `host_gpu_num` | 1 | 注入 devices、worker 数 | I |
| `taiji_multi_host_mode` | native | native/split_jobs；支持 unified/fanout/split 别名 | I |
| `gpu_name` | V100 | 调度 GPU 型号 | N/I |
| `max_parallel_jobs` | 1 | orchestrator 并发 | N/I |
| `template_json` | None | Taiji 提交模板 | N/I |
| `image_full_name` | None | 运行镜像 | N/I |
| `timeout_seconds` | 3600 | 超时与重试 | I |
| `taiji_options` | `{}` | 调度扩展口 | N/I |

资源参数一般不改变模型语义，但会改变分片、成功率和实际调用数量。资源失败时内部策略会将 `devices` 减半、扩展 timeout；总 round budget 仍应保持不变。

### 14.2 RuntimeSpec

- `boltzgen_root`：选择 BoltzGen 安装与默认 artifacts；若版本/权重不同可产生 D0。
- `skill_registry_path`、`extend_memory`：I。
- `output_dir`、`project_root`：N；`project_root` 当前未发现生产消费者。
- `python_bin`：当前 BoltzGen closed-loop 构造固定 `boltzgen` 命令，故 N；对 ODesign/legacy path 有效。
- `odesign_root`：当前 BoltzGen 主闭环 N。

### 14.3 主入口 CLI

策略/生成相关：`--config`、`--max-rounds`、`--llm-config`、`--llm-model`、`--llm-thinking`、`--require-llm`、`--submit`、`--checkpoint-dir`、`--cache-dir`、`--moldir`、memory 开关、self-improvement 开关。

调度/日志相关：`--out`、`--taiji-client`、`--taiji-task-prefix`、`--taiji-remote-run-root`、`--taiji-poll-seconds`、`--taiji-wait-timeout`、`--no-wait-taiji`、`--result-sync-mode`、`--secret-config`、`--conda-base`、`--conda-env-name`、`--boltzgen-heartbeat-seconds`、`--silence/--boltzgen-silence-log`、`--force-new-run`。

## 15. 模块 M：环境变量与别名

### 15.1 直接影响实际进程

- `CUDA_VISIBLE_DEVICES`：每个 shard 的 GPU 可见性；同时 shard command 改成 `--devices 1`。这是直接调用控制，但不是模型策略参数。

### 15.2 多机调度

`HARNESS_HOST_COUNT`、`HARNESS_GPUS_PER_HOST`、`HARNESS_MULTI_HOST_MODE`、`HARNESS_RUN_TOKEN`、`HARNESS_HOST_REGISTRATION_TIMEOUT`（600）、`HARNESS_CLUSTER_BARRIER_TIMEOUT`（7200），以及 MPI rank/world-size 变量。均为 I/N。

### 15.3 日志

- silence 别名：`silence`、`silent`、`silence_logging`、`boltzgen_silence_log`、`boltzgen_silent_log` → `BOLTZGEN_SILENCE_LOG`
- heartbeat 别名：`log_heartbeat_seconds`、`boltzgen_log_heartbeat_seconds`、`heartbeat_seconds` → `BOLTZGEN_LOG_HEARTBEAT_SECONDS`

仅影响日志/活性观测，N。

### 15.4 凭据与 artifacts

`CEPH_SECRET` 和 endpoint `api_key_env` 决定挂载/API 可用性。`CHECKPOINT_DIR`、`CACHE_DIR`、`MOLDIR` 环境变量主要用于生成脚本检查；最终命令通常已经固化具体 CLI path，单独覆盖环境变量不一定改写 BoltzGen 参数。

`NUM_DESIGNS` 虽可写入 Taiji env，但运行脚本未用它构造命令；真正设计数来自 `--num_designs`，因此当前为 N/重复元数据。

## 16. 模块 N：隐藏可调项与硬编码策略常量

这些字段/阈值有真实消费者，但没有完整进入 typed 用户 schema；由于 `search_space.boltzgen` 是开放 dict，其中部分仍可由静态 YAML 写入。

| 项目 | 默认 | 影响 |
|---|---:|---|
| `fragment_template_min_quality` | 0.70 | I→D0 |
| `allow_agent_epitope_crop` | false | I→D0 |
| `negative_binding_residues` | `[]` | D0 |
| `binder_template_proximity` | 8.0 Å | D0 |
| `max_templates` | 20 | I |
| template `library_size` | 30 | I |
| max fixed fraction | 0.5 | I→D0 |
| min designable residues | 8 | I→D0 |
| auxiliary hotspot `max_items` | 3 | D0 |
| auxiliary hotspot `max_distance` | 15 residues | D0 |
| structure `contact_cutoff` | 5.0 Å | I |
| heavy-atom `clash_cutoff` | 2.0 Å | I |
| `clash_density_max` | 0.02 | I |
| fragment window/stride | 8 / 4 | I |
| prompt compaction limits | 多组固定上限 | I |
| agent temperature/max_tokens | 各 agent 内硬编码 | I |
| resource retry devices/timeout policy | GPU 减半、timeout 扩展 | I |

这些项应正式纳入 schema，或明确标记为不可运行时调整的 policy constants。

## 17. 模块 O：已定义但当前不生效/断链

| 字段 | 当前状态 |
|---|---|
| `hotspot_weight`、`prioritize_hotspots`、`clash_filter`、`module_guided_repair`、`module_guided_exploitation`、`exploit_fragment_modules` | deprecated audit-only；不进入 CLI/spec |
| `selection_policy` | 有 clash intent 和消费报告标记，但未确认最终候选 gate/rank 生产消费者 |
| `sampler_policy` | `sampler_explore` 只写 `explore`，未自动翻译为 noise/step 等参数 |
| `target_context_policy` | 写入 `focus` intent，实际 crop 仍依赖独立 epitope translator |
| `active_learning.strategy` | 未发现主闭环读取 |
| `active_learning.promote_top_branches` | 未发现消费者 |
| `active_learning.branch_budget_policy` | 实际预算逻辑不读取；按 `round_budget_weight` 或均分 |
| `scoring.*` | YAML 可加载，但主闭环不读取 `cfg.scoring` |
| `runtime.project_root` | 未发现生产消费者 |
| `runtime.python_bin` | 当前 BoltzGen closed-loop 不使用 |
| `DesignJob.seed` | BoltzGen 无 seed CLI；仅 harness/ODesign 兼容 |
| Taiji env `NUM_DESIGNS` | 运行脚本未消费 |
| `run_filtering` | 看似配置项，实际从 loader 到 executor 均固定 true |
| ODesign 参数 | 对 legacy pipeline 有效，对当前 BoltzGen closed-loop 无效 |
| 原生 BoltzGen YAML 未抽取字段 | 不透传，可能丢失 binder id 等信息 |

## 18. 配置默认值冲突

1. `num_designs`：BoltzGen 原生 10000；adapter/DesignSpecAgent fallback 50；DesignParameterPlan 100；正式闭环由用户 round cap 派生。
2. `budget`：BoltzGen 原生 30；adapter fallback 10；实验常见 20 或 99999。
3. `exploration_ratio`：dataclass 0.30；PARAM_BOUNDS baseline 0.35；实验配置多为 0.35。
4. `step_scale/noise_scale`：BoltzGen 原生为 schedule；bounds baseline 0.8/0.7；遗留 DesignParameterAgent 可提出 1.8/1.0，超过当前安全上限，但该 agent 未见生产调用。
5. `alpha`：protocol/CLI/agent/contract 多层都可补默认。
6. `model_order`：SearchSpace dataclass 默认含 BoltzGen 和 ODesign；loader 默认只 BoltzGen。

因此“未显式设置”时的实际值取决于调用入口。正式闭环应以提交后的 `effective_execution_plan.json`、最终 spec、最终 command 和 shard plan 为准。

## 19. 历史 v22 与当前源码差异

- v22 `module_exploitation` job 实际 `template_conditioned=false`，最终 CLI/spec 没有模板、module ID、hotspot weight 或 clash repair；它是有 exploitation 名称但无对应生成 intervention 的普通任务。
- v22 parameter plan 保留旧伪策略字段，但这些字段没有进入最终 BoltzGen command/spec。
- v22 旧脚本存在 diffusion batch 被 shard size 覆盖的问题；当前源码已修复。
- 当前源码会写 `boltzgen_parameter_consumption.json` 和 `effective_execution_plan.json`；被审计的 v22 产物没有这些文件，说明其来自较早版本。

审计优先级应为：

1. 最终 shard shell command；
2. 每长度 design spec；
3. redesign mask；
4. cluster shard plan；
5. 当前版本 consumption report / effective execution plan；
6. 之后才是 parameter plan、next-round config、LLM proposal 和策略 metadata。

## 20. `final_executable_strategy_governance.plan.md` 状态核对

### 已基本实现但文档仍标 pending

- 移除 mountability 目录黑名单，改为 packagability 检查
- stable `template_artifacts` staging
- valid-template-only `template_exploit`
- template-free control
- typed canonical arm catalog
- deprecated strategy audit
- unknown model 不再回退 BoltzGen
- full-job validation 与 agent delta 分离
- shard 独立保留 diffusion batch size
- primary/expanded/negative binding 物化与 provenance
- separate coverage
- heavy-atom clash metrics
- parameter consumption report
- effective execution plan artifact
- exact non-contiguous motif groups
- insertion metadata
- template library fallback

### 已在本轮完成或显著推进

- canonical schema：正式任务 YAML 已迁移至严格 owner schema，未知/旧用户字段会拒绝，并单向编译为运行时配置。
- deterministic resolver：新增集中 execution governance，统一 `num_designs`、multi-GPU budget floor、sampler bounds、applicability 与 lineage；旧 orchestrator 的 merge/pressure/rollback 仍需后续完全搬入。
- immutable plan：effective plan 现包含 plan digest、artifact digests、consumer receipts 和 num-designs parity；job params 在 resolver 前仍存在兼容期原地修改。
- 单一 CLI renderer：adapter 与 DesignSpecAgent 已共用 `boltzgen_renderer.py`。
- template package failure：template exploit 现 fail closed，不再静默降级为 template-free。
- clash selection：heavy-atom clash policy 已接入候选 gate/rank 并输出 selection artifact。
- motif retention：已接入 source-template → final-refold 生产指标；initial/inverse-fold 阶段仍因稳定 lineage 缺失而明确标记 unavailable。

### 尚未完成

- 将现有 lineage/parity 从关键字段覆盖扩展为完整逐字段 semantic projection
- template blacklist/decay/target-specific utility
- safe shortening/cropping
- target-patch alignment 与 aligned motif RMSD
- initial-design/inverse-fold/final-refold 四阶段稳定 candidate lineage 与完整归因
- weighted hotspot 原生模型扩展

因此该 plan 的方向基本正确，但 front matter 中所有 todo 为 pending 已明显过期。

## 21. 是否可直接控制 BoltzGen：最终矩阵

### 可以直接控制 diffusion/backbone

- target structure/context：`target_structure_path`、`target_include`、`structure_groups`
- binding condition：`hotspots`、有效的 `target_binding_types`、`auxiliary_hotspots`/negative residues 翻译结果
- length：`binder_length_range`、`binder_length_step`、最终 `binder_lengths`/`binder_sequence`
- sampler：`protocol`、`num_designs`、`diffusion_batch_size`、`step_scale`、`noise_scale`、`design_checkpoints`
- generic override：影响 design step 的合法 `config_overrides`
- template：有效 `binder_template`、`binder_template_proximity`、redesign mask、insertions

### 只控制序列、验证或最终输出集合

- `inverse_fold_num_sequences`、`inverse_fold_avoid`、inverse/folding/affinity checkpoints
- `budget`、`alpha`、`filter_biased`、`refolding_rmsd_threshold`
- `additional_filters`、`metrics_override`、`size_buckets`
- analysis/filtering step overrides

### 通过任务物化间接变成直接控制

- `fragment_templates_enabled`、gate、PAE threshold、quality、Top-K、conditioned fraction
- `auto_binder_length`
- `binding_site_policy`
- `epitope_crop_mode` 与 allow-agent-crop
- `branch_width`、`round_budget_weight`
- 回滚/replay 最佳轮

### 不能直接控制 BoltzGen

- LLM model/thinking、memory、自改进、质量协作
- success thresholds、failure taxonomy、结构分析阈值
- parent selection、near-miss、strategy skills
- resource、heartbeat、silence、同步模式
- deprecated hotspot/clash/module 字段
- 当前无消费者的 typed intents 和配置项

## 22. 建议的 canonical 分类

建议把配置正式收敛为以下 owner 模块：

1. `task_hard_constraints`：target、primary hotspots、include/binding/groups、length range/step、round cap、freeze flags。
2. `boltzgen_design_native`：protocol、num designs、diffusion batch、step/noise、design checkpoints、steps。
3. `boltzgen_inverse_fold_and_validation`：inverse-fold count/avoid/checkpoint、folding/affinity checkpoint。
4. `boltzgen_filtering_ranking`：budget、alpha、biased filter、RMSD、metrics/additional filters、size buckets、config overrides。
5. `harness_target_translator`：auxiliary/negative residues、binding-site policy、crop mode、allow-agent-crop。
6. `harness_template_policy`：enabled、gate、PAE、quality、Top-K、fraction、proximity、fixed/designable/library/failure policy。
7. `harness_selection_and_evidence`：clash/contact/fragment thresholds、success gates、selection policy。
8. `active_learning_and_rollback`：rounds/retries/top-k/exploration/branching/exploit/near-miss/backtracking。
9. `runtime_resources`：backend、hosts/GPUs/mode/type/timeout/artifact paths/Taiji/logging。
10. `llm_context_learning`：endpoint、memory、self-improvement、quality collaboration；明确标注“不直接进入 BoltzGen”。
11. `deprecated_audit_only`：旧 hotspot/clash/module 字段和 legacy arm names。

## 23. 安全发现

审计发现 `configs/llm_endpoints.ds.json` 含明文真实 API key 与 Ceph secret。本文不记录其值。该文件应视为敏感凭据文件，避免提交、分享或保留在普通配置目录；建议立即轮换已暴露凭据并迁移到环境变量或受控 secret store。

## 24. 最终结论

当前 Harness 已具备较清晰的“LLM 稀疏提案 → 约束/校验 → job 物化 → spec/CLI”的框架，但还没有完全达到唯一 canonical schema 和唯一 resolver。最可靠的判断原则是：一个策略变量只有在最终出现在 CLI、design spec、redesign mask、明确的 harness transformation/selection，或实际 budget/branch materialization 中，才应被称为 executable。

当前最重要的后续治理点：

1. 在 YAML 加载边界建立完整 typed BoltzGen schema，关闭开放 dict 的隐式接口。
2. 合并多套默认值和 adapter/DesignSpecAgent 的双 CLI renderer。
3. 将分散的 merge/freeze/clamp/pressure/rollback 收敛为单一 resolver。
4. 对 `selection_policy`、`sampler_policy`、`target_context_policy` 要么实现消费者，要么标记 metadata-only。
5. 将 clash selection 与 motif retention 接入生产闭环和效果归因。
6. 为无模板、template drop、shard override 等建立逐字段 lineage/parity 断言。
7. 清理明文 secret 与旧伪 executable 命名。
