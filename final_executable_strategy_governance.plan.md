---
name: 最终可执行策略治理
overview: 基于前一版合并方案形成最终执行计划：先修复 mountability 对 template exploitation 的系统性阻断和失真审计，再统一配置契约、策略目录与 deterministic execution plan，最后升级 fragment-template 的结构语义和效果验证。核查确认 mountability 不直接过滤普通 LLM 参数；被异常排除的是 LLM 可调的 `template_conditioned_fraction` 所依赖的内部 `binder_template(s)` 物化，以及 LLM 排名可选的 `module_exploitation` 策略线。
todos:
  - id: restore-template-path
    content: 移除目录名 mountability 黑名单，建立稳定模板 staging 与可打包校验
    status: pending
  - id: restore-template-arms
    content: 仅在有效模板存在时启用 template arm 和 template_conditioned_fraction，并修复降级审计
    status: pending
  - id: canonical-resolution
    content: 统一完整配置 schema、LLM delta、策略 intent、deterministic resolver 与 immutable execution plan
    status: pending
  - id: cleanup-real-strategies
    content: 删除伪 executable 字段并建立唯一 typed arm catalog 和 intervention digest
    status: pending
  - id: upgrade-template-conditioning
    content: 实现 coherent frame、exact motif groups、长度 insertions、library fallback 与模板级失败管理
    status: pending
  - id: downstream-translators
    content: 修复 primary/expanded/negative binding 物化，实现 coverage、clash、module 显式转换与消费报告
    status: pending
  - id: validate-replay
    content: 增加 motif retention、control 归因、逐字段 lineage、生产路径端到端测试和 v22 离线 replay
    status: pending
  - id: weighted-hotspot-rfc
    content: 独立评估 BoltzGen 原生 weighted hotspot conditioning、checkpoint 兼容与重训需求
    status: pending
isProject: false
---

# 最终执行计划：可执行策略、Mountability 与 Fragment Template 治理

## 最终核查结论
- LLM 当前可直接提出的参数来自 [`config_parameter_contract.py`](/aceph/daweihuang/program/binder-harness/binder_harness/agents/config_parameter_contract.py)，包括 sampling、targeting、filtering 和 `template_conditioned_fraction` 等；mountability 规则不直接检查或删除这些 LLM 数值。
- `binder_template`、`binder_templates`、`binder_template_proximity`、`exploit_fragment_modules`、`module_guided_exploitation` 是内部字段，只允许 [`fragment_template_mining_agent.py`](/aceph/daweihuang/program/binder-harness/binder_harness/agents/fragment_template_mining_agent.py) 经 provenance/PAE gate 生成，不允许 LLM 直接构造。
- 被 mountability 系统性失效的 LLM/策略输出有两类：
  1. `template_conditioned_fraction`：LLM/policy 可以调整该比例，但标准 Taiji source 被拒绝后没有 `binder_template(s)`，[`strategy.py`](/aceph/daweihuang/program/binder-harness/binder_harness/active_learning/strategy.py) 不会生成模板分支，该变量成为无效配置。
  2. `module_exploitation`：LLM ranking 可以从封闭 arm catalog 中选择该 arm，但 mountability 清空模板 payload 后仍可能生成名为 module exploitation、实际 `template_conditioned=false` 的普通任务。
- 其余 LLM 可调参数，如 `alpha`、`noise_scale`、`step_scale`、`diffusion_batch_size`、`inverse_fold_avoid`、`filter_biased`、`config_overrides`、`auxiliary_hotspots` 和 `epitope_crop_mode`，不会被 mountability 系统性排除。`hotspot_weight`、`prioritize_hotspots`、`clash_filter`、`module_guided_repair` 虽不受 mountability 影响，但按前一版治理结论需要取消伪 executable 语义或转换为真实下游操作。

## 第一阶段：立即恢复被异常排除的模板策略线
1. **用 packagability 取代目录名黑名单**
   - 修改 [`fragment_template_mining_agent.py`](/aceph/daweihuang/program/binder-harness/binder_harness/agents/fragment_template_mining_agent.py)：移除对 `taiji_project_package`、`/outputs/boltzgen_output/`、`/intermediate_designs` 的无条件拒绝。
   - 新资格检查只验证：文件存在、是普通可读 CIF/PDB/mmCIF、大小合理、可解析、binder chain/span 有效，并能复制到新 package；目录名称不再决定资格。
   - 保留对不存在的远端路径、目录、断链 symlink、不可读文件和不支持格式的拒绝。

2. **建立稳定 template artifact staging**
   - 在 round 结构回收完成后，将入选 source complex 发布到 run-owned `template_artifacts/` 稳定目录；采用内容 digest 命名，原子写入并去重。
   - artifact 保存 `original_source_path`、`staged_source_path`、digest、round/job/candidate、binder/target chain map、source length、motif residues、PAE/quality 与发布时间。
   - [`design_spec_agent.py`](/aceph/daweihuang/program/binder-harness/binder_harness/agents/design_spec_agent.py) 只消费 staged path，并继续将其复制/打包为当前 Taiji package 的 `inputs/template_*`；worker 不需要重新挂载上一轮 package。
   - staging 失败时拒绝单个 template，不关闭整个模板功能，并记录结构化原因。

3. **恢复 `module_exploitation` 与 `template_conditioned_fraction` 的真实语义**
   - [`strategy.py`](/aceph/daweihuang/program/binder-harness/binder_harness/active_learning/strategy.py) 只有在存在至少一个已验证 `binder_template` 时才允许物化 `module_exploitation`。
   - 如果模板 payload 不存在：删除/拒绝该 arm，不得生成同名 template-free job；由 `baseline_hold`、`sampler_explore` 等真实 arm 补位。
   - `template_conditioned_fraction` 仅在模板集合非空时进入 execution plan；否则标记 `not_applicable:no_effective_templates`，不得作为已执行变量参与 pressure、rollback 或效果归因。
   - 模板 Top-K 分支始终保留明确的 template-free control；预算按最终有效模板数重新计算。

4. **修复后期降级和审计状态**
   - 在 [`design_spec_agent.py`](/aceph/daweihuang/program/binder-harness/binder_harness/agents/design_spec_agent.py) 统一生成 `template_requested`、`template_staged`、`template_applied`、`template_drop_reason` 和 `effective_template_id`。
   - package 阶段若模板失效，必须同步修改 effective job params、run manifest、strategy exposure 和 lineage，不得保留 `template_conditioned=true`。
   - pre-submit 阶段对 requested/applied 状态做强断言；无模板却声明 template arm 的任务直接拒绝并重新物化，而不是静默伪装执行。

## 第二阶段：统一配置契约和唯一执行计划
1. **建立 canonical BoltzGen schema**
   - 在 [`config.py`](/aceph/daweihuang/program/binder-harness/binder_harness/config.py)、[`config_parameter_contract.py`](/aceph/daweihuang/program/binder-harness/binder_harness/agents/config_parameter_contract.py)、[`model_input_spec.py`](/aceph/daweihuang/program/binder-harness/binder_harness/agents/model_input_spec.py) 中区分：用户 YAML、完整 runner 字段、LLM delta、策略 intent、内部派生、资源/runtime 和 deprecated metadata。
   - 每个字段声明 owner、类型、默认值、来源优先级、下游消费位置、是否允许 LLM 提议、是否需要 adapter translator，以及无消费者时的处理方式。
   - 首批明确分类：
     - BoltzGen native：`noise_scale`、`step_scale`、`alpha`、`diffusion_batch_size`、`inverse_fold_avoid`、真实 CLI filters/config overrides 与 binder lengths；
     - adapter-translated：primary/expanded/negative binding residues、harness clash selection、可物化的 structure template/motif；
     - harness-only intent/deprecated：旧 `hotspot_weight`、`prioritize_hotspots` 与抽象 module 开关；
     - unsupported/native extension：per-residue weighted hotspot 和不能转成坐标/掩码的抽象 fragment guidance。
   - full-job schema 不再复用不完整的 Agent delta 白名单；未知 model 不再静默回退 BoltzGen。
   - 明确 YAML、CLI、resolver defaults、executor-derived 参数和 runtime env 的优先级；收敛 heartbeat/silence 与 multi-host alias，用户环境变量不得绕过已审计的预算和资源计划。

2. **改为 intent → sparse proposal → deterministic resolver**
   - [`active_learning_policy_agent.py`](/aceph/daweihuang/program/binder-harness/binder_harness/agents/active_learning_policy_agent.py) 仅输出稀疏候选变化。
   - [`strategy_arm_ranking_agent.py`](/aceph/daweihuang/program/binder-harness/binder_harness/agents/strategy_arm_ranking_agent.py) 保持只排序封闭 arm，不生成值或 payload。
   - [`strategy.py`](/aceph/daweihuang/program/binder-harness/binder_harness/active_learning/strategy.py) 的 arm 只表达策略族、方向、证据和 branch role，不直接发明参数值。
   - [`orchestrator.py`](/aceph/daweihuang/program/binder-harness/binder_harness/orchestration/orchestrator.py) 引入唯一 resolver，按固定顺序处理 owner、用户冻结、下游 capability、方向一致性、物理 bounds、inertia、pressure conflict、rollback、离散长度、template provenance/可用性和最终预算，输出 immutable execution plan。
   - 删除 `_directional_hotspot_weight()` 等由 arm 自动发明数值的逻辑；`hold` 表示保留 resolver 结果，而不是恢复 parent value；controlled comparison 也必须先形成 typed intent，再由 resolver 生成分支值。
   - 保留 `BinderLengthPolicyAgent` 对离散长度集合的确定性 ownership；LLM 不接管用户预算、长度外边界、target 定义、资源、rollback 和 template provenance。
   - 策略 materializer、预算分配、validator、adapter 与 executor 不得再修改 model parameters；所有执行器派生值必须作为独立字段写入 plan/lineage。

3. **修复 validation/executor 覆盖**
   - full-job validation 输出完整 sanitized partition，并显式记录 removed keys/tombstones。
   - [`run_closed_loop_orchestrator.py`](/aceph/daweihuang/program/binder-harness/scripts/run_closed_loop_orchestrator.py) 使用 domain replacement，而不是对原 job `dict.update()`；local、Taiji 和 retry 共用同一应用函数。
   - 清除 shard 对 `diffusion_batch_size` 的静默覆盖，或将 shard design count 改名并建模为明确的 executor-derived 字段；逐 shard 断言 batch/spec/command parity。
   - 将 adapter defaults 前移到 resolver/final plan；adapter 仅序列化，不再补写会导致 `next_round_config.yaml` 与最终命令不一致的默认值。
   - 收敛 [`boltzgen_adapter.py`](/aceph/daweihuang/program/binder-harness/binder_harness/models/boltzgen_adapter.py) 与 DesignSpecAgent 的重复 CLI 渲染，确保只有一套最终命令来源。

## 第三阶段：清理伪策略并建立唯一 executable arm catalog
1. 将 `hotspot_weight`、`prioritize_hotspots`、`clash_filter`、`module_guided_repair`、`module_guided_exploitation`、`exploit_fragment_modules` 从可执行参数表移出；旧 snapshot 只作为 deprecated audit 读取。
2. 建立唯一 typed arm catalog：
   - `baseline_hold`
   - `site_primary_condition`
   - `site_expanded_condition`
   - `site_negative_exclusion`
   - `target_context_focus`
   - `sampler_explore`
   - `clash_select`
   - `template_exploit`
3. hotspot 通过真实 `BINDING`/`NOT_BINDING`、primary/expanded/negative coverage 表达；clash 通过 harness 重原子 gate/rank 表达；模板只通过有效 native template spec 表达。
4. 合并 [`strategy.py`](/aceph/daweihuang/program/binder-harness/binder_harness/active_learning/strategy.py) 中动态和 builtin 两套 arm 定义；Skill registry、LLM ranking、rollback 和 materializer 共用同一 catalog。
5. 每个 arm 在预算分配前生成 effective intervention digest；spec、CLI 和 selection policy 均与 control 相同的 arm 自动去重或拒绝。
6. 删除旧伪字段关联的 bounds、prompt、merge owner、pressure conflict、rollback 比较、数值 arms 与 quality guidance；兼容逻辑集中到版本化 legacy reader，当前文档和新 artifact 统一使用 `exploration_arm`/typed policy 名称。

## 第四阶段：补齐真实 Binding-site、Coverage 与 Clash 转换
1. **统一 residue 单一事实源**
   - 在每个 job 物化前，从用户 primary residues、证据支持的 expanded residues 和明确 off-patch negative residues 重新构造 `target_binding_types`；禁止继承 stale `target_binding_types` 后再叠加热点。
   - 修复当前已有 binding types 可能抑制 `DesignJob.hotspots`/`auxiliary_hotspots` 的路径；输出 `accepted`、`rejected`、`effective` residues artifact，并保留 primary/expanded/negative provenance。
   - BoltzGen 当前只能把 primary 与 expanded 正残基序列化为同等级 `BINDING`；不得将旧数值 weight 描述为 diffusion guidance。

2. **实现可执行 site policies**
   - `site_primary_condition` 仅写入用户 primary residues；`site_expanded_condition` 合并经 resolver 校验的 expanded residues；`site_negative_exclusion` 将 off-patch residues 写入 `NOT_BINDING`。
   - `target_context_focus` 只通过真实 `target_include`、`include_proximity` 和 `structure_groups` 改变目标上下文，并遵守用户 crop hard constraint。
   - 分开计算 primary、expanded、negative coverage、真实跨链接触和 bindsite RMSD/ranking，避免 expanded 命中掩盖 primary miss。

3. **实现 harness clash selection**
   - 基于最终结构计算真实跨链重原子 clash count/density/gate/rank，并将其作为 harness 候选选择策略；只有存在明确 BoltzGen metrics column 时才生成 `additional_filters`。
   - 不再把 `clash_filter=true` 描述为 BoltzGen 内部 repair，也不复用训练数据清洗的 `ClashingChainsFilter`。

4. **生成参数消费报告**
   - [`design_spec_agent.py`](/aceph/daweihuang/program/binder-harness/binder_harness/agents/design_spec_agent.py) 为每个字段记录 `CLI`、`design_spec`、`harness_transform`、`allocation`、`runtime`、`rejected` 或 `metadata_only`。
   - purported executable 参数若未进入最终 CLI/spec，也没有已测试的 harness transformation，必须在提交前失败。

## 第五阶段：将 Fragment Template 升级为合理的片段模块优化
1. **精确模板数据模型**
   - 使用 `binder_residue_ids` 生成 BoltzGen 1-based chain-local comma/range，支持不连续 motif，不再把 `min..max` 当作完整模块。
   - pre-submit 解析 staged complex，验证 chain、residue IDs、insertion codes、source length、target patch、motif 连续性和坐标完整性。
   - 真正使用可配置 `min_quality`；同时记录 PAE matched/missing/invalid 数、mountable/staged 数和最终入选原因。

2. **一致坐标系与结构条件化**
   - 优先从同一 prior complex 提取 binder motif 和 target patch；如使用 canonical target，先完成 target alignment 并记录变换矩阵。
   - 仅 exact motif 与必要 target patch 设置正 `structure_groups`；新 scaffold 保持 group 0，避免默认条件化整条旧 binder。
   - motif 与 target patch 同组表示保留相对 pose；不同组只保留各自内部几何。所有 artifact 明确说明这是 soft coordinate conditioning，不是原子冻结。

3. **长度和 design insertion**
   - template branch 默认绑定 source binder length，不再按多个 `binder_lengths` 重复运行相同 spec。
   - 目标长度更长时，通过 `design_insertions` 明确 N/C 端或连接区插入，并重算 motif 与 inverse-fold indices。
   - 目标长度更短时，只允许经验证的非 motif 裁剪；不能安全裁剪时拒绝 template-length pair。
   - 计算 fixed residue count/fraction 和 designable count；超过固定比例上限或设计自由度不足时拒绝模板。

4. **历史 library 与模板级失败管理**
   - executable 选择池使用“当前轮 staged templates + prior staged library”；当前轮不足时才从历史库补齐。
   - 按 source candidate、target patch、cluster、round 和结构相似性去重，避免 Top-K 被近重复片段占满。
   - 为 template ID 维护 success、retention、config failure 和 regression 计数；支持衰减、blacklist 和 target-specific utility。

## 第六阶段：结果验证、Lineage、回放和测试
1. final refold 后计算 source motif、initial design、inverse-fold result、final refold 之间的 motif RMSD，并记录 target-patch-aligned RMSD、sequence identity、接触/热点保持率、clash、pTM/pLDDT 和核心 refolding 指标。
2. template-conditioned 与 template-free control 使用同一轮、同类 sampler 和可比较预算；模板比例只能根据模板级增量效果调整，不能根据混合总体指标归因。
3. 每个 job 输出逐字段 lineage：base、typed intent、LLM proposal、resolver decision/clamp、arm/branch、validation、executor derivation、最终 CLI/spec/selection 位置和消费状态；模板另含 provenance、residue mapping、staging digest 和 drop/reject 原因。
4. rollback/replay artifact 记录 config contract、resolver、adapter、BoltzGen 版本、template digest 和 execution-plan digest；明确区分“配置精确 replay”与“无稳定 seed 时输出不可逐结构复现”。
5. 扩展测试：
   - [`test_strategy_improvements.py`](/aceph/daweihuang/program/binder-harness/scripts/test_strategy_improvements.py)：policy 稀疏 delta、value-free arms、`hold` 不覆盖、controlled comparison 经 resolver、typed site policies、auxiliary hotspot 全链路、无有效模板时禁止 template arm、模板预算/control、library fallback、template blacklist。
   - [`test_boltzgen_taiji_agents.py`](/aceph/daweihuang/program/binder-harness/scripts/test_boltzgen_taiji_agents.py)：最终 binding residues、primary/expanded/negative coverage、clash selection、标准上一轮 Taiji output staging、coherent complex、exact motif groups、insertions、post-insertion indices、CLI/spec/shard parity、manifest effective state和未消费字段拒绝。
   - [`test_length_and_pae_gate.py`](/aceph/daweihuang/program/binder-harness/scripts/test_length_and_pae_gate.py)：真实 min-quality、PAE/mountability 分层统计、source/patch 多样性。
   - [`test_config_cleanup.py`](/aceph/daweihuang/program/binder-harness/scripts/test_config_cleanup.py)：参数分区完整性、owner、merge-order independence、resolver clamp/方向一致性。
   - [`test_retry_limits.py`](/aceph/daweihuang/program/binder-harness/scripts/test_retry_limits.py)：local/Taiji/retry replacement、删除 tombstone、空 corrected config 与 runtime key 保留。
   - [`test_resume_support.py`](/aceph/daweihuang/program/binder-harness/scripts/test_resume_support.py)：immutable execution-plan replay、版本兼容、template/config provenance。
6. 使用 SC2RBD/PD-L1/TNFA v22 artifacts 离线 replay：证明修复前 `module_exploitation` 无 template intervention；修复后标准 Taiji source 能生成 staged artifact、template job、redesign spec/mask，并在最终结果中得到 motif retention 指标。

## 后续独立能力：BoltzGen 原生 Weighted Hotspot
- 当前主计划只实现 binary `BINDING`/`NOT_BINDING`、coverage/ranking 和 post-generation selection，不修改 BoltzGen 网络或 checkpoint。
- 若后续确认需要 diffusion-time per-residue float weight/priority conditioning，单独修改 [`schema.py`](/aceph/daweihuang/program/binder-harness/models/boltzgen/src/boltzgen/data/parse/schema.py)、feature pipeline、[`trunk.py`](/aceph/daweihuang/program/binder-harness/models/boltzgen/src/boltzgen/model/modules/trunk.py) 与训练配置，并评估 checkpoint 兼容、微调/重训和回归基线。
- 该模型能力必须单独 RFC、单独验收，不与 harness 配置治理和 template staging 修复混入同一变更。

## 执行顺序与验收门槛
1. 先完成 mountability/staging、template arm eligibility、effective audit 修复，并运行生产路径集成测试；这是恢复现有策略线的 P0 阶段。
2. 再完成 canonical schema、resolver、validation replacement、真实 binding/coverage/clash translators 和 arm catalog；此阶段后所有获得预算的策略必须有唯一有效 intervention。
3. 然后升级 motif-only conditioning、长度 insertion 和模板级 outcome attribution；在此之前功能名称应保持 `structure_redesign_template`，不得宣称为独立 motif transplantation。
4. weighted hotspot 原生 conditioning 仅在独立 RFC 获批后进入模型侧开发，不阻塞上述 harness 改造。

最终验收必须同时满足：
- 标准 `taiji_project_package/outputs/boltzgen_output/...cif` 可被安全 staged 并用于下一轮，不依赖旧 package remount。
- 无有效模板时 `module_exploitation/template_exploit` 不会出现，`template_conditioned_fraction` 被标为不可应用。
- 有效模板时，Top-K template jobs 与 template-free control 均真实生成，manifest 的 requested/applied 状态与最终 spec/command 一致。
- primary/expanded/negative residues 从单一事实源物化，accepted/rejected/effective artifact 与最终 `target_binding_types` 一致；primary coverage 不被 expanded coverage 掩盖。
- clash 策略基于真实跨链重原子指标，旧 `clash_filter`/module/hotspot weight 开关只存在于 deprecated audit。
- 所有 LLM 参数和策略 arm 都能追溯到实际 design spec、CLI、harness transformation 或 selection policy；无消费者字段不得获得预算或进入效果归因。
- template branch 完成 chain/residue/frame/length/fixed-fraction 校验，并输出 post-refold motif retention 指标。
- `next_round_config`、immutable execution plan、validated params、spec、CLI、shard 命令和 lineage 保持一致。