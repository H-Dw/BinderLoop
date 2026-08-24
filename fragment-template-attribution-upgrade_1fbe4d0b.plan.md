---
name: fragment-template-attribution-upgrade
overview: 把现有 fragment template 的 soft structure-redesign 升级为 target-frame 一致、残基映射可追踪、整轮预算语义明确、四阶段可归因且具备模板 utility 生命周期的生产闭环。改造集中在 harness、BoltzGen writer/manifest 和分析层，不修改模型网络或 checkpoint。
todos:
  - id: template-policy-plan
    content: 扩展 harness_template_policy 和 immutable TemplateApplicationPlan，统一整轮 conditioned budget
    status: in_progress
  - id: target-alignment
    content: 实现 source/current target patch mapping、Kabsch alignment、coherent frame 与 fail-closed gates
    status: pending
  - id: residue-length-map
    content: 实现安全 insertion/cropping 和 source→effective residue mapping
    status: pending
  - id: stage-lineage
    content: 在 BoltzGen writer/CLI 与 ingestion 中建立四阶段 global candidate lineage
    status: pending
  - id: motif-attribution
    content: 实现 self-aligned/target-patch-aligned 四阶段 motif、pose、sequence/contact attribution
    status: pending
  - id: template-utility
    content: 实现 matched control、target-specific outcome ledger、utility、cooldown 和 blacklist
    status: pending
  - id: template-parity
    content: 增加 template alignment/residue-map/spec/mask/shard/lineage digest parity 与 fail-closed 重物化
    status: pending
  - id: template-tests
    content: 扩展 alignment、mapping、lineage、attribution、budget、resume 和 SC2RBD replay 测试
    status: pending
isProject: false
---

# Fragment Template 生产归因升级

## 目标与验收定义
- 将模板策略从“source motif 局部形状约束 + source→final 终点检查”升级为“target patch 对齐后的 coherent template intervention + source/initial/inverse-fold/final 四阶段归因”。
- 每个 template job 必须证明：模板来源有效、source/current target 在同一坐标系、source→effective residue mapping 完整、整轮 conditioned budget 符合用户配置、每个最终 candidate 可追溯到唯一 backbone/sequence/shard。
- 保持 BoltzGen 现有 binary `BINDING/NOT_BINDING` 和 checkpoint；不引入 weighted-hotspot 网络修改。

## 1. 扩展 owner schema 与模板执行计划
- 在 [`binder_harness/config.py`](/aceph/daweihuang/program/binder-harness/binder_harness/config.py) 将模板字段从 `boltzgen_design_native` 收敛到 typed `harness_template_policy`：`enabled`、gate、PAE、min quality、Top-K、`round_conditioned_fraction`、proximity、max fixed fraction、min designable residues、library size、alignment thresholds、failure policy、utility/cooldown/blacklist thresholds。
- 明确 `round_conditioned_fraction` 是整轮全局比例，不再由 template arm 内部权重二次归一化；resolver 在所有 arms 确定后分配 conditioned/template-free/other-arm 预算，确保总和严格等于用户 `num_designs`。
- 在 [`execution_governance.py`](/aceph/daweihuang/program/binder-harness/binder_harness/execution_governance.py) 增加 immutable `TemplateApplicationPlan`：template ID/source digest、source/current target identity、alignment、residue map、length transform、budget、applicability、consumer receipts 和 digest。

## 2. Target-patch alignment 与 coherent frame
- 新增 `binder_harness/analysis/template_alignment.py`：
  1. 从 staged source complex 提取 binder motif 和真实 target contact patch；
  2. 在 current target 中按 chain、author residue ID、insertion code 做映射，必要时允许受控 sequence fallback；
  3. 用 target patch backbone/CA 原子做 Kabsch alignment；
  4. 将 source binder motif/必要 target patch变换到 current target frame；
  5. 输出 mapping coverage、target-patch RMSD、rotation/translation matrix、alignment status/digest。
- 在 [`fragment_template_mining_agent.py`](/aceph/daweihuang/program/binder-harness/binder_harness/agents/fragment_template_mining_agent.py) 把缺 PAE、patch mapping 不足、patch RMSD 超限、chain/residue 不一致设为 `not_evaluable`/reject；生产默认不再在缺 PAE 时退化为“全部 source 可用”。
- 在 [`boltzgen_adapter.py`](/aceph/daweihuang/program/binder-harness/binder_harness/models/boltzgen_adapter.py) 渲染同一 frame 中的 template motif 与 target patch，并通过 structure groups 明确表达保持 motif-target 相对 pose；pre-submit 校验 alignment digest 与 packaged files 一致。

## 3. 长度变换与 source→effective residue mapping
- 新增 `binder_harness/templates/length_mapping.py`，为每个 template-length pair 生成 typed transform：
  - 增长：在 N 端、C 端或非 motif/non-contact linker 中选择最安全 insertion；
  - 缩短：第一版仅允许 terminal non-motif、non-contact、非保护带裁剪，不允许未经验证的内部 deletion；
  - 无安全方案：拒绝 pair，不静默回退 source length。
- 对每个阶段保存 `source_to_effective_residue_map`，包含 chain、residue number、insertion code；所有 design spec、inverse-fold mask、motif retention 和 sequence identity 都使用映射后的 IDs。
- 校验 chain continuity、motif/contact 不被裁剪、fixed fraction、designable count、post-transform index 和 spec/mask parity。

## 4. BoltzGen 四阶段 candidate lineage
- 修改 vendored BoltzGen writer/CLI（`models/boltzgen/src/boltzgen/task/predict/`、`task/filter/`、`cli/boltzgen.py`）为每个产物生成稳定的 `global_candidate_id` 和阶段 manifest：
  - round/job/template/branch
  - host/GPU/shard
  - backbone ID
  - inverse-fold sequence ID
  - parent candidate ID
  - stage
  - structure/sequence/metrics path
  - source template/residue-map digest
- 阶段固定为 `initial_design`、`inverse_folded`（若仅序列则记录 sequence-only/not-structural）、`before_refolding`、`final_refold`；一对多 inverse-fold 展开必须通过 parent ID 显式表示。
- 扩展 [`ResultIngestionAgent`](/aceph/daweihuang/program/binder-harness/binder_harness/agents/result_ingestion_agent.py) 优先读取 manifest，禁止生产 attribution 依赖文件排序或模糊文件名匹配；历史产物只能标记 lineage unavailable。

## 5. 四阶段 motif 与 pose attribution
- 新增 `binder_harness/analysis/motif_attribution.py`，对同一 candidate lineage 计算：
  - source→initial：structure conditioning 是否保持 motif；
  - initial→inverse-folded/before-refolding：sequence design/folding 是否破坏 motif；
  - before-refolding→final：final refold 是否破坏 motif；
  - source→final：端到端 retention。
- 每组比较同时输出：
  - `motif_self_aligned_rmsd`（局部形状）；
  - `target_patch_aligned_motif_rmsd`（相对 target pose）；
  - target-patch alignment RMSD/coverage；
  - sequence identity；
  - primary hotspot/contact retention；
  - clash 与核心质量指标。
- 替换 [`orchestrator.py`](/aceph/daweihuang/program/binder-harness/binder_harness/orchestration/orchestrator.py) 当前仅 source→final 的循环，发布版本化 `template_motif_attribution.json`；阶段缺失必须写 `not_available`，不得伪算。

## 6. Matched control 与模板 outcome utility
- Template branches 和 template-free control 必须共享同轮 target、length、sampler、filters 和可比 backbone budget；只有 template intervention 不同。
- 新增 `binder_harness/templates/outcome_ledger.py`，按 `target_identity_digest + template_id` 保存：使用次数、package/runtime failure、各阶段 retention、primary coverage、clash、最终核心质量、matched-control uplift、最近使用轮。
- Utility 只基于 matched-control 增量和置信度/时间衰减；实现：
  - hard blacklist：解析失败、mapping/digest/chain 不一致；
  - soft cooldown：多次无增益或 retention 回归；
  - target-specific utility；
  - 一次随机失败不得永久 blacklist。
- Template Top-K 排序从静态 PAE/quality 扩展为 eligibility → target compatibility → utility/uncertainty → source/patch/cluster diversity。

## 7. Fail-closed、lineage 与 parity
- Template package、alignment、mapping、length transform 或 stage lineage 失败时，不提交伪 template job；由 resolver 重新物化并把预算分配给显式 control/其他合法 arm。
- 扩展 execution plan、run manifest、result manifest、consumption report：统一引用 template source/alignment/residue-map/plan/spec/mask/shard digests。
- Pre-submit 和 post-ingest 分别断言 requested/applied 状态、spec/mask parity、candidate lineage 完整性和 attribution 输入一致性。

## 8. 测试与回放
- 扩展 `scripts/test_length_and_pae_gate.py`：缺 PAE fail-closed、target patch identity/rigid-transform/mismatch、insertion code、N/C insertion、terminal cropping、无安全 pair 拒绝。
- 扩展 `scripts/test_boltzgen_taiji_agents.py`：manifest global ID、一对多 inverse fold、multi-host/shard 唯一性、template alignment/residue-map/spec/mask parity、package failure 预算重分配。
- 新增/扩展 template attribution 测试：已知 motif deformation、纯 pose drift、sequence mutation、contact loss、阶段缺失，验证 self-aligned 与 target-patch-aligned RMSD 能区分。
- 扩展 strategy tests：整轮 `round_conditioned_fraction` 在多 arm/Top-K 下保持准确，template/control 除 intervention 外完全匹配。
- 扩展 resume/replay：alignment/residue-map/template source digest 变化时拒绝 exact replay；历史 v22 只标记 lineage unavailable。
- 使用 SC2RBD 的历史结构做离线 replay，并执行一个小预算 production-path dry-run/smoke，验证从模板 eligibility 到四阶段 attribution 和 utility ledger 的完整 artifact 链。

## 实施顺序
1. P0：owner schema、全局预算语义、target-patch alignment、residue mapping、缺 PAE fail-closed。
2. P1：BoltzGen stage manifest、ResultIngestion lineage、四阶段 attribution、pose RMSD。
3. P1：matched control、outcome ledger、utility/cooldown/blacklist。
4. P2：更高级 linker insertion/internal cropping；只有充分结构验证后启用。