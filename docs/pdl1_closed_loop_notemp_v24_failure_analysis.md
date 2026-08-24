# PD-L1 closed-loop no-template v24 失败分析

## 1. 执行摘要

本次 `pdl1_closed_loop_llm_np_si_notemp_160s_8r_v24` 运行并未进入 Taiji 提交阶段。8 个逻辑轮次（`round_00`—`round_07`）中，每轮 2 个 arm 均在提交前被本地执行器标记为失败，因此 16 个 job 全部没有产生实际运行结果，所有轮次均为 `candidate_count=0`、`realized_job_count=0`。

直接失败信息是 `pre-submit config validation failed`，但这并不意味着 validator 判定配置无效。相反，所有 16 份 `pre_submit_config_validation.json` 的顶层 `is_valid` 均为 `true`；确定性预校验也为 `true`。validator 发现的内容都是已经解决的 warning：删除不可执行的编排元数据/不支持字段，以及把 `filter_biased` 的布尔值规范化为 BoltzGen CLI 要求的小写字符串。

根本原因位于 `scripts/run_closed_loop_orchestrator.py` 的 Taiji executor：代码把完整、不可变的 `job.params` 与 validator 输出的“仅可执行、已清理配置”直接比较。validator 按设计删除 `arm_id`、`job_identity`、`immutable_branch_plan`、`round_budget_resolution`、fragment-template policy 等编排元数据后，这些删除被 `_changed_config_values` 记为 tombstones；调用方随后将“存在任何 diff”错误地等同于“校验失败”，即使 `validation.is_valid == true` 仍进入失败分支。这违反了 `ConfigValidationAgent` 明确规定的契约：确定性 sanitizer 是 full-job 可提交性的权威，删除 unsupported keys 是已解决的 warning，LLM advisory 不得否决提交。

结论：这是提交前配置域边界和 diff 判定错误，不是 Taiji、GPU、LLM、timeout、PD-L1 数据、无模板策略或 BoltzGen 运行时故障。由于没有执行到 `taiji_agent.submit(...)`，本次证据不能用于评价 Taiji 环境或 GPU 是否健康。

## 2. 分析范围与证据基线

本报告基于以下实际文件：

- 顶层日志：`pdl1_closed_loop_notemp_160s_8r_v24.log`
- 完整运行产物：`outputs/pdl1_closed_loop_llm_np_si_notemp_160s_8r_v24/`
- 执行入口：`scripts/run_closed_loop_orchestrator.py`
- 配置校验器：`binderloop/agents/config_validation_agent.py`
- 闭环编排器：`binderloop/orchestration/orchestrator.py`
- 任务配置：`configs/pdl1_structured_task_notemp_iptm035.yaml`

行号均指本次分析时工作区中的文件版本。大型 JSON 产物以代表性 `round_00`/`round_07` 文件给出精确位置，并通过同类文件全集核验重复性。

## 3. 现象与影响

### 3.1 用户可见现象

顶层日志只有三类结果：

1. 四个 agent 使用 `execution_failure_noop` 的 warning，见 `pdl1_closed_loop_notemp_160s_8r_v24.log:2-9`。
2. LLM preflight 成功，endpoint 为 `deepseek`、model 为 `deepseek-v4-pro`，见同文件 `:10`。
3. 汇总文件生成，但候选指标图因没有候选而跳过，见同文件 `:11-12`。

这些日志是下游保护性降级和零候选的表现，不是根因。尤其是 `execution_failure_noop` 表示编排器已识别到执行失败，主动跳过质量学习，避免把基础设施/执行故障误当成设计质量信号。

### 3.2 运行影响

- 轮次范围：`round_00` 至 `round_07`，共 8 轮。
- 每轮预期 job：2。
- 每轮失败 job：2。
- 每轮 realized job：0。
- 每轮 quality candidate：0。
- 总计：16 次逻辑 job 均在 pre-submit 阶段失败，Taiji 提交次数为 0。
- 未产生可用于 binder 质量评估、主动学习、回滚比较或参数优化的真实候选。
- 8 轮预算被重复的确定性软件错误消耗，顶层日志却没有直接打印 validator 的 `is_valid`、diff 分类或 artifact 路径，增加了定位成本。

代表性证据：

- `round_00/execution_state.json:2-20`：`expected_job_count=2`、两个 job 均在 `failed_job_ids`、`quality_candidate_count=0`、`realized_job_count=0`。
- `round_07/execution_state.json:3-17`：最后一轮仍是 `expected_job_count=2`、`quality_candidate_count=0`、`realized_job_count=0`。
- 目录中 8 份 `round_00`—`round_07/execution_state.json` 均包含 `quality_candidate_count: 0`；16 份 pre-submit validation artifact 均包含 `is_valid: true`；8 份 `execution_records.json` 均包含 `pre-submit config validation failed`。

## 4. 失败时间线

### 阶段 A：任务与运行参数加载

任务配置指定 PD-L1 目标、160 个总设计、binder 长度 50—120、Taiji backend、2 hosts × 8 GPUs、1200 秒 timeout，并关闭 harness template policy：

- 目标及设计预算：`configs/pdl1_structured_task_notemp_iptm035.yaml:3-13`
- `filter_biased: 'true'` 的配置源值：同文件 `:28-37`
- active-learning 配置：同文件 `:57-68`
- Taiji 资源与 timeout：同文件 `:69-84`
- 无模板策略：同文件 `:111-115`

配置内容解释了运行意图，但没有证据表明这些设置触发了失败。

### 阶段 B：LLM preflight 成功

入口在 `scripts/run_closed_loop_orchestrator.py:168-183` 执行 required-LLM preflight；顶层日志确认成功（`pdl1_closed_loop_notemp_160s_8r_v24.log:10`）。因此不存在“LLM 接口不可用导致任务未提交”的证据。

### 阶段 C：Taiji executor 组装远端参数并校验

Taiji executor 在 `scripts/run_closed_loop_orchestrator.py:351-358`：

1. 调用 `_params_for_remote_boltzgen(job.params)`；
2. 把 `analysis_location=taiji`、`run_analysis_on_taiji=true` 注入副本（实现见 `:611-615`）；
3. 对这个包含可执行参数和编排元数据的 full job config 调用 `validate_full_job_config`；
4. 写出 `pre_submit_config_validation.json`。

### 阶段 D：validator 成功清理配置

以首轮 baseline arm 为例：

- `.../r0/arms/00_baseline_hold_b45690779c24/jobs/r0_arm00_b45690779c24_job00_d9a4caa647/pre_submit_config_validation.json:1-6` 明确记录 `llm_used=true`、`is_valid=true`。
- 同文件 `:178-360` 的问题全部为 resolved warning；其中 `:257-352` 显示 `arm_id`、`job_identity`、`round_budget_resolution`、`immutable_branch_plan`、`execution_slot` 等元数据被删除，`:355-359` 显示 `filter_biased` 从 bool 规范化为字符串 `'true'`。
- 同文件 `:634-638` 显示确定性预校验 `is_valid=true`。
- 同文件 `:1176-1229` 明确列出 deterministic sanitizer 的 `removed_keys`/`tombstones`。
- 同文件 `:1233` 标记 `llm_advisory_only=true`。

这一步的语义是“配置已清理且可提交”，不是“配置失败”。

### 阶段 E：调用方错误生成 correction proposal

随后 `scripts/run_closed_loop_orchestrator.py:359-363`：

1. `_job_params_from_remote_correction(job.params, validation.corrected_config)`；
2. `_config_correction_proposal(job.params, corrected_job_params, ...)`；
3. `if correction_proposal or not validation.is_valid:`。

问题在于 `_job_params_from_remote_correction` 的实现（`:505-511`）基本直接返回 `corrected_config`，仅移除原配置中不存在的两个远端字段。它没有把 sanitizer 输出重新合并回完整 job，也没有区分 executable payload 与 orchestration metadata。

`_config_correction_proposal`（`:514-525`）调用 `_changed_config_values`；后者（`:584-594`）把 `original` 中存在、`corrected` 中缺失的所有 key 记为 `__tombstones__`。于是 validator 按契约清理元数据，必然产生 non-empty proposal。

最终，虽然 `validation.is_valid` 为 true，`correction_proposal` 仍为 truthy，代码在 `:363-377` 返回：

- `status=failed`
- `error=pre-submit config validation failed`
- `retryable=false`

并且不会到达 run spec 创建（`:378`）和 Taiji submit（`:415-422`）。

### 阶段 F：失败被编排器归类并跨轮复制

`binderloop/orchestration/orchestrator.py` 的行为如下：

- `_detect_round_execution_failure` 在 `:5285-5333` 将“零候选 + 已识别配置/执行错误”归为 execution failure。
- execution failure 被排除出 reward/rollback 学习，见 `:938-941`，避免污染设计质量学习。
- 对 execution failure，`next_jobs_module` 清空参数更新并调用 `_retry_jobs_after_execution_failure`，见 `:1581-1593`。
- `_retry_jobs_after_execution_failure` 在 `:2691-2733` 读取 `requires_refinalization` proposal，并用 `corrected_params` 替换下一轮 params，再重新 finalize identity。
- 每轮完成后把 `next_jobs` 作为下一轮 `current_jobs`，见 `:1884-1889`。

这里缺少对“跨轮重复的确定性失败类别 + 等价 config diff”的熔断。现有 `failure_fingerprint` 会受到 job identity、轮次或具体 payload 的影响：例如 `round_00/execution_records.json:1241-1242` 和 `:3036-3037` 两个 arm 的 fingerprint 不同，`round_07/execution_records.json:1007-1008`、`:2240-2241` 也不同。因此不能仅按现有 hash 字符串去重；需要生成排除轮次/job identity 的语义 fingerprint。

## 5. 证据链

从输入到失败的完整链条如下：

1. 合法任务配置进入 closed loop；`filter_biased` 在 YAML 中已经是字符串（`configs/pdl1_structured_task_notemp_iptm035.yaml:28-34`），但某个中间对象把它变成了 bool。
2. Taiji executor 复制完整 `job.params` 并添加远端字段（`scripts/run_closed_loop_orchestrator.py:351-356,611-615`）。
3. validator 对 full job config 进行白名单清理，并返回 `is_valid=true`。
4. validator artifact 表明所有 unsupported-key 删除均是 `resolved=true`，类型规范化也已解决。
5. 调用方把 sanitized executable config 当成完整 job params（`scripts/run_closed_loop_orchestrator.py:505-511`）。
6. 调用方比较不同配置域：完整 immutable params 对 sanitized executable config（`:359-362`）。
7. 元数据删除被记成 tombstones（`:584-594`）。
8. 任意 proposal 即触发失败（`:363-377`），与 `is_valid=true` 无关。
9. `round_00/execution_records.json:1583-1786` 记录第一 arm 的 `retry_correction_proposal`、大量 tombstones、`requires_refinalization=true`、`retryable=false`；第二 arm 在 `:3381-3586` 重复相同行为。
10. 因失败发生在 `create_boltzgen_run_spec` 和 `taiji_agent.submit` 之前，没有 Taiji job id、远端运行日志或 GPU 产出。
11. 编排器把该错误当 execution/infrastructure failure，保护性跳过质量分析并复制到下一轮，最终 8 轮全空。

## 6. 原因分层与信心

### 6.1 直接原因：pre-submit 分支把有效配置标成失败

**结论：** `if correction_proposal or not validation.is_valid` 将“存在任何清理差异”视为失败。

**信心：高（>99%）**

证据是 execution record 同时存在 `config_validation.is_valid=true`、non-empty tombstones 和 `error=pre-submit config validation failed`；对应分支可在 `scripts/run_closed_loop_orchestrator.py:359-377` 直接复现逻辑关系。

### 6.2 根本原因：混淆 executable config 与 orchestration metadata 的配置域

**结论：** caller 用完整不可变 `job.params` 与 validator 的 sanitized executable payload 比较，比较对象语义不等价；`_job_params_from_remote_correction` 又丢失完整 params 的 identity/编排元数据。

**信心：高（>99%）**

`pre_submit_config_validation.json` 的 tombstones 正是 `arm_id`、`job_identity`、`immutable_branch_plan`、`round_budget_resolution` 和 fragment-template policy 等非提交域字段。代码实现和 artifact 一致。

### 6.3 契约违背：调用方没有遵守 validator 的 submittability 语义

**结论：** validator 把 deterministic sanitizer 定义为 full-job submittability 的权威，而 caller 仍用 sanitized diff 额外否决提交。

**信心：高（>99%）**

- `binderloop/agents/config_validation_agent.py:94-100`：full job config 的 deterministic sanitizer 是权威，LLM review 仅 advisory，不得 veto。
- `:195-220`：full-job 模式保留 base `is_valid` 和 corrected config，LLM issues 转为 advisory。
- `:353-371`：unsupported/invalid-shape keys 被删除并记录为 `resolved=true` warning。
- `:391-402`：LLM advisory issue 强制 resolved，明确不得 veto。

### 6.4 促成因素 A：跨轮重试没有语义熔断

**结论：** execution failure 路径有保护性重试，但没有检测“相同失败类别 + 等价 executable diff”连续出现，导致耗尽本次 8 轮。

**信心：高（约 95%）**

代码明确在 execution failure 时重建下一轮 job（`orchestrator.py:1588-1593,2691-2733`），8 轮 artifact 均重复同一类错误。现有 raw hash 不稳定，因此需新增归一化语义 fingerprint，而不是复用现有 fingerprint。

### 6.5 促成因素 B：顶层可观测性不足

**结论：** 顶层日志没有打印 job id、`validation.is_valid`、blocking issue、diff 分类和 artifact 路径，导致最终用户只看到 noop warnings 和“无候选”。

**信心：高（>99%）**

顶层日志仅 13 行，关键错误只能下钻到 per-job JSON 才能看到。

### 6.6 促成因素 C：类型边界不稳定

**结论：** YAML 中 `filter_biased` 是字符串 `'true'`，但 validator 收到过 `True (bool)` 并再次规范化，说明中间层存在类型漂移。

**信心：中（约 80%）**

证据能证明发生了漂移，但仅凭当前文件不能唯一定位是哪一个中间转换函数造成。该问题已被 sanitizer 修复，不是本次阻断提交的根因。

## 7. 明确排除项

以下因素没有证据支持为本次根因：

- **Taiji 环境或 GPU 故障**：根本未执行到 `taiji_agent.submit`（`scripts/run_closed_loop_orchestrator.py:422`），不能宣称已验证或未验证成功，只能确认“本次没有提交”。
- **资源请求、GPU 型号、host 数**：虽配置为 2 hosts × 8 V100（config `:76-84`），但资源调度尚未发生。
- **timeout=1200**：等待逻辑未启动，timeout 不可能导致 pre-submit 失败。
- **LLM 接口失败**：preflight 成功；validation artifact 中 LLM 也有返回。四个 noop 是 execution failure 下的保护性降级。
- **LLM 的 `is_valid` 判定**：顶层、deterministic prefilter、LLM result 都是 true。
- **无模板模式**：被删除的是不支持的 fragment-template policy 元数据，validator 已将其作为 resolved warning；是否启用模板不改变错误的配置域比较方式。
- **PD-L1 数据、target residue 或 `iptm>0.35`**：没有进入 BoltzGen，也没有候选被过滤，数据质量和阈值尚未被运行验证。
- **设计数、binder 长度、采样参数**：未进入模型执行阶段。
- **`filter_biased` 本身**：虽发生 bool→string 规范化，但 validator 已解决且仍判定有效；真正阻断来自 caller 对 non-empty diff 的处理。

## 8. 改进方向

1. 建立明确的两域模型：
   - `orchestration_metadata`：identity、arm、branch、budget resolution、retry lineage、模板策略元数据等；
   - `executable_payload`：真正传给 DesignSpec/BoltzGen 的参数。
2. validator 只决定 executable payload 是否可提交；resolved key stripping 不得成为阻断条件。
3. 只有 executable 域的真实语义值修复才生成 correction proposal；metadata tombstones 不参与。
4. 不得把 sanitized executable config 整体回写为完整 `job.params`，否则会破坏 identity 和重试语义。
5. run spec 必须使用 validator 已验证的 executable corrected config，而不是未经校验的 `remote_params`。
6. 对跨轮重复的确定性失败建立语义 fingerprint 和熔断。
7. 顶层日志输出结构化 validation 摘要和 artifact 定位信息。
8. 在配置解析/合并边界统一 choice 类型，避免字符串在中间步骤变回 bool。

## 9. 分优先级实施策略

### P0：修复提交阻断和 payload 使用

#### P0.1 改为在同一配置域比较

建议修改 `scripts/run_closed_loop_orchestrator.py` 的 `_build_taiji_executor.execute`（当前 `:351-378`）：

- 保留 `remote_params = _params_for_remote_boltzgen(job.params)` 作为 validator 输入快照。
- 令 `validated_exec_params = validation.corrected_config`。
- diff 应比较 `remote_params` 与 `validated_exec_params`，但只检查 validator 支持的 executable 域；由 sanitizer 删除的 unsupported/internal keys归类为 `resolved_stripping`，不生成 blocking proposal。
- 更清晰的实现是先显式 partition：`metadata, executable = partition_job_params(remote_params)`，validator 仅接收或输出 executable；之后仅比较 `executable` 与 corrected executable。
- 阻断条件改为：
  - `not validation.is_valid`；或
  - executable 域出现需要 refinalization 的真实语义变化，并且当前策略明确不允许在本次提交直接采用该修复。
- 仅 resolved tombstones、advisory issue 或格式规范化不应阻断提交。

推荐逻辑语义：

```python
validated_exec_params = validation.corrected_config
blocking_diff = executable_semantic_diff(executable_input, validated_exec_params)
if not validation.is_valid or blocking_diff.requires_refinalization:
    ... fail ...
```

对于安全、确定性的规范化（例如 bool→`'true'`），更合理的是直接使用 corrected executable payload 提交，并记录 `applied_normalizations`，而不是要求跨轮 refinalization。

#### P0.2 保持完整 job identity，不回写 sanitized payload

建议修改或替换 `_job_params_from_remote_correction`（当前 `:505-511`）：

- 不要 `result = dict(corrected)` 后把它当完整 params 返回。
- 应以 `original` 为基底，只更新 executable keys 的真实修复值；metadata 原样保留。
- sanitizer 删除 unsupported executable key 时，要依据显式 ownership/partition 决定是否从 executable 子域移除，不能删除 `job_identity` 等编排字段。
- `_retry_jobs_after_execution_failure`（`orchestrator.py:2717-2724`）也不应再用 `params = dict(corrected)` 整体替换；应应用 typed patch 或 executable delta。

#### P0.3 run spec 使用已验证配置

当前 `scripts/run_closed_loop_orchestrator.py:378` 使用未经 sanitizer 最终确认的 `remote_params` 创建 run spec。修复后应传入 `validated_exec_params`，或传入“完整 metadata + 已验证 executable 域”的组合视图，确保真正提交的内容与 validation artifact 一致。

同类风险也存在 local executor：`scripts/run_closed_loop_orchestrator.py:282-306` 同样以“有任何 correction proposal”为失败条件，并在成功路径使用 `job.params`。虽然本次事故发生在 Taiji 路径，建议 P0 同步修正，避免 backend 行为分叉。

### P1：回归测试与跨轮熔断

#### P1.1 添加 pre-submit 回归测试

围绕 `_build_taiji_executor`、`_job_params_from_remote_correction`、`_config_correction_proposal` 或新 partition/diff helper 添加测试，至少覆盖：

1. **合法 full job + 内部元数据**：包含 `arm_id`、`job_identity`、`immutable_branch_plan`、`round_budget_resolution`；validator valid，Taiji submit mock 必须被调用。
2. **resolved tombstones**：unsupported fragment-template policy 被清理但不阻断；artifact 保留 warning，submit 被调用。
3. **`filter_biased` 规范化**：输入 bool `True`，最终 run spec/提交 payload 为字符串 `'true'`；不触发 refinalization。
4. **真实无效配置**：构造 unresolved error；submit 不得调用，execution record 包含 blocking issue。
5. **真实 executable 语义修复**：明确测试可安全原地应用与必须 refinalize 两类策略。
6. **metadata 保留**：成功提交前后 job identity、arm/branch/budget metadata 不丢失。
7. **payload 一致性**：传给 `create_boltzgen_run_spec` 的参数等于 validator 的 corrected executable config，而不是原始 remote params。
8. **local/Taiji 一致性**：相同 valid full job 在两个 backend 不应因 resolved stripping 失败。

#### P1.2 添加 deterministic failure fingerprint 熔断

建议在 `BinderDesignOrchestrator._detect_round_execution_failure`、`_retry_jobs_after_execution_failure` 或 `next_jobs_module` 周边实现：

- fingerprint 输入应包括：failure class、validator activation、`validation.is_valid`、未解决 blocking issues、归一化 executable diff、resolved tombstone 类别。
- 排除：round id、job id、output path、identity digest、arm-specific bookkeeping 等易变元数据。
- 同一语义 fingerprint 连续出现 2 轮（阈值可配置）时停止自动重试。
- 输出结构化根因，例如：`pre_submit_contract_mismatch: valid sanitizer result rejected due only to metadata tombstones`。
- 熔断时保留 resume 所需状态，但不再生成等价 next jobs。

### P2：可观测性和类型治理

#### P2.1 顶层结构化日志

在 pre-submit 判定处输出一行稳定格式日志，至少包含：

- `round_id`、`job_id`、backend、attempt；
- `validation.is_valid`；
- unresolved blocking issue 数量及参数名；
- diff 分类：`semantic_value_change`、`resolved_stripping`、`metadata_only`、`normalization_only`；
- 是否调用 submit；
- `pre_submit_config_validation.json` 与 `execution_record.json` 路径。

同时在 orchestrator round summary 中汇总每类失败数量，而不是只打印四个 noop warning。

#### P2.2 统一类型边界

- 在 YAML load 后或构造 executable payload 时一次性规范化 choice 类型。
- 对 `filter_biased` 定义单一 canonical 类型（建议 executable 边界为小写字符串，内部模型若使用 bool 则必须显式转换）。
- 增加 round-trip 测试：YAML → config model → job params → remote params → validator → run spec，确保 `'true'` 不意外变成 `True`，或即使转换也只产生 non-blocking normalization。

## 10. 测试与验收标准

### 单元测试验收

- 合法 full job 含内部 metadata 时，`validation.is_valid == true` 且仅有 resolved stripping，executor 返回 `submitted`/`dry_run`，而不是 `failed`。
- Taiji agent mock 的 `submit` 被调用恰好一次。
- unresolved validator error 时，submit 调用次数为 0。
- `filter_biased=True` 的最终 executable payload 为 `'true'`。
- corrected executable payload 中不存在 unsupported keys，但原始 job 的 identity metadata 完整保留。
- metadata-only tombstones 不产生 `requires_refinalization` proposal。
- 真正 executable 值变化可被明确分类并按策略应用或阻断。

### 集成测试验收

使用最小 dry-run/fake-Taiji 集成测试执行 1 轮、2 arms：

- 两个 arm 均通过 pre-submit；
- 两次 submit/create-spec mock 被调用；
- 每个 validation artifact 顶层 `is_valid=true`；
- execution record 不包含 `pre-submit config validation failed`；
- run spec 参数与 `corrected_config` 一致；
- 顶层日志包含 job id、validation 状态、diff 分类和 artifact 路径。

### 熔断验收

注入同一确定性 pre-submit 故障：

- 第一次记录语义 fingerprint；
- 达到阈值后提前终止，不继续耗尽 `max_rounds`；
- summary 给出结构化根因、首次/末次发生轮次及关联 artifacts；
- 不把该失败计入设计质量 reward。

### 真实运行前验收

在重新提交 PD-L1 大任务前，先运行一个低成本 smoke test。验收点仅是“成功越过 pre-submit 并实际得到 Taiji submission id”；只有到这一步之后，才有资格继续验证 Taiji 环境、GPU 调度和 BoltzGen runtime。不要把本次零提交事故当作这些环境已经验证的证据。

## 11. 临时绕过方案及风险

### 推荐临时方案：最小 caller hotfix 后新建输出目录重跑

在正式重构前，可做一个最小、可审计的临时修复：

1. Taiji executor 的失败条件暂改为仅 `if not validation.is_valid`；
2. `create_boltzgen_run_spec` 使用 `validation.corrected_config`；
3. 不调用 `_job_params_from_remote_correction` 把 sanitized config 回写为完整 params；
4. 保留 validation artifact 和 normalization 日志；
5. 用新输出目录运行 1 轮 smoke test，确认出现 submission id 后再扩大规模。

**风险：** 如果 validator 将某个需要 identity/refinalization 的真实 executable 修复也标为 valid，这个临时方案会直接采用修复值提交。当前 validator 契约倾向允许 deterministic repair，但临时方案仍应通过 diff 分类和人工检查降低风险。

### 次选临时方案：提交前显式拆分 metadata/executable

在 executor 外增加 wrapper，按参数 contract 提取 executable payload，validator 和 run spec 都只使用该 payload；完整 `job.params` 仅用于编排和 identity。

**风险：** 手工维护白名单可能与 `ALL_EXECUTABLE_CONFIG_KEYS` 漂移；应复用 validator/contract 的权威集合，而不是复制一份新列表。

### 不推荐方案

- 删除 `arm_id`、`job_identity`、`immutable_branch_plan` 等元数据来“消除 tombstones”：会破坏 identity、重试、预算和审计语义。
- 关闭 LLM：确定性 sanitizer 仍会删除 unsupported keys，根因仍存在。
- 调大 timeout、修改 GPU/host、改 PD-L1 阈值或启用模板：都无法绕过发生在 Taiji submit 之前的 caller 逻辑错误。
- 直接复用当前输出目录强行 resume：可能继续沿用已经生成的 correction proposal 和 retry 状态。建议修复后使用新输出目录，或在充分理解 resume artifact 契约后做专门迁移。

## 12. 最终结论

本次 v24 失败是一个确定性的 pre-submit caller bug：validator 正确地把 full job 清理成可提交 executable config，并明确返回 valid；executor 却把“sanitization 产生差异”误当作“校验失败”。由于比较了不同语义域，编排元数据必然成为 tombstones，导致每个 arm 在 Taiji submit 之前失败。编排器的 execution-failure 保护机制随后避免了错误学习，但缺少跨轮语义熔断，因此同类失败持续 8 轮。

修复重点不是调整实验配置，而是：统一比较域、保留 identity metadata、使用 validated executable payload 提交、仅让 unresolved/真实 executable 变化阻断，并补齐回归测试、语义熔断和顶层可观测性。
