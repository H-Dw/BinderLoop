# BoltzGen + Taiji Agent 配置说明

本文档说明 `DesignParameterAgent`、`DesignSpecAgent`、`TaijiExecutionAgent`、`RunMonitorAgent` 针对 `examples/bg_example` 的 BoltzGen 运行和 Taiji 提交能力。

## 1. DesignParameterAgent

路径：`binderloop/agents/design_parameter_agent.py`

职责：把简洁 Harness YAML 扩展为 BoltzGen 可执行参数计划。

已支持的 BoltzGen 参数选择包括：

- `protocol`：根据 binder 类型、长度、小分子上下文选择 `protein-anything`、`peptide-anything`、`nanobody-anything`、`protein-small_molecule`。
- `num_designs` / `budget`：设置生成数量和 filtering 最终预算。
- `diffusion_batch_size`：根据生成数量、长度范围和多样性需求自动选择。
- `step_scale` / `noise_scale`：支持探索/多样性相关调节。
- `inverse_fold_num_sequences` / `inverse_fold_avoid`：控制 inverse folding 序列数量和禁用氨基酸。
- `filter_biased` / `alpha` / `refolding_rmsd_threshold` / `additional_filters`：控制 BoltzGen filtering。
- `config_overrides`：支持向 BoltzGen 内部 step 注入配置，例如 `filtering filter_bindingsite=true`。
- `run_filtering`：可关闭 filtering，只执行到 `analysis`，用于保留更多失败样本。
- `keep_unfiltered_for_failure_analysis`：默认开启，用于提示后续 agent 从 pre-filter / analysis 文件中提取失败原因。

注意：为了捕获失败案例，不建议只使用 `final_ranked_designs`。即使开启 filtering，也应由后处理 Agent 读取 `intermediate_designs*` 与 analysis CSV。

## 2. DesignSpecAgent

路径：`binderloop/agents/design_spec_agent.py`

职责：将 `DesignJob + BoltzGenParameterPlan` 转换为完整 BoltzGen 运行规格。

关键优化：

- 不再只运行示例脚本中的 `design` 或 `design inverse_folding`。
- 默认生成完整流程：

```text
design -> inverse_folding -> folding -> design_folding -> analysis -> filtering
```

即命令中显式包含：

```bash
--steps design inverse_folding folding design_folding analysis filtering
```

如果 `run_filtering=false`，则流程变为：

```text
design -> inverse_folding -> folding -> design_folding -> analysis
```

该 Agent 会生成：

- `boltzgen_design_spec.yaml`
- `boltzgen_parameter_plan.yaml`
- `boltzgen_run_manifest.json`
- `run_boltzgen_full.sh`

## 3. TaijiExecutionAgent

路径：`binderloop/agents/taiji_execution_agent.py`

职责：生成 Taiji simple config JSON，并通过如下命令提交：

```bash
taiji_client start -scfg /path/to/json_file.json
```

可参考模板：

```text
examples/bg_example/boltzgen_test_v100.json
examples/bg_example/benchmark_v100.json
```

注意：示例 JSON 中包含 `Token`、mount secret 等敏感字段。Agent 写审计 manifest 时会对敏感 key 做脱敏；但生成的实际 simple config 仍需保留 Taiji 运行所需字段。

## 4. RunMonitorAgent

路径：`binderloop/agents/run_monitor_agent.py`

职责：单次检查 Taiji 任务状态、日志和 BoltzGen 期望产物。

它不会内部 sleep/poll 循环；重复调度应由上层任务系统或 cron 负责。

检查内容：

- `taiji_client instance_list` 或 `taiji_client instance_detail`
- `taiji_client logs`
- `steps.yaml`
- `intermediate_designs`
- `intermediate_designs_inverse_folded`
- `final_ranked_designs`
- `boltzgen_full.log`

并给出失败提示，例如：

- `cuda_out_of_memory`
- `missing_boltzgen_cli`
- `missing_input_file`
- `conda_env_error`
- `taiji_resource_or_queue_issue`
- `boltzgen_config_error`
- `missing_expected_outputs:*`

## 5. Smoke test

```bash
cd /projects/design_harness/BinderLoop
python3 scripts/test_boltzgen_taiji_agents.py
```

该脚本只做 dry-run，不会真实提交任务。

## 6. Filtering 与失败案例捕获建议

BoltzGen filtering 很有价值，因为它会产生统一排序和最终候选集合。但 Harness 的研究目标不是只拿 top hits，而是学习失败模式。因此建议：

1. 默认开启 filtering，用于得到标准最终候选。
2. 同时保留 `analysis` 和 intermediate 输出。
3. 后续由 `ResultIngestionAgent` / `FailureTaxonomyAgent` 读取 filtering 前后的所有样本。
4. 对被 filtering 拦截的样本打标签，例如：
   - refolding RMSD 失败
   - binding-site contact 失败
   - pTM/ipTM 低
   - composition bias
   - cysteine 或 motif liability
   - diversity collapse
5. 如果某轮目标是系统性收集失败案例，可设置 `run_filtering=false`，只运行到 analysis。
