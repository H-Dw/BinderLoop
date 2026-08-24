# BoltzGen 完整路径测试报告

## 1. 测试目标

本次测试覆盖 BinderLoop 的最小完整闭环：

```text
需求/模块检查
  -> 设计参数思考
  -> BoltzGen 完整设计脚本生成
  -> Taiji simple config JSON 生成
  -> Taiji 提交尝试
  -> 运行监测
  -> 结果收集
  -> 评估成功/失败设计因素
  -> 下一轮参数迭代建议
```

测试输出目录：

```text
outputs/boltzgen_complete_path_test/
```

主报告：

```text
outputs/boltzgen_complete_path_test/path_test_report.json
```

## 2. 执行前模块检查

对照：

- `docs/research_plan.md`
- `docs/boltzgen_taiji_agents.md`
- `../Binder-Harness.pdf`

检查结果：初始 Harness 已有参数、spec、taiji、monitor Agent，但缺少独立结果收集、评估和下一轮策略更新 Agent。已补齐：

| Agent | 路径 | 状态 |
|---|---|---|
| `DesignParameterAgent` | `binderloop/agents/design_parameter_agent.py` | 已存在 |
| `DesignSpecAgent` | `binderloop/agents/design_spec_agent.py` | 已存在 |
| `TaijiExecutionAgent` | `binderloop/agents/taiji_execution_agent.py` | 已存在 |
| `RunMonitorAgent` | `binderloop/agents/run_monitor_agent.py` | 已存在 |
| `ResultIngestionAgent` | `binderloop/agents/result_ingestion_agent.py` | 本次新增 |
| `EvaluationAgent` | `binderloop/agents/evaluation_agent.py` | 本次新增 |
| `ActiveLearningPolicyAgent` | `binderloop/agents/active_learning_policy_agent.py` | 本次新增 |

模块覆盖检查文件：

```text
outputs/boltzgen_complete_path_test/00_module_coverage.json
```

检查结论：`status=pass`，无缺失 Agent。

## 3. 设计参数思考

参数计划文件：

```text
outputs/boltzgen_complete_path_test/01_design_parameter_plan.yaml
```

本轮采用小规模测试参数：

- Target：`examples/bg_example/IL-17A.cif`
- Binder 长度：50 aa
- Hotspot：`A:67`, `A:89`, `B:49`
- Protocol：`protein-anything`
- `num_designs=4`
- `budget=2`
- `diffusion_batch_size=1`
- `inverse_fold_num_sequences=1`
- `run_filtering=true`
- 保留 unfiltered / intermediate 输出用于失败案例分析。

参数选择理由：

1. 默认 protein binder，使用 `protein-anything`。
2. 小规模路径测试优先验证流程，所以 `num_designs` 设为 4。
3. `diffusion_batch_size=1`，避免 batch 内同长度/随机性耦合，方便观察单样本失败。
4. 有 hotspot，启用 `filter_bindingsite=true`。
5. 开启 filtering，但保留 filtering 前分析结果，避免只看到 top hits。

## 4. BoltzGen 完整设计脚本

生成的完整运行脚本：

```text
outputs/boltzgen_complete_path_test/round0_len50_seed0/run_boltzgen_full.sh
```

生成的 BoltzGen spec：

```text
outputs/boltzgen_complete_path_test/round0_len50_seed0/boltzgen_output/boltzgen_design_spec.yaml
```

关键命令包含完整 pipeline：

```bash
boltzgen run ... \
  --protocol protein-anything \
  --num_designs 4 \
  --budget 2 \
  --diffusion_batch_size 1 \
  --inverse_fold_num_sequences 1 \
  --alpha 0.001 \
  --refolding_rmsd_threshold 2.0 \
  --additional_filters 'design_ptm>0.5' \
  --additional_filters 'iptm>0.4' \
  --config filtering filter_bindingsite=true \
  --steps design inverse_folding folding design_folding analysis filtering
```

重点：已修正之前只运行 `design` / `design inverse_folding` 的问题，当前脚本包含结构生成后的 folding、analysis 和 filtering。

## 5. Taiji 平台 JSON 撰写与提交

生成的 Taiji simple config：

```text
outputs/boltzgen_complete_path_test/02_taiji_simple_config.json
```

提交指令：

```bash
taiji_client start -scfg outputs/boltzgen_complete_path_test/02_taiji_simple_config.json
```

本次真实调用了 `taiji_client start -scfg ...`，但未复制示例 JSON 中的 Token / mount secret。

提交结果：

```text
returncode=0
stdout=[error][user token not set. Please run: taiji_client config ...]
taiji_job_id=null
```

结论：Taiji 客户端当前环境未配置 user token，因此任务未获得 instance/job id，无法进入真实 GPU 运行阶段。该失败已记录在：

```text
outputs/boltzgen_complete_path_test/path_test_report.json
```

## 6. 任务运行监测

因为 Taiji 未返回 instance/job id，`RunMonitorAgent` 记录状态为：

```text
state=not_started_or_no_instance_id
reason=Taiji submission did not return an instance/job id; inspect submission stdout/stderr.
```

如果后续配置好 Taiji token，`RunMonitorAgent` 会检查：

- `taiji_client instance_list`
- `taiji_client instance_detail`
- `taiji_client logs --tail ...`
- `steps.yaml`
- `intermediate_designs`
- `intermediate_designs_inverse_folded`
- `final_ranked_designs`
- `boltzgen_full.log`

并识别常见问题，例如 OOM、输入文件缺失、conda 环境错误、BoltzGen 配置错误等。

## 7. 结果收集与评估

由于真实 Taiji 任务未启动，本地没有真实 BoltzGen metrics。为了验证结果收集与评估路径，脚本写入了 4 条 mock BoltzGen metrics，用于测试 downstream Agent：

```text
outputs/boltzgen_complete_path_test/round0_len50_seed0/boltzgen_output/final_ranked_designs/all_designs_metrics.csv
```

评估输出：

```text
outputs/boltzgen_complete_path_test/06_evaluation_summary.json
outputs/boltzgen_complete_path_test/06_scores_preview.csv
```

评估结果：

- 总候选数：4
- 成功候选：1
- 失败候选：3

失败/成功标签统计：

| 标签 | 数量 | 含义 |
|---|---:|---|
| `pass_compute_gate` | 1 | 通过当前计算门控 |
| `hotspot_miss` | 1 | hotspot 接触不足 |
| `folding_failure` | 1 | binder 自身折叠/复折叠稳定性不足 |
| `clash` | 1 | RMSD/clash 类几何惩罚偏高 |
| `binding_pose_failure` | 1 | complex/interface 置信度不足 |

### 成功设计因素

成功样本：`success_high_interface`

关键指标：

- `interface_confidence=0.76`
- `hotspot_contact=0.82`
- `binder_plddt=0.88`
- `clash_penalty=0.12`
- `diversity=0.61`
- `sequence_designability=0.84`

成功原因：

1. interface 置信度高。
2. hotspot 接触满足度高。
3. binder pLDDT / sequence designability 较好。
4. refolding/clash penalty 低，说明结构几何稳定性较好。

### 失败设计因素

失败样本主要分为三类：

1. `failure_hotspot_miss`
   - interface 还可以，但 hotspot_contact 低。
   - 表明生成姿态可能靠近 target，但没有命中关键功能位点。

2. `failure_folding_unstable`
   - binder pLDDT 低，refolding RMSD 高。
   - 表明 backbone 或 sequence 不稳定，即使有局部接触也不适合作为候选。

3. `failure_pose_low_confidence`
   - ipTM/interface confidence 低。
   - 表明复合物结合姿态不可信，可能是 target patch 或约束不足。

## 8. 下一轮设计参数迭代思考

下一轮建议文件：

```text
outputs/boltzgen_complete_path_test/07_next_round_parameter_proposal.json
```

由于本轮 mock 评估中存在 1 个 pass candidate，策略建议是：

- 保留当前 `protein-anything` 方案。
- 保留 `filter_bindingsite=true`。
- 适度扩大采样量：`num_designs` 从 4 提高到 20。
- 继续开启 filtering，但继续保留 intermediate / analysis 输出用于失败归因。

当前自动策略给出的 rationale：

```text
At least one pass candidate found: exploit the current strategy with a modestly larger sample budget.
```

人工补充建议：

1. 如果真实运行中 hotspot_miss 比例升高：
   - 提高 hotspot conditioning。
   - 尝试 hotspot 子集采样。
   - 扩大 target patch 或将部分 hotspot 从硬约束改为软约束。

2. 如果 folding_failure 比例升高：
   - 减小 binder 长度，例如测试 40/45/50。
   - 增加 inverse_fold_num_sequences。
   - 引入 topology / secondary structure bias。

3. 如果 binding_pose_failure 比例升高：
   - 增强 interface contact 约束。
   - 切换 target conformer。
   - 增加 target ensemble 或 patch robustness 检查。

4. 如果 diversity collapse 出现：
   - 保持 `diffusion_batch_size=1`。
   - 提高 `alpha`。
   - 加 length/topology quota。

## 9. 当前阻塞与下一步

当前真实 Taiji 阻塞：

```text
user token not set
```

下一步需要在运行环境中配置 Taiji token，或由用户确认允许使用已有示例 JSON 中的 Token / mount secret 后，再执行真实 GPU 任务。

在 token 配置完成后，建议直接重跑：

```bash
cd /projects/design_harness/BinderLoop
python3 scripts/run_boltzgen_complete_path_test.py --submit
```

如果需要使用示例 JSON 中的 token/mount 信息，则必须显式确认后运行：

```bash
python3 scripts/run_boltzgen_complete_path_test.py --submit --allow-template-secrets
```

该选项会复制示例 Taiji JSON 中的敏感字段到本次 generated simple config，因此需要明确授权。
