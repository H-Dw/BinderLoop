# BinderLoop 测试脚本与使用方法

当前服务器没有 GPU，因此这里提供 CPU-only 的功能推测/结构测试，主要验证：配置读取、BoltzGen/ODesign 适配器配置生成、dry-run 命令生成、mock scoring 与 failure tags。

## 1. 适配器与 dry-run 命令测试

```bash
cd /projects/design_harness/BinderLoop
python3 scripts/test_adapters.py
```

预期输出：

```text
OK: generated 12 dry-run commands. See outputs/example_run/commands.json
```

检查内容：

- `outputs/example_run/commands.json`：BoltzGen/ODesign 的待执行命令；
- `outputs/example_run/r0/*/boltzgen/boltzgen_design_spec.yaml`：BoltzGen design spec；
- `outputs/example_run/r0/*/odesign/odesign_input.json`：ODesign input JSON。

## 2. 主 pipeline dry-run

```bash
cd /projects/design_harness/BinderLoop
python3 scripts/run_strategy_al.py --config configs/example_binder_task.yaml --dry-run
```

用途：根据配置生成所有首轮策略任务，不真正调用 GPU 模型。

## 3. mock scoring 与 failure taxonomy 测试

```bash
cd /projects/design_harness/BinderLoop
python3 scripts/test_mock_scoring.py
```

预期输出：

```text
OK: wrote outputs/mock_test/scores.csv and outputs/mock_test/failure_tags.jsonl
```

检查内容：

- `outputs/mock_test/scores.csv`：模拟候选多目标分数；
- `outputs/mock_test/failure_tags.jsonl`：根据阈值生成的失败标签。

## 4. GPU 环境上的真实运行建议

1. 先保持 `dry_run: true` 检查 spec 和命令。
2. 将 `configs/example_binder_task.yaml` 中 target 替换为真实 PDB/mmCIF 路径和 hotspots。
3. 在 GPU 环境安装 BoltzGen/ODesign 依赖和 checkpoint。
4. 将 `runtime.dry_run` 改为 `false` 或自行复制 `commands.json` 中命令执行。
5. 首轮建议小预算：BoltzGen `num_designs=50~200`，ODesign `N_sample=2~5`。
6. 得到真实输出后，扩展 `analysis/parsers.py` 中真实指标映射，再运行后续主动学习策略。

## 5. 已知限制

- 当前没有 GPU，未执行真实生成。
- ODesign/BoltzGen 的不同版本输出字段可能变化，真实 parser 需要在 GPU 跑完后根据实际 CSV/log/CIF 补齐。
- 当前主动学习是 MVP：支持首轮生成、评分接口、failure tags 和规则式下一轮提案骨架；Bayesian surrogate/多臂 bandit 可作为下一阶段增强。

## 6. 自迭代 Skill 回归测试

```bash
python scripts/test_self_improvement_skills.py
python scripts/test_strategy_improvements.py
python scripts/test_prompt_byte_budget.py
python scripts/replay_self_improvement_skill.py --out outputs/<completed-run>
```

覆盖唯一文件创建、已有 Skill copy-on-write、resume、固定模块更新、target
去特异化、语义关系、晋升/退役、Prompt 优先级、参数族冲突与安全 fallback。

CPU-only 集成 smoke test：

```bash
python scripts/run_closed_loop_orchestrator.py \
  --config configs/example_binder_task.yaml \
  --out /tmp/binder-self-improvement-smoke \
  --max-rounds 1 \
  --enable-self-improvement-skill
```
