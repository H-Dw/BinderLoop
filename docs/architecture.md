# BinderLoop 初步项目架构

## 1. 设计原则

- **用户提供 target + hotspots**：不做上游靶点挖掘。
- **只优化计算设计成功率**：不纳入湿实验验证。
- **底层模型可替换**：BoltzGen 与 ODesign 均通过 adapter 接入。
- **策略级主动学习**：优化模型选择、约束、长度、采样预算和过滤器。
- **dry-run first**：无 GPU 环境下必须能生成配置、命令和测试 mock parser。

## 2. 模块结构

```text
binderloop/
  config.py                    # YAML 配置 dataclass
  pipeline.py                  # 主 pipeline
  models/
    base.py                    # DesignJob / ModelAdapter
    boltzgen_adapter.py         # 生成 BoltzGen spec 与命令
    odesign_adapter.py          # 生成 ODesign input JSON 与 Hydra 命令
  analysis/
    scoring.py                 # weighted score / rank / csv
    parsers.py                 # 输出解析与 mock metrics
    failure_taxonomy.py         # failure tag 规则
  active_learning/
    strategy.py                # 初始 jobs 与下一轮策略建议
  orchestration/
    runner.py                  # dry-run/subprocess 执行与命令记录
scripts/
  run_strategy_al.py            # 运行 pipeline
  test_adapters.py              # 适配器 dry-run 测试
  test_mock_scoring.py          # mock parser/scorer 测试
configs/
  example_binder_task.yaml      # 示例任务
```

## 3. 一轮运行产物

```text
outputs/<run>/
  commands.json
  r0/
    len80_seed0/
      boltzgen/
        boltzgen_design_spec.yaml
      odesign/
        odesign_input.json
        odesign_run.sh
  scores.csv
  failure_tags.jsonl
```

## 4. 推荐 GPU 侧执行顺序

1. `python scripts/run_strategy_al.py --config configs/example_binder_task.yaml --dry-run` 检查命令与配置。
2. 将 `dry_run: false` 后在 GPU 机器运行。
3. 首轮只使用小预算：BoltzGen `num_designs=50~200`，ODesign `N_sample=2~5`。
4. 解析输出，生成 `scores.csv`。
5. 调用下一轮策略更新。

## 5. 后续可扩展点

- target ensemble；
- off-target panel；
- Boltz/AF2/Rosetta 多评分器；
- Bayesian surrogate；
- GPU budget scheduler；
- Web dashboard。
