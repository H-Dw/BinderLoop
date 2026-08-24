# BinderLoop 研究计划：Strategy-level Active Learning for Binder Design

## 1. 研究背景分析

蛋白 binder 计算设计通常包含：用户给定 target 结构与功能热点位点，生成 backbone/scaffold，进行序列设计或逆折叠，复折叠/复合物预测，最后用界面、折叠稳定性、热点命中和可开发性指标筛选候选。`Binder-Harness.docx` 的核心判断是正确的：真正值得研究的对象不只是“再造一个生成模型”，而是把 binder 设计过程从一次性 top-k 筛选提升为可复现、可审计、能利用失败样本的策略级主动学习系统。

本项目不关注上游 target 挖掘，也不包含湿实验验证；默认用户提供：target 结构、目标链、初始 hotspots，以及基本设计边界。Harness 的目标是：在固定底层模型（当前重点为 BoltzGen 与 ODesign/RFDiffusion3 优化版）的条件下，学习“什么设计策略更可能产生高质量计算候选”。

## 2. 方向判断：创新点是否突出、实现概率是否高

### 2.1 突出创新点

1. **策略级主动学习，而非候选级排序**：多数 pipeline 只问“哪些候选最好”；本方向进一步问“哪些设计配置/约束/采样策略最好”。优化对象从 candidate 转为 strategy。
2. **失败样本显式建模**：binder 设计失败样本高度有信息量，例如自身不折叠、热点未接触、界面消失、clash、pAE 高、候选同质化、scorer disagreement。Harness 可将失败标签转为下一轮策略更新，而不是仅丢弃失败样本。
3. **多目标、阶段性权重调度**：早期强调探索和覆盖，后期强调界面质量、refolding 稳定性和可开发性。Harness 可使权重调度、门控和停止标准程序化。
4. **模型无关的控制面**：BoltzGen/ODesign 都可以作为执行后端。研究贡献集中在策略空间、反馈表示、acquisition function、审计与可复现执行，而不是依赖某个单一模型性能。
5. **可复现与可审计**：记录模型版本、checkpoint、输入哈希、随机种子、参数 schema、失败项和输出指标，能显著降低普通 Agent workflow 中“参数漂移/失败被跳过/结果不可追溯”的问题。

### 2.2 实现概率

实现概率较高，因为第一阶段不需要训练新生成模型，也不依赖湿实验闭环。可先完成统一任务 schema、BoltzGen/ODesign dry-run 与 GPU 运行适配、结果解析与评分表、简单 successive halving/epsilon-greedy 策略选择、失败分类和参数变异规则、离线 replay 测试。

主要风险在于真实计算指标与真实实验 binder 活性可能不一致。因此论文/项目早期应明确定位为 **in-silico computational hit-rate optimization**，不夸大为真实活性优化。

## 3. 当前方向不足与改进建议

### 不足 1：目标函数容易被预测器偏差劫持

如果 reward 只来自同一类结构预测器，系统可能学习到 scorer reward hacking。改进：使用多模型/多指标交叉验证；引入 scorer disagreement 作为不确定性；对过高但不物理的界面分数设置异常检测；保留一部分探索样本和负控样本。

### 不足 2：失败标签若过粗，难以指导策略更新

| 失败类别 | 可观测指标 | 下一轮策略 |
|---|---|---|
| folding failure | binder pLDDT 低、设计链 refold RMSD 高 | 缩短长度、提高 scaffold 稳定性、降低探索噪声 |
| binding pose failure | ipTM/interface confidence 低 | 增强 hotspot/patch conditioning |
| hotspot miss | hotspot contact recall 低 | 提高 hotspot 权重、扩展 patch 或改为子集采样 |
| clash | clash count 高 | 增加几何过滤/relax 或降低局部约束强度 |
| diversity collapse | 序列/结构聚类单一 | 提高 diversity 权重、扩大长度/拓扑配额 |
| scorer disagreement | 指标冲突大 | 进入 uncertainty sampling 或人工审查队列 |

### 不足 3：策略空间过大，容易组合爆炸

改进：使用分阶段 search space：Round 0 少量长度、模型、hotspot 子集、种子做粗探索；Round 1 淘汰低成功率区域，扩大高潜力区域采样；Round 2 局部 refinement，调高 interface/hotspot 权重；Round 3+ 多样性补偿和负向设计。

### 不足 4：没有湿实验标签时，泛化能力评价困难

改进：用 compute budget 下的 high-quality computational hit rate、cross-target replay、对 target conformer/hotspot 扰动后的 robustness 作为评价。

## 4. 相关文献与技术脉络

1. **RFdiffusion / diffusion-based binder design**：以扩散模型生成满足 motif/interface 条件的蛋白 backbone，是当前 de novo binder 设计的重要路线。
2. **ProteinMPNN / inverse folding**：常用于 backbone 后序列设计，也可通过温度、固定/可变残基、避免氨基酸等参数影响序列多样性与稳定性。
3. **AlphaFold2/Multimer、Boltz 系列结构/复合物预测**：提供 pLDDT、pTM/ipTM、PAE 等结构置信指标，是 in-silico 筛选的重要反馈。
4. **BoltzGen: Toward Universal Binder Design**：本地仓库 README 显示其支持 protein/peptide/antibody/nanobody/small-molecule 等 binder protocol，具备 design spec YAML、pipeline 输出、analysis/filtering 等模块，适合作为 Harness 的主生成后端之一。
5. **ODesign: all-atom generative world model for biomolecular interaction design**：本地 README 显示其支持蛋白、配体、核酸、motif scaffold、partial diffusion 等模式，允许指定 epitope/hotspot 和生成结合伙伴，适合作为第二后端。
6. **Active learning / Bayesian optimization / AutoML for scientific discovery**：本项目的 ML 表述可借鉴 acquisition function、多目标优化、successive halving、multi-armed bandit、surrogate modeling 与 uncertainty sampling。

## 5. 可控参数调研总结

### 5.1 BoltzGen 可控参数

**任务 spec 层**：target 文件路径；include/exclude chains/residue ranges；include_proximity；binding_types；structure_groups；designed protein/peptide sequence 固定片段、长度和长度范围；residue_constraints；secondary_structure；design_insertions；cyclic peptide；bond constraints；total_len constraints。

**CLI / pipeline 层**：`--protocol`、`--num_designs`、`--budget`、`--diffusion_batch_size`、`--design_checkpoints`、`--step_scale`、`--noise_scale`、`--inverse_fold_num_sequences`、`--inverse_fold_avoid`、`--skip_inverse_folding`、`--only_inverse_fold`、`--alpha`、`--metrics_override`、`--additional_filters`、`--size_buckets`、`--refolding_rmsd_threshold`、`--reuse`、`--devices`、`--num_workers`、`--use_kernels`。

**适合主动学习优化的 BoltzGen 参数**：binder length range；hotspot/patch 子集与 binding/not_binding 类型；secondary structure/topology bias；protocol 选择；num_designs/budget/diffusion_batch_size；design checkpoint mix；step_scale/noise_scale；inverse_fold_num_sequences/avoid residues；alpha/metrics_override/hard filters；refolding_rmsd_threshold 和 size_buckets。

### 5.2 ODesign 可控参数

**输入 JSON 层**：`ref_file`；`chains` 的 chain_type、sequence 片段/长度；`hotspot`；`center_method`；`motif_scaffolding`；`partial_diff`；ligand/SMILES 模式。

**Hydra / CLI 层**：`infer_model_name`；`design_modality`；`input_json_path`；`exp_name`；`seeds`；`N_sample`；`use_msa`；`num_workers`；`invfold_topk`；`invfold_temp`；`invfold_use_beam`；`sample_diffusion.N_step`；`diffusion_chunk_size`；partial diffusion enable/snr；inference scheduler 的 `rho`、`s_max`、`s_min`、`gamma0`、`gamma_min`、`noise_scale_lambda`、`step_scale_eta`。

**适合主动学习优化的 ODesign 参数**：flex vs rigid receptor model；chain specification 中 binder length；hotspot 子集；N_sample/seeds；invfold_topk/temp/beam；partial_diff 区域与 snr；diffusion steps/chunk；motif scaffolding vs protein-binding-protein design mode。

## 6. 具体研究计划与阶段性目标

### 阶段 0：任务边界与数据协议（1 周）

定义 target spec、strategy spec、candidate metrics schema、run provenance。交付：示例 YAML、schema 文档、dry-run 命令生成。

### 阶段 1：底层模型适配（1–2 周）

将 BoltzGen 与 ODesign 抽象为统一 `ModelAdapter`。BoltzGen 生成 design spec YAML 与 `boltzgen run/configure` 命令；ODesign 生成 input JSON 与 Hydra override 命令；支持 dry-run 与 GPU-run；记录 commands.json、manifest.json、strategy.csv。

### 阶段 2：结果解析与计算评分（1–2 周）

BoltzGen 解析 `all_designs_metrics.csv`、`final_designs_metrics_*.csv`、`aggregate_metrics_analyze.csv`；ODesign 遍历 `predictions/*.cif` 与 run.log。输出 `scores.csv` 与 `failure_tags.jsonl`。

### 阶段 3：策略级主动学习 MVP（2 周）

初始策略采用 grid/random exploration；策略选择采用 successive halving + epsilon exploration；参数变异覆盖长度、热点子集、hotspot_weight、model choice、budget、inverse fold temp；引入 failure-aware rule 和 early/mid/late 多目标权重调度。

### 阶段 4：离线 replay 与消融实验（2 周）

比较固定 BoltzGen、固定 ODesign、random search 与 strategy-level AL。指标包括 top-k computational hit rate、每 100 GPU-hour 的有效候选数、diversity-adjusted hit rate、failure rate 降低幅度、scorer disagreement 降低幅度。

### 阶段 5：工程完善与论文化表达（2–4 周）

完善 provenance/audit；引入 off-target/negative design panel 接口；加入 target ensemble 策略；增加安全与合规 gate；整理论文结构。

## 7. Harness 转换方案：从平面设计到 Binder 设计

原有平面设计 Harness 中应保留 Planner/Executor/Critic 角色拆分、上下文隔离、可审计事件流、多阶段 workflow、生成与审查分离、产物目录与 revision loop。应删除或弱化品牌诊断、市场洞察、受众画像、视觉风格、图片生成、文案审查、UI/HTML 交付逻辑等。

BinderLoop 的简化角色：

```text
TargetSpecValidator
  → StrategyPlanner
  → ModelExecutor(BoltzGen/ODesign)
  → ResultParser
  → ScientificCritic/Scorer
  → ActiveLearner
  → next round
```

优先做极简 Python 包，而不是复刻完整 TypeScript control plane。待核心闭环跑通后，再补 daemon、SSE、审批和多用户队列。
