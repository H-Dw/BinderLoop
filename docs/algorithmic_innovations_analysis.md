# BinderLoop 当前算法创新点分析

## 1. 范围、口径与结论

### 1.1 分析范围

本文分析当前 `binderloop/` 主闭环，重点覆盖：

- `analysis/`：目标函数、失败分类、坐标与片段分析；
- `active_learning/`：对比样本、策略选择、质量回退；
- `agents/`：长度策略、模板挖掘、冲突消解、自改进规则；
- `orchestration/orchestrator.py`：各算法如何组成反馈闭环。

### 1.2 判定口径

本文把会改变优化对象、反馈表示、信用分配或下一轮候选分布的机制视为“算法创新候选”。以下内容不单独计为算法创新：

- Taiji 提交、GPU 分片、并发、重试、checkpoint、断点续跑；
- API 适配、prompt 压缩、日志、可视化；
- adapter、配置 schema、白名单、原子写入。

这些是重要的工程能力或算法安全边界，但不是本文讨论的核心算法贡献。

### 1.3 总结论

项目不是新的蛋白生成模型，而是一个**多尺度、失败感知、可回退的策略级主动学习控制器**。其优化对象由“哪个候选最好”提升为：

> 在固定预算和用户硬约束下，下一轮应选择哪种长度、热点压力、采样强度、结构模板和修复策略，才能提高可靠 binder 的计算命中率。

最有辨识度的贡献不是某个孤立公式，而是以下链条：

```text
分层核心目标
  → 正例 / 近失正例 / 困难负例
  → 候选—结构—片段三级信用分配
  → 结构表型驱动策略臂
  → PAE 门控模板迁移与形态自适应
  → 质量回退、反向压力控制
  → 跨轮经验沉淀与冲突消解
```

这里的“创新”指当前代码中形成了有辨识度的算法设计，不等同于已经证明学术首创。多臂搜索、MMR、回退和滑窗评分均有既有方法；论文级原创性仍需文献检索、基线和消融实验。

---

## 2. 创新点 A：分层核心目标与连续边界对比反馈

### 2.1 方法

项目不再用补偿式加权和决定候选优先级，而是先统一成功门控：

```text
iPTM >= 0.50
inter-chain PAE <= 10 Å
design pTM >= 0.70
refold RMSD <= 2.5 Å
```

候选按以下 `CoreRankKey` 词典序比较：

```text
(
  primary_gate_pass,
  min(normalized metric margins),
  iPTM,
  -PAE,
  -RMSD
)
```

证据：

- `binderloop/analysis/core_objective.py:11-18,39-96`
- `binderloop/agents/evaluation_agent.py:47-66,193-211`
- `binderloop/active_learning/rollback.py:272-300`

候选又被构造成三类反馈：

- `strict_positive`：iPTM、PAE、RMSD、pTM 四个 margin 全部通过；
- `near_miss`：未完全通过、但连续置信度较高的边界候选；
- `other_negative`：非正例池中除去 near miss 后的其他候选。

```text
m_iptm = (iPTM - 0.5) / 0.20
m_pae  = (10 - PAE) / 8
m_rmsd = (2.5 - RMSD) / 2.5
m_ptm  = (pTM - 0.7) / 0.3

c = sigmoid(2 × (0.40m_iptm + 0.25m_pae + 0.20m_rmsd + 0.15m_ptm))
```

证据：

- `binderloop/active_learning/examples.py:15-25,94-158,233-252`

### 2.2 算法价值

- 防止“高 iPTM、低折叠可靠性”候选劫持闭环；
- 避免阈值附近候选因二值切分而丢失信息；
- 给回退、父代排序、失败假设提供统一反馈底座；
- 使用最差 margin 防止某个强指标补偿失败的物理门控。

### 2.3 创新性与局限

创新强度：**中等，基础性组合创新**。  
词典序和 margin 本身不新；较有价值的是“统一物理门控 + 最弱项优先 + 连续 near-miss”贯穿整个 binder 闭环。

局限：

- margin 的归一化区间仍是启发式常数；
- 词典序表达强偏好，指标顺序仍需实验验证；
- 主要依赖计算预测器，仍存在 scorer bias；
- `min` inter-chain PAE 比平均 interface PAE 更容易通过。

---

## 3. 创新点 B：结构表型驱动的策略级主动学习

### 3.1 方法

传统流程通常是：

```text
生成 → 排序 → 取 top-k
```

当前项目进一步执行：

```text
失败类型 / 结构表型
  → 修复假设
  → 可执行策略臂
  → 改变下一轮生成分布
```

策略臂包括：

- `exploit_validated_strategy`
- `diversity_explore`
- `hotspot_repair`
- `clash_repair`
- `module_exploitation`
- `balanced_hold`

这些臂不再每轮全部出现：exploit 需要足够 strict positives 且轨迹未退化；diversity 需要多样性塌缩、平台期、无 strict positives 或没有其他可信干预。候选臂先由确定性证据门控，再由 LLM 在封闭目录内排序，非法输出回退到确定性优先级。

证据：

- `binderloop/active_learning/strategy.py:47-189,264-374`
- `binderloop/agents/active_learning_policy_agent.py:64-189`
- `binderloop/agents/hypothesis_agent.py:73-92`

父代选择显式划分探索与利用：

```text
exploit_n = round(n × (1 - exploration_ratio))
```

利用父代来自 `CoreRankKey` 排序，diversity 分支使用探索父代。`branch_width > 1` 时，LLM 排序后的 top-N 臂在固定轮预算内做受控比较；当前正例不足时模板利用等 exploitation 路径关闭。

证据：

- `binderloop/active_learning/strategy.py:191-262`
- `binderloop/orchestration/orchestrator.py:349-381,1183-1216`

### 3.2 算法价值

- 优化变量从候选升级为生成策略；
- 失败样本变成定向干预，不再只是被丢弃；
- 结构表型与采样动作形成可审计映射；
- 为 bandit、受控比较和按臂预算分配建立了接口。

### 3.3 创新性与局限

创新强度：**高，是项目最核心的算法定位创新**。  
多臂搜索本身不新，创新更可能成立于 binder 失败表型到可执行策略臂的领域映射及闭环组合。

局限：

- LLM 排序仍不是通过在线 posterior 学到的 arm value；
- 未显式估计不确定性、期望改进或 GPU-hour 收益；
- 多参数同轮变化造成因果归因混杂；
- 臂参数需要系统消融。

---

## 4. 创新点 C：候选—结构—片段三级信用分配

### 4.1 方法

评价由候选整体下沉到局部片段：

```text
候选级：iPTM、PAE、pTM、RMSD
  ↓
结构级：界面大小、热点覆盖、clash、链断裂、几何紧凑性
  ↓
片段级：局部接触、热点贡献、化学平衡、冲突、主链连续性
```

片段用默认窗口 8、步长 4 的滑窗构造。质量先经过硬门控：

```text
local_chain_break == 0
clash_density <= 0.15
interface_density >= 0.25
```

通过后按 `interface density → contact density → hotspot contact → specific contact → chemistry balance → clash` 词典序排序；未通过门控的片段标记为 low，通过且界面密集或命中 hotspot 的片段标记为 high，其余为 medium。旧 `quality_score` 只用于历史 artifact 兼容。

证据：

- `binderloop/analysis/structure_features.py:272-322,326-438`

### 4.2 算法价值

整体成功不表示所有片段都值得复用，整体失败也不表示所有局部结构都无价值。三级表示实现：

- 将稀疏候选奖励分解为局部反馈；
- 从失败结构中保留局部成功模块；
- 定位 clash、链断裂和非特异疏水斑块；
- 为模板迁移与模块修复提供信用来源。

### 4.3 创新性与局限

创新强度：**中高，是模板迁移成立的关键中间表示**。  
滑窗和手工结构特征常见；创新在于该表示直接进入下一轮动作，而不只是生成分析报告。

局限：

- 固定窗口可能切断真实二级结构；
- 未识别序列不连续但空间连续的 motif；
- 接触类型是几何近似而非精确能量；
- 门控阈值与词典序优先级仍需实验校准。

---

## 5. 创新点 D：PAE 门控的跨轮结构模板迁移与混合探索

### 5.1 方法

高质量片段不是无条件复用。默认资格门控为：

```text
preserve_eligible(structure)
  ⇔ min_design_to_target_pae <= 10 Å
```

随后：

1. 按 inter-chain PAE 升序优先选择局部最可信结构；
2. 再按片段质量降序；
3. 检查质量、跨度和模板源；
4. 生成 `structure_redesign` 模板；
5. 固定模板片段，只设计其余 binder 区域。

证据：

- `binderloop/agents/fragment_template_mining_agent.py:55-88,90-197`
- `binderloop/agents/fragment_template_mining_agent.py:220-248,323-419`

模板进入下一轮后，预算被拆为：

```text
template-conditioned exploitation
  +
template-free exploration
```

`template_conditioned_fraction` 最大为 0.8；自由探索分支会主动剥离模板字段。若出现折叠或 refolding 退化，policy 会降低模板条件化比例。

证据：

- `binderloop/active_learning/strategy.py:100-180`
- `binderloop/agents/active_learning_policy_agent.py:129-148`

### 5.2 算法价值

- 用局部交互置信度替代全局复合分数作为模板资格；
- 将局部信用转成可执行的结构 redesign；
- 兼顾知识复用和避免单模板锁死；
- 模板分配比例本身成为可反馈调整的策略变量。

### 5.3 创新性与局限

创新强度：**高，是当前最具体的领域算法创新之一**。  
最有辨识度的是“局部 PAE 门控 + 片段级信用 + 结构 redesign + 模板/自由混合搜索”的整体设计。

局限：

- `min PAE <= 10 Å` 只要求一个局部对可信，可能偏宽松；
- 模板 top-k 仍按启发式排序；
- 没有显式模板多样性或模板间互补优化；
- 结构复用的真实增益需要与无模板、随机模板、iPTM 门控做消融。

---

## 6. 创新点 E：结构反馈驱动的搜索形态自适应

该创新包含两个互补方向：搜索对象的长度自适应，以及 target 条件区域的表位自适应。

### 6.1 Binder 长度策略

结构按实际长度分桶。长度桶先检查支持样本数与 foldability gate，再按词典序比较：

```text
(
  support_count,
  foldability_gate_pass,
  reliability,
  -chain_break_fraction,
  PAE_available,
  -interchain_PAE,
  interface_size,
  -clash_density
)
```

决策规则：

- 折叠失败占优 → 整体向短移动；
- 界面弱但折叠尚可 → 向长移动；
- 某长度明显更优 → 聚焦其邻域；
- 无显著信号 → 保持；
- 删除已观测且 foldability gate 失败的长度；缺失 PAE 的桶不会因少参与一项平均而占优。

证据：

- `binderloop/agents/binder_length_policy_agent.py:35-147`
- `binderloop/agents/binder_length_policy_agent.py:184-313`

### 6.2 共识接触驱动的表位裁剪

系统聚合可靠结构实际接触的 target residues，形成共识 engaged epitope。`auto` 模式下：

- 已命中用户 hotspot → 收紧至 hotspot 附近；
- 持续接触错误 patch → 不裁到错误区域，保留整链并提高 hotspot 优先级；
- 样本不足 → 延迟激进裁剪。

证据：

- `binderloop/analysis/epitope.py:66-93,96-199`

### 6.3 创新性与局限

创新强度：**中高，属于受约束的形态自适应搜索**。  
长度和裁剪规则本身是启发式的，但它们共同让搜索域随结构反馈重塑，而不是只微调采样温度。

局限：

- 长度桶样本少时均值不稳定；
- 当前只用简单支持率，没有置信区间或贝叶斯收缩；
- 共识接触可能强化早期错误 patch；
- crop 与 hotspot 压力之间存在耦合，需要单因素比较。

---

## 7. 创新点 F：质量感知的非线性轨迹控制

### 7.1 最佳轮回退与失败臂剪枝

每轮使用 `RoundRankKey` 比较：

```text
(
  strict_positive_yield,
  median(top-k worst_margin),
  median(top-k iPTM),
  -median(top-k PAE),
  -median(top-k RMSD)
)
```

若当前轮相对历史最佳轮的下降超过容忍度，或连续非最佳轮超过 patience，则：

- 抑制退化轮的所有策略建议；
- 精确恢复最佳轮 config/jobs；
- replay 最佳轮；
- 记录导致退化的 arm signature，避免再次执行同一分支；
- 可在持续无改善时停止。

基础设施或配置失败不进入质量历史，不会被误判为科学退化。

证据：

- `binderloop/active_learning/rollback.py:88-257,272-300`
- `binderloop/orchestration/orchestrator.py:1027-1061,1193-1216`
- `binderloop/active_learning/strategy.py:57-68`

### 7.2 加压后退化的负反馈控制

系统检测“增加 hotspot/contact/template 压力后，iPTM、PAE、pTM 或 RMSD 反而恶化”的轨迹冲突。触发后：

- 不再继续抬高 hotspot_weight；
- 撤销新增 auxiliary hotspots 或过紧 crop；
- 降低模板压力；
- 提高探索比例；
- 转向替代 patch、长度或拓扑。

证据：

- `binderloop/orchestration/orchestrator.py:3113-3229`
- `binderloop/orchestration/orchestrator.py:3269-3389`

### 7.3 算法价值

这不是普通异常重试，而是搜索轨迹控制：

- 从线性 hill-climbing 变为带分支恢复的搜索；
- 区分“实验动作无效”和“任务根本没执行”；
- 用负反馈阻止“界面加压 → 折叠退化 → 再加压”的正反馈失控；
- 近似提供一次跨轮反事实判断：该策略改变之后是否系统退化。

### 7.4 创新性与局限

创新强度：**高，是闭环稳定性与搜索效率的主要算法贡献**。  
回退本身常见；领域创新在于统一核心 reward、arm signature、执行失败隔离和压力冲突反转。

局限：

- 仍是观察性前后比较，不是严格因果推断；
- exact replay 可能消耗预算却不产生新信息；
- reward 噪声可能造成错误回退；
- 缺少基于置信区间的序贯检验。

---

## 8. 创新点 G：受治理的跨轮经验学习与冲突消解

### 8.1 方法

系统把跨轮经验存成目标无关的结构化规则，而非无限追加自然语言：

- 规则被分配到成功模式、失败规避、参数效应、结构上下文、探索利用、回退恢复等固定模块；
- 经验先去除 target 名称、路径、残基和绝对长度；
- 新规则先按 canonical signature 召回相似规则；
- 再判断 equivalent、subsumes、complementary、contradictory 等关系；
- 支持数和 utility 达标才激活，矛盾积累后退役；
- 冲突规则不会直接叠加，而由物理指标、历史最佳和受控比较消解。

证据：

- `binderloop/agents/self_improvement_skill_agent.py:34-85,102-174`
- `binderloop/agents/self_improvement_skill_agent.py:203-298,301-361`
- `binderloop/skills/self_improvement.py:25-39,98-215,218-293`
- `binderloop/agents/strategy_conflict_resolution_agent.py:14-124,163-250`

长期记忆召回还采用“结构字段召回 → 可选语义重排 → 规则簇去重 → MMR”的流程，以兼顾相关性与多样性。

证据：

- `binderloop/agents/memory_retrieval_agent.py:47-99,101-139,197-266`

### 8.2 创新性与局限

创新强度：**中等，属于元学习/经验治理层创新**。  
结构化规则生命周期比普通 prompt memory 更可靠，但 MMR、语义去重和规则投票均有成熟先例。其贡献主要是防止 binder 闭环把偶然相关性立即固化为策略。

局限：

- 规则效用仍受短轨迹和混杂参数影响；
- 去 target 化不保证真正跨靶点泛化；
- LLM 关系判断可能不稳定；
- 尚无 held-out target 上的 transfer 验证。

---

## 9. 各创新点之间的逻辑联系

### 9.1 主因果链

```text
[A 分层目标与连续对比反馈]
  提供统一质量标尺与近失信息
        ↓
[C 三级信用分配]
  把整体质量分解为结构表型和局部模块
        ↓
┌──────────────────────┬─────────────────────────┐
│ [B 策略级主动学习]   │ [D PAE门控模板迁移]      │
│ 选择修复/探索策略臂  │ 把局部成功转成结构先验   │
└──────────┬───────────┴────────────┬────────────┘
           │                        │
           └──────────┬─────────────┘
                      ↓
       [E 长度与表位的搜索形态自适应]
          重塑下一轮可搜索区域
                      ↓
         受约束 BoltzGen/ODesign 生成
                      ↓
             新一轮真实观测
                      ↓
[F 回退与压力负反馈] ──保护──→ A/B/D/E
  防止坏分支、过度加压和执行故障污染学习
                      ↓
[G 跨轮经验规则] ─────增强──→ B/F
  将多轮证据沉淀为可激活、可冲突、可退役的策略知识
```

### 9.2 功能分层

| 层次 | 对应创新 | 解决的问题 |
|---|---|---|
| 目标层 | A | 什么叫“更好的 binder” |
| 表示层 | A、C | 如何表示成功、近失、失败及局部贡献 |
| 决策层 | B、E | 下一轮改变什么 |
| 迁移层 | D | 如何把局部成功直接复用 |
| 轨迹控制层 | F | 何时继续、回退或反转策略 |
| 元学习层 | G | 如何跨轮积累经验又避免错误固化 |

### 9.3 必要依赖

- 没有 A，回退、父代排序和正负样本会使用不一致目标；
- 没有 C，D 只能复用整条结构，无法做局部信用迁移；
- 没有 B，结构分析只会停留在报告层；
- 没有 F，B/D/E 的错误动作会被线性继承；
- 没有 G，每次运行只能从固定启发式重新开始；
- D、E、G 都依赖 A/C 提供可信证据门控。

---

## 10. 不应包装成算法创新的内容

| 内容 | 判定 | 原因 |
|---|---|---|
| 多 Agent 名称与顺序 | 不是独立创新 | 固定工作流拆分，价值主要在工程组织 |
| Taiji/GPU 多卡分片 | 工程创新 | 改善吞吐，不改变科学搜索准则 |
| retry/checkpoint/resume | 工程创新 | 提高可靠性，不直接优化候选分布 |
| LLM API 多 provider | 工程创新 | 接入能力，不是 binder 算法 |
| prompt compaction | 工程/推理效率 | 防止超上下文，但不构成科学策略 |
| 配置白名单与原子写 | 算法护栏/工程 | 保护算法，不是主要搜索方法 |
| 单独使用 MMR | 现有算法组件 | 可作为 G 的组成部分，不宜单列首创 |
| 旧 `core_objective` 加权分 | 历史兼容指标 | 仅保留用于旧 artifact 展示，新决策使用门控词典序 |
| `diversity` / `vendi` 指标 | 上游指标消费 | 当前仓库主要读取 BoltzGen 输出列，未实现序列聚类、Hamming 距离或 MMseqs 多样性选择 |
| 配置中的 `successive_halving` 名称 | 尚非已实现算法 | 当前实际行为是规则策略臂、探索—利用和回退，不能表述为经典 successive halving |

---

## 11. 创新强度排序

### 第一梯队：最适合作为核心贡献

1. **B：结构表型驱动的策略级主动学习**
2. **D：PAE 门控的片段模板迁移与模板/自由混合搜索**
3. **F：质量回退、失败臂剪枝与压力负反馈**

### 第二梯队：支撑核心贡献

4. **C：候选—结构—片段三级信用分配**
5. **E：长度与表位的搜索形态自适应**
6. **A：统一核心目标与连续 near-miss 对比反馈**

### 第三梯队：增强型贡献

7. **G：受治理的跨轮经验规则与策略冲突消解**

建议对外表述为一个主贡献和三个子机制：

> **主贡献：多尺度失败感知、可回退的策略级 binder 主动学习。**
>
> 子机制 1：连续 near-miss 与三级结构信用分配；  
> 子机制 2：局部 PAE 门控的片段模板迁移和搜索形态自适应；  
> 子机制 3：质量回退、压力反转与受治理的跨轮经验学习。

---

## 12. 论文级验证建议

要把“代码创新”提升为可信研究贡献，至少需要以下消融：

1. 固定 BoltzGen vs 当前完整闭环；
2. best-iPTM reward vs `RoundRankKey`；
3. 二值正负例 vs strict-positive/near-miss/other-negative；
4. 无片段信用 vs 片段信用；
5. 无模板 vs 随机模板 vs iPTM 门控模板 vs PAE 门控模板；
6. 100% 模板利用 vs 模板/自由混合搜索；
7. 固定长度 vs动态长度；
8. 无回退 vs exact replay vs 分支切换；
9. 无压力冲突控制 vs 负反馈控制；
10. 无经验规则 vs run-local 规则 vs cross-target transfer。

关键指标应包括：

- 每 100 GPU-hour 的严格成功候选数；
- strict positive yield 和 margin-weighted yield；
- top-k 最差归一化 margin、iPTM、PAE、RMSD 的分层统计与置信区间；
- 候选结构/序列多样性；其中序列多样性需新增本地独立计算，不能只依赖 BoltzGen 透传列；
- 模板分支相对自由分支的增益；
- 回退前后恢复速度和被节省的无效轮次；
- held-out target 上的迁移增益；
- 对不同阈值、随机种子和预算的稳健性。

当前最需要补强的理论与实验部分是：

- 把规则臂升级为可估计收益与不确定性的 contextual bandit；
- 用分层贝叶斯或置信区间处理小样本长度桶；
- 用受控单因素分支降低多参数混杂；
- 对模板使用平均/分位数 interface PAE，而不仅是最小 PAE；
- 引入跨预测器 disagreement 和最终湿实验标签，减少 scorer bias。

---

## 13. 最终判断

当前项目已经存在明确的算法创新主线，且不是简单的“LLM + 蛋白生成工具”：

1. 用统一物理核心目标和连续 near-miss 表示定义反馈；
2. 用三级结构信用定位可保留与可修复的局部模块；
3. 用结构表型选择下一轮策略臂；
4. 用 PAE 门控把局部成功转成结构 redesign 模板，同时保留自由探索；
5. 用长度与表位自适应重塑搜索域；
6. 用质量回退和压力负反馈控制非线性搜索轨迹；
7. 用可激活、可冲突、可退役的规则沉淀跨轮经验。

其中最值得强调的不是任一常见组件，而是：

> **将多尺度结构反馈、局部模板迁移、失败定向修复和非线性轨迹控制耦合成一个可执行的策略级 binder 主动学习算法。**

这一表述与当前代码证据一致，同时避免把工程可靠性或尚未验证的学术首创夸大为算法贡献。
