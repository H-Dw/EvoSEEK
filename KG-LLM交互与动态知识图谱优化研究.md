# KG 参与 Fitness 决策、LLM Agent 交互与动态知识图谱优化研究

> 研究日期：2026-08-15  
> 分析范围：当前 fitness-agents 实现、KG+LLM Agent 代表性论文及其官方 GitHub 代码  
> 交付性质：需求澄清与架构研究；本次不修改系统代码

## 0. 结论摘要

当前系统中的 KG 可以准确地定义为：

> 一个以本轮实验 Observation 为事实核心，分开保存 Prediction、Evidence、Hypothesis 和 Agent Query 的、按运行隔离的 SQLite 结构化记忆与审计层。

它已经参与思考和决策，但参与方式有两条彼此并行、能力仍较受限的路径：

1. **定性路径**：KG 预先生成一个固定的 hypothesis context，注入 LLM；LLM 只调用一次，输出结构化 Hypothesis；Hypothesis 再用于候选过滤或排序。
2. **定量路径**：KG 根据已观察变体计算位点—残基统计证据，与理化、保守性、结构等 Evidence 合并为 knowledge score，进入 acquisition score，影响最终实验批次。

所以，当前实现不是“LLM 自主使用 KG 反复查询、验证和修正”的 Agentic KG 推理，而是：

> 固定 KG 预查询 + 一次 LLM 假设生成 + 数值型知识加权。

建议保留现有 SQLite KG 作为可信的实验事实账本，先在其上增加**有类型、可审计、有限步数的查询与变更接口**，不要立即替换存储引擎。目标形态应是：

> LLM 负责提出查询计划和可证伪假设；KG 负责确定性执行、证据路径、反证和时间边界；实验系统负责产生真实 Observation；只有经过校验的 ChangeProposal 才能提交到 KG。

文献和代码复用方面，最适合当前项目的不是直接移植某一个完整框架，而是组合复用四类机制：

- **RoG / BYOKG-RAG**：关系计划 → 确定性图执行 → 证据路径 → LLM 决策；
- **ToG / ReKG-MCTS**：有预算、可停止、兼顾探索与利用的图搜索；
- **KG-RAG**：只向 LLM 提供最小、带出处、带统计和反证的 EvidencePack；
- **Graphiti**：增量 episode、事实有效期、历史保留和来源追溯。

其中，RoG、KG-RAG、HippoRAG、Graphiti、BYOKG-RAG 的许可证相对清晰，可作为接口或局部实现参考；ToG、DALK、MindMap 的仓库在本次检查快照中没有完整 LICENSE 文件或实现较为脚本化，应只借鉴设计，不直接复制代码。

---

## 1. 研究问题与检索方法

### 1.1 研究问题

本报告回答四个问题：

1. 当前 KG 如何参与思考、候选生成和最终选择？
2. KG 内部实际保存了什么，现有知识如何被结构化？
3. 如何把 KG 升级为 LLM 的外挂结构化知识库，并在每轮实验后动态优化？
4. 哪些 KG+LLM Agent 论文和 GitHub 实现可以复用，复用到什么程度？

### 1.2 检索与核验策略

检索时间为 2026-08-15。检索词围绕以下组合展开：

- knowledge graph + large language model + agent + reasoning
- dynamic knowledge graph + LLM agent + memory
- knowledge graph retrieval augmented generation + biomedical
- iterative graph traversal + LLM
- temporal knowledge graph + agent memory

优先采用论文正式页面、OpenReview、ACL Anthology、NeurIPS Proceedings、Bioinformatics/Oxford Academic 和作者官方 GitHub。初步检索使用 OpenAlex 等聚合来源扩大召回，最终结论只依赖论文原文和官方代码仓库。纳入标准为：

- KG 实际进入 LLM 检索、规划、推理或动态记忆流程；
- 论文给出足够的算法或系统描述；
- 能找到官方或作者声明的代码仓库时，进一步检查模块边界、依赖和许可证。

本次不是严格系统综述，也没有对所有 KGQA 或 GraphRAG 工作穷举。学术检索 MCP 在当前会话未挂载，因此采用公开学术索引和正式出版页面进行回退检索；这会降低引文数据库覆盖的一致性，但不影响本文对核心论文及官方代码的逐项核验。

---

## 2. 当前系统中的 KG 到底是什么

### 2.1 当前流程中的真实位置

当前每轮流程可表示为：

~~~text
初始可见 Observation
        |
        +------------------------------> 写入 KG
        |
        v
训练轻量 Predictor
        |
        v
对未测 Variant 产生 Prediction(mean, std, OOD, components)
        |
        +--> 计算 physchem / conservation / structure / KG Evidence
        |             |
        |             +---------------> Prediction + Evidence 写入 KG
        |
        +--> KG.hypothesis_context(round)
                       |
                       v
                单次调用 LLM
                       |
                       v
             结构化 Hypothesis
                 |             |
                 |             +-------> 写入 KG，status=active
                 v
        候选过滤 / 候选优先级

Prediction ----------------------------+
Evidence -> knowledge_score -----------+--> acquisition score
Hypothesis -> eligible candidates -----+         |
                                                 v
                                   批次选择 + diversity penalty
                                                 |
                                                 v
                                         实验 / oracle reveal
                                                 |
                                                 v
                                    新 Observation 写回 KG
                                                 |
                                                 +--> 下一轮
~~~

对应实现位置：

- 主循环：src/fitness_agents/loop/orchestrator.py
- KG 持久层：src/fitness_agents/knowledge/graph.py
- KG 受限工具：src/fitness_agents/knowledge/tool.py
- 知识证据计算：src/fitness_agents/knowledge/engine.py
- LLM Scientist Agent：src/fitness_agents/agents/scientist.py
- 结构化 LLM 输出：src/fitness_agents/agents/llm.py
- 候选生成：src/fitness_agents/mutation/generators.py
- acquisition 与 diversity：src/fitness_agents/acquisition/policies.py

### 2.2 KG 内部结构

当前并没有单独的 edge 表或通用 RDF/Neo4j 图。它是一个关系型数据库实现的“属性图式语义层”，共有七类表：

| 表 | 角色 | 关键字段 | 语义 |
|---|---|---|---|
| variants | 变体实体 | variant_id、code、sequence、mutation_notation、mutation_count、split_role | 被预测、被测量和被解释的中心对象 |
| mutations | 变体—突变展开 | variant_id、position、wt_aa、mutant_aa | 将四字符变体拆成位点级结构 |
| observations | 实验事实 | observation_id、variant_id、assay_id、fitness、round_id、source | 已揭示的真实测量；事实层 |
| predictions | 模型输出 | mean、std、interval、OOD、model_version、component scores、round_id | 可失效、可重新计算；不能当成测量 |
| evidence | 知识证据 | channel、statement、score、source_id、confidence、evidence_type、round_id | 支持或削弱某个候选的计算证据 |
| hypotheses | LLM 假设 | statement、evidence_ids、round_id、status | Agent 产生的可追踪主张 |
| agent_queries | 查询审计 | operation、parameters、result、round_id | 记录 Agent 看过什么以及何时看过 |

把这些表还原成逻辑图，可理解为：

~~~text
(Variant)-[HAS_MUTATION]->(Mutation at Position)
(Variant)-[MEASURED_AS]->(Observation)-[IN_ASSAY]->(Assay)
(Variant)-[PREDICTED_AS]->(Prediction)-[BY_MODEL]->(ModelVersion)
(Variant)-[EVALUATED_BY]->(Evidence)-[FROM]->(Source)
(Hypothesis)-[CITES]->(Evidence)
(AgentQuery)-[RETURNED]->(Observation | Prediction | Evidence | Hypothesis)
~~~

这些逻辑边目前主要由外键、查询和 JSON 引用隐式表达，而不是一等公民的图边。

### 2.3 已有知识如何被整合为结构化信息

当前知识进入系统有四条路径。

#### A. 真实实验数据

每条 FitnessObservation 被保存为 Observation，并绑定完整 Variant、assay、round 和 source。可见性规则要求 LLM 在第 r 轮只能看到 round_revealed < r 的测量，从而避免当前轮或最终测试标签泄漏。

这是当前 KG 中可信度最高的内容。

#### B. 轻量模型输出

Predictor 的均值、标准差、90% 区间、OOD、模型版本、分量得分和干预标签被保存为 Prediction。系统明确保留 Prediction 与 Observation 的类型差异，因此不会把模型估计伪装成真实实验结果。

#### C. 计算知识

KnowledgeEngine 为候选生成多通道 Evidence：

- physchem：理化变化；
- conservation：位点保守性或预设 profile；
- structure：预计算结构容忍度，不直接宣称亲和力；
- kg：来自当前运行可见 Observation 的位点—残基统计。

KG 通道首先计算各突变残基在已观察完整变体中的平均 fitness 和计数，然后用 count / (count + 3) 做小样本收缩，再与当前全局均值比较。各通道 Evidence 最终按 confidence 加权平均成 knowledge score。

#### D. LLM 生成的假设

KG 为 ScientistAgent 生成：

- 可见 Observation 数和全局均值；
- 观测上较有利的位点—残基组合及 support；
- 高 fitness 的已测变体；
- 当前候选的 Prediction、区间、OOD 和 Evidence；
- 历史 Hypothesis。

LLM 返回严格 JSON 结构的 Hypothesis，再被写入 KG，并通过 preferred_residues 影响候选过滤或排序。

### 2.4 KG 目前如何影响决策

当前 KG 对决策有三个层次的作用：

| 层次 | 当前作用 | 是否直接改变候选 |
|---|---|---|
| 事实记忆 | 保存可见 Observation、模型输出、证据、假设和查询历史 | 间接 |
| LLM 引导 | 给 LLM 一次固定的 hypothesis_context；Hypothesis 引导候选生成 | 是 |
| 数值融合 | Evidence → confidence-weighted knowledge score → acquisition score | 是 |

Acquisition 的核心形式为：

~~~text
base_score = Greedy / UCB / Thompson
final_score = base_score + knowledge_weight × knowledge_score
batch_select = final_score - diversity_penalty
~~~

因此，KG 不是只用于解释结果；它已经能改变 eligible candidates 和最终 acquisition rank。

### 2.5 当前能力边界

1. **不是通用生物知识图谱**：WT 固定为 VDGV，位点固定为 39、40、41、54，主要是 GB1 单任务、单运行记忆。
2. **不是自主 Agent 查询循环**：主流程只自动调用 hypothesis_context；虽然 explain_variant 已存在，但没有进入主推理循环。
3. **LLM 不能选择关系、路径或下一步工具**：KG 查什么由代码预先决定，而不是由 QueryPlan 决定。
4. **Hypothesis 缺少自动验证状态机**：新结果会写入 Observation，但不会自动把旧假设改成 supported、contradicted 或 inconclusive。
5. **证据相关性可能被高估**：位点—残基平均值来自完整变体，存在强 epistasis 混杂；当前 caveat 有提醒，但数值层仍是简单加权。
6. **外部知识较薄**：结构、保守性和理化知识是轻量 profile 或预计算指标，还没有文献实体、PDB/UniProt 来源路径、原子接触、assay condition 等更丰富的语义关系。

---

## 3. 示例案例：KG 如何影响一轮 fitness 优化

下面是一个为了说明机制而构造的示例，不是仓库中的真实运行结果。

### 3.1 第 2 轮开始时的可见事实

设 WT 为 VDGV，四个字符对应位点 39、40、41、54。KG 中已有：

| Observation | Variant | Fitness | Round |
|---|---:|---:|---:|
| obs-1 | FDGV | 1.20 | 0 |
| obs-2 | VDGW | 1.10 | 0 |
| obs-3 | ADGV | 0.35 | 1 |
| obs-4 | VDGV | 1.00 | 1 |

全局均值为 0.9125。

对于候选 FDGW：

- 39:F 的可见均值为 1.20，support=1；
- 54:W 的可见均值为 1.10，support=1；
- 两个单点信号都高于全局均值，但尚未观察 F 与 W 的组合效应。

当前 KG Evidence 的近似计算为：

~~~text
effect(39:F) = 1/(1+3) × (1.20 - 0.9125) = 0.071875
effect(54:W) = 1/(1+3) × (1.10 - 0.9125) = 0.046875
raw_score      = mean(0.071875, 0.046875) = 0.059375
kg_score       = tanh(raw_score) ≈ 0.0593
confidence     = 0.25 + 0.03 × support(2) = 0.31
~~~

这个信号很弱，符合小样本条件；它只能说明“值得测试”，不能证明组合一定增益。

### 3.2 KG 给 LLM 的信息

实际字段会更多，这里压缩为：

~~~json
{
  "as_of_round": 2,
  "visible_global_mean_fitness": 0.9125,
  "beneficial_site_residues": [
    {"position": 39, "residue": "F", "lift": 0.2875, "support": 1},
    {"position": 54, "residue": "W", "lift": 0.1875, "support": 1}
  ],
  "current_candidate_predictions": [
    {
      "variant": "FDGW",
      "fitness_mean": 1.25,
      "fitness_std": 0.18,
      "ood_score": 0.31,
      "evidence": [
        {"channel": "kg", "score": 0.0593, "confidence": 0.31},
        {"channel": "structure", "score": 0.62, "confidence": 0.45}
      ]
    }
  ],
  "caveats": [
    "site-level association may be confounded by complete-variant epistasis"
  ]
}
~~~

### 3.3 LLM 产生的 Hypothesis

合理的结构化输出应类似：

~~~json
{
  "hypothesis_id": "hyp-r2-01",
  "statement": "39:F 与 54:W 的组合可能提高 fitness，但存在未观测的负向 epistasis 风险。",
  "preferred_residues": {"39": ["F"], "54": ["W"]},
  "evidence_ids": ["ev-39F", "ev-54W", "ev-structure-FDGW"],
  "expected_outcome": "FDGW 高于当前可见全局均值",
  "falsification": "若 FDGW 明显低于 0.9125，或低于单点 FDGV/VDGW，则组合增益假设被削弱。"
}
~~~

该 Hypothesis 会让匹配 39:F 和 54:W 的候选更容易进入 eligible set。假设 beta=1.5，则 FDGW 的 UCB 为：

~~~text
UCB = 1.25 + 1.5 × 0.18 = 1.52
~~~

knowledge score 只做附加修正，之后还会应用批次 diversity penalty。因此，最终选择是 Predictor、KG Evidence、Hypothesis 和多样性共同作用，而不是 LLM 单独拍板。

### 3.4 新结果如何反馈

假设实验测得 FDGW=0.40。该结果说明两个单点的正向关联没有组合成正向结果，可能存在负向 epistasis。

**当前系统会做：**

- 新建 FDGW 的 Observation；
- 更新当前运行的 residue statistics；
- 下一轮重新生成 kg Evidence；
- 历史 Hypothesis 仍保留为 active。

加入 0.40 后，39:F 的均值变成 0.80，54:W 的均值变成 0.75；两者不再呈现原来的正向信号。下一轮 KG 数值会自然下降。

**目标系统还应该做：**

~~~text
(hyp-r2-01)-[TESTED_BY]->(Observation: FDGW=0.40)
(Observation)-[CONTRADICTS]->(hyp-r2-01)
hyp-r2-01.status = contradicted
hyp-r2-01.valid_to_round = 2

生成新的 Evidence：
"39:F 与 54:W 在独立单点背景中表现较高，
但组合 FDGW 出现负向 epistasis；不要继续把二者视为可加性增益。"
~~~

这样，下一轮 LLM 获得的不是原始历史堆积，而是直接可行动的指引：

- 避免继续押注 F+W 的可加性；
- 优先测试仅保留一个信号的邻域；
- 若要重试 F+W，应改变 40/41 背景并显式验证 epistasis；
- 将该反例纳入候选解释和不确定性评估。

---

## 4. 目标 KG：从运行日志升级为外挂结构化知识库

### 4.1 核心原则

1. **事实、计算和主张分层**：Observation 不能被 Prediction 或 Hypothesis 覆盖。
2. **LLM 不直接写可信事实**：LLM 只能提交 ChangeProposal；真实 Observation 只能来自实验后端或经批准的数据导入。
3. **保留历史，不覆盖历史**：错误或过时的事实应失效、被反证或被 supersede，而不是删除。
4. **最小证据包，而不是整图塞入 prompt**：只提供与当前决策有关的路径、数字、反证和 caveat。
5. **图负责约束，LLM 负责语义规划**：查询执行、可见性、聚合和评分必须确定性。
6. **每个结论可追溯**：DecisionRecord 必须引用 Evidence、Prediction、Hypothesis 和 query_id。

### 4.2 建议的语义层

#### 实体

| 层 | 建议实体 |
|---|---|
| 生物对象 | Protein、Sequence、Residue、Variant、Mutation、Structure、AtomContact |
| 实验对象 | Assay、Condition、Campaign、Round、Experiment、Observation |
| 计算对象 | Model、Prediction、Feature、Evidence、CalibrationRecord |
| 知识对象 | Source、Publication、Claim、Hypothesis、Decision |
| 审计对象 | AgentRun、Query、ChangeProposal、ValidationResult |

#### 关键关系

~~~text
Protein HAS_SEQUENCE Sequence
Variant DERIVED_FROM Protein
Variant HAS_MUTATION Mutation
Mutation AT_RESIDUE Residue
Residue HAS_CONTACT AtomContact
Variant MEASURED_IN Assay
Experiment PRODUCED Observation
Prediction PREDICTS Variant
Prediction GENERATED_BY Model
Evidence ABOUT Variant
Evidence DERIVED_FROM Observation | Prediction | Source
Hypothesis SUPPORTED_BY Evidence
Hypothesis CONTRADICTED_BY Evidence
Hypothesis TESTED_BY Experiment
Decision SELECTED Variant
Decision JUSTIFIED_BY Evidence | Hypothesis | Prediction
Hypothesis SUPERSEDES Hypothesis
~~~

### 4.3 Observation、Prediction、Evidence、Hypothesis 的规范

#### Observation：不可变的实验事实

至少包含：

- observation_id、variant_id、assay_id；
- 原始值、标准化值及单位；
- 条件、重复、批次、QC 状态；
- event_time、ingested_at、round_id；
- source、数据哈希；
- supersedes 或 correction_of，仅用于更正链。

Observation 原记录不可被 LLM 更新。实验修正应生成新记录并指向旧记录。

#### Prediction：有版本的模型输出

至少包含：

- prediction_id、variant_id、target；
- mean、uncertainty、interval、OOD；
- model_id、model_version、training_snapshot_id；
- feature/evidence 版本；
- round_id、generated_at、calibration_status。

新模型结果应新增 Prediction，不覆盖旧版本。

#### Evidence：可计算、可反驳、带来源的证据单元

建议增加：

- polarity：support / contradict / neutral；
- evidence_type：measurement、aggregate、model、structure、literature、rule；
- target_claim_id 或 target_variant_id；
- score、confidence、support_count；
- source_ids、derivation_rule、input_snapshot；
- scope：assay、condition、background、positions；
- caveat、valid_from、valid_to。

Evidence 不能只是一句自然语言；应同时有机器可用的方向、强度、适用范围和出处。

#### Hypothesis：可证伪的暂时主张

建议增加：

- claim：受控结构，例如 interaction(39:F,54:W) increases fitness；
- expected_effect、uncertainty；
- preferred 与 avoided constraints；
- evidence_ids 与 counterevidence_ids；
- falsification_rule；
- proposed_test；
- status：proposed / testing / supported / contradicted / inconclusive / superseded；
- valid_from_round、valid_to_round、supersedes。

#### Decision：不要只保留结果，还要保留决策依据

建议独立增加 DecisionRecord：

- candidate set 与 selected variants；
- predictor score、knowledge score、acquisition score；
- hypothesis_id、evidence_ids、query_ids；
- 未选择的主要反证；
- policy/version、预算和多样性约束；
- decision confidence；
- 结果揭示后 outcome_link。

这比保存 LLM 的自由文本“思考过程”更稳健，也避免把模型私有 chain-of-thought 当成审计证据。

---

## 5. LLM 与 KG 的合理交互设计

### 5.1 不让 LLM 直接写 SQL/Cypher

LLM 应调用稳定的领域工具，而不是看到数据库 schema 后自由生成查询。推荐的只读工具包括：

| 工具 | 输入 | 输出目的 |
|---|---|---|
| get_round_summary | round_id、scope | 当前轮事实、模型和假设概览 |
| explain_variant | variant_id、as_of_round | 单个候选的完整证据包 |
| compare_variants | variant_ids、criteria | 并排比较候选、反证和不确定性 |
| find_supporting_paths | claim、max_hops | 返回支持路径 |
| find_counterevidence | claim、scope | 主动寻找失败案例和冲突 |
| get_residue_effects | positions、background、assay | 带 support 和 epistasis caveat 的局部效应 |
| get_hypothesis_history | filters | 历史假设、测试结果和状态 |
| get_uncertainty_frontier | candidate_set、budget | exploitation/exploration 边界 |
| get_source_provenance | evidence_ids | 文献、结构、模型或实验来源 |

写接口应与查询接口分离，只暴露：

- propose_hypothesis；
- propose_evidence_link；
- propose_hypothesis_status_change；
- propose_curated_claim；
- propose_alias_merge。

任何写操作都先生成 ChangeProposal，再由 schema、权限、来源、重复、矛盾和时间校验器决定是否提交。

### 5.2 有限步 Agentic KG 循环

建议把当前“一次固定查询 + 一次 LLM 调用”升级为最多 2–3 次查询的有限循环：

~~~text
[优化目标 + 本轮预算 + 候选范围]
                |
                v
        LLM 生成 KGQueryPlan
                |
                v
     Policy/Schema/Leakage Validator
                |
                v
     KG 确定性执行一个领域工具
                |
                v
 EvidencePack：事实 + 路径 + 数值 + 反证 + 时效
                |
                v
       Sufficiency / Conflict Gate
           | 足够          | 不足且未超预算
           v               |
   Hypothesis/Decision     +--> 第二个领域查询
           |
           v
    Candidate Generator + Acquisition
           |
           v
       DecisionRecord
           |
           v
       Experiment Result
           |
           v
 Trusted Observation Ingest
           |
           v
 Hypothesis Evaluation + Derived Evidence Refresh
~~~

停止条件必须由控制器执行：

- 达到最大查询次数；
- EvidencePack 已满足最小证据覆盖；
- 新查询不再增加有效证据；
- 查询成本或 token 预算到达上限；
- 候选间没有可区分信息时，转为探索策略而不是继续问 LLM。

### 5.3 查询计划契约

~~~json
{
  "plan_id": "plan-r2-01",
  "objective": "选择 4 个最有希望且信息增益互补的 GB1 变体",
  "operations": [
    {
      "tool": "compare_variants",
      "args": {
        "variant_ids": ["FDGW", "FDGV", "VDGW"],
        "criteria": ["fitness", "uncertainty", "epistasis", "structure"]
      },
      "why": "区分可加性增益与组合风险"
    },
    {
      "tool": "find_counterevidence",
      "args": {
        "claim": "39:F and 54:W jointly increase fitness",
        "max_results": 6
      },
      "run_if": "first_result_has_insufficient_interaction_evidence"
    }
  ],
  "max_tool_calls": 2
}
~~~

Validator 应检查：

- tool 在 allow-list；
- variant 属于当前允许候选范围；
- as_of_round 满足无泄漏规则；
- max_rows、max_hops、max_calls 不超限；
- 不包含任意 SQL/Cypher；
- 不允许访问隐藏 final test 或 oracle labels。

### 5.4 EvidencePack：KG 给 LLM 的最小直接指引

KG 不应只返回三元组，也不应返回整个邻域。建议统一为：

~~~json
{
  "query_id": "kgq-r2-05",
  "as_of_round": 2,
  "scope": {
    "protein": "GB1",
    "assay": "IgG_binding",
    "positions": [39, 40, 41, 54]
  },
  "facts": [],
  "predictions": [],
  "supporting_paths": [],
  "counterevidence": [],
  "directional_signals": [
    {
      "target": "FDGW",
      "direction": "test_with_caution",
      "reason_code": "positive_single_site_negative_interaction_unknown",
      "strength": 0.31
    }
  ],
  "uncertainty": {
    "epistasis": "high",
    "measurement_support": 2,
    "ood": 0.31
  },
  "caveats": [],
  "provenance": [],
  "freshness": {
    "latest_observation_round": 1,
    "derived_at_round": 2
  }
}
~~~

真正能够“指引 LLM 选择正确思考方向”的不是更多文字，而是五项同时出现：

1. 支持方向；
2. 反证；
3. support 和不确定性；
4. 可追溯路径；
5. 明确的适用范围和 caveat。

LLM 的任务由“凭常识猜测”变为“在这些相互约束的证据中提出下一条可证伪假设”。

### 5.5 DecisionRecord 契约

~~~json
{
  "decision_id": "decision-r2-01",
  "round_id": 2,
  "selected_variants": ["FDGW", "FAGV", "VDAV", "YDGW"],
  "query_ids": ["kgq-r2-05", "kgq-r2-06"],
  "hypothesis_ids": ["hyp-r2-01"],
  "evidence_ids": ["ev-39F", "ev-54W", "ev-structure-FDGW"],
  "policy": {
    "name": "ucb",
    "beta": 1.5,
    "knowledge_weight": 0.3,
    "diversity_lambda": 0.2
  },
  "candidate_reasons": [
    {
      "variant_id": "FDGW",
      "role": "hypothesis_test",
      "main_support": ["ev-39F", "ev-54W"],
      "main_risk": "unmeasured epistasis"
    }
  ]
}
~~~

这使每个选择能够完整 replay，也能比较“没有 KG”“只用一次 KG”“Agentic KG”三种策略。

---

## 6. 每轮测试后如何动态优化 KG

### 6.1 写回不能等同于“LLM 修改图”

建议把更新权限分成三类：

| 来源 | 可写内容 | 信任级别 |
|---|---|---|
| ExperimentBackend / 受控数据导入 | Observation、Assay、Condition、QC | 最高 |
| 确定性计算模块 | Prediction、aggregate Evidence、Calibration、Conflict | 中高，可重算 |
| LLM Agent | Hypothesis、Claim/Link/Status ChangeProposal | 待验证 |

### 6.2 动态更新流水线

~~~text
实验结果
  |
  v
1. Ingestion Validator
   - variant/assay/round 是否存在
   - 数值、单位、重复、QC 是否合法
   - idempotency key / data hash 是否重复
  |
  v
2. Append immutable Observation
  |
  v
3. Derived Knowledge Updater
   - 更新位点/残基统计
   - 更新 pairwise/背景条件 epistasis
   - 更新 predictor calibration
   - 生成新的 supporting/contradicting Evidence
  |
  v
4. Hypothesis Evaluator
   - 读取 falsification_rule
   - 链接 tested_by Observation
   - 状态变更为 supported / contradicted / inconclusive
  |
  v
5. Temporal/Provenance Commit
   - 保留旧事实
   - 设置 valid_to 或 supersedes
   - 记录 updater version、input snapshot、transaction id
  |
  v
6. Next-round summary/materialized views refresh
~~~

### 6.3 ChangeProposal 契约

~~~json
{
  "proposal_id": "kgcp-r3-02",
  "actor": "scientist_agent",
  "operation": "change_hypothesis_status",
  "target_id": "hyp-r2-01",
  "proposed_value": "contradicted",
  "basis": {
    "observation_ids": ["obs-FDGW-r2"],
    "rule": "measured fitness below baseline and both single-site parents"
  },
  "confidence": 0.94,
  "expected_effect": "avoid treating 39:F and 54:W as additive",
  "idempotency_key": "hyp-r2-01:obs-FDGW-r2:v1"
}
~~~

校验器通过后返回：

~~~json
{
  "proposal_id": "kgcp-r3-02",
  "status": "committed",
  "transaction_id": "kgtx-00918",
  "created_evidence_ids": ["ev-negative-epistasis-FDGW"],
  "updated_entities": ["hyp-r2-01"],
  "rejected_fields": []
}
~~~

### 6.4 时间建模

建议至少同时记录：

- event_time：实验或知识在现实中发生的时间；
- ingested_at：系统何时收到；
- valid_from_round / valid_to_round：结论在哪些轮次有效；
- system_version：哪个 updater、规则或模型产生。

Graphiti/Zep 的 temporal KG 工作和开源 Graphiti 对 episode、valid_at、invalid_at、增量整合和来源追溯的实现很有启发，但其论文目前属于预印本/工程系统证据，不应把作者报告的性能直接当作本项目效果保证。可借鉴时间模型，不必在第一阶段整体替换当前 SQLite。[论文](https://arxiv.org/abs/2501.13956)；[Graphiti 代码](https://github.com/getzep/graphiti)

### 6.5 防止动态图自我污染

必须加入以下保护：

- LLM 自己生成的 Claim 不能再次被当成独立 Evidence；
- 支持数必须按独立 Observation、独立实验或独立来源计数；
- 相同来源派生出的多个 Evidence 需要 source-group 去重；
- Hypothesis 被实验支持前，不能晋升为 curated fact；
- 图中所有统计都绑定 assay、condition 和背景；
- 历史结果永不物理覆盖；
- 定期检测 graph bloat、孤立节点、重复实体和失效证据；
- 检索时默认优先当前有效事实，同时允许显式查询历史。

---

## 7. 相关论文与 GitHub 代码：可借鉴什么

### 7.1 文献综合

| 工作 | 核心机制 | 对当前项目的启发 | 官方代码与复用判断 |
|---|---|---|---|
| Think-on-Graph，ICLR 2024 | LLM 在 KG 上迭代 beam search，对关系和实体评分、剪枝并判断是否停止 | 给 KG 查询增加有限深度、beam 和 stop gate；适合“先探索哪种证据关系” | [论文](https://openreview.net/forum?id=nnVO1PvbTv)；[代码](https://github.com/DataArcTech/ToG)。研究脚本与 Freebase/Wikidata 强耦合；检查快照未见独立 LICENSE 文件，README 虽声明 Apache 2.0，复制前仍需法务确认。建议只借控制流程 |
| Reasoning on Graphs，ICLR 2024 | planning → retrieval → reasoning；LLM 先给关系路径计划，图执行器只返回 KG 中真实存在的路径 | 最适合当前项目：LLM 只规划 typed relation path，确定性工具执行，再由 LLM形成假设 | [论文](https://openreview.net/pdf?id=ZGNWW7xZ6Q)；[代码](https://github.com/RManLuo/reasoning-on-graphs)。MIT；可借接口和路径执行思想，但原实现面向 Freebase KGQA/Hugging Face |
| MindMap，ACL 2024 | 提取实体，查询最短路径和邻域，将多条证据路径组织成可解释 inference map | EvidencePack 同时提供 path evidence 和 neighbor/counter evidence；适合生成可视化决策依据 | [论文](https://aclanthology.org/2024.acl-long.558/)；[代码](https://github.com/wyl-willing/MindMap)。实现较单体且检查快照无 LICENSE，不直接复制 |
| DALK，Findings EMNLP 2024 | LLM 从新文献构建演化疾病 KG；KG 再通过 coarse-to-fine/self-aware retrieval 增强 LLM | 对“外部文献知识层 + 每轮实验知识层”双向增强最直接；支持动态加入新结构化知识 | [论文](https://aclanthology.org/2024.findings-emnlp.119/)；[代码](https://github.com/David-Li0406/DALK)。脚本含手工 key/Neo4j 配置并基于 MindMap fork，检查快照无 LICENSE；只借流程 |
| Biomedical KG-RAG，Bioinformatics 2024 | 实体识别与链接、SPOKE 邻域、类型过滤、embedding pruning、最小 schema、来源和统计证据 | EvidencePack 应最小化且保留 provenance、统计和 source type；避免整图/整 schema 进入 prompt | [论文](https://academic.oup.com/bioinformatics/article/40/9/btae560/7759620)；[代码](https://github.com/BaranziniLab/KG_RAG)。Apache-2.0；可选择性参考检索和 context pruning，疾病/SPOKE 逻辑不能直接复用 |
| HippoRAG，NeurIPS 2024 | OpenIE 构图、文档/实体/事实关联、Personalized PageRank 检索，持续整合外部文档 | 可作为未来“文献长期记忆 sidecar”，根据实体激活跨文档证据；不应替代实验 truth store | [论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf)；[代码](https://github.com/OSU-NLP-Group/HippoRAG)。MIT；模块和测试较完整，但依赖较重 |
| KG-Agent，2024 preprint | LLM、工具箱、KG executor 和 knowledge memory 组成迭代循环，选择工具并更新记忆 | 直接支持“查询—思考—再查询”；适合定义 agent controller 的职责 | [论文](https://arxiv.org/abs/2402.11163)。本次未核验到作者可用的正式代码发布，因此只作概念参考 |
| ReKG-MCTS，Findings ACL 2025 | 用 UCB 选择、图约束扩展、LLM rollout、价值回传做 training-free KG 路径搜索 | 与本项目已有 UCB acquisition 思路相容；未来可给复杂关系搜索加入探索/利用控制 | [论文](https://aclanthology.org/2025.findings-acl.484/)；[代码](https://github.com/ShawnKS/rekgmcts)。适合借搜索控制思想，现阶段不需要完整 MCTS |
| BYOKG-RAG，EMNLP 2025 | LLM 生成实体、候选答案、关系路径和 OpenCypher 等 artifacts；专用图工具完成 linking、triplet/path/query retrieval 与 verbalization | 对自定义 fitness KG 的可插拔接口最有参考价值；EntityLinker、Traversal、PathRetriever、Verbalizer 可对应拆层 | [论文](https://aclanthology.org/2025.emnlp-main.1417/)；[代码](https://github.com/awslabs/graphrag-toolkit)。Apache-2.0 + NOTICE；接口很值得参考，但 AWS/云图依赖需隔离 |
| Graphiti / Zep temporal KG，2025 preprint + 持续开发 | 增量 episode、实体关系、有效时间、失效而不删除、混合检索和 provenance | 适合实现历史轮次、过时假设、纠正链和实时写回 | [论文](https://arxiv.org/abs/2501.13956)；[代码](https://github.com/getzep/graphiti)。Apache-2.0；时间数据模型可复用，整体引入需评估图数据库和 LLM 抽取成本 |

### 7.2 代码快照核查

为避免只看 README，本次将官方仓库克隆到工作区之外的临时目录，并检查核心调用链。检查快照如下：

| 仓库 | 检查 commit | 主要检查内容 |
|---|---|---|
| ToG | 7ccbb92e17579f934bb778386230de47eca0ab67 | main_freebase.py、relation/entity search、prune、reasoning stop |
| RoG | ccf8ec847bf61005a1b27cc9e5aff5c8ead7a24b | gen_rule_path、graph_utils、build_qa_input、predict_answer |
| DALK | bcdedb387fb4336b26953ccf0efbd46d3534e847 | llm2kg、MindMap_revised、Neo4j 查询 |
| KG_RAG | 01b9f6e6414221eb186a030e3c103e4ec620ab0f | entity recognition、SPOKE context、embedding pruning |
| MindMap | 0411a54719e9b90176d02e937ab73d2df5e25963 | shortest path、neighbor retrieval、prompt packaging |
| HippoRAG | c617143f01477243992a63b2e2151cc003dd3b21 | OpenIE、incremental index、PPR retrieval |
| Graphiti | b2ff2eadd9a6b75a261a5cf0b19557883a13f752 | add_episode、search、valid_at/invalid_at、drivers |
| BYOKG-RAG | f438df318682676a47401194843370590f8d9b43 | QueryEngine、EntityLinker、AgenticRetriever、PathRetriever、GraphQueryRetriever |

代码快照仅用于可行性评估，不代表已经将这些代码复制进当前项目。

### 7.3 建议的复用等级

#### 可优先复用或适配

1. **BYOKG-RAG 的抽象层次**  
   参考 EntityLinker、GraphTraversal、Triplet/Path Retriever、Verbalizer、QueryEngine 的拆分；本项目实现自己的 SQLite adapter，避免直接绑定 AWS 服务。

2. **RoG 的关系计划—执行结构**  
   将自然语言目标先变成受控 QueryPlan，再由 KG 工具执行；不要让 LLM 自由拼 SQL。

3. **KG-RAG 的 context pruning**  
   把高相关、带出处、带统计的路径打包为 EvidencePack，并用硬上限控制 token。

4. **Graphiti 的时间字段和 episode/provenance 模型**  
   第一阶段可以只复用数据思想；如果后续跨 campaign 长期记忆明显增大，再做 sidecar 或存储替换评估。

#### 可作为可选 sidecar

- HippoRAG：管理论文摘要、实验日志和非结构化知识，通过实体/PPR 找跨文档证据；
- Graphiti：管理跨轮次、跨 campaign 的时间记忆；
- Microsoft GraphRAG 一类 corpus/community summarization：适合大规模文献全局问答，不适合代替当前实验事实图。

#### 暂不直接复用

- ToG：Freebase/Wikidata 研究脚本耦合高，且许可证文件需进一步确认；
- DALK/MindMap：思路有价值，但仓库脚本化、凭证配置和许可证边界不适合作为生产依赖；
- KG-Agent：没有核验到可用的作者正式代码；
- ReKG-MCTS：当前只有四个位点和有限工具，完整 MCTS 会增加调用成本；等关系和外部知识层显著扩展后再考虑。

---

## 8. 面向当前仓库的目标模块

以下是后续实现方向，本次没有创建或修改这些代码：

~~~text
src/fitness_agents/knowledge/
├── contracts.py
│   ├── KGQueryPlan
│   ├── EvidencePack
│   ├── DecisionRecord
│   ├── KGChangeProposal
│   └── KGUpdateResult
├── query_gateway.py
│   ├── allow-listed tool registry
│   ├── visibility / budget validator
│   └── query audit
├── retrievers/
│   ├── variant_explainer.py
│   ├── comparison.py
│   ├── path_search.py
│   ├── counterevidence.py
│   └── uncertainty_frontier.py
├── evidence_pack.py
│   ├── source deduplication
│   ├── conflict detection
│   ├── token / row budget
│   └── provenance packaging
├── write_gateway.py
│   ├── proposal validator
│   ├── permission policy
│   └── transactional commit
└── updaters/
    ├── observation_ingest.py
    ├── derived_evidence.py
    ├── hypothesis_evaluator.py
    └── temporal_validity.py

src/fitness_agents/agents/
└── kg_reasoning_controller.py
    ├── plan
    ├── tool call
    ├── sufficiency gate
    ├── hypothesis
    └── decision record
~~~

### 8.1 与现有 LLM API 的兼容方式

现有 LLM client 已经通过严格 JSON schema 生成 Hypothesis，这一点应保留。升级时可将一次 generate_hypothesis 拆成三个结构化能力：

1. plan_kg_queries：输出 KGQueryPlan；
2. generate_hypothesis：输入 EvidencePack，输出 Hypothesis；
3. propose_kg_changes：输入实验结果和旧假设，输出 ChangeProposal。

底层 LLM provider 继续走统一客户端，Agent controller 不依赖特定模型。这也便于与 PG-LLM 类调用策略对接和在相同任务/提示/输出契约下做后续测试。

---

## 9. 实施优先级

### P0：先定义契约和不可违反的规则

- 固定 Observation、Prediction、Evidence、Hypothesis、Decision 的 schema；
- 定义 as_of_round 与 final-test 隔离；
- 定义 Evidence polarity、scope、source group 和 provenance；
- 为 ChangeProposal 定义权限矩阵；
- 用 JSON fixture 写契约测试。

### P1：在当前 SQLite 上增加 EvidencePack facade

- 不换数据库；
- 把 hypothesis_context 和 explain_variant 统一成 EvidencePack；
- 新增 compare_variants、find_counterevidence、get_hypothesis_history；
- 增加查询预算、行数、hop 和 token 上限；
- 所有查询保持审计记录。

### P2：增加最多两步的 Agentic KG 查询

- LLM 先输出 QueryPlan；
- 执行首个工具；
- 只有证据不足时允许第二个工具；
- 每次都要求支持证据与反证；
- 最终仍输出当前 Hypothesis schema 的向后兼容版本。

### P3：动态 Hypothesis 评估和安全写回

- 新 Observation 到达后自动产生 support/contradict Evidence；
- 更新假设状态；
- 记录 tested_by、supersedes、valid_to_round；
- LLM 只提交 ChangeProposal，不能直接改 Observation。

### P4：扩展科学知识层

- 序列与结构位置映射；
- PDB/UniProt/文献出处；
- 原子接触、溶剂暴露、二级结构、能量/稳定性指标；
- assay condition、背景依赖和 pairwise epistasis；
- 跨 campaign 可复用、但严格按 assay/context 限定的知识。

### P5：评估外部组件

- 若文献数据成为主要知识源，试验 HippoRAG sidecar；
- 若跨轮次时间关系显著复杂，试验 Graphiti adapter；
- 若换用通用图数据库，参考 BYOKG-RAG 的 linker/traversal/verbalizer；
- 只有多跳关系空间明显增大后，再评估 ToG/ReKG-MCTS 搜索。

---

## 10. 如何验证升级确实有效

### 10.1 离线对照

在相同数据划分、随机种子和实验预算下比较：

| 组 | 策略 |
|---|---|
| A | Predictor + acquisition，无 KG |
| B | 当前固定 hypothesis_context |
| C | EvidencePack，但无多步查询 |
| D | 有限两步 Agentic KG 查询 |
| E | D + 动态 Hypothesis 状态和反证回写 |

### 10.2 主要指标

#### Fitness 优化指标

- best observed fitness；
- cumulative/max regret；
- top-k hit rate；
- 在固定实验预算下达到阈值的轮次；
- batch diversity；
- 新发现相对 WT 和当前 best 的增益。

#### 预测与决策指标

- calibration、NLL/coverage、OOD 分层误差；
- Hypothesis 的支持率、反证率和校准；
- 选择记录中有效 Evidence 覆盖率；
- counterevidence recall；
- evidence path faithfulness；
- 无效或越权 KG 写入率。

#### 系统指标

- 每轮 LLM 调用数、token、延迟和失败率；
- KG 查询成功率、空结果率；
- query replay 一致性；
- 同一输入在不同 LLM provider 下的结构兼容性；
- 动态更新后的 stale fact 命中率和 graph growth。

### 10.3 必须保留的消融

- evidence deletion；
- score shuffle；
- knowledge ablation；
- 不使用 counterevidence；
- 不更新 hypothesis status；
- 只使用位点统计 vs 加入 pairwise epistasis；
- 一步查询 vs 两步查询；
- 不同 knowledge_weight。

最关键的问题不是“回答写得是否更像专家”，而是：

> 在不泄漏隐藏标签、相同实验预算和相同 Predictor 下，KG 是否让系统更快找到更高 fitness，同时保持决策可追溯和假设可被反证？

---

## 11. 风险与控制

| 风险 | 后果 | 控制 |
|---|---|---|
| 小样本关联被解释为因果 | LLM 过早收敛到错误方向 | shrinkage、support、counterevidence、背景限定、探索配额 |
| epistasis 被位点均值掩盖 | 组合候选失败 | pairwise/conditional evidence、完整变体路径、明确 caveat |
| LLM 自证循环 | 自己生成的主张重复增强自己 | source-group 去重；LLM Claim 不计独立 Evidence |
| 当前轮/最终测试泄漏 | 评测失真 | as_of_round 强制校验、不可见字段黑名单、replay 测试 |
| 动态写回污染事实层 | KG 长期失真 | proposal/validate/commit；Observation 只由可信后端写入 |
| 过时事实仍被检索 | 决策使用旧结论 | valid_from/valid_to、默认 current view、可显式查历史 |
| KG 过度膨胀 | 检索噪声、成本上升 | salience、归档、物化 summary、去重和维护任务 |
| 外部代码许可证不清 | 合规风险 | 仅复用明确许可证代码；无 LICENSE 项目只借论文思想 |
| 框架过重 | 研发复杂度超过收益 | 先 SQLite facade；达到规模阈值后才引入图数据库/sidecar |

---

## 12. 最终建议

当前 KG 的方向是正确的，但它目前更像“实验记忆 + 证据缓存 + 一次性 LLM 上下文”，还不是完整的外挂结构化知识库。

近期最有价值的优化顺序是：

1. 先规范 Observation、Prediction、Evidence、Hypothesis、Decision；
2. 在现有 SQLite 上实现统一 EvidencePack；
3. 增加 compare、counterevidence、history 三个领域工具；
4. 将 LLM 交互改成最多两步的受限 QueryPlan；
5. 新结果到达后自动评价 Hypothesis，并用时间化、可追溯方式写回；
6. 只有在文献和跨 campaign 知识达到一定规模后，再引入 Graphiti、HippoRAG 或通用图数据库。

推荐的总体边界是：

~~~text
LLM：提出问题、制定有限查询计划、形成可证伪假设
KG：保存分层知识、执行确定性查询、返回路径/反证/不确定性
Predictor：给出数值预测与校准不确定性
Acquisition：在预算和多样性约束下做最终选择
ExperimentBackend：产生唯一可晋升为事实的真实结果
Validator：控制可见性、权限、时间和写回一致性
~~~

这能让 KG 真正“指导 LLM 朝正确方向思考”，同时避免让 LLM 成为事实数据库的直接管理员。

---

## 参考文献与代码

1. Sun, J. et al. Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph. ICLR 2024. [论文](https://openreview.net/forum?id=nnVO1PvbTv)；[代码](https://github.com/DataArcTech/ToG)。
2. Luo, L. et al. Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning. ICLR 2024. [论文](https://openreview.net/pdf?id=ZGNWW7xZ6Q)；[代码](https://github.com/RManLuo/reasoning-on-graphs)。
3. Wen, Y., Wang, Z. & Sun, J. MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models. ACL 2024. DOI: 10.18653/v1/2024.acl-long.558. [论文](https://aclanthology.org/2024.acl-long.558/)；[代码](https://github.com/wyl-willing/MindMap)。
4. Li, D. et al. DALK: Dynamic Co-Augmentation of LLMs and KG to answer Alzheimer’s Disease Questions with Scientific Literature. Findings of EMNLP 2024. DOI: 10.18653/v1/2024.findings-emnlp.119. [论文](https://aclanthology.org/2024.findings-emnlp.119/)；[代码](https://github.com/David-Li0406/DALK)。
5. Soman, K. et al. Biomedical knowledge graph-optimized prompt generation for large language models. Bioinformatics 40, btae560 (2024). DOI: 10.1093/bioinformatics/btae560. [论文](https://academic.oup.com/bioinformatics/article/40/9/btae560/7759620)；[代码](https://github.com/BaranziniLab/KG_RAG)。
6. Gutiérrez, B. J. et al. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. NeurIPS 2024. [论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf)；[代码](https://github.com/OSU-NLP-Group/HippoRAG)。
7. Jiang, J. et al. KG-Agent: An Efficient Autonomous Agent Framework for Complex Reasoning over Knowledge Graph. arXiv:2402.11163 (2024). [论文](https://arxiv.org/abs/2402.11163)。
8. Song, X., Zhang, S. & Yu, T. ReKG-MCTS: Reinforcing LLM Reasoning on Knowledge Graphs via Training-Free Monte Carlo Tree Search. Findings of ACL 2025. DOI: 10.18653/v1/2025.findings-acl.484. [论文](https://aclanthology.org/2025.findings-acl.484/)；[代码](https://github.com/ShawnKS/rekgmcts)。
9. Mavromatis, C. et al. BYOKG-RAG: Multi-Strategy Graph Retrieval for Knowledge Graph Question Answering. EMNLP 2025. DOI: 10.18653/v1/2025.emnlp-main.1417. [论文](https://aclanthology.org/2025.emnlp-main.1417/)；[代码](https://github.com/awslabs/graphrag-toolkit)。
10. Rasmussen, P. et al. Zep: A Temporal Knowledge Graph Architecture for Agent Memory. arXiv:2501.13956 (2025). [论文](https://arxiv.org/abs/2501.13956)；[Graphiti 代码](https://github.com/getzep/graphiti)。

---

## AI 辅助研究说明

本报告使用 AI 辅助完成代码审计、检索式扩展、论文与代码仓库对照、架构综合和文档撰写。关键论文信息均回到正式论文页面核验；代码复用结论基于上述固定 commit 的只读检查。所有示例 fitness 数值均为说明性构造，不应作为真实生物结论或模型性能结果。
