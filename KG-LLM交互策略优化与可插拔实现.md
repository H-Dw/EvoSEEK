# KG–LLM 交互策略优化与可插拔实现

> 研究与实现日期：2026-08-15  
> 范围：fitness-agents 中 KG 查询、LLM 推理、fitness 优化建议生成与受控写回  
> 状态：已新增独立接口骨架、消融配置与单元测试；未接入当前默认 orchestrator

## 0. 核心结论

当前系统的 KG 已经参与决策，但主要形态是“固定查询一次、将摘要注入 LLM、再把结果用于候选筛选”。它还不是一个可由 LLM 按需、多步、可验证地调用的外挂知识工具。

推荐把目标交互收敛为以下闭环：

```text
任务目标
  -> LLM/规则生成受限 KGQueryPlan
  -> Policy Gateway 校验轮次、候选范围、工具白名单和预算
  -> 可插拔 QueryOperator 确定性执行
  -> 返回最小 EvidencePack（事实、路径、反证、出处、局限）
  -> Evidence Sufficiency 判断是否需要继续查询
  -> LLM 生成结构化 Hypothesis / FitnessRecommendation / ChangeProposal
  -> 非 LLM 规则完成候选约束、acquisition、diversity 和最终选择
  -> Proposal Gateway 校验写回
  -> 新实验 Observation 触发假设评估和下一轮知识更新
```

关键边界是：

- LLM 负责语义规划、提出假设和解释；
- KG 负责确定性查询、路径、时间边界、来源和历史；
- Predictor 与 acquisition policy 负责数值估计和最终排序；
- ExperimentBackend 是 Observation 的唯一可信写入者；
- LLM 不获得任意 SQL/Cypher，也不能直接修改实验事实或模型输出。

本次已实现一套轻量、无新增运行依赖的 `kg_interaction` 包。其设计吸收了 RoG、KAG、BYOKG-RAG、KG-RAG、G-Retriever 和 Graphiti 的可复用模式，但没有直接搬入重型框架，也没有把新模块接到现有主循环，因此当前基线行为不变。

## 1. 当前系统与目标系统的差距

### 1.1 当前 KG 如何进入 LLM 和 fitness 决策

当前路径可概括为：

```text
Visible Observation
  -> KG / KnowledgeEngine
  -> Prediction + physchem/conservation/structure/kg Evidence
  -> 固定 hypothesis_context
  -> 单次 LLM generate_hypothesis
  -> preferred_residues 约束候选
  -> acquisition score + knowledge score + diversity
  -> 实验批次
```

它已经具备三个有价值的基础：

1. Observation、Prediction、Evidence、Hypothesis 类型分开；
2. 查询结果带 `as_of_round`，并有轮次可见性控制；
3. `explain_variant` 已经能够返回单候选证据，只是尚未进入主推理循环。

主要缺口为：

| 缺口 | 当前影响 | 目标状态 |
|---|---|---|
| 查询固定 | LLM 不能根据不足信息追问 | 显式、多步、有限预算的 QueryPlan |
| 返回值面向 prompt | 不同查询输出难统一 | 统一 EvidencePack |
| 缺少反证要求 | 易形成确认偏差 | 支持证据与反证独立字段和消融开关 |
| 没有充分性判断 | 要么一次不足，要么无限查询 | 可替换 EvidenceSufficiencyPolicy |
| LLM 写回边界不完整 | 容易把主张写成事实 | propose → validate → commit |
| 假设无自动状态机 | 新结果不会直接修正旧知识 | TESTED_BY、SUPPORTED/CONTRADICTED、SUPERSEDES |
| 工具与模型耦合 | 难做消融和 PG-LLM 测试 | provider-neutral LLM + registry-based KG tools |

## 2. 文献与 GitHub 代码给出的可借鉴模式

本次先检索论文正式页面，再检查作者或官方 GitHub 仓库。下表只列与本系统直接相关的设计点。

| 工作 | 论文结论或代码模式 | 对本项目的复用判断 |
|---|---|---|
| [RoG, ICLR 2024](https://openreview.net/forum?id=ZGNWW7xZ6Q) / [代码](https://github.com/RManLuo/reasoning-on-graphs) | 将关系路径规划、图上确定性检索、答案推理分离 | 高度适合；复用“plan → retrieve → reason”边界，不复制其数据集专用代码 |
| [ToG, ICLR 2024](https://openreview.net/forum?id=nnVO1PvbTv) / [代码](https://github.com/DataArcTech/ToG) | LLM 引导逐跳图搜索，并按预算停止 | 适合受限探索；当前只实现预算与早停，不引入自由图遍历 |
| [BYOKG-RAG, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1417/) / [代码](https://github.com/awslabs/graphrag-toolkit) | 实体、候选答案、路径、图查询等多策略检索器可组合，LLM 可迭代修正链接 | 高度适合 Operator Registry；后续可将结构、文献、历史分别做成 retriever |
| [KAG](https://arxiv.org/abs/2409.13731) / [代码](https://github.com/OpenSPG/KAG) | 逻辑形式规划器、DAG 任务、多个 executor/operator 和知识—文本互索引 | 高度适合接口设计；完整 OpenSPG 栈过重，不作为当前依赖 |
| [KG-RAG, Bioinformatics 2024](https://academic.oup.com/bioinformatics/article/40/9/btae560/7759620) / [代码](https://github.com/BaranziniLab/KG_RAG) | 实体识别、有限邻域、语义裁剪、最小 prompt context、出处和统计证据 | 直接用于 EvidencePack 和 token 预算设计 |
| [G-Retriever, NeurIPS 2024](https://openreview.net/forum?id=MPJ3oXtTZl) / [代码](https://github.com/XiaoxinHe/G-Retriever) | 用 prize-collecting Steiner tree 选择相关子图 | 可作为高级子图检索插件；依赖 GNN/PyTorch/PCST，暂不进入 MVP |
| [GCR](https://arxiv.org/abs/2410.13080) / [代码](https://github.com/RManLuo/graph-constrained-reasoning) | 通过 KG-Trie 将生成限制在合法图路径 | 适合作为“结构约束解码”高级消融；模型侵入性较强，暂不直接复用 |
| [ReKG-MCTS, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.484/) / [代码](https://github.com/ShawnKS/rekgmcts) | MCTS 平衡图路径探索与利用 | 只在多跳空间显著增大后引入；当前蛋白局部图不值得承担复杂度 |
| [Graphiti / Zep](https://arxiv.org/abs/2501.13956) / [代码](https://github.com/getzep/graphiti) | episode、来源、事实有效期、增量更新、混合检索 | 高度适合历史和动态写回语义；不要求现在更换 SQLite |

代码检查采用固定快照以便重现：RoG `ccf8ec8`、BYOKG-RAG `f438df3`、KAG `fdab15b`、KG-RAG `01b9f6e`、GCR `9518e8e`、Graphiti `b2ff2ea`。RoG/GCR 为 MIT，BYOKG-RAG、KAG、KG-RAG、Graphiti 为 Apache-2.0。ToG 仓库快照未发现根目录 LICENSE，因此只借鉴论文策略，不复制代码。

### 2.1 为什么不直接选择一个框架整体移植

这些仓库主要面向通用 KGQA、文本问答或大规模图推理，目标函数和本项目不同。fitness 优化必须额外满足：

- 轮次可见性和最终测试标签隔离；
- Observation 与 Prediction 的严格类型隔离；
- 少量真实测量、较高噪声和强 epistasis；
- 查询结果必须能进入 acquisition，而不是只生成自然语言答案；
- 每个模块需要独立消融；
- PG-LLM 评测时必须冻结 KG、禁止结果写回。

因此最稳妥的方案是复用接口模式和算法思想，保留当前领域契约，逐个引入可测插件。

## 3. 目标 KG–LLM 交互架构

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Fitness Agent                                                      │
│  objective + candidate scope + round snapshot                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            v
┌─────────────────────────────────────────────────────────────────────┐
│ Query Planner                                                       │
│  KGQueryPlan[step_id, operator, intent, args, depends_on, rationale]│
└───────────────────────────┬─────────────────────────────────────────┘
                            v
┌─────────────────────────────────────────────────────────────────────┐
│ KG Interaction Controller                                          │
│  allow-list | round/scope guard | call budget | dependency | stop  │
└───────────────┬─────────────────────────────┬───────────────────────┘
                v                             v
       QueryOperator Registry       EvidenceSufficiencyPolicy
       context / explain /          facts + paths + counterevidence
       compare / paths / history
                |
                v
┌─────────────────────────────────────────────────────────────────────┐
│ Read-only KG Gateway                                                │
│  current SQLite KG now; external multimodal KG later               │
└───────────────────────────┬─────────────────────────────────────────┘
                            v
┌─────────────────────────────────────────────────────────────────────┐
│ EvidencePack                                                        │
│ facts | predictions | evidence | paths | counterevidence | caveats │
│ provenance | query_id | as_of_round                                │
└───────────────────────────┬─────────────────────────────────────────┘
                            v
                LLM structured recommendation
                            |
              ┌─────────────┴─────────────┐
              v                           v
    Acquisition/selection          KGChangeProposal
    deterministic scoring          validate -> dry-run/commit
```

### 3.1 QueryPlan：让 LLM 计划，但不让它自由访问数据库

每个查询步骤至少包含：

- `operator`：已注册的领域操作，而非 SQL/Cypher；
- `intent`：context、explain、compare、support、counterevidence、history 或 uncertainty；
- `arguments`：受 schema 校验的参数；
- `depends_on`：只有依赖步骤成功才执行；
- `rationale`：用于审计的简短查询理由；
- `max_tool_calls`：硬预算。

推荐由两类 Planner 并存：

1. `TemplatePlanner`：规则生成，作为可复现基线；
2. `LLMPlanner`：根据目标产生结构化计划，失败时回退模板。

两者输出同一 `KGQueryPlan`，因此可以直接做 planner 消融。

### 3.2 QueryOperator：每类知识一个插件

MVP 操作符：

| Operator | 输入 | 输出重点 | 当前状态 |
|---|---|---|---|
| `hypothesis_context` | round、limit | 全局观测摘要、候选预测、历史假设 | 已实现适配 |
| `explain_variant` | variant_id、round | 单候选测量、预测、正负证据 | 已实现适配 |
| `compare_variants` | variant_ids、round | 同一结构下的横向比较 | 已实现适配 |
| `find_supporting_paths` | claim、hop limit | 支持路径和端点出处 | 预留插件 |
| `find_counterevidence` | claim、scope | 冲突事实、负向结果、失效关系 | 预留插件 |
| `get_history` | entity/claim、time range | 版本、状态变化、测试结果 | 预留插件 |
| `retrieve_structure_context` | positions、radius | 局部结构/原子相互作用 | 预留插件 |
| `retrieve_evolution_context` | positions | 保守性、共进化、同源证据 | 预留插件 |
| `retrieve_literature_claims` | entity/claim | 文献 Claim 及适用范围 | 预留插件 |

所有 Operator 返回同一种 `EvidencePack`。这样可以替换检索算法，而不改变 LLM 或 orchestrator。

### 3.3 EvidencePack：向 LLM 提供可行动的最小证据

推荐字段为：

```text
EvidencePack
  query_id, operator, as_of_round
  facts[]                 # 真实观测和稳定图事实
  predictions[]           # 明确标注模型版本和不确定性
  evidence[]              # 计算/文献/结构证据
  supporting_paths[]      # 有端点、有关系、有来源的短路径
  counterevidence[]       # 反例和冲突
  directional_signals[]   # 支持“往哪里试”的结构化信号
  caveats[]               # epistasis、低支持数、映射不确定等
  provenance[]            # source_id、method/version、query receipt
```

不要将整个子图序列化进 prompt。KG-RAG 的结果表明，对图上下文先裁剪再注入可以显著减少 token；本项目更应采用“每个候选最多 N 条证据、每个 claim 最多 K 条路径”的显式预算。

### 3.4 Evidence Sufficiency：查询何时停止

停止条件不应由 LLM 一句话决定，而应由可替换策略计算：

```text
sufficient =
  fact_count >= min_fact_count
  AND supporting_path_count >= min_path_count
  AND (counterevidence_present OR counterevidence_ablation=true)
```

未来可以增加：来源独立数、最低置信度、候选覆盖率、token 成本和时间成本。该策略本身是消融轴：固定一跳、两步查询、证据充分早停、无早停全计划。

## 4. Fitness 优化建议的提出流程

目标不是让 LLM 直接给出一个“最佳突变”，而是让它生成可检验且能被数值决策层使用的结构化建议。

### 4.1 推荐的完整步骤

```text
1. Predictor 生成候选 mean/std/OOD/component scores
2. Rule Planner 获取 round summary
3. 根据高潜力、高不确定或证据冲突筛出需要解释的候选
4. compare_variants 做候选横向比较
5. 对领先候选主动查询 counterevidence
6. 如信息仍不足，查询结构/进化/文献插件
7. 组装 EvidencePack，不暴露隐藏标签
8. LLM 输出结构化 FitnessRecommendation
9. Validator 检查引用、范围、可证伪性和已知反证覆盖
10. Recommendation 转换为候选约束或弱先验
11. acquisition + uncertainty + diversity 产生最终批次
12. 保存 DecisionRecord 和使用过的 query/evidence IDs
13. 实验 Observation 写回并评估 Hypothesis
```

### 4.2 FitnessRecommendation 建议契约

```json
{
  "recommendation_id": "rec:r3:001",
  "scope": {"protein_id": "...", "assay_id": "...", "round_id": 3},
  "candidate_ids": ["v17", "v42"],
  "direction": "prioritize|deprioritize|explore",
  "claim": "...",
  "expected_effect": {"metric": "fitness", "direction": "increase", "baseline": "WT"},
  "supporting_evidence_ids": ["..."],
  "counterevidence_ids": ["..."],
  "uncertainty": "low|medium|high",
  "falsification_rule": "...",
  "proposed_test": {"variants": ["..."], "comparison": "..."},
  "caveats": ["..."],
  "query_ids": ["..."]
}
```

建议只能改变三个受控量：

- candidate eligibility；
- knowledge prior/bonus 的有限权重；
- batch 中 exploitation 与 exploration 的配额。

它不应覆盖 Predictor 输出，也不应绕过 diversity、预算和安全约束。这样即使 LLM 失误，影响仍被限制在可解释的决策层。

### 4.3 正反证对称原则

每个“优先”建议至少回答：

1. 哪些事实支持？
2. 哪些事实反对或限制外推？
3. 证据是否来自独立来源？
4. 结论适用于哪个 assay、condition、background 和 round？
5. 用哪一组最小实验可以证伪？

如果没有反证，不代表不存在反证；应记录 `counterevidence_search_performed=true/false`。这能区分“查过但没有”与“没有查”。

## 5. 动态写回：LLM 可以建议修改，不能直接改事实

### 5.1 两阶段写回

```text
LLM -> KGChangeProposal
        |
        v
Schema / scope / evidence / authority / idempotency validation
        |
        +-> rejected
        +-> dry_run
        +-> needs_review
        +-> committed as a new version
```

允许的 LLM 操作：

- 新建 Hypothesis 草稿；
- 将现有 Evidence 链接到 Hypothesis；
- 提议 Hypothesis 状态变化；
- 新建低权威的 curated/agent Claim；
- 提议别名合并。

禁止的 LLM 操作：

- 新建或修改 Observation；
- 把 Prediction 改成 Observation；
- 删除历史记录；
- 无 Evidence ID 地宣告假设“已证实”；
- 直接运行 SQL/Cypher/SPARQL；
- 读取 final/oracle 或超出当前轮可见范围的数据。

### 5.2 新结果如何优化 KG

每轮实验完成后应执行非 LLM 的 `HypothesisEvaluator`：

```text
new Observation
  -> 找到 TESTED_BY 指向该实验的 Hypothesis
  -> 执行结构化 falsification_rule
  -> 产生 EvaluationEvidence
  -> supported / contradicted / inconclusive
  -> 旧版本 valid_to_round 关闭
  -> 如需修订，新 Hypothesis SUPERSEDES 旧版本
```

这样“动态优化”指知识状态在证据驱动下演进，而不是让模型自我强化。

## 6. 本次已实现的可插拔代码

### 6.1 模块清单

```text
src/fitness_agents/
  plugin_registry.py
  kg_interaction/
    contracts.py       # QueryPlan、EvidencePack、ChangeProposal
    ablation.py        # operator、反证、预算、早停、只读开关
    operators.py       # context/explain/compare 与 callable adapter
    controller.py      # 白名单、依赖、范围、泄漏防护、预算、早停
    writeback.py       # proposal validator、dry-run/commit、幂等 writer
```

对应配置：

- `configs/kg/interaction_modules.yaml`

对应测试：

- `tests/unit/test_kg_interaction_modules.py`

### 6.2 已实现的安全不变量

- 禁止计划参数包含 `sql`、`cypher`、`sparql`、`oracle`、`final_test`、`raw_fitness` 等键；
- 查询的 variant 必须属于 `allowed_variant_ids`；
- Operator 返回的 `as_of_round` 必须与请求轮次一致；
- 工具调用数取计划预算和实验配置预算的较小值；
- 反证查询可以独立开启/关闭；
- 每个步骤依赖关系显式；
- 写回默认 `read_only=True`；
- ChangeProposal 有置信度、Evidence 要求和幂等键；
- 重复提交不会重复写入。

### 6.3 当前没有实现的部分

- 尚未把 QueryPlanner 接入 ScientistAgent；
- 尚未把 controller 接到当前 `AgentKnowledgeGraphTool` 的主调用路径；
- 尚未实现真实 SQLite/Neo4j ChangeWriter；
- 尚未实现 HypothesisEvaluator；
- 尚未实现 structure/evolution/literature/path retriever；
- 尚未将 FitnessRecommendation 接入 acquisition；
- 尚未提供 PG-LLM endpoint。

这种状态是刻意的：先冻结接口和消融边界，验证后再更改现有主流程。

## 7. 消融实验设计

### 7.1 交互模块消融

| 条件 | 开关 | 要回答的问题 |
|---|---|---|
| LLM only | 不查询 KG | KG 是否带来净增益 |
| one-shot context | 仅 `hypothesis_context` | 复现当前范式 |
| explain | context + explain | 个体证据是否提高候选质量 |
| compare | context + compare | 横向比较是否改善排序 |
| no counterevidence | 关闭反证 | 反证搜索是否降低错误方向 |
| no early stop | 执行完整计划 | 早停是否节省成本而不损失效果 |
| agentic read | LLM planner + 只读工具 | 自主规划是否优于模板 |
| proposal dry-run | 生成但不提交 | 写回提案质量 |
| dynamic writeback | 受控提交 + 状态评估 | 历史学习是否改善后续轮次 |

### 7.2 评估指标

最终不能只看自然语言答案。建议报告：

- top-k hit rate、NDCG、Spearman；
- 每轮 best observed fitness、regret、达到阈值所需实验数；
- calibration error、interval coverage、OOD 分层表现；
- 推荐中 Evidence ID 引用有效率；
- 支持/反证覆盖率；
- 无效查询率、平均 tool calls、token、延迟和费用；
- oracle/final 泄漏测试通过率；
- 假设被支持、反驳和 inconclusive 的比例；
- 动态 KG 相对冻结 KG 的后续轮次增益。

所有比较固定 dataset split、seed、候选集、Predictor 和 acquisition policy，只改变一个交互组件。

## 8. 与 PG-LLM 的兼容策略

[PG-LLM](https://github.com/rohitarorayyc/proteingym-llm) 适合用作“给定 WT 和候选集的跨蛋白排序”测试，不覆盖完整主动学习闭环。推荐通过 OpenAI-compatible endpoint 对接，而不是让本项目 import PG-LLM 内部包。

运行模式必须分开：

| 模式 | KG | 写回 | 结果解释 |
|---|---|---|---|
| PG canonical LLM-only | 无 | 禁止 | 模型本身能力 |
| PG + frozen KG read | 固定、评测前冻结 | 禁止 | Agent/KG 增强能力 |
| PG + predictor | 固定模型工具 | 禁止 | 工具增强能力 |
| 本项目 campaign | 运行内动态 KG | 受控 | 闭环优化能力 |

PG 评测候选和得分不得进入持久 KG，也不得用于 prompt、query policy 或超参数调优。每个结果保存 `prompt_hash`、`model_id`、`kg_snapshot_id/hash`、operator 配置和 exposure manifest。

## 9. 推荐实施顺序

### P0：保持基线，接只读接口

1. 用现有 `AgentKnowledgeGraphTool` 注册三个已实现 Operator；
2. 使用 TemplatePlanner 生成固定两步计划；
3. 保存 EvidencePack 和 query receipt；
4. 新路径放在 feature flag 后；
5. 与当前 one-shot context 做配对回归。

退出条件：无泄漏回归通过；相同 seed 下关闭开关得到当前结果。

### P1：结构化 fitness 建议

1. 新增 FitnessRecommendation schema；
2. 强制 Evidence/Counterevidence/Query 引用；
3. 将建议仅转换为有限 knowledge bonus 或配额；
4. 保存 DecisionRecord。

退出条件：非法引用和无范围建议被拒绝；LLM 不能绕过 acquisition。

### P2：动态假设评估

1. 增加 TESTED_BY 和结构化 falsification rule；
2. 实验结果触发 HypothesisEvaluator；
3. 用 append-only 状态和 supersession 保存历史；
4. 先 dry-run，再允许自动提交低风险状态变更。

退出条件：每次状态变化都能追溯到 Observation 和 evaluator 版本。

### P3：高级检索插件

按收益逐步增加 structure、evolution、literature、path 和 PCST/MCTS 插件。只有当简单 compare/path 基线显示多跳瓶颈后，才引入 G-Retriever 或 ReKG-MCTS 类重型搜索。

### P4：PG-LLM 桥接

提供只读、无状态的 ranking endpoint；先用合成 fixture，再单 assay 冒烟，最后执行冻结快照的正式测试。

## 10. 验收清单

- [ ] 关闭全部新开关时，当前 campaign 行为不变；
- [ ] 每次 KG 查询均有 query_id、round、参数、结果 hash 和来源；
- [ ] 查询不接受原始 SQL/Cypher/SPARQL；
- [ ] 隐藏标签无法通过任何 Operator 返回；
- [ ] EvidencePack 明确区分 Observation、Prediction 和 Evidence；
- [ ] “优先”建议包含反证搜索状态和可证伪规则；
- [ ] LLM 只能提交 ChangeProposal；
- [ ] Observation/Prediction writer 权限与 Agent 隔离；
- [ ] 重复 proposal 幂等；历史不原地覆盖；
- [ ] 每个 Operator、Planner、SufficiencyPolicy 和 Writer 都可单独替换；
- [ ] 每个模块有一项对应消融配置；
- [ ] PG-LLM 模式使用冻结 KG 且禁止写回。

## 11. 最终建议

当前最优路线不是立即让 LLM “自由遍历并修改 KG”，而是先形成一个受控的知识工具层：

1. QueryPlan 让查询意图结构化；
2. Operator Registry 让不同知识源可插拔；
3. EvidencePack 让上下文最小、可引用、含反证；
4. SufficiencyPolicy 让查询成本可控；
5. Proposal Gateway 让写回可验证、可回放；
6. Predictor/acquisition 保留最终数值决策权。

这条路线能够同时支持当前小数据闭环、后续外部 KG、多模型 LLM API、PG-LLM 测试，以及逐模块消融。更重要的是，它把“LLM 的建议”转换成可核验的决策输入，而不是不可追溯的自由文本结论。

## 12. 主要参考资料

- [RoG: Reasoning on Graphs, ICLR 2024](https://openreview.net/forum?id=ZGNWW7xZ6Q)；[GitHub](https://github.com/RManLuo/reasoning-on-graphs)
- [Think-on-Graph, ICLR 2024](https://openreview.net/forum?id=nnVO1PvbTv)；[GitHub](https://github.com/DataArcTech/ToG)
- [BYOKG-RAG, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1417/)；[GitHub](https://github.com/awslabs/graphrag-toolkit)
- [KAG paper](https://arxiv.org/abs/2409.13731)；[GitHub](https://github.com/OpenSPG/KAG)
- [KG-RAG, Bioinformatics 2024](https://academic.oup.com/bioinformatics/article/40/9/btae560/7759620)；[GitHub](https://github.com/BaranziniLab/KG_RAG)
- [G-Retriever, NeurIPS 2024](https://openreview.net/forum?id=MPJ3oXtTZl)；[GitHub](https://github.com/XiaoxinHe/G-Retriever)
- [Graph-Constrained Reasoning](https://arxiv.org/abs/2410.13080)；[GitHub](https://github.com/RManLuo/graph-constrained-reasoning)
- [ReKG-MCTS, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.484/)；[GitHub](https://github.com/ShawnKS/rekgmcts)
- [Zep: Temporal KG Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)；[Graphiti GitHub](https://github.com/getzep/graphiti)
- [ProteinGym, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee-Paper-Datasets_and_Benchmarks.pdf)；[GitHub](https://github.com/OATML-Markslab/ProteinGym)

