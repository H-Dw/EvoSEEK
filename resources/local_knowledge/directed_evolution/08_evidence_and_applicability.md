---
title: 证据等级、适用性与引用边界
knowledge_type: evidence_applicability
language: zh-CN
version: "1.0.0"
evidence_level: methodological_policy
rule_scope: [evidence_grading, citation_validation, rag_usage, kg_claims]
topics: [evidence hierarchy, applicability, citation support, uncertainty, deep mutational scanning]
citation_keys: [FowlerFields2014, NgHenikoff2003, PackerLiu2015]
applies_to: [rag_retrieval, agent_reasoning, kg_materialization]
excludes: [automatic_causal_upgrade, metadata_only_support]
---

# 证据等级、适用性与引用边界

突变规则只有在明确适用范围时才有用。同一替换的效果可受位点、parent、结构、assay 和条件影响；因此 RAG 命中应作为可回溯证据，而不是自动升级为目标蛋白的因果事实。

## 规则 EVID-001：证据层级

| 等级 | 证据 | 默认用途 |
|---|---|---|
| A | 同一 parent、同一 assay/条件、独立复测的目标突变 | 可作为高置信历史实验依据 |
| B | 同一目标、条件或 parent 有差异的测量 | 部分支撑，必须写明差异 |
| C | 同源蛋白、相似结构或相关 assay 的测量 | 类比支撑，不作直接效果承诺 |
| D | 综述、教材、通用机制或计算预测 | 背景先验与候选解释 |
| E | 仅标题/metadata 相关，未核验摘要或正文 | 只能作为待筛选 citation candidate |

深度突变扫描可提供大量位点特异实验数据，但仍受 assay 和实验体系约束。[Fowler & Fields, 2014](https://doi.org/10.1038/nmeth.3027)

## 规则 EVID-002：支撑等级

每条 citation 对具体 statement 标注：

- `strong_support`：直接测试同一核心关系且上下文相近；
- `partial_support`：只支持句子的一部分或更窄条件；
- `background_support`：只提供通用机制/领域背景；
- `limiting_or_contradictory`：限制或反驳该主张；
- `metadata_only`：未核验内容，不得作为正文支撑。

## 规则 EVID-003：预测方法不等于实验证据

基于同源序列的替换容忍度工具可用于保守性先验，但不能替代目标条件测量。SIFT 明确使用序列同源信息预测替换是否影响功能，这类结果应保留方法、版本、输入 alignment 和不确定性。[Ng & Henikoff, 2003](https://doi.org/10.1093/nar/gkg509)

## 规则 EVID-004：工程策略与科学事实分离

诸如“默认最多 3 个突变”“优先组合历史单点”属于 evidence-informed policy；其理由来自 library size、筛选能力、epistasis 与可解释性，但阈值本身不是普适生物学定律。KG Claim 必须在 `applicability` 中写入 `policy`、适用模式和可覆盖条件。

## 规则 EVID-005：引用必须就近对应观点

- citation 应位于它支持的句子或规则附近；
- 不用综述替代可获得的关键原始实验，但综述可支撑方法框架；
- DOI、作者、年份和标题至少通过 PubMed、Crossref 或官方出版页交叉核验；
- 不因高被引或期刊级别自动提高支撑等级；
- RAG 输出必须带文档路径、chunk span、file hash、citation keys 和 knowledge type。

定向进化综述可以支撑“多样化—筛选/选择的迭代框架”和聚焦文库原则，但不能替代某个具体突变在当前 assay 中的验证。[Packer & Liu, 2015](https://doi.org/10.1038/nrg3927)

## 规则 EVID-006：冲突证据并存

若不同来源对同类突变结论不一致，保留双方 provenance，比较 parent、结构、assay、条件与误差，不通过删除负面文献制造虚假一致。Agent 的最终理由应同时给出支持、限制和待验证项。

## 参考文献

1. Fowler DM, Fields S. *Nat Methods* 11, 801–807 (2014). https://doi.org/10.1038/nmeth.3027
2. Ng PC, Henikoff S. *Nucleic Acids Res* 31, 3812–3814 (2003). https://doi.org/10.1093/nar/gkg509
3. Packer MS, Liu DR. *Nat Rev Genet* 16, 379–394 (2015). https://doi.org/10.1038/nrg3927
