---
title: 定向进化知识库使用与解释约定
knowledge_type: corpus_guide
language: zh-CN
version: "1.0.0"
evidence_level: operational_contract
rule_scope: [rag_routing, kg_provenance, evidence_interpretation]
topics: [document types, retrieval filters, evidence levels, citations]
citation_keys: []
applies_to: [canonical_protein_point_mutation, directed_evolution_planning]
excludes: [clinical_variant_interpretation, target_specific_fitness_labels]
---

# 定向进化知识库使用与解释约定

本目录是面向蛋白质定向进化与点突变设计的通用规则库。它只保存可公开核验的通用知识、工程策略和出处，不保存目标蛋白的实验标签、最优序列或隐藏测试结果。

## 文档类型

| `knowledge_type` | 内容 | 主要用途 |
|---|---|---|
| `amino_acid_properties` | 氨基酸疏水性、电荷、极性、体积与特殊骨架性质 | 解释单点替换的物理化学变化 |
| `substitution_conservativeness` | 保守、适中、激进替换及其上下文修正 | 生成候选与风险分层 |
| `structure_context` | 核心、表面、界面、活性位点、二级结构和二硫键约束 | 对通用替换规则作结构覆盖 |
| `mutation_burden` | 突变数量、组合空间、筛选覆盖与默认上限 | 控制一次引入的突变数 |
| `sequence_safeguards` | 标准氨基酸、终止密码子、开放阅读框和简并密码子 | 作为候选硬门禁 |
| `directed_evolution_strategy` | 多轮突变—筛选—学习闭环与聚焦文库 | 规划每轮探索策略 |
| `history_guided_combination` | 历史单点突变、组合优先级和 epistasis | 组合已验证单点而不假设可加性 |
| `evidence_applicability` | 证据等级、适用范围与 citation 使用边界 | 防止把背景知识当作目标实验证据 |

RAG 查询可通过 `knowledge_types` 过滤一个或多个类型。返回的 chunk、Evidence、KG `Document`、`DocumentChunk`、`Claim` 均应保留相同的 `knowledge_type` 和原文 citation metadata。

## 规则强度

- **hard gate**：序列或表达层面的确定性非法状态，例如内部终止、未知残基、位点越界、野生型不匹配。
- **evidence-informed policy**：由文献、筛选能力和风险共同推导的工程默认值，可由显式实验设计覆盖。
- **scientific prior**：可用于排序或解释，但不能代替目标体系中的测量。
- **context override**：结构、功能位点、条件或 epistasis 证据优先于简单的全局氨基酸分类。

## Citation 约定

正文中的关键观点就近附 DOI、PubMed、NCBI Bookshelf 或官方出版页链接。`citation_keys` 只用于 provenance 和检索提示；真正使用结论时仍应回到正文对应句和参考文献核验。检索命中默认是“待复核证据”，不得仅凭标题相关性升级为强支撑，也不得直接贡献候选选择分数。
