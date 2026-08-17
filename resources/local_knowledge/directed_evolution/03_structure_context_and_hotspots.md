---
title: 结构环境、功能位点与突变热点
knowledge_type: structure_context
language: zh-CN
version: "1.0.0"
evidence_level: textbook_review_and_method
rule_scope: [structure_filter, hotspot_selection, context_override]
topics: [buried core, surface, interface, active site, secondary structure, disulfide, hotspot]
citation_keys: [Alberts2002, Pavelka2009, GoldenzweigFleishman2018, FowlerFields2014]
applies_to: [structure_available, predicted_structure_with_uncertainty]
excludes: [structure_blind_absolute_rules]
---

# 结构环境、功能位点与突变热点

蛋白质功能来自三维结构，氨基酸替换的影响必须结合局部环境解释。教材层面的总体规律是疏水侧链更常埋藏于内部、极性侧链更常位于表面；真正设计时还要检查 packing、氢键、盐桥、界面与构象自由度。[Alberts et al., 2002](https://www.ncbi.nlm.nih.gov/books/NBK26830/)

## 规则 STRUCT-001：结构区域标签

每个位点至少赋一个环境标签：`buried_core`、`partially_buried`、`solvent_exposed`、`protein_interface`、`ligand_interface`、`active_or_functional_site`、`secondary_structure`、`loop_or_turn`、`disulfide_or_reactive_cysteine`、`unknown`。结构缺失或置信度低时不得伪装成确定标签。

## 规则 STRUCT-002：埋藏核心

- 检查疏水匹配、侧链体积、rotamer 可容纳性与空腔。
- 对埋藏电荷或极性残基，确认现有氢键、盐桥、金属或功能网络；不能仅因“核心应疏水”而替换。
- 引入更大侧链须检查 clash；引入更小侧链须检查 cavity 与水进入风险。

蛋白稳定性设计综述强调，疏水堆积、氢键和局部相互作用必须在具体构象中共同评估，而非按单一尺度优化。[Goldenzweig & Fleishman, 2018](https://doi.org/10.1146/annurev-biochem-062917-012102)

## 规则 STRUCT-003：表面与界面分开处理

溶剂暴露表面可优先考虑溶解性与表面电荷，但蛋白或配体界面需要保留形状、电荷和氢键互补。界面疏水残基可能是结合热点，不能按“表面应亲水”自动删除。

## 规则 STRUCT-004：功能位点采用高风险门槛

催化残基、配体直接接触残基、金属配位残基和已知变构网络默认不做无证据激进替换。若优化目标正是改变底物特异性或活性，可把邻近且进化可变的位点作为 focused library，而不是无差别随机化。HotSpot Wizard 的方法正是联合结构、功能和进化信息选择工程热点。[Pavelka, Chovancova & Damborsky, 2009](https://doi.org/10.1093/nar/gkp410)

## 规则 STRUCT-005：Gly、Pro、Cys 的结构覆盖

- Gly 位于紧转角、正 φ 构象或狭窄空间时，替换风险上调。
- Pro 位于 helix、strand 或关键 turn 时，增删 Pro 均上调风险。
- 成对二硫键中的 Cys 为 hard protection；新增 Cys 时检查潜在错误配对和氧化环境。

## 规则 STRUCT-006：热点优先级

热点优先级从高到低建议为：

1. 同条件实验已显示可改进且未破坏表达/稳定性的位点；
2. 结构邻近目标功能区域、同时在同源序列中可变的位点；
3. 有可靠计算或深度突变扫描支持的位点；
4. 仅凭全局理化规则推测的位点。

深度突变扫描提供大规模位点特异测量，可用于检验结构先验；但实验 assay 的适用范围必须与当前目标一致。[Fowler & Fields, 2014](https://doi.org/10.1038/nmeth.3027)

## 规则 STRUCT-007：不确定性随结构来源传播

实验结构、同源模型和预测结构具有不同误差。低置信度 loop、缺失残基、替代构象和界面装配不确定性必须进入候选 provenance；结构不确定时降级为软先验，不能作为 hard reject 的唯一依据。

## 参考文献

1. Alberts B, et al. *Molecular Biology of the Cell*, 4th ed. NCBI Bookshelf, 2002. https://www.ncbi.nlm.nih.gov/books/NBK26830/
2. Pavelka A, Chovancova E, Damborsky J. *Nucleic Acids Res* 37, W376–W383 (2009). https://doi.org/10.1093/nar/gkp410
3. Goldenzweig A, Fleishman SJ. *Annu Rev Biochem* 87, 105–129 (2018). https://doi.org/10.1146/annurev-biochem-062917-012102
4. Fowler DM, Fields S. *Nat Methods* 11, 801–807 (2014). https://doi.org/10.1038/nmeth.3027
