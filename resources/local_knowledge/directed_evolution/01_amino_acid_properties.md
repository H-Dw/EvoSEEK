---
title: 氨基酸理化性质与突变解释
knowledge_type: amino_acid_properties
language: zh-CN
version: "1.0.0"
evidence_level: textbook_and_primary
rule_scope: [single_substitution, physicochemical_delta, protein_stability]
topics: [hydrophobicity, charge, polarity, side-chain size, glycine, proline, cysteine]
citation_keys: [Cooper2000, Alberts2002, KyteDoolittle1982, Grantham1974]
applies_to: [canonical_amino_acids, soluble_proteins, protein_interfaces]
excludes: [noncanonical_amino_acids, condition_free_absolute_predictions]
---

# 氨基酸理化性质与突变解释

氨基酸侧链的大小、形状、电荷、极性、疏水性和反应性共同影响蛋白质折叠、稳定性与功能；非极性侧链通常更倾向于埋藏，极性或带电侧链通常更常见于水相表面，但局部结构与功能环境可以覆盖这一总体趋势。[Cooper, 2000](https://www.ncbi.nlm.nih.gov/books/NBK9879/)；[Alberts et al., 2002](https://www.ncbi.nlm.nih.gov/books/NBK26830/)

## 常用侧链分组

| 分组 | 残基 | 突变解释要点 |
|---|---|---|
| 疏水脂肪族 | A, V, I, L, M | 核心区可参与疏水堆积；体积增大仍可能造成 clash |
| 芳香族 | F, W, Y | 可形成疏水、π 或特异界面相互作用；Y 还含极性羟基 |
| 极性不带电 | S, T, N, Q, C | 可形成氢键；埋藏时应检查是否满足氢键或特定化学作用 |
| 酸性 | D, E | 在常见生理条件下通常带负电；检查盐桥、金属结合和 pH |
| 碱性 | K, R | 在常见生理条件下通常带正电；检查盐桥、溶剂暴露和界面互补 |
| 可变质子化 | H | 电荷状态强烈依赖局部 pKa 与实验 pH，不能简单固定为正电 |
| 骨架特殊 | G, P | G 高度灵活且无侧链碳；P 对主链构象有强约束 |
| 化学特殊 | C | 可参与二硫键或亲核化学，增删 Cys 均需显式检查 |

这些分组用于特征化，不是保证功能等价的替换表。标准教材也明确指出侧链化学性质决定其在结构和功能中的角色，而极性、带电与非极性残基的空间分布与折叠环境相关。[Cooper, 2000](https://www.ncbi.nlm.nih.gov/books/NBK9879/)

## 规则 AA-PROP-001：记录多维变化

每个候选替换至少记录：`delta_hydrophobicity`、`charge_change`、`polarity_change`、`size_or_volume_change`、`aromaticity_change`、`special_residue_change`。Kyte–Doolittle hydropathy 可作为疏水性的一种量化尺度，但它不能单独决定突变好坏。[Kyte & Doolittle, 1982](https://doi.org/10.1016/0022-2836(82)90515-0)

## 规则 AA-PROP-002：位置环境优先

- 埋藏核心：优先检查疏水匹配、空腔填充、过度堆积与未满足极性基团。
- 溶剂暴露表面：优先检查溶解性、表面电荷与非特异聚集风险。
- 配体或蛋白界面：优先检查几何互补、氢键、盐桥、芳香或疏水接触；不能简单套用“表面亲水、核心疏水”。
- 跨膜区：需要单独的膜环境模型，不使用本文件的可溶蛋白默认值。

## 规则 AA-PROP-003：电荷变化分级

同号保留通常比电荷消失或反转更保守；`D/E ↔ K/R` 视为高风险电荷反转。H 的状态必须结合 pH 和局部环境。若电荷改变同时破坏已知盐桥、催化基团或结合互补，升级为 `context_override` 高风险。

## 规则 AA-PROP-004：特殊残基必须单独标记

- `to/from Pro`：检查 α-helix、β-strand、turn 和 cis/trans 肽键环境。
- `to/from Gly`：检查是否需要极小体积、正 φ 角或局部柔性。
- `to/from Cys`：检查二硫键配对、氧化环境、暴露游离硫醇与错误配对。

## 规则 AA-PROP-005：距离是先验，不是答案

Grantham 距离把组成、极性与分子体积合并为氨基酸差异度，并观察到进化替换频率与总体化学差异相关；因此可作为“激进程度”先验，但不能替代位点特异的结构、保守性或实验数据。[Grantham, 1974](https://doi.org/10.1126/science.185.4154.862)

## 参考文献

1. Cooper GM. *The Cell: A Molecular Approach*, 2nd ed., “The Molecular Composition of Cells”. NCBI Bookshelf, 2000. https://www.ncbi.nlm.nih.gov/books/NBK9879/
2. Alberts B, et al. *Molecular Biology of the Cell*, 4th ed., “The Shape and Structure of Proteins”. NCBI Bookshelf, 2002. https://www.ncbi.nlm.nih.gov/books/NBK26830/
3. Kyte J, Doolittle RF. *J Mol Biol* 157, 105–132 (1982). https://doi.org/10.1016/0022-2836(82)90515-0
4. Grantham R. *Science* 185, 862–864 (1974). https://doi.org/10.1126/science.185.4154.862
