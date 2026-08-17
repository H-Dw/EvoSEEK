---
title: 保守替换、激进替换与上下文覆盖
knowledge_type: substitution_conservativeness
language: zh-CN
version: "1.0.0"
evidence_level: primary_and_review
rule_scope: [candidate_generation, substitution_risk, evolutionary_prior]
topics: [conservative substitution, radical substitution, Grantham distance, BLOSUM62, conservation]
citation_keys: [Grantham1974, Henikoff1992, NgHenikoff2003, FowlerFields2014]
applies_to: [canonical_amino_acid_substitution]
excludes: [universal_context_free_tolerance]
---

# 保守替换、激进替换与上下文覆盖

“保守”表示替换在某些理化或进化维度上相似，并不表示该位点一定容忍。Grantham 距离描述组成、极性和体积差异；BLOSUM 描述保守蛋白区块中实际观察到的替换对数优势。两者回答的问题不同，均不能单独作为目标功能的真值。[Grantham, 1974](https://doi.org/10.1126/science.185.4154.862)；[Henikoff & Henikoff, 1992](https://doi.org/10.1073/pnas.89.22.10915)

## 规则 SUB-CLASS-001：保留原始特征，不只保留标签

每个替换应同时保留：Grantham distance、BLOSUM62 score、名义电荷变化、极性变化、疏水性变化、体积变化、位点保守性和结构环境。最终标签采用 `conservative`、`moderate`、`radical` 或 `context_override`，并附可审计原因。

## 规则 SUB-CLASS-002：保守候选

以下信号共同出现时可标记为 `conservative` 候选：

- 名义电荷不变；
- 极性与疏水类别相近；
- 侧链尺寸变化有限，未显示空腔或 steric clash 风险；
- BLOSUM62 为非负或同源序列中可观察到该替换；
- 位点不属于已知催化、结合、二硫键或主链几何特殊位置。

这是一组工程判据，不是普适等价关系。序列保守性方法如 SIFT 的基本思想也是利用同源序列中可接受的残基分布判断替换是否可能影响功能，说明“同一替换在不同位置的后果不同”。[Ng & Henikoff, 2003](https://doi.org/10.1093/nar/gkg509)

## 规则 SUB-CLASS-003：激进或高风险替换

出现任一强信号时至少标记为 `radical` 或 `context_override`：

- 正负电荷反转；
- 疏水核心中引入未满足的带电或强极性侧链；
- 大幅体积增加造成碰撞，或大幅缩小造成核心空腔；
- `to/from Gly`、`to/from Pro` 或改变结构性 Cys；
- BLOSUM62 显著不利且 Grantham distance 较大；
- 高保守位点或已知功能位点发生非同类替换。

## 规则 SUB-CLASS-004：上下文可以推翻“保守”判断

- `D→E` 虽保持负电，但在精确几何的盐桥或催化位点仍可能有强影响。
- `L→I` 虽均疏水，但 β-branching 和 rotamer 几何可能改变堆积。
- `S→T` 虽均极性，但额外甲基可能造成局部冲突或改变构象偏好。
- `K→R` 虽保持正电，但长度、氢键几何和 guanidinium 化学并不相同。

## 规则 SUB-CLASS-005：经验测量覆盖先验

当同一位点存在可靠的深度突变扫描、同条件单点测量或多轮定向进化数据时，经验结果优先于通用分类。深度突变扫描能够并行测量大量氨基酸变体，因此适合把“保守/激进”先验校正为位点特异经验分布。[Fowler & Fields, 2014](https://doi.org/10.1038/nmeth.3027)

## 规则 SUB-CLASS-006：输出反证

候选解释除支持理由外还应输出至少一个可能失效的机制，例如“保持疏水性但可能过度堆积”或“保持电荷但可能破坏几何”。这样可防止 Agent 把保守替换误写成确定性结论。

## 参考文献

1. Grantham R. *Science* 185, 862–864 (1974). https://doi.org/10.1126/science.185.4154.862
2. Henikoff S, Henikoff JG. *PNAS* 89, 10915–10919 (1992). https://doi.org/10.1073/pnas.89.22.10915
3. Ng PC, Henikoff S. *Nucleic Acids Res* 31, 3812–3814 (2003). https://doi.org/10.1093/nar/gkg509
4. Fowler DM, Fields S. *Nat Methods* 11, 801–807 (2014). https://doi.org/10.1038/nmeth.3027
