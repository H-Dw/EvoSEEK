---
title: 突变数量、组合空间与筛选覆盖
knowledge_type: mutation_burden
language: zh-CN
version: "1.0.0"
evidence_level: evidence_informed_policy
rule_scope: [mutation_count_limit, library_size, screening_capacity, round_planning]
topics: [mutation burden, combinatorial explosion, library coverage, mutation rate, focused library]
citation_keys: [PackerLiu2015, Drummond2005, Reetz2008, Kille2013, Wu2016]
applies_to: [point_mutation_campaign, site_saturation_mutagenesis]
excludes: [universal_optimal_mutation_rate, unrestricted_recombination]
---

# 突变数量、组合空间与筛选覆盖

定向进化中的最优突变率与具体蛋白、文库生成方法和筛选条件有关，不存在适用于所有项目的固定数字。高突变负荷通常降低功能保留率，而 PCR 和抽样分布会改变观测到的文库组成，因此应把突变数量阈值写成可配置的工程策略，而不是自然定律。[Drummond et al., 2005](https://doi.org/10.1016/j.jmb.2005.05.023)

## 规则 BURDEN-001：本库默认突变数策略

以下阈值是风险控制默认值：

| 每个变体的氨基酸编辑数 | 默认决策 | 最低证据要求 |
|---:|---|---|
| 1 | 优先 | 理化、结构、进化或实验理由之一 |
| 2 | 允许 | 两个单点均有可解释依据；优先至少一个有同条件测量支持 |
| 3 | 条件允许 | 需要组合理由、筛选容量说明和更高不确定性惩罚 |
| >3 | 默认阻断 | 仅在显式 `high_order_combination`、重组或已验证模块替换模式下放行，并要求逐级验证 |

这里的“3”是本项目的保守工程上限，不是文献声称的普适最优值。Agent 输出必须把它标记为 `evidence-informed policy`。

## 规则 BURDEN-002：避免一次引入过多突变

优先使用逐轮策略：单点探索 → 选择命中 → 双点组合 → 复测 → 必要时三点组合。若一次跨越多个未验证编辑，无法区分有益、无效与有害贡献，并显著增加 epistasis 和归因不确定性。

## 规则 BURDEN-003：组合空间必须与筛选能力匹配

若 n 个位点各允许 20 种氨基酸状态，完整组合空间为 `20^n`；即使只随机化少数位点，空间也会迅速超过实验筛选能力。一个完整四位点、20 状态的实验景观包含 160,000 个组合，直接展示了高阶组合空间的规模与 epistasis 复杂性。[Wu et al., 2016](https://doi.org/10.7554/eLife.16965)

设计文库前应记录：理论氨基酸多样性、编码密码子多样性、预计有效 transformants、期望覆盖率、筛选上限、WT 与 stop 比例。若理论多样性超过可筛选量，先缩小位点或氨基酸字母表。

## 规则 BURDEN-004：聚焦文库优先于无约束全组合

结构、计算、系统发育或历史实验数据可用于把多样性聚焦到更可能相关的位点；这正是处理 library-size 限制的常用策略。[Packer & Liu, 2015](https://doi.org/10.1038/nrg3927)

## 规则 BURDEN-005：控制密码子冗余

简并密码子会使某些氨基酸过度代表、引入终止并增加达到目标覆盖率所需的筛选量。NDT 等缩减字母表可在特定项目中降低筛选成本；22c-trick 用 22 个密码子覆盖 20 个标准氨基酸并避免 stop，但其合成与装配成本也需计入。[Reetz, Kahakeaw & Lohmer, 2008](https://doi.org/10.1002/cbic.200800298)；[Kille et al., 2013](https://doi.org/10.1021/sb300037w)

## 规则 BURDEN-006：编辑数的计算口径

- 以最终蛋白序列相对当前 parent 的氨基酸改变数计数。
- 同一位点重复编辑合并为一个最终编辑，并在生成阶段禁止冲突表示。
- indel、片段重组和非天然残基不并入普通点突变计数，必须进入独立设计模式。
- 相对初始 WT 的总距离与相对上一轮 parent 的新增编辑数同时记录。

## 参考文献

1. Packer MS, Liu DR. *Nat Rev Genet* 16, 379–394 (2015). https://doi.org/10.1038/nrg3927
2. Drummond DA, Iverson BL, Georgiou G, Arnold FH. *J Mol Biol* 350, 806–816 (2005). https://doi.org/10.1016/j.jmb.2005.05.023
3. Reetz MT, Kahakeaw D, Lohmer R. *ChemBioChem* 9, 1797–1804 (2008). https://doi.org/10.1002/cbic.200800298
4. Kille S, et al. *ACS Synth Biol* 2, 83–92 (2013). https://doi.org/10.1021/sb300037w
5. Wu NC, et al. *eLife* 5, e16965 (2016). https://doi.org/10.7554/eLife.16965
