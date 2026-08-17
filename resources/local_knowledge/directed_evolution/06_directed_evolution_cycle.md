---
title: 定向进化闭环与聚焦文库策略
knowledge_type: directed_evolution_strategy
language: zh-CN
version: "1.0.0"
evidence_level: review_and_protocol
rule_scope: [round_planning, diversification, screening, selection, focused_library]
topics: [directed evolution, iterative cycle, random mutagenesis, site saturation, screening, selection]
citation_keys: [PackerLiu2015, ZeymerHilvert2018, ReetzCarballeira2007, Currin2015]
applies_to: [iterative_protein_engineering]
excludes: [single_pass_unvalidated_design]
---

# 定向进化闭环与聚焦文库策略

定向进化是多轮循环：产生遗传多样性，筛选或选择目标表型，确认优良变体，再以选定 parent 进入下一轮。文库生成和筛选能力必须共同设计，而不是先产生巨大文库再假设能够覆盖。[Packer & Liu, 2015](https://doi.org/10.1038/nrg3927)；[Zeymer & Hilvert, 2018](https://doi.org/10.1146/annurev-biochem-062917-012034)

## 规则 DE-CYCLE-001：每轮最小闭环

1. 定义单一主目标和必须守住的 counter-screen 指标。
2. 选择 parent，并冻结其序列、测量与 provenance。
3. 根据不确定性选择随机、聚焦或组合多样化策略。
4. 生成可被真实 screen capacity 覆盖的候选集合。
5. 使用与目标耦合的 assay 筛选，并保留 parent、WT、阳性/阴性对照和重复。
6. 对命中变体复测、测序并确认序列—表型链接。
7. 更新历史证据、失败机制与下一轮 mutation budget。

## 规则 DE-CYCLE-002：策略选择

| 已知信息 | 首选多样化方式 | 原因 |
|---|---|---|
| 几乎无位点信息 | 低到中等负荷 random mutagenesis | 广泛探索，但要控制功能丢失和归因难度 |
| 有结构/保守性/模型热点 | focused site-saturation 或小字母表 | 把筛选量集中到高价值位点 |
| 已有可靠单点命中 | 逐步组合或 ISM | 利用历史结果，同时显式检测 epistasis |
| 有同源家族功能片段 | 受控 recombination | 组合天然已存在的模块，但仍需筛选 |

聚焦位点可来自结构、计算模型或系统发育信息；这一策略用于缓解实际 library-size 限制。[Packer & Liu, 2015](https://doi.org/10.1038/nrg3927)

## 规则 DE-CYCLE-003：迭代饱和突变

Iterative saturation mutagenesis 在一组位点/位点簇上逐轮饱和，使用上一轮更优变体作为下一轮模板，适合在可管理的局部文库中探索组合效应。[Reetz & Carballeira, 2007](https://doi.org/10.1038/nprot.2007.72)

采用 ISM 时必须保留多条候选路径，而不是每轮只保留单一 top-1；否则早期噪声或局部最优可能永久收窄后续空间。

## 规则 DE-CYCLE-004：命中必须复核

初筛命中至少进行：独立重建或复测、序列确认、同批次 parent 对照、主要目标复现、关键 counter-screen。若信号只在单次高通量初筛出现，证据等级不得超过 `provisional_hit`。

## 规则 DE-CYCLE-005：保留失败知识

失败候选不是无信息样本。记录其位点、替换类型、表达/稳定性、主 assay、反筛和不确定性，可帮助下一轮避开重复失败、识别 trade-off 并校正模型。定向进化综述强调，多样化与可判别的筛选/选择共同决定循环是否有效。[Zeymer & Hilvert, 2018](https://doi.org/10.1146/annurev-biochem-062917-012034)

## 规则 DE-CYCLE-006：随机性与知识引导叠加

知识引导不等于只生成高分保守替换。每轮可保留一小部分机制多样、风险受控的探索候选，用于发现先验之外的改进；同时用 hard gate、mutation budget 和 counter-screen 控制代价。面向序列空间的综述也把随机、聚焦和合成生物学文库视为互补工具。[Currin et al., 2015](https://doi.org/10.1039/C4CS00351A)

## 参考文献

1. Packer MS, Liu DR. *Nat Rev Genet* 16, 379–394 (2015). https://doi.org/10.1038/nrg3927
2. Zeymer C, Hilvert D. *Annu Rev Biochem* 87, 131–157 (2018). https://doi.org/10.1146/annurev-biochem-062917-012034
3. Reetz MT, Carballeira JD. *Nat Protoc* 2, 891–903 (2007). https://doi.org/10.1038/nprot.2007.72
4. Currin A, Swainston N, Day PJ, Kell DB. *Chem Soc Rev* 44, 1172–1239 (2015). https://doi.org/10.1039/C4CS00351A
