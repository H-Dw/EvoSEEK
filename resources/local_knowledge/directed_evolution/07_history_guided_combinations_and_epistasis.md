---
title: 历史单点优先、组合验证与突变互作
knowledge_type: history_guided_combination
language: zh-CN
version: "1.0.0"
evidence_level: primary_and_review
rule_scope: [historical_hit_ranking, mutation_combination, epistasis, path_planning]
topics: [beneficial single mutation, combination, additivity, epistasis, fitness landscape]
citation_keys: [RomeroArnold2009, StarrThornton2016, Olson2014, Wu2016]
applies_to: [measured_single_mutations, iterative_combination]
excludes: [blind_top_hit_stacking, cross_assay_score_comparison]
---

# 历史单点优先、组合验证与突变互作

优先组合历史上表现较好的单点突变是合理的搜索策略，但不是“好单点相加必然更好”的规则。蛋白质内 epistasis 广泛存在：一个突变的效果会随遗传背景改变，原因可来自直接/间接结构相互作用，也可来自稳定性或结合能到实验表型之间的非线性映射。[Starr & Thornton, 2016](https://doi.org/10.1002/pro.2897)

## 规则 HIST-001：历史单点证据排序

单点突变进入组合池前按以下顺序分层：

1. 同一 assay、同一条件、独立重复且序列确认的改善；
2. 同一目标但条件略有差异的可复现改善；
3. 同源蛋白或相关 assay 的改善；
4. 仅由模型、理化或文献类比预测的改善。

只有第 1 层可标记为 `measured_hit`。跨 assay 的原始分数不得直接排序；应先统一方向、尺度、误差和适用条件。

## 规则 HIST-002：优先但不盲目堆叠

组合候选优先选择：

- 单点效果可复现；
- 位点不冲突；
- 结构机制互补或空间上不过度拥挤；
- 不同时放大同一风险，例如多个去稳定化核心突变；
- 保留机制与位置多样性，而不是只取同一局部区域的 top-k。

## 规则 HIST-003：逐级组合与回拆

推荐路径为 `single → pair → triple`。每增加一级都保留组成单点和关键子组合，以便判断哪个编辑贡献改善、哪个产生负 epistasis。三点组合若没有对应 pair 数据，证据等级下调并增加不确定性。

## 规则 HIST-004：显式计算 epistasis

在选择了适合可加比较的表型尺度后，可用

`epsilon_AB = y_AB - y_A - y_B + y_WT`

描述双突变相对加性预期的偏离。必须同时报告误差传播与 assay 变换；原始比例、对数活性、结合自由能和生长 fitness 的可加尺度不同。对完整蛋白域的大规模双突变研究观察到广泛的正、负 epistasis，并表明稳定性与功能的非线性可产生表型互作。[Olson, Wu & Sun, 2014](https://doi.org/10.1016/j.cub.2014.09.072)

## 规则 HIST-005：负互作不是自动删除全部单点

若组合低于加性预期：

- 检查两个位点的直接接触、共同 packing、盐桥或氢键网络；
- 检查是否共同降低稳定性并跨过功能阈值；
- 分别保留单点结果，不把组合失败错误归因到每个单点；
- 必要时搜索补偿突变或替代组合路径。

实验 fitness landscape 显示，直接适应路径可能被 sign epistasis 阻断，而额外的中间状态或绕行路径可以改变可达性。[Wu et al., 2016](https://doi.org/10.7554/eLife.16965)

## 规则 HIST-006：历史 parent 必须进入 provenance

同一突变在不同 parent 上可能有不同效果。每条历史记录必须保存 parent sequence/hash、assay、条件、round、replicates、effect、uncertainty 和数据来源；缺少 parent 的“有益突变”只能作为低等级背景支撑。

## 规则 HIST-007：多条路径优于单一路径贪心

蛋白 fitness landscape 可呈 rugged 和多峰结构。每轮组合时保留数条不同 parent 或机制路径，能降低早期噪声、局部最优和背景依赖带来的锁定风险。[Romero & Arnold, 2009](https://doi.org/10.1038/nrm2805)

## 参考文献

1. Romero PA, Arnold FH. *Nat Rev Mol Cell Biol* 10, 866–876 (2009). https://doi.org/10.1038/nrm2805
2. Starr TN, Thornton JW. *Protein Sci* 25, 1204–1218 (2016). https://doi.org/10.1002/pro.2897
3. Olson CA, Wu NC, Sun R. *Curr Biol* 24, 2643–2651 (2014). https://doi.org/10.1016/j.cub.2014.09.072
4. Wu NC, et al. *eLife* 5, e16965 (2016). https://doi.org/10.7554/eLife.16965
