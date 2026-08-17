---
title: 序列合法性、终止密码子与非标准氨基酸门禁
knowledge_type: sequence_safeguards
language: zh-CN
version: "1.0.0"
evidence_level: textbook_review_and_method
rule_scope: [hard_gate, translation_validation, codon_library_design]
topics: [canonical amino acids, stop codon, open reading frame, NNK, NNS, noncanonical amino acids]
citation_keys: [Alberts2002Translation, Kille2013, LiuSchultz2010]
applies_to: [standard_genetic_code, canonical_point_mutation]
excludes: [explicit_genetic_code_expansion, programmed_stop_suppression]
---

# 序列合法性、终止密码子与非标准氨基酸门禁

标准遗传码中 61 个密码子编码氨基酸，UAA、UAG、UGA 是翻译终止信号；DNA 层面对应 TAA、TAG、TGA。内部 stop 可导致提前终止和截短蛋白，因此是默认 hard reject。[Alberts et al., 2002](https://www.ncbi.nlm.nih.gov/books/NBK26829/)

## 规则 SAFE-001：标准点突变模式的残基白名单

默认只允许 20 个标准单字母氨基酸：

`ACDEFGHIKLMNPQRSTVWY`

以下符号默认拒绝：`*`、`X`、`B`、`Z`、`J`、gap、空字符及任意非字母 token。`U` 和 `O` 只有在任务明确提供宿主、密码子、翻译机制和验证流程时才可放行；普通 canonical campaign 不将其视为默认候选。

## 规则 SAFE-002：突变表示必须自洽

- 位点必须在参考序列范围内。
- 声明的 WT 残基必须与采用的 parent/参考序列一致。
- 同一候选不得对同一位置给出两个不同终态。
- 点突变模式不得隐式包含插入、缺失、移码或片段替换。
- 生成最终序列后重新计算 mutation count，不信任仅由文本声明的数量。

## 规则 SAFE-003：DNA 构建设计必须翻译回检

对任何回译或密码子优化后的构建执行：

1. 长度与 reading frame 检查；
2. 使用指定宿主/细胞器的遗传码翻译完整 ORF；
3. 验证起始、预期终止位置和所有内部 codon；
4. 拒绝内部 TAA/TAG/TGA 或非预期移码；
5. 验证翻译蛋白与目标氨基酸序列完全一致。

不能假设所有宿主都严格使用同一密码子解释；若任务不是标准核基因遗传码，必须显式配置 translation table。

## 规则 SAFE-004：简并密码子设计检查

NNK/NNS 各包含 32 个密码子，可覆盖 20 个标准氨基酸，但包含一个 TAG stop，且不同氨基酸的表示不均衡。若 screen capacity 有限，应优先考虑无 stop、低冗余或按目标氨基酸集合优化的密码子设计。22c-trick 使用三个引物组合产生 22 个密码子覆盖 20 个标准氨基酸并避免 stop，可作为完整标准字母表的低冗余方案之一。[Kille et al., 2013](https://doi.org/10.1021/sb300037w)

## 规则 SAFE-005：非天然氨基酸必须进入独立模式

非天然/非标准氨基酸的定点掺入通常需要正交 aminoacyl-tRNA synthetase/tRNA pair、可用密码子和宿主兼容的翻译系统；因此不能把未知字符直接塞入普通蛋白序列并假定可表达。[Liu & Schultz, 2010](https://doi.org/10.1146/annurev.biochem.052308.105824)

放行非天然残基至少要求：

- 明确化学实体和唯一残基 ID；
- 明确编码/掺入方法、宿主、正交翻译组件和 stop suppression 风险；
- 单独的表达、质谱或等价身份验证；
- 不与 canonical mutation count 和普通保守替换评分混算。

## 规则 SAFE-006：异常终止或读穿不得自动修复

若发现内部 stop、ambiguous residue 或 ORF 不一致，系统应返回具体位置和原因并阻断候选；不得静默替换为最相近标准氨基酸，也不得通过删除字符“修复”序列。

## 参考文献

1. Alberts B, et al. *Molecular Biology of the Cell*, 4th ed., “From RNA to Protein”. NCBI Bookshelf, 2002. https://www.ncbi.nlm.nih.gov/books/NBK26829/
2. Kille S, et al. *ACS Synth Biol* 2, 83–92 (2013). https://doi.org/10.1021/sb300037w
3. Liu CC, Schultz PG. *Annu Rev Biochem* 79, 413–444 (2010). https://doi.org/10.1146/annurev.biochem.052308.105824
