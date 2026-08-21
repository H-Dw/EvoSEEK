# 为什么 RAG 没有带来预期提升：检索—Prompt—选择链审计

## 结论先行

本实验不能概括为‘RAG 让所有性能下降’。RAG 相对 `kg_base` 提高了 best-seen AULC、Round 3 batch mean/median 和部分排序指标，但没有提高最终 best-seen，且 Spearman、Pearson、MSE 与高峰发现能力变差。更准确的结论是：RAG 没有形成稳定的候选级增益，其主要原因是检索内容与当前 GB1 四位点闭池选择的决策接口不匹配。

## 1. 性能变化是混合的，不是单向退化

| 指标 | kg_base | kg_base_rag | RAG − base | 解释 |
|---|---:|---:|---:|---|
| 最终峰值 | 5.393 ± 0.550 | 5.075 ± 0.000 | -0.317 | 下降 |
| 峰值轨迹效率 | 4.765 ± 0.126 | 4.888 ± 0.035 | +0.123 | 改善 |
| 末轮批次均值 | 1.154 ± 0.106 | 1.355 ± 0.499 | +0.200 | 改善 |
| 末轮批次中位数 | 0.583 ± 0.345 | 1.151 ± 0.778 | +0.568 | 改善 |
| 预测排序 | 0.243 ± 0.049 | 0.208 ± 0.028 | -0.034 | 下降 |
| 预测误差 | 0.357 ± 0.148 | 0.487 ± 0.314 | +0.130 | 下降 |
| Top-k regret | 4.140 ± 1.106 | 3.513 ± 1.072 | -0.627 | 改善 |

因此，问题不是 RAG 完全无效，而是它没有把局部批次富集转化为更高、可重复的峰值发现或更可靠的预测对齐。

## 2. 实际检索内容：高重复、通用、不可直接用于候选选择

9 个 fold × round 共记录 72 条检索结果；每轮 8 条。只有 1 种检索 claim 集合，说明查询和返回内容高度重复。全部 72/72 条均为 `selection_eligible=false`，包含 GB1、39/40/41/54 位点或具体残基方向的 target-specific claim 数为 0。

固定检索问题为：

> optimization objective maximize; protein structure and stability; binding interface mutation effects; physicochemical substitution mechanisms; epistasis and residue interactions; protein property optimization; kg

检索内容主要落在以下类别：

| 类别 | 出现次数 | 对当前闭池选择的作用 |
|---|---:|---|
| `interpretation_or_future_combination` | 18 | 提示上位性或后续组合实验，只能约束解释强度 |
| `library_design_not_current_candidate_score` | 18 | 用于文库设计，不能区分本轮32个候选的fitness |
| `requires_structure_or_site_validation` | 9 | 需要结构或热点验证，当前四位点闭池没有新增结构证据 |
| `requires_unavailable_stability_assay` | 27 | 需要稳定性、表达或可溶性读出，当前实验没有该通道 |

### 被反复检索的原始 claim

| Claim | 次数 | Mean confidence | 原文 |
|---|---:|---:|---|
| `de:claim:pairwise-epistasis-is-widespread` | 9 | 0.825 | Large-scale double-mutant measurements can reveal widespread positive and negative pairwise epistasis across a protein domain. |
| `binding:claim:validate-computational-hotspots-experimentally` | 9 | 0.794 | Use computational alanine scanning or interface-energy scores to prioritize sites, then validate those sites experimentally before allocating the focused library budget. |
| `binding:claim:preserve-structural-core-during-interface-diversification` | 9 | 0.782 | Concentrate affinity-maturation diversity at experimentally permissive interface or adjacent positions and preserve buried structural-core residues unless stability is being co-optimized. |
| `binding:claim:coselect-affinity-and-stability` | 9 | 0.776 | Measure expression or folding stability during affinity maturation and retain compensatory mutations when affinity-enhancing substitutions destabilize the binding domain. |
| `binding:claim:map-permissive-interface-positions` | 9 | 0.774 | Use experimental alanine scanning or another single-substitution scan to identify interface positions that tolerate mutation before constructing a combinatorial affinity-maturation library. |
| `de:claim:prefer-stable-parent-for-broad-mutagenesis` | 9 | 0.752 | When candidate starting parents have comparable target activity, prefer the parent with greater verified stability for broad mutagenesis because it is more likely to retain folding after additional substitutions. |
| `binding:claim:combine-validated-mutations-in-secondary-libraries` | 9 | 0.751 | Build a secondary library from individually validated affinity or stability mutations across site groups, then remeasure the resulting combinations instead of assuming additive effects. |
| `binding:claim:use-wild-type-inclusive-reduced-alphabets` | 9 | 0.707 | At each diversified interface position, include the wild-type residue and a justified reduced amino-acid set when full saturation would exceed the experimentally screenable library size. |

## 3. RAG 如何进入 LLM Prompt

执行链为 `local_rag_retrieval.json` → `local_rag_evidence.json` → Prompt 顶层 `rag_claims` → Scientist `evidence_ids`/`preferred_residues` → `hypothesis_score` → Agent-UQ hypothesis-target arm。RAG 不直接写入 fitness 或 predictor score，只能通过改变 Scientist 假设间接影响选择。

RAG Prompt 平均为 42711 tokens，base 平均为 28108 tokens，约为 1.52 倍；平均增加 14603 tokens。Prompt 每轮平均包含 12.0 张 RAG claim cards，但只有 4/9 轮的最终 Scientist 输出显式引用了至少一条 RAG 短 ID。未引用不等于完全没有上下文影响，但说明可审计的直接使用很有限。

RAG与base的四个位点 residue-set Jaccard 平均为 0.711；RAG每轮preferred-residue笛卡尔组合数平均为 37.4，base为 24.9。Scientist statement 文本相似度平均为 0.098。总体方向仍由可见GB1测量主导，RAG主要改变residue set的宽窄、措辞和置信边界，而没有产生稳定的新GB1特异证据。

更关键的是，这些soft preferences会在LLM之后、Agent-UQ之前先影响32-candidate pool。`KnowledgeCandidateGenerator`首先按每个variant命中多少个preferred positions排序，再按selection-eligible evidence score和确定性tie-break排序，最后从约119k候选截取32个。因此，语义上‘soft’的逐位偏好在大空间硬截断下具有近似门控效应。

## 4. Prompt 投影存在证据身份混合

原始检索每轮为 8 条，但 Prompt 中平均扩展为 12.0 张 cards，其中包括 KG 中已有的通用先验。9/9 轮都存在 `claim_text_mismatch_across_paths`，每轮12张cards中有6张带该warning。没有发现同一短ID跨多张card复用，但一张card内部可合并来自不同claim路径的source refs。

这意味着 LLM 看到的不是八条彼此独立、身份稳定的证据：同一statement可能合并多个不一致source refs。虽然运行时保留了warning，但Scientist仍可引用该短ID，导致‘引用闭合’不等于‘语义身份清晰’。这会增加错误归因风险。

## 5. 对候选选择的实际影响

| Fold | Round | Pool overlap | Selected overlap | RAG−base mean | RAG−base median | RAG−base best | Preferred-set similarity | RAG citations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 9/32 | 5/16 | +0.434 | +0.470 | +0.590 | 0.742 | 1 |
| 0 | 2 | 6/32 | 4/16 | +0.831 | +1.862 | +1.127 | 0.792 | 0 |
| 0 | 3 | 4/32 | 1/16 | +0.666 | +1.341 | -1.535 | 0.833 | 0 |
| 1 | 1 | 4/32 | 2/16 | -1.256 | -1.848 | +0.408 | 0.604 | 2 |
| 1 | 2 | 9/32 | 4/16 | -0.452 | -0.460 | +0.000 | 0.629 | 1 |
| 1 | 3 | 2/32 | 2/16 | -0.247 | -0.182 | -0.783 | 0.521 | 0 |
| 2 | 1 | 11/32 | 4/16 | +0.181 | +0.015 | +0.000 | 0.838 | 1 |
| 2 | 2 | 4/32 | 0/16 | +1.348 | +2.944 | +0.307 | 0.733 | 0 |
| 2 | 3 | 1/32 | 0/16 | +0.182 | +0.543 | +0.428 | 0.708 | 0 |

虽然 9/9 个fold-round使用相同seed、sampling namespace和`knowledge_filtered`策略，但没有任何一轮候选池完全相同（0/9）；Round 1也只重合4–11/32。结合生成器的确定性排序逻辑，这表明RAG引起的preferred-residue变化已经在Scientist输出之后、最终acquisition之前改写了候选池。后续轮次还叠加了前序入选差异。

### RAG偏好匹配数与wet结果

| 满足Scientist偏好的位点数 | n | wet mean |
|---:|---:|---:|
| 2 | 19 | 0.440 |
| 3 | 40 | 0.866 |
| 4 | 85 | 2.745 |

偏好匹配数与pooled wet mean呈明显递增：四个位点全部匹配的85个候选均值为2.745，而仅匹配2–3个位点的候选显著更低。这说明RAG/Scientist的逐位偏好确实能做粗粒度富集，也解释了RAG为何改善部分batch mean/median。问题在于85个全匹配候选内部仍有巨大组合差异，而当前匹配分数只计数位点、不编码配对或四位点上位性，因此难以稳定找到最高峰。

## 6. 为什么没有得到预期提升

1. **检索问题没有候选锚点。** 查询只包含‘结构、稳定性、界面、理化、上位性’等宽泛主题，没有当前32个variant、已观测反例或本轮待区分的残基组合。返回结果因此高度重复。
2. **知识粒度与决策粒度错位。** 文献claim回答‘应如何设计或验证实验’，Agent-UQ需要的是‘本轮哪些完整四位点组合更可能提高GB1 fitness’。所有claim都被正确标为不可直接选择，但系统仍允许其改变假设方向。
3. **现有实验数据已经提供了更强的方向。** base Prompt中的96+轮次观测已经清楚支持40位芳香残基、41G等模式；RAG主要重复‘注意上位性、验证稳定性、保留WT’等常识，新增信息边际很小。
4. **RAG影响是间接且未经校准的，但作用被候选池截断放大。** 通用claim可改变`preferred_residues`；候选生成器先按逐位匹配数从约119k候选截到32个，之后才进入`hypothesis_score`和8个hypothesis-target名额。RAG confidence不是候选fitness效应，也没有GB1 selection calibration。
5. **证据融合降低了信号清晰度。** claim-text mismatch和单张card内的source-ref混合增加了Prompt长度与语义歧义；更多上下文没有等比例增加可行动信息。
6. **信息接口无法表达RAG自己检索到的上位性知识。** Prompt反复出现‘epistasis is widespread/background-dependent’，但Scientist输出与`_hypothesis_matches`仍把四个位点表示为独立residue sets并逐位相加。结果可以提高平均富集，却会在全匹配组合之间随机tie-break，错过像`LWAA`这类特定组合高峰。

## 7. 证据边界

该审计证明RAG内容进入了Scientist可见Prompt，并在部分轮次被引用、改变了假设和候选选择；但不能把RAG与base的所有差异解释为纯RAG因果效应。后续轮次候选池会因先前选择而分叉，LLM调用也不是严格配对重复。更严格的验证应固定每轮32-candidate pool、复用同一visible-observation snapshot，并对同一Prompt执行RAG on/off配对重放。

## 8. 建议的修正优先级

1. 将查询改为round-specific：纳入当前候选组合、已观测支持/反例、需要区分的competing directions。
2. RAG默认只生成解释与实验建议；只有通过GB1或任务级校准的claim才能进入 `hypothesis_score`。
3. 修复claim/source/evidence-ID一对一身份，出现 `claim_text_mismatch_across_paths` 时禁止该claim影响selection。
4. 从逐位preferred residues升级为候选级、组合级的支持/反证表示，并显式建模上位性。
5. 对RAG进行no-answer门控：若检索内容不能区分当前候选，则返回‘解释性上下文’，不扩张selection prior。

## Prompt案例

### largest_negative_batch_delta：fold 1 / round 1

- RAG−base batch mean：-1.256
- 候选池重合：4/32；入选重合：2/16
- Scientist显式引用的RAG IDs：E34, E37
- base preferred residues：`{"39": ["L", "I", "M"], "40": ["Y", "W", "F", "H", "A", "I", "V", "N"], "41": ["G"], "54": ["A", "C"]}`
- RAG preferred residues：`{"39": ["I", "L"], "40": ["Y", "F"], "41": ["G"], "54": ["A"]}`
- 原始检索：`artifacts/hierarchical-scientist-kg_base_kg_base_rag_kg_base_al/runs/knowledge_agent-s11-f01-GB1-hierarchical-kg_base_rag-f01-20260820T221240262683Z/round_01/local_rag_retrieval.json`
- 完整Scientist Prompt：`artifacts/hierarchical-scientist-kg_base_kg_base_rag_kg_base_al/runs/knowledge_agent-s11-f01-GB1-hierarchical-kg_base_rag-f01-20260820T221240262683Z/round_01/llm/scientist/conversations/00001_reasoning_draft_attempt-00.json`
- 可复制的精简Prompt与输出已保存到对应 evidence-case JSON；provider reasoning_content 未复制。

该负向案例中，RAG输出显式引用的是‘pairwise epistasis widespread’与‘epistasis background-dependent’，二者都没有提供Y/F、I/L或A的候选级方向；但最终preferred sets从base的较宽集合收缩为39={I,L}、40={Y,F}、41={G}、54={A}，候选池随之只与base重合4/32。该收缩缺少RAG claim到具体残基的可验证蕴含关系，却被确定性候选池截断放大。

### largest_positive_batch_delta：fold 2 / round 2

- RAG−base batch mean：+1.348
- 候选池重合：4/32；入选重合：0/16
- Scientist显式引用的RAG IDs：none
- base preferred residues：`{"39": ["L", "I", "C"], "40": ["Y", "W", "F"], "41": ["G"], "54": ["V"]}`
- RAG preferred residues：`{"39": ["L", "I", "C"], "40": ["Y", "W", "F", "H", "A"], "41": ["G"], "54": ["C", "A", "V"]}`
- 原始检索：`artifacts/hierarchical-scientist-kg_base_kg_base_rag_kg_base_al/runs/knowledge_agent-s11-f02-GB1-hierarchical-kg_base_rag-f02-20260820T222600583330Z/round_02/local_rag_retrieval.json`
- 完整Scientist Prompt：`artifacts/hierarchical-scientist-kg_base_kg_base_rag_kg_base_al/runs/knowledge_agent-s11-f02-GB1-hierarchical-kg_base_rag-f02-20260820T222600583330Z/round_02/llm/scientist/conversations/00067_reasoning_draft_attempt-00.json`
- 可复制的精简Prompt与输出已保存到对应 evidence-case JSON；provider reasoning_content 未复制。

该正向案例中，RAG batch mean高出1.348，但Scientist没有显式引用任何RAG短ID；其方向主要来自当轮已揭示GB1观测。因此，这个改善不能作为外部检索claim带来收益的直接证据。

