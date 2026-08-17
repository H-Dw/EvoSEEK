# KnowledgeEngine 蛋白特征理解、证据校准与动态配置重构方案

> 状态：设计与实施计划  
> 审计基线：2026-08-17 当前工作区代码  
> 研究范围：KnowledgeEngine、蛋白理化性质、MSA/保守性、三维结构、运行期 KG 证据、Scientist 可查询工具、GB1 解耦和可学习参数  
> 目标：把当前适合演示的确定性启发式，重构为“有来源、有适用域、有质量控制、有不确定性、可缓存、可校准”的蛋白特征证据系统；不改变 wet validation 的最终科研权威性。

## 1. 结论先行

当前 KnowledgeEngine 生成的 statement 确实存在显著误差风险，但四个通道的风险程度不同：

1. physchem 计算本身是可重复的，但它只是一个人为定义的四维残基距离。把这个距离映射成“理化保守性”，再进一步作为 fitness 支持证据，没有经过任务或 assay 校准。它可以保留为低保真描述符，不能作为独立 fitness 结论。
2. conservation 不是从 MSA 得到的保守性，而是人工 tolerated list 上的 ±0.5 打分。它只适合作为 GB1 测试配置或专家先验，不能继续以通用 conservation provider 的名义运行。
3. structure 没有读取 PDB/mmCIF、原子坐标、链、界面或结构置信度，只把每个位点的人工 structure_risk 取平均。它不具备三维结构理解能力，当前 statement 中的 “precomputed structural tolerance” 容易产生不应有的权威感。这一通道应立即标为 heuristic/test-only，默认不得进入生产候选排序。
4. kg 使用当前可见的 wet observation 计算残基均值，基本数据来源是可信的；但它把位点边际均值当作可加和效应，在 GB1 这种强上位性景观中会混入背景序列和采样策略偏差。它应保留为“当前运行内的描述性关联”，不能解释为因果或可迁移知识。

这些误差会影响下游：statement 会进入 KG 和 Scientist 上下文，score/confidence 会直接参与 candidate ranking，Scientist 的 preferred_residues 又会改变候选生成，随后被测到的样本再回写 KG，可能形成自增强偏差。现有 hard validation、approval 和 wet reveal 能防止计算证据替代湿实验事实，因此风险主要表现为探索效率下降、候选多样性受损和假设叙事失真，而不是直接篡改 wet truth。

推荐的目标架构不是让 KnowledgeEngine 自行调用 LLM “理解蛋白”，而是：

- KnowledgeEngine 继续是确定性、可审计的证据编排器。
- 专业代码或外部科学工具负责产生理化、进化和结构特征。
- LLM 只在经过权限和轮次过滤的结构化证据上做假设综合，不负责创造底层特征。
- 每个证据同时携带原始特征、来源、版本、适用域、数据质量、不确定性和校准状态。
- 所有 GB1 常量及经验数值从运行时代码移除，由 Task/Knowledge/Structure 配置和版本化资源动态注入。

## 2. 当前实现的事实审计

### 2.1 KnowledgeEngine 当前是什么

主实现位于 src/fitness_agents/knowledge/engine.py。它是一个有状态的领域服务：

- 持有本轮已观察 Variant、FitnessObservation、ValidationRecord 和 ReThinkReflection。
- 将 observation、prediction、evidence、hypothesis 和 validation 同步到运行 KG/structured KG。
- 注册 physchem、conservation、structure、kg 四个 provider。
- 为候选生成 Evidence，并对 evidence score 做 confidence 加权平均。
- 暴露 AgentKnowledgeGraphTool，让受控的 KG Operator 查询已持久化信息。

它目前不调用 LLM，也没有真正调用外部蛋白分析软件。这个职责边界总体是正确的，应保留。

### 2.2 四个 provider 的精确规则

| 通道 | 当前规则 | 当前来源/置信度 | 科学评价 |
|---|---|---|---|
| physchem | 对 “VDGV” 与候选残基的 hydropathy、side-chain volume、charge、polarity group 做归一化欧氏距离；用人工尺度 [9, 170, 2, 1]，再经 clip=2 映射到 1 - distance/2 | aa_properties:v1；0.65 | 可作低成本 mutation descriptor；不能代表局部结构、结合、稳定性或 assay fitness |
| conservation | 从 site_profiles[position].tolerated 查表；在表内 +0.5，否则 -0.5；无突变时 0.5 | gb1_site_profile:v1；0.55 | 不是 MSA conservation；仅是人工分类测试桩或显式专家先验 |
| structure | 对突变位点的 structure_risk 求均值，再做 1 - mean；无突变时 0.5 | 5LDE_site_risk:v1；0.45 | 没有读取 5LDE 或任何坐标；不是三维结构分析 |
| kg | 从历史可见 observation 得到 position-residue 均值与计数；以 count/(count+3) 收缩到全局均值，跨位点平均后 tanh | observation_graph:current_run；min(0.85, 0.25+0.03×support) | 可作运行内关联统计；不是因果效应，且忽略上位性、采样偏差和异方差 |

### 2.3 当前最严重的语义问题：不同 score 并不可直接平均

KnowledgeEngine.scores 和 mutation/uncertainty.py 会按 confidence 对各通道 score 求加权平均。但当前四种 score 并没有共同的统计语义：

- physchem 通常落在 0 到 1 附近。
- conservation 使用 -0.5 与 +0.5，且 WT/无突变返回 +0.5。
- structure 使用 0 到 1，且 WT/无突变返回 +0.5。
- kg 经 tanh 后处于 -1 到 1。

因此即使每个规则内部完全无 bug，合并后的 knowledge score 仍不代表概率、标准化效应、期望 fitness 或统一效用。正值通道还会对排序形成结构性偏置。confidence 在这里同时承担“证据可靠度”和“合并权重”两种职责，也没有校准依据。

目标实现必须先定义统一语义，推荐两种可选方案：

- 方案 A：各 provider 输出独立 feature 和 uncertainty，由候选选择器在公开训练数据上学习 assay-specific combiner。
- 方案 B：每个 provider 先校准成相同语义，例如“相对 WT 为有利的概率”或标准化预期效应，再由受约束的融合器组合。

在缺少校准数据时，应采用方案 A 的保守形态：保留独立证据，不生成一个伪精确的总 knowledge score。

### 2.4 误差如何传播到下游

当前传播链为：

~~~text
人工规则
  → Evidence(statement, score, confidence)
  → KG 持久化与 top_knowledge_evidence
  → Scientist 假设、preferred_residues、evidence_ids
  → candidate generator / AgentUncertaintySelector
  → dry validation 与被选中的 wet batch
  → observation 回写 KG
  → 下一轮 residue statistics 与 Scientist context
~~~

具体影响包括：

- KnowledgeCandidateGenerator 会用 evidence score 排序。
- AgentUncertaintySelector 的 utility 含 evidence_weight × evidence_score。
- Scientist 会看到 statement，并可能把人为规则误当成实验或结构结论。
- structured KG 会长期保存这些 statement；如果来源与质量字段不足，测试启发式会被“永久化”。
- 选择偏差会改变后续 wet 数据分布，使 KG residue mean 进一步偏向早期规则推荐的区域。
- GB1 多位点上位性会使边际 residue mean 对未观测背景的外推失真。

缓冲机制也要明确：

- hard validator 与 approval 仍在候选进入实验前执行。
- oracle/final-test 不由 KnowledgeEngine 直接调用。
- wet evidence 仍是唯一高保真科研状态。
- 因而当前问题是证据质量和实验设计效率问题，不是状态机权限失控。

## 3. 哪些确定性规则合理，哪些只适用于测试

### 3.1 应保留的确定性机制

| 机制 | 是否保留 | 原因与改进 |
|---|---|---|
| provider 可独立启停 | 保留 | 便于消融、故障隔离和按资源降级 |
| typed Evidence | 保留并扩展 | 需增加 raw_features、quality、uncertainty、applicability、calibration 和 provenance |
| 稳定 evidence_id | 保留 | 但 ID 应加入 provider/config/resource/model 版本，避免不同算法生成同语义 ID |
| prediction/evidence/measurement 类型分离 | 必须保留 | 防止计算推断升级为观测事实 |
| round visibility | 必须保留 | MSA/结构外部资源可全程可见；campaign wet observation 仍按轮次可见 |
| KG residue statistics 的收缩思想 | 保留思想 | shrinkage 是合理的低样本稳定化方法，但参数、模型和上位性处理必须校准 |
| bounded confidence | 保留思想 | 防止计算证据表现为绝对确定；上限不应硬编码为 0.85 |
| heartbeat | 保留为运维机制 | 256 只是进度日志间隔，不是 evidence 数量上限，也不是科学参数 |
| deterministic provider | 保留 | 同输入、同版本应产生同输出；LLM 不进入底层特征计算 |

### 3.2 可以保留但必须降格的规则

1. AA_PROPERTIES：
   - 可以作为快速 mutation delta 的一组基础描述符。
   - 必须保留每一维名称和来源，不能先压成一个欧氏距离。
   - 体积、疏水性、电荷、极性属于不同量纲，不应靠人工 range 直接混合。
   - 电荷依赖 pH 和局部 pKa；His 尤其不能固定为 0.1 后当成普适值。
   - Gly、Pro、Cys 的特殊几何/二硫键作用无法由四维向量表达。

2. KG residue mean：
   - 可以作为“在当前可见数据中，某残基与 fitness 的边际关联”。
   - statement 必须说明 conditioning scope、背景序列分布、样本数和不确定性。
   - 单残基效应不能简单相加；至少增加 pairwise/epistasis 和 background-stratified 统计。
   - 不能与外部自然序列保守性混为同一类 evidence。

### 3.3 应标为 test-only 或删除的规则

- tolerated list 直接等同 conservation。
- 突变在 tolerated list 中固定 +0.5，否则固定 -0.5。
- 人工 structure_risk 直接等同 structural tolerance。
- 不读坐标却使用 5LDE_site_risk:v1 作为来源。
- WT/无突变在 conservation 和 structure 中固定得到 0.5。
- 固定 confidence：0.65、0.55、0.45。
- KG shrinkage_pseudocount=3.0、confidence_base=0.25、support_gain=0.03、confidence_cap=0.85。
- physchem scale=[9,170,2,1]、clip=2.0。
- 所有运行时 “VDGV” 与 (39,40,41,54)。

如果短期仍需兼容现有基线，应把旧实现重命名为 LegacyHeuristicProvider，并要求显式配置：

~~~yaml
mode: legacy_test_only
production_eligible: false
statement_prefix: "[UNVALIDATED HEURISTIC]"
contributes_to_selection: false
~~~

这样旧测试仍可复现，但不会默默进入生产证据融合。

## 4. 蛋白设计 Agent 文献给出的架构启示

### 4.1 文献结论

蛋白设计 Agent 的共同方向不是让通用 LLM 在 prompt 中手工估计氨基酸性质，而是让 Agent 编排专用工具：

- [ProteinMCP](https://pubmed.ncbi.nlm.nih.gov/41877581/) 将文献/知识、序列/MSA、fitness、结构预测、设计、性质预测、分子动力学和数据库拆成专用模块，由 Agent 统一编排 38 个工具。其公开实现的 fitness workflow 组合 MSA、PLMC/共进化、one-hot、ESM 和 ProtTrans，而不是依赖一张手工残基表。
- [ProtAgents](https://doi.org/10.1039/D4DD00013G) 强调不同 agent/profile 分别连接知识检索、结构分析、物理模拟和机器学习模型；LLM 的价值主要是拆解任务、组织工具和批判结果。
- [Agent Rosetta](https://arxiv.org/abs/2603.15952) 的关键发现是，仅靠 prompt engineering 往往不能稳定地产生正确 Rosetta action；环境设计、受约束接口和明确工具合同才是科学软件集成的核心。
- 2026 年的 [BioDesignBench 预印本](https://doi.org/10.64898/2026.05.06.723381) 发现，Agent 常能选择合适工具，但容易只看少数指标、缺少候选间比较并过早结束。强制多指标、成组比较和更深评估可改善表现。

对本系统的直接启示是：

1. KnowledgeEngine 应成为特征工具编排和证据标准化层，而不是用更多 prompt 替代科学计算。
2. Scientist 应查询结构化结果和反证，不能直接接收没有适用域说明的单一综合分。
3. 每个候选至少应有进化、结构兼容、物理能量和 assay/KG 关联等互补维度；缺失维度必须显式显示为 unavailable，而不是使用 0.5 填充。
4. 工具调用结果必须可比较、可追溯和可复现；LLM 不得直接决定科学参数。

### 4.2 证据等级建议

| 等级 | 示例 | 可用于什么 | 不可宣称什么 |
|---|---|---|---|
| D0 描述符 | AAindex delta、ProtParam delta | 描述突变改变了哪些理化性质 | 不可直接称为有利 fitness |
| E1 进化先验 | MSA entropy、PSSM、DCA、MSA Transformer | 自然序列可容忍性和共进化相容性 | 不等于特定 assay 的改造方向 |
| S1 静态结构 | RSA、邻居、界面、H-bond、clash | 局部环境与明显几何风险 | 不等于动态稳定性或 binding affinity |
| S2 模型结构 | ProteinMPNN/ESM-IF likelihood、AF confidence | sequence-backbone compatibility | 不等于真实折叠和功能 |
| P2 物理/经验能量 | Rosetta/FoldX ΔΔG | 稳定性或结合能的 dry prior | 不等于 wet fitness，不能无校准地跨蛋白比较 |
| K2 运行 KG | 已揭示 observation 的关联统计 | 当前 assay 内经验关联 | 不等于因果或外部可迁移规律 |
| W3 wet | 当前 assay 的真实实验结果 | 科研状态与最终评估 | 仍需保留实验噪声、批次和测量条件 |

## 5. 更合理的氨基酸理化性质分析

### 5.1 不应继续使用单一“残基距离”

当前四维欧氏距离的问题包括：

- 量纲和尺度由人工范围决定。
- 所有性质被压成一个值，LLM 无法判断变化来自疏水性、体积还是电荷。
- 不考虑 assay pH、温度、离子强度、氧化还原环境。
- 不考虑位点是埋藏、表面、界面、活性中心还是柔性环。
- 不考虑邻近残基、盐桥、氢键、二硫键和 packing。
- 多突变被简单求均值，丢失补偿和协同效应。

### 5.2 建议的工具与职责

1. AAindexRepository
   - 从 [AAindex](https://www.genome.jp/aaindex/) 读取有文献来源的 amino-acid index、substitution matrix 和 pairwise contact potential。
   - 配置只选择少量与任务相关、低共线、来源明确的 accession。
   - 原始数据固定 hash；每一维保留 accession、单位/方向和引用。

2. PhyschemDeltaProvider
   - 输出每个 mutation 的命名 delta，而不是一个不可解释距离：
     hydropathy、side-chain volume、formal charge at assay pH、aromaticity、H-bond donor/acceptor class、flexibility/turn propensity、helix/sheet propensity、Gly/Pro/Cys flags。
   - 多突变同时输出 sum、max、site-wise vector 和 pairwise interaction flags。
   - 标准化参数从公开训练集或当前 task 的 public training split 拟合；禁止读取 oracle/final-test。

3. ProtParamAdapter
   - [ExPASy ProtParam](https://web.expasy.org/protparam/protparam-doc.html)、[Biopython ProtParam](https://biopython.org/docs/latest/api/Bio.SeqUtils.ProtParam.html) 或 [EMBOSS pepstats](https://emboss.bioinformatics.nl/cgi-bin/emboss/help/pepstats) 可计算 molecular weight、pI、composition、extinction coefficient、instability index、aliphatic index、GRAVY 等。
   - 应计算 WT 与 mutant 的 delta，用于全序列可制造性、纯化和 developability 描述。
   - 这些全局指标对 GB1 四个位点 fitness 通常只是辅助信息，不能替代局部结构分析。

4. IonizationProvider
   - 对结构可用的任务，可调用 [PROPKA](https://github.com/jensengroup/propka) 估计局部环境下的 pKa。
   - 输入必须包括结构、链、assay pH；输出为预测值和模型适用域。
   - PROPKA 本身也是经验模型，仍需标为 computed prediction。

5. LocalEnvironmentJoiner
   - 把理化 delta 与 RSA、二级结构、界面、邻居、氢键和电荷网络连接。
   - 示例：疏水性增加在埋藏核心可能有利于 packing，在暴露表面可能降低溶解性，在结合界面则取决于 partner 环境。

### 5.3 推荐输出合同

~~~python
@dataclass(frozen=True)
class PhyschemEvidencePayload:
    mutation: str
    assay_conditions: dict[str, float | str]
    raw_deltas: dict[str, float]
    special_flags: tuple[str, ...]
    local_context: dict[str, float | str | bool] | None
    applicability: str
    uncertainty: float | None
    source_refs: tuple[str, ...]
~~~

statement 应由固定模板生成，例如：

> D40W increases side-chain volume and hydrophobicity under the selected AAindex scales. The site is interface-exposed in structure resource X; this is a descriptor and not an affinity claim.

这比 “mean physicochemical conservativeness=0.734” 更可解释，也更不容易被 LLM 误读。

## 6. conservation 应重构为动态 MSA 分析

### 6.1 是否只在第一轮计算一次

原则上合理，但更准确的时机是 campaign prepare/bootstrap 阶段，即第一轮 Scientist 运行之前：

- 用目标蛋白的完整 WT 序列或明确 domain 序列搜索同源序列。
- 搜索、过滤、聚类、对齐和质量评估只做一次。
- 后续每轮复用同一 MSA、sequence weights、position mapping、PSSM/entropy 和可选共进化模型。
- 对每个候选只重新计算其相对于缓存模型的 variant score。

不能用四字符 “VDGV” 去做 MSA 搜索。必须用完整蛋白/结构 domain 序列，再把配置中的 mutable_positions 映射到 alignment columns。

### 6.2 可复用的必要条件

缓存键至少包含：

~~~text
reference_sequence_sha256
protein/domain boundaries
sequence database name + release/hash
search tool + version
E-value / coverage / identity thresholds
clustering and sequence-weighting parameters
alignment tool + version
chain/complex pairing mode
taxonomy filters
position-map version
~~~

以下情况必须重新计算：

- 更换 reference sequence、domain 边界或 residue numbering。
- 任务包含 indel，导致 alignment coordinate mapping 改变。
- 更换数据库版本或搜索/聚类参数。
- 从单体任务切换到复合物/paired-MSA 任务。
- MSA 质量不达标，需要扩大数据库或更换搜索策略。
- 突变改变了被研究的蛋白家族定义，而不再是同一 WT 邻域内的点突变任务。

### 6.3 推荐流水线

1. SequenceResourceResolver
   - 解析 WT 全长序列、domain、链和 mutable position map。
2. HomologySearchAdapter
   - 首选 [MMseqs2](https://www.nature.com/articles/nbt.3988)；高敏感备选 [HHblits](https://www.nature.com/articles/nmeth.1818)。
   - [ColabFold](https://www.nature.com/articles/s41592-022-01488-1) 证明 MMseqs2 可用于快速构建多样 MSA，并强调 MSA 深度与多样性质量。
3. MSAFilterAndCluster
   - 按覆盖率、identity、gap、低复杂度、重复序列和 taxonomy 过滤。
   - 聚类或 sequence reweighting，避免近缘序列重复造成伪保守。
4. MSAQualityAnalyzer
   - 输出 raw depth、Neff、每列 coverage/gap、identity distribution、taxonomic diversity。
   - 质量不足时不得生成高 confidence conservation statement。
5. ConservationProfileProvider
   - 输出每位点 entropy、WT frequency、mutant frequency、log-odds/PSSM、gap fraction 和置信区间。
6. EpistasisProvider
   - 对深度足够的 MSA，拟合 pairwise Potts/DCA 或 PLMC。
   - [EVmutation](https://doi.org/10.1038/nbt.3769) 表明显式建模位点依赖可优于仅保守性方法。
7. OptionalEvolutionModelProvider
   - MSA Transformer 可把 row/column attention 用于对齐序列的结构与功能表示：[ICML 2021](https://proceedings.mlr.press/v139/rao21a/rao21a.pdf)。
   - EVE 展示了从进化序列拟合生成模型以评估 variant effect 的思路：[Nature 2021](https://doi.org/10.1038/s41586-021-04043-8)。
   - 如果 MSA 过浅，应降级到单序列 PLM 并显式降低证据等级。

### 6.4 对 GB1 多突变的特别要求

GB1 四位点景观存在强 epistasis。仅用四个位点的独立 conservation score 会遗漏“单个位点看似可容忍，但组合不可容忍”以及补偿性组合。

建议同时输出：

- independent_score：各 mutation 的 PSSM/log-odds 和。
- epistatic_score：Potts/DCA 或 MSA model 对完整 mutant sequence 的能量差。
- observed_assay_score：当前运行 KG 的 assay-specific 关联。

三者必须分开显示。自然进化保守性、共进化相容性和 GB1 binding assay 不是同一个目标。

## 7. fitness 所需的三维结构特征

### 7.1 structure provider 至少要理解什么

结构相关的 fitness 分析不能只问“这个位点风险是多少”，而应按任务目标分析以下特征：

1. 结构资源与质量
   - experimental/predicted、PDB/mmCIF ID、链、分辨率。
   - predicted structure 的 pLDDT/PAE 或等价置信度。
   - 缺失残基、alternate location、非标准残基、配体、金属、辅因子。
2. residue mapping
   - task position、UniProt position、PDB chain/residue、alignment column 的双向映射。
   - insertion code、signal peptide、mature chain 和 domain offset。
3. 局部几何
   - backbone φ/ψ/ω、secondary structure、Cα/Cβ 邻居距离、contact number。
   - residue depth、SASA/RSA、burial、packing density、cavity。
4. 相互作用
   - backbone/side-chain H-bond、salt bridge、disulfide、aromatic、cation-π。
   - van der Waals overlap/clash、unsatisfied polar atoms。
5. 界面与功能位点
   - partner chain contact、buried interface area、interface contact density。
   - 到 ligand/cofactor/metal/catalytic residue 的距离。
   - binding hotspot 或已知 functional annotation。
6. 柔性与不确定性
   - B-factor、pLDDT、PAE、NMR/多构象 ensemble 方差。
   - mutation 在不同 conformer 上结果是否一致。
7. 变异影响
   - WT 与 mutant 的局部 packing、H-bond、SASA、electrostatics 变化。
   - folding ΔΔG、binding ΔΔG、clash 和 rotamer strain。
8. sequence-backbone compatibility
   - ProteinMPNN 或 ESM-IF 的 residue/sequence log-likelihood delta。

[ProteinMPNN](https://doi.org/10.1126/science.add2187) 用 N、Cα、C、O 和 virtual Cβ 原子间距离构建结构图并进行 message passing，这体现了“理解结构”需要真实几何表示，而不是位点标签。其论文也使用邻近 Cβ 距离表征埋藏程度。[Geometric Vector Perceptrons](https://arxiv.org/abs/2009.01411) 和 [ESM-IF1](https://icml.cc/virtual/2022/oral/16886) 同样以旋转/平移不变的几何特征进行结构学习或 inverse folding。

### 7.2 建议的代码模块

~~~text
src/fitness_agents/protein_features/
  context.py
  contracts.py
  cache.py
  provenance.py

  sequence/
    aaindex_repository.py
    physchem_provider.py
    protparam_adapter.py

  evolution/
    msa_search.py
    msa_filter.py
    msa_quality.py
    conservation_provider.py
    epistasis_provider.py

  structure/
    resource_resolver.py
    residue_mapper.py
    quality.py
    static_features.py
    interactions.py
    mutation_environment.py
    inverse_folding.py
    rosetta_adapter.py
    foldx_adapter.py
    calibrator.py

  tools/
    registry.py
    execution.py
    schemas.py
~~~

模块职责：

- StructureResourceResolver：只负责解析配置、校验 hash、选择 experimental/predicted resource。
- ResidueMappingService：建立 task/sequence/PDB/alignment 坐标映射；任何映射不确定性都必须阻止高置信结构证据。
- StructureQC：检查链、缺失原子、分辨率、pLDDT/PAE、配体和 assembly。
- StaticStructureFeatureExtractor：用 Bio.PDB/MDAnalysis 等解析坐标；用 DSSP 类工具计算二级结构；用 [FreeSASA](https://pmc.ncbi.nlm.nih.gov/articles/PMC4776673/) 计算 SASA/RSA。
- InteractionFeatureExtractor：计算 contact、H-bond、salt bridge、disulfide、界面和明显 clash。
- MutationEnvironmentComparator：生成 WT→mutant 的局部环境 delta；不得仅复制 WT 位点特征。
- InverseFoldingAdapter：调用 ProteinMPNN/ESM-IF，输出结构条件下的序列兼容性。
- RosettaAdapter/FoldXAdapter：封装外部可执行程序、输入快照、版本、随机种子、重复数、超时和日志。
- StructureEvidenceCalibrator：把工具输出校准到当前任务或公开 benchmark；未校准时只输出 raw dry metric。

### 7.3 结构 statement 的最低要求

任何 structure statement 必须回答：

- 使用了哪个结构、哪条链、哪个 assembly？
- task position 如何映射到 structure residue？
- 是 WT 静态环境、mutant 重建结果，还是能量模型结果？
- 使用了哪些原子/构象？
- 结构质量与模型适用域是什么？
- 结论是 folding、binding、packing，还是仅几何描述？
- uncertainty 来自哪里？

如果缺少这些字段，provider 应返回 unavailable/insufficient_quality，不应返回默认 0.5。

## 8. 是否调用 Rosetta，以及如何调用

### 8.1 可以调用，但不应把它当成“理化属性表”

Rosetta 适合回答上下文相关问题，例如：

- 突变后局部 packing 和 side-chain rotamer 是否合理。
- folding ΔΔG 是否明显不利。
- 蛋白复合物界面 binding ΔΔG 是否变化。
- 哪些 score term 导致总能量变化。

[Rosetta REF2015](https://doi.org/10.1021/acs.jctc.7b00125) 包含 van der Waals attraction/repulsion、solvation、electrostatics、H-bond、torsion、rotamer 和 residue reference 等项；[官方 score type 文档](https://docs.rosettacommons.org/docs/latest/rosetta_basics/scoring/score-types) 可用于固定 scorefunction 和 term 解释。[cartesian_ddG 文档](https://docs.rosettacommons.org/docs/latest/cartesian-ddG) 也明确要求比较 relaxed WT 与 mutant。

Rosetta 不适合替代 AAindex/ProtParam 来回答“某个氨基酸一般是否疏水”。两者职责不同。

### 8.2 推荐成本分层

| 层级 | 工具 | 运行时机 |
|---|---|---|
| Tier 0 | AAindex/ProtParam delta | 全候选，CPU，低成本 |
| Tier 1 | 缓存 MSA 的 PSSM/entropy/epistasis | 全候选，首次准备后复用 |
| Tier 2 | static structure + ProteinMPNN/ESM-IF | 全候选或初筛后候选 |
| Tier 3 | FoldX/Rosetta ΔΔG、interface analysis | top-N 候选，dry validation 前后 |
| Tier 4 | MD/ensemble free-energy 类分析 | 少量 finalist 或专门科学问题 |
| Tier 5 | wet validation | 唯一高保真状态更新 |

### 8.3 Rosetta/FoldX 的边界

- [FoldX](https://doi.org/10.1093/nar/gki387) 是适合较高通量 mutation scan 的经验 force field，可作为 Rosetta 前的成本较低层。
- [Potapov 等人的 benchmark](https://doi.org/10.1093/protein/gzp030) 显示多种稳定性方法能把握平均趋势，但单个 mutation 的精确值仍可能明显错误。
- 所以 ΔΔG 必须标为 dry prediction；不能把 Rosetta Energy Unit 直接写成实验 kcal/mol，也不能把 folding stability 直接写成 binding fitness。
- 多突变要在完整组合上重建和评分，不能只相加单突变 ΔΔG。
- 需要多次 relax/repack、固定随机种子集合并报告分布。
- 对预测结构，应在多个可信 conformer 上重复，避免对单一坐标过拟合。
- external binary 版本、数据库、flags、scorefunction、license 和容器 digest 必须写入 artifact。
- Scientist 不应直接获得 shell 或 batch submission 权限；它只能查询已批准分析的结果，或提交 request_analysis proposal 交由 Runner 审批执行。

## 9. KG 与新特征工具的关系

### 9.1 当前关系

当前 KnowledgeEngine._kg 直接调用 ObservationKnowledgeGraph.residue_statistics，不经过 KGInteractionController。这是可信的内部读路径。Scientist 侧则通过 AgentKnowledgeGraphTool、Operator registry 和 KGInteractionController 查询受轮次约束的信息。

这两个路径不应混为一体：

- provider path：由系统在明确阶段批量计算并写入 evidence。
- agent query path：Scientist 只读取已经计算、已经通过可见性过滤的 evidence。

### 9.2 推荐新增 Operator

| Operator | 输入 | 返回 | 约束 |
|---|---|---|---|
| query_physchem_delta | variant_id | 命名 delta、assay condition、来源 | 只读、max rows |
| query_evolutionary_profile | variant_id/position | PSSM、entropy、Neff、epistasis | 缓存资源版本固定 |
| query_structure_environment | variant_id/position | mapping、RSA、contacts、interface、quality | 只返回允许的结构资源 |
| query_mutation_energy | variant_id | FoldX/Rosetta component、replicates、uncertainty | 只读已完成任务；不能触发 batch |
| compare_feature_evidence | variant_ids | 同尺度多维比较 | 数量上限；必须含缺失项 |
| query_evidence_provenance | evidence_id | config/resource/model/hash/calibration | 所有 statement 可追溯 |
| query_counterevidence | variant_id/hypothesis_id | 相反方向或低质量证据 | 不得只返回支持证据 |

继续由 KGInteractionController 执行：

- tool whitelist。
- variant scope。
- max rows。
- round visibility。
- query budget。
- raw SQL 禁止。
- query trace。

Scientist 可以获得辅助知识，但不应直接运行 Rosetta、访问 oracle、读取 final-test 或修改 KG。

## 10. GB1 硬编码的系统性重构

### 10.1 当前硬编码范围

本次代码扫描发现至少包括：

- knowledge/engine.py：VDGV、(39,40,41,54)、AA_PROPERTIES、固定结构来源、所有通道经验常量。
- knowledge/graph.py：_WT_CODE 和 _POSITIONS。
- mutation/generators.py：position→variant index。
- mutation/uncertainty.py：position→variant index。
- agents/llm.py 与当前 Scientist profile：固定四个位点。
- models/ensemble.py：对 VDGV 计算 Hamming distance。
- configs/model/kermut.yaml 与部分 module test：GB1 resource positions。
- data/gb1.py：GB1 专用常量；这部分可保留在明确命名的 GB1 dataset adapter 中，但不能泄漏到通用运行时模块。

现有 TaskConfig 已包含 wild_type_sites 与 mutable_positions，但 Runner 创建 KnowledgeEngine 时只传 assay_id/protein_id，没有注入这两个字段。这是重构的直接切入点。

### 10.2 新增统一 ProteinTaskContext

~~~python
@dataclass(frozen=True)
class ProteinTaskContext:
    task_id: str
    protein_id: str
    assay_id: str
    full_sequence: str
    mutable_positions: tuple[int, ...]
    wild_type_residues: tuple[str, ...]
    position_to_variant_index: dict[int, int]
    numbering_scheme: str
    assay_conditions: AssayConditions
    structure_resources: tuple[StructureResource, ...]
    sequence_resource: SequenceResource
~~~

由 ProteinTaskContextFactory 在 campaign 初始化时一次构造并校验：

- mutable_positions 长度等于 wild_type_residues 长度。
- 所有位置唯一且可映射到 full_sequence。
- compact variant 与 full sequence 的投影一致。
- 结构链和 residue mapping 可解析。
- resource hash 与版本存在。

Runner 将同一个不可变 context 注入：

- KnowledgeEngine/各 provider。
- ObservationKnowledgeGraph。
- candidate generators 和 selectors。
- Scientist sanitized context/profile renderer。
- hard validator 和 artifact writer。

任何模块不得再自行声明 GB1 position map。

### 10.3 配置拆分

~~~yaml
task:
  task_id: gb1_binding
  protein_id: GB1
  assay_id: binding
  sequence_resource: configs/resources/sequences/gb1.yaml
  mutable_positions: [39, 40, 41, 54]
  wild_type_sites: VDGV
  numbering_scheme: domain
  assay_conditions:
    pH: 7.4
    temperature_c: 25.0
    ionic_strength_mM: null

structure:
  resources:
    - resource_id: gb1_complex_experimental_v1
      kind: experimental
      path: data/structures/...
      format: mmcif
      chain: A
      partner_chains: [B]
      assembly_id: 1
      sha256: ...
      residue_map: configs/resources/mappings/...
~~~

这里的示例条件不能以猜测值提交生产配置；未知值必须为 null，并在 evidence applicability 中降级。

## 11. 所有经验数值动态配置化与“可学习”标记

### 11.1 参数分为三类

1. scientific_parameter
   - 影响证据数值或科学结论。
   - 例如 shrinkage、confidence mapping、score clip、MSA thresholds、structure cutoff。
2. policy_parameter
   - 影响选择和资源分配。
   - 例如 evidence_weight、top-N Rosetta、tool budget。
3. operational_parameter
   - 只影响性能/日志，不影响结果。
   - 例如 evidence heartbeat interval=256。

只有前两类可考虑学习。heartbeat 256 可配置，但不应标为可学习科学参数。

### 11.2 统一参数合同

~~~python
class LearnableParameterSpec(BaseModel):
    value: float
    category: Literal["scientific", "policy", "operational"]
    status: Literal[
        "fixed_by_definition",
        "literature_default",
        "expert_prior",
        "calibrated",
        "learned",
        "experimental_override",
    ]
    learnable: bool = False
    bounds: tuple[float, float] | None = None
    transform: Literal["identity", "log", "logit"] = "identity"
    prior: dict[str, float | str] | None = None
    update_policy: str = "never"
    min_evidence: int = 0
    source: str
    version: str
~~~

### 11.3 当前数值的迁移表

| 当前数值 | 含义 | 迁移建议 |
|---|---|---|
| [9,170,2,1] | physchem 人工归一化尺度 | 删除；按命名特征在 public train/reference population 拟合 scaler |
| 2.0 | radicality clip | 删除或配置；优先不压成单一 score |
| 0.65/0.55/0.45 | provider confidence | 删除固定值；由质量与校准模型计算 |
| ±0.5 | tolerated list score | legacy_test_only；生产删除 |
| 0.5 | 无突变或缺失结构默认 | 改为 neutral only when mathematically defined；缺失必须是 unavailable |
| 3.0 | KG pseudo-count | scientific、learnable；按公开训练/已揭示 wet 校准 |
| 0.25 | KG confidence base | scientific、learnable，或改为 uncertainty model |
| 0.03 | 每 support 的 confidence gain | scientific、learnable；线性模型通常不合理，建议用 posterior/calibration |
| 0.85 | KG confidence cap | scientific/policy；不再硬编码，可从 held-out calibration 学习 |
| 1.0 | validation prior shrinkage | 一并进入参数审计 |
| 0.20 | knowledge soft_weight | policy、learnable；必须避免 oracle/final-test |
| 0.65 | generation evidence_weight | policy、learnable；与 evidence calibration 联合评估 |
| 0.85 | hypothesis recency weight | policy、learnable；从 mutation/uncertainty.py 移出 |
| 256 | heartbeat interval | operational、不可学习；可配置环境覆盖 |

### 11.4 如何学习，如何避免泄漏

- 初始值来自文献、公开 benchmark 或训练 split；来源写入 manifest。
- campaign 内只允许用已经 reveal 的 wet observations 更新。
- final-test/oracle 未揭示数据不得参与 scaler、calibration、threshold 或 policy tuning。
- 参数只能在 round boundary 更新，生成新 parameter_set_id。
- 已持久化 evidence 不回写重算；新版本产生新 evidence ID。
- optimizer 只能提出 proposal，需离线验证/approval 后成为 active policy。
- 训练样本不足时保持 prior，不允许 LLM 自由修改。
- 主动学习可优化 candidate policy；强化学习若接入，应使用离线/模拟环境并设置约束，不能在真实 wet backend 上无审批探索。

### 11.5 建议配置

~~~yaml
knowledge:
  schema_version: 2
  fusion:
    mode: independent_features
    contributes_to_selection: true

  providers:
    physchem:
      kind: aaindex_delta
      enabled: true
      accessions: []
      scaler_artifact: null
      missing_policy: unavailable

    conservation:
      kind: msa_profile
      enabled: true
      msa_resource: gb1_msa_v1
      minimum_neff:
        value: 32
        category: scientific
        status: expert_prior
        learnable: false
        source: project_policy
        version: v1

    structure:
      kind: structure_ensemble
      enabled: true
      resource_ids: [gb1_complex_experimental_v1]
      static_features: [rsa, secondary_structure, contacts, interface, clashes]
      inverse_folding: proteinmpnn
      missing_policy: unavailable

    kg:
      kind: assay_observation_association
      enabled: true
      model: hierarchical_position_pairwise
      shrinkage_pseudocount:
        value: 3.0
        category: scientific
        status: expert_prior
        learnable: true
        bounds: [0.1, 50.0]
        transform: log
        update_policy: round_boundary_calibration
        min_evidence: 32
        source: legacy_baseline
        version: v1

runtime:
  evidence_heartbeat_interval: 256
~~~

示例中的数值只展示合同，不表示已被科学验证。

## 12. 新 Evidence 合同

当前 Evidence 字段不足以表达证据质量。建议升级为：

~~~python
class EvidenceQuality(BaseModel):
    status: Literal["ok", "degraded", "unavailable", "failed"]
    applicability: Literal["in_domain", "partial", "out_of_domain", "unknown"]
    data_quality: float | None
    calibrated: bool
    uncertainty_kind: str | None
    uncertainty_value: float | None
    warnings: tuple[str, ...] = ()

class EvidenceProvenance(BaseModel):
    provider: str
    provider_version: str
    config_sha256: str
    resource_ids: tuple[str, ...]
    resource_sha256: tuple[str, ...]
    tool_versions: dict[str, str]
    parameter_set_id: str
    generated_at_stage: str

class ScientificEvidence(BaseModel):
    evidence_id: str
    variant_id: str
    channel: str
    evidence_type: str
    claim_scope: str
    statement: str
    raw_features: dict[str, float | str | bool | None]
    calibrated_score: float | None
    quality: EvidenceQuality
    provenance: EvidenceProvenance
    round_id: int
~~~

关键变化：

- score 可以为 null；不是每个 descriptor 都必须伪装成总分。
- confidence 拆成 data quality、applicability、uncertainty 和 calibrated。
- statement 不再是唯一信息载体。
- missing/unavailable 不用 0.5 代替。
- provider/config/resource 版本共同决定 evidence identity。

## 13. Provider 生命周期与缓存

建议协议：

~~~python
class EvidenceProvider(Protocol):
    channel: str

    def prepare(self, context: ProteinTaskContext) -> PreparedResource:
        ...

    def evaluate_batch(
        self,
        variants: Sequence[Variant],
        *,
        context: ProteinTaskContext,
        prepared: PreparedResource,
        round_context: RoundEvidenceContext,
    ) -> Sequence[ScientificEvidence]:
        ...
~~~

- prepare：每个 target/resource 只做一次重任务，例如 MSA 搜索、结构 QC、WT relax。
- evaluate_batch：对候选批处理，避免逐个进程启动。
- PreparedResource 写入 artifact manifest 并带 cache key。
- cache 是内容寻址、只读复用；失效条件明确。
- provider failure 隔离，不应让一个外部工具失败把缺失结果写成中性证据。
- external tool 的 stdout/stderr、退出码、资源用量和 timeout 写入 trace，但不进入科研状态判断。

## 14. 代码实施 PLAN

### Phase 0：冻结当前行为并建立科学风险护栏

目标：在重构前可复现旧结果，同时防止 legacy heuristic 被误当生产证据。

改动：

- 为现有四个规则增加 legacy snapshot/golden tests。
- 将 tolerated-list 和 structure_risk provider 标为 legacy_test_only。
- Evidence statement 加未校准前缀，并允许 contributes_to_selection=false。
- 增加测试，证明 unavailable 不会变成 0.5。
- 记录当前三折结果作为重构前 benchmark。

验收：

- 旧配置显式选择 legacy 模式时结果完全一致。
- 新生产配置默认不启用人工 structure/conservation。

### Phase 1：引入 ProteinTaskContext，消除通用模块中的 GB1 常量

主要文件：

- 新增 protein_features/context.py 和 contracts.py。
- 修改 config.py、loop/orchestrator.py。
- 修改 knowledge/engine.py、knowledge/graph.py。
- 修改 mutation/generators.py、mutation/uncertainty.py。
- 修改 agents/llm.py 和 Scientist profile renderer。
- 修改 models/ensemble.py。

步骤：

1. 从 TaskConfig 和资源配置构造 immutable context。
2. Runner 注入所有消费者。
3. 用 position_to_variant_index 替代所有局部 map。
4. GB1 常量只保留在 data/gb1.py 的专用 adapter。
5. 添加非 GB1 toy protein 测试，mutable positions 使用不同长度与编号。

验收：

- rg 扫描通用运行时不再出现 VDGV 或固定四个位点。
- GB1 回归通过。
- 非 GB1 task 能运行 knowledge、candidate selection 和 KG query。

### Phase 2：升级 Evidence 与参数配置合同

改动：

- Pydantic 配置取代 KnowledgeConfig 的四个 bool + 任意 site_profiles。
- 引入 provider discriminated union。
- 引入 LearnableParameterSpec/ParameterSet。
- Evidence 增加 quality/provenance/raw_features/calibrated_score。
- structured KG schema 与 adapter 迁移。

兼容：

- 编写 v1→v2 config migration。
- 旧 Evidence 读取时映射为 legacy_unvalidated。

验收：

- 每条 evidence 可追到 config hash、resource hash、tool version。
- 任意 scientific 数值都能在配置或校准 artifact 中找到来源。

### Phase 3：实现序列理化 provider

改动：

- AAindexRepository：固定 accession、缓存和 hash。
- PhyschemDeltaProvider：输出命名 delta 与特殊残基 flags。
- ProtParamAdapter：输出全序列 WT/mutant delta。
- 可选 PROPKA adapter 放在结构阶段，不与基础序列属性混合。

测试：

- 20 种标准残基、非法字符、His/pH、Gly/Pro/Cys、multi-mutant。
- 对已知 ProtParam 示例做 parity test。
- 证明 feature scaling 只读取 public train/reference。

验收：

- 不再有人工四维归一化欧氏总分。
- Scientist 可查询每一维及来源。

### Phase 4：实现一次准备、多轮复用的 MSA provider

改动：

- SequenceResourceResolver。
- MMseqs2Adapter；HHblits 作为可插拔备选。
- MSAFilterAndCluster、MSAQualityAnalyzer、PositionMapper。
- ConservationProfileProvider。
- 可选 PLMC/EVmutation-style epistasis provider。

运行：

- campaign prepare 阶段计算/加载 MSA。
- artifact 保存 A3M/Stockholm、search manifest、filter report、profile、mapping。
- 各 round 只对候选计算变体分数。

测试：

- cache hit/miss、database version invalidation。
- mapping offset/indel/insertion code。
- shallow MSA 降级。
- hidden oracle sequence/fitness 不进入 MSA 或 calibration。

验收：

- conservation statement 全部来自 MSA artifact。
- 输出 Neff、coverage/gap 和适用域。
- 后续轮次不重复搜索。

### Phase 5：实现真实静态结构分析

改动：

- StructureResourceResolver 和 ResidueMappingService。
- StructureQC。
- StaticStructureFeatureExtractor：secondary structure、SASA/RSA、depth、contacts。
- InteractionFeatureExtractor：H-bond、salt bridge、disulfide、interface、clash。
- FreeSASA/DSSP 等外部依赖通过 adapter，非核心依赖放 optional extra。

测试：

- 小型固定 PDB fixture。
- missing atom、alternate location、多 chain、insertion code。
- 错误 residue mapping 必须 fail closed。
- 结构缺失返回 unavailable。

验收：

- structure provider 实际读取坐标。
- statement 明确 resource/chain/residue/quality。
- 删除生产 structure_risk。

### Phase 6：接入 inverse folding 与 Rosetta/FoldX

改动：

- ProteinMPNNAdapter/ESMIFAdapter。
- FoldXAdapter。
- RosettaAdapter：relax、ddG、interface analyzer 的受控子集。
- ToolExecutionPolicy：allowlist、timeout、CPU/GPU、并发、license、container digest。
- ReplicateAggregator 与 calibration。

测试：

- 单元测试使用 fake executable，不要求 CI 安装 Rosetta。
- 可选 integration marker 在有 license/runtime 的环境运行。
- 检查 multi-mutant 完整组合、replicate、seed 和能量项解析。
- 禁止 Scientist 直接启动 batch。

验收：

- raw score 与 calibrated score 分离。
- 所有 external run 可重现。
- failure 不生成中性/正向证据。

### Phase 7：扩展 KG Operator 与 Scientist 上下文

改动：

- 新增第 9.2 节的 feature query operators。
- controller 继续执行 scope/max rows/round/query budget。
- Scientist prompt 只接收结构化 evidence summary 与 query packs。
- 强制至少比较多个候选和多个 evidence dimension。
- 强制呈现 counterevidence 与 missing evidence。

验收：

- Scientist 无 raw SQL、filesystem、oracle、external batch 权限。
- tool trace 能映射 run/round/variant/evidence/resource。
- cited evidence ID 必须可见且可追溯。

### Phase 8：校准、可学习参数与渐进上线

改动：

- 对 public train/fold 内数据拟合 scaler、calibrator 和 fusion。
- 每个 task/family 输出 calibration report。
- 参数更新只在 round boundary 发生。
- 添加 ablation：
  legacy、physchem-only、MSA-only、structure-only、KG-only、full。

指标：

- 证据与 wet fitness 的 Spearman/校准误差。
- top-k recall、regret、diversity。
- 不同 mutation count、site、burial、MSA depth 的分层性能。
- 候选选择稳定性和 evidence disagreement。
- 三折间方差，而不只看平均值。

上线门槛：

- 新 provider 在 held-out public fold 上不劣于 legacy，且 provenance/quality 完整。
- 未校准 provider 只能作为显示证据，不参与 selection。
- 发现分布外输入时自动降级。

## 15. 测试矩阵

### 15.1 单元测试

- ProteinTaskContext：长度、位置、序列、mapping。
- 参数合同：bounds、version、learnable/update policy。
- physchem：命名 delta、特殊残基、pH。
- MSA：过滤、Neff、entropy、PSSM、cache。
- structure：mapping、RSA、contact、interface、missing data。
- external adapter：输入模板、输出 parser、timeout、非零退出。
- Evidence：unavailable、provenance、稳定 ID。

### 15.2 集成测试

- prepare→evaluate_batch→KG write→Scientist query。
- Round 1 建 MSA，Round 2/3 cache hit。
- structure resource 更新导致 cache invalidation。
- Rosetta/FoldX 不可用时按 policy 降级。
- parameter_set 在 round boundary 切换但旧 evidence 不变。

### 15.3 泄漏与权限测试

- MSA 搜索只使用 reference/full sequence，不读 oracle label。
- scaler/calibrator 不读 final-test。
- KG query 不返回当前轮未 reveal wet observation。
- Scientist 不能触发 oracle、final-test、实验 backend 或 external batch。
- trace 不参与 campaign recovery truth。

### 15.4 科学验证

- 在 ProteinGym/公开 DMS 或当前训练 folds 上比较。
- 进化通道按 MSA Neff 分层。
- 结构通道按 experimental/predicted 和质量分层。
- ΔΔG 按 buried/exposed/interface、mutation class 和单/多突变分层。
- 检查证据方向是否与 assay objective 对齐。
- 对 GB1 特别报告 pairwise/epistatic 模型相对独立位点模型的增益。

## 16. 决策清单

建议立即采纳：

- structure_risk 和 tolerated list 降为 test-only。
- 所有缺失证据返回 unavailable，不使用 0.5。
- WT/positions 从 TaskConfig 构造统一 context 并动态注入。
- 0.85、0.03、3.0 等全部迁入版本化参数合同。
- MSA 在 campaign prepare 阶段计算一次并缓存，多轮复用。
- Scientist 只查询科学工具的结构化结果，不直接执行外部程序。

建议第二阶段采纳：

- AAindex/ProtParam 命名 delta。
- MSA PSSM/entropy + quality。
- 真实坐标的 RSA/contact/interface。
- ProteinMPNN/ESM-IF compatibility。

建议经 benchmark 后采纳：

- PLMC/EVmutation-style pairwise model。
- FoldX/Rosetta ΔΔG。
- assay-specific evidence fusion 与 learnable policy。

不建议：

- 让 KnowledgeEngine 内置 LLM 来“理解”原子结构。
- 让 LLM 自行修改 confidence、阈值或 selection weight。
- 将所有通道继续压成没有共同语义的单一平均分。
- 将 Rosetta/FoldX 输出直接称为 fitness 或 wet evidence。
- 为了避免工具失败而用人工 0.5 填充。

## 17. 最终目标状态

重构完成后，KnowledgeEngine 的定义应是：

> 一个确定性、可插拔、可审计的蛋白证据编排器。它从 Task/Sequence/Structure 配置构造目标上下文，准备并缓存专业科学资源，批量调用受控特征 provider，将输出标准化为带质量、适用域、不确定性和 provenance 的 Evidence，再写入受轮次约束的 KG。它不自行调用 LLM，也不把 dry prediction 升级为 wet truth。

Scientist 的定义应是：

> 一个只使用当前轮 sanitized context 和白名单 KG tools 的假设综合器。它可以比较理化、进化、结构、能量和 assay/KG 证据，但不能运行外部实验、访问隐藏数据或创造底层科学数值。

这样既能保持现有 CampaignRunner 的科研状态机和数据可见性，又能使蛋白“特征理解”从人工 GB1 规则升级为真正的序列、进化和三维结构分析。

## 18. 主要参考文献与工具

### Agent 与工具编排

1. Xu et al. ProteinMCP: An agentic AI framework for autonomous protein engineering. Protein Science, 2026. [PubMed](https://pubmed.ncbi.nlm.nih.gov/41877581/)；[开源实现](https://github.com/charlesxu90/ProteinMCP)
2. Ghafarollahi & Buehler. ProtAgents: protein discovery via large language model multi-agent collaborations combining physics and machine learning. Digital Discovery, 2024. [DOI](https://doi.org/10.1039/D4DD00013G)
3. Teneggi et al. Protein Design with Agent Rosetta: A Case Study for Specialized Scientific Agents. 2026 preprint. [arXiv](https://arxiv.org/abs/2603.15952)
4. Kim & Romero. Benchmarking and behavioral characterization of LLM agents for protein design. 2026 preprint. [DOI](https://doi.org/10.64898/2026.05.06.723381)

### MSA、进化与 fitness

5. Hopf et al. Mutation effects predicted from sequence co-variation. Nature Biotechnology, 2017. [DOI](https://doi.org/10.1038/nbt.3769)
6. Frazer et al. Disease variant prediction with deep generative models of evolutionary data. Nature, 2021. [DOI](https://doi.org/10.1038/s41586-021-04043-8)
7. Rao et al. MSA Transformer. ICML, 2021. [论文](https://proceedings.mlr.press/v139/rao21a/rao21a.pdf)
8. Steinegger & Söding. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nature Biotechnology, 2017. [论文](https://www.nature.com/articles/nbt.3988)
9. Remmert et al. HHblits: lightning-fast iterative protein sequence searching by HMM-HMM alignment. Nature Methods, 2012. [论文](https://www.nature.com/articles/nmeth.1818)
10. Mirdita et al. ColabFold: making protein folding accessible to all. Nature Methods, 2022. [论文](https://www.nature.com/articles/s41592-022-01488-1)
11. Notin et al. ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction. [PubMed](https://pubmed.ncbi.nlm.nih.gov/38106144/)

### 理化性质与结构

12. AAindex: amino acid indices, substitution matrices and pairwise contact potentials. [官方数据库](https://www.genome.jp/aaindex/)
13. ExPASy ProtParam. [官方文档](https://web.expasy.org/protparam/protparam-doc.html)
14. EMBOSS pepstats. [官方文档](https://emboss.bioinformatics.nl/cgi-bin/emboss/help/pepstats)
15. Olsson et al. PROPKA3. Journal of Chemical Theory and Computation, 2011. [DOI](https://doi.org/10.1021/ct100578z)
16. Dauparas et al. Robust deep learning-based protein sequence design using ProteinMPNN. Science, 2022. [DOI](https://doi.org/10.1126/science.add2187)
17. Jing et al. Learning from Protein Structure with Geometric Vector Perceptrons. ICLR, 2021. [论文](https://arxiv.org/abs/2009.01411)
18. Hsu et al. Learning inverse folding from millions of predicted structures. ICML, 2022. [会议页](https://icml.cc/virtual/2022/oral/16886)
19. Mitternacht. FreeSASA: An open source C library for solvent accessible surface area calculations. 2016. [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4776673/)
20. Alford et al. The Rosetta All-Atom Energy Function for Macromolecular Modeling and Design. Journal of Chemical Theory and Computation, 2017. [DOI](https://doi.org/10.1021/acs.jctc.7b00125)
21. Schymkowitz et al. The FoldX web server: an online force field. Nucleic Acids Research, 2005. [DOI](https://doi.org/10.1093/nar/gki387)
22. Potapov et al. Assessing computational methods for predicting protein stability upon mutation: good on average but not in the details. Protein Engineering, Design & Selection, 2009. [DOI](https://doi.org/10.1093/protein/gzp030)

### 检索说明

本方案按“蛋白设计 Agent/工具编排 → MSA/进化模型 → 三维结构与 inverse folding → Rosetta/FoldX/理化工具”的路径进行多源检索。优先采用论文原文、PubMed、期刊页面、官方软件文档和官方代码仓库；2026 年尚为预印本的 Agent Rosetta 与 BioDesignBench 已显式标注为预印本，不将其结果当作成熟共识。
