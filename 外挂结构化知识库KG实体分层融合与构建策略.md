# 外挂结构化知识库 KG：实体分层、跨模态融合与可消融构建策略

> 研究与实现日期：2026-08-15  
> 范围：将 fitness-agents 的运行日志型 KG 升级为可复用、可追溯、可动态更新的蛋白 fitness 外挂知识库  
> 状态：已新增独立 schema-first 构建骨架、适配器、融合、校验、消融配置与测试；未替换当前 SQLite KG

## 0. 核心结论

KG 不应只记录“这一轮发生了什么”，而应成为连接以下信息的结构化知识中枢：

```text
蛋白/序列身份
  + 残基坐标与突变
  + assay/condition 下的真实 Observation
  + Predictor/轻量模型输出及不确定性
  + 结构与局部原子环境
  + 保守性、MSA、共进化和同源信息
  + 功能本体与结构域
  + 文献 Claim
  + Agent Evidence/Hypothesis/Decision
  + 来源、版本、轮次与有效期
```

但不应从第一天就构建“大而全”的生物医学 KG。对小数据 fitness 优化，优先级最高的是能够直接减少数据泄漏、坐标混淆和证据误用的实体：

1. P0：Protein、Sequence、ResiduePosition、Variant、Mutation、Assay、Condition、Observation、CampaignRound；
2. P1：Prediction、ModelRun、Evidence、Hypothesis、Decision、Structure、ResidueEnvironment、EvolutionProfile；
3. P2：AtomicInteraction、Domain、OntologyTerm、Homolog、BindingPartner、Publication、Claim、Artifact；
4. P3：大范围通路、疾病、药物和跨物种网络，只在任务证明需要时引入。

推荐采用 BioCypher 式的模块边界：

```text
Source Adapter -> Canonical Schema -> Normalizer -> Fusion -> Validator -> Sink
      ↑                ↑                 ↑          ↑          ↑        ↑
   可插拔/消融      类型/关系消融       ID策略     融合策略    QA规则   存储后端
```

本次已按这一边界实现无新增依赖的 `kg_knowledge` 包。它可把当前项目的 Variant、Observation、Prediction、Evidence 和 Hypothesis 转换成统一图记录，并支持按 adapter、layer、modality、entity type、relation type 独立消融。

## 1. 从“运行日志”到“外挂知识库”的定义变化

### 1.1 当前 KG 的合理定位

当前 SQLite KG 是一个可靠的运行内事实账本和审计层：保存 Variant、Observation、Prediction、Evidence、Hypothesis 和 Agent Query，并通过轮次控制可见性。它对当前 GB1 闭环非常有用，不应被直接丢弃。

但它仍有四个限制：

- 结构、序列、原子、进化和文献知识没有统一实体模式；
- 多数关系由外键或 JSON 引用隐式表达；
- 同一个残基在序列、PDB、MSA 和论文中的坐标缺少显式映射；
- 外部知识、运行内事实和 Agent 主张没有统一 provenance/validity 语义。

### 1.2 外挂结构化知识库应具备的能力

“外挂”不是指换成 Neo4j 就完成，而是指知识可独立于一次 run 存在，并由稳定接口供多个 Agent、模型和实验复用：

- 跨 run 的稳定实体 ID；
- 按 protein/assay/condition/round 查询；
- Observation、Prediction、Claim、Hypothesis 权威等级分离；
- 序列、结构、原子、MSA、文本和模型输出能通过明确关系连接；
- 每条关系有来源、方法版本、置信度、适用范围和有效期；
- 新实验增量写入，旧知识保留并可失效或被 supersede；
- 任何知识层都能独立关闭，以验证真实贡献。

## 2. 前沿文献与 GitHub 代码的设计启示

### 2.1 KG 构建与模式治理

| 工作 | 可借鉴点 | 对本项目的采用方式 |
|---|---|---|
| [BioCypher, Nature Biotechnology 2023](https://www.nature.com/articles/s41587-023-01848-y) / [GitHub](https://github.com/biocypher/biocypher) | 数据源 adapter、schema/ontology 配置和输出后端三层解耦 | 直接采用其模块边界；当前先做轻量原生实现，后续可增加 BioCypher sink |
| [Biolink Model](https://ascpt.onlinelibrary.wiley.com/doi/10.1111/cts.13302) / [GitHub](https://github.com/biolink/biolink-model) | 与存储无关的生物医学实体/association 模式 | 使用其命名与 association 思想，但仅投影项目需要的子集 |
| [LinkML](https://github.com/linkml/linkml) | 一份 schema 生成 JSON Schema、Python、OWL、SHACL 等表示 | P1 后将已稳定的 dataclass schema 迁移成 LinkML 源定义 |
| [KGX](https://github.com/biolink/kgx) | Biolink 图的转换、合并、验证与多格式 IO | 作为外部交换/验证插件，不作为核心运行依赖 |
| [PheKnowLator](https://pmc.ncbi.nlm.nih.gov/articles/PMC11009265/) / [GitHub](https://github.com/callahantiff/PheKnowLator) | 可选择不同语义模型、ontology cleaning、构建 QA | 用于高级 ontology/OWL 路线；MVP 不需要完整语义网复杂度 |

### 2.2 多层生物知识与蛋白表示

| 工作 | 证据 | 对本项目的含义 |
|---|---|---|
| [PrimeKG, Scientific Data 2023](https://www.nature.com/articles/s41597-023-01960-3) / [GitHub](https://github.com/mims-harvard/PrimeKG) | 20 个来源、10 个生物尺度、统一标识和文本特征；提供可更新脚本 | 每个来源一个 adapter；先标准化再合并；文本是属性/制品而不是替代图关系 |
| [Multimodal learning with graphs, Nature Machine Intelligence 2023](https://www.nature.com/articles/s42256-023-00624-6) | 图可作为跨模态几何关系的中枢，并区分知识驱动、语言驱动等融合形态 | 对 fitness KG 优先采用 graph-centric late fusion，保留各模态独立性 |
| [OntoProtein, ICLR 2022](https://openreview.net/forum?id=yfe1VMYAXa4) / [GitHub](https://github.com/zjunlp/OntoProtein) | 将蛋白序列与 GO KG 联合用于蛋白表示学习 | GO/功能层有潜在价值，但应晚于实验、序列、结构和进化 P0/P1 层 |
| [OpenBioLink, Bioinformatics 2020](https://academic.oup.com/bioinformatics/article/36/13/4097/5825726) / [GitHub](https://github.com/OpenBioLink/OpenBioLink) | 区分高质量与噪声边、提供泄漏更谨慎的划分和可复现图生成 | KG 消融必须按来源质量和时间切分，不能只随机删边 |
| [Graphiti](https://arxiv.org/abs/2501.13956) / [GitHub](https://github.com/getzep/graphiti) | episode、双时间语义、事实失效而非删除、增量构建 | 用于 CampaignRound、valid_from/to 和来源事件的设计 |

代码检查快照：BioCypher `f2061939`、Biolink `fc44947`、PrimeKG `32a3818`。三者许可证分别为 Apache-2.0、Apache-2.0 和 MIT。另检查了一个序列+文本+图的 BioMedKG 研究仓库，但快照 `3f719ac` 未发现根目录 LICENSE，因此本实现没有复制其代码，只将“模态独立编码、图上后融合”作为概念参考。

### 2.3 对“前沿”的使用边界

2026 年出现的扩展 PrimeKG 工作可以用于观察趋势，但预印本和快速变化仓库不应成为 MVP 基础。核心架构应建立在已经正式发表、模式清晰且许可证明确的 BioCypher、Biolink、PrimeKG、OpenBioLink 和 Graphiti 上。

## 3. 分层实体与优先级

评分说明：潜在贡献和实现难度均为 1–5；优先级综合考虑对 fitness 决策的直接性、可追溯性和实现成本。

### 3.1 P0：先建立可信坐标、实验事实和时间边界

| 实体 | 模态 | 核心字段 | 潜在贡献 | 难度 | 为什么先做 |
|---|---|---|---:|---:|---|
| Protein | identity/sequence | accession、organism、target name | 5 | 1 | 所有外部知识的身份锚点 |
| Sequence | sequence | sequence、length、version、checksum | 5 | 1 | 防止不同参考序列混用 |
| ResiduePosition | sequence | sequence_id、position、reference residue | 5 | 2 | 连接 mutation、PDB、MSA、文献的核心坐标 |
| Variant | sequence | full sequence、mutation notation、background | 5 | 1 | fitness 预测中心对象 |
| Mutation | sequence | ref、position、alt、mutation class | 5 | 1 | 支持单点、多点和 epistasis 表达 |
| Assay | tabular | assay definition、target metric、directionality、normalization | 5 | 1 | fitness 不能脱离 assay 解释 |
| Condition | tabular | pH、temperature、ligand、buffer、host 等 | 5 | 2 | 相同突变在不同条件可能方向相反 |
| Observation | tabular/time-series | raw/normalized value、unit、replicates、QC、round | 5 | 1 | 最高权威事实 |
| CampaignRound | time-series | run、round、snapshot、visibility | 5 | 1 | 防泄漏、历史重放和动态学习 |

P0 建完后，系统才能可靠回答：哪个序列背景、哪个 assay、什么条件、哪一轮可见的哪个完整变体产生了什么真实结果。

### 3.2 P1：直接提升候选排序和可解释性的知识

| 实体 | 模态 | 核心字段 | 贡献 | 难度 | 接口建议 |
|---|---|---|---:|---:|---|
| Prediction | tabular | mean、interval、uncertainty、OOD、model run | 5 | 1 | `ModelOutputAdapter` |
| ModelRun | provenance | model/version、training snapshot、feature hash | 5 | 2 | 独立模型版本实体 |
| Evidence | tabular/text | direction、score、confidence、scope、source | 5 | 1 | 所有建议必须引用 |
| Hypothesis | text/structured | claim、assumptions、falsification、status | 5 | 2 | append-only 版本 |
| Decision | structured | candidates、scores、policy、query/evidence IDs | 5 | 2 | 与自由思维文本分开 |
| Structure | structure-3D | PDB/AlphaFold ID、chain、model version、quality | 5 | 2 | 保存坐标制品引用 |
| ResidueEnvironment | structure-3D | ASA、secondary structure、local geometry、pLDDT | 5 | 3 | 直接解释位点容忍度 |
| EvolutionProfile | MSA | conservation、entropy、frequency、coupling、MSA version | 5 | 2 | 小数据下通常是高价值先验 |
| AtomicInteraction | atomic | contact type、atoms、distance、geometry、method | 4 | 3 | 局部机制证据；应受结构质量约束 |

P1 的建议内部顺序为：Prediction/ModelRun/Evidence/Hypothesis/Decision → EvolutionProfile → Structure/ResidueEnvironment → AtomicInteraction。原因是前五项几乎直接复用当前数据，进化和残基层结构的投入产出比通常高于完整原子知识。

### 3.3 P2：扩展功能、文献和外部生物学上下文

| 实体 | 贡献 | 难度 | 适用条件 |
|---|---:|---:|---|
| Domain | 4 | 2 | 多结构域蛋白或位点功能解释 |
| OntologyTerm（GO 等） | 4 | 2 | 跨蛋白迁移、功能相似性 |
| Homolog | 3 | 3 | 同源实验和跨物种证据充分时 |
| BindingPartner/Ligand | 3 | 3 | binding assay、复合体结构 |
| Publication | 3 | 2 | 需要文献可追溯时 |
| Claim | 4 | 3 | 文献事实必须携带上下文、极性和证据句 |
| Artifact | 4 | 2 | 结构、MSA、embedding、报告等大对象的版本引用 |

### 3.4 P3：大范围背景知识

Pathway、Disease、Drug、CellType、Tissue、Phenotype 等实体有助于药物发现和临床任务，但对单蛋白、单 assay 的 fitness 闭环往往是远端信息。除非任务扩展为跨蛋白迁移、宿主效应或疾病机制，不应优先投入。

## 4. 核心关系设计

### 4.1 身份与序列关系

```text
(Variant)-[:VARIANT_OF]->(Protein)
(Variant)-[:HAS_SEQUENCE]->(Sequence)
(Variant)-[:HAS_MUTATION]->(Mutation)
(Mutation)-[:AT_POSITION]->(ResiduePosition)
(ResiduePosition)-[:IN_SEQUENCE]->(Sequence)
(Sequence_v2)-[:SUPERSEDES]->(Sequence_v1)
```

关键约束：`position=39` 只有在绑定明确 Sequence 后才有意义。不要用全局“Residue 39”连接不同蛋白或不同参考序列。

### 4.2 实验关系必须上下文化

不要直接写：

```text
(Mutation A39W)-[:INCREASES]->(Fitness)
```

应将 Observation 作为上下文实体重ification：

```text
(Observation)-[:OBSERVES_VARIANT]->(Variant)
(Observation)-[:MEASURED_IN]->(Assay)
(Observation)-[:UNDER_CONDITION]->(Condition)
(Observation)-[:REVEALED_IN]->(CampaignRound)
(Observation)-[:DERIVED_FROM]->(ExperimentArtifact)
```

原始值、标准化值、重复、误差和 QC 保存在 Observation。若需要“突变提升 fitness”的 Claim，应由 Evidence 从一个或多个 Observation 推导，并声明 sequence background、assay 和 condition。

### 4.3 结构和原子关系

```text
(Protein)-[:HAS_STRUCTURE]->(Structure)
(ResiduePosition)-[:MAPPED_TO_STRUCTURE]->(ResidueEnvironment)
(ResidueEnvironment)-[:IN_STRUCTURE]->(Structure)
(ResidueEnvironment)-[:CONTACTS]->(ResidueEnvironment)
(AtomicInteraction)-[:INVOLVES]->(ResidueEnvironment)
(AtomicInteraction)-[:INVOLVES_LIGAND]->(BindingPartner)
```

映射边必须保存：sequence position、chain、author residue number、insertion code、alignment method、mapping confidence。结构来源和质量（实验分辨率或 AlphaFold pLDDT/PAE）应影响下游证据置信度。

### 4.4 进化与功能关系

```text
(ResiduePosition)-[:HAS_EVOLUTION_PROFILE]->(EvolutionProfile)
(ResiduePosition)-[:COEVOLVES_WITH]->(ResiduePosition)
(Protein)-[:HAS_DOMAIN]->(Domain)
(Protein)-[:ANNOTATED_WITH]->(OntologyTerm)
(Homolog)-[:HOMOLOG_OF]->(Protein)
```

MSA 原文件不放进节点属性；通过 Artifact 保存 URI、checksum、数据库版本、过滤阈值和构建参数。EvolutionProfile 保存可直接查询的摘要统计。

### 4.5 模型、证据和 Agent 关系

```text
(Prediction)-[:PREDICTS]->(Variant)
(Prediction)-[:GENERATED_BY]->(ModelRun)
(Evidence)-[:ABOUT]->(Variant | Mutation | ResiduePosition | Claim)
(Evidence)-[:DERIVED_FROM]->(Observation | Prediction | Artifact | Publication)
(Hypothesis)-[:SUPPORTED_BY]->(Evidence)
(Hypothesis)-[:CONTRADICTED_BY]->(Evidence)
(Hypothesis)-[:TESTED_BY]->(Observation | Experiment)
(Decision)-[:SELECTS]->(Variant)
(Decision)-[:JUSTIFIED_BY]->(Evidence | Hypothesis | Prediction)
(Hypothesis_v2)-[:SUPERSEDES]->(Hypothesis_v1)
```

Evidence 是跨层桥梁，不能只是一段自然语言。至少保存方向、强度、置信度语义、适用范围、推导方法和 source IDs。

## 5. 不同模态如何存储和融合

### 5.1 图内保存摘要，图外保存大制品

| 模态 | 图内保存 | 图外 Artifact |
|---|---|---|
| Sequence | checksum、length、version；短序列 MVP 可直接保存 | FASTA、完整版本包 |
| Structure | structure ID、chain、质量、映射 | PDB/mmCIF、AlphaFold 文件 |
| Atomic | contact type、关键距离、置信度 | 完整坐标、轨迹、能量文件 |
| MSA | conservation/coupling 摘要、版本 | A3M/Stockholm、搜索数据库快照 |
| Text | Claim、证据句位置、publication ID | PDF/HTML/全文索引 |
| Embedding | model ID、维度、checksum、artifact URI | 向量文件或向量库 |
| Time series | 轮次、窗口摘要 | 原始仪器/实验序列 |

高维向量、完整原子坐标或全文直接塞进节点属性会使 schema、版本和消融都变差。使用 Artifact 节点还能精确追踪同一事实由哪个文件和哪个处理版本产生。

### 5.2 五级融合流程

```text
L0 Schema alignment
   source fields -> canonical entity/relation types

L1 Identity resolution
   UniProt/PDB/assay/local ID -> stable canonical ID + alias trail

L2 Coordinate alignment
   reference sequence position <-> variant position <-> PDB chain residue <-> MSA column

L3 Evidence fusion
   relation-level source, confidence, scope, time and conflict handling

L4 Decision fusion
   query-specific subgraph -> EvidencePack -> predictor/acquisition/LLM
```

不要在 L0–L3 就把所有模态压成单一分数。先保留可区分的实体和关系，只有在具体查询或模型输入时做 late fusion，才便于解释和消融。

### 5.3 来源相关性的置信度融合

多个数据库可能复制同一篇论文或同一实验，不能把它们当成独立证据简单 noisy-or。推荐：

1. 先按 `source_group` 分组；同一来源族取最大置信度；
2. 再对相互独立的来源族使用 noisy-or：

```text
c_fused = 1 - Π_g (1 - max(c_i in source_group g))
```

例如同一数据库的两个导出置信度 0.6、0.8，只贡献 0.8；另一个独立实验来源为 0.5，融合为 `1-(1-0.8)(1-0.5)=0.9`。

该公式只是默认插件，不代表统计真值。后续可以替换成 Bayesian、Dempster–Shafer、校准回归或简单 max，并作为融合策略消融。

### 5.4 冲突不删除，显式保留

冲突处理规则：

- 属性冲突保存 `_conflicts` 或独立 Claim；
- 相反关系可以共存，但各自有 assay、condition、source 和 valid time；
- 新事实不覆盖旧事实，通过 `valid_to` 或 `SUPERSEDES` 关闭旧版本；
- Agent 查询时优先返回冲突摘要和反证，而不是静默选一条；
- 只有 identifier alias 可以在高置信校验后合并，生物结论不能因名称相同而合并。

## 6. 可插拔、可消融接口

### 6.1 Source Adapter

```python
class KnowledgeAdapter(Protocol):
    name: str
    def extract(self, context: BuildContext) -> KnowledgeBatch: ...
```

每个数据源独立 adapter：

- `CampaignObservationAdapter`
- `InferenceKnowledgeAdapter`
- `ProteinSequenceAdapter`
- `StructureAdapter`
- `AtomicInteractionAdapter`
- `EvolutionAdapter`
- `OntologyAdapter`
- `LiteratureClaimAdapter`

关闭某个 adapter 即完成数据源消融，不需要改 schema 或查询层。

### 6.2 Normalizer

```python
class KnowledgeNormalizer(Protocol):
    name: str
    def normalize(self, batch: KnowledgeBatch) -> KnowledgeBatch: ...
```

建议拆分：identifier alias、amino-acid notation、assay unit、sequence coordinate、PDB mapping、ontology mapping。每个 normalizer 输出映射收据，避免不可追溯的清洗。

### 6.3 FusionPolicy

```python
class FusionPolicy(Protocol):
    name: str
    def fuse(self, batches: Iterable[KnowledgeBatch]) -> KnowledgeGraphSnapshot: ...
```

可比较：no fusion、deduplicate only、provenance-aware noisy-or、learned/calibrated fusion。

### 6.4 Validator

```python
class KnowledgeValidator(Protocol):
    name: str
    def validate(self, snapshot: KnowledgeGraphSnapshot) -> tuple[ValidationIssue, ...]: ...
```

至少检查：

- ID 唯一；
- 关系端点存在；
- confidence 范围；
- valid interval 合法；
- 关系具有 source IDs；
- Observation 有 assay/condition/round；
- 序列—结构坐标映射完整；
- 不允许 Prediction 通过错误 predicate 伪装为测量；
- final/oracle visibility 不进入训练或 Agent snapshot。

### 6.5 Sink

```python
class KnowledgeGraphSink(Protocol):
    name: str
    def write(self, snapshot: KnowledgeGraphSnapshot) -> None: ...
```

建议顺序：

1. `InMemoryGraphSink`：单元测试；
2. `SQLiteSnapshotSink`：最小改造并兼容当前系统；
3. `Parquet/KGXExportSink`：离线分析和交换；
4. `Neo4jSink`：确有多跳、并发和图可视化需求后再启用；
5. `BioCypherSink`：与更广泛生物医学数据管线对接。

存储后端不是 schema；切换后端不应改变实体和关系含义。

## 7. 本次已实现的代码

```text
src/fitness_agents/
  plugin_registry.py
  kg_knowledge/
    schema.py          # layer、modality、Entity/Relation/Batch/Snapshot
    catalog.py         # 实体/关系优先级和消融分组
    ablation.py        # adapter/layer/modality/type 开关
    adapters.py        # 当前 campaign 与 inference 记录适配器
    normalization.py   # identity 与 alias normalizer
    fusion.py          # provenance-aware dedup/confidence fusion
    validation.py      # 端点、ID、置信度、时间、来源 QA
    sinks.py           # sink 协议与内存实现
    builder.py         # extract -> normalize -> filter -> fuse -> validate -> write
```

配置：`configs/kg/knowledge_modules.yaml`。  
测试：`tests/unit/test_kg_knowledge_modules.py`。

### 7.1 已能完成的工作

- 将现有 Variant/Observation 转换成 Protein、Sequence、ResiduePosition、Mutation、Assay、Condition、Observation、CampaignRound；
- 将 Prediction/Evidence/Hypothesis 转换成 ModelRun、Prediction、Evidence、Hypothesis 及关系；
- 稳定生成实体/关系 ID；
- 对 adapter、layer、modality、entity type 和 relation type 独立过滤；
- 消融实体后自动移除悬空边；
- 按 source_group 防止相关来源重复增信；
- 合并来源、模态、证据 ID，并保留属性冲突；
- 在写入 sink 前进行核心 schema 校验。

### 7.2 尚未实现的工作

- Structure、Evolution、Literature 等真实外部 adapter；
- SQLite/Neo4j/BioCypher 持久 sink；
- PDB/UniProt/MSA 坐标映射；
- LinkML/Biolink schema 文件和 KGX 导出；
- Observation 的 replicate/QC/raw unit 扩展；
- 实际增量 update/diff，而非完整 snapshot build；
- 与 LLM QueryOperator 的正式连接。

## 8. 消融设计

### 8.1 知识层消融

| 条件 | 保留内容 | 目的 |
|---|---|---|
| observations-only | P0 实验、身份、序列、历史 | 测运行日志基线 |
| + model | Prediction/ModelRun | 测轻量模型输出价值 |
| + evidence/agent | Evidence/Hypothesis/Decision | 测结构化推理历史价值 |
| + evolution | MSA/保守性/共进化 | 测进化先验 |
| + structure | Structure/ResidueEnvironment | 测残基层结构先验 |
| + atom chemistry | AtomicInteraction | 测细粒度几何/化学增益 |
| + function | Domain/GO | 测功能迁移 |
| + literature | Publication/Claim | 测外部文本知识 |
| full | 全部 | 上限系统 |

### 8.2 正交消融轴

仅按 layer 做消融还不够，至少保留以下正交开关：

- source adapter：PDB vs AlphaFold、不同 MSA 或数据库；
- modality：structure-3D、atomic、MSA、text、embedding；
- entity/relation type：只去掉 AtomicInteraction 或 COEVOLVES_WITH；
- fusion policy：max、noisy-or、calibrated；
- provenance：有/无来源权重；
- temporal：latest-only vs as-of-round；
- retrieval：one-hop、path、semantic、hybrid；
- writeback：frozen、append-only、动态状态更新。

每次实验只改变一个轴，并冻结同一个 graph snapshot、dataset split、seed、predictor 和 selection budget。

## 9. 具体构建路线

### Phase 0：冻结 P0 schema（1 个迭代）

工作：

1. 确认 Protein/Sequence/ResiduePosition/Variant/Mutation ID 规则；
2. 将 Assay 与 Condition 从自由字符串提升为实体；
3. Observation 增加 raw/normalized、unit、QC、replicate、round；
4. 建立 visibility、source、checksum、valid_from/to；
5. 用当前 GB1 数据生成 snapshot 并与现有 KG 对账。

验收：所有 Observation 能追溯到完整 Variant、Sequence、Assay、Condition 和 Round；无悬空边；隐藏标签不进入可见快照。

### Phase 1：模型与 Agent 知识（1 个迭代）

工作：

1. 接入本次 `InferenceKnowledgeAdapter`；
2. 增加 Decision；
3. 将 Evidence 方向、scope、method version 结构化；
4. 为 Hypothesis 增加 counterevidence、status、falsification 和 supersession；
5. 实现 SQLiteSnapshotSink，但保持旧表只读兼容。

验收：每个选择能回溯 Prediction、Evidence、Hypothesis、query 和 policy version。

### Phase 2：进化与结构层（2–3 个迭代）

先做进化，再做残基层结构：

1. MSA Artifact + EvolutionProfile；
2. Structure Artifact + residue mapping；
3. ResidueEnvironment：ASA、secondary structure、local contacts、quality；
4. 与 P0 ResiduePosition 对齐；
5. 分别运行 no-evolution 和 no-structure 消融。

验收：映射覆盖率、错位率和低置信映射均可报告；结构/进化关闭后基线可复现。

### Phase 3：原子化学与文献 Claim（按需求）

原子层只保存任务相关摘要，不急于图化所有原子：

- 受突变影响的氢键、盐桥、芳香堆积、疏水接触、金属/配体接触；
- 距离、角度、occupancy、structure quality、计算方法；
- 计算制品和参数 hash。

文献层采用 `Publication -> Claim -> Evidence`，Claim 必须带适用蛋白、位置、assay、condition、极性和原文定位。LLM 抽取的 Claim 初始 authority 为 `agent_inferred`，不能自动升级成 curated fact。

### Phase 4：外部图与跨蛋白迁移

当至少有第二个蛋白任务并需要功能迁移时，再接 Biolink/BioCypher/PrimeKG 子图。只导入查询可达、许可证允许且版本明确的最小投影，不复制整个大图。

## 10. 一个具体示例

以候选 `A3W` 为例，目标图不是一条“W 有利”边，而是一组带上下文的事实：

```text
(Variant:v1 {mutation_notation:A3W})
  -[:VARIANT_OF]-> (Protein:PTEST)
  -[:HAS_SEQUENCE]-> (Sequence:sha256/...)
  -[:HAS_MUTATION]-> (Mutation:PTEST:A3W)

(Mutation:PTEST:A3W)
  -[:AT_POSITION]-> (ResiduePosition:PTEST:3)

(Observation:o1 {fitness:0.90, round:1, source:experiment})
  -[:OBSERVES_VARIANT]-> (Variant:v1)
  -[:MEASURED_IN]-> (Assay:binding)
  -[:UNDER_CONDITION]-> (Condition:pH7)
  -[:REVEALED_IN]-> (CampaignRound:run1:1)

(Prediction:p1 {mean:0.80, std:0.10, ood:0.20})
  -[:PREDICTS]-> (Variant:v1)
  -[:GENERATED_BY]-> (ModelRun:m1:r2)

(Evidence:e1 {direction:supports, confidence:0.80})
  -[:ABOUT]-> (Variant:v1)
  -[:DERIVED_FROM]-> (StructureArtifact:af-test)

(Hypothesis:h1 {status:testing})
  -[:SUPPORTED_BY]-> (Evidence:e1)
  -[:TESTED_BY]-> (Observation:o1)
```

LLM 查询时获得的是：A3W 在明确 assay/condition 下的真实值、模型估计、结构证据和来源，而不是脱离背景的“3W beneficial”。如果另一个 assay 中 A3W 下降，两条 Observation 可以共存，查询按 scope 返回冲突和适用边界。

## 11. 数据质量与验收指标

### 11.1 构建质量

- schema validation error = 0；
- 悬空关系 = 0；
- P0 provenance coverage = 100%；
- Observation 的 assay/condition/round coverage = 100%；
- sequence checksum coverage = 100%；
- sequence–structure mapping coverage 和 confidence 分布；
- 重复实体率、alias merge 错误率；
- 冲突属性/关系率；
- 每个 adapter 的输入、保留、过滤和失败计数。

### 11.2 决策价值

- 相对 observations-only 的 top-k ranking 增益；
- 每层加入后的 best observed fitness/regret 改善；
- 少样本区间的 calibration 和 OOD 改善；
- 结构/进化证据方向与真实 Observation 的一致率；
- 单位 token/tool-call 带来的 fitness 增益；
- 动态 snapshot 相对冻结 snapshot 的后续轮次收益。

### 11.3 泄漏与时间验证

- 所有查询必须指定 `as_of_round` 或 snapshot；
- 用 observation event time 而不是 ingest time 决定是否可见；
- final/oracle 节点不进入训练/Agent namespace；
- 外部数据库 release date 记录到 Artifact；
- link prediction 或 KG embedding 使用按时间/实体隔离的 split，参考 OpenBioLink 的泄漏防护思想；
- PG-LLM 正式评测使用评测前冻结快照，候选和得分不写回。

## 12. 风险与缓解

| 风险 | 表现 | 缓解 |
|---|---|---|
| KG 越大越好 | 噪声和远端知识淹没局部证据 | 按 P0–P3 建设、query-specific subgraph |
| 坐标错配 | PDB residue 与序列位置错连 | 显式 Mapping 实体/边、置信度和 alignment artifact |
| 重复来源增信 | 多数据库转载同一实验 | source_group 内 max，独立组间融合 |
| Prediction 污染事实 | 模型输出被当成测量 | 类型、predicate、writer 权限分开 |
| 文献抽取幻觉 | LLM Claim 伪装 curated fact | authority 分级、证据句定位、人工/规则校验 |
| 原子图过细 | 图规模大但查询价值低 | 只保存突变局部交互摘要，原文件走 Artifact |
| 过早引入重型平台 | 开发与运维成本高 | 先原生接口 + SQLite sink，需求证明后接 Neo4j/BioCypher |
| 消融不正交 | 无法判断哪层有效 | adapter/layer/modality/type/fusion 分离开关 |

## 13. 最终推荐

最先构建的不是文献或大规模通路，而是“身份—坐标—实验—轮次—来源”骨架。它们贡献高、实现难度低，而且是所有高级知识正确连接的前提。

具体优先顺序为：

1. P0 的九类锚点实体和上下文化 Observation；
2. 当前已有的 Prediction、Evidence、Hypothesis，加上 Decision；
3. EvolutionProfile 和 ResidueEnvironment；
4. 任务相关 AtomicInteraction；
5. Domain/GO 和带来源定位的 Publication/Claim；
6. 只有在跨蛋白任务出现后才引入广域 biomedical KG。

技术上保持 `Adapter -> Normalize -> Fuse -> Validate -> Sink`，实验上保持每个 adapter、layer、modality、entity/relation type 可独立关闭。这样 KG 是否真正提高 fitness 优化，不需要依赖主观解释，而可以由严格配对消融回答。

## 14. 主要参考资料

- [Democratizing knowledge representation with BioCypher, Nature Biotechnology 2023](https://www.nature.com/articles/s41587-023-01848-y)；[GitHub](https://github.com/biocypher/biocypher)
- [Biolink Model, Clinical and Translational Science 2022](https://ascpt.onlinelibrary.wiley.com/doi/10.1111/cts.13302)；[GitHub](https://github.com/biolink/biolink-model)
- [LinkML GitHub](https://github.com/linkml/linkml)
- [KGX GitHub](https://github.com/biolink/kgx)
- [Building a knowledge graph to enable precision medicine / PrimeKG, Scientific Data 2023](https://www.nature.com/articles/s41597-023-01960-3)；[GitHub](https://github.com/mims-harvard/PrimeKG)
- [Multimodal learning with graphs, Nature Machine Intelligence 2023](https://www.nature.com/articles/s42256-023-00624-6)
- [OntoProtein, ICLR 2022](https://openreview.net/forum?id=yfe1VMYAXa4)；[GitHub](https://github.com/zjunlp/OntoProtein)
- [OpenBioLink, Bioinformatics 2020](https://academic.oup.com/bioinformatics/article/36/13/4097/5825726)；[GitHub](https://github.com/OpenBioLink/OpenBioLink)
- [PheKnowLator, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11009265/)；[GitHub](https://github.com/callahantiff/PheKnowLator)
- [Graphiti / Zep temporal KG](https://arxiv.org/abs/2501.13956)；[GitHub](https://github.com/getzep/graphiti)
- [ProteinGym, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee-Paper-Datasets_and_Benchmarks.pdf)；[GitHub](https://github.com/OATML-Markslab/ProteinGym)

