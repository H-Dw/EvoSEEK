# KnowledgeEngine Phase 0–5、7–8 实施记录

> 实施日期：2026-08-17  
> 对应计划：`docs/knowledge-engine-feature-reasoning-and-dynamic-configuration-plan.md`  
> Phase 6：明确保留为未来 TODO，本次没有运行或接入 Rosetta、FoldX、ProteinMPNN、ESM-IF。

## 1. 当前结果

本轮已把 KnowledgeEngine 从“GB1 固定启发式打分器”改造成由 Task、资源和版本化参数驱动的确定性证据编排器。底层 provider 不调用 LLM；Scientist 仍只消费 CampaignRunner 提供的 sanitized context 和白名单 KG 查询结果；wet/dry/KG/artifact 仍是科研状态源。

当前 GB1 生产配置的实际状态是：

- physchem：加载带 hash 的命名残基属性资源；在仅使用已 reveal observation 的前提下完成轮次内校准后，才允许参与选择。
- conservation：代码能力已实现，但 GB1 配置尚未提供 A3M/FASTA 或 MMseqs2 database，因此显式返回 `unavailable`，不使用 tolerated list 或 0.5 填充。
- structure：代码能力已实现，但 GB1 配置尚未提供结构资源和 residue map，因此显式返回 `unavailable`，不使用 `structure_risk` 填充。
- KG：仅使用当前运行已可见 observation 形成描述性关联；保留“非因果、可能遗漏 epistasis”的 warning。

这个状态是有意的 fail-closed 行为：代码实现完成不代表当前任务已经具备相应科学资源。

## 2. 分 Phase 改动

### Phase 0：legacy 风险护栏

- legacy physchem/conservation/structure statement 标为未验证启发式。
- legacy evidence 默认不进入候选选择。
- provider 缺失返回 `quality_status=unavailable`、`confidence=0`、`contributes_to_selection=false`，不再伪装成中性 0.5。
- 提供显式 `configs/ablations/knowledge/legacy.yaml` 用于旧行为消融，不作为生产默认值。

### Phase 1：动态 ProteinTaskContext

- 新增不可变 `ProteinTaskContext`，统一管理完整序列、mutable positions、WT 残基、编号映射、assay 条件和结构资源。
- Runner 将同一 context 注入 KnowledgeEngine、Scientist、候选生成器和 uncertainty selector。
- Scientist 输出的位点 key 按当前 Task 动态验证，不再固定为 39/40/41/54。
- 通用 ensemble 不再用 `VDGV` 计算训练深度，而是从训练序列确定性推导 reference code。
- GB1 全长 56 aa domain 序列已进入 Task 配置；GB1 常量仅保留在专用 data adapter 和测试数据中。

### Phase 2：Evidence、provider 与参数合同

- Evidence 增加 `raw_features`、quality、applicability、uncertainty、calibration、selection eligibility、warnings 和 provenance。
- SQLite KG 已增加兼容迁移列；旧数据库打开时自动补列。
- structured KG Evidence entity 同步保存扩展字段；未校准 descriptor 与 Hypothesis 使用 `CITES_EVIDENCE`，不错误升级为 `SUPPORTED_BY`。
- 新增 `KnowledgeProviderConfig`、`LearnableParameterSpec` 和 `parameter_set_id`。
- KG 的 3.0、0.25、0.03、0.85 已迁入版本化配置并标记来源、范围、变换、更新策略和可学习状态；运行时代码缺失这些参数时 fail closed。
- 运行 artifact 保存 protein context、provider status、parameter snapshot 和 evidence/calibration contract。

### Phase 3：理化与全序列属性

- 新增 YAML-backed 命名 descriptor provider；每个 property 保留 accession、资源 hash 和逐位点 delta。
- 输出 hydropathy、side-chain volume、nominal charge、residue mass 等独立维度，以及 Gly/Pro/Cys flags。
- 输出同长度 WT/mutant 的 molecular-weight、mean hydropathy、nominal-charge 和 aromatic-fraction delta。
- nominal charge 明确标注不等于局部 pKa；缺少 assay pH 时增加 warning。
- 未校准结果只作为 descriptor；不能直接宣称 affinity、stability 或 fitness。

当前仍未实现的低成本扩展：独立 Biopython/EMBOSS ProtParam parity adapter 与 PROPKA adapter。它们不影响当前 provider 的 fail-safe 合同。

### Phase 4：一次准备、多轮复用的 MSA provider

- 支持 FASTA/A3M 读取、A3M insertion 清理、coverage/gap 过滤、identity reweighting、Neff、column coverage、gap fraction、entropy 和 residue frequency。
- 输出独立位点 log-odds 与明确标为 `pairwise-frequency` 的组合项；没有把它称为 DCA/PLMC。
- 支持受配置约束的 MMseqs2 external search；保存 stdout/stderr、query、alignment 和 profile。
- cache key 包含 reference sequence hash、alignment resource hash 和全部配置；资源内容变化会触发 cache miss。
- prepare manifest 保存 cache key、profile hash、参数集和过滤设置；provider 在 Runner 初始化、第一轮 Scientist 之前准备，后续轮次复用。
- MSA query 必须与 Task reference sequence 一致，否则 fail closed。

当前未实现：HHblits、paired MSA、indel mapping、PLMC/DCA/EVmutation 与 MSA Transformer。它们属于可插拔后续 provider，不应由当前 pair-frequency 结果冒充。

### Phase 5：真实静态结构特征

- 原生读取 PDB 与常见 atom-site mmCIF。
- 支持 chain/residue/insertion-code 映射；结构残基与配置 WT 不一致时整个 provider fail closed。
- 计算 Shrake–Rupley 近似 SASA、Tien 2013 maximum-ASA 归一化 RSA、原子接触、chain interface 接触、主链 phi/psi 和 coarse secondary-structure label。
- 输出 H-bond、salt bridge、disulfide 和非相邻重原子 clash candidates，以及 missing-backbone-atom QC。
- statement 强制说明 mutant side chain 未建模/未 relax，不能宣称 folding 或 affinity。
- cutoff、sphere points 和 context-flag threshold 均由配置注入，不在 provider 中使用静默经验默认值。

当前未实现：DSSP/FreeSASA parity adapter、结构 ensemble、预测结构 pLDDT/PAE、mutant rotamer/relax 和动力学。这些不能由静态 WT 坐标特征替代。

### Phase 6：未来 TODO

保留后续接口方向，但本次不接入高计算开销模块：

- ProteinMPNN/ESM-IF sequence–backbone compatibility；
- FoldX stability/binding ΔΔG；
- Rosetta relax、cartesian ddG、interface analyzer；
- execution allowlist、container/license/version、timeout、replicate、seed、资源配额；
- raw energy 与 assay-calibrated score 的严格分离。

Phase 6 provider 未来必须复用当前 Evidence、quality、provenance、KG scope 和 artifact 合同，且不得成为 wet truth。

### Phase 7：KG Operator 与 Scientist context

- 新增理化、进化、结构、assay-association 和 evidence-provenance 查询 operator。
- 所有 operator 继续经过 `KGInteractionController` 的 allowlist、variant scope、round visibility、max rows 和 query budget。
- Scientist profile 要求比较多个候选（存在 compare pack 时）、至少两个可用证据维度，并显式处理 counterevidence/unavailable。
- Scientist 无 raw SQL/Cypher、filesystem、oracle、final-test、experiment backend 或 batch submission 权限。
- Hypothesis 的 `evidence_ids` 在 Pydantic/context validation 中必须属于本次可见 evidence。

### Phase 8：校准、可学习参数与消融

- `visible_linear` calibration 只从 KnowledgeEngine 已登记的 visible observations 拟合；candidate/oracle/final-test label 不进入拟合。
- 每轮开始根据上一轮已 reveal 数据重新生成校准副本；calibration provenance 保存 sample count、slope/intercept、residual 和 label scope。
- 只有 `quality=ok`、成功校准且 provider 配置允许的证据才进入选择。
- learnable 参数当前只记录 snapshot，不自动更新；避免未经 benchmark 的在线漂移。未来更新只能发生在 round boundary。
- 新增 legacy、physchem-only、MSA-only、structure-only、KG-only 和 full 消融配置。

当前未实现：跨 fold 的自动参数优化、分层 calibration report、held-out 上线门槛自动判定。它们需要新的 benchmark 运行，不应在没有实验结果时伪造完成状态。

## 3. 关键运行合同

1. `CampaignRunner` 仍拥有 campaign state、visibility、selection、hard validation、approval、wet reveal、KG write 和 artifact。
2. KnowledgeEngine 是确定性 provider 编排器，不自行调用 LLM。
3. 未校准 descriptor 可供 Scientist 阅读，但默认不参与 selection。
4. resource 缺失、mapping 错误或工具失败产生 unavailable/failed evidence，不产生中性或正向替代值。
5. MSA/structure 外部资源可以跨轮复用；wet measurement 必须遵守 reveal round。
6. SDK trace 不存在于本实现；原生 trace 仍是观测副本，不参与恢复判定。

## 4. 验证结果

- Ruff：全部 `src` 和本轮相关测试通过。
- 单元测试：排除一个与本任务无关、且对应文件没有本轮改动的 assay-list 既有失败后，104 passed、1 skipped；跳过项为本机未缓存 ESM-2 650M checkpoint。
- Integration：11 passed。
- E2E + leakage：5 passed。
- 新增测试覆盖：非 GB1 position mapping、source-backed physchem、MSA cache hit/miss、PDB coordinate features、可见数据校准、resource unavailable fail-closed、feature/provenance KG tool scope。

仓库当前仍有一个独立失败：`test_mvp_assay_list_excludes_out_of_scope_receptor_binding_assays` 期望 assay list 不含 SPIKE 条目，但被测试的 tracked 文件当前包含这些条目。本轮没有修改该文件，也没有为通过测试而删除用户现有数据范围。

## 5. 启用 MSA/structure 前必须补充的配置

要让当前 GB1 的 conservation 与 structure 从 unavailable 变为 ready，至少需要：

1. 为 MSA provider 配置经过审计的 A3M/FASTA `resource_path`，或 MMseqs2 database、数据库版本/hash、搜索参数和 timeout。
2. 为 Task 配置 PDB/mmCIF `structure_resources`、resource hash、chain、partner chains 和 task-position→structure-residue mapping。
3. 在 public/visible fold 上生成 calibration/ablation report；未通过门槛前保持 `contributes_to_selection=false`。

这三步属于资源准备和科学验证，不应由 LLM 自动猜测或由代码填入未经验证的 GB1 数值。
