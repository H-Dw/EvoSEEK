# GB1 三类特征工具参数的文献依据与优化审计

日期：2026-08-19  
审计对象：`configs/knowledge/gb1_features.example.yaml`、`configs/resources/aaindex_minimal.yaml` 以及 `PhyschemDescriptorProvider`、`MSAProfileProvider`、`StaticStructureProvider` 实现。第 3–6 节保留对审计前 v1 参数的判断；第 7 节记录已经落地的 v2/v3 设计。

## 1. 结论摘要

审计前 v1 配置适合继续作为“可审计、不可直接驱动选择”的示例配置，但不适合把三通道的原始分数解释为 fitness、稳定性或亲和力效应。2026-08-19 已按本报告建议更新 conservation 配置和 provider；其余两通道仍维持描述性、未校准状态。

1. **理化通道**：残基平均质量、Kyte–Doolittle hydropathy 和侧链体积表基本有明确来源；`nominal_charge` 中 D/E/K/R 的符号是合理简化，但 H=0.1 依赖约 pH 7 和游离侧链 pKa 假设。真正缺乏文献标定的是“按每个描述符全局极差归一、等权平均、再以 `1/(1+mean_change)` 生成 conservativeness”的聚合规则，以及只对 G/P/C 设置特殊标记的规则。它们是项目启发式，不是 Grantham distance、BLOSUM 或实验校准的保守/激进替换分类。
2. **保守性通道（v1 问题）**：`identity_threshold=0.8` 有 DCA 序列重加权先例；但 v1 的 `pseudocount=0.5` 被逐状态加入 20 个单点类别和 400 个成对类别，不等于 DCA 文献中的总伪计数/混合权重。对 GB1 A3M，它使先验占完整单点列总质量的 39.8%，占成对频率总质量的 93.0%。`minimum_neff=12` 没有找到通用文献阈值，而且 `Neff/L=0.27` 不足以支持未经质量校正的 pairwise coevolution。v2 已改用 Neff-scaled total prior、拆分质量门槛并关闭当前 pairwise 项。
3. **结构通道**：5 Å contact、3.5 Å H-bond donor–acceptor 距离、4 Å salt bridge、1.4 Å SASA probe、2.3 Å Cys Sγ–Sγ 阈值均有文献或成熟工具先例；然而当前 H-bond 只检查任意 N/O 距离、没有供体/受体类型和角度，因此只能叫候选近邻。`sasa_sphere_points=96` 是数值精度设置；`dense_contact_count=12`、`clash_distance_fraction=0.75` 以及手写 φ/ψ 矩形是项目启发式。
4. **GB1 特异限制**：1PGB 是 56 aa、单链 A 的游离 B1 domain，能描述折叠环境，不能产生 IgG-Fc 跨链界面证据。当前任务却是 `GB1_IgG_binding_Wu2016`，所以 `interface_cutoff_angstrom=5.0` 虽有几何依据，在当前资源上实际不起作用。1FCC 提供 protein G C2–Fc 复合物，可作为同源界面参考，但必须显式进行 B1/C2 序列和编号映射，不能无条件替换 1PGB。
5. **优先级**：第一优先级是关闭/重写当前 pairwise-frequency 项；第二优先级是删除结构“风险分数”中的两个任意规则；第三优先级是保留原始理化描述符、另增连续 Grantham/BLOSUM 特征，而不是继续把启发式 conservativeness 当作保守替换强度。

当前三个 provider 都配置为 `contributes_to_selection: false` 且 `calibration: none`。因此这些问题暂时不会直接改变 acquisition score，但 evidence 会进入 KG/LLM 上下文并影响假设措辞和候选解释，仍应避免把启发式数值包装成已验证的效应估计。

## 2. 方法与证据等级

本审计先逐行检查当前配置和实现，再使用学术检索与多搜索引擎发现候选文献，最终尽量回到论文、PubMed、RCSB、ExPASy、Biopython 或官方源代码页面核实。搜索结果本身不作为证据。

参数分为三类：

- **A：直接有据**——当前数值及其定义与论文/成熟工具基本一致；仍不等于已经对 GB1 fitness 校准。
- **B：有先例但实现或任务依赖**——数值可作候选阈值，但当前实现省略了文献中的其他必要条件，或阈值依赖数据集。
- **C：项目启发式/占位**——没有发现与当前公式和用途相符的来源，或配置本身已标记 project prior。

## 3. 审计前 v1 参数总表

| 通道 | 当前参数/规则 | 等级 | 判断 | 建议 |
|---|---:|:---:|---|---|
| Physchem | ExPASy average residue mass | A | 数值与 ExPASy FindMod 的 peptide-residue average masses 一致 | 保留并补正式来源元数据 |
| Physchem | Kyte–Doolittle hydropathy | A | 表值与 KYTJ820101/原始尺度一致 | 保留为独立描述符；不要解释为 fitness |
| Physchem | side-chain volume 表 | A/B | 数值与 Zamyatnin 1972 residue-volume 表一致，但当前 accession 写成项目近似值 | 将 accession 改为明确文献 ID，并记录单位/定义 |
| Physchem | D/E=-1、K/R=+1、H=0.1 | B/C | 前四者是近中性 pH 简化；H=0.1 隐含 pH/pKa，且蛋白环境会移动 pKa | assay pH 未知时保留为 nominal feature；有结构和 pH 时用 PROPKA 类方法 |
| Physchem | 极差归一、跨性质/位点等权平均 | C | 没有尺度协方差、任务权重或实验标定 | 保留各维原值；权重只从可见训练折学习 |
| Physchem | `1/(1+mean_change)` conservativeness | C | 单调压缩函数是项目规则，不是经典替换距离 | 改名 heuristic similarity 或不产出总分；新增 Grantham/BLOSUM |
| Physchem | G/P/C special flags | C | 三者确有特殊化学/构象性质，但当前二元规则和 fitness 关系未标定 | 拆成 glycine flexibility、proline backbone、cysteine chemistry 等明确特征 |
| Conservation | `identity_threshold=0.8` | A/B | Morcos 等 DCA 工作使用 >80% identity 邻居重加权，并报告 70%–90% 范围较稳健 | 暂保留 0.8，同时报告 0.7/0.8/0.9 敏感性 |
| Conservation | `pseudocount=0.5` per state | C（当前用途） | 可能被误读为 DCA λ=0.5，但代码逐类别加 0.5；pair prior mass=200 | 改成显式 total prior/mixing weight；不要把相同 per-cell α 用于 q 与 q² |
| Conservation | `minimum_neff=12` | C | 仅决定 quality flag；没有找到适用于 56 aa GB1 和当前 pair-frequency 算法的通用阈值 | 分离 single-site 与 pairwise QC；pairwise 至少按 Neff/L 约束 |
| Conservation | `minimum_sequence_coverage=0.7` | B/C | 是常见操作性过滤量，但 HHfilter 本身并没有统一 70% 默认值 | 保留为显式工程选择并做 0.7/0.8/0.9 敏感性 |
| Conservation | `maximum_sequence_gap_fraction=0.5` | C | 对当前等长、无 query gap 的 1PGB A3M，coverage≥0.7 已使该规则基本冗余 | 删除，或收紧到与 coverage 一致的 0.3，并分别报告非标准字符 |
| Structure | contact 5.0 Å | A/B | “任意重原子最短距离 <5 Å”是常见 residue-contact 定义 | 保留，但把序列相邻与非局部接触分开 |
| Structure | interface 5.0 Å | A/B | 常见跨链界面定义 | 保留；必须使用含生物学 partner/assembly 的结构 |
| Structure | H-bond 3.5 Å | B | 距离有据，但文献还要求 donor/acceptor 化学与角度 | 保留距离；增加类型和角度≥100°，排除邻接伪候选 |
| Structure | salt bridge 4.0 Å | A/B | 经典 N–O cutoff；但 His 质子化和溶剂/pH 依赖 | 保留为候选，加入 protonation policy 和 assay pH |
| Structure | SASA probe 1.4 Å | A | 约为水分子半径，也是成熟 Shrake–Rupley 实现默认值 | 保留 |
| Structure | SASA 96 sphere points | B/C | 数值积分精度，不是生物学阈值；成熟实现常用 100 且允许提高精度 | 对 GB1 提到 960，或先做收敛标准再选 200/480/960 |
| Structure | `dense_contact_count=12` | C | 未发现依据；强依赖 cutoff、蛋白大小及是否计入相邻残基 | 删除二元阈值；保留连续接触数/分位数 |
| Structure | `clash_distance_fraction=0.75` | C | 与 MolProbity 的绝对 vdW overlap >0.4 Å 定义不同 | 改为 overlap Å，并加氢、化学例外和每千原子归一化 |
| Structure | Cys Sγ–Sγ 2.3 Å | A | 文献直接以 2.3 Å 识别 disulfide | 保留；同时读取 mmCIF/SSBOND 注释并报告异常长键 |
| Structure | 手写 alpha/beta φ/ψ 方框 | C | 仅为 coarse Ramachandran bins，不是 DSSP/STRIDE | 使用 DSSP；若保留则明确标为 coarse geometry |

## 4. 理化通道详细分析

### 4.1 有文献依据的原始表值

`residue_mass` 与 [ExPASy FindMod amino-acid mass table](https://web.expasy.org/findmod/findmod_masses.html) 的平均 peptide-residue masses 一致，例如 A=71.0788、W=186.2132。`hydropathy` 与 [ExPASy 的 Kyte–Doolittle scale](https://web.expasy.org/protscale/pscale/Hphob.Doolittle.html) 一致；原始论文 DOI 为 [10.1016/0022-2836(82)90515-0](https://doi.org/10.1016/0022-2836(82)90515-0)。

当前 `side_chain_volume` 的 20 个值实际对应 Zamyatnin 的 residue-volume 表，而不仅是无法追溯的“项目近似”；原始工作可由 [PubMed 记录](https://pubmed.ncbi.nlm.nih.gov/4566650/) 和 DOI [10.1016/0079-6107(72)90005-3](https://doi.org/10.1016/0079-6107(72)90005-3) 核实。因此这里首先需要修正的是 provenance/accession，而不是数值本身。

这些原始性质有来源，并不意味着它们对 GB1 binding fitness 的线性权重有来源。Kyte–Doolittle 原始用途是序列 hydropathy profile，不是四位点组合突变的 affinity predictor。

### 4.2 属于项目启发式的聚合规则

代码对每项性质使用 `(mutant-WT)/(max-min)` 的绝对值，随后跨性质、位点等权平均，再计算：

```text
conservativeness = 1 / (1 + mean_normalized_absolute_delta)
```

这个公式保证结果落在 (0,1]，但没有处理性质之间的相关性、量纲可靠性、位点环境、assay 类型或多突变互作。它也不是经典的 Grantham distance。Grantham 的原始方法明确联合 composition、polarity 和 molecular volume，并根据观察到的替换频率构造距离；可见 [Grantham 1974](https://pubmed.ncbi.nlm.nih.gov/4843792/)。BLOSUM 则从保守 blocks 的观测替换频率推导 substitution score；可见 [Henikoff & Henikoff 1992](https://pubmed.ncbi.nlm.nih.gov/1438297/)。

因此，如果需求包括“保守替换与激进替换”，建议输出连续的 `grantham_distance` 和 `blosum62_score`，而不是立即再引入一组无出处的 hard class thresholds。类别阈值应在训练/验证折上预注册或学习，最终测试标签必须隔离。

### 4.3 charge 的限制

`H=0.1` 可被理解为游离 His 在约 pH 7、pKa 约 6 时的平均质子化分数近似，但当前 assay pH 为空，局部介电、埋藏、氢键和盐桥又会移动蛋白内 pKa。PROPKA3 的工作显示结构环境对 Asp/Glu/Tyr/Lys/His pKa 预测均重要；见 [Olsson et al. 2011](https://pubmed.ncbi.nlm.nih.gov/26596171/)。因此该表只能保留 `nominal_charge` 名称，不能升级为局部 electrostatic effect。

## 5. 保守性通道详细分析

### 5.1 0.8 identity 有先例，但不是唯一正确值

当前权重为每条序列 `1 / number_of_neighbors(identity >= threshold)`。Morcos 等在 DCA 中使用超过 80% identity 的邻居计数进行重加权，并指出 70%–90% 范围内结果对具体阈值相对稳健；见 [Direct-coupling analysis, PNAS 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3241805/)。所以 `0.8` 属于有直接先例的默认值。

但本地敏感性检查显示，对同一个 D40K 示例，阈值从 0.7/0.8/0.9 变化时：

| identity threshold | Neff | Neff/L | 当前总 log-odds |
|---:|---:|---:|---:|
| 0.7 | 9.78 | 0.175 | -5.60 |
| 0.8 | 15.13 | 0.270 | -8.18 |
| 0.9 | 18.98 | 0.339 | -9.60 |

这说明“阈值有文献先例”不等于当前 score 已稳健。LLM/KG 应同时看到 threshold sensitivity 或 uncertainty，而不是只看到单个 -8.18。

### 5.2 pseudocount=0.5 的语义不匹配

当前实现给每个单点氨基酸类别加 0.5，因此 total single prior mass = `20×0.5=10`；给每个氨基酸对加 0.5，因此 total pair prior mass = `400×0.5=200`。当前 `non_pairing.a3m` 经筛选后有 39 条序列、`Neff=15.133`，四个可变位点覆盖率均为 1.0，所以：

- 单点频率先验占比：`10/(10+15.133)=39.8%`；
- 成对频率先验占比：`200/(200+15.133)=93.0%`。

DCA 文献中的频率平滑将总伪计数 λ 分配到 q 或 q² 个状态，并常令 λ 与 `Meff` 成比例；不是对 q 和 q² 的每个 cell 使用同一个 0.5。明确的公式示例也可见 [hoDCA 方法](https://pmc.ncbi.nlm.nih.gov/articles/PMC6311078/)。

对 D40K，把当前 per-cell pseudocount 从 0.5 改为 0.1/0.05/0.01 时，总分分别从 -8.18 变为 -14.13/-16.83/-23.22。简单减小 0.5 并不能解决问题，只会让未观察类别的 log-odds 更极端。需要改变参数语义，例如：

```text
single: f_i(a)  = (n_i(a)  + lambda_total/q)   / (N_i  + lambda_total)
pair:   f_ij(ab)= (n_ij(ab)+ lambda_total/q^2) / (N_ij + lambda_total)
```

并把 `lambda_total` 或 `pseudocount_weight=lambda_total/Neff` 作为明确、版本化参数。若要复现某个 DCA 方法，应实现该方法完整的 gap alphabet、重加权和直接耦合推断，而不是只借用一个相同数值。

### 5.3 当前 pairwise 项不是 DCA，也会重复计数单点效应

代码计算所有四个可变位点的 raw pair frequency ratio，并把六个 pair log-ratios与四个 single-site log-ratios直接相加。它没有用 Potts/global model 区分直接与间接相关；而 DCA 的核心正是解除间接相关。对于只有 D40K 的单突变，D40 的边际变化会在 single term 中出现一次，又在 D40–39、D40–41、D40–54 三个 pair term 中出现，不能视作纯 epistasis。

序列深度也不足。当前三份文件在相同过滤参数下为：

| A3M | 原始条数 | 过滤后条数 | Neff | Neff/L |
|---|---:|---:|---:|---:|
| `non_pairing.a3m`（当前） | 41 | 39 | 15.13 | 0.270 |
| `pairing.a3m` | 120 | 120 | 25.83 | 0.461 |
| `hmmsearch.a3m` | 457 | 251 | 5.85 | 0.104 |

经典 evolutionary-coupling 工作通常要求更深的 MSA；一项复合物研究概括早期经验为至少约 1 个 non-redundant sequence/residue，并为低深度另行设计 quality score，最低扩展到约 0.3/residue；见 [Hopf et al., eLife 2014](https://elifesciences.org/articles/03430)。当前 0.270 低于前者，也略低于后者，而当前代码没有对应的低深度质量模型。`pairing.a3m` 是配对/物种组织的输入，不应仅因 Neff 较高就在单体 profile 中静默替换 `non_pairing.a3m`。

建议在当前 GB1 配置中：

1. 暂时设置 `pairwise_enabled: false`，只把 single-site profile 作为低置信度 evolutionary prior；
2. 将 `minimum_neff=12` 拆成 single-site QC 与 pairwise QC；pairwise 默认要求 `Neff/L>=1`，若采用 0.3 下限必须同时实现并验证对应 quality score；
3. 给每个位点输出 effective count、coverage、bootstrap/Dirichlet interval；
4. 在 0.7/0.8/0.9 identity 和若干 coverage 阈值上做预注册敏感性分析；
5. 若最终想得到 epistasis，应采用真正的 Potts/DCA 或用可见 DMS 训练数据估计 interaction，并单独进入 `MutationPair/Interaction — HAS_EPISTASIS_ESTIMATE → EffectEstimate`。

## 6. 结构通道详细分析

### 6.1 contact 与 interface

以任意重原子最短距离约 5 Å 定义 residue contact 是常见做法；例如 [Marks et al. 2011](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0028766) 使用最短原子距离判定结构近邻，其他 residue-interaction network 工作也明确使用任意重原子 <5 Å 的定义。这个阈值可以保留，但“contact count≥12 即风险”没有文献支持。

本地 1PGB 敏感性结果如下：

| 位点 | 4.0 Å | 4.5 Å | 5.0 Å | 5.5 Å | 6.0 Å |
|---:|---:|---:|---:|---:|---:|
| V39 | 6 | 9 | 11 | 11 | 16 |
| D40 | 4 | 4 | 7 | 8 | 9 |
| G41 | 5 | 7 | 7 | 7 | 7 |
| V54 | 9 | 13 | 15 | 16 | 16 |

因此 `dense_contact_count=12` 在 5 Å 时把 V39（11）与 V54（15）硬切开，且在 cutoff 变为 4.5 Å 时结果又变化。建议输出连续 contact count、非局部 contact count、packing/RSA，不再把 12 作为静态风险分数的一部分。

[RCSB 1PGB](https://www.rcsb.org/structure/1PGB) 是 1.92 Å、56 residue、单体 A1。对 IgG binding 任务，它没有 partner chain，所以当前所有 `interface_contact_count` 必然为 0。可增加 [RCSB 1FCC](https://www.rcsb.org/structure/1FCC) 作为 protein G C2–human Fc 复合物的同源界面资源；该结构分辨率约 3.2 Å，且不是目标 B1 的完全同序列结构，必须保留 1PGB 用于折叠环境，并对 1FCC 做显式 homolog mapping。蛋白 G–Fc 的 alanine-scan 也证明该界面由特定极性 hot spot 主导，不能由游离单体 contact density 代替；见 [Sloan & Hellinga 1999](https://pmc.ncbi.nlm.nih.gov/articles/PMC2144421/)。

### 6.2 SASA

当前算法是 Shrake–Rupley rolling-sphere approximation；原始论文为 [Shrake & Rupley 1973](https://doi.org/10.1016/0022-2836(73)90011-9)。1.4 Å probe 是水探针的成熟默认值；[Biopython ShrakeRupley 文档](https://biopython.org/docs/latest/api/Bio.PDB.SASA.html) 也以 1.40 Å、100 surface points 为默认，并明确说明提高点数会增加精度。RSA 分母使用的 extended-tripeptide maximum ASA 与 [Tien et al. 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3836772/) 一致。

96 points 与 960 points 的本地比较：

| 位点 | SASA@96 Å² | SASA@960 Å² | 相对差异（相对 960） |
|---:|---:|---:|---:|
| V39 | 17.17 | 15.02 | +14.3% |
| D40 | 121.69 | 123.55 | -1.5% |
| G41 | 20.80 | 22.96 | -9.4% |
| V54 | 0.00 | 0.57 | 绝对差很小，但 96 点给出零 |

因此 96 不是错误的文献阈值，而是对本目标略粗的数值精度。GB1 仅 56 aa，建议审计/缓存计算使用 960；若要降低成本，先规定“关键位点 RSA 与 960 点参考差异 <2%”的收敛准则，再选择 200 或 480。

### 6.3 hydrogen bond、salt bridge、disulfide

H-bond 的 3.5 Å donor–acceptor 距离有充分先例，但结构调查还要求正确的 donor/acceptor 类型和角度；例如 [Adhikari et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8261469/) 使用 donor–acceptor≤3.5 Å 且 antecedent angle>100°。当前实现只要两残基间任意 N/O≤3.5 Å 就计数，还没有排除共价相邻/错误化学类型，所以应继续称 `candidate`，不能称真实 H-bond。

Salt bridge 的 N–O≤4 Å 与 Barlow–Thornton 定义一致，综述核实见 [Donald et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3069487/)。但当前把所有 His ND1/NE2 都视为正电，不考虑 assay pH、局部 pKa 或 tautomer，因此也只能是 candidate。

2.3 Å Cys Sγ–Sγ cutoff 有直接数据集先例；[Bhattacharyya, Pal & Chakrabarti 2004](https://pubmed.ncbi.nlm.nih.gov/15576382/) 以 2.3 Å 识别 disulfide（DOI [10.1093/protein/gzh093](https://doi.org/10.1093/protein/gzh093)）。该值可以保留，同时应读取结构中的显式连接注释，避免把少数异常长但已注释的键静默丢弃。

### 6.4 clash 与 secondary structure

当前 clash 条件是 `distance < 0.75 × (r_vdw1+r_vdw2)`。Bondi vdW radii 本身有经典来源，见 [Bondi 1964](https://pubs.acs.org/doi/10.1021/j100785a001)，但乘 0.75 的规则未找到与当前用途相符的来源。以 C/O 为例，该规则只在距离 <2.415 Å 时报警，即 overlap >0.805 Å；这比 [MolProbity](https://pmc.ncbi.nlm.nih.gov/articles/PMC2803126/) 的非 donor–acceptor vdW overlap >0.4 Å 严重 clash 定义更不敏感，而且当前没有加氢、优化 Asn/Gln/His 或处理 H-bond 例外。建议改为 `vdw_overlap_angstrom`，若不能复现 MolProbity 全流程则明确标记 heavy-atom severe-overlap heuristic。

手写的 alpha/beta φ/ψ 矩形只反映粗 Ramachandran 区域。标准二级结构赋值还依赖氢键模式；应优先采用 [Kabsch–Sander DSSP](https://onlinelibrary.wiley.com/doi/10.1002/bip.360221211)，或者把当前结果名称固定为 `coarse_phi_psi_geometry`，不与 DSSP helix/sheet 等同。

## 7. 建议的下一版参数合同

下面是设计建议，不是已经验证优于当前配置的结果。

```yaml
providers:
  physchem:
    kind: aaindex_delta
    options:
      aggregate_score: none          # 保留各维原始值
      substitution_descriptors:
        - grantham_distance
        - blosum62_score
      charge_mode: nominal           # assay pH/结构 pKa 不足时不得升级

  conservation:
    kind: msa_profile
    options:
      identity_threshold: 0.8        # 有 DCA 先例；同时输出 0.7/0.9 敏感性
      pseudocount_mode: neff_scaled_uniform
      pseudocount_weight: 0.5        # 总 prior mass = weight * Neff
      minimum_single_site_neff: 10.0
      minimum_site_effective_count: 5.0
      minimum_sequence_coverage: 0.7
      maximum_sequence_gap_fraction: 0.3
      single_site_aggregation: sum_log_odds
      pairwise_enabled: false        # 当前 GB1 Neff/L=0.27
      pairwise_mode: marginal_corrected_log_odds
      pairwise_minimum_neff_per_length: 1.0
      estimated_parameters:
        - pseudocount_weight
        - minimum_single_site_neff
        - minimum_site_effective_count
        - minimum_sequence_coverage
        - maximum_sequence_gap_fraction
        - pairwise_minimum_neff_per_length

  structure:
    kind: static_structure
    options:
      contact_cutoff_angstrom: 5.0
      interface_cutoff_angstrom: 5.0
      hbond_cutoff_angstrom: 3.5
      hbond_min_antecedent_angle_degrees: 100.0
      salt_bridge_cutoff_angstrom: 4.0
      sasa_probe_radius_angstrom: 1.4
      sasa_sphere_points: 960
      dense_contact_count: null       # 不再二元切分
      clash_overlap_angstrom: 0.4
      disulfide_sg_cutoff_angstrom: 2.3
      secondary_structure_method: dssp
```

这里的 `pseudocount_weight=0.5` 只表示“总伪计数相对 Neff 的候选权重”，并与 q/q² 均匀分配公式绑定；建议在训练折进行 0.1/0.25/0.5 等预注册敏感性分析。它不再复用旧版 per-cell 语义。`minimum_single_site_neff=10`、`minimum_site_effective_count=5`、coverage/gap filters 和 `pairwise_minimum_neff_per_length=1.0` 都很可能有助于质量控制，但仍是待任务验证的估计值，不应冒充性能最优阈值。

## 8. 验证计划

1. **单元合同**：测试 single/pair prior 总质量不随类别数从 q 到 q² 暴涨；测试只发生单突变时 pair term 不重复计数单点边际效应。
2. **输入敏感性**：记录 A3M 文件、hash、过滤前后条数、Neff/L、每位点覆盖；对 identity、coverage 和 pseudocount 输出 score interval。
3. **结构数值收敛**：以 960 点为参考，要求关键位点 RSA 的选定低成本设置误差 <2%；contact 同时报告 4.5/5.0 Å 敏感性。
4. **界面资源**：保持 1PGB 为 target fold 结构，另建 1FCC homolog/interface resource，保存序列比对、position map、assembly 和分辨率 caveat。
5. **预测效度**：只在训练/validation DMS 折上比较 Spearman、NDCG、top-k enrichment 和校准；最终测试不可用于选阈值。分别做 physchem、conservation、structure 及联合 ablation。
6. **KG/LLM 语义**：raw descriptor 进入 `SubstitutionDescriptor`、`EvolutionProfile`、`ResidueEnvironment`；只有经过可见数据校准的结果才能成为 `MutationEffectEstimate`。LLM prompt 必须保留 `uncalibrated`、`not fitness`、Neff/coverage、结构是否含 partner、是否建模 mutant side chain 等 caveat。

## 9. 审计边界

- 2026-08-19 已将 conservation 建议落入 GB1 example config 和 provider v3；这证明配置与 evidence 合同可运行，不代表已经提高 GB1 fitness 优化效果。
- 文献中的常用 cutoff 只证明“该几何/统计定义曾被合理使用”，不能替代对本 assay 的校准。
- 本报告聚焦三类 feature tools；config 中 `kg.shrinkage_pseudocount=3.0`、`kg.confidence_base=0.25`、`kg.support_gain=0.03`、`kg.confidence_cap=0.85` 已明确标记为 `project_prior_pending_calibration`，不应与上述文献支持参数混为一谈。
