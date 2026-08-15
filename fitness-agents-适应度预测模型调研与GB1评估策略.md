# 蛋白质适应度预测模型调研与 GB1 评估策略

> 版本：2026-08-15  
> 目标：为 Fitness Agents 选择独立的突变体 fitness 评估模型，并建立能够检验“发现高 fitness 变体”而非仅拟合平均趋势的评估协议。  
> 具体任务：GB1（Protein G B1 domain）对 IgG-Fc 的结合适应度。

## 0. 结论摘要

### 0.1 对当前问题的直接回答

1. **如果 GB1 只能选择一个主 fitness 模型，优先选择 Kermut。**它是带后验不确定性的监督式高斯过程，将 ESM-2 序列表征、ProteinMPNN 局部结构环境和突变位点几何关系组合到复合核中，原生支持多突变体。在 ProteinGym 的 GB1 直接结果里，Kermut 在 random、contiguous、modulo 三种切分上的 Spearman 分别为 **0.781、0.778、0.778**；其他深度模型在 random 切分上可能更好，但在更接近外推的切分中明显失效。例如 ProteinNPT 为 **0.858、-0.322、-0.322**，ESM-1v embedding 为 **0.731、0.310、0.259**。因此 Kermut 是目前证据最直接、外推稳定性和不确定性最均衡的 GB1 主模型。[Kermut 论文](https://papers.nips.cc/paper_files/paper/2024/hash/34547650b2ca69d91f3b3c3ae8b21962-Abstract-Conference.html)、[ProteinGym 官方逐数据集结果](https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/DMS_supervised/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv)  
   <!--ref:kermut2024--><!--anchor:section:Abstract--><!--ref:proteingym-supervised-gb1--><!--anchor:row:SPG1_STRSG_Wu_2016-->

2. **Kermut 应是“可更新的代理模型”，不应被称为客观真值 oracle。**真正客观的 GB1 目标仍是 IgG-Fc 结合实验；在离线模拟中，密封的 Wu 等人完整四位点 landscape 可以代替实验裁判，但不得进入生成、训练、调参或停止决策。模型的“独立”只能通过冻结版本、隔离标签、盲测和模型家族多样性来操作化。

3. **推荐一个数值头加一个排序头，而不是只优化 MSE。**主头使用 Kermut 预测连续 fitness 和不确定性；第二头使用 Bradley–Terry/ListMLE 类相对排序损失，或在资源允许时加入 FSFP-SaProt。GB1 的已发表实验显示，排序损失在 `low-vs-high` 切分上把 Spearman 从 **0.499** 提高到 **0.567**、Top-10 recall 从 **0.381** 提高到 **0.443**，更贴近“从低 fitness 数据中找到高 fitness 变体”的实际目标。[NeurIPS 2024 排序损失研究](https://proceedings.neurips.cc/paper_files/paper/2024/file/a9b938e79504889f905d549f8d53e405-Paper-Conference.pdf)  
   <!--ref:contrastive-ranking-2024--><!--anchor:page:7-->

4. **结合置信度辅助模型可用，但不应取代 assay-specific fitness 模型。**建议对 Kermut 排名前部的少量候选使用 Pythia-PPI 预测突变引起的 PPI 结合自由能变化，再用 AlphaFold 3 的 ipTM/interface PAE 检查复合物界面是否结构上可信。Pythia-PPI 更接近“结合强弱”，AF3 的置信度更接近“复合物构象是否可信”；ipTM 不是亲和力，也不是 Kd。两者应作为独立列、过滤器或重排序特征，不应未经 GB1 校准就与 fitness 简单加权相加。[Pythia-PPI](https://pmc.ncbi.nlm.nih.gov/articles/PMC12199698/)、[AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w)  
   <!--ref:pythia-ppi-2025--><!--anchor:section:Abstract--><!--ref:alphafold3-2024--><!--anchor:section:Inference-regime-->

5. **模型验收的主指标必须面向高 fitness 尾部。**Spearman 是全局排序的主指标，Pearson 和 MSE 只作为校准/数值误差指标；项目主终点应是 `Precision/Recall@B`、`NDCG@B`、`Enrichment Factor`、`Success@B`、最佳命中值和 simple regret，并分别在 top 0.1%、1%、5% 真值集合上报告。

### 0.2 最终推荐配置

| 角色 | 首选 | 备选或消融 | 使用原则 |
|---|---|---|---|
| 候选生成器 | 约束条件下的生成模型/离散搜索器 | ProteinMPNN、PoET、ESM 类生成器、遗传搜索 | 从定义好的序列设计域直接提出候选，不读取完整候选池 |
| 主 fitness 代理 | **Kermut** | 当前 one-hot + pairwise ensemble | 仅使用初始和已揭示标签；输出均值与后验不确定性 |
| 尾部排序头 | **Bradley–Terry/ListMLE** | FSFP-SaProt | 强化高 fitness 排序；不能接触密封测试集 |
| 结合能辅助 | **Pythia-PPI** | RDE-PPI 等 ΔΔG 模型 | 只对前部候选重排序/否决；先做 GB1 校准 |
| 复合物结构置信度 | **AF3 ipTM + interface PAE** | Boltz 结构置信度 | 只判断结构/界面可信度，不解释成亲和力 |
| 最终 oracle | **IgG-Fc 结合实验** | 离线基准中的密封完整 landscape | 与生成和建模进程物理/权限隔离 |

---

## 1. 问题重构：筛选已知候选池不等于序列设计

当前 GB1 实现把完整 landscape 切成初始集、验证集、最终测试集和 `oracle_pool`，Agent 只能在 `oracle_pool` 中排序选择。该模式适合测试**给定候选的评分和批次选择**，但没有测试模型能否在未枚举的序列空间中提出候选。

应区分三层：

1. **生成/搜索层**：在预先定义的序列设计域中提出新序列。
2. **fitness 代理层**：预测实验目标及其不确定性，负责排序和采集决策。
3. **实验/oracle 层**：返回真正的 IgG-Fc 结合 fitness；在离线基准中由密封标签模拟。

```mermaid
flowchart LR
    A["受约束序列空间"] --> B["生成器 / 搜索策略"]
    B --> C["去重与硬约束过滤"]
    C --> D["Kermut：fitness 均值 + 不确定性"]
    D --> E["排序头与多样性采集"]
    E --> F["Pythia-PPI / AF3：少量候选复核"]
    F --> G["IgG-Fc 实验或密封 oracle"]
    G --> H["仅释放本轮已测标签"]
    H --> D
```

### 1.1 “全蛋白质空间”必须被工程化定义

字面上的全部氨基酸序列空间不可穷举，也没有统一的可测 fitness。建议把目标写成**固定 GB1 骨架上的受约束开放设计域**：

- 固定序列长度或明确允许的 indel 范围；
- 指定可突变残基集合、单序列最大编辑距离和允许氨基酸；
- 保留关键折叠/界面残基或把其作为软约束；
- 明确单点、多点、indel 是否都在本项目范围内；
- 将可表达性、稳定性、聚集倾向等设为硬过滤或多目标约束。

**关键限制：Wu/FLIP GB1 数据只覆盖 39、40、41、54 四个位点的组合。**其名义空间是 `20^4 = 160,000`，实验获得约 149,361 个变体，而不是整条 GB1 序列的所有突变。因此：

- `GB1-4` 轨道可以测试“面对不可见的完整四位点 landscape，生成器能否盲找高结合变体”；
- `GB1-full` 轨道若允许其他位点或 indel，就不能再用这份数据提供真值，必须增加新的 DMS、实验或另一项可验证任务；
- 把完整四位点表从 Agent 内存中隐藏，只能消除候选池泄漏，不能把任务升级为全序列泛化。

[Wu 等人的 GB1 四位点 landscape](https://pmc.ncbi.nlm.nih.gov/articles/PMC4985287/)  
<!--ref:wu2016-gb1--><!--anchor:section:Results-->

---

## 2. 调研方法与证据等级

### 2.1 研究问题

- 哪些模型在蛋白质 fitness/variant-effect 预测中具有足够强的公开证据？
- 不同模型分别擅长零样本、少样本、结构外推、多突变、PPI 亲和力还是主动学习？
- 对 GB1 IgG-Fc 结合任务，哪个模型有最直接的同任务证据？
- 如何评价模型识别极少数高 fitness 变体的能力？

### 2.2 检索与核验流程

本次检索截至 **2026-08-15**，使用多搜索引擎和学术检索工作流交叉检索以下组合：`protein fitness prediction`、`variant effect prediction`、`GB1`、`IgG-Fc binding`、`few-shot protein engineering`、`ProteinGym supervised/zero-shot`、`PPI mutation ΔΔG`、`uncertainty calibration`、`top-k recall`。优先采用：

1. 同行评议论文原文和会议正式论文；
2. ProteinGym 官方仓库的逐 assay 结果，而不是模型论文自行汇总的单一数字；
3. 官方代码库/模型卡；
4. 对尚未正式发表的方法，明确标为预印本或 workshop 证据。

纳入标准是模型能够对突变序列输出连续分数或排序，且报告 DMS、蛋白质工程或 PPI 突变任务。排除只做结构生成、没有突变效应输出、或只有二手宣传而没有可核验结果的方法。

### 2.3 证据解释规则

- **直接证据**：同一个 `SPG1_STRSG_Wu_2016` assay 上的官方结果。
- **相邻证据**：其他 DMS/PPI 数据集上的公开基准或湿实验验证。
- **推断/建议**：由模型机制和相邻证据推导出的项目选择，不等价于 GB1 实测结论。
- 不跨 assay 比较原始 MSE；只在相同标签变换和切分内比较。

本地学术 MCP 在本次会话不可调用，因此 Nature Academic Search 的多源检索与 DOI 核验流程通过 OpenAlex 回退检索、出版商原文和官方仓库交叉完成。该限制不会改变下列 GB1 官方 CSV 数字，但意味着 2026 年尚未进入 ProteinGym 主榜的新模型不能获得完全同条件的直接排名。

---

## 3. 模型 landscape：性能与特长

| 模型/路线 | 输入与监督 | 特长 | 主要短板 | 对 GB1 的定位 |
|---|---|---|---|---|
| **Kermut**（NeurIPS 2024） | 少量 assay 标签；ESM-2 序列 + ProteinMPNN 结构 + 位点距离；高斯过程 | 多突变、结构感知、标签较少时稳定；自带后验不确定性；跨位点外推强 | 精确 GP 随样本数近似立方增长；实例级不确定性仍非完美 | **主模型首选** |
| **FSFP-SaProt**（Nature Communications 2024） | 20–100 个单点标签；PLM + LoRA + MAML + ListMLE | 极少标签、未见位点/突变、多点排序；直接优化排名 | 训练流程较重；没有原生概率不确定性；GB1 官方同切分表不如 Kermut 完整 | 排序头/第二模型首选 |
| **ProteinNPT**（NeurIPS 2023） | 序列与 assay 标签联合建模，可多性质预测 | random 插值、少样本、多个性质联合学习 | GB1 contiguous/modulo 直接失效，说明位置外推风险大 | 插值对照，不作主模型 |
| **eUniRep**（Nature Methods 2021） | 进化微调的 UniRep 表征 + 低样本回归 | 约 24 个测量也能启动；曾从千万级虚拟库中筛选；适合早期低 N | 较旧表征；缺少直接 GB1 优势与原生 UQ | 强低样本基线 |
| **EVOLVEpro**（Science 2025） | PLM 表征 + 轻量监督器 + 多轮主动学习 | 10 个左右初始数据/轮；适合真实闭环、多目标和跨蛋白工程 | 是优化工作流而非冻结、独立的客观评分器 | 采集策略参考，不作最终 oracle |
| **AIDO Protein-RAG / ProSST / VenusREM** | 零样本；分别强调检索、结构 token、序列/结构/MSA 检索融合 | 没有 assay 标签时提供合理先验；当前 ProteinGym 总榜较强 | GB1 直接相关性仍有限；大模型成本或 MSA/结构依赖 | 冷启动 prior 或集成成员 |
| **PoET / TranceptEVE / EVE / GEMME** | 家族序列、MSA 或自回归似然 | 同源家族信息充分时擅长进化约束；PoET 能做条件序列生成 | assay 特异结合目标不等同于进化适应度；GB1 直接表现不领先 | 生成/先验，不作 GB1 主评分器 |
| **Pythia-PPI**（NSR 2025） | 复合物结构 + 突变，预测 PPI ΔΔG | 任务语义最接近“突变是否增强蛋白–蛋白结合”；在 SKEMPI 2.0 上较强 | ΔΔG 不等于 GB1 DMS fitness；结构、构象和跨体系校准误差 | 前部候选辅助重排序 |
| **AlphaFold 3 / Boltz 结构置信度** | 复合物结构预测与界面置信度 | 检查是否形成可信复合物界面、排除明显结构不合理序列 | ipTM/iPAE 不是结合亲和力；Boltz-2 的 affinity head 主要面向小分子 IC50 | 结构否决器，不是 fitness 模型 |

来源：[Kermut](https://papers.nips.cc/paper_files/paper/2024/file/34547650b2ca69d91f3b3c3ae8b21962-Paper-Conference.pdf)、[FSFP](https://www.nature.com/articles/s41467-024-49798-6)、[ProteinNPT](https://papers.nips.cc/paper_files/paper/2023/hash/6a4d5d85f7a52f062d23d98d544a5578-Abstract-Conference.html)、[eUniRep](https://www.nature.com/articles/s41592-021-01100-y)、[EVOLVEpro](https://doi.org/10.1126/science.adr6006)、[ProSST](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3ed57b293db0aab7cc30c44f45262348-Abstract-Conference.html)、[VenusREM](https://pubmed.ncbi.nlm.nih.gov/40662802/)、[ProteinGym 官方仓库](https://github.com/OATML-Markslab/ProteinGym)、[Boltz 官方说明](https://github.com/jwohlwend/boltz)。  
<!--ref:kermut2024--><!--anchor:section:Methods--><!--ref:fsfp2024--><!--anchor:section:Results--><!--ref:proteinnpt2023--><!--anchor:section:Abstract--><!--ref:eunirep2021--><!--anchor:section:Abstract--><!--ref:evolvepro2024--><!--anchor:section:Abstract-->

### 3.1 为什么“更大的通用 PLM”不自动等于更好的 fitness 模型

通用 PLM 的似然分数主要反映自然序列分布、折叠与进化约束，而 GB1 实验目标是特定条件下的 IgG-Fc 富集。两者相关但不等价。2026 年关于 fitness 预测的 scaling 分析还显示，参数增大并不保证单调变好；当野生型残基似然处在极端位置时，模型可能把不同突变打成接近的分数。[Nature Computational Science 2026](https://pubmed.ncbi.nlm.nih.gov/42443524/)  
<!--ref:plm-scaling-fitness-2026--><!--anchor:section:Abstract-->

因此更合理的分工是：PLM/结构模型提供先验和表征，少量 GB1 assay 标签决定任务特异后验。

### 3.2 GB1 上的零样本直接证据

下表取自 ProteinGym 当前官方 `SPG1_STRSG_Wu_2016` 逐 assay 文件；所有数值越高越好。这里的 `Top-recall` 和 `NDCG` 使用 ProteinGym 官方定义，不能与下文按实验预算定义的 `Recall@B` 混为同一个统计量。

| 零样本模型 | Spearman | 官方 Top-recall | 官方 NDCG | 解读 |
|---|---:|---:|---:|---|
| AIDO Protein-RAG 16B | **0.347** | 0.383 | 0.387 | 全局相关略高，但计算代价大 |
| ProSST K=1024 | 0.343 | **0.392** | **0.388** | GB1 零样本中最均衡，结构 token 有帮助 |
| ProSST K=4096 | 0.310 | 0.377 | 0.363 | 仍强，但 K 并非越大越好 |
| VenusREM | 0.239 | 0.361 | 0.324 | 检索增强有效，但 GB1 全局排序仍有限 |
| ESM-2 15B | 0.199 | 0.320 | 0.315 | 大参数量未带来 GB1 最优表现 |
| ESM-IF1 | 0.180 | 0.334 | 0.301 | 逆折叠先验可用，但不够 assay-specific |
| PoET 200M | 0.114 | 0.269 | 0.252 | 更适合作为家族条件生成/先验 |
| TranceptEVE-L | 0.027 | 0.228 | 0.204 | 进化先验与 GB1 结合测定存在显著任务差距 |

原始数据：[Spearman](https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv)、[Top-recall](https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/DMS_zero_shot/substitutions/Top_recall/DMS_substitutions_Top_recall_DMS_level.csv)、[NDCG](https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/DMS_zero_shot/substitutions/NDCG/DMS_substitutions_NDCG_DMS_level.csv)。  
<!--ref:proteingym-zero-shot-gb1--><!--anchor:row:SPG1_STRSG_Wu_2016-->

**推断：**这些零样本模型适合在第 0 轮作为 prior 或模型多样性来源，但 Spearman 约 0.35 的直接证据不足以支持其单独担任 GB1 最终评分器。

### 3.3 GB1 上的监督式直接证据

| 模型 | Random Spearman | Contiguous Spearman | Modulo Spearman | Random MSE | Contiguous MSE | Modulo MSE |
|---|---:|---:|---:|---:|---:|---:|
| **Kermut** | 0.781 | **0.778** | **0.778** | 0.722 | **4.601** | **4.601** |
| ProteinNPT | **0.858** | -0.322 | -0.322 | **0.455** | 6.146 | 6.146 |
| ESM-1v embeddings | 0.731 | 0.310 | 0.259 | 0.458 | 6.427 | 6.532 |
| Tranception embeddings | 0.691 | 0.087 | 0.087 | 0.461 | 5.455 | 5.455 |
| One-hot | 0.710 | -0.222 | -0.222 | 0.710 | 5.634 | 5.634 |

原始数据：[random Spearman](https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/DMS_supervised/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level_fold_random_5.csv)、[contiguous Spearman](https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/DMS_supervised/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level_fold_contiguous.csv)、[modulo Spearman](https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/DMS_supervised/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level_fold_modulo_5.csv)、[random MSE](https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/DMS_supervised/substitutions/MSE/DMS_substitutions_MSE_DMS_level_fold_random_5.csv)、[contiguous MSE](https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/DMS_supervised/substitutions/MSE/DMS_substitutions_MSE_DMS_level_fold_contiguous.csv)、[modulo MSE](https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/DMS_supervised/substitutions/MSE/DMS_substitutions_MSE_DMS_level_fold_modulo_5.csv)。  
<!--ref:proteingym-supervised-gb1-splits--><!--anchor:row:SPG1_STRSG_Wu_2016-->

解释：

- random 切分测的是邻近组合的插值；ProteinNPT 在这种条件下最好。
- contiguous/modulo 更容易暴露未见位置/突变模式上的失效；Kermut 显著更稳。
- Kermut 在 random MSE 上不是最好，说明“尾部排序与外推更稳”和“平均数值误差最小”不是同一目标。
- 三种切分上 Kermut 的 GB1 平均 Spearman 约 **0.779**，明显高于零样本模型的直接结果。

### 3.4 低样本与多点突变的补充证据

FSFP 使用 20、40、60、80、100 个单点标签，通过元学习、LoRA 和 listwise 排序微调 PLM；论文在 11 个单点/多点任务上显示少样本排名优势，且强调对未见位点和多突变体的外推。[FSFP 原文](https://www.nature.com/articles/s41467-024-49798-6)  
<!--ref:fsfp2024--><!--anchor:section:Results-->

因此建议将 **FSFP-SaProt 作为资源充足时的第二监督模型**，而不是替换 Kermut：它补充非线性排序能力，Kermut则补充小样本稳定性和可用不确定性。只有在相同 GB1 split、相同标签预算下复现以后，才决定是否让 FSFP 升级为主模型。

SP-Kermut 在 2025 ICML workshop 报告中通过 projected covariance stabilization 改善了 Kermut 的 OOD、MSE、不确定性和高 fitness recovery，但目前证据等级低于正式主会和官方 ProteinGym 结果，适合作为实验分支，不宜直接替换生产基线。[SP-Kermut workshop 论文](https://openreview.net/pdf?id=tC4PVCM7oI)  
<!--ref:sp-kermut-2025--><!--anchor:section:Abstract-->

---

## 4. GB1 模型选择：推荐的两阶段评分架构

### 4.1 第一阶段：Kermut 主 fitness 代理

采用 Kermut 的原因不是“它在所有蛋白上永远最好”，而是：

- 有同一个 GB1 assay 的直接、官方、跨切分证据；
- 复合核同时编码序列相似性、局部结构环境、位点距离和突变组合，契合 GB1 的强上位性；
- GP 后验可用于 UCB、Thompson sampling 和批次多样性采集；
- 初始 96 个左右标签时，精确 GP 的规模完全可控。

实施时应保留当前 `onehot_heterogeneous_ensemble` 作为强任务特异基线。由于 GB1 只有四个位点，显式 pairwise 特征有可能在插值条件下非常有竞争力；如果 Kermut 不能在低到高外推和 tail 指标上稳定胜出，就不应因为模型更新而替换它。

Kermut 的样本数瓶颈也必须预先处理：精确 GP 训练近似为 `O(N^3)`。当累计标签达到数千时，应切换稀疏 GP/诱导点近似，或把旧样本做覆盖性压缩，而不是无限累积到精确核矩阵。

### 4.2 第二阶段：排序头与模型融合

建议同时训练一个 Bradley–Terry 或 ListMLE 排序头，输入可使用：

- Kermut 的组成特征或 ESM/SaProt embedding；
- 训练集中两两偏序，或按小批列表构造 listwise 目标；
- 对高 fitness 样本加权，但权重和阈值只能由训练/验证集决定。

融合不建议直接平均未校准分数。可选方法：

1. 在验证集上把各模型做 rank-normalization，再进行加权 Borda/rank fusion；
2. 训练一个只使用验证外预测的 stacking 层；
3. 保守条件下采用 Pareto/共识选择：Kermut 排名高且排序头不否决。

推荐默认 acquisition：

`score(x) = calibrated_mean_rank(x) + β_t × uncertainty_rank(x) + λ × diversity(x)`

其中 `β_t` 随轮次下降；结构/PPI 辅助分数不直接进入，除非已在 GB1 独立验证集上完成校准。

### 4.3 模型独立性的操作合同

要把 fitness 模型作为 Agent 的独立评估器，至少满足：

1. 每轮开始前冻结模型版本、训练数据哈希、随机种子和超参数；
2. 模型只能读取初始标签和历轮已测标签；完整 landscape 的序列列表也不提供给生成器；
3. 验证集只用于模型/采集策略选择，最终测试集永久密封到一次性评估；
4. Agent 不得用测试分数决定提示词、生成温度、采集参数或提前停止；
5. 生成器与评分器若共用同一 PLM，应增加不同模型家族的消融，以避免“自我偏好”被误认为真实 fitness；
6. 离线基准中由单独 oracle service 只返回本轮查询标签；生产中由盲化实验返回结果；
7. 报告每个候选的来源、父序列、编辑距离、模型均值、UQ、辅助分数和最终选择理由。

---

## 5. 是否需要额外的结合置信度模型

### 5.1 需要，但只做辅助

GB1 DMS fitness 是实验富集结果，可能同时混合折叠、表达、展示效率和 Fc 结合。一个专门的 PPI 模型可以补充“界面结合能”这一因果维度，但不能把不同实验机制压缩成更客观的单一真值。

推荐三层使用：

| 层级 | 输出 | 决策作用 | 禁止的解释 |
|---|---|---|---|
| Kermut | assay-specific fitness 均值与不确定性 | 所有生成候选的主排序 | 不能称为真实 Kd |
| Pythia-PPI | 突变 PPI ΔΔG/亲和变化方向 | 对排名前 `5–10 × B` 的候选重排序、共识或否决 | 不能未经校准替代 GB1 fitness |
| AF3 | ipTM、interface PAE、界面构象 | 排除低可信复合物或识别构象风险 | ipTM 不能解释为结合强度或成功概率 |

Pythia-PPI 论文在 SKEMPI 2.0 上报告 Pearson 从基础模型约 0.645 提升到约 0.785，并通过多任务稳定性与自蒸馏改善跨复合物预测；这是相邻任务上的强证据，不是 GB1 直接验证。[Pythia-PPI 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12199698/)  
<!--ref:pythia-ppi-2025--><!--anchor:section:Results-->

### 5.2 上线前的校准检查

在 GB1 验证集上分别计算：

- Pythia-PPI 分数与 DMS fitness 的 Spearman/Pearson；
- 对 `fitness > WT` 和 top 1% 的 AUROC/AUPRC；
- 当 Pythia 与 Kermut 一致/冲突时各自的 tail precision；
- AF3 ipTM/interface PAE 与绝对预测误差、高 fitness 命中的关系。

只有当辅助模型在密封验证集上带来稳定的 `NDCG@B`、`Precision@B` 或 regret 改善，才允许进入正式重排序。否则保留为解释性证据列。

Boltz-2 文档中的 `affinity_pred_value` 是围绕小分子配体和 `log10(IC50)` 的输出，不应直接用于 GB1-Fc 蛋白–蛋白亲和力；Boltz 的复合物结构置信度仍可作为 AF3 的开源替代。[Boltz 官方仓库](https://github.com/jwohlwend/boltz)  
<!--ref:boltz--><!--anchor:section:Prediction-->

---

## 6. 评估策略：从平均拟合转向高 fitness 发现

### 6.1 数据切分必须回答不同问题

至少并行保留以下离线协议：

| 切分 | 训练/测试构造 | 回答的问题 | 优先级 |
|---|---|---|---|
| Random | 随机分层，控制 Hamming depth 和 fitness 分布 | 邻近组合插值是否正常 | 仅 sanity check |
| 1-vs-rest / 2-vs-rest / 3-vs-rest | 用指定突变深度训练，测试更远组合 | 多点组合外推 | 高 |
| Low-vs-high | 训练只含 `fitness ≤ WT`，测试 `fitness > WT` | 能否从普通样本发现更优变体 | **最高** |
| Leave-substitution-out | 某位点的某些氨基酸身份只出现在测试 | 未见替换身份泛化 | 高 |
| Minimum-distance split | 测试序列到任何训练序列的 Hamming 距离不小于 d | 控制邻域泄漏 | 高 |
| Sequential closed loop | 固定初始集、轮次、每轮预算；只揭示已查询标签 | 真实优化效率 | **最终主协议** |

FLIP 对 GB1 的统计显示，超过 96% 的变体 fitness 低于 0.5，而 WT 被归一化为 1；这意味着普通随机相关性会被大量低 fitness 样本主导。[FLIP GB1 说明](https://flip.protein.properties/assets/FLIP_2021_manuscript.pdf)  
<!--ref:flip2021--><!--anchor:page:4-->

当前项目的随机分层 `initial=96 / validation=96 / final_test=2048` 可以保留，但不足以支持“能发现高 fitness”这一主张。最终测试需要加入 low-vs-high、距离外推和闭环轨道。

### 6.2 指标分层

设测试真值为 `y`，预测为 `ŷ`；`H_q` 是真值最高 `q` 比例的集合；`S_B` 是模型按分数选择的前 `B` 个候选。

#### A. 全局排序指标

- **Spearman ρ：主全局指标。**`corr(rank(y), rank(ŷ))`，对单调变换稳健，适合 DMS 排名。
- **Kendall τ-b：辅助指标。**GB1 有大量相同或接近零分数，τ-b 对 ties 的含义更清楚。
- **Pearson r：校准后报告。**用于判断线性数值关系；预测若只保证排序，Pearson 可能不公平。

#### B. 连续值误差指标

- **RMSE/MSE 与 MAE**：只在同一 assay、同一标签变换内比较。
- 同时报告对训练集拟合的线性或 isotonic calibration 后误差，避免“尺度错误”掩盖排序能力。
- 如果 fitness 长尾严重，附加 log/分位数变换后的误差，但原尺度指标也必须保留。

#### C. 高 fitness 尾部指标——项目主终点

- `Precision@B(q) = |S_B ∩ H_q| / B`
- `Recall@B(q) = |S_B ∩ H_q| / |H_q|`
- `EF@B(q) = Precision@B(q) / q`，随机选择的期望为 1
- `Success@B(q) = 1[|S_B ∩ H_q| > 0]`
- `NDCG@B`：考虑前 B 个候选内部的顺序和 fitness 增益
- `BestFitness@B = max_{x∈S_B} y(x)`
- `SimpleRegret@B = max_{x∈test} y(x) - BestFitness@B`，越低越好
- `HitAboveWT@B`：前 B 个候选中高于 WT 的数量与比例
- `Diversity@B`：两两 Hamming 距离、突变位点覆盖和序列簇覆盖，防止选出一个局部簇的近重复序列

默认报告 `q ∈ {0.1%, 1%, 5%}`；对完整 GB1 landscape，约对应 149、1,494、7,468 个真值高分变体。实验预算建议至少报告 `B ∈ {16, 48, 96}`，并把实际单轮预算设为预注册主 `B`。

#### D. 分类与不平衡指标

对 `fitness > WT`、top 1% 等阈值报告：

- **AUPRC 为主**，并明确随机基线等于阳性率；
- AUROC 为辅助，因为极端不平衡时 AUROC 可能显得乐观；
- Matthews correlation coefficient 用于需要固定二值决策阈值的场景。

#### E. 不确定性与风险指标

- 50%、80%、90% prediction interval 的 empirical coverage 和平均宽度；
- Gaussian NLL 或 CRPS；
- `corr(预测标准差, |误差|)`；
- risk–coverage 曲线/AURC：只接受高置信预测时错误如何下降；
- “进入 top q% 的概率”的 reliability diagram、Brier score 和 expected calibration error。

Kermut 原文表明整体不确定性校准有意义，但实例级校准仍困难，因此不能只看一个 90% coverage 数字。[Kermut 校准分析](https://papers.nips.cc/paper_files/paper/2024/file/34547650b2ca69d91f3b3c3ae8b21962-Paper-Conference.pdf)  
<!--ref:kermut2024--><!--anchor:section:Uncertainty calibration-->

### 6.3 闭环评估

建议固定：

- 10 个独立初始随机种子；
- 初始标签数 96，与当前配置一致；另做 24/48 的低样本敏感性分析；
- 每轮预算 16 或 32，共 5–10 轮；
- 所有方法使用完全相同的初始集、总查询预算和 oracle；
- 每轮报告 cumulative best、cumulative top-q hits、regret、覆盖的序列簇和实验失败率；
- 无重复查询；无效/不可测序列计入预算，避免算法通过提出不可实验候选获得虚假优势。

必须比较：

1. 随机选择；
2. 只按 Hamming 邻域的贪心/遗传搜索；
3. 当前 one-hot + pairwise ensemble；
4. Kermut mean-only；
5. Kermut + UCB/Thompson；
6. Kermut + 排序头；
7. 加/不加 Pythia-PPI 与 AF3 的消融。

### 6.4 置信区间与显著性

- 所有方法在相同 seed/test candidates 上做**配对比较**；
- 至少 2,000 次 bootstrap，报告 95% CI；
- 对距离/位点切分按 Hamming shell 或留出块做 block bootstrap，避免把强相关近邻当独立样本；
- 同时报告均值、中位数和最差四分位 seed，避免少数幸运轮次主导结果；
- 预注册一个主终点，例如 `top 1% Precision@96`，其余指标标为次要，控制多重比较。

### 6.5 建议验收门槛

不建议脱离标签预算设一个通用绝对 Spearman 门槛。更稳妥的项目 gate 是：

1. 在 low-vs-high 和至少一种距离外推切分上，Kermut/新方案的主 tail 指标显著优于当前 one-hot + pairwise 基线；
2. `EF@B(top 1%) > 2`，且配对 bootstrap 95% CI 下界高于随机基线 1；
3. 相同预算下 simple regret 和 cumulative best 均有稳定改善，而不只是 MSE 下降；
4. 90% 区间实际覆盖建议落在 85%–95%，并且区间宽度优于朴素分位数基线；
5. 新模型不得以 random split 的高分掩盖 contiguous/low-vs-high 上的失败。

可把下列已发表 GB1 数字当作**参考线而非硬阈值**：

| 切分 | CNN + MSE：Spearman / Top-10 recall | CNN + Bradley–Terry：Spearman / Top-10 recall |
|---|---:|---:|
| 1-vs-rest | 0.133 / 0.097 | 0.091 / 0.138 |
| 2-vs-rest | 0.564 / 0.250 | **0.607 / 0.282** |
| 3-vs-rest | 0.814 / 0.539 | **0.880 / 0.664** |
| low-vs-high | 0.499 / 0.381 | **0.567 / 0.443** |
| sampled | 0.930 / **0.823** | **0.951** / 0.816 |

这些数字说明排序损失主要改善外推和高分尾部，但并非在每种插值指标上都绝对领先。[原始表格](https://proceedings.neurips.cc/paper_files/paper/2024/file/a9b938e79504889f905d549f8d53e405-Paper-Conference.pdf)  
<!--ref:contrastive-ranking-2024--><!--anchor:page:7-->

---

## 7. 推荐的实施顺序

### 阶段 A：先建立可信基准

1. 保留现有 `onehot_heterogeneous_ensemble`，冻结为 baseline v1。
2. 扩展 `evaluation/metrics.py`：加入 Kendall τ-b、MAE、Precision/Recall/EF/Success/NDCG@B、simple regret、AUPRC、CRPS/AURC。
3. 在数据层新增 low-vs-high、1/2/3-vs-rest、leave-substitution-out、minimum-distance split。
4. 加入 oracle 权限测试：Agent 和 scorer 都不能列出完整 landscape 或读取 hidden labels。

### 阶段 B：接入 Kermut

1. 缓存 ESM-2 与结构表征，避免每轮重复计算；
2. 用当前 96 个初始标签拟合精确 GP，记录均值、方差和校准参数；
3. 与当前 ensemble 在完全相同 split/seed 下对比；
4. 先通过离线 gate，再进入闭环采集。

### 阶段 C：强化高 fitness 排序

1. 增加 Bradley–Terry/ListMLE 排序头；
2. 比较 mean-only、UCB、Thompson、rank fusion；
3. 如果资源允许，再复现 FSFP-SaProt 作为第二监督模型；
4. 对前 `5–10 × B` 候选离线计算 Pythia-PPI 和 AF3，验证其增量价值。

### 阶段 D：从候选池转向开放生成

1. 生成器只接收 WT、允许的编辑操作和实验约束，不接收候选表；
2. 候选经过去重、有效性与多样性检查后交给 scorer；
3. `GB1-4` 使用四位点 landscape 作为完全密封 oracle；
4. `GB1-full` 必须连接新的全序列实验/数据任务，不能复用四位点标签伪造真值；
5. 在至少一个全序列或更大突变域数据集上外部验证后，才宣称具备“全蛋白质空间候选检索”能力。

---

## 8. 风险、边界与尚待验证的假设

1. **Assay shift**：PPI ΔΔG、自然序列似然、折叠稳定性与 DMS 富集不是同一标签。
2. **完整 landscape 的反向泄漏**：即使 hidden labels 不可见，如果生成器能枚举所有序列、观察频次或用测试结果调参，仍然不是严格盲测。
3. **GB1 的规模偏小**：四位点任务特别适合显式上位性模型，不能据此证明对整条蛋白通用。
4. **高 fitness 极少且 ties 多**：单个相关系数会掩盖是否真正命中尾部。
5. **结构模型置信度误用**：高 ipTM 只说明模型相信该复合物构象，不说明结合更强。
6. **不确定性并非天然可靠**：GP 后验受核、噪声和数据分布假设影响，必须做 OOD 校准。
7. **新模型证据不齐**：2025–2026 的一些方法尚未进入相同版本 ProteinGym 官方逐 assay 表，不能与 Kermut 做严格同条件结论。

---

## 9. 最终决策

对当前项目，建议把模型方向定为：

> **Kermut 作为 GB1 assay-specific 主 fitness 代理；Bradley–Terry/ListMLE 或 FSFP-SaProt 作为高 fitness 排序补充；Pythia-PPI 与 AF3 仅对前部候选做结合能和结构可信度复核；最终真值始终来自盲化 IgG-Fc 实验或离线时完全密封的 landscape。**

模型是否成功，不能由 random split 的 Spearman 或 MSE 单独决定。主判据应是，在 low-vs-high、距离外推和固定预算闭环中，是否以统计显著的方式提高 top 0.1%/1% 变体的命中、富集和最佳发现值，并降低 simple regret。

---

## 参考文献与可复核来源

1. Wu NC, et al. Adaptation in protein fitness landscapes is facilitated by indirect paths. *eLife* (2016). [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4985287/)
2. Dallago C, et al. FLIP: Benchmark tasks in fitness landscape inference for proteins. (2021). [论文 PDF](https://flip.protein.properties/assets/FLIP_2021_manuscript.pdf)
3. Notin P, et al. ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design. *NeurIPS Datasets and Benchmarks* (2023). [论文](https://papers.nips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee-Paper-Datasets_and_Benchmarks.pdf)、[官方仓库](https://github.com/OATML-Markslab/ProteinGym)
4. Groth PM, et al. Kermut: Composite kernel regression for protein variant effects. *NeurIPS* (2024). [论文](https://papers.nips.cc/paper_files/paper/2024/file/34547650b2ca69d91f3b3c3ae8b21962-Paper-Conference.pdf), DOI: 10.52202/079017-0929
5. Zhou Z, et al. Enhancing efficiency of protein language models with minimal wet-lab data through few-shot learning. *Nature Communications* (2024). [论文](https://www.nature.com/articles/s41467-024-49798-6), DOI: 10.1038/s41467-024-49798-6
6. Notin P, et al. ProteinNPT: Improving protein property prediction and design with non-parametric transformers. *NeurIPS* (2023). [论文](https://papers.nips.cc/paper_files/paper/2023/hash/6a4d5d85f7a52f062d23d98d544a5578-Abstract-Conference.html)
7. Biswas S, et al. Low-N protein engineering with data-efficient deep learning. *Nature Methods* (2021). [论文](https://www.nature.com/articles/s41592-021-01100-y)
8. Jiang K, et al. Rapid in silico directed evolution by a protein language model with EVOLVEpro. *Science* (2025). [论文](https://doi.org/10.1126/science.adr6006)
9. Brookes DH, Otwinowski J, Sinai S. Contrastive losses as generalized models of global epistasis. *NeurIPS* (2024). [论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a9b938e79504889f905d549f8d53e405-Abstract-Conference.html), DOI: 10.52202/079017-2962
10. Wang Y, et al. ProSST: Protein language modeling with quantized structure and disentangled attention. *NeurIPS* (2024). [论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3ed57b293db0aab7cc30c44f45262348-Abstract-Conference.html)
11. Retrieval-Enhanced Mutation Mastery / VenusREM. *Bioinformatics/ISMB-ECCB* (2025). [PubMed](https://pubmed.ncbi.nlm.nih.gov/40662802/)、[代码](https://github.com/ai4protein/VenusREM)
12. Tao F, et al. Reliable prediction of protein–protein binding affinity changes upon mutations with Pythia-PPI. *National Science Review* (2025). [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12199698/), DOI: 10.1093/nsr/nwaf231
13. Abramson J, et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* (2024). [论文](https://www.nature.com/articles/s41586-024-07487-w)
14. Groth PM, et al. Stabilizing protein fitness predictors via projected covariance stabilization. ICML Workshop (2025). [OpenReview PDF](https://openreview.net/pdf?id=tC4PVCM7oI)

## 研究透明度声明

本报告由 AI 辅助执行检索、官方 CSV 读取、证据分层和写作；关键性能数字均链接到论文原文或 ProteinGym 官方逐 assay 文件。模型选择与实施建议属于基于这些证据和当前代码架构的工程推断，应在统一标签预算、统一切分和盲化测试下复现后再用于实验决策。
