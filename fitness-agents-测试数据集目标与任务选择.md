# fitness-agents：测试数据集的优化目标与 1–2 个具体任务的选择策略

> 版本：v1.0（2026-08-15）
> 关联文档：[fitness-agents-虚拟蛋白质定向进化智能体-项目搭建策略.md](./fitness-agents-虚拟蛋白质定向进化智能体-项目搭建策略.md)（本文对应其 §5 数据策略的展开）
> 结论速览：**主任务选 FLIP GB1（IgG-Fc 结合，完整四位点组合景观）；第二任务选 avGFP（荧光亮度，多点突变外推）。** ProteinGym 与 FLIP-2 不作 MVP 闭环 oracle，分别承担"跨蛋白外部泛化"与"真实工程 split 消融"角色。

## 1. 统一定义：一个"样本"的测试目标到底是什么

在所有这些数据集里，**一条样本 =（某蛋白的一条突变体序列 + 该突变体在特定 assay 条件下实测的 fitness 分数）**。样本的测试目标不是"预测蛋白结构"，也不是"预测功能分类标签"，而是一个三层的序列→标量回归/排序问题：

```text
输入：variant_sequence（或相对 WT 的 mutation set，如 D40N+V54E）
输出：该变体在某 assay 下的 normalized fitness（统一为 higher-is-better）
评测：
  ① 排序质量 —— Spearman ρ、NDCG@k（能否把好的排在前面）
  ② 头部发现 —— Top-k recall / EF@k（能否找回真实 top 1% / 5%）
  ③ 闭环效率 —— 在隐藏标签 oracle 下，固定预算内 best_seen_fitness 逐轮提升多少
```

"目标优化的蛋白质工程性质"完全由该样本所属的 **assay** 决定：同一个蛋白在不同 assay 下（如 GB1 的结合 vs 稳定性）是不同的优化目标，不可混报。这正是策略文档 §10.1 要求把 Observation 绑定到 Assay/Condition 的原因。

## 2. 各数据集的测试目标与优化性质

### 2.1 总览表

| 数据集 | 蛋白 | 样本的测试目标（assay 实测值） | 优化的工程性质 | 规模 | 性质类别 |
|---|---|---|---|---:|---|
| FLIP GB1 | 链球菌 protein G B1 domain | 与 IgG-Fc 结合的筛选富集分数（Olson et al. 2014 / Wu et al. 2016） | **结合亲和力（binding）** | full 149,361；curated 8,733 | Binding |
| FLIP AAV | AAV2 衣壳 VP1（561–588 区段，28 aa） | 变体携带自身 barcode 完成包装/生产的富集度（Ogden et al. 2019） | **病毒衣壳包装活性（capsid viability / production fitness）** | 284,009（82,583 sampled + 201,426 designed） | Activity / 病毒载体产量 |
| FLIP Meltome | 13 个物种的数万种天然蛋白 | 熔解温度 Tm（Meltome atlas） | **热稳定性（thermostability）** | 27,951 | Stability |
| avGFP | 维多利亚水母 GFP（Sarkisyan et al. 2016） | 荧光亮度（log 荧光强度，相对 WT） | **荧光亮度（brightness）** | 约 5.1 万条，1–15 个突变 | 光学性质 / Activity |
| β-lactamase | TEM-1（ProteinGym 中 BLAT_ECOLX 系列） | 氨苄青霉素选择压力下的相对生长/耐药分数（Firnberg 2014、Stiffler 2015 等） | **催化活性（β-内酰胺水解 → 抗生素耐药）** | 各 assay 数千~万余条，以单点为主 | Activity |
| ProteinGym v1.3 | 217 个 substitution DMS（+74 个 indel assays） | 每个 assay 各自的 DMS score | 混合：官方按 **Activity / Binding / Expression / Organismal Fitness / Stability** 五类归组 | 约 270 万 missense + 约 30 万 indels | 混合集合 |
| FLIP-2 Amylase | 枯草杆菌 α-淀粉酶 | 去污活性（stain removal activity） | **工业酶活性** | 3,706，≤8 突变 | Activity |
| FLIP-2 IRED | 亚胺还原酶 | 微流控活性筛选分数 | **酶活性（生物催化）** | 17,143，≤15 突变 | Activity |
| FLIP-2 NucB | 核酸酶 B | pH 7 活性（分 4 档） | **酶活性** | 55,760 | Activity |
| FLIP-2 TrpB | 色氨酸合酶 β 亚基 | 催化产物生成速率 | **酶活性（含强上位性）** | 228,298，10 个组合完备子景观 / 20 位点 | Activity |
| FLIP-2 Hydro | 3 个蛋白的疏水核心（7 残基随机化） | 稳定性测量 | **稳定性（跨 scaffold）** | 24,935 | Stability |
| FLIP-2 Rhomax | 75 条微生物视紫红质 | 最大吸收波长 | **光谱性质（光遗传学）** | 884（含嵌合体） | 光学性质 |
| FLIP-2 PDZ3 | PDZ3 domain–CRIPT 短肽 | 双突变结合亲和力（刻意筛选的上位子集） | **PPI 结合 + 上位性压力测试** | 734 对 | Binding |

### 2.2 关键数据集详解

**GB1（结合）**。样本测试目标：四位点（V39/D40/G41/V54）组合突变体与 IgG-Fc 的结合富集分数。名义空间 20^4 = 160,000，实测 149,361 条全覆盖（WT 1、单点 76、双点 2,091、三点 26,019、四点 121,174）。它是罕见的**近完整组合景观**：候选空间可穷举、上位性强、真值可隐藏，因此天然适合做"虚拟定向进化"的闭环 oracle。FLIP 的 split 语义（1/2/3-vs-rest、low-vs-high、sampled）直接对应"从低阶突变外推到高阶组合"和"从低 fitness 外推到高 fitness"两类真实工程问题。

**AAV（衣壳包装活性）**。样本测试目标：VP1 衣壳 28-aa 区段突变体能否成功包装成携带自身条码的病毒颗粒——即基因治疗载体的**可生产性/包装效率**，而非组织嗜性或转导效率。特点是序列长、突变位点多（最多 39 个突变）、含 indel，且 designed 子集（201,426 条）本身是前一轮工程结果，分布明显偏移。适合做"长序列 + 高阶突变"的难任务，但 CPU 基线成本高。

**avGFP（荧光亮度）**。样本测试目标：突变体相对 WT 的荧光亮度（log 尺度）。约 5 万条变体、突变数 1–15，亮度分布高度集中（多数突变体暗淡），是经典的"窄峰高 fitness 区域 + 大空间"问题，考验模型从低阶/低亮度样本向高阶组合的排序外推。ProteinGym 亦收录（GFP_AEQVI_Sarkisyan_2016），可与 FLIP 之外的榜单结果对照。

**β-lactamase TEM-1（催化活性/耐药）**。样本测试目标：氨苄青霉素梯度下的相对生长分数，本质是 TEM-1 水解 β-内酰胺的催化活性在细胞层面的读数。各 assay 以单点饱和突变为主（数千条量级），**组合上位性信号弱**，不适合做多点组合闭环的主任务；但作为 ProteinGym 中 Activity 类别的代表，非常适合做跨蛋白外部泛化验证。

**Meltome（热稳定性）**。样本测试目标：跨 13 物种天然蛋白的 Tm 回归。这是"全局景观"（不同蛋白之间比），不是定向进化的"局部景观"（同一 WT 附近组合优化），因此只作为稳定性性质的辅助参考，不进闭环。

**ProteinGym（集合，不是单一任务）**。它的"测试目标"必须逐 assay 理解：217 个 substitution assays 各自测量不同蛋白的不同性质，官方按 Activity / Binding / Expression / Organismal Fitness / Stability 五类归组。样本层面的目标统一为"预测该 assay 的 DMS score 并对变体排序"。使用时必须按性质类别挑选与目标任务同类的 assays，否则"跨蛋白泛化"的结论会被性质差异污染。

**FLIP-2（真实工程 split 的集合）**。7 个数据集覆盖酶活（Amylase/IRED/NucB/TrpB）、稳定性（Hydro）、光谱性质（Rhomax）和 PPI 结合（PDZ3）；16 个 split 归入 5 类真实分布偏移：

| split 类型 | 对应的工程决策时刻 | 代表 |
|---|---|---|
| Number（突变数） | 用低阶突变数据预测高阶组合 | single-to-double、one-to-many |
| Position（位点） | 已饱和活性中心，转向未扰动的远端位点 | close-to-far / far-to-close |
| Mutation（突变身份） | 遇到训练中未见过的氨基酸替换 | by-mutation |
| Fitness（低→高） | 用前几轮差变体外推后几轮好变体（定向进化的本质） | low-to-high |
| Wild Type（骨架） | 把一个同源蛋白上学到的效应迁移到目标蛋白 | by-wild-type |

FLIP-2 的核心结论对本项目有硬约束：random split 系统性高估性能；one-hot ridge（尤其加 zero-shot likelihood 特征）是必须击败的强基线；position / fitness / wild-type 三类 split 最难。这直接支持策略文档"默认 one-hot Ridge + ExtraTrees 异质集成、先做强基线"的决策。

## 3. 只选 1–2 个任务时的选择标准

按重要性排序：

1. **真值完整、可隐藏**：候选空间内每条变体都有实验标签，才能构建严格无泄露的虚拟 oracle（选中才揭示、final test 全程封闭）。这一条直接排除绝大多数 ProteinGym assays（多为单点、空间不完备）。
2. **性质单一、量纲清晰**：闭环优化的 reward 必须是单一可比的标量；多 assay 混合会引入方向性和量纲问题。
3. **组合上位性足够强**：单点加性模型会失效，才能体现"模型 + 知识 + Agent"相对穷举/随机的价值。
4. **规模 CPU 友好**：一周 MVP 内可完成下载、特征、3 轮闭环和 ≥20 seeds 消融。
5. **有公开基线可对齐**：ALDE、EVOLVEpro、FLIP、FLIP-2、UQ benchmark 已跑过的任务，结果可横向核验，失败时可定位是实现问题还是方法问题。
6. **结构/知识通道可接入**：有 PDB/AlphaFold 结构、界面注释、MSA，知识增强消融才有意义。

按工程性质补一个考量：1–2 个任务最好**分属不同性质类别**（如 binding + activity/光学），否则"系统有效"的结论只对单一性质成立。

## 4. 推荐方案与测试数据选法

### 4.1 推荐组合：GB1（主）+ avGFP（第二）

| 维度 | GB1（主任务） | avGFP（第二任务） |
|---|---|---|
| 优化性质 | IgG-Fc 结合亲和力 | 荧光亮度 |
| 性质类别 | Binding | 光学/Activity |
| 空间结构 | 20^4 完备，可穷举 | 1–15 突变、稀疏采样、不可穷举 |
| 检验能力 | 闭环 oracle、上位性、严格防泄露 | 低阶→高阶、低亮度→高亮度外推 |
| 公开基线 | ALDE / EVOLVEpro / PLMeAE / ProteinGenerator / UQ benchmark | TAPE / Design-Bench / ProteinGym |
| 结构通道 | 有 PDB、界面明确，ipTM/界面特征可接入 | 有晶体结构，发色团环境明确 |

两者性质不同、景观形态不同（完备 vs 稀疏）、split 语义不同（穷举组合 vs 深度外推），组合起来可以同时回答"系统在可验证的小空间里是否学得对"和"系统能否迁移到更真实的稀疏大空间"。

**备选与放弃理由**：

- **GB1 + AAV**：FLIP 同源、对齐成本低，但 AAV 序列长、含 indel、28 万条规模的 embedding 预计算对一周 MVP 偏重；放入 Phase 2。
- **GB1 + TrpB**：两者都是完备组合景观（TrpB 在 FLIP-2 中有 10 个组合完备子景观），与 ALDE 完全可比；但两个任务都偏"小景观组合优化"，不如 GFP 能补充稀疏大空间外推场景。若工期紧可优先考虑（复用 ALDE 闭环代码路径）。
- **β-lactamase 不作主任务**：单点为主、上位性弱、空间不完备，留给 ProteinGym 外部验证。
- **Meltome 不作任务**：跨物种 Tm 回归不是定向进化闭环问题。
- **ProteinGym / FLIP-2 不作 MVP oracle**：定位见 §4.3。

### 4.2 GB1 测试数据的具体选法

- **数据源**：FLIP `four_mutations_full_data.csv`（149,361 条全量），manifest 中固定 `full` 而非 `keep=True` 的 8,733 条 curated 子集，二者不可混报。
- **分层切分**（对应策略文档 §5.3/§5.5）：`initial_observed`（Round 0 可见）/ `validation`（HPO 与校准）/ `oracle_pool`（选中才揭示）/ `final_test`（全程封闭），采样按 Hamming depth（1/2/3/4 点）分层，保证各层突变深度分布一致。
- **split 语义对齐消融**：除项目自定义闭环 split 外，另跑 FLIP 官方 `sampled`、`low-vs-high`、`1/2/3-vs-rest`，使结果可与 FLIP/FLIP-2 文献数字直接对照。
- **预算**：demo 512 条（3 轮 × 16）；正式 96/轮 × 3 轮，与 ALDE 的 384 总预算对齐。

### 4.3 avGFP 测试数据的具体选法

- **数据源**：Sarkisyan et al. 2016（约 5.1 万条）；注意与 ProteinGym 收录版本（GFP_AEQVI_Sarkisyan_2016）核对行数与归一化方式，两处数字不可混报。
- **split 设计（核心）**：不用 random split。采用两类真实工程 split——
  - **Number split**：训练仅含 WT + 单点 + 双点，测试 ≥3 点突变（对应 FLIP-2 one-to-many）；
  - **Fitness split**：训练取亮度分布的低分位（如最低 60%），测试高分位（对应 low-to-high，即"用差样本找好样本"）。
- **闭环化**：与 GB1 相同地包装成 oracle_pool + final_test；候选生成用枚举不可行，改用位点×氨基酸两阶段采样（策略文档 §9.1）。
- **预算**：与 GB1 相同的轮数和每轮预算，保证两任务的闭环指标（best_seen、regret）可横向比较。

### 4.4 ProteinGym 与 FLIP-2 的正确用法

- **ProteinGym = Phase 2 跨蛋白外部泛化**：按性质类别挑与两主任务同类的 assays（Binding 类如 GRB2/PDZ 相关，Activity 类如 BLAT_ECOLX β-lactamase、PABP 等），评估协议用 contiguous / modulo CV 而非 random split，报告每 assay Spearman 与类别均值。它回答的是"模型换个蛋白还行不行"，不进入闭环。
- **FLIP-2 = split 语义与基线压力测试**：MVP 消融直接借用其 5 类 split 的定义（Number/Position/Mutation/Fitness/Wild Type）在 GB1+GFP 上实现，并把"one-hot ridge、one-hot + zero-shot likelihood ridge"列为必须击败的基线。若未来扩展第三任务，TrpB（组合完备子景观 + 酶活）是首选。
- **禁止**：不要把 ProteinGym 的 random-split Spearman 当作系统的主要性能声明；不要把不同版本（v1.0/v1.2/v1.3）的榜单数字混用。

## 5. 与本项目模块的对应落点

| 本文结论 | 策略文档落点 |
|---|---|
| 样本目标 = assay 绑定的标量 fitness | §5.4 统一 schema 的 `assay_id / directionality / normalized_fitness` |
| GB1 主任务 + 四层 split | §3.1 MVP 科学问题、§5.3 demo、§5.5 防泄露 |
| GFP 第二任务的 number/fitness split | §13.2 消融矩阵的 split 维度；§15.2 Phase 2 |
| one-hot ridge（+likelihood）为必击败基线 | §6.1 Level 0 基线、§13 实验设计 |
| ProteinGym 仅作外部泛化 | §15.2 Phase 2 |
| Oracle/指标不跨 assay 混报 | §7 指标体系（Spearman/NDCG/Top-k/EF + 闭环指标） |
