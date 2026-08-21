## 4. 主要结果、对照实验与消融矩阵

### 4.1 分析对象、完成性门槛与统计口径

本节以六种策略的正式 AL96 三折结果替换原占位信息：`random` 表示随机选择；`fitness_direct` 表示由 Kermut 预测器直接选择；`kg_base` 表示仅使用 Experimental Memory Layer；`kg_base_rag` 在此基础上加入外部数据库 RAG；`kg_base_al` 在基础 KG 上加入主动学习；`kg_3features_rag` 则在双层 KG–RAG 路线上进一步整合理化性质、conservation 和 structure 三个特征通道。每折从 96 个初始可见观测出发，连续执行 3 轮，每轮揭示 16 个候选，因而总查询预算为 48，最终可见观测数为 144。

只有同时满足 `completion_manifest.pass_eligible=true`、完成 3 轮、无 aborted round 且实际批量为 16/16/16 的 run 才进入统计。六种策略的 fold 0–2 均通过该门槛，共纳入 18 个 run；旧组合目录中的三个失败 `kg_3features_rag` run 仅作为被新实验替代的审计记录，不参与均值、标准差、排名或案例筛选。[I17][I20]

| 条件 | fold 0 | fold 1 | fold 2 | 正式纳入 |
|---|---:|---:|---:|---:|
| `random` | 通过 | 通过 | 通过 | 3/3 |
| `fitness_direct` | 通过 | 通过 | 通过 | 3/3 |
| `kg_base` | 通过 | 通过 | 通过 | 3/3 |
| `kg_base_rag` | 通过 | 通过 | 通过 | 3/3 |
| `kg_base_al` | 通过 | 通过 | 通过 | 3/3 |
| `kg_3features_rag` | 通过 | 通过 | 通过 | 3/3 |

三折汇总均报告均值 ± 样本标准差（s.d., `ddof=1`）。发现能力的主指标为最终 `best-seen`、相对初始观测的 `best-seen` 增量、按 48 次查询归一化的 best-seen AULC，以及末轮 batch best/mean/median。Spearman、Pearson、MSE、RMSE、NDCG@10、top-k hit/recall、regret@10、90% 区间 coverage 偏差和 Gaussian NLL 用于评价隔离测试集上的预测排序、误差与校准，不替代 wet-fitness 发现结论。表中粗体表示第一名，下划线表示第二名；对 fitness、相关性、NDCG 和命中指标取越大越好，对 MSE、RMSE、regret、coverage 偏差和 NLL 取越小越好。

六种策略在同一 fold 使用相同 assignment hash 和初始观测，但 `random`/`fitness_direct` 使用 seed 42，KG 路线使用 seed 11，逐轮候选池也未预先固定。即使 `kg_base_rag` 与 `kg_3features_rag` 使用相同 fold 和 seed，9 个 fold–round 候选池的平均 Jaccard 相似度也仅为 0.270，范围为 0.049–0.684。因此，下述均值、标准差和 fold 对齐差值是描述性证据；在 n=3 下不进行显著性检验，也不把条件差异解释为严格隔离的组件因果效应。

### 4.2 三轮 fitness 变化趋势

![六种策略的三轮 fitness 轨迹](../analysis/gb1_al96_report_20260821/outputs/figures/figure2_fitness_trajectories.png)

**图 2｜六种策略的三轮 fitness 轨迹。** a，累计查询预算下的 best-seen；b、c，各轮选中批次的 mean 和 median。粗线和阴影分别表示三折均值和 ±1 s.d.，浅色细线为单折轨迹（n=3 folds）。未进行推断性显著性检验。源数据见 `round_metrics_by_fold.csv` 与 `round_metrics_mean_sd.csv`。

首先，六种策略均从相同的初始 best-seen 4.073 出发。在固定预算下，`random` 与 `fitness_direct` 三轮后仍为 4.073 ± 0.000；与此相比，四条 KG 路线均在第一轮获得增益。`kg_base` 的 best-seen 从 4.682 ± 0.340、4.879 ± 0.340 逐轮升至 5.393 ± 0.550；`kg_base_rag` 在第一轮达到 5.015 ± 0.105、第二轮达到 5.075 ± 0.000 后进入平台；`kg_base_al` 第一轮即达到 5.075 ± 0.000，第三轮小幅升至 5.127 ± 0.089；`kg_3features_rag` 则从第一轮 4.846 ± 0.397 上升到第二轮 5.075 ± 0.000，第三轮未再刷新。由此可见，基础 KG 的终点峰值最高但折间差异较大，RAG、主动学习和三通道路线更早到达约 5.075 的高值区间。

其次，批次分布给出了与单点峰值不同的信息。`kg_base_al` 的 batch mean 由 3.331 ± 0.169 降至 2.164 ± 0.210 和 1.880 ± 0.418，对应 median 为 3.382 ± 0.134、2.538 ± 0.061 和 2.055 ± 0.768；虽然后期下降，其末轮批次主体仍优于其他条件。`kg_base_rag` 的 batch mean 为 2.160 ± 0.670、2.242 ± 0.129 和 1.355 ± 0.499；`kg_3features_rag` 的对应值为 2.450 ± 0.714、2.467 ± 0.188 和 1.356 ± 0.430，两条路线在末轮几乎重合。`kg_base` 则由 2.373 ± 0.243 连续降至 1.667 ± 0.802 和 1.154 ± 0.106。相反，`random` 与 `fitness_direct` 的末轮 batch mean 仅为 0.110 ± 0.057 和 0.118 ± 0.069，median 均接近零。因此，KG 家族的共同优势不仅是命中单个高值，还包括把整个实验批次推向更高的 wet-fitness 区域；与此同时，后期批次质量下降提示高值区域被逐步消耗，探索、对照和低置信候选在后续轮次中的占比可能上升。

### 4.3 跨策略性能比较

**表 1｜闭环 wet-fitness 发现指标。** 三折均值 ± s.d.；粗体为第一名，下划线为第二名。

| 策略 | 最终 best-seen | best-seen 增量 | best-seen AULC | R3 batch best | R3 batch mean | R3 batch median |
|---|---|---|---|---|---|---|
| 方向 | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ |
| `random` | 4.073 ± 0.000 | 0.000 ± 0.000 | 4.073 ± 0.000 | 1.094 ± 0.742 | 0.110 ± 0.057 | 0.004 ± 0.001 |
| `fitness_direct` | 4.073 ± 0.000 | 0.000 ± 0.000 | 4.073 ± 0.000 | 1.094 ± 0.742 | 0.118 ± 0.069 | 0.008 ± 0.002 |
| `kg_base` | **5.393 ± 0.550** | **1.320 ± 0.550** | 4.765 ± 0.126 | **4.526 ± 1.312** | 1.154 ± 0.106 | 0.583 ± 0.345 |
| `kg_base_rag` | 5.075 ± 0.000 | 1.002 ± 0.000 | <u>4.888 ± 0.035</u> | 3.896 ± 0.672 | 1.355 ± 0.499 | 1.151 ± 0.778 |
| `kg_base_al` | <u>5.127 ± 0.089</u> | <u>1.053 ± 0.089</u> | **4.917 ± 0.015** | <u>4.450 ± 0.740</u> | **1.880 ± 0.418** | **2.055 ± 0.768** |
| `kg_3features_rag` | 5.075 ± 0.000 | 1.002 ± 0.000 | 4.832 ± 0.132 | 3.921 ± 0.188 | <u>1.356 ± 0.430</u> | <u>1.186 ± 0.833</u> |

表 1 支持两个层次的结论。其一，四条 KG 路线在最终 best-seen、best-seen 增量、AULC 和末轮批次分布上均高于 `random` 与 `fitness_direct`，因此 KG 家族在本次固定预算闭池筛选中呈现发现优势。其二，各 KG 条件的优势侧重点不同：`kg_base` 在最终峰值、峰值增量和 R3 batch best 上排名第一，`kg_base_al` 则在 AULC、R3 batch mean 和 median 上排名第一；`kg_3features_rag` 的末轮 batch mean/median 排名第二，但终点峰值与 `kg_base_rag` 相同。换言之，“命中最高峰”和“持续提高批次主体”是两个不同目标，不能用单一总分合并。

**表 2｜隔离测试集上的排序指标。** 三折均值 ± s.d.；Top-k 为 k=10。

| 策略 | Spearman | Pearson | NDCG@10 | Top-k hit | Top-k recall | Regret@10 |
|---|---|---|---|---|---|---|
| 方向 | ↑ | ↑ | ↑ | ↑ | ↑ | ↓ |
| `random` | 0.216 ± 0.013 | 0.185 ± 0.015 | 0.674 ± 0.010 | **0.333 ± 0.577** | **0.033 ± 0.058** | <u>3.570 ± 1.781</u> |
| `fitness_direct` | 0.235 ± 0.030 | 0.203 ± 0.032 | 0.690 ± 0.023 | **0.333 ± 0.577** | **0.033 ± 0.058** | <u>3.570 ± 1.781</u> |
| `kg_base` | <u>0.243 ± 0.049</u> | **0.241 ± 0.071** | 0.707 ± 0.035 | <u>0.000 ± 0.000</u> | <u>0.000 ± 0.000</u> | 4.140 ± 1.106 |
| `kg_base_rag` | 0.208 ± 0.028 | 0.224 ± 0.034 | **0.714 ± 0.016** | **0.333 ± 0.577** | **0.033 ± 0.058** | **3.513 ± 1.072** |
| `kg_base_al` | 0.242 ± 0.062 | 0.209 ± 0.053 | 0.699 ± 0.013 | <u>0.000 ± 0.000</u> | <u>0.000 ± 0.000</u> | 4.282 ± 1.062 |
| `kg_3features_rag` | **0.243 ± 0.004** | <u>0.238 ± 0.003</u> | <u>0.711 ± 0.003</u> | <u>0.000 ± 0.000</u> | <u>0.000 ± 0.000</u> | 4.361 ± 1.151 |

**表 3｜隔离测试集上的误差与校准指标。** 三折均值 ± s.d.。

| 策略 | MSE | RMSE | \|Coverage−0.90\| | Gaussian NLL |
|---|---|---|---|---|
| 方向 | ↓ | ↓ | ↓ | ↓ |
| `random` | **0.172 ± 0.007** | **0.414 ± 0.008** | 0.079 ± 0.002 | **−0.250 ± 0.032** |
| `fitness_direct` | <u>0.220 ± 0.053</u> | <u>0.466 ± 0.055</u> | 0.079 ± 0.003 | <u>−0.157 ± 0.058</u> |
| `kg_base` | 0.357 ± 0.148 | 0.589 ± 0.124 | 0.074 ± 0.014 | 0.139 ± 0.079 |
| `kg_base_rag` | 0.487 ± 0.314 | 0.675 ± 0.217 | <u>0.065 ± 0.027</u> | 0.221 ± 0.175 |
| `kg_base_al` | 0.666 ± 0.329 | 0.796 ± 0.220 | **0.058 ± 0.027** | 0.317 ± 0.187 |
| `kg_3features_rag` | 0.530 ± 0.110 | 0.725 ± 0.078 | 0.075 ± 0.008 | 0.224 ± 0.069 |

然而，表 2–3 没有复现 wet-fitness 指标上的统一排序。`kg_3features_rag` 的 Spearman 均值最高且折间离散最小，Pearson 和 NDCG@10 排名第二；`kg_base_rag` 的 NDCG@10 和 regret@10 最优。另一方面，`random` 的 MSE、RMSE 和 NLL 最低，却没有获得任何 best-seen 增量。这一冲突与 GB1 隔离测试集被大量低值样本主导、闭环目标关注高值尾部相符。因而，相关系数和全局误差只能作为 predictor 的辅助诊断，不能替代 `best-seen`、batch 分布与 AULC 所反映的实验发现能力。

### 4.4 KG、外部 RAG、主动学习与三通道特征的增量作用

![RAG、主动学习和三通道特征的 fold 对齐差值](../analysis/gb1_al96_report_20260821/outputs/figures/figure3_module_deltas.png)

**图 3｜三个模块条件的 fold 对齐差值。** `kg_base_rag` 与 `kg_base_al` 以 `kg_base` 为参照；`kg_3features_rag` 以 `kg_base_rag` 为参照。空心圆表示三个 fold，菱形表示均值，水平线表示观测范围，虚线为零差值。候选池未严格配对，因此该图描述方向与折间一致性，不表示配对统计推断。源数据见 `kg_module_fold_deltas.csv`。

| 模块条件 | 参照 | 最终 best-seen Δ | best-seen AULC Δ | R3 batch mean Δ | R3 batch median Δ |
|---|---|---:|---:|---:|---:|
| `kg_base_rag` | `kg_base` | −0.317 ± 0.550（0/3 正向） | +0.123 ± 0.118（2/3 正向） | +0.200 ± 0.457（2/3 正向） | +0.568 ± 0.762（2/3 正向） |
| `kg_base_al` | `kg_base` | −0.266 ± 0.599（1/3 正向） | +0.152 ± 0.111（3/3 正向） | +0.725 ± 0.382（3/3 正向） | +1.472 ± 0.553（3/3 正向） |
| `kg_3features_rag` | `kg_base_rag` | 0.000 ± 0.000（0/3 正向；3/3 持平） | −0.056 ± 0.153（1/3 正向） | +0.001 ± 0.197（2/3 正向） | +0.035 ± 0.157（2/3 正向） |

以 `kg_base` 为参照，外部 RAG 的帮助是混合的。`kg_base_rag` 提高了平均 AULC 和末轮 batch mean/median，但 fold 方向不一致；最终 best-seen 在两个 fold 持平、一个 fold 下降。其 NDCG@10 和 regret@10 改善，Spearman、MSE 和 NLL 则变差。因此，现有证据只支持“外部知识改变了排序和部分批次分布”，不支持“外部 RAG 稳定提高峰值发现”。

主动学习的帮助更集中在批次层面。`kg_base_al` 的 AULC、R3 batch mean 和 R3 batch median 差值均在 3/3 folds 为正，且运行记录确认每轮完成 8 个 exploitation、4 个 exploration 和 4 个 knowledge 名额，posterior 在 96、112 和 128 个可见观测阶段均为 calibrated。相反，最终 best-seen 仅一折提高、一折持平、一折降低。由此可见，主动学习在本实验中提高了高质量候选的密度和早期收益，但没有稳定抬高单个最高峰。

进一步地，三通道功能确实进入了运行，而不是仅存在于配置中。3 folds × 3 rounds × 3 channels 的 27 条路径全部执行 feature tool、启动 Sub-Scientist 请求并获得子 Critic `APPROVED`；各通道每个 fold–round 覆盖 9–10 个样本。通道输出共形成 205 条有界 finding，但没有生成任何 child candidate hypothesis。

| 三通道 | 执行并通过 | 覆盖样本总数（9 个 fold–round） | 有界 findings | child candidate hypotheses |
|---|---:|---:|---:|---:|
| Physchem | 9/9 | 87 | 68 | 0 |
| Conservation | 9/9 | 87 | 69 | 0 |
| Structure | 9/9 | 87 | 68 | 0 |

尽管如此，`kg_3features_rag` 相对 `kg_base_rag` 的最终 best-seen 在 3/3 folds 完全相同，末轮 batch mean 与 median 的平均增量接近零，AULC 反而降低 0.056 ± 0.153。三通道提高了最终 predictor 的 Spearman 稳定性，却没有同步改善高峰发现、regret 或全局误差。更重要的是，`kg_3features_rag` 的 144 个已选候选均只记录一条直接 selection evidence，类型均为 `kg/measured_aggregate`；physchem、conservation 和 structure 输出没有成为候选级 selection evidence。综合执行审计与性能结果，当前数据只能说明系统能够读取并形成有界的三通道观察，不能说明这些观察已被转化为更准确的定向进化方向或更优的候选选择。

### 4.5 Prompt—证据—KG 子图案例分析

案例从四种 KG 条件的 576 个已选候选中按预先声明的确定性规则筛选。全局正例取 wet fitness 最高者；全局反例限定在条件内 wet fitness 后 25% 且 acquisition 前 25%，再取 surprise score 最高者；为直接检查三通道，还在 `kg_3features_rag` 内分别保留同规则正例和反例。完整 Prompt、最终结构化输出、selection evidence、KG 子图与全量 shortlist 见 `outputs/case_studies/`；以下只展示模型可见内容，不导出隐藏推理。

| 案例 | 条件 / fold / round | 变体 | 预测 fitness | 实测 fitness | acquisition / knowledge |
|---|---|---|---:|---:|---:|
| 全局正例 | `kg_base` / 0 / 3 | `LWAA`（V39L;D40W;G41A;V54A） | 1.166 ± 0.722 | **6.027** | 2.163 / 0.552 |
| 全局反例 | `kg_base_rag` / 2 / 3 | `LWTC`（V39L;D40W;G41T;V54C） | 2.527 ± 0.648 | **0.015** | 2.514 / 0.799 |
| 三通道正例 | `kg_3features_rag` / 0 / 1 | `LYGV`（V39L;D40Y） | 4.173 ± 0.486 | **5.075** | 1.437 / 0.672 |
| 三通道反例 | `kg_3features_rag` / 0 / 3 | `VYGY`（D40Y;V54Y） | 2.077 ± 0.686 | **0.217** | 3.359 / 0.697 |

在全局正例中，Scientist 从可见实验记忆与 KG 聚合节点得到软方向：“position 39 favors hydrophobic I or L; position 40 favors aromatic/hydrophobic Y, W, F, or H; position 41 favors wild-type G; position 54 favors C or A”。对应 KG 节点显示 Y40、W40、F40 和 L39 的 visible mean fitness 均高于全局可见均值，同时明确附带“association only; complete-variant epistasis may confound residue effects”的 caveat。`LWAA` 的 G41A 不符合 G41 软偏好，Kermut 也将其明显低估，但软约束没有把该候选排除，最终 wet fitness 达到 6.027。该案例表明基础 KG 的优势来自“可追溯的方向先验 + 保留组合探索”，而不是把残基统计写成硬规则；同时，G41A 的成功也证明单点聚合不能覆盖完整上位性。

相反，全局反例 `LWTC` 同时命中 L39、W40 和 C54 的正向聚合信号，selection evidence score 为 0.799、support 为 34，因而获得较高 acquisition；但其 G41T 偏离了 Scientist 对 G41 的软偏好，最终 wet fitness 仅 0.015。对应 KG 子图在结果揭示后写入 Variant、Evidence、MutationEffectEstimate 和 ReThinkReflection，并记录 2.527 → 0.015 的干湿偏差。该案例说明残基级正向证据可能掩盖组合背景中的致损突变；由于直接 selection evidence 不包含 local-RAG 记录，该失败不能简单归因于外部数据库，更准确的瓶颈是外部知识尚未被转化为候选特异、可校准的上位性约束。

三通道正例 `LYGV` 同时符合 L39、Y40、G41 和 V54 的主方向，wet fitness 为 5.075。其 Sub-Scientist Prompt 确实包含理化描述符、MSA 单点 log-odds 和 1PGB 静态结构环境；然而 conservation 输出同时指出 `Neff/L=0.270`、pairwise analysis 关闭，structure 输出指出 mutant side chains 未建模。因此，这个正例与三通道方向一致，却不能证明三通道是候选成功的必要原因，因为相同 `LYGV` 也出现在其他 KG 条件的高值结果中，直接 selection evidence 仍是实验记忆聚合节点。

三通道反例 `VYGY` 更直接暴露了接口断点。Scientist 输出已经写明 V54 偏好 C/V/A，并要求保护 buried V54 pocket；structure 子通道也记录 position 54 在静态模型中 relative SASA 为 0、contact count 为 15。尽管如此，包含 V54Y 的候选仍以 3.359 的 acquisition 被选中，wet fitness 仅 0.217，远低于 2.077 的预测值。该轮 feature 分析每个通道只覆盖 9–10 个样本，且没有形成 candidate hypothesis 或候选级 selection evidence；ReThink 只能在测量后识别“V54Y 非偏好且可能造成负面组合效应”。因此，问题不是三通道没有运行，而是其有界观察没有稳定绑定到全部候选，也没有进入控制 acquisition 的决策接口。

四个案例共同解释了图表结果：KG 的可追溯软先验能够富集高值区域并保留上位性发现机会；同一软性设计也会让局部正向信号掩盖不利组合。外部 RAG 和三通道增加了上下文，但只要它们没有被压缩为候选特异、可校准且可参与选择的证据，就难以转化为稳定的 wet-fitness 增益。

### 4.6 主要发现、对照矩阵与证据边界

上述结果按“运行有效性 → fitness 轨迹 → 跨策略排名 → 模块差值 → Prompt/KG 个案”的顺序形成一条闭合论证链。首先，18 个正式 run 证明比较对象均完成同等预算；在此基础上，图 2 说明 KG 家族相对两个基线具有高值发现和批次富集优势。随后，表 1–3 将这种优势限定在 wet-fitness 目标，而没有扩展为所有 predictor 指标上的领先。进一步的图 3 把组件作用分开：主动学习的批次收益最一致，外部 RAG 的收益混合，三通道没有形成增量发现优势。最后，Prompt/KG 案例把总体差异连接到“软残基先验、上位性缺失、特征覆盖与 selection-evidence 接口”三个可检查机制，同时保留非因果边界。

| 实验目的 | 参照条件 | 实验条件 | 本次状态 | 可支持的结论 |
|---|---|---|---|---|
| 随机下界 | `random` | 四条 KG 路线 | 3 折完成 | 支持描述性 wet-fitness 发现优势；seed/候选池未严格配对 |
| 预测模型基线 | `fitness_direct` | 四条 KG 路线 | 3 折完成 | 支持区分实验发现指标与 surrogate predictor 指标 |
| 外部知识增量 | `kg_base` | `kg_base_rag` | 3 折完成 | 部分批次与排序指标改善，峰值发现无稳定增益 |
| 主动学习增量 | `kg_base` | `kg_base_al` | 3 折完成 | AULC 与末轮批次质量在 3/3 folds 改善；峰值不一致 |
| 三通道特征增量 | `kg_base_rag` | `kg_3features_rag` | 3 折完成；27/27 通道路径通过 | 证明通道执行与有界观察生成；不支持增量 wet-fitness 优势 |
| 分层 Agent | 单 Scientist | Scientist、Critic、Sub-Agent | 未提供同折简化对照 | 不判定层级结构的独立贡献 |
| 反馈闭环 | 关闭 ReThink | 开启 ReThink | 未提供同折对照 | 证明反思可写回，不判定独立 fitness 增益 |
| 不确定性 | No-UQ | Agent-UQ | 未提供 No-UQ 对照 | 不判定 UQ 的独立贡献 |

完整分析由 `analysis/gb1_al96_report_20260821/` 中的模块化 Python 包执行。源数据目录逐一保存折级指标、mean ± s.d.、候选池重叠、模块差值、主动学习执行审计、三通道执行审计和全部 selected candidates；案例目录保存确定性筛选记录、Prompt 摘录、证据记录与 KG 子图；图件目录保存可编辑 SVG、PDF、600-dpi TIFF 和 PNG。固定复现命令为：

```powershell
uv run --no-project --with numpy --with pandas --with matplotlib --with pillow python analysis/gb1_al96_report_20260821/run_analysis.py
```

鉴于不同策略的逐轮候选池没有严格固定，本报告不使用“显著提高”“证明组件因果效应”等表述。下一阶段应预先固定各 fold–round 的候选池与 seed，至少增加重复 seed，并为 RAG、三通道和 UQ 分别设置同折开关对照；只有这样才能把当前的条件差异推进为组件级因果结论。

