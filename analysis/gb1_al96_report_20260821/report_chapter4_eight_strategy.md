## 4. 主要结果、对照实验与消融矩阵

### 4.1 分析对象、完成性门槛与统计口径

本节以八种策略的正式 AL96 三折结果更新比较矩阵。三种 baseline 分别为随机选择的 `random`、由 Kermut 预测器直接选择的 `fitness_direct`，以及仅由 LLM 根据当前可见观测形成方向先验、但关闭知识模块与 KG 交互的 `agent_only`。五种知识增强策略包括：仅使用 Experimental Memory Layer 的 `kg_base`；在基础 KG 上整合理化性质、conservation 和 structure 三个通道但不使用外部 RAG 的 `kg_3features_base`；加入外部数据库 RAG 的 `kg_base_rag`；同时使用三通道与 RAG 的 `kg_3features_rag`；以及在基础 KG 上加入主动学习的 `kg_base_al`。从实际配置看，两个 `kg_3features_*` 条件同时启用了分层假设整合，因此它们检验的是“三通道分层整合包”，而不是任一单独 feature channel 的孤立贡献。

每折从 96 个初始可见观测出发，连续执行 3 轮，每轮揭示 16 个候选，总查询预算为 48，最终可见观测数为 144。只有同时满足 `completion_manifest.pass_eligible=true`、完成 3 轮、无 aborted round 且实际批量为 16/16/16 的 run 才进入统计。八种策略的 fold 0–2 均通过该门槛，共纳入 24 个 run；旧组合目录中的三个失败 `kg_3features_rag` run 仅保留为被新实验替代的审计记录，不参与均值、标准差、排名或案例筛选。[I17][I20]

| 条件 | fold 0 | fold 1 | fold 2 | 正式纳入 |
|---|---:|---:|---:|---:|
| `random` | 通过 | 通过 | 通过 | 3/3 |
| `fitness_direct` | 通过 | 通过 | 通过 | 3/3 |
| `agent_only` | 通过 | 通过 | 通过 | 3/3 |
| `kg_base` | 通过 | 通过 | 通过 | 3/3 |
| `kg_3features_base` | 通过 | 通过 | 通过 | 3/3 |
| `kg_base_rag` | 通过 | 通过 | 通过 | 3/3 |
| `kg_3features_rag` | 通过 | 通过 | 通过 | 3/3 |
| `kg_base_al` | 通过 | 通过 | 通过 | 3/3 |

三折汇总均报告均值 ± 样本标准差（s.d., `ddof=1`）。发现能力的主指标为最终 `best-seen`、相对初始观测的 `best-seen` 增量、按 48 次查询归一化的 best-seen AULC，以及末轮 batch best/mean/median。Spearman、Pearson、MSE、RMSE、NDCG@10、top-k hit/recall、regret@10、90% 区间 coverage 偏差和 Gaussian NLL 用于评价隔离测试集上的预测排序、误差与校准，不替代 wet-fitness 发现结论。表中粗体表示第一名，下划线表示第二名；对 fitness、相关性、NDCG 和命中指标取越大越好，对 MSE、RMSE、regret、coverage 偏差和 NLL 取越小越好。

八种策略在同一 fold 使用相同 assignment hash 和初始观测；`random`/`fitness_direct` 使用 seed 42，其余 Agent/KG 条件使用 seed 11。新增比较虽然在 fold、assignment 和 seed 上对齐，但逐轮候选池没有预先固定：`agent_only` 与 `kg_base` 的 9 个 fold–round 候选池平均 Jaccard 为 0.106，`kg_base` 与 `kg_3features_base` 为 0.081，`kg_3features_base` 与 `kg_3features_rag` 为 0.213。因此，下述 fold 对齐差值是描述性证据；在 n=3 下不进行显著性检验，也不把条件差异解释为严格隔离的组件因果效应。

### 4.2 三轮 fitness 变化趋势

![八种策略的三轮 fitness 轨迹](../analysis/gb1_al96_report_20260821/outputs/figures/figure2_fitness_trajectories.png)

**图 2｜八种策略的三轮 fitness 轨迹。** a，累计查询预算下的 best-seen；b、c，各轮选中批次的 mean 和 median。粗线和阴影分别表示三折均值和 ±1 s.d.，浅色细线为单折轨迹（n=3 folds）。所有策略均从相同的 96 个初始可见观测出发，每轮选择 16 个候选；未进行推断性显著性检验。源数据见 `round_metrics_by_fold.csv` 与 `round_metrics_mean_sd.csv`。

首先，所有策略均从相同的初始 best-seen 4.073 出发。`random` 与 `fitness_direct` 三轮后仍为 4.073 ± 0.000；`agent_only` 则依次达到 4.128 ± 0.096、4.404 ± 0.142 和 4.453 ± 0.057，说明只使用可见观测进行 LLM 方向归纳已经能够越过两个非 Agent baseline，但其累计峰值提升较慢。`kg_base` 的 best-seen 由 4.682 ± 0.340、4.879 ± 0.340 升至 5.393 ± 0.550；新增的 `kg_3features_base` 由 4.856 ± 0.452、5.228 ± 0.830 继续升至 5.657 ± 0.534，是唯一在三轮均持续提高且最终均值超过 5.5 的条件。`kg_base_rag` 和 `kg_3features_rag` 均在第二轮达到 5.075 ± 0.000 后进入平台；`kg_base_al` 第一轮即达到 5.075 ± 0.000，第三轮小幅升至 5.127 ± 0.089。

其次，批次分布显示“命中更高峰值”和“提高整个批次”并不等价。`agent_only` 的 batch mean 为 1.904 ± 0.144、1.906 ± 0.232 和 1.656 ± 0.564，对应 median 为 2.147 ± 0.236、1.965 ± 0.348 和 1.589 ± 1.306；尽管其 best-seen 低于所有 KG 条件，其末轮批次主体却高于除 `kg_base_al` 外的所有策略。`kg_3features_base` 的 batch mean 为 2.114 ± 0.662、2.326 ± 0.606 和 1.215 ± 0.309，median 为 2.001 ± 1.328、2.416 ± 0.671 和 0.684 ± 0.687，表现为累计峰值持续提高、但末轮批次主体明显回落。相比之下，`kg_base_al` 的末轮 batch mean/median 仍为最高的 1.880 ± 0.418 和 2.055 ± 0.768。由此，新数据把两类目标进一步分开：三通道无 RAG 条件偏向发现极值，主动学习偏向维持批次富集，而 Agent-only 在不依赖知识承载的情况下也能形成较强的批次级方向性。

### 4.3 跨策略性能比较

**表 1｜闭环 wet-fitness 发现指标。** 三折均值 ± s.d.；粗体为第一名，下划线为第二名。

| 策略 | 最终 best-seen | best-seen 增量 | best-seen AULC | R3 batch best | R3 batch mean | R3 batch median |
|---|---|---|---|---|---|---|
| 方向 | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ |
| `random` | 4.073 ± 0.000 | 0.000 ± 0.000 | 4.073 ± 0.000 | 1.094 ± 0.742 | 0.110 ± 0.057 | 0.004 ± 0.001 |
| `fitness_direct` | 4.073 ± 0.000 | 0.000 ± 0.000 | 4.073 ± 0.000 | 1.094 ± 0.742 | 0.118 ± 0.069 | 0.008 ± 0.002 |
| `agent_only` | 4.453 ± 0.057 | 0.380 ± 0.057 | 4.265 ± 0.025 | 4.290 ± 0.126 | <u>1.656 ± 0.564</u> | <u>1.589 ± 1.306</u> |
| `kg_base` | <u>5.393 ± 0.550</u> | <u>1.320 ± 0.550</u> | 4.765 ± 0.126 | <u>4.526 ± 1.312</u> | 1.154 ± 0.106 | 0.583 ± 0.345 |
| `kg_3features_base` | **5.657 ± 0.534** | **1.584 ± 0.534** | **4.983 ± 0.444** | **5.082 ± 0.927** | 1.215 ± 0.309 | 0.684 ± 0.687 |
| `kg_base_rag` | 5.075 ± 0.000 | 1.002 ± 0.000 | 4.888 ± 0.035 | 3.896 ± 0.672 | 1.355 ± 0.499 | 1.151 ± 0.778 |
| `kg_3features_rag` | 5.075 ± 0.000 | 1.002 ± 0.000 | 4.832 ± 0.132 | 3.921 ± 0.188 | 1.356 ± 0.430 | 1.186 ± 0.833 |
| `kg_base_al` | 5.127 ± 0.089 | 1.053 ± 0.089 | <u>4.917 ± 0.015</u> | 4.450 ± 0.740 | **1.880 ± 0.418** | **2.055 ± 0.768** |

表 1 支持三个层次的观察。其一，`agent_only` 相对 `random` 和 `fitness_direct` 已获得 0.380 ± 0.057 的 best-seen 增量，并在末轮 batch mean/median 排名第二，说明 LLM 对可见测量的直接归纳本身构成了有效 baseline，而不是随机选择的等价替代。其二，`kg_base` 相对 `agent_only` 的最终 best-seen 和 AULC 在三折均提高，表明 Experimental Memory Layer 的主要优势集中在累计高值发现。其三，`kg_3features_base` 在最终峰值、增量、AULC 和 R3 batch best 上排名第一，而 `kg_base_al` 在 R3 batch mean/median 上排名第一。故本实验没有单一的“全面最优策略”：前者偏向极值发现，后者偏向批次主体，Agent-only 则说明不使用 KG 时仍可获得相当程度的批次富集。

**表 2｜隔离测试集上的排序指标。** 三折均值 ± s.d.；Top-k 为 k=10。

| 策略 | Spearman | Pearson | NDCG@10 | Top-k hit | Top-k recall | Regret@10 |
|---|---|---|---|---|---|---|
| 方向 | ↑ | ↑ | ↑ | ↑ | ↑ | ↓ |
| `random` | 0.216 ± 0.013 | 0.185 ± 0.015 | 0.674 ± 0.010 | **0.333 ± 0.577** | **0.033 ± 0.058** | 3.570 ± 1.781 |
| `fitness_direct` | 0.235 ± 0.030 | 0.203 ± 0.032 | 0.690 ± 0.023 | **0.333 ± 0.577** | **0.033 ± 0.058** | 3.570 ± 1.781 |
| `agent_only` | 0.241 ± 0.043 | 0.199 ± 0.041 | 0.700 ± 0.026 | <u>0.000 ± 0.000</u> | <u>0.000 ± 0.000</u> | 4.471 ± 1.219 |
| `kg_base` | <u>0.243 ± 0.049</u> | **0.241 ± 0.071** | 0.707 ± 0.035 | <u>0.000 ± 0.000</u> | <u>0.000 ± 0.000</u> | 4.140 ± 1.106 |
| `kg_3features_base` | 0.213 ± 0.094 | 0.201 ± 0.102 | 0.692 ± 0.043 | **0.333 ± 0.577** | **0.033 ± 0.058** | **3.384 ± 1.432** |
| `kg_base_rag` | 0.208 ± 0.028 | 0.224 ± 0.034 | **0.714 ± 0.016** | **0.333 ± 0.577** | **0.033 ± 0.058** | <u>3.513 ± 1.072</u> |
| `kg_3features_rag` | **0.243 ± 0.004** | <u>0.238 ± 0.003</u> | <u>0.711 ± 0.003</u> | <u>0.000 ± 0.000</u> | <u>0.000 ± 0.000</u> | 4.361 ± 1.151 |
| `kg_base_al` | 0.242 ± 0.062 | 0.209 ± 0.053 | 0.699 ± 0.013 | <u>0.000 ± 0.000</u> | <u>0.000 ± 0.000</u> | 4.282 ± 1.062 |

**表 3｜隔离测试集上的误差与校准指标。** 三折均值 ± s.d.。

| 策略 | MSE | RMSE | \|Coverage−0.90\| | Gaussian NLL |
|---|---|---|---|---|
| 方向 | ↓ | ↓ | ↓ | ↓ |
| `random` | **0.172 ± 0.007** | **0.414 ± 0.008** | 0.079 ± 0.002 | **−0.250 ± 0.032** |
| `fitness_direct` | <u>0.220 ± 0.053</u> | <u>0.466 ± 0.055</u> | 0.079 ± 0.003 | <u>−0.157 ± 0.058</u> |
| `agent_only` | 0.806 ± 0.333 | 0.886 ± 0.181 | 0.080 ± 0.064 | 0.439 ± 0.260 |
| `kg_base` | 0.357 ± 0.148 | 0.589 ± 0.124 | 0.074 ± 0.014 | 0.139 ± 0.079 |
| `kg_3features_base` | 0.535 ± 0.282 | 0.716 ± 0.185 | <u>0.061 ± 0.023</u> | 0.282 ± 0.138 |
| `kg_base_rag` | 0.487 ± 0.314 | 0.675 ± 0.217 | 0.065 ± 0.027 | 0.221 ± 0.175 |
| `kg_3features_rag` | 0.530 ± 0.110 | 0.725 ± 0.078 | 0.075 ± 0.008 | 0.224 ± 0.069 |
| `kg_base_al` | 0.666 ± 0.329 | 0.796 ± 0.220 | **0.058 ± 0.027** | 0.317 ± 0.187 |

表 2–3 的排序再次没有复现 wet-fitness 发现指标。`kg_3features_base` 获得最低 regret@10 和最高实际发现指标，却只有 0.213 ± 0.094 的 Spearman；`agent_only` 的 Spearman 达到 0.241 ± 0.043，但最终 best-seen 在数值上仍低于全部 KG 条件。另一方面，`random` 的 MSE、RMSE 和 NLL 最低，却没有获得任何 best-seen 增量。该冲突说明全局预测误差受大量低值样本影响，而闭环实验关注有限预算下的高值尾部；因此，predictor 指标只能作为辅助诊断，不能替代 best-seen、AULC 和 batch 分布。

### 4.4 Experimental Memory、外部 RAG、主动学习与三通道的增量作用

![知识承载、三通道、RAG 与主动学习的 fold 对齐差值](../analysis/gb1_al96_report_20260821/outputs/figures/figure3_module_deltas.png)

**图 3｜六组模块条件的 fold 对齐差值。** a–d 分别为最终 best-seen、best-seen AULC、R3 batch mean 和 R3 batch median 的实验条件减参照条件。空心圆表示三个 fold，菱形表示均值，水平线表示观测范围，虚线为零差值（n=3 folds）。候选池未严格配对，故该图描述方向与折间一致性，不表示配对统计推断。源数据见 `kg_module_fold_deltas.csv`。

| 比较目的 | 实验条件 − 参照 | 最终 best-seen Δ | best-seen AULC Δ | R3 batch mean Δ | R3 batch median Δ |
|---|---|---:|---:|---:|---:|
| Experimental Memory | `kg_base − agent_only` | +0.940 ± 0.524（3/3 正向） | +0.500 ± 0.151（3/3 正向） | −0.502 ± 0.479（1/3 正向） | −1.005 ± 1.207（1/3 正向） |
| 三通道，无 RAG | `kg_3features_base − kg_base` | +0.264 ± 1.068（2/3 正向） | +0.218 ± 0.341（2/3 正向） | +0.060 ± 0.414（2/3 正向） | +0.101 ± 0.924（2/3 正向） |
| RAG，无三通道 | `kg_base_rag − kg_base` | −0.317 ± 0.550（0/3 正向） | +0.123 ± 0.118（2/3 正向；1/3 持平） | +0.200 ± 0.457（2/3 正向） | +0.568 ± 0.762（2/3 正向） |
| 三通道，有 RAG | `kg_3features_rag − kg_base_rag` | 0.000 ± 0.000（3/3 持平） | −0.056 ± 0.153（1/3 正向） | +0.001 ± 0.197（2/3 正向） | +0.035 ± 0.157（2/3 正向） |
| RAG，有三通道 | `kg_3features_rag − kg_3features_base` | −0.582 ± 0.534（0/3 正向；1/3 持平） | −0.151 ± 0.569（1/3 正向；1/3 持平） | +0.141 ± 0.704（2/3 正向） | +0.502 ± 1.474（2/3 正向） |
| 主动学习 | `kg_base_al − kg_base` | −0.266 ± 0.599（1/3 正向；1/3 持平） | +0.152 ± 0.111（3/3 正向） | +0.725 ± 0.382（3/3 正向） | +1.472 ± 0.553（3/3 正向） |

`agent_only → kg_base` 提供了知识承载作用的新增基线。运行审计显示，`agent_only` 的三折均关闭 knowledge、KG interaction、hierarchical hypothesis、local RAG 和全部 feature channels；其结构化 KG 均为 0 实体、0 关系，144 个入选候选也没有候选级 evidence record。相比之下，`kg_base` 仅增加 Experimental Memory Layer 及其 KG 查询能力。`kg_base` 的最终 best-seen 和 AULC 在 3/3 folds 均高于 `agent_only`，但末轮 batch mean/median 仅 1/3 folds 提高。因此，现有结果更支持“Experimental Memory 提高累计峰值发现与早期收益”，而不是“KG 使每一轮整个批次都更优”。

`kg_3features_base` 是最有希望、也最需要谨慎解释的新条件。其最终 best-seen、AULC 和 R3 batch best 均居首；相对 `kg_base`，四项差值平均为正，但只在 2/3 folds 同向，且最终 best-seen 差值的 s.d. 达到 1.068。两条件逐轮候选池平均 Jaccard 仅为 0.081，因此该结果只支持完整“三通道分层整合包”在本次运行中表现最佳，尚不能把增益因果归于某一 feature channel。

三通道执行审计覆盖两个条件。`kg_3features_base` 与 `kg_3features_rag` 的 54 条 fold–round–channel 路径全部执行 feature tool、启动 Sub-Scientist 请求并获得子 Critic `APPROVED`。前者三个通道各覆盖 90 个样本并形成 67、71、70 条 findings，后者各覆盖 87 个样本并形成 68、69、68 条 findings；两者均没有生成 child candidate hypothesis。

| 条件 / 通道 | 执行并通过 | 覆盖样本总数 | 有界 findings | child candidate hypotheses |
|---|---:|---:|---:|---:|
| `kg_3features_base` / Physchem | 9/9 | 90 | 67 | 0 |
| `kg_3features_base` / Conservation | 9/9 | 90 | 71 | 0 |
| `kg_3features_base` / Structure | 9/9 | 90 | 70 | 0 |
| `kg_3features_rag` / Physchem | 9/9 | 87 | 68 | 0 |
| `kg_3features_rag` / Conservation | 9/9 | 87 | 69 | 0 |
| `kg_3features_rag` / Structure | 9/9 | 87 | 68 | 0 |

然而，两个三通道条件的 288 个已选候选仍各自只记录一条直接 selection evidence，且全部为 `kg/measured_aggregate`；feature-channel observations 没有成为候选级 selection evidence。无 RAG 时三通道条件的平均发现指标较高，但有 RAG 时其相对 `kg_base_rag` 的最终 best-seen 在 3/3 folds 持平，AULC 还下降 0.056 ± 0.153。feature-by-RAG 的描述性交互为：最终 best-seen −0.264 ± 1.068、AULC −0.275 ± 0.487、R3 batch mean −0.059 ± 0.534、R3 batch median −0.065 ± 0.880，方向和折间差异均不足以支持加性协同。因此，新数据支持“三通道分层条件值得进一步验证”，仍不支持“三通道已通过直接选择证据稳定改善定向进化方向”。

外部 RAG 的作用在两个分层中均表现为权衡，而非稳定增益。无三通道时，RAG 提高平均 AULC 和末轮 batch mean/median，但最终 best-seen 不提高；有三通道时，RAG 相对 `kg_3features_base` 的最终 best-seen 降低 0.582 ± 0.534，末轮 batch mean/median 平均提高但折间不一致。候选池差异和 n=3 排除了“RAG 抑制三通道”的因果表述；较稳妥的结论是，当前 RAG 上下文没有显示与三通道的可重复加性协同。

相比之下，主动学习的批次作用保持最一致：AULC、R3 batch mean 和 R3 batch median 在 3/3 folds 均高于 `kg_base`，运行记录也确认每轮完成 8 个 exploitation、4 个 exploration 和 4 个 knowledge 名额。其最终 best-seen 只有一折提高、一折持平、一折降低，说明主动学习主要提高高质量候选密度和早期收益，而不是保证单点峰值。

### 4.5 Prompt—证据—KG 子图案例分析

案例从六种 Agent/KG 条件的 864 个已选候选中按预先声明的确定性规则筛选。核心 KG 正反例沿用基础 KG、RAG 和 AL 条件中的全局规则；两个三通道条件与 `agent_only` 分别保留条件内 wet fitness 最高者，以及条件内 wet fitness 后 25% 且 acquisition 前 25% 的 surprise failure。完整模型可见 Prompt、最终结构化输出、selection evidence、KG 子图与全量 shortlist 见 `outputs/case_studies/`；以下不导出隐藏推理。

| 案例 | 条件 / fold / round | 变体 | 预测 fitness | 实测 fitness | acquisition / knowledge | 直接证据 / KG 子图 |
|---|---|---|---:|---:|---:|---:|
| 核心 KG 正例 | `kg_base` / 0 / 3 | `LWAA` | 1.166 ± 0.722 | **6.027** | 2.163 / 0.552 | 1 / 3 entities |
| 核心 KG 反例 | `kg_base_rag` / 2 / 3 | `LWTC` | 2.527 ± 0.648 | **0.015** | 2.514 / 0.799 | 1 / 4 entities |
| 三通道 + RAG 正例 | `kg_3features_rag` / 0 / 1 | `LYGV` | 4.173 ± 0.486 | **5.075** | 1.437 / 0.672 | 1 / 10 entities |
| 三通道 + RAG 反例 | `kg_3features_rag` / 0 / 3 | `VYGY` | 2.077 ± 0.686 | **0.217** | 3.359 / 0.697 | 1 / 5 entities |
| 无 RAG 三通道正例 | `kg_3features_base` / 2 / 2 | `VWAA` | 3.549 ± 0.745 | **6.124** | 2.465 / 0.726 | 1 / 8 entities |
| 无 RAG 三通道反例 | `kg_3features_base` / 0 / 2 | `LYWC` | 3.405 ± 0.715 | **0.020** | 2.542 / 0.792 | 1 / 5 entities |
| Agent-only 正例 | `agent_only` / 0 / 2 | `IYGC` | 3.804 ± 0.590 | **4.486** | 1.716 / 0.000 | 0 / 0 entities |
| Agent-only 反例 | `agent_only` / 1 / 1 | `IFHA` | 1.736 ± 0.781 | **0.003** | 1.510 / 0.000 | 0 / 0 entities |

新增的 Agent-only 对照解释了“LLM 能做什么、KG 又额外提供什么”。在正例 `IYGC` 中，Scientist 仅根据当前可见观测形成了“39 位大疏水残基、40 位芳香残基、41 位保留 G、54 位 C/A/V”的软方向；该候选符合方向并达到 4.486。其输出没有 evidence ID，结构化 KG 为 0 实体/0 关系。相反，Agent-only 反例的同类 Prompt 已明确写出“residue 41 should remain G”，但包含 G41H 的 `IFHA` 仍被选中，最终 wet fitness 仅 0.003。两例说明 LLM 能从当前表格归纳有用方向，却缺少跨轮、候选级的知识承载与证据可追溯性；同时，软方向没有被下游 selection hard validator 强制执行。

`kg_3features_base` 的正例 `VWAA` 达到本次全部入选候选中的最高 wet fitness 6.124。Main Scientist 的结构化输出却把最终方向明确限定为 “soft assay-only direction”：D40W、G41A 和 V54A 主要由可见测量关联支持，conservation 与静态 structure 对其中部分替换持反对或不支持意见，physchem 仍为 analysis-only。该案例表明三通道没有把成功组合错误地禁止掉，也显示该成功不能简单归因为 feature-channel 直接选择；其候选级 evidence 仍是 support=23 的 `kg/measured_aggregate`。

对应反例 `LYWC` 更直接暴露决策接口问题。Scientist 已输出“position 41 retains G because visible G41 substitutions collapse fitness”，conservation 也只提供单点 log-odds，structure 明确声明未建模或 relax 突变侧链；但包含 G41W 的候选仍以 2.542 的 acquisition 和 0.792 的 knowledge score 被选中，实测仅 0.020。与 Agent-only 的 `IFHA` 一样，这不是信息完全缺失，而是可见方向没有稳定约束候选选择。三通道增加了对冲突和局限的描述，却尚未形成候选特异、可校准、能够影响 acquisition 或 hard validation 的证据。

此前四个案例的解释仍成立。`LWAA` 说明基础 KG 的软实验记忆能够保留组合探索并发现被 Kermut 低估的高值上位性组合；`LWTC` 说明残基级正向聚合可能掩盖 G41T 的组合损害；`LYGV` 与三通道方向一致，但不是三通道成功的必要证据；`VYGY` 则显示 structure 已提示 V54 buried pocket，包含 V54Y 的候选仍能越过软先验。综合八个案例，Experimental Memory 的新增价值主要是持久、可追溯的实验关联，而 Agent-only、KG 与三通道路线共享的主要失败模式是“方向先验与候选级选择约束脱节”。

### 4.6 主要发现、对照矩阵与证据边界

上述结果按“运行有效性 → fitness 轨迹 → 跨策略排名 → 明确参照的模块差值 → Prompt/证据/KG 个案”形成论证链。首先，24 个正式 run 建立等预算比较前提。随后，图 2 和表 1 把结论分成三层：LLM-only 已能产生批次富集，Experimental Memory 进一步提高累计峰值发现，无 RAG 三通道条件获得最高极值指标，而主动学习保持最高批次 mean/median。表 2–3 再把这些 wet-fitness 结论与 predictor 指标分离。图 3 则说明 RAG 和三通道均未形成跨分层、跨 fold 一致的加性收益。最后，案例将总体差异连接到实验记忆可追溯性、组合上位性和 selection interface 三个可检查环节。

| 实验目的 | 参照条件 | 实验条件 | 本次状态 | 可支持的结论 |
|---|---|---|---|---|
| 随机下界 | `random` | 全部 Agent/KG 条件 | 3 折完成 | 支持固定预算下的描述性发现比较；seed/候选池未严格配对 |
| 预测模型基线 | `fitness_direct` | 全部 Agent/KG 条件 | 3 折完成 | 支持区分 wet-fitness 发现与 surrogate predictor 指标 |
| LLM-only baseline | `random`/`fitness_direct` | `agent_only` | 3 折完成；KG 0 实体/0 关系 | LLM-only 可提高批次富集和有限峰值发现，但没有候选级证据承载 |
| Experimental Memory | `agent_only` | `kg_base` | 同 seed、同 assignment；3 折完成 | final best-seen 与 AULC 在 3/3 folds 提高；末轮批次主体不提高 |
| 三通道分层整合，无 RAG | `kg_base` | `kg_3features_base` | 3 折完成；27/27 通道路径通过 | 完整条件获得最高极值指标，但折间差异大，不能归因到单一 channel |
| 外部知识，无三通道 | `kg_base` | `kg_base_rag` | 3 折完成 | 部分 AULC/批次指标改善，峰值无稳定增益 |
| 三通道分层整合，有 RAG | `kg_base_rag` | `kg_3features_rag` | 3 折完成；27/27 通道路径通过 | 支持通道已执行；不支持增量 wet-fitness 优势 |
| 外部知识，有三通道 | `kg_3features_base` | `kg_3features_rag` | 3 折完成 | 峰值和 AULC 平均下降，批次差异混合；不支持加性协同 |
| 主动学习 | `kg_base` | `kg_base_al` | 3 折完成 | AULC 与末轮批次质量在 3/3 folds 改善；峰值不一致 |
| 反馈闭环 | 关闭 ReThink | 开启 ReThink | 未提供同折开关对照 | 记录反思写回，不判定独立 fitness 增益 |
| 不确定性 | No-UQ | Agent-UQ | 未提供 No-UQ 对照 | 不判定 UQ 的独立贡献 |

完整分析由 `analysis/gb1_al96_report_20260821/` 中的模块化 Python 包执行。源数据目录逐一保存折级指标、mean ± s.d.、候选池重叠、六组模块差值、feature-by-RAG 交互、主动学习执行审计、54 条三通道执行记录、新条件运行边界和全部 selected candidates；案例目录保存确定性筛选记录、Prompt 摘录、证据记录与 KG 子图；图件目录保存可编辑 SVG、PDF、600-dpi TIFF 和 PNG。固定复现命令为：

```powershell
uv run --no-project --with numpy --with pandas --with matplotlib --with pillow python analysis/gb1_al96_report_20260821/run_analysis.py
```

鉴于每个比较只有三个 fold，且逐轮候选池没有严格固定，本报告不使用“显著提高”“证明组件因果效应”等表述。下一阶段应预先固定 fold–round 候选池并增加重复 seed；对三通道还需分别设置 physchem、conservation、structure 的单通道与 leave-one-channel-out 对照，并把 feature observation 显式绑定到候选级 selection evidence。只有这样才能区分“三通道条件整体表现较好”与“具体特征真正改变了定向进化决策”。
