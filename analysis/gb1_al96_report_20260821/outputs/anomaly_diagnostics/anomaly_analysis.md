# Best fitness 上升而 batch mean/median 下降：异常诊断

## Figure contract

- Core conclusion：best-seen 是不可下降的累计极值；本批实验的后期上升由极少数新纪录候选驱动，而批次主体同时向低 fitness 移动。
- Figure archetype：quantitative grid。
- Backend：Python/matplotlib。
- Statistics：3 folds；候选分布每个 condition × round 合并 48 个已揭示候选；不进行显著性检验。
- Reviewer risk：未获得未选中候选的 wet 标签，因此不能把下降严格归因于候选池真值耗竭。

## 诊断结论

### 1. 这首先是累计极值与批次分布的统计口径差异

`best_seen_fitness` 定义为截至当前轮所有已测候选的累计最大值，因此数学上只能持平或上升；batch mean 和 median 只描述当轮 16 个新候选。一个候选刷新纪录即可抬高 best-seen，即使其余 15 个候选整体变差。实际新纪录也高度稀疏：

| 条件 | Round 1 新纪录 folds | Round 2 | Round 3 |
|---|---:|---:|---:|
| `kg_base` | 3/3 | 1/3 | 1/3 |
| `kg_base_rag` | 3/3 | 1/3 | 0/3 |
| `kg_base_al` | 3/3 | 0/3 | 1/3 |

因此，图中的‘逐轮上升’不代表每个 fold 或整批候选持续改善。Round 3 只有 `kg_base` fold 0 的 `LWAA`（6.027）和`kg_base_al` fold 2 的 `IWGM`（5.229） 刷新既有纪录；`kg_base_rag` 在 Round 3 没有任何 fold 刷新纪录。

### 2. mean 和 median 下降来自批次主体真实下移，而非单个离群值造成

| 条件 | pooled median R1 → R3 | fitness ≤0.05 | fitness ≥2 | 去除每轮最大值后的 mean R1 → R3 |
|---|---:|---:|---:|---:|
| `kg_base` | 2.435 → 0.563 | 14.6% → 41.7% | 72.9% → 27.1% | 2.316 → 1.051 |
| `kg_base_rag` | 2.376 → 0.523 | 14.6% → 29.2% | 62.5% → 37.5% | 2.098 → 1.288 |
| `kg_base_al` | 3.359 → 2.276 | 0.0% → 16.7% | 91.7% → 58.3% | 3.294 → 1.808 |

三个条件均出现低值质量增加、高值质量减少。尤其是 `kg_base`，≤0.05 的候选从 14.6% 增至 41.7%，≥2 的候选从 72.9% 降至 27.1%；即使删除每轮最高候选，mean 仍从 2.316 降至 1.051。这证明下降是批次主体变化，而不是最高点把统计量‘拉坏’。

### 3. 后期入选候选在 dry predictor 看来也更弱

| 条件 | selected predicted mean R1 → R3 | wet mean R1 → R3 | acquisition–wet Pearson R3 |
|---|---:|---:|---:|
| `kg_base` | 2.291 → 1.263 | 2.373 → 1.154 | 0.274 |
| `kg_base_rag` | 2.202 → 1.809 | 2.160 → 1.355 | -0.066 |
| `kg_base_al` | 2.958 → 2.118 | 3.331 → 1.880 | 0.469 |

这排除了‘模型仍认为后期批次同样优秀，只是测量偶然变差’这一解释。三条路线的 selected predicted mean 都下降；同时 `kg_base_rag` Round 3 的 acquisition–wet 相关接近零，说明综合 acquisition 中的知识、先验、覆盖不确定性和控制配额未能继续稳定映射到 wet fitness。

### 4. 固定分臂使批次目标不等于纯粹最大化当轮均值

`kg_base`/`kg_base_rag` 的请求配额为 8 个 hypothesis-target、3 个 evidence-prior、3 个 coverage-exploration 和 2 个 matched-control，出现 shortfall 时由 fallback 补齐；`kg_base_al` 的请求配额为 8 个 exploitation、4 个 exploration 和 4 个 knowledge。后半批候选承担覆盖、证伪和对照功能，本来就不保证具有最高即时 fitness。各分臂 wet 结果已输出到 `acquisition_arm_outcomes_mean_sd.csv`。

### 5. acquisition 分臂放大下降，但核心优化臂本身也在变弱

| 条件 | 核心臂 wet mean R1 → R3 | 支持臂 wet mean R1 → R3 | R3 核心臂相对全批次差值 |
|---|---:|---:|---:|
| `kg_base` | 2.716 → 1.599 | 2.031 → 0.710 | +0.445 |
| `kg_base_rag` | 2.251 → 1.839 | 2.069 → 0.870 | +0.485 |
| `kg_base_al` | 3.591 → 2.461 | 2.552 → 0.136 | +0.581 |

`kg_base_al` 的 exploration arm 下降最明显（2.552 → 0.136），Round 3 有 58.3% exploration 候选 ≤0.05；若只看 exploitation + knowledge，Round 3 mean 为 2.461，高于全批次 1.880。类似地，`kg_base` 的 coverage-exploration 从 2.845 降至 0.374，`kg_base_rag` 的 matched-control 在 Round 3 只有 0.081。尽管如此，核心臂也同步下降：`kg_base` hypothesis-target 为 2.716 → 1.599，`kg_base_al` exploitation + knowledge 为 3.591 → 2.461。因此，探索/控制配额是放大器，不是唯一原因。

### 6. 后期序列组成更少满足早期高值残基模式

| 条件 | 平均 preferred-site count R1 → R3 | position 41=G 比例 | R3 wet mean：41G / non-G |
|---|---:|---:|---:|
| `kg_base` | 3.35 → 2.77 | 89.6% → 56.2% | 1.68 / 0.38 |
| `kg_base_rag` | 3.12 → 2.94 | 83.3% → 56.2% | 1.97 / 0.47 |
| `kg_base_al` | 3.71 → 3.04 | 100.0% → 87.5% | 2.12 / 0.03 |

`kg_base` 与 `kg_base_rag` 中 41G 比例均降至 56%，而 Round 3 的 41G 候选平均 fitness 明显高于 non-G。`kg_base_al` 保持了较高的 41G 比例，但平均 preferred-site count 仍下降。该组成漂移与 batch body 下移一致。这里的 preferred-site 集合属于事后描述性诊断，不是独立验证的因果规则；它也不应被转化为硬过滤，因为 `LWAA` 等非 G41 高值变体证明了上位性例外存在。

### 7. 不能把下降解释为 119k 全局候选空间被耗尽

| 条件 | pool utility mean R1 → R3 | selected utility mean R1 → R3 | unselected utility mean R1 → R3 |
|---|---:|---:|---:|
| `kg_base` | 1.814 → 1.592 | 1.927 → 1.791 | 1.701 → 1.393 |
| `kg_base_rag` | 1.747 → 1.603 | 1.874 → 1.855 | 1.621 → 1.352 |
| `kg_base_al` | 0.500 → 0.500 | 0.631 → 0.655 | 0.369 → 0.345 |

utility 的量纲在条件内解释；`kg_base_al` 使用秩归一化 utility，不能与其他条件横向比较。每个条件中 selected utility 仍高于 unselected utility，说明选择器按自身 dry 目标在小池内实现了偏好；但 `kg_base`/`kg_base_rag` 的 pool 和 selected utility 总体下降，且 dry–wet 对齐在后期变弱。

每轮实际只从约 119,442 个可用候选中抽取 32 个并选择 16 个。到 Round 3，目录仅减少 32 个，即 0.027%；相邻轮 32-candidate pools 的平均 Jaccard 为 0.115。因此，全局候选空间耗竭不是数据支持的主因。更符合证据的解释是：首轮优先抓住明显的高值模式；后续小候选池构成改变，且 acquisition 继续为探索、知识验证和控制分配预算，使入选集合的预测质量与实际质量同步下移。由于未选中候选没有 wet 标签，无法进一步区分‘小池本身变难’与‘dry 目标对 wet fitness 的排序变差’各占多少。

## 论文表述建议

不应写成‘best fitness 随轮次持续改善，但 batch quality 异常下降’。更准确的表述是：

> 累计 best-seen 在少数 fold 中被稀疏的新纪录候选继续抬高，而当轮批次的中位数、高值候选比例和 dry-predicted mean 同时下降，说明峰值发现与批次富集发生分离。该现象主要来自累计极值的单调定义、首轮高值候选的前置捕获，以及后续轮次对探索/知识/控制预算的持续分配；现有数据不支持全局候选空间耗竭这一解释。
