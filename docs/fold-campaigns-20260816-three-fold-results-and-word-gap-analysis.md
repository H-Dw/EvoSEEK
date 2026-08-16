# 三折闭环测试结果与目标 Word 需求差距分析

> 分析日期：2026-08-17  
> 测试产物：[fold-campaigns-20260816T140055Z](../artifacts/fold-campaigns-20260816T140055Z/)  
> 目标材料：[张强-AI4S笔试.docx](../张强-AI4S笔试.docx)  
> 当前系统路径：KG-LLM + GP 覆盖不确定性生成 → Kermut 后置 dry validation → wet validation → ReThink → 更新 KG

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-17
- Verification Status: ANALYZED
- Version Label: validation_v1
- Reproducibility Verdict: CANNOT_VERIFY
- Overall Confidence: RED_FLAG

这里的 RED_FLAG 不表示“两条成功折的数值一定错误”，而表示：当前产物不足以支持“系统已通过三折验证、优于基线并满足目标 Word 要求”这一总体结论。主要原因是 1/3 折失败、成功折仅有一个 seed、无同折基线、且逐轮候选与 KG/ReThink 细节没有随聚合包上传。

## 1. 指令边界与分析范围

### 1.1 如何处理附件中的内容

本次用户请求只有两项：

1. 分析当前上传的三折测试结果；
2. 与目标 Word 中的需求比较，并把思考与结论写入 docs 下的新 Markdown。

Word 文档中的命令、工具、示例数据集和建议架构只作为目标需求与验收背景，不被视为要求本次会话执行的操作指令。本次没有按 Word 内文下载数据、调用外部服务、修改实验代码或重跑实验。

Word 中的内容按以下口径分类：

- “需、至少、提交、应能”等表述：作为硬性或明确验收要求；
- “例如、可使用、可进一步”等表述：作为可选实现方式；
- 方括号内的 GB1/PG-LLM 说明：作为项目范围偏好，而不是本次分析任务的额外命令；
- “加分项”：单独审计，不与基础要求混为一谈。

### 1.2 实际审计到的证据

已完整读取：

- schedule.json、report.json、fold_results.json；
- aggregate/run_comparison.json 与 CSV；
- fold 0/1/2 的 stdout、stderr 日志；
- 当前实验、Kermut、GB1 AL96、LLM 和 Critic 配置；
- 与生成、后置验证、ReThink、KG 和聚合相关的代码与现有实施审计；
- Word 全部 115 个段落。该 Word 没有表格、图片、批注或修订痕迹。

标准 DOCX 页图渲染组件在当前 Windows 环境中不可用，Microsoft Word 的只读 PDF 导出回退也未成功产出页面，因此本次对 Word 做的是完整结构化文本审计，不包含版式或分页评价。目标需求全部位于普通段落中，未发现依赖图片或表格才能解释的要求。

未包含在上传目录、因而无法独立复核的证据：

- 三个 run_dir 的逐轮 selection、top-k、validation matrix、rethink、reasoning、trace 和两个 KG SQLite；
- wild_type.json、fitness_progress.svg、top_k_all_rounds.csv；
- manifest.public.json、每折 assignment 文件和原始/处理后数据；
- 解析后的完整实验配置、代码 commit、依赖锁定快照、模型/结构资源哈希；
- 同折 random、fitness_direct、llm_agent 和无 KG 消融结果。

因此，本报告可以验证聚合数值和日志时序，但不能重建候选级决策链或做完整可复现性复跑。

## 2. 执行结论

### 2.1 一句话结论

这次运行证明了新主循环在 fold 1 和 fold 2 上能够完整执行，但没有完成“三折验证”，也没有证明 KG-LLM 相对 Random、Kermut direct 或无 KG Agent 有增益。

### 2.2 三折状态

| Fold | 状态 | 完成轮数 | 查询数 | 关键结果 | 运行时间 |
|---:|---|---:|---:|---|---:|
| 0 | 失败 | 0/3 | 0 | 第一轮 Scientist JSON 缺 hypothesis_id，触发 KeyError | 约 2 分 45 秒 |
| 1 | 完成 | 3/3 | 96 | best_seen=7.5547；末轮 batch mean=2.2590 | 约 4 小时 30 分 |
| 2 | 完成 | 3/3 | 96 | best_seen=5.3907；末轮 batch mean=1.8760 | 约 4 小时 25 分 |

实际成功率为 2/3，即 66.7%。聚合文件只包含成功的 fold 1 和 fold 2，所以任何“跨折平均”都存在明显的完成者偏差。

### 2.3 新系统路径是否真的跑通

成功折的日志顺序一致：

1. 对 119,538 个变体生成知识 evidence；
2. 远程 Scientist 产生 hypothesis；
3. 生成 64 个 eligible candidates；
4. 以 agent_uq 完成设计分数与初选；
5. Kermut 在 96、128、160 个可见样本规模上依次拟合，并对 64 个候选做 dry validation；
6. Hard Validator 与 Critic 审批，选择 32 个变体；
7. 提交 dataset oracle，作为本测试中的模拟 wet validation；
8. 对 32 个已测变体执行 ReThink；
9. 写入每轮 wet/dry validation 与 ReThink/KG 更新；
10. 三轮后用 192 个已揭示观察拟合最终 Kermut，并在 29,439 个 final-test 变体上评估。

两条成功折均报告：

- selection_driver=agent_uq；
- fitness_predictors_used_for_generation=false；
- 96 条 selection record；
- 96 条 ReThink reflection；
- 192 条 validation record，即每个已测候选各一条 wet 和一条 dry；
- rounds_aborted=0，finalized=true。

这组日志足以支持“初选先于 Kermut dry validation”的时序结论。它不等于“LLM/KG 有效”，因为流程时序正确与相对方法增益是两个不同问题。

### 2.4 Kermut 在本轮实际发挥了多大作用

成功折每轮先从 64 个 Agent 候选中选 32 个，再看 Kermut 排名。跨三轮的平均 selected model rank fraction 为：

- fold 1：0.5163；
- fold 2：0.5138。

该值接近 0.5，符合“选择没有由 Kermut 排名驱动”的设计目标。但它只能证明解耦，不能证明 Agent 选择优于随机；而且此排名只发生在 Agent 已过滤出的 64 个候选内，不是 119,000 余个未观测变体的全空间排名。

六个成功轮次中：

- hard_conflicts 全部为 0；
- Critic 最终全部 APPROVE；
- 没有任何一轮因 dry validation 发生可见的拒绝、修订或批次替换。

因此，在当前聚合证据中，Kermut 确实被调用并进入 Critic/ReThink 上下文，但它对最终提交批次的可观察干预次数为 0。若目标是证明“后置验证有价值”，还需要报告它改变、否决或降级了哪些候选，以及这些干预是否改善 wet 结果。

## 3. 逐轮优化表现

### 3.1 Fold 1

| Round | best seen | batch best | batch mean | batch median | visible mean | hypothesis |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 6.0421 | 6.0421 | 2.9727 | 2.7893 | 1.1905 | SUPPORTED |
| 2 | 7.5547 | 7.5547 | 0.8684 | 0.3377 | 1.1260 | CONTRADICTED |
| 3 | 7.5547 | 6.1238 | 2.2590 | 1.8623 | 1.3149 | SUPPORTED |

解释：

- round 2 找到全程最优值，比 round 1 提高 1.5126，约 25.0%；
- round 3 没有刷新 best seen，batch best 反而低于 round 2；
- batch mean 呈 2.9727 → 0.8684 → 2.2590，明显不单调；
- round 3 的 batch mean 仍比 round 1 低约 24.0%；
- visible mean 先降后升。

所以 fold 1 只支持“第二轮找到了更高的单个候选”，不支持“推荐序列整体 fitness 逐轮提升”。

### 3.2 Fold 2

| Round | best seen | batch best | batch mean | batch median | visible mean | hypothesis |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 4.1128 | 4.1128 | 0.7780 | 0.0630 | 0.6418 | CONTRADICTED |
| 2 | 4.7553 | 4.7553 | 0.9578 | 0.6822 | 0.7050 | SUPPORTED |
| 3 | 5.3907 | 5.3907 | 1.8760 | 2.1444 | 0.9002 | SUPPORTED |

解释：

- best seen 连续提高 4.1128 → 4.7553 → 5.3907；
- batch mean 和 visible mean 也连续提高；
- round 3 相对 round 1，batch mean 提高约 141%，visible mean 提高约 40%。

fold 2 是这批结果中最符合“闭环逐轮改善”目标的一折。

### 3.3 为什么不能只用 best_seen 证明优化

best_seen 是累计历史最优值，按定义只能不变或上升。即使每轮新推荐越来越差，只要旧最优仍保留，best_seen 也不会下降。因此应把以下指标设为主要闭环终点：

- 每轮新 batch 的 mean、median 与分位数；
- 相对初始集和 Random 的 enrichment；
- 单位查询预算的 improvement；
- best-so-far 曲线面积，而不是只看最终最大值；
- paired baseline delta；
- 候选多样性与新颖性。

## 4. 最终 Kermut 预测与不确定性

### 4.1 每折结果

两条成功折的 final-test 样本数均为 29,439，Top-k 的 k=10。

| 指标 | Fold 1 | Fold 2 | 解释 |
|---|---:|---:|---|
| Spearman ↑ | 0.1550 | 0.2423 | 都是弱排序相关 |
| Pearson ↑ | 0.2786 | 0.1914 | 都是弱线性相关 |
| MSE ↓ | 0.9475 | 2.8449 | fold 2 约为 fold 1 的 3.0 倍 |
| RMSE ↓ | 0.9734 | 1.6867 | fold 2 是 fold 1 的 1.73 倍 |
| NDCG ↑ | 0.7729 | 0.7129 | 全局排序尚可，但不能替代 top-tail 指标 |
| Top-k hit ↑ | 1.0 | 1.0 | 两折至少命中 1 个，但门槛很低 |
| Top-k recall ↑ | 0.5 | 0.1 | 分别命中真实 Top-10 的 5 个与 1 个 |
| Regret@k ↓ | 0.0000 | 1.7826 | fold 1 找到全局最优；fold 2 未找到 |
| 90% interval coverage | 0.9669 | 0.4876 | 一折偏宽，一折严重欠覆盖 |
| Gaussian NLL ↓ | 0.5761 | 1.6091 | 概率预测跨折不稳定 |

### 4.2 跨折汇总只能作描述

对仅有的两条成功折做描述性均值：

| 指标 | 两折均值 | 两折范围 |
|---|---:|---:|
| best seen | 6.4727 | 5.3907–7.5547 |
| 末轮 batch mean | 2.0675 | 1.8760–2.2590 |
| Spearman | 0.1986 | 0.1550–0.2423 |
| RMSE | 1.3300 | 0.9734–1.6867 |
| NDCG | 0.7429 | 0.7129–0.7729 |
| Top-k recall | 0.3000 | 0.1–0.5 |
| Regret@k | 0.8913 | 0–1.7826 |
| 90% coverage | 0.7273 | 0.4876–0.9669 |

这些均值不能作为交叉验证结论：n=2，没有有效置信区间，且失败的 fold 0 被排除。

### 4.3 主要指标诊断

1. 排序能力偏弱  
   Spearman 和 Pearson 均低于 0.3。Kermut 可以找到部分 top-tail 候选，但对完整 final-test 的总体 fitness 排序仍较弱。

2. Top-k hit 过于宽松  
   两折都是 1.0，但 fold 2 的 recall 只有 0.1 且 regret 为 1.7826。只报告 hit 会掩盖 top-tail 性能差异。

3. 不确定性没有跨折校准  
   名义 90% 区间在 fold 1 覆盖 96.7%，在 fold 2 只覆盖 48.8%，相差 47.9 个百分点。当前不能把 Kermut std 直接解释为可靠的候选级置信度，也不能据此断言 dry validation 能稳定识别风险。

4. NDCG 与 top-tail 指标不一致  
   NDCG 约 0.71–0.77，但 Top-10 recall 可低至 0.1。最终目标是发现高 fitness 变体时，应把 recall、regret、enrichment 和 query efficiency 放在全局 NDCG 之前。

5. 模型的跨折稳定性不足  
   RMSE 相差 0.7133，MSE 相差 1.8974，NLL 相差 1.0329。需要按 fold 的序列距离、fitness 分布和突变组合难度分层诊断。

## 5. 运行稳定性与工程问题

### 5.1 Fold 0 的根因

fold 0 的远程 Scientist 返回了可解析 JSON，但缺少 hypothesis_id。当前远程调用层只保证“能解析成 JSON object”：

- complete_json 明确丢弃传入的 schema，只把 schema 放在 prompt 中；
- 返回后先记录 llm_request_completed；
- llm.py 随后直接读取 payload["hypothesis_id"]；
- 缺字段的 KeyError 位于 complete_json 的重试边界之外，导致整个 campaign 立即失败。

这说明“JSON 可解析”被误当成“结构契约满足”。fold 1/2 也出现了 JSONDecodeError、ValueError 和 Critic 输出校验失败，只是被上层重试后恢复。

### 5.2 远程 LLM 稳定性

成功折中的可恢复异常：

- fold 1：至少 5 次结构化输出或语义校验失败，其中含 Scientist、Critic 和 ReThink；
- fold 2：至少 2 次 ReThink JSON 解析失败；
- 多个 Critic/ReThink 调用耗时超过 90–130 秒。

这表明重试机制有价值，但契约校验位置不一致；Scientist 缺字段可以杀死整折，而 Critic/ReThink 的部分错误可以恢复。

### 5.3 Critic 门禁尚未被实证

六个成功轮次全部 APPROVE，hard_conflicts 全为 0。当前 Critic 配置中的 OOD warning 和 model disagreement 阈值为空，结果中也没有报告任何 dry-validator 导致的修改。可能有三种解释：

1. 所有候选确实合理；
2. 门禁过松或阈值未启用；
3. 上传包缺少候选级审计，无法看到实际影响。

在没有 reject/revise 对照和 wet 后验收益之前，不能把“Critic 存在”写成“Critic 已证明有效”。

### 5.4 计算成本

两条成功折的最终 29,439 个变体预测各耗时约 3 小时 59 分，几乎占据整折运行时间。日志还出现 CUDA driver 过旧警告，尽管 Kermut 明确在 CPU 上运行。

当前全池 evidence 每轮也重复扫描 119,538 个变体。若计划扩展到 5 folds × 多 seed × 4 个方法，这一实现会使正式比较成本迅速放大。需要优先使用预计算 Kermut feature store、批量预测优化和 evidence 增量缓存。

### 5.5 可复现性不足

schedule 记录了命令、fold、seed 和 manifest SHA，但仍缺少：

- resolved YAML 快照；
- git commit 与 dirty state；
- Python/torch/gpytorch/fair-esm 版本；
- ESM checkpoint、ProteinMPNN probability、结构坐标文件哈希；
- LLM provider 请求 ID、模型版本快照和完整结构化失败类型；
- 随聚合包携带的 manifest 与 run artifacts。

因此本报告标记为 ANALYZED，而不是 VERIFIED。

## 6. 统计完整性与 11 类谬误扫描

当前结果没有 p 值、置信区间或效应量，也没有预先声明 primary endpoint。完成折只有 2 个，无法做可靠的折级统计推断。以下 11/11 项均已检查。

| # | 谬误/偏差 | 当前风险 | 结论与处理 |
|---:|---|---|---|
| 1 | Simpson's paradox | 高 | 不同 fold 的趋势和误差差异很大；fold 0 缺失，不能用两折平均代表各折。完整 5 折后同时报告逐折与配对总体结果。 |
| 2 | Ecological fallacy | 中 | 折级平均不能推出每类突变或每个位点都受益。需要按单点/双点/多点、Hamming distance 和位点组合分层。 |
| 3 | Berkson's paradox | 中 | 逐轮 batch 只来自 Agent 过滤后的 64 个候选，候选内关系不能外推到全空间。 |
| 4 | Collider bias | 中 | 在“已被 Agent 选入候选集”这一条件下分析 Kermut 排名，可能人为改变模型分数与 wet fitness 的关系。 |
| 5 | Base-rate neglect | 高 | Top-k hit=1.0 没有说明命中比例；fold 2 仅命中 1/10。必须同时报告 recall、regret、enrichment 和 top 候选基率。 |
| 6 | Regression to the mean | 高 | best_seen 累计最优天然单调，不能单独证明闭环学习。使用新 batch 分布和 paired baseline。 |
| 7 | Survivorship bias | 高 | fold 0 失败后未进入聚合，成功折均值可能偏乐观。所有失败折必须保留并纳入 completion/failure endpoint。 |
| 8 | Look-elsewhere effect | 高 | 同时输出十余个指标但未指定主指标，容易只挑选好看的 NDCG 或 hit。预注册 primary/secondary endpoints。 |
| 9 | Garden of forking paths | 中高 | generation 权重、dry cap、Critic 阈值和指标选择缺少正式敏感性分析。使用固定分析计划与预定义消融。 |
| 10 | Correlation ≠ causation | 高 | 没有同折 Random、fitness_direct、llm_agent 对照，不能把 fold 2 的上升归因于 KG、LLM、GP UQ 或 ReThink。 |
| 11 | Reverse causality | 中 | 后轮看到前轮 wet 结果，同时候选分布也发生变化；逐轮改善可能来自搜索空间迁移，而不一定是 KG 学到了正确机制。 |

多重比较方面：当前没有正式假设检验，所以不存在可校正的 p 值；但指标选择本身有多重性风险。建议预先指定一个主要闭环指标和一个主要 final-test 指标，其余作为次要或诊断指标。

## 7. 与目标 Word 需求的逐项差距

状态口径：

- 已满足：当前代码或本次结果有直接证据；
- 部分满足：能力存在，但本次证据不完整或与原要求存在语义偏差；
- 未证明：仓库可能具备能力，但上传结果无法证明；
- 未满足：本次结果明确缺失。

### 7.1 项目任务

| Word 目标 | 状态 | 当前证据 | 主要不足 |
|---|---|---|---|
| 选择公开定向进化数据并评估数据大小 | 部分满足 | 使用 GB1/FLIP，README 有下载与拆分命令 | 上传包不含 manifest、数据规模报告和下载验收记录 |
| 提供本地下载 command 与极小 demo | 已满足（代码层） | README 和 scripts/data 已提供 | 本次正式结果没有把命令、版本与 checksum 快照纳入报告 |
| 合理划分 train/validation/test，避免泄漏 | 部分满足 | GB1-AL96-5CV-v1、manifest/assignment SHA、角色隔离设计 | 只跑 3 个计划折且 1 个失败；manifest 未上传，无法独立验真 |
| 以 GB1 为主任务 | 已满足 | task_id=gb1_binding_al96，96 initial，3×32 查询 | Word 中“BG1”应在最终报告统一更正为 GB1 |
| 整理 WT、突变、位点与 fitness | 未证明 | 成功 run 引用了 wild_type.json 和 top-k 文件 | 这些文件没有随当前上传目录提供 |
| 建立 fitness baseline | 部分满足 | Kermut 确实完成逐轮和 final-test 预测 | 缺简单模型 baseline，也缺同折 fitness_direct 比较 |
| 报告 Spearman/Pearson/MSE/Top-k | 已满足 | 两成功折均输出完整指标 | 主要指标弱、Top-k hit 容易误导，且没有统计区间 |
| Agent 读取当前实验与 top variants | 未证明 | 远程 Scientist 每轮产生 hypothesis | 缺 prompt/context、reasoning 和 KG query artifact |
| 总结有益位点 | 未证明 | hypothesis 与 evidence 路径存在 | 上传包没有位点级总结或频次/效应图 |
| 生成下一轮候选 | 已满足 | 每轮 64 proposed、32 selected | fold 0 第一轮未生成 |
| 调用 fitness predictor 对候选打分 | 已满足但后置 | Kermut 对 64 候选做 dry validation | 不在生成期打分，是本系统有意的架构变更 |
| 根据预测结果选择 Top-k | 不符合默认路径原义 | 默认 selection_driver=agent_uq，predictor 不参与生成 | 应以独立 fitness_direct arm 满足此要求，并与 Agent arm 对照；不要重新让 Kermut 掩盖 Agent |
| 给出推荐理由 | 未证明 | summary 记录 selection_records，run_dir 引用 reasoning.md | reasoning.md 未上传 |
| Data Analyst/Hypothesis/Designer/Evaluator/Critic 模块化 | 部分满足 | Scientist、generator、Kermut、Critic、ReThink 代码与日志存在 | 当前结果不展示各模块输入输出契约和失败降级 |
| 构建知识库/规则库 | 部分满足 | Operational KG、structured KG、evidence 路径存在 | 没有随结果上传 KG 文件或关系统计 |
| 比较有无知识增强，模块可拆分消融 | 未满足 | 仓库存在 no_kg 等消融入口 | 本次 campaign 只有 knowledge_agent |
| 至少 2–3 轮虚拟进化 | 部分满足 | fold 1/2 完成 3 轮 | fold 0 在第一轮前失败，整体三折未完成 |
| 用测试真实 fitness 或模型作为虚拟实验器 | 已满足 | dataset oracle 作为模拟 wet，Kermut 作为 dry | 必须明确这不是现实湿实验 |
| 观察 fitness 是否逐轮提升 | 部分满足 | fold 2 的 batch 和 best 均提升 | fold 1 的 batch mean 不单调，best 在 round 3 平台 |
| 展示每轮 Top-k | 未满足（上传包） | run_dir 声称生成相关文件 | 聚合包没有任何候选表 |
| 分析关键位点 | 未满足（上传包） | 无位点级结果 | 需要 residue/mutation enrichment 与交互分析 |
| 比较 Random、模型直接、LLM、知识增强 LLM | 未满足 | same_fold_baseline_available=false | 这是目前最大的实验归因缺口 |
| 成功与失败案例分析 | 部分满足 | 有 hypothesis contradicted 和 LLM 结构化失败 | 没有候选级 dry/wet 冲突、机制失败和反事实配对 |
| 讨论 Agent 是否真正具有科学家思维 | 未满足（实证层） | 架构上已把 predictor 从生成期移除 | 还缺 no-LLM、no-KG、score-shuffle、evidence-deletion 的正式同折结果 |

### 7.2 交付物要求

| Word 交付要求 | 状态 | 差距 |
|---|---|---|
| 3–5 页 PDF 实验报告 | 未满足/未提供 | 当前上传的是运行聚合包，没有目标格式报告 |
| 报告覆盖背景、数据、模型、Agent、知识、虚拟实验、失败案例和未来工作 | 未满足/未提供 | 本分析可作为报告素材，但不是最终 PDF |
| 提交 GitHub 链接 | 未满足 | README 的 clone 地址仍是占位符，不是实际仓库 URL |
| 包含数据处理、训练、Agent、生成和评估代码 | 部分满足 | 本地仓库具备这些模块；尚未通过外部 GitHub 交付验收 |
| README 说明环境、来源、命令和结果 | 部分满足 | 环境、来源和命令较完整；当前结果未回写，且 baseline 章节仍描述 predictor top-μ，和新 agent_uq 后置验证路径冲突 |
| 小规模数据可复现 | 部分满足 | demo/测试入口存在 | 本次正式结果缺 resolved config、commit、依赖和资源哈希 |
| 展示 WT 与突变序列 | 未满足（上传包） | wild_type.json 和逐轮候选文件未上传 |
| 表格/曲线展示 fitness | 未满足（上传包） | fitness SVG 和 top-k CSV 只在外部 run_dir 路径中被引用 |
| 展示推理过程与失败原因 | 未满足（上传包） | reasoning、ReThink、KG interaction 文件未上传 |
| 注明外部数据、模型、代码或 ChatGPT 的来源与使用方式 | 部分满足 | README 有部分来源；最终报告和本次结果包没有完整 provenance 表 |

### 7.3 加分项

| 加分项 | 状态 | 评价 |
|---|---|---|
| 主动学习/强化学习选择 | 部分满足 | 3×32 budget 的闭环与 agent_uq 属于主动选择；没有 RL |
| GP 或集成不确定性 | 已实现但未校准 | generation 使用 coverage GP，Kermut 输出 std；90% coverage 跨折严重不稳定 |
| 比较单点、双点、多点优化 | 未满足 | 没有按 mutation count 分层结果 |
| 结构信息或保守位点 | 部分满足 | Kermut/knowledge 配置有结构与保守先验；本次无候选级证据展示 |
| 位点—突变—性质—fitness KG | 部分满足 | schema 和写入路径存在；结果包没有图谱快照与查询示例 |
| 可交互 demo | 未满足 | 有命令行 demo，无“输入 WT 自动推荐”的交互产品证据 |
| 连接真实自动化实验平台 | 仅讨论/接口层 | backend 可替换为 LIMS，但当前 wet 是 dataset oracle，没有真实 QC、批次、复测和机器人接入 |

## 8. 与新架构目标本身的差距

即使不考虑 Word，当前新架构还存在以下内部验收缺口。

### 8.1 已被本次实跑确认

- Agent/coverage-UQ 初选在 Kermut fit 之前；
- Kermut 对初选候选做后置 dry validation；
- Critic 审批后才提交 oracle；
- wet 与 dry 都被计入 validation records；
- 每个 wet 候选都有 ReThink；
- 成功折会执行 final-test 预测；
- predictor 没有直接参与 generation score。

### 8.2 尚未被本次实跑确认

- KG prior 相比无 KG 条件是否提高发现效率；
- coverage GP 相比无 UQ 是否提高信息效率；
- Kermut dry validation 是否减少失败候选；
- ReThink 是否改变下一轮 mutation，而非只生成文本记录；
- wet/dry 权重 1.0:0.2 与 0.85 recency 是否合理；
- Critic 是否能在真实冲突中拒绝或修订；
- 相同 fold 下 KG-LLM 是否优于 fitness_direct；
- 远程 LLM 输出契约在长时间并发运行中是否稳定；
- 结果在不同 seed、其余 fold 和不同模型版本下是否可重复。

### 8.3 文档与实现发生漂移

现有实施审计已说明 knowledge_agent 默认使用 agent_uq 且关闭 generation predictor，但 README 的“四种规定 baseline”仍把 LLM Agent 和 Knowledge-enhanced LLM Agent 描述为 predictor top-μ 选择。最终交付前必须统一：

- 新系统的主路径；
- 为满足 Word 原始“根据预测选择 Top-k”而保留的 fitness_direct 对照路径；
- 两条路径分别回答什么科学问题。

否则读者会把 Kermut 后置验证误解成前置选择器，或反过来认为系统没有完成 Word 的预测模型调用要求。

## 9. 优先级整改建议

### P0：先让结果可用于作结论

1. 修复远程 JSON 契约  
   在重试边界内执行 JSON Schema/Pydantic 校验，包括 required、类型、枚举和附加字段；只有领域对象构造成功后才记录 completed。缺字段应触发带字段路径的 retry，而不是让 KeyError 杀死整折。

2. 增加安全恢复  
   每个 phase 写可恢复 checkpoint。LLM 契约失败耗尽重试后，fold 应标记为 paused/failed-at-phase，并支持同 run_id 或明确 continuation ID 从安全边界恢复，避免重做数小时计算。

3. 先原样补跑 fold 0  
   使用同一 manifest SHA、assignment、seed=42、budget=96 和代码版本补跑。只有 3/3 都完成后，才允许生成“三折汇总”。

4. 运行严格同折四方法对照  
   每个 fold、seed、budget、candidate universe 和 Kermut 资源完全配对：
   - Random；
   - fitness_direct；
   - llm_agent + coverage UQ；
   - knowledge_agent + coverage UQ。

5. 打包完整证据  
   fold campaign 目录应复制或相对链接每个成功 run 的 summary、resolved config、top-k、validation matrix、ReThink、reasoning、curve、KG query audit 和 manifest，不能只留下另一台机器上的绝对路径。

### P1：校准与归因

1. 预先声明主要终点  
   建议主闭环终点为 paired delta of area-under-best-so-far 或末轮 batch median；主 final-test 终点为 Top-k recall 或 regret。Spearman、RMSE、NDCG、coverage 和 NLL作为次要指标。

2. 做 fold 级配对推断  
   先报告每折差值，再用 paired bootstrap 或精确配对置换；只有 5 折时推断能力仍有限，正式结论宜增加 3–5 个 paired seeds，并用 fold/seed 分层模型或层级 bootstrap。

3. 校准 Kermut  
   使用每折专属 validation 做 conformal/calibration，不得查看 final-test；报告 coverage-risk、interval width、NLL、分 Hamming distance/OOD 的 coverage。当前 48.8%–96.7% 的跨度不可接受为稳定 dry validator。

4. 证明后置 validation 的边际价值  
   增加 no-dry、dry-only-report、dry-gated 三个条件；报告因 Kermut/Critic 被修改或拒绝的候选数、这些候选的 wet 结果，以及 gating 对 batch fitness 的净收益。

5. 做关键消融  
   至少包括 no-KG、no-UQ、no-ReThink、no-Critic，以及 dry_weight_cap=0/0.05/0.1/0.2/0.4。所有消融必须同折、同 seed、同预算。

6. 做候选级失败分析  
   对 dry 高/wet 低、dry 低/wet 高、hypothesis contradicted、Critic approve 但 wet 失败四类样例进行 matched counterfactual 分析。

### P2：交付与产品化

1. 更新 README，使 baseline 描述与新主循环一致，并填入真实 GitHub URL；
2. 生成 Word 要求的 3–5 页 PDF，压缩为“问题—方法—对照—结果—失败—局限”结构；
3. 补齐 WT、逐轮 Top-k、fitness 曲线、关键位点图、单/双/多点分层图；
4. 建立一个输入 WT/允许位点/预算即可运行的小型交互 demo；
5. 将 dataset oracle 明确标为 simulated wet，另行实现 LIMS/robot adapter、QC、重复测量、assay batch 与失败重试。

## 10. 建议的下一轮最小验收矩阵

为了先用可控成本回答最关键问题，建议分三道门执行。

### Gate A：可靠性修复

| 验收项 | 通过条件 |
|---|---|
| 原三折补齐 | fold 0/1/2 全部 finalized |
| LLM 契约 | 缺字段会 schema retry，不出现裸 KeyError |
| 断点恢复 | 人为制造一次 LLM 失败后可从安全 phase 恢复 |
| 证据包 | 三折每个 run 的候选、validation、ReThink、KG、配置均可本地打开 |

### Gate B：最小归因实验

在 fold 0/1/2、seed 42 上跑四个配对条件：Random、fitness_direct、llm_agent、knowledge_agent。先回答：

- KG 是否优于无 KG LLM；
- Agent 是否优于 Random；
- Agent 是否优于或补充 Kermut direct；
- fold 2 的逐轮上升能否在其他折复现。

### Gate C：正式报告

扩展到完整 5 folds，至少 3 个 paired seeds；固定 primary endpoint、统计计划和消融矩阵。只有同时满足以下条件，才建议写“系统有效”：

- 所有折和 seed 的 completion rate 达到预设阈值；
- knowledge_agent 相对至少一个强基线有一致、可量化的 paired 改善；
- Kermut calibration 在各折达到可接受范围；
- 结论不依赖单个 outlier fold 或只看 best_seen；
- 候选级失败案例、负结果和外部资源 provenance 完整公开。

## 11. 最终判断

### 对新架构

工程路径验证为“部分通过”：两折完整跑通了生成、dry、模拟 wet、ReThink 和 KG 更新，且初选与 Kermut 已解耦。最严重的问题是 LLM 结构化契约不稳、fold 0 直接崩溃、Kermut 不确定性跨折失准、Critic/dry validation 没有显示任何候选级干预。

### 对科学结论

当前不能回答“KG-LLM 是否优于直接适应度模型或随机选择”。fold 2 很有希望，fold 1 混合且第三轮平台，fold 0 缺失；没有配对 baseline 和消融，任何归因都属于推测。

### 对目标 Word

代码层已覆盖数据、模型、Agent、KG、三轮闭环和不确定性等大部分模块，但实验层与交付层仍有明显缺口。最关键的未完成项是：

1. 四方法正式比较；
2. 有/无知识增强消融；
3. 完整三折或五折与多 seed；
4. 每轮 Top-k、关键位点、成功/失败案例；
5. 3–5 页 PDF、真实 GitHub 链接和完整可复现证据包；
6. 明确区分 dataset oracle 的模拟 wet 与真实湿实验。

只有补齐这些内容，才能从“功能完整的工程 MVP”升级为“对 Word 目标有充分证据支撑的科学智能体实验”。
