# Open-design 运行时与本地交互界面

## 已实现边界

`open_design` 现在从完整参考序列生成新的单点替换序列，不读取候选池作为 proposal 来源。运行时只有一个 `ResolvedDesignSpace`：`OpenDesignRunner` 在入口解析一次位置策略，或复用交互 preview 已解析并确认的对象；随后把同一对象传给 proposer、Scientist、审计产物和 hard validator。

运行时明确区分两类范围：

- `computation_positions`：完整参考序列，供 sequence feature、posterior、知识和结构上下文计算；
- `allowed_mutation_positions`：唯一突变权限集合，由 `all`、`include` 或 `all_except` 解析得到。

Scientist 可看到完整序列上下文，但其输出位置只允许来自 `allowed_mutation_positions`。完整序列观测通过 `residues_by_position` 提供真实编号到残基的映射，不能把稀疏允许位置列表的索引当成完整序列索引。

## 安全与审批链路

`OpenDesignHardValidator` 会从参考序列重新推导每条候选的真实 edits，并校验：

- 完整序列长度与 canonical amino-acid alphabet；
- `variant`/`sequence` 一致性和 sequence-derived SHA-256 ID；
- `mutation_count`、mutation depth 和 mutation notation；
- source/target residue 与参考序列一致；
- 每个编辑位置属于同一个 resolved design space；
- 批次预测、重复、已观测/待处理状态、evidence snapshot 和 draft hash。

验证通过后仍必须经过 Critic；只有 `APPROVE` 决策才能由 `ApprovalGateway` 签发 receipt。最终 FASTA/CSV/JSON 只从 approved batch 导出。

## Predictor capability gate

每个 posterior predictor 暴露 `supports_full_sequence` 和 `supports_generated_sequences`。`open_design` 配置加载时要求所有 posterior predictor 同时支持两项能力。内置 `full_sequence_onehot` 支持；`gb1_onehot_pairwise` 等四位点 compact provider 会在配置加载时被拒绝，不再延迟到模型特征计算阶段失败。外部 predictor 必须在可信 model config 中显式声明 capability，未声明时按不支持处理。

## 命令行与 UI

安装本地 UI 可选依赖：

```powershell
.venv\Scripts\python.exe -m pip install -e ".[ui]"
```

只生成结构化 preview，不创建 run directory：

```powershell
fitness-agents design configs/experiments/knowledge_agent_open_design.yaml `
  --prompt "希望提高结合能力，开放全部位置，输出 8 条"
```

核对 preview 后显式确认运行：

```powershell
fitness-agents design configs/experiments/knowledge_agent_open_design.yaml `
  --prompt "仅优化位置 2,5,17，提高结合能力，输出 8 条" `
  --confirm
```

启动只绑定本机回环地址的 Gradio Blocks 界面：

```powershell
fitness-agents serve configs/experiments/knowledge_agent_open_design.yaml `
  --host 127.0.0.1 --port 7860
```

推理脚本入口：

```powershell
.venv\Scripts\python.exe scripts/run_open_design.py `
  --experiment configs/experiments/knowledge_agent_open_design.yaml `
  --prompt "希望对该序列进行定向进化，提高结合能力" `
  --fasta path\to\reference.fasta
```

交互入口只能覆盖 objective、位置策略和不超过可信配置上限的预算。它不能从 prompt 修改 API key、路径、backend factory、Critic/Approval 或泄漏控制。输入序列必须与所选可信配置的 reference 一致；新蛋白需要单独的 task config 和匹配的初始测量，系统不会把 GB1 标签套用到任意序列。

## 当前科学限制

- 当前 proposer 只支持精确单点替换；多点 beam/joint posterior 尚未实现。
- 当前 runner 是单轮 design/export；尚未接 sequence-aware lab queue 或多轮 measurement backend。
- posterior、知识、结构和 Scientist 输出均为干式决策证据，不是实验测量，也不表示已经取得 fitness 改善。
- 至少需要 4 条可见初始观测；更可靠的 calibration 仍取决于配置的 training/calibration 最小样本量。

