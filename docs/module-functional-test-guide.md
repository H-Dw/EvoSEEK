# 主要模块独立功能测试指南

## 1. 目的与模块边界

本测试集按“可独立验收的业务能力”划分，而不是机械地为每个 Python 文件创建脚本。当前系统的主要数据流为：数据规范化与隔离切分 → 特征和 fitness 预测 → 候选生成与采集 → 知识证据与 KG → Scientist/Critic 审批 → 实验闭环 → 指标和报告。

共设置 8 个独立功能测试：

| 模块 | 主要源码 | 测试脚本 | 配置 |
|---|---|---|---|
| 数据规范化与防泄漏切分 | `data/adapters`、`data/splitting`、`data/loader.py` | `scripts/module_tests/test_data_pipeline.py` | `configs/module_tests/data_pipeline.yaml` |
| 特征与 fitness 预测模型 | `features`、`models`、`evaluation/metrics.py` | `scripts/module_tests/test_predictive_model.py` | `configs/module_tests/predictive_model.yaml` |
| 突变设计、采集与冲突检测 | `mutation`、`acquisition` | `scripts/module_tests/test_design_acquisition.py` | `configs/module_tests/design_acquisition.yaml` |
| 运行时知识引擎 | `knowledge` | `scripts/module_tests/test_knowledge_runtime.py` | `configs/module_tests/knowledge_runtime.yaml` |
| 结构化知识图谱构建 | `kg_knowledge`、`plugin_registry.py` | `scripts/module_tests/test_kg_construction.py` | `configs/module_tests/kg_construction.yaml` |
| KG 安全交互与提案写回 | `kg_interaction` | `scripts/module_tests/test_kg_interaction.py` | `configs/module_tests/kg_interaction.yaml` |
| Scientist、Critic 与审批网关 | `agents`、`validation`、`loop/review.py` | `scripts/module_tests/test_agents_review.py` | `configs/module_tests/agents_review.yaml` |
| 闭环 Campaign、评估与报告 | `loop`、`evaluation`、`reporting` | `scripts/module_tests/test_campaign_evaluation.py` | `configs/module_tests/campaign_evaluation.yaml` |

这些脚本是功能验收脚本：每个脚本均可直接启动、自建小型确定性模拟输入、正常完成时返回退出码 0，并在自己的输出目录写入 `result.json`。它们补充现有 `tests/unit`、`tests/integration`、`tests/leakage` 和 `tests/e2e`，不替代 pytest 回归测试。

## 2. 环境与总启动命令

先在仓库根目录安装开发依赖：

```bash
python -m pip install -e ".[dev]"
```

Windows PowerShell 推荐使用：

```powershell
.\.venv\Scripts\python.exe scripts\module_tests\run_all.py
```

Linux/macOS 推荐使用：

```bash
.venv/bin/python scripts/module_tests/run_all.py
```

也可以使用当前环境中的 `python`：

```bash
python scripts/module_tests/run_all.py --output-root artifacts/module_tests
```

总入口顺序执行 8 个模块，并生成：

- `artifacts/module_tests/summary.json`：总通过/失败统计；
- `artifacts/module_tests/<module>/result.json`：每个模块的结构化结果；
- 各模块自己的数据库、CSV、manifest、campaign trace 和报告等中间验收产物。

加入 `--stop-on-failure` 可在首个失败模块后停止。总入口不会启用真实 LLM API，也不会加载真实外部模型 checkpoint。

## 3. 各模块命令、输入输出与主要流程

### 3.1 数据规范化与防泄漏切分

启动命令：

```bash
python scripts/module_tests/test_data_pipeline.py --config configs/module_tests/data_pipeline.yaml
```

输入：配置中的 GB1 四个位点参考序列、每个位点的模拟残基集合、折数和三个切分协议参数。脚本通过残基笛卡尔积构造 81 条带确定性 target 的原始记录，不依赖外部下载。

主要流程：

1. 经通用 sequence adapter 规范化为 features/labels 两张表；
2. 验证 target 及 target proxy 不进入公开特征；
3. 分别执行 `al96_closed_loop`、`flip_static_ood`、`mutation_identity_ood`；
4. 对每种切分执行泄漏审计、manifest/哈希写入和幂等重写；
5. 分别以 agent、controller、oracle、evaluator、auditor 权限加载 fold；
6. 验证 agent 看不到 oracle/final labels，oracle 和 evaluator 权限互相隔离。

输出：`synthetic_raw.csv`、三套版本化 split 目录、公开/分折 manifest、审计文件和 `result.json`。

### 3.2 特征与 fitness 预测模型

启动命令：

```bash
python scripts/module_tests/test_predictive_model.py --config configs/module_tests/predictive_model.yaml
```

输入：脚本生成的 81 条模拟 public/oracle benchmark；配置提供 baseline 超参数、外部后端工厂和 checkpoint 占位配置。

主要流程：

1. 拟合并转换 GB1 one-hot + pairwise 特征；
2. 拟合 heterogeneous ensemble，输出 mean、std、90% 区间、OOD 和 component scores；
3. 用 evaluator-only 标签计算相关性、RMSE、NDCG、coverage 和 Gaussian NLL；
4. 对 Kermut、ProteinNPT、ProSST、Pythia-PPI 四个别名执行外部 predictor adapter 完整契约测试；
5. 验证外部后端返回乱序结果时，adapter 恢复请求顺序；
6. 可选加载配置中的真实外部模型。

默认外部契约测试使用 `scripts/module_tests/fakes.py`，它只验证本系统的 adapter、fit/predict、顺序恢复与 schema，不声称真实 checkpoint 已通过。真实模型测试见第 4 节。

输出：模拟 benchmark CSV、预测器/指标摘要和 `result.json`。

### 3.3 突变设计、采集与冲突检测

启动命令：

```bash
python scripts/module_tests/test_design_acquisition.py --config configs/module_tests/design_acquisition.yaml
```

输入：脚本生成的候选 Variant、Prediction、正反 Evidence、Hypothesis 和 CampaignState；阈值来自配置。

主要流程：

1. 执行 enumerating、hypothesis-filtered、knowledge-filtered 候选生成路径；
2. 执行 Random、Greedy、UCB、Thompson/TS 五个采集入口；
3. 应用知识加权、batch budget 和序列多样性惩罚；
4. 对合法/非法残基编辑执行 residue-level hard validation；
5. 触发并检查高 OOD、模型分歧、证据极性冲突和缺失预测；
6. 分别计算完整 posterior 下的 sign epistasis 和缺少 constituent 时的 UNKNOWN 结果。

输出：各生成器排序、各采集策略选择结果、冲突 code、epistasis 结果和 `result.json`。

### 3.4 运行时知识引擎

启动命令：

```bash
python scripts/module_tests/test_knowledge_runtime.py --config configs/module_tests/knowledge_runtime.yaml
```

输入：模拟 observation、candidate 和 prediction；配置提供四个内置知识通道及位点 profile。脚本还注册一个自定义 EvidenceProvider。

主要流程：

1. 写入已揭示 observation；
2. 计算 physchem、conservation、structure、KG 和自定义证据；
3. 融合证据分数并记录 prediction/evidence 类型；
4. 写入上一轮 hypothesis；
5. 通过 allow-listed KG tool 查询 hypothesis context 和单变体解释；
6. 注入一个当前轮隐藏测量，验证其不会出现在同轮上下文；
7. 验证 query audit、typed graph edge 和 evidence deletion 消融。

输出：`knowledge.sqlite`、证据/查询/边类型摘要和 `result.json`。

### 3.5 结构化知识图谱构建

启动命令：

```bash
python scripts/module_tests/test_kg_construction.py --config configs/module_tests/kg_construction.yaml
```

输入：模拟 Variant、Observation、Prediction、Evidence、Hypothesis，以及一个带别名的外部 Protein 实体。

主要流程：

1. 通过插件注册表加载 campaign observation、inference、external 三个 adapter；
2. 执行 alias normalization；
3. 执行来源感知 fusion、去重、关系端点清理和 core schema validation；
4. 将 snapshot 写入 in-memory sink；
5. 再以 observations-only profile 构建一次，验证 adapter/layer/modality 消融。

输出：full 与 observations-only snapshot 的实体/关系统计、adapter report 和 `result.json`。

### 3.6 KG 安全交互与提案写回

启动命令：

```bash
python scripts/module_tests/test_kg_interaction.py --config configs/module_tests/kg_interaction.yaml
```

输入：脚本先建立一个真实运行时 SQLite KG，再构造四步查询计划和变更提案。

主要流程：

1. 注册 hypothesis context、variant explanation、variant comparison 三个 allow-listed operator；
2. 在 tool-call budget 内执行 context → explain → counterevidence → compare；
3. 验证 evidence pack、counterevidence、依赖关系、stop reason 和确定性 query ID 审计；
4. 验证原始 SQL 参数和越权 variant ID 被拒绝；
5. 验证 proposal 的 dry-run、commit、幂等 duplicate 和非法状态 reject。

输出：`interaction.sqlite`、查询计划/证据包/安全边界/写回状态摘要和 `result.json`。

### 3.7 Scientist、Critic 与审批网关

离线启动命令：

```bash
python scripts/module_tests/test_agents_review.py --config configs/module_tests/agents_review.yaml
```

输入：模拟 benchmark、ensemble prediction、KnowledgeEngine evidence；默认使用确定性 Mock Scientist 和 Rule Critic。

主要流程：

1. Scientist 仅从可见 observation、evidence 和安全 KG 查询生成可证伪假设；
2. 检查 hidden-label sanitizer 和单候选 KG inspection；
3. 首次构造未注册 executable falsification 的 DraftBatch，Critic 返回 REVISE；
4. 第二次冻结 falsification spec，重新 hard validation 后 Critic APPROVE；
5. ApprovalGateway 生成不可变 receipt，后端只接受 ApprovedBatch；
6. CSV oracle 揭示已批准候选，篡改 receipt 被拒绝；
7. DeterministicHypothesisEvaluator 输出 SUPPORTED/CONTRADICTED/INCONCLUSIVE 判定。

输出：`agent_knowledge.sqlite`、假设/审批/揭示/评估/安全守卫摘要和 `result.json`。

### 3.8 闭环 Campaign、评估与报告

启动命令：

```bash
python scripts/module_tests/test_campaign_evaluation.py --config configs/module_tests/campaign_evaluation.yaml
```

输入：脚本生成 legacy public/oracle benchmark；配置提供轮数、预算、候选限制、baseline model、knowledge 和 critic 参数。

主要流程：

1. 分别运行 random、fitness_direct、llm_agent、knowledge_agent 四种模式；
2. 检查 fit → propose → validate → critic → approve → submit → collect → final-test 的完整闭环；
3. 检查批准先于测量、query budget、finalize 和 round artifact；
4. 汇总四种模式的预测/闭环指标；
5. 对 knowledge ablation、score shuffle、evidence deletion 分别运行反事实干预；
6. 计算科学思维行为指标并生成 Markdown 报告。

输出：每次 campaign 的 `config.json`、`trace.jsonl`、`status.json`（当前 phase 覆盖写入）、round artifact、SQLite KG、summary；另有 `aggregate/run_comparison.csv|json`、`scientific_thinking.md` 和模块 `result.json`。闭环脚本会把进度打到 stderr；stdout 仍是最终 JSON。

## 4. LLM API 与模型 checkpoint 占位配置

### 4.1 真实 LLM API

配置位置：`configs/module_tests/agents_review.yaml` 的 `remote_llm`。

必须替换：

```yaml
remote_llm:
  api_key: REPLACE_WITH_REAL_LLM_API_KEY
  base_url: REPLACE_WITH_OPENAI_COMPATIBLE_BASE_URL_OR_DELETE
  scientist_model: REPLACE_WITH_SCIENTIST_MODEL_NAME
  critic_model: REPLACE_WITH_CRITIC_MODEL_NAME
```

若直接使用 OpenAI 官方端点，可将 `base_url` 设为 `null`。替换后执行：

```bash
python scripts/module_tests/test_agents_review.py \
  --config configs/module_tests/agents_review.yaml \
  --enable-remote
```

脚本只在进程内把 key 复制到 `FITNESS_AGENTS_LLM_API_KEY`，不会把真实 key 写入 `result.json`、prompt trace 或 campaign artifact。未替换占位符时启用远程测试会立即失败并指出对应字段。

### 4.2 真实外部预测模型

配置位置：`configs/module_tests/predictive_model.yaml` 的 `real_external_model` 与 `checkpoint_placeholders`。

Kermut 示例要求替换：

- `checkpoint`：模型 checkpoint 的绝对路径；
- `precomputed_features_path`：预计算 embedding/zero-shot NPZ；
- `conditional_probs_path`：ProteinMPNN 条件概率 NPY；
- `coords_path`：C-alpha 坐标 NPY。

替换后把 `real_external_model.enabled` 改为 `true`，再运行 predictive-model 测试。ProteinNPT、ProSST、Pythia-PPI 的 checkpoint 占位路径集中记录在 `checkpoint_placeholders`；接入真实 backend 时还需把对应 `backend_factory` 和 backend-specific options 填入 `real_external_model`。

配置仍含 `REPLACE_WITH_...` 时，脚本不会尝试下载或加载模型，也不会用随机分数冒充真实模型输出。Kermut 真实后端还需安装：

```bash
python -m pip install -e ".[kermut]"
```

## 5. 结果判定与排错

- 退出码 0 且 `result.json.status == "passed"`：该模块功能测试通过；
- 非 0：脚本在首个不满足的功能断言、配置错误或外部依赖错误处停止；
- 总入口的 `summary.json` 记录每个子进程的 return code 和结果路径；
- `scientific_thinking.verdict == "not_supported"` 本身不等于程序崩溃，它表示当前小样本干预没有通过全部行为门槛；模块脚本仍会要求指标完整、rank tracking 完整且所有 campaign 正常结束；
- 重复运行会复用或覆盖 `artifacts/module_tests` 下的测试产物。数据 split writer 自身仍按 manifest/hash 规则验证幂等性。

最后，现有回归测试仍建议同时执行：

```bash
python -m pytest tests -q
python -m ruff check src tests scripts/module_tests
```

