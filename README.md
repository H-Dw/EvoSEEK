# fitness-agents

`fitness-agents` 是一个可审计、可消融、接口可替换的虚拟蛋白质定向进化系统。当前 MVP
以 GB1 四位点 IgG-binding landscape 为主任务，在隐藏真实 fitness 的条件下运行
Design → Score → Select → Test → Learn 闭环。

核心原则：LLM 负责提出可检验假设、组织证据和批判决策；专用 fitness 模型负责数值预测；
实验后端只在候选被正式提交后揭示标签。LLM 永远收不到未揭示 oracle 或 final-test 标签。

当前版本特意不包含 EVOLVEpro 的轻量 PLM+RF 架构。默认模型是 CPU 可运行的
one-hot 加性/成对上位性特征 + bootstrap Ridge + ExtraTrees 异质集成，并输出校准区间、
epistemic uncertainty 和 OOD 分数。

## 1. Linux 快速开始

要求：Linux、Python 3.10–3.13、`curl`、`sha256sum`。推荐 Python 3.11。

```bash
git clone <your-repository-url> fitness-agents
cd fitness-agents
bash scripts/setup_linux.sh dev
source .venv/bin/activate
```

也可使用 Conda：

```bash
conda env create -f environment.yml
conda activate fitness-agents
```

检查环境：

```bash
python scripts/check_environment.py
```

## 2. 数据下载与准备

下载脚本固定到已核验的 `J-SNACKKB/FLIP` commit，并检查压缩包 SHA-256；固定地址不可用时会回退到同一仓库的 `main`，但任何回退文件仍必须通过相同校验：

```bash
bash scripts/data/download_flip_gb1.sh
python scripts/data/prepare_gb1.py \
  --source data/raw/flip/gb1/four_mutations_full_data.csv
python scripts/data/validate_data.py
```

若曾使用旧版脚本并看到 GitHub Raw `404`，请确认脚本中的仓库为 `J-SNACKKB/FLIP`，然后强制重新下载：

```bash
grep 'REPOSITORY=' scripts/data/download_flip_gb1.sh
FITNESS_AGENTS_FORCE_DOWNLOAD=1 bash scripts/data/download_flip_gb1.sh
```

成功时会显示 `Verified archive`；期望 SHA-256 为
`85692d808dcd3ae54fa2ac31f4e590858d4582369b6c7b05df299b9b6c383bff`。

生成两套数据：

| 数据 | 划分 | 用途 |
|---|---|---|
| `data/demo/gb1_demo_*` | 64 initial + 32 validation + 352 oracle pool + 64 final | CPU demo、CI、快速消融 |
| `data/processed/gb1_full_*` | 96 initial + 96 validation + 147121 oracle pool + 2048 final | 完整 landscape 实验 |

每套数据都拆成 `*_public.csv` 和 `*_oracle.csv`。public 文件不包含任何 fitness；oracle
文件只能交给 `ExperimentBackend`。这种拆分用于防泄露测试，不是密码学安全边界。

原始 GB1 测量为 CC BY 4.0，FLIP 派生文件及 split 为 AFL-3.0。来源和统计写入
`data/demo/data_manifest.json`。

## 3. 模型与结构资产

默认模型从可见 GB1 标签现场训练，没有外部预训练权重。仍应执行模型准备脚本来生成可追踪
manifest：

```bash
python scripts/models/download_models.py --profile baseline
```

若要准备结构证据所需的 GB1 参考结构：

```bash
python scripts/models/download_models.py --profile structure
# 或：bash scripts/models/download_models.sh --profile all
```

这会下载 RCSB 5LDE、记录来源与 SHA-256。AlphaFold、Boltz、Rosetta、SaProt 或 Kermut
可以在后续通过 `EvidenceProvider` / `FitnessPredictor` 注册表接入。当前 structure 通道是版本化的
5LDE 位点风险先验；它不把 ipTM 等同于结合亲和力或实验 fitness。

## 4. 运行 demo

默认运行 Knowledge-enhanced LLM Agent，离线使用确定性的 `MockScientistLLMClient`：

```bash
python scripts/run_demo.py
```

快速烟雾测试：

```bash
python scripts/run_demo.py --rounds 1 --budget 4
```

指定其他实验配置：

```bash
python scripts/run_demo.py --config configs/experiments/fitness_direct.yaml --seed 42
```

每轮都会在 `artifacts/runs/<run_id>/round_XX/selection.csv` 记录：

- 候选在全部未观测候选中的 predictor mean 排名 `model_rank_all`；
- 候选在全部未观测候选中的 acquisition 排名 `acquisition_rank_all`；
- 候选在 Agent 过滤后集合中的排名 `eligible_rank`；
- mean、uncertainty、knowledge score、证据 ID、假设 ID、干预标签和选择理由。

`trace.jsonl` 是追加式事件日志；`state.json`、`summary.json`、SQLite KG 和图谱边导出支持重放与审计。

## 5. 四种规定 baseline

四种模式共享相同 initial/validation/oracle/final split、查询预算、fitness predictor 和 seed。
主比较均使用 greedy predictor mean，避免把 UQ 策略增益错误归因给 Agent；UCB 单独在消融实验中评估。

| 模式 | 候选生成 | 选择 |
|---|---|---|
| Random | 全部未观测候选 | 随机 |
| Fitness model direct | 全部未观测候选 | predictor top-μ |
| LLM Agent | 可检验假设过滤候选 | 同一 predictor top-μ |
| Knowledge-enhanced LLM Agent | 假设 + 理化/保守/结构/KG | 同一 predictor top-μ + 配置化软证据 |

运行默认五个 paired seeds：

```bash
python scripts/run_baselines.py --seeds 11,22,33,44,55
```

快速验证四条路径：

```bash
python scripts/run_baselines.py --seeds 101 --rounds 1 --budget 4
```

完整 149,361-variant landscape 可使用分批预测和 5,000 条无标签 UCB evidence prefilter：

```bash
python scripts/run_demo.py --config configs/experiments/knowledge_agent_full.yaml
python scripts/run_baselines.py --seeds 11,22,33 \
  --task-config configs/task/gb1_binding_full.yaml
```

比较表写入 `artifacts/baseline-comparison-*/run_comparison.{csv,json}`。

## 6. 模块消融

所有知识证据通道均是独立开关。提供的单因素配置包括：

- `no_physchem`、`no_conservation`、`no_structure`、`no_kg`；
- `no_knowledge`、`no_uq`、`no_llm`。

```bash
python scripts/run_ablation.py --seed 17
```

仅测试选定模块：

```bash
python scripts/run_ablation.py \
  --seed 17 --rounds 1 --budget 4 \
  --ablations no_structure,no_kg,no_uq
```

参考条件使用 UCB；`no_uq` 将 acquisition 改为 greedy。不要把完整笛卡尔积作为默认实验；应先做
单因素，再对最重要的 `LLM × KG × UQ` 做 paired factorial test。

## 7. 科学思维测试

科学思维测试不评价文字“像不像科学家”，而检查 Agent 是否：提出可证伪假设、引用可追踪证据、
记录全局候选排名、根据新观察更新假设，并对因果干预产生可解释的行为变化。

```bash
python scripts/run_scientific_thinking.py --seed 23
```

该命令固定同一 seed、数据和预算，自动运行四个条件：

1. 完整 Knowledge-enhanced Agent；
2. Knowledge ablation；
3. Score-shuffle test：打乱 predictor mean 与候选的对应关系，并在 trace 中显式标记；
4. Evidence deletion test：删除提供给 Agent 和 acquisition 的证据 bundle。

输出为 `artifacts/scientific-thinking-*/report.{json,md}`。判据包括假设可证伪率、假设更新率、
证据引用率、排名记录完整率、三个干预下的 batch Jaccard change，以及 score shuffle 是否在理由中披露。
正面 verdict 只说明此 benchmark 中观察到科学型行为，不等同于证明具有人类科学理解。

## 8. 测试

测试代码按目的拆分：

```bash
bash scripts/tests/run_unit.sh
bash scripts/tests/run_integration.sh
bash scripts/tests/run_leakage.sh
bash scripts/tests/run_e2e.sh
bash scripts/tests/run_all.sh
```

也可直接执行：

```bash
python -m pytest tests -q
python -m pytest tests/unit -q
python -m pytest -m leakage -q
```

当前测试覆盖数据 schema、mutation canonicalization、特征、ensemble/UQ、acquisition、多样性、
知识通道、四 baseline、oracle 单次揭示、final gate、prompt 泄露、全候选排名，以及三类科学思维干预。

## 9. 可插拔接口

核心 Protocol 位于 `src/fitness_agents/contracts/interfaces.py`：

- `FeatureProvider.fit/transform`
- `FitnessPredictor.fit/predict`
- `CandidateGenerator.generate`
- `EvidenceProvider.evaluate`
- `AcquisitionPolicy.score/select`
- `ExperimentBackend.submit/collect/open_final_test`
- `LLMClient.generate_hypothesis`

特征、predictor、candidate generator 和 acquisition 均提供注册函数。例如：

```python
from fitness_agents.models import register_predictor

register_predictor("my_kermut_adapter", my_predictor_factory)
```

`CampaignRunner` 也允许构造时注入 `ExperimentBackend`、predictor factory 和 `ScientistAgent`，
因此可以把 CSV oracle 换成 LIMS/机器人队列，而不修改闭环状态机。真实实验 backend 必须保证：提交幂等、
QC 状态显式、重复测量保留、失败可重试，并且 final test gate 不可逆。

## 10. LLM API

离线 demo 不需要 API key。使用 OpenAI-compatible structured output 时：

```bash
bash scripts/setup_linux.sh llm
export FITNESS_AGENTS_LLM_API_KEY='...'
export FITNESS_AGENTS_LLM_MODEL='gpt-5-mini'
sed 's/llm_provider: mock/llm_provider: openai/' \
  configs/experiments/knowledge_agent.yaml > /tmp/knowledge_agent_openai.yaml
python scripts/run_demo.py --config /tmp/knowledge_agent_openai.yaml
```

API context 仅包含已揭示 observation、当前轮次、上轮假设和带来源 evidence。API key 不写入配置、
prompt 或 trace。LLM 不负责生成数值 fitness，也不能执行任意 shell。

## 11. 项目结构

```text
configs/                  task/model/experiment/knowledge/ablation 配置
data/                     raw、processed、demo 与数据许可说明
src/fitness_agents/
  contracts/              typed schemas 与 Protocol
  data/                    下载后清洗、split、公开/oracle 隔离
  features/                one-hot/pairwise 特征及注册表
  models/                  ensemble、校准、UQ 及 predictor 注册表
  mutation/                全枚举、假设过滤、知识过滤
  acquisition/             Random/Greedy/UCB/Thompson + batch diversity
  knowledge/               理化、保守、结构、observation-centric KG
  agents/                  Mock/remote LLM、hypothesis、critic、sanitizer
  loop/                    状态机、CSV oracle、依赖注入
  evaluation/              prediction/loop/scientific-thinking metrics
  reporting/               baseline、消融与干预报告
scripts/data/              数据下载、准备、验证
scripts/models/            模型/结构资产准备
scripts/run_*.py           demo、四 baseline、消融、科学思维测试
scripts/tests/             分层测试命令
tests/                     unit/integration/leakage/e2e
services/structure/        可选 GPU sidecar 接口约定
```

## 12. Docker 与 CI

```bash
docker build -t fitness-agents:local .
docker run --rm -v "$PWD/artifacts:/workspace/artifacts" fitness-agents:local
# 或
docker compose run --rm fitness-agents
```

GitHub Actions 在 Ubuntu 24.04 上测试 Python 3.10、3.11、3.12 和 3.13，并运行 pytest 与 Ruff。

## 13. 迁移来源与边界

本实现没有直接打包第三方源码。它借鉴了 ALDE 的离散 batch acquisition/UQ 分层、FLIP 的 GB1
数据语义、BioDesignBench 的 typed trace 与 intervention test、Virtual Lab 的 hypothesis/critic
职责，以及 protein-design-mcp 的小工具边界。详细 commit、许可证和不可直接复制的项目见
[`THIRD_PARTY.md`](THIRD_PARTY.md)。

## 14. 已知限制

- demo 结果是隐藏标签模拟，不是新的湿实验结论；
- 5LDE 位点风险是轻量先验，不替代变体结构预测或自由能计算；
- ensemble uncertainty 必须同时用 coverage/NLL 和闭环 acquisition utility 评估；
- 小样本 KG residue aggregate 可能受上位性混杂，故 fitness 始终绑定完整 variant、assay 和 observation；
- 正式结论应使用 paired seeds、bootstrap 置信区间和多重比较校正。
