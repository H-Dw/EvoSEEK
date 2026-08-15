# fitness-agents

`fitness-agents` 是一个可审计、可消融、接口可替换的虚拟蛋白质定向进化系统。当前 MVP
以 GB1 四位点 IgG-binding landscape 为主任务，在隐藏真实 fitness 的条件下运行
Design → Score → Select → Test → Learn 闭环。

需求定位：项目主目标不是单独建设一个 fitness 知识图谱，而是验证知识增强 Agent 能否在相同
实验预算下提高候选发现效率。知识图谱承担统一的知识、模型证据、历史记忆和审计层；fitness
predictor 仍是数值预测组件，Agent 则基于图谱查询结果生成和修订可检验假设。

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

### 2.1 正式五折数据集拆分

上面的 `prepare_gb1.py` 保留给旧 demo。正式闭环与 OOD 实验使用 manifest-driven split，
一次命令必须生成 `fold_00` 至 `fold_04`，而不是把五个随机 seed 当作五折。

构建 GB1-AL96 closed-loop：

```bash
python scripts/data/build_splits.py \
  --dataset-spec configs/data/splits/gb1.yaml \
  --strategy al96_closed_loop \
  --n-folds 5 \
  --seed 20260815 \
  --protocol-version GB1-AL96-5CV-v1 \
  --output-root data/processed/splits
```

该配置的 96 条初始实验由 WT 1 条、全部单点 76 条和标签盲选择的双点 19 条组成。
HD3/HD4 deployable universe 被分成五个互斥 outer final-test shard；每折还有独立的
benchmark validation 和只可通过 oracle 查询的 candidate pool。

构建 FLIP-compatible static OOD：

```bash
python scripts/data/build_splits.py \
  --dataset-spec configs/data/splits/gb1.yaml \
  --strategy flip_static_ood \
  --ood-rule two_vs_rest \
  --population full \
  --n-folds 5 \
  --protocol-version FLIP-two-vs-rest-5CV-v1
```

构建 Mutation-identity OOD：

```bash
python scripts/data/build_splits.py \
  --dataset-spec configs/data/splits/gb1.yaml \
  --strategy mutation_identity_ood \
  --mutation-row-policy contains_unseen \
  --mixed-policy quarantine \
  --n-folds 5 \
  --protocol-version Mutation-OOD-5CV-v1
```

一次生成三种策略：

```bash
python scripts/data/build_splits.py \
  --dataset-spec configs/data/splits/gb1.yaml \
  --strategy all \
  --n-folds 5 \
  --protocol-version v1
```

检查某一折的 manifest、文件哈希和角色数量：

```bash
python scripts/data/validate_data.py \
  --split-root data/processed/splits/GB1/al96_closed_loop/GB1-AL96-5CV-v1 \
  --fold-index 0
```

输出按能力隔离为 `agent/`、`controller/`、`oracle/` 和 `evaluator/`。candidate 文件没有
target；queryable labels 不含 final-test ID；相同 source/config/code 才允许复用已有目录。
更完整的数据 spec、PDZ3 paired-sequence 配置和 label-dependent split 注意事项见
[`数据集分割与处理使用指南.md`](数据集分割与处理使用指南.md)。

### 2.2 单折闭环与五折 agents 并行启动

标准闭环 task 直接配置 `split_root + fold_index`，不再把 manifest 数据重新拼成旧的
public/oracle 文件：

```yaml
# configs/task/gb1_binding_al96.yaml
split_root: data/processed/splits/GB1/al96_closed_loop/GB1-AL96-5CV-v1
fold_index: 0
expected_split_strategy: al96_closed_loop
expected_protocol_version: GB1-AL96-5CV-v1
```

运行单折：

```bash
python -m fitness_agents.cli \
  configs/experiments/knowledge_agent_al96.yaml \
  --fold-index 0 \
  --seed 11
```

五折应是五个相互隔离的 campaign 进程，而不是同一 agent 的五轮。使用 fold scheduler
同时启动五折，并通过 `--max-parallel` 限制并行 agent 数量：

```bash
python scripts/run_fold_campaigns.py \
  --config configs/experiments/knowledge_agent_al96.yaml \
  --folds all \
  --max-parallel 2 \
  --seed 11
```

常用调度示例：

```bash
# 顺序运行，适合单 GPU 或内存较小的机器
python scripts/run_fold_campaigns.py --folds all --max-parallel 1

# 只运行尚未完成的折或先做小范围验证
python scripts/run_fold_campaigns.py --folds 0,2,4 --max-parallel 2

# 只检查将要启动的命令，不创建 campaign
python scripts/run_fold_campaigns.py --folds all --max-parallel 3 --dry-run

# 为每个 fold 设置最长运行时间；超时记为失败，但其他 fold 继续
python scripts/run_fold_campaigns.py \
  --folds all --max-parallel 2 --timeout-seconds 21600
```

调度器的设计约束：

- 每折通过独立 Python 子进程启动，因此不共享模型权重、随机状态、replay buffer、LLM memory、
  KG 写回或已揭示标签；
- `--max-parallel N` 表示任意时刻最多运行 N 个 fold；单 GPU 通常设为 1，CPU/多 GPU 环境再按
  实际内存和显存提高；
- 所有 fold 默认使用同一 paired seed；fold 由 `--fold-index` 决定，seed 不参与重新拆分数据；
- 单折失败、超时或输出无法解析时会写入失败记录，其他已启动/排队 fold 仍会完成；只要任一折失败，
  调度命令最终返回非零状态；
- 不允许把一个 fold 的 checkpoint、KG 或查询结果作为下一 fold 的初始状态。

每次调度写入 `artifacts/fold-campaigns-<timestamp>/`：

```text
schedule.json                    实际配置、manifest、fold 和启动命令
fold_logs/fold_XX-seed_Y.stdout.log
fold_logs/fold_XX-seed_Y.stderr.log
fold_results.json                每折退出状态、run_id、run_dir 和错误
aggregate/run_comparison.csv     五折逐折指标与 manifest/assignment hash
aggregate/run_comparison.json
report.json                      成功/失败数量及聚合文件位置
```

`run_comparison.csv` 保留每个 fold 的结果，不先把候选行混在一起计算指标。统计分析应以 fold 为
一级重复，使用配对 fold/seed 差值比较不同 agent 或消融条件。

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

`trace.jsonl` 是追加式事件日志；`state.json`、`summary.json`、SQLite KG、图谱边导出和
`knowledge_graph_queries.json` 支持重放与审计。KG 将实验 observation、模型 prediction、
理化/保守性/结构 evidence 与 hypothesis 分开存储，避免把计算输出误写成实验事实。

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
- `KnowledgeGraphTool.hypothesis_context/explain_variant`
- `AcquisitionPolicy.score/select`
- `ExperimentBackend.submit/collect/open_final_test`
- `LLMClient.generate_hypothesis`

特征、predictor、candidate generator 和 acquisition 均提供注册函数。例如：

```python
from fitness_agents.models import register_predictor

register_predictor("my_kermut_adapter", my_predictor_factory)
```

内置 predictor 可直接在 model YAML 中选择：`onehot_heterogeneous_ensemble`、`kermut`、
`proteinnpt`、`prosst`、`pythia_ppi`。其中 Kermut 已接入真实的 ESM-2 + ProteinMPNN +
结构复合核 Exact-GP 后端；后三者保留同一插件契约，未配置 backend 时会显式报错而不是生成伪分数。

### 9.1 安装 Kermut 后端

核心环境不强制安装 PyTorch。需要 Kermut 时安装对应可选依赖：

```bash
python -m pip install -e ".[kermut]"
```

这会安装 PyTorch、GPyTorch 和 `fair-esm`。Kermut 的复合核与 Exact-GP 核心已经包含在项目中，
不需要另外安装上游 Kermut wheel。默认使用 CPU；如需 GPU，应安装与本机 CUDA 匹配的 PyTorch。

Kermut 还需要两个 assay/蛋白特异的外部资源，项目不会用占位数据替代：

- ProteinMPNN 条件氨基酸概率，形状为 `L × 20`；
- 蛋白质 C-alpha 坐标，形状为 `L × 3`。

### 9.2 配置 GB1 fitness 打分

编辑 [`configs/model/kermut.yaml`](configs/model/kermut.yaml)，至少设置两个资源路径：

```yaml
name: kermut
device: cpu
allow_device_fallback: false
batch_size: 8
backend_factory: fitness_agents.models.backends.kermut:create_backend
checkpoint: null  # null 时由 fair-esm 获取 esm2_t33_650M_UR50D

options:
  wild_type_sequence: VDGV
  feature_mode: live_esm2
  esm_model: esm2_t33_650M_UR50D
  esm_representation_layer: 33
  cache_dir: artifacts/model_cache/kermut_esm2

  conditional_probs_path: /path/to/SPG1_STRSG_Wu_2016.conditional_probs.npy
  coords_path: /path/to/SPG1_STRSG_Wu_2016.coords.npy
  resource_positions: [39, 40, 41, 54]
  positions_are_one_indexed: true

  composition: weighted_sum
  learning_rate: 0.1
  n_steps: 150
```

GB1 候选表可以使用 `VDGV` 这样的四位点序列，而结构资源仍可保留完整蛋白长度；
`resource_positions` 会从完整资源中抽取第 39、40、41、54 位。反过来，完整蛋白候选序列也可以
使用仅包含这四个位点的裁剪资源。

在实验 YAML 中选择该模型：

```yaml
model_config: configs/model/kermut.yaml
```

CPU 是默认激活方式。GPU 和显式回退分别配置为：

```yaml
device: cuda:0
allow_device_fallback: false  # GPU 不可用时直接报错
# allow_device_fallback: true # 明确允许回退到 CPU
```

### 9.3 实时序列与固定候选池

开放序列空间使用实时模式。系统会缓存 ESM-2 embedding，并按 WT 位点缓存 masked-marginal：

```yaml
options:
  feature_mode: live_esm2
  cache_dir: artifacts/model_cache/kermut_esm2
```

固定 GB1 benchmark 可先生成不含 fitness 标签的特征文件：

```bash
python scripts/models/build_kermut_feature_store.py \
  --public-csv data/processed/gb1_full_public.csv \
  --output models/kermut/gb1_features.npz \
  --cache-dir artifacts/model_cache/kermut_esm2 \
  --device cpu
```

然后切换为预计算模式：

```yaml
options:
  feature_mode: precomputed
  precomputed_features_path: models/kermut/gb1_features.npz
```

NPZ 必须包含 `variant_ids` 或 `sequences`，以及 `embeddings` 和 `zero_shot`。无论选择哪种
特征模式，都仍然需要 `conditional_probs_path` 和 `coords_path`。如果资源缺失，后端会在加载
650M ESM-2 权重前终止并报告缺少的配置。更完整的插件契约和其他模型接入方法见
[`docs/predictor-plugins.md`](docs/predictor-plugins.md)。

`CampaignRunner` 也允许构造时注入 `ExperimentBackend`、predictor factory 和 `ScientistAgent`，
因此可以把 CSV oracle 换成 LIMS/机器人队列，而不修改闭环状态机。真实实验 backend 必须保证：提交幂等、
QC 状态显式、重复测量保留、失败可重试，并且 final test gate 不可逆。

Knowledge-enhanced Agent 默认通过受控的 `AgentKnowledgeGraphTool` 查询 KG，而不是执行任意 SQL。
`hypothesis_context` 只返回当前轮之前已揭示的 observation，以及明确标记为 prediction/evidence 的
当前轮计算结果；每次查询都会写入 `agent_queries`，从而支持轮次历史、推理追溯和消融。未来的
Mutation Designer 或 Scientific Critic 可复用 `explain_variant` 获取单个候选的序列、预测与证据上下文。

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

### 10.1 Scientific Critic 控制流

所有 campaign 提交现在都经过 `DraftBatch → hard validation → CriticAgent → ApprovedBatch`。
`CampaignRunner` 使用审批网关包装实验后端，因此裸候选 ID、被修改的审批凭证或仍含 hard conflict 的
batch 都无法进入提交路径。`REVISE` 最多执行两次；`REJECT` 与循环耗尽默认中止本轮，也可显式配置
`safe_fallback`，回退批次仍须重新验证和审批。

默认离线配置位于 `configs/critic/scientific_v1.yaml`。要启用独立远程 Critic，可在 experiment YAML
中设置：

```yaml
critic:
  enabled: true
  mode: remote
  provider: openai
  model: gpt-5-mini
  profile: scientific_v1
  max_revision_attempts: 2
  fallback_policy: rule
  on_reject: abort_round
  on_exhausted: abort_round
```

Critic 的审查方法位于
`src/fitness_agents/agents/critic_profiles/scientific_v1/SKILL.md`，使用结构化英文编写；它只定义
evidence audit、epistasis、batch design 和 falsification 四种审查视角。实际权限、schema、循环上限、
审批哈希和提交 gate 均由 Python 代码强制执行。

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

除已明确标注并保留 MIT 许可证的 Kermut 计算核心外，本实现没有直接打包第三方源码。它借鉴了
ALDE 的离散 batch acquisition/UQ 分层、FLIP 的 GB1 数据语义、BioDesignBench 的 typed trace
与 intervention test、Virtual Lab 的 hypothesis/critic 职责，以及 protein-design-mcp 的小工具边界。
详细 commit、许可证和不可直接复制的项目见
[`THIRD_PARTY.md`](THIRD_PARTY.md)。

## 14. 已知限制

- demo 结果是隐藏标签模拟，不是新的湿实验结论；
- 5LDE 位点风险是轻量先验，不替代变体结构预测或自由能计算；
- ensemble uncertainty 必须同时用 coverage/NLL 和闭环 acquisition utility 评估；
- 小样本 KG residue aggregate 可能受上位性混杂，故 fitness 始终绑定完整 variant、assay 和 observation；
- 正式结论应使用 paired seeds、bootstrap 置信区间和多重比较校正。
