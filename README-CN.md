# EvoSEEK

`EvoSEEK` 是一个可审计、可消融、接口可替换的虚拟蛋白质定向进化系统。当前 MVP
以 GB1 四位点 IgG-binding landscape 为主任务，在隐藏真实 fitness 的条件下运行
Design → Score → Select → Test → Learn 闭环。

## 1. 环境配置

### 1.1 克隆仓库

```bash
git clone https://github.com/H-Dw/EvoSEEK.git EvoSEEK
cd EvoSEEK
```

### 1.2 使用 Conda 安装

```bash
conda env create -f environment.yml
conda activate EvoSEEK
python scripts/check_environment.py
```

`environment.yml` 固定 Python 3.11，并安装核心依赖（含 `httpx>=0.27,<1`）与可编辑包。远程 LLM、RAG、Kermut、UI
仍需在激活后按第 1.6 节补装 extras。

### 1.3 按用途安装可选 extras

核心环境不强制安装 PyTorch。按实际要跑的脚本再装：

```bash
conda activate EvoSEEK

# 远程 Scientist / Critic（DeepSeek 等 OpenAI 兼容 API）
python -m pip install -e ".[llm]"

# 本地 RAG 向量检索（sentence-transformers）
python -m pip install -e ".[rag]"

# 科学文档解析入库
python -m pip install -e ".[rag-docs]"

# Kermut / ESM-2 fitness 后端（会安装 PyTorch、GPyTorch、fair-esm）
python -m pip install -e ".[kermut]"

# 本地 Gradio 交互界面
python -m pip install -e ".[ui]"
```

常用组合可以一次装齐：

```bash
python -m pip install -e ".[dev,llm,rag]"
```

### 1.4 配置密钥（`.env`）

密钥**不要**写入 YAML，也不要提交到 git。在仓库根目录创建 `.env`（已列入 `.gitignore`）：

```bash
cat > .env <<'EOF'
# Scientist / Critic（默认 DeepSeek V4）
DEEPSEEK_API_KEY='sk-...'

# 若使用阿里云 DashScope / Qwen embedding 与 reranker，取消下一行注释
# DASHSCOPE_API_KEY='sk-...'
EOF
```

运行时会读取项目根目录 `.env`，且**不会覆盖**已经存在的进程环境变量。默认
`llm_agent` / `knowledge_agent*` 实验使用 `api_key: env:DEEPSEEK_API_KEY`。也可用
`FITNESS_AGENTS_LLM_API_KEY` 或 `OPENAI_API_KEY` 作为通用覆盖。

离线复现不需要密钥：把实验 YAML 改回 `llm_provider: mock`，或先跑单元测试。

常用环境变量：

| 变量 | 作用 |
|---|---|
| `DEEPSEEK_API_KEY` | 默认 Scientist / Critic 密钥 |
| `DASHSCOPE_API_KEY` | Qwen embedding / reranker |
| `FITNESS_AGENTS_LLM_API_KEY` | 通用 LLM 密钥覆盖 |
| `FITNESS_AGENTS_LLM_BASE_URL` / `OPENAI_BASE_URL` / `DEEPSEEK_BASE_URL` | API 网关 |
| `FITNESS_AGENTS_LOG_LEVEL` | 进度日志级别（默认 `INFO`） |
| `FITNESS_AGENTS_FORCE_DOWNLOAD` | 设为 `1` 时强制重新下载数据压缩包 |

### 1.5 检查环境是否就绪

```bash
python scripts/check_environment.py
python -c "import fitness_agents; print('import ok')"
python -m pytest tests/unit -q
```

`check_environment.py` 会打印 Python 版本、平台和核心包版本。单元测试不需要 API 密钥，
也不依赖外部数据下载。

数据与模型资产仍按第 2、3 节准备；默认 `python scripts/run_demo.py` 需要 `[llm]`、`[kermut]`、
`.env` 中的 `DEEPSEEK_API_KEY`，以及 Kermut 的条件概率/坐标文件。

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

### 4.1 本地 RAG 向量检索与 KG 物化诊断

默认配置使用英文原子事实语料、FTS5 + BGE dense hybrid 检索。通用 corpus/vector index 位于
`artifacts/local_knowledge/corpus/directed_evolution-v4.sqlite`；GB1 泄漏策略与查询审计单独位于
`artifacts/local_knowledge/overlays/gb1.sqlite`，不会把目标状态写回通用向量库。首次运行前显式
安装依赖并下载固定 revision；campaign runtime 不联网：

```bash
python -m pip install -e ".[rag]"
python scripts/setup_local_rag_models.py --model bge-small-en-v1.5
python -m fitness_agents.cli knowledge index \
  configs/experiments/knowledge_agent.yaml
python -m fitness_agents.cli knowledge inspect \
  configs/experiments/knowledge_agent.yaml

python scripts/rag_diagnostics/simulate_local_rag_to_kg.py \
  --embedding-model models/embeddings/bge-small-en-v1.5 \
  --output-dir artifacts/rag-diagnostics/manual-run \
  --strict

export FITNESS_RAG_TEST_MODEL="$(pwd)/models/embeddings/bge-small-en-v1.5"
python -m pytest -q \
  tests/integration/test_local_rag_real_embedding_to_kg.py
```

诊断会输出 `diagnostic.json`、`summary.md`、真实向量 SQLite 和 structured KG SQLite，并比较
lexical、dense、hybrid 的 gold-query 命中率。`--strict` 还会检查 chunk token budget、模型截断、
embedding 覆盖率、no-answer 阈值以及从既有 lexical 索引启用 dense 时是否完成向量回填。查询和
目标数据库统一为英文；模型实际 tokenizer 控制 chunk 上限，禁止静默截断。

新增外部知识时使用项目 skill 并执行 bundle 校验：

```bash
python \
  skills/ingest-scientific-knowledge/scripts/validate_knowledge_bundle.py \
  resources/local_knowledge/directed_evolution \
  --embedding-model models/embeddings/bge-small-en-v1.5
```

检索 chunk 只生成 `context:<protein>` 的非排序上下文。若以后开启
`contributes_to_selection=true`，还必须同时使用 `calibrated_candidate_projection` 和一个
`status: validated` 的候选级校准文件；draft 示例位于
`configs/knowledge/local_rag_selection.example.yaml`，默认会被拒绝。

### 4.2 API embedding 与 reranker

远程向量化通过独立 YAML 配置，不在仓库中保存密钥。默认示例是 Qwen
`text-embedding-v4`；另有 Jina v5、TEI 托管 BGE-M3/E5，以及 Qwen/Jina/BGE reranker
示例，均位于 `configs/knowledge/api/`。先复制示例、替换 endpoint 中的 workspace/host，
再通过环境变量提供密钥：

```bash
export DASHSCOPE_API_KEY="<YOUR_API_KEY>"

python scripts/rag_api_embeddings.py probe \
  --embedding-config configs/knowledge/api/embedding.default-qwen.example.yaml \
  --prompt "How does epistasis constrain combinatorial mutation design?" \
  --document "Epistatic effects make mutation outcomes depend on genetic background."

python scripts/rag_api_embeddings.py index \
  --experiment-config configs/experiments/knowledge_agent.yaml \
  --embedding-config configs/knowledge/api/embedding.default-qwen.example.yaml \
  --index-path artifacts/local_knowledge/corpus/directed_evolution-qwen-v4.sqlite
```

`probe` 分别调用 query/document 编码，并只输出向量维度、范数、哈希与八维预览；`index`
复用生产解析、原子 chunk、manifest 和 SQLite 写入流程。若需要重排，在两个命令中增加
`--reranker-config configs/knowledge/api/reranker.qwen3.example.yaml`。默认 20 条原子事实
语料仍不启用 reranker；先用项目查询集校准 Recall@K、MRR/nDCG、no-answer 和阈值，再决定上线。

若希望 campaign 直接使用 API，在 knowledge YAML 的 `retrieval` 中配置：

```yaml
embedding_backend: api
embedding_model_path: null
embedding_api_config: configs/knowledge/api/embedding.default-qwen.example.yaml
reranker_backend: api  # 或 none
reranker_api_config: configs/knowledge/api/reranker.qwen3.example.yaml
```

API 返回向量会检查数量、顺序、维度、有限值与零向量，并在本地做 L2 归一化；请求禁止服务端
静默截断。manifest 记录 provider、模型家族、模型/部署版本、endpoint 哈希、task/instruction、
维度和 tokenizer 策略，但不会记录 API key。

第 4.2 节的 `directed_evolution-qwen-v4.sqlite` 只覆盖 directed_evolution 语料，供
`knowledge_agent_qwen_rag` 使用。Hierarchical Scientist 的 RAG 条件需要另一份含 binding
claims 的共享索引，见第 4.3 节。

### 4.3 正式矩阵的共享 Qwen corpus index

并行 RAG worker 只读一份预构建的 Qwen 语料索引，各自写入 per-condition/fold overlay，禁止
边跑边建。密钥从 `.env` 的 `DASHSCOPE_API_KEY` 读取（第 1.7 节）。

Hierarchical Scientist 的 `kg_base_rag` 与 `kg_3features_rag` 要求文件：

`artifacts/local_knowledge/corpus/gb1-reasoning-routes-qwen-v4.sqlite`

该路径由 `configs/knowledge/gb1_reasoning_routes.yaml` 给出，语料同时包含
`resources/local_knowledge/directed_evolution` 与 `resources/local_knowledge/binding` 的英文
claims。启动含 RAG 条件的 `scripts/run_hierarchical_scientist.py` 前必须先构建；缺文件时
调度器会立即退出，12 个 job 都不会启动。

```bash
python -m fitness_agents.cli knowledge index \
  configs/experiments/gb1_reasoning_routes_base.yaml
python -m fitness_agents.cli knowledge inspect \
  configs/experiments/gb1_reasoning_routes_base.yaml
```

`inspect` 应打印 corpus 统计。不要把第 4.2 节的 `directed_evolution-qwen-v4.sqlite` 拷贝或改名
成 `gb1-reasoning-routes-qwen-v4.sqlite`：两份索引的 roots、chunk 与 embedding manifest 不同。

Qwen knowledge-agent AL96（`run_agent_baselines.py --modes knowledge_agent_qwen_rag`）仍使用
第 4.2 节的 `directed_evolution-qwen-v4.sqlite`。若该文件尚不存在，用同一套 DashScope 密钥构建：

```bash
python -m fitness_agents.cli knowledge index \
  configs/experiments/knowledge_agent_qwen_al96.yaml
python -m fitness_agents.cli knowledge inspect \
  configs/experiments/knowledge_agent_qwen_al96.yaml
```

## 5. Baseline

共享相同 initial/validation/oracle/final split、查询预算、fitness predictor 和 seed。
主比较均使用 greedy predictor mean，避免把 UQ 策略增益错误归因给 Agent；UCB 单独在消融实验中评估。

| 模式 | 候选生成 | 选择 |
|---|---|---|
| Random | 全部未观测候选 | 随机 |
| Fitness model direct | 全部未观测候选 | predictor top-μ |

### 5.1 GB1-AL96 并行 baseline（`run_agent_baselines.py`）

`scripts/run_baselines.py` 按 seed 串行跑 demo/full 四模式。正式 AL96 五折闭环请用
`scripts/run_agent_baselines.py`：每个 `(mode, seed, fold)` 是独立进程，可用
`--max-parallel` 并行。先按第 2.1 节生成 `GB1-AL96-5CV-v1`，并安装 `[llm]` 与 `[kermut]`。

`random` / `fitness_direct` 不调用 LLM、不启用 RAG 或 KG 工具。Scientist 类模式需要
`DEEPSEEK_API_KEY`。`knowledge_agent_qwen_rag` 另外需要 `DASHSCOPE_API_KEY` 和第 4.2 节的
Qwen 索引。

检查调度表（不启动 campaign）：

```bash
python scripts/run_agent_baselines.py \
  --preset al96 \
  --modes random,fitness_direct \
  --seeds 42 \
  --folds 0,1,2 \
  --max-parallel 3 \
  --dry-run
```

后台跑 random 与 fitness_direct（6 个 job；`--max-parallel 3` 分两波）：

```bash
nohup python scripts/run_agent_baselines.py \
  --preset al96 \
  --modes random,fitness_direct \
  --seeds 42 \
  --folds 0,1,2 \
  --max-parallel 3 \
  --cuda-devices 0,1,2 \
  > random_fitness_direct_b16.log 2>&1 &
```

`--preset al96` 可用模式：

| `--modes` | 作用 |
|---|---|
| `random` | 在配置的 `candidate_limit` 池内随机选湿实验 batch |
| `fitness_direct` | 同一池内 Kermut greedy |

`--comparison rag`、`agents`、`llm_vs_qwen_rag` 等命名集合见脚本内 `COMPARISON_SETS`。产物写入
`artifacts/agent-baselines-<时间戳>/`（`schedule.json`、`job_logs/`、`report.json`、`aggregate/`）。
`--folds config`（默认）沿用各 YAML 的 `fold_index`；正式三折比较请显式传 `--folds 0,1,2`。

Kermut 配置见 `configs/model/kermut.yaml`（`device: cuda:0`）。请在 conda 环境
`EvoSEEK` 中启动（第 6.1 节）。`--max-parallel 3` 或 `4` 时加上
`--cuda-devices 0,1,2,3`（默认 `auto` 也会按可见卡数分配），让并发 job 各占一张卡。
不要与第 16 节 Hierarchical Scientist 同时打满同一组 GPU。

### 6.1 安装 Kermut 后端

核心环境不强制安装 PyTorch。需要 Kermut 时，先看 `nvidia-smi` 右上角的 **CUDA Version**（这是驱动
最高支持的 toolkit，不是 `nvcc` 版本），再安装匹配的 PyTorch。然后装 GPyTorch 与 `fair-esm`。
Kermut 的复合核与 Exact-GP 核心已经包含在项目中，不需要另外安装上游 Kermut wheel。

若只跑 CPU，或驱动已支持 CUDA 12.4+，才可以直接：

```bash
python -m pip install -e ".[kermut]"
```

`pip install -e ".[kermut]"` 会从 PyPI 拉最新默认 torch（目前为 cu130）。在 CUDA 12.1 驱动上
`torch.cuda.is_available()` 为 false，日志出现 `NVIDIA driver too old`，进程会继续在 CPU 上跑。

Kermut 还需要两个 assay/蛋白特异的外部资源，项目不会用占位数据替代：

- ProteinMPNN 条件氨基酸概率，形状为 `L × 20`；
- 蛋白质 C-alpha 坐标，形状为 `L × 3`。

### 6.2 配置 GB1 fitness 打分

编辑 [`configs/model/kermut.yaml`](configs/model/kermut.yaml)，至少设置两个资源路径：

```yaml
name: kermut
device: cuda:0
allow_device_fallback: false
batch_size: 8
backend_factory: fitness_agents.models.backends.kermut:create_backend
checkpoint: ~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt

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

本仓库 `configs/model/kermut.yaml` 已按本机 GPU 设为 `device: cuda:0`。GB1 候选是 265 残基
FLIP fusion，ESM-2 650M 在独占 24GB 3090 上可以把 `batch_size` 提到 16–32；多进程或与其它作业
共享显存时保持 8。`allow_device_fallback: false` 表示 GPU 不可用时直接报错，避免再静默落到 CPU。

```yaml
device: cuda:0
allow_device_fallback: false  # GPU 不可用时直接报错
# allow_device_fallback: true # 明确允许回退到 CPU
batch_size: 8                 # 共享 3090 时的安全值；独占时可改为 16 或 32
```

不要把 YAML 改成 `cuda:0` / `cuda:1` / `cuda:2` / `cuda:3` 来做四卡并行：所有 worker 读同一份
配置，会一起挤在 `cuda:0` 上。正确做法是保持 `device: cuda:0`，由调度器给每个子进程设置
`CUDA_VISIBLE_DEVICES`，让该进程只看见一张卡（在进程内仍叫 `cuda:0`）。

`scripts/run_hierarchical_scientist.py` 与 `scripts/run_agent_baselines.py` 提供
`--cuda-devices`：

| 值 | 行为 |
|---|---|
| `auto`（默认） | 发现可见 GPU；`--max-parallel` 不能超过卡数 |
| `0,1,2,3` | 四卡池，并发 job 各占一张 |
| `none` | 不隔离，所有 worker 继承父进程设备（多进程会争用 GPU 0） |

```bash
conda activate EvoSEEK
python scripts/run_hierarchical_scientist.py \
  --max-parallel 4 \
  --cuda-devices 0,1,2,3 \
  --dry-run
```

单 GPU 或显存紧张时用 `--max-parallel 1`。卡数少于并行度时调度器会直接退出，而不是让两份
ESM-2 650M 挤在同一张 3090 上。

### 6.3 实时序列与固定候选池

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

## 7. 项目结构

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
scripts/run_*.py           demo、四 baseline、AL96 并行 baseline、Hierarchical Scientist、消融、科学思维测试
scripts/tests/             分层测试命令
tests/                     unit/integration/leakage/e2e
services/structure/        可选 GPU sidecar 接口约定
```

## 8. 已知限制

- demo 结果是隐藏标签模拟，不是新的湿实验结论；
- 5LDE 位点风险是轻量先验，不替代变体结构预测或自由能计算；
- ensemble uncertainty 必须同时用 coverage/NLL 和闭环 acquisition utility 评估；
- 小样本 KG residue aggregate 可能受上位性混杂，故 fitness 始终绑定完整 variant、assay 和 observation；
- 正式结论应使用 paired seeds、bootstrap 置信区间和多重比较校正。

## 9. Hierarchical Scientist 正式矩阵

`scripts/run_hierarchical_scientist.py` 在 GB1-AL96 前三折上跑四组条件。Scientist / Critic
走 DeepSeek；RAG embedding / reranker 走 Qwen；fitness 为 Kermut。Agent-UQ 条件不把 fitness
混进 acquisition；`kg_base_al` 使用显式 Kermut posterior。`--placeholder-predictor` 会被拒绝。

| 条件 | 层级 | 文档 RAG | 三通道特征工具 | KG | 采集 |
|---|---|---|---|---|---|
| `kg_base` | 否 | 否 | 否 | 基础 KG | Agent-UQ |
| `kg_base_rag` | 否 | 是 | 否 | 基础 KG | Agent-UQ |
| `kg_base_al` | 否 | 否 | 否 | 基础 KG | Kermut active learning |
| `kg_3features_rag` | 是（三路子 Scientist） | 是 | physchem / conservation / structure | 基础 KG | Agent-UQ |
| `kg_3features_base`（可选） | 是（三路子 Scientist） | 否 | physchem / conservation / structure | 基础 KG | Agent-UQ |
| `agent_only`（可选） | 否 | 否 | 否 | 完全关闭（knowledge runtime 消融） | Agent-UQ |

`kg_3features_base` 与 `agent_only` 不在默认 12-job 矩阵中，通过 `--conditions`
显式调度，例如 `--conditions kg_3features_base,agent_only`。`agent_only` 关闭
`knowledge_enabled` / `knowledge.kg` / `kg_interaction.enabled`，Scientist 不挂任何
KG 工具，纯靠 LLM 做多轮假设-选择迭代；审计会校验 KG runtime 未泄露。

只检查调度表：

```bash
python scripts/run_hierarchical_scientist.py \
  --config configs/experiments/hierarchical_scientist.deepseek.yaml \
  --conditions kg_base,kg_base_rag,kg_base_al,kg_3features_rag \
  --folds 0,1,2 \
  --max-parallel 4 \
  --cuda-devices 0,1,2,3 \
  --dry-run
```

正式 12-job 矩阵（4 条件 × 3 折）。默认 `--max-parallel 4` 分三波，每波恰好一个
`kg_3features_rag`（内部再扇出三路子 Scientist）。DeepSeek 争用时改用 `--max-parallel 2`。
四张 3090 上同时跑 Kermut 时加 `--cuda-devices 0,1,2,3`（默认 `auto` 等价于发现全部可见卡）；
YAML 保持 `device: cuda:0`，由调度器按 job 设置 `CUDA_VISIBLE_DEVICES`。

```bash
conda activate EvoSEEK
nohup python scripts/run_hierarchical_scientist.py \
  --config configs/experiments/hierarchical_scientist.deepseek.yaml \
  --conditions kg_base,kg_base_rag,kg_base_al,kg_3features_rag \
  --folds 0,1,2 \
  --max-parallel 4 \
  --cuda-devices 0,1,2,3 \
  > hierarchical_scientist.log 2>&1 &
```

产物写入 `artifacts/hierarchical-scientist-<时间戳>/`（`schedule.json`、`fold_logs/`、
`report.json`、`aggregate/`）。总控看 `hierarchical_scientist.log`；单 job 看
`fold_logs/<condition>-fXX-sYY.stderr.log`。

只跑不依赖 RAG 的两组时可以暂缓第 4.3 节索引：

```bash
nohup python scripts/run_hierarchical_scientist.py \
  --conditions kg_base,kg_base_al \
  --folds 0,1,2 \
  --max-parallel 2 \
  > hierarchical_scientist_base.log 2>&1 &
```
