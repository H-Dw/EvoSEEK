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

## 1. Linux 本地环境配置

本节说明如何在 Linux 上从源码部署并运行本仓库脚本。GitHub Actions 在 Ubuntu 24.04 上测试
Python 3.10–3.13；Debian/Ubuntu 可直接按下面的 `apt` 步骤操作。推荐 Python 3.11。

不需要 GPU 即可运行默认 CPU ensemble、单元测试和 mock LLM 路径。默认 `run_demo.py`
（DeepSeek + Kermut）还需要第 1.6 节的可选 extras、第 1.7 节的 `.env`，以及第 2、3、9 节的
数据与模型资源。

### 1.1 系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux（推荐 Ubuntu 22.04/24.04 或同等发行版） |
| Python | 3.10–3.13，推荐 3.11 |
| 系统工具 | `git`、`curl`、`sha256sum`、`python3-venv` |
| 编译工具 | `build-essential` 与 `python3-dev`（缺少预编译 wheel 时需要） |
| 磁盘 | 代码本身很小；完整 GB1 landscape、ESM-2 650M 与本地 RAG 模型会各占数 GB |

### 1.2 安装系统软件包

Debian / Ubuntu：

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl ca-certificates \
  python3 python3-venv python3-pip python3-dev \
  build-essential
python3 --version   # 应为 3.10–3.13
```

若系统默认 Python 不在 3.10–3.13，请安装对应版本（例如 `python3.11`），并用
`PYTHON_BIN=python3.11` 调用第 1.4 节的安装脚本。

Fedora / RHEL：

```bash
sudo dnf install -y git curl ca-certificates python3 python3-pip python3-devel gcc gcc-c++ make
```

### 1.3 克隆仓库

```bash
git clone <your-repository-url> fitness-agents
cd fitness-agents
```

### 1.4 创建虚拟环境并安装 Python 依赖（推荐）

一键创建 `.venv`、安装开发依赖并自检：

```bash
bash scripts/setup_linux.sh dev
source .venv/bin/activate
```

`scripts/setup_linux.sh` 会在仓库根目录创建 `.venv`，升级 pip，按 profile 安装依赖，然后运行
`scripts/check_environment.py`。可用 profile：

| Profile | 安装内容 | 适用场景 |
|---|---|---|
| `base` | 核心运行时（numpy / pandas / scikit-learn / scipy / PyYAML / pydantic / httpx） | 仅跑 CPU ensemble 与 mock LLM |
| `dev`（默认） | `base` + pytest、ruff | 本地开发、CI、单元测试 |
| `llm` | `dev` + `openai` | 调用 DeepSeek 或 OpenAI 兼容的远程 Scientist / Critic |

指定解释器：

```bash
PYTHON_BIN=python3.11 bash scripts/setup_linux.sh llm
source .venv/bin/activate
```

等价的手动步骤：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

也可用 Make：`make setup`（等同 `dev` profile）。

每次新开 shell 都需要重新激活：

```bash
cd /path/to/fitness-agents
source .venv/bin/activate
```

激活后，下文所有 `python` / `pip` 命令都使用该虚拟环境。未激活时可显式调用
`.venv/bin/python`。

### 1.5 使用 Conda（可选）

```bash
conda env create -f environment.yml
conda activate fitness-agents
python scripts/check_environment.py
```

`environment.yml` 固定 Python 3.11，并安装核心依赖（含 `httpx>=0.27,<1`）与可编辑包。远程 LLM、RAG、Kermut、UI
仍需在激活后按第 1.6 节补装 extras。

### 1.6 按用途安装可选 extras

核心环境不强制安装 PyTorch。按实际要跑的脚本再装：

```bash
source .venv/bin/activate

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

Kermut 默认走 CPU。若要用 GPU，**不要**直接 `pip install -e ".[kermut]"`：当前 PyPI 默认
torch 是 CUDA 13.0 wheel，在 CUDA 12.1 驱动上会报 `NVIDIA driver too old` 并无法调用 GPU。
应先安装与 `nvidia-smi` 中 CUDA Version 匹配的 PyTorch，再装其余 kermut 依赖；步骤见第 9.1 节。

### 1.7 配置密钥（`.env`）

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

离线复现不需要密钥：把实验 YAML 改回 `llm_provider: mock`（见第 10 节），或先跑单元测试。

常用环境变量：

| 变量 | 作用 |
|---|---|
| `DEEPSEEK_API_KEY` | 默认 Scientist / Critic 密钥 |
| `DASHSCOPE_API_KEY` | Qwen embedding / reranker |
| `FITNESS_AGENTS_LLM_API_KEY` | 通用 LLM 密钥覆盖 |
| `FITNESS_AGENTS_LLM_BASE_URL` / `OPENAI_BASE_URL` / `DEEPSEEK_BASE_URL` | API 网关 |
| `FITNESS_AGENTS_LOG_LEVEL` | 进度日志级别（默认 `INFO`） |
| `FITNESS_AGENTS_FORCE_DOWNLOAD` | 设为 `1` 时强制重新下载数据压缩包 |

### 1.8 检查环境是否就绪

```bash
python scripts/check_environment.py
python -c "import fitness_agents; print('import ok')"
python -m pytest tests/unit -q
```

`check_environment.py` 会打印 Python 版本、平台和核心包版本。单元测试不需要 API 密钥，
也不依赖外部数据下载。

Docker 替代方案见第 12 节。数据与模型资产仍按第 2、3 节准备；准备完成后可用第 4 节的
demo 命令做端到端验证。默认 `python scripts/run_demo.py` 需要 `[llm]`、`[kermut]`、
`.env` 中的 `DEEPSEEK_API_KEY`，以及 Kermut 的条件概率/坐标文件（第 9.2 节）。

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

默认运行 Knowledge-enhanced LLM Agent。实验配置已接到 DeepSeek-v4-flash
（`configs/llm/deepseek.yaml`，密钥只从 `.env` 的 `DEEPSEEK_API_KEY` 读取）和
KERMUT（`configs/model/kermut.yaml`）：

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

`trace.jsonl` 是追加式事件日志；`status.json` 覆盖写入当前 phase / 轮次 / 耗时，便于 `watch`
或 `tail` 查看卡在哪一步。长步骤（模型 fit/predict、Scientist/Critic LLM）还会把进度打到
**stderr**（`[fitness-agents] ...`），stdout 仍只输出最终 JSON。可用 `--quiet` 或
`FITNESS_AGENTS_LOG_LEVEL=WARNING` 关闭进度行。`state.json`、`summary.json`、SQLite KG、图谱边导出和
`knowledge_graph_queries.json` 支持重放与审计。KG 将实验 observation、模型 prediction、
理化/保守性/结构 evidence 与 hypothesis 分开存储，避免把计算输出误写成实验事实。

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

完整 149,361-variant landscape 可使用分批预测和 5,000 条无标签 UCB evidence prefilter。
四种 baseline 实验 YAML 已指向 KERMUT；LLM/Knowledge 模式同时使用 DeepSeek：

```bash
python scripts/run_demo.py --config configs/experiments/knowledge_agent_full.yaml
python scripts/run_baselines.py --seeds 11,22,33 \
  --task-config configs/task/gb1_binding_full.yaml
```

比较表写入 `artifacts/baseline-comparison-*/run_comparison.{csv,json}`。

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

默认 `--modes` 为 `random,fitness_direct,knowledge_agent`。加上 Knowledge Agent：

```bash
nohup python scripts/run_agent_baselines.py \
  --preset al96 \
  --modes random,fitness_direct,knowledge_agent \
  --seeds 42 \
  --folds 0,1,2 \
  --max-parallel 3 \
  --cuda-devices 0,1,2 \
  > agent_baselines.log 2>&1 &
```

`--preset al96` 可用模式：

| `--modes` | 作用 |
|---|---|
| `random` | 在配置的 `candidate_limit` 池内随机选湿实验 batch |
| `fitness_direct` | 同一池内 Kermut greedy |
| `llm_agent` | DeepSeek Scientist，无 KG / RAG |
| `knowledge_agent` | observation KG，无文档 RAG |
| `knowledge_agent_rag` | 本地 BGE RAG（第 4.1 节） |
| `knowledge_agent_qwen_rag` | Qwen embedding / rerank RAG（第 4.2 节索引） |

`--comparison rag`、`agents`、`llm_vs_qwen_rag` 等命名集合见脚本内 `COMPARISON_SETS`。产物写入
`artifacts/agent-baselines-<时间戳>/`（`schedule.json`、`job_logs/`、`report.json`、`aggregate/`）。
`--folds config`（默认）沿用各 YAML 的 `fold_index`；正式三折比较请显式传 `--folds 0,1,2`。

Kermut 配置见 `configs/model/kermut.yaml`（`device: cuda:0`）。请在 conda 环境
`fitness-agents` 中启动（第 9.1 节）。`--max-parallel 3` 或 `4` 时加上
`--cuda-devices 0,1,2,3`（默认 `auto` 也会按可见卡数分配），让并发 job 各占一张卡。
不要与第 16 节 Hierarchical Scientist 同时打满同一组 GPU。

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

核心环境不强制安装 PyTorch。需要 Kermut 时，先看 `nvidia-smi` 右上角的 **CUDA Version**（这是驱动
最高支持的 toolkit，不是 `nvcc` 版本），再安装匹配的 PyTorch。然后装 GPyTorch 与 `fair-esm`。
Kermut 的复合核与 Exact-GP 核心已经包含在项目中，不需要另外安装上游 Kermut wheel。

本机当前是驱动 530.41.03 / CUDA 12.1、4× RTX 3090。请使用 `fitness-agents` conda 环境，**不要**
用仓库 `.venv`（其中是 `torch 2.13+cu130`，GPU 不可用）。cu121 官方 wheel 的最高版本是 2.5.1：

```bash
conda activate fitness-agents
cd /path/to/fitness-agents

python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.5.1" \
  "gpytorch>=1.11,<2" \
  "fair-esm>=2.0,<3"

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 期望：2.5.1+cu121  12.1  True
```

若只跑 CPU，或驱动已支持 CUDA 12.4+，才可以直接：

```bash
python -m pip install -e ".[kermut]"
```

`pip install -e ".[kermut]"` 会从 PyPI 拉最新默认 torch（目前为 cu130）。在 CUDA 12.1 驱动上
`torch.cuda.is_available()` 为 false，日志出现 `NVIDIA driver too old`，进程会继续在 CPU 上跑。

Kermut 还需要两个 assay/蛋白特异的外部资源，项目不会用占位数据替代：

- ProteinMPNN 条件氨基酸概率，形状为 `L × 20`；
- 蛋白质 C-alpha 坐标，形状为 `L × 3`。

### 9.2 配置 GB1 fitness 打分

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

四卡同时跑 4 个 Kermut 进程（conda 环境 `fitness-agents`，不要用 `.venv` 的 cu130 torch）：

```bash
conda activate fitness-agents
python scripts/run_hierarchical_scientist.py \
  --max-parallel 4 \
  --cuda-devices 0,1,2,3 \
  --dry-run
```

单 GPU 或显存紧张时用 `--max-parallel 1`。卡数少于并行度时调度器会直接退出，而不是让两份
ESM-2 650M 挤在同一张 3090 上。

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

Scientist / Critic 的实验配置在 `configs/llm/deepseek.yaml` 与
`configs/critic/deepseek_remote.yaml`。密钥**不要**写进 YAML，只放在 gitignored 的 `.env`：

```bash
# .env
DEEPSEEK_API_KEY='...'
```

`llm_agent` 与 `knowledge_agent*` 实验会读取 `api_key: env:DEEPSEEK_API_KEY`。
`random` 与 `fitness_direct` 仍使用 mock LLM，但共享同一 KERMUT 预测器。

离线复现可把实验 YAML 改回：

```yaml
llm_provider: mock
model_config: configs/model/baseline.yaml
```

并删掉 `llm_config` / `critic_config`。API context 仅包含已揭示 observation、当前轮次、上轮假设和带来源 evidence。API key 不写入 prompt 或 trace。LLM 不负责生成数值 fitness，也不能执行任意 shell。

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
scripts/run_*.py           demo、四 baseline、AL96 并行 baseline、Hierarchical Scientist、消融、科学思维测试
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


## 15. RAG/KG/三通道/主动学习路线验证

路线矩阵位于 `configs/experiments/gb1_reasoning_routes.matrix.yaml`，公共实验参数位于
`configs/experiments/gb1_reasoning_routes_base.yaml`。矩阵固定使用五折中的前三折
`[0, 1, 2]`，并行进程数默认是 2。所有路线共享同一 fold manifest、seed、候选预算、
Scientist LLM 和本地规则 Critic，避免把资源差异误当成路线差异。

| route ID | RAG | physchem | conservation | structure | 主动学习 | 测试目标 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `rag_kg_none` | ✓ |  |  |  |  | RAG 检索和 observation-KG 能进入 LLM，上述三类 feature evidence 均不出现 |
| `rag_kg_all` | ✓ | ✓ | ✓ | ✓ |  | 三类 provider 均 ready、联合 tool 被调用、evidence/KG/RAG 进入 LLM 上下文 |
| `rag_kg_physchem` | ✓ | ✓ |  |  |  | 只出现理化 evidence，排除 MSA 与结构 evidence |
| `rag_kg_physchem_structure` | ✓ | ✓ |  | ✓ |  | 理化与坐标结构联合出现，MSA evidence 缺席 |
| `rag_kg_physchem_conservation` | ✓ | ✓ | ✓ |  |  | 理化与 A3M 单位点保守性联合出现，结构 evidence 缺席 |
| `kg_all` |  | ✓ | ✓ | ✓ |  | 在没有本地文档 RAG 时验证 KG 与三通道到 LLM 的路径 |
| `kg_none` |  |  |  |  |  | 纯 observation-centric KG 到 LLM 的对照路线 |
| `rag_kg_all_active_learning` | ✓ | ✓ | ✓ | ✓ | ✓ | 除完整 evidence 路径外，验证 posterior 与 acquisition 产物及 selection driver |

首次运行前先按第 2.1 节生成 `GB1-AL96-5CV-v1`。检查 24 个任务及有效配置而不调用 LLM：

```bash
python scripts/run_reasoning_routes.py --dry-run
```

执行全部 8 条路线 × 前 3 折；`--max-parallel` 可覆盖默认值 2：

```bash
python scripts/run_reasoning_routes.py --max-parallel 2
```

只验证一条或若干路线：

```bash
python scripts/run_reasoning_routes.py \
  --routes rag_kg_all,kg_all,rag_kg_all_active_learning \
  --folds 0,1,2 \
  --max-parallel 2
```

Linux 已 `source .venv/bin/activate` 后直接使用 `python`。Windows PowerShell 使用仓库虚拟环境时，
将 `python` 换为 `.\.venv\Scripts\python.exe`。DeepSeek key 只通过 `DEEPSEEK_API_KEY` 环境变量提供。

调度器不仅检查进程退出码，还逐折审计：effective config、provider ready/disabled 状态、
`evidence_contract.json` 的通道集合、`structured_kg.sqlite`、`kg_interaction.json`、RAG 产物、
LLM hypothesis、主动学习 posterior/acquisition，以及 `kg_truncation_audit.json` 中对
`MutationEffectEstimate`、关系类型和已启用 feature 关系的关键词查询。汇总写入
`route_validation.json`；其中 `any_max_rows_truncation` 显式报告 `max_rows` 截断，而不是静默忽略。

该测试能证明“组件执行、evidence 生成、KG 注入、上下文交付和选择器切换”。它不能单凭一次
trace 证明 LLM 在语义上依赖某条 evidence，也不能证明某路线提高 fitness。后者应在相同 fold/seed
上增加 evidence 删除或置换干预、多个 seed 和置信区间；三通道与未校准 RAG evidence 默认不直接
升级为测量或 fitness 选择证据。

## 16. Hierarchical Scientist 正式矩阵

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

前置：第 2.1 节 split、第 1.7 节 `DEEPSEEK_API_KEY` 与 `DASHSCOPE_API_KEY`、第 9.2 节 Kermut
资源。只要 `--conditions` 含 RAG 项，必须先完成第 4.3 节共享 Qwen 索引，否则调度器 fail-closed。

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
YAML 保持 `device: cuda:0`，由调度器按 job 设置 `CUDA_VISIBLE_DEVICES`。必须用 conda 环境
`fitness-agents`，不要用 `.venv`。

```bash
conda activate fitness-agents
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
