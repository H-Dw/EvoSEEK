# 数据集分割代码模块改进 PLAN

> 目标：把当前 GB1 专用、单次随机抽样的数据准备代码，改造成可配置、可审计、一次生成完整五折、可复用于 GB1 与 FLIP-2 PDZ3 等数据集的通用 split 模块。  
> 首批策略：`AL96 closed-loop`（`al96_closed_loop`）、`FLIP-compatible static OOD`（`flip_static_ood`）、`Mutation-identity OOD`（`mutation_identity_ood`）。  
> 本文只定义代码改进计划；主动学习及消融实验如何使用这些数据，见独立文档《主动学习使用数据划分与消融实验设计》。  
> 编制日期：2026-08-15

## 1. 交付目标与边界

完成后，一个命令必须能够：

1. 读取一个由 dataset spec 描述的序列—fitness 数据集；
2. 规范化序列、WT、突变身份和唯一 `variant_id`，在 split 前完成去重与冲突检查；
3. 通过参数选择三种 split 策略之一或一次生成全部策略；
4. 一次执行输出 `fold_00` 至 `fold_04` 五个完整 fold，而不是只输出一个 seed；
5. 为 agent、训练控制器、query oracle 和 final evaluator 生成不同权限视图；
6. 生成 source、配置、salt commitment、fold assignment 和每个输出文件的 SHA-256 manifest；
7. 自动执行泄露审计，并在任何硬性检查失败时拒绝发布结果；
8. 继续支持当前 GB1 campaign，同时可通过 adapter 应用于 FLIP-2 PDZ3 这类双组分序列数据。

本阶段不负责重新训练模型，也不负责实现新的 acquisition algorithm；但为了让 split 的安全边界真实生效，需要同步改造 loader、oracle backend 和 campaign runner 的数据访问接口。

## 2. 当前实现审计

### 2.1 已有优点

当前 `src/fitness_agents/data/gb1.py` 已经具备：

- canonical `variant_id`；
- public 特征与 oracle 标签分文件；
- 固定随机 seed；
- 对 GB1 四位点的 WT、突变表示和 Hamming depth 计算；
- loader 对 public 表出现 `fitness` 的拒绝检查；
- oracle 对重复查询和 final-test 一次性开启的基础保护。

这些能力应保留，但移入通用接口，不继续散落在 GB1 专用函数中。

### 2.2 必须修正的问题

| 位置 | 当前行为 | 风险或限制 | 计划修正 |
|---|---|---|---|
| `data/gb1.py::_assign_splits` | 全数据按 mutation count 比例随机抽 96/96/2048 | 不符合 WT + 全 singles + 19 doubles 的 AL96；高阶 test 未先锁定 | 替换为策略对象和明确的 role assignment |
| `data/gb1.py` | WT、位置、四字母编码、列名全部硬编码 | 无法复用于 PDZ3、多 WT 或双组分序列 | 移至 dataset adapter/spec |
| `data/loader.py` | 固定要求四个角色及单一 public/oracle 文件 | 无法表示 static OOD、quarantine 和五折 | 改为 manifest 驱动的 `FoldBundle` |
| `loop/backends.py` | 硬编码 `oracle_pool` 和 `final_test` | 不能按策略/阶段限制查询权限 | 使用 capability/allowed-role 配置 |
| `loop/orchestrator.py` | 每一轮都把固定 validation 标签传入 predictor | 多轮反复使用 benchmark validation，形成验证集过拟合 | AL 模式改为 observed-only 校准；static OOD 才使用 fold validation |
| `models/ensemble.py` | validation 用于每轮 conformal radius | 固定验证集信息持续影响每轮预测和 acquisition | 增加 observed-only/OOB 校准模式 |
| 输出 | 只有一个 split，没有 fold manifest | 无法做真正的五折复现与配对比较 | 单次生成五个 fold 及总 manifest |
| 随机性 | `numpy` 按输入行顺序抽样 | 输入排序或 pandas 行号改变会改变结果 | 用 canonical ID + keyed hash 做稳定随机分配 |

## 3. 从 FLIP 与 FLIP-2 借鉴什么

### 3.1 FLIP

[FLIP GB1 生成 notebook（固定 commit）](https://github.com/J-SNACKKB/FLIP/blob/62cace8735f5610e2743cf06ce0f944b37fffaa6/collect_splits/3_gb1.ipynb)采用：

- 按 Hamming depth 定义 `one/two/three-vs-rest`；
- 从语义训练侧再随机抽取 validation；
- 输出统一的 `sequence,target,set,validation`；
- 通过 split semaphore 区分正式、讨论和废弃 split。

需要保留的是“先定义工程分布偏移，再在合法训练域内构造验证集”的顺序，以及兼容导出格式。不能照搬的部分是：

- GB1 `keep` 子集通过 fitness 重平衡产生，属于 label-conditioned population；
- notebook 使用单个 `random_state=11` 和一次 10% validation，不是五折；
- `random.sample`/行号依赖不适合作为版本稳定的 split 标识；
- notebook 和数据路径高度数据集专用。

### 3.2 FLIP-2

[FLIP-2](https://flip.protein.properties/)把工程偏移分为 Number、Position、Mutation、Fitness 和 Wild-type，并延续统一的 `set` 与 `validation` 字段。其 `by-mutation` 明确要求训练和测试使用不同突变，PDZ3 `single-to-double` 则展示了双组分序列和非加和测试集的特殊性。[FLIP-2 论文](https://doi.org/10.64898/2026.02.23.707496)

需要借鉴：

- strategy category 与数据 adapter 分离；
- validation 是 train domain 的子集，而不是把 test 拿来早停；
- 对多 WT、多组分、position/mutation group 的支持；
- 每个 split 记录来源、许可和实际有效 train/validation/test 数量。

需要改进：

- 上游部分处理脚本仍包含绝对路径、逐行赋值及 dataset-specific 逻辑；
- 发布文件只有 `sequence,target,set,validation`，不足以安全重建 mutation identity；
- PDZ3 使用 `PDZ3:CRIPT` 形式的双组分序列，不能用单一字符串位置直接定义 mutation group；
- 上游 `by-mutation` 语义不应被模糊复刻。本项目将发布更严格、可机器验证的“原子突变身份不跨训练/测试”定义，并在 manifest 中标记为本项目协议。

## 4. 总体架构

### 4.1 目录与模块

建议新增：

```text
src/fitness_agents/data/
  specs.py                         # DatasetSpec、字段与参考序列配置
  canonical.py                     # CanonicalDataset：features 与 labels 分离
  adapters/
    base.py                        # DatasetAdapter protocol
    flip_gb1.py                    # GB1 四位点/全长表示规范化
    flip2.py                       # 通用 FLIP-2 CSV adapter
    paired_sequence.py             # PDZ3 等多组分序列解析
  splitting/
    contracts.py                   # SplitRequest、FoldAssignment、SplitResult
    hashing.py                     # keyed SHA-256、salt commitment、稳定排序
    initialization.py              # low-order/coverage initial set
    al96.py                        # GB1-AL96 closed-loop
    flip_ood.py                    # FLIP-compatible static OOD
    mutation_ood.py                # mutation-identity group OOD
    audit.py                       # 完整性、标签盲和 group 泄露审计
    writer.py                      # capability views、manifest、hash
    registry.py                    # strategy registry
scripts/data/build_splits.py       # 唯一 split CLI
configs/data/splits/
  gb1.yaml
  flip2_pdz3.yaml
```

保留 `data/gb1.py::build_gb1_benchmark` 作为一个版本周期的兼容 wrapper，并打印 deprecation warning；wrapper 内部调用新 engine，不能保留旧随机逻辑。

### 4.2 核心数据契约

目标值与可用于分割的元数据必须从对象层面分开：

```python
@dataclass(frozen=True)
class CanonicalDataset:
    features: pd.DataFrame  # 不含 target、原始计数或 target proxy
    labels: pd.DataFrame    # variant_id, target；只交给 writer/oracle/evaluator
    spec: DatasetSpec

@dataclass(frozen=True)
class SplitRequest:
    strategy: str
    n_folds: int
    seed: int
    public_salt: str | None
    options: Mapping[str, object]

@dataclass(frozen=True)
class FoldAssignment:
    variant_id: str
    fold_index: int
    split_role: str
    queryable: bool
    label_visibility: str
```

所有 label-blind strategy 的 `build()` 只接收 `features`。只有显式声明 `requires_labels=True` 的策略才能接收 labels；CLI 默认拒绝此类策略，除非提供 `--allow-label-dependent-membership`。

### 4.3 通用 canonical features

每条变体至少包含：

| 字段 | 含义 |
|---|---|
| `dataset_id` / `assay_id` | 数据集与实验身份 |
| `variant_id` | `assay + backbone + normalized components` 的稳定哈希 |
| `backbone_id` | 单 WT 数据集可固定；多 WT 数据集必须显式提供 |
| `sequence` | 面向模型的规范序列表示 |
| `component_sequences` | JSON/结构化多组分序列；GB1 为一个 component，PDZ3 为 PDZ3 与 CRIPT |
| `mutation_tokens` | 排序后的原子 token 列表 |
| `mutation_count` | token 数量或 dataset spec 指定的工程计数 |
| `mutated_positions` | component-aware 位置集合 |
| `source_row_id` / `source_sha256` | 来源追踪，不进入模型特征 |

原子 mutation token 定义为：

```text
(backbone_id, component_id, position, wild_type_residue, mutant_residue)
```

PDZ3 等数据集必须由 spec 提供经过验证的各 component WT/reference sequence。代码不得通过全数据多数票静默推断 WT，也不得把 `:` 等分隔符计作残基位置。

## 5. 五折的统一原则

“一次执行输出五折”不等于三种策略都用 row-wise `KFold`。每种策略的 fold unit 必须与要测的分布偏移一致：

| strategy | 随机分配单位 | 五折中变化的角色 | 保持不变的角色 |
|---|---|---|---|
| `al96_closed_loop` | 高阶 deployable variant，按非标签 strata 分层 | outer `final_test`；每折的内部 validation | 固定 AL96 initial set |
| `flip_static_ood` | OOD 规则训练域内的 eligible row | 五个 train-side validation fold | 语义 OOD test |
| `mutation_identity_ood` | 原子 mutation identity group | test/validation identity shard | token 定义、WT anchors |

所有 fold assignment 使用 keyed SHA-256，而不是依赖输入行顺序。建议键：

```text
HMAC_SHA256(salt, dataset_version | strategy_version | stratum | canonical_id)
```

公开开发 split 可使用固定 public salt；正式盲测只发布 salt commitment，结束后再公开 salt。`seed`、salt、输入 source hash 和 strategy version 都必须写入 manifest。

## 6. Strategy A：`al96_closed_loop`

### 6.1 Initial set

GB1 默认配置必须精确得到：

- WT 1；
- 全部 HD=1 单点 76；
- 从 HD=2 中用 sequence-only coverage objective 选择 19；
- 合计 `initial_observed=96`，五个 fold 共用同一初始实验集。

19 个双点的选择不得读取 fitness。coverage objective 至少覆盖：位置对、原子 mutation token、每个位点 mutant AA 及氨基酸理化类别；并用 keyed hash 处理并列。

通用模式使用 `--initial-budget 96 --initial-policy low_order_coverage`：按 mutation depth 从低到高加入完整层；下一层超过预算时做 coverage selection。若数据集无 WT、single 数量超过预算或 mutation token 不可用，adapter 必须按 spec 给出明确行为或失败，不能套用 GB1 假设。

### 6.2 Outer test fivefold

GB1 默认只在 HD=3/4 deployable universe 上做 outer fivefold，并按 `mutation_count` 分层：

1. 每个 HD stratum 由 keyed hash 排序；
2. round-robin 分配到五个 test shard；
3. fold `f` 的 `final_test` 为 shard `f`；
4. 五个 `final_test` 两两不相交，合集恰为全部 26,019 个 triples + 121,174 个 quadruples；
5. 每个 HD 在各 fold 的数量差不得超过 1。

GB1 每折 `final_test` 预计为 29,437–29,439 条。它是 `GB1-AL96-5CV`，不是先前 10% sealed test 配置的同义词；manifest 必须明确 `outer_test_fraction≈20%`。

### 6.3 Validation 与 candidate pool

从当前 fold 非 test 的高阶序列中，再用独立 validation salt、完全 label-blind 地抽取 HD3 192 + HD4 192：

- `benchmark_validation=384`；
- 剩余双点、高阶变体进入 `candidate_pool`；
- candidate label 只有被 oracle 接受查询后才揭示；
- `final_test` 永远不进入 queryable ID 集合。

GB1 每折 candidate pool 约 119,442–119,444 条。validation 抽样只按 mutation count、backbone/component 等非标签字段分层，不按 fitness、`keep`、原始计数或已有 split 列分层。

### 6.4 非 GB1 参数化

支持：

```text
--initial-budget 96
--test-depth-min 3
--validation-size 384
--validation-strata mutation_count,backbone_id
--initial-policy low_order_coverage
```

PDZ3 若以公开 734 条 curated split 文件为输入，只能说明是在该 curated universe 内重新划分；不得声称覆盖原研究 200,000+ 个组合。`dataset_scope` 必须写入 manifest。

## 7. Strategy B：`flip_static_ood`

### 7.1 OOD 规则

CLI 提供：

```text
--ood-rule one_vs_rest|two_vs_rest|three_vs_rest|low_vs_high
--population full|flip_keep
```

默认 `population=full`。Number OOD 规则为：

```text
train_domain: mutation_count <= k
test_domain:  mutation_count > k
```

其中 `k=1/2/3`。`sampled` 是 IID sanity check，不属于本策略。`low_vs_high` 与 `flip_keep` 都使用 fitness 决定成员身份，只有显式提供 `--allow-label-dependent-membership` 才能运行，并在 manifest 标记：

```yaml
label_conditioned_population: true
eligible_for_closed_loop: false
eligible_for_clean_label_blind_claim: false
```

### 7.2 五折语义

OOD test 必须在五折中保持完全不变，否则会破坏“低阶训练、高阶测试”的任务定义。五折只作用于 train domain：

1. 从 train domain 中排除配置为 always-train 的 anchors；GB1 默认 WT 为 anchor；
2. 对其余训练域按 mutation count/backbone 做 label-blind 五折；
3. fold `f` 的一个 shard 为 validation，其余四个 shard为 effective train；
4. 每个 eligible training row 恰好在一个 fold 中作为 validation；
5. OOD test 的 variant IDs 与顺序哈希在五折中相同。

这应在命名上写作 `train-domain 5-fold validation + fixed OOD test`，不能误称为五个独立 test fold。

### 7.3 兼容导出

每个 fold 额外输出：

```text
sequence,target,set,validation
```

- effective train：`set=train, validation=False`；
- validation：`set=train, validation=True`；
- OOD test：`set=test, validation=False`。

该文件只用于兼容静态 FLIP 工具，不作为 agent runtime 的安全数据包。

## 8. Strategy C：`mutation_identity_ood`

### 8.1 分组单位

禁止对 variant row 做普通随机 KFold。先从 adapter 得到每个 variant 的 `mutation_tokens`，再把原子 identity 分为五个 shard。为降低某个 fold 只有极少数变体的问题，使用按 component/position 分层、按 token support frequency 做 deterministic greedy bin packing；fitness 不进入平衡目标。

### 8.2 Fold role 规则

对 fold `f`：

- test identities：identity shard `f`；
- validation identities：identity shard `(f + 1) mod 5`；
- train identities：其余三个 shard；
- WT/空 token anchor：始终为 train。

variant 的角色：

| token 构成 | 角色 |
|---|---|
| 不含 test/validation identity | `train_observed` |
| 至少一个 validation identity，且无 test identity | `benchmark_validation` |
| 至少一个 test identity，且无 validation identity | `final_test` |
| 同时含 test 与 validation identity | `quarantine` |

test/validation variant 可以同时含 train identities；本任务测的是“至少出现一个未见原子突变”的泛化。硬性约束是：指定为 test 的 identity 不能出现在 train 或 validation variant 中，指定为 validation 的 identity 不能出现在 train variant 中。

提供可选严格模式：

```text
--mutation-row-policy contains_unseen   # 默认
--mutation-row-policy pure_group_only   # 所有 token 必须来自同一角色，数据会显著减少
--mixed-policy quarantine               # v1 唯一允许值
```

不得通过“test 优先”把 val+test 混合 row 直接塞入 test，因为这样 validation identity 会借 test 路径泄露到评分分析；v1 一律 quarantine 并报告数量。

### 8.3 五折覆盖要求

- 每个 atomic identity 恰好作为 test identity 一次、validation identity 一次；
- 每个 fold 都必须有非空 effective train/validation/test；
- held-out test identity 与 train/validation 使用的 identity 严格隔离；
- 若某个 position/component 的 identity 数不足以支持五折，命令失败并给出位置、数量和建议，而不是退化为 row-wise split。

## 9. CLI 设计

建议入口：

```bash
python scripts/data/build_splits.py \
  --dataset-spec configs/data/splits/gb1.yaml \
  --strategy al96_closed_loop \
  --n-folds 5 \
  --seed 20260815 \
  --output-root data/processed/splits
```

其他示例：

```bash
# FLIP number OOD；固定 OOD test，训练域五折 validation
python scripts/data/build_splits.py \
  --dataset-spec configs/data/splits/gb1.yaml \
  --strategy flip_static_ood \
  --ood-rule two_vs_rest \
  --n-folds 5

# Mutation-identity group OOD
python scripts/data/build_splits.py \
  --dataset-spec configs/data/splits/flip2_pdz3.yaml \
  --strategy mutation_identity_ood \
  --n-folds 5 \
  --mixed-policy quarantine

# 一次生成三种策略；每种策略均有五折
python scripts/data/build_splits.py \
  --dataset-spec configs/data/splits/gb1.yaml \
  --strategy all \
  --n-folds 5
```

参数约束：

- `--n-folds` 默认且正式配置固定为 5；允许其他值仅供开发测试；
- build 命令永远生成全部 fold，不提供只生成单 fold 的选项；
- `--fold-index` 只属于训练/评测命令；
- 已存在输出若 manifest 不同则拒绝覆盖，要求新 `--protocol-version` 或新目录；
- 所有 CLI override 写入 manifest，不能只存在终端历史中。

## 10. 输出目录与权限视图

```text
data/processed/splits/<dataset>/<strategy>/<protocol_version>/
  manifest.public.json
  audit_summary.json
  fold_00/
    fold_manifest.json
    agent/
      initial_or_train_observed.csv.gz
      candidate_pool.csv.gz
    controller/
      benchmark_validation.csv.gz
    oracle/
      queryable_labels.csv.gz
    evaluator/
      final_test_inputs.csv.gz
      final_test_labels.csv.gz
    quarantine/
      excluded_variants.csv.gz
    compat/
      flip.csv.gz                 # 仅 static OOD 生成
  ... fold_01 至 fold_04
```

安全要求：

- `agent/candidate_pool` 没有 target、计数、`keep`、上游 split 或 target-derived 统计；
- AL 的 validation 不挂载给 agent/predictor 进程；
- query oracle 文件不含 final-test IDs；
- final inputs/labels 由 evaluator 单独挂载；
- `manifest.public.json` 只含 salt commitment，不含 private salt；
- 每个文件记录 SHA-256、字节数、行数、schema、角色计数和 mutation-depth/group 审计。

## 11. Loader、oracle 与 runner 的配套改造

### 11.1 Manifest-driven loader

用 `load_fold_bundle(root, fold_index, consumer_role)` 替代把 public/oracle 全表读进同一进程：

```text
consumer_role=agent       -> observed labels + candidate features
consumer_role=controller  -> 再增加 benchmark validation
consumer_role=oracle      -> queryable label map
consumer_role=evaluator   -> final inputs + labels
```

`DatasetBundle` 不再无条件携带 validation observations 和 final-test variants。

### 11.2 Oracle capability

`CsvOracleBackend` 改为从 fold manifest 读取 allowed query roles；提交时同时检查：

- ID 是否属于本 fold、本 strategy 的 queryable pool；
- 是否重复、超预算、已 finalized；
- 是否误用其他 fold、validation、quarantine 或 final-test ID；
- `round_id` 是否严格递增，feedback 是否只在下一决策状态可见。

### 11.3 Validation access

- `al96_closed_loop`：每轮模型拟合与不确定性校准只能使用 `initial_observed + acquired_before_round`；固定 benchmark validation 只供外部控制器在预注册 checkpoint 做方法选择，不传入 acquisition loop；
- `flip_static_ood` / `mutation_identity_ood`：允许静态训练过程使用当前 fold validation 做 early stopping/校准，test 只在模型冻结后评分；
- 若需要 observed-only conformal calibration，从 observed history 内部拆分或使用 OOB/bootstrap，不读取全局 validation。

## 12. 数据泄露防护实现

### 12.1 Canonical 去重

- 主键为 `assay_id + backbone_id + normalized component sequences`，GB1 同时保存 four-site tuple；
- FLIP 四字母表示与 ProteinGym 全长表示必须先映射为同一 canonical genotype；
- 精确重复且 target 一致时按 spec 的 replicate policy 合并；target 冲突超过容忍度时失败并输出冲突报告；
- 不允许同一 canonical variant 在一个 fold 的两个角色中出现。

### 12.2 Label-blind 不变性测试

对三个默认策略执行 target permutation test：保持 features 不变，随机置换或用 sentinel 替换全部 target，fold assignment 必须逐字节相同。`low_vs_high` 和 `flip_keep` 例外，但必须显式声明 label-dependent。

### 12.3 Target proxy 黑名单

agent/public schema 拒绝：

```text
target, fitness, DMS_score, raw_fitness, normalized_fitness,
input_count, selected_count, enrichment, keep,
low_vs_high, top_k, percentile, source_set, source_validation
```

上游 `set/validation` 只可由 compatibility importer 读取；重新生成本项目 split 时不得作为 feature 或分层依据。

### 12.4 Preprocessing 边界

scaler、target transform、feature selection、监督降维、校准器和阈值必须在当前 fold 的合法 observed/train 数据上拟合。split builder 只做非监督 canonicalization；即使输出 audit target 分布，也必须在 assignment 冻结后由只读 evaluator 计算，不能触发重抽。

### 12.5 五折使用边界

五个 fold 是五次独立训练/评测，不是同一 agent 的五个连续 episode。跨 fold 必须清空：模型权重、replay buffer、LLM memory、KG 写回、缓存和已揭示标签。若使用一个 fold 调参并查看其 test，再修改系统，其他 fold 不自动恢复“完全盲测”资格；正式结论仍应预注册并使用独立 sealed benchmark version。

### 12.6 公共数据污染

代码 split 不能消除基础模型曾见过 GB1/FLIP/ProteinGym 的预训练污染。manifest 增加 `contamination_status`、模型版本/知识截止和 canary 字段；该限制与本地 exact-label 隔离分开报告。

## 13. 测试计划

### 13.1 Unit tests

新增 `tests/unit/data/splitting/`：

1. `test_canonical_ids_are_source_representation_invariant`；
2. `test_hash_assignment_is_row_order_invariant`；
3. `test_target_permutation_does_not_change_label_blind_splits`；
4. `test_gb1_al96_initial_is_wt_all_singles_19_doubles`；
5. `test_al96_outer_tests_are_disjoint_and_cover_all_high_order`；
6. `test_al96_validation_and_candidate_roles_are_disjoint`；
7. `test_flip_ood_predicate_and_fixed_test_across_folds`；
8. `test_flip_train_rows_validate_exactly_once`；
9. `test_mutation_identities_do_not_cross_forbidden_boundaries`；
10. `test_mixed_val_test_identity_rows_are_quarantined`；
11. `test_paired_sequence_tokens_are_component_aware`；
12. `test_duplicate_conflicting_targets_fail_closed`；
13. `test_public_views_reject_target_and_proxy_columns`；
14. `test_manifest_hashes_match_all_outputs`。

### 13.2 Golden/reference tests

- 固定 FLIP commit，核对 importer 能正确解释 `set=train, validation=True` 为 effective validation；
- 核对 GB1 源数据 149,361、WT 1、single 76、double 2,091、triple 26,019、quadruple 121,174；
- 核对 FLIP-2 PDZ3 官方文件有效 train/validation/test 为 124/31/579；
- 核对本项目 `flip_static_ood` 兼容导出 schema，但不强求与上游单次随机 validation 成员完全相同。

### 13.3 Integration/leakage tests

- CLI 一次产生且只产生五个 fold；同一配置重跑 byte-identical；
- agent 视图无法打开 controller/oracle/evaluator 文件；
- oracle 拒绝其他 fold、validation、quarantine、final-test、重复和超预算 ID；
- round `r` 的 selection artifact 中没有 round `r` feedback；
- final evaluator 只能打开一次，打开后 oracle 停止查询；
- 五个 campaign 连续运行时验证不存在跨 fold memory/checkpoint 复用；
- sentinel final label 不出现在 trace、prompt、KG、异常栈、配置或缓存。

## 14. 实施阶段

### Phase 0：协议冻结

- 确认三个 strategy 名称、role enum、五折语义和 manifest schema；
- 固定 `GB1-AL96-5CV-v1` 的 public salt 或 salt commitment；
- 收集并校验 GB1、PDZ3 reference/component metadata。

完成标准：协议样例 manifest 通过 schema validation，所有尚未决定的行为均显式报错而非使用隐式默认值。

### Phase 1：Canonical data 与 adapters

- 实现 `DatasetSpec`、features/labels 分离；
- 迁移 GB1 canonicalization；
- 实现 FLIP-2 单组分和 paired-sequence adapter；
- 增加去重、replicate conflict 与 source hash。

完成标准：GB1 和 PDZ3 均能生成无 split 的 canonical dataset；target permutation 不影响 features。

### Phase 2：五折 split engine

- 实现 hashing、initial coverage 和三个 strategy；
- 输出 `FoldAssignment`；
- 先在 synthetic landscape 上验证，再跑 GB1 全量。

完成标准：三种 strategy 各自一次返回五个 fold，并通过第 13.1 节核心不变量测试。

### Phase 3：安全 writer、loader 与 oracle

- 生成 capability views 和 hashes；
- manifest-driven loader；
- role-aware oracle 与 final evaluator；
- 保留旧 loader 的迁移警告。

完成标准：agent 进程不再读取含全部标签的统一 oracle CSV。

### Phase 4：Runner 与验证语义

- AL runner 移除每轮固定 benchmark validation；
- 增加 observed-only calibration；
- static OOD runner 使用 train/validation/test；
- run config 增加 strategy、fold、manifest hash。

完成标准：同一实验模式只能取得其允许的数据 view，现有 campaign integration tests 全部迁移通过。

### Phase 5：CLI、审计和兼容导出

- 完成 `build_splits.py` 与三个 dataset spec 示例；
- 增加 audit report、salt commitment 和 FLIP compatibility CSV；
- 更新 `validate_data.py` 支持整个五折目录。

完成标准：README 中三条命令可从原始数据完整复现输出，第二次执行 hash 完全一致。

### Phase 6：Golden tests 与迁移

- 加入 GB1/FLIP/FLIP-2 golden counts；
- 迁移 tests fixtures、configs 和 run scripts；
- `build_gb1_benchmark` 标为 deprecated；
- 文档说明旧 `validation/oracle_pool` 到新角色的映射。

完成标准：unit、leakage、integration 和一个小规模 e2e 全部通过；旧配置得到明确迁移错误或兼容映射，不发生静默语义变化。

## 15. Definition of Done

以下条件全部满足才算实现完成：

- 三个策略均可由 CLI 参数选择；
- 一次执行为每个选中策略输出五折；
- GB1 AL96 initial 集严格为 1 + 76 + 19；
- AL96 五个 outer final test 覆盖全部高阶序列且两两不重叠；
- FLIP static OOD 的 test 在五折中固定，训练域 validation 完整轮换；
- Mutation OOD 在 identity 层分折，所有 forbidden identity overlap 为 0；
- label-blind 策略通过 target permutation invariance；
- agent、controller、oracle、evaluator 文件与运行权限隔离；
- AL campaign 不再每轮使用固定 benchmark validation；
- 输入行顺序变化不改变 fold；
- manifest 包含 source/config/code/split/output hashes、实际计数和审计结论；
- GB1 与至少一个非 GB1 数据集（PDZ3 synthetic/official curated universe）通过端到端构建与加载测试。

## 参考实现与资料

1. [FLIP 官方仓库](https://github.com/J-SNACKKB/FLIP)
2. [FLIP GB1 split notebook，固定 commit](https://github.com/J-SNACKKB/FLIP/blob/62cace8735f5610e2743cf06ce0f944b37fffaa6/collect_splits/3_gb1.ipynb)
3. [FLIP-2 项目、split 类型与数据格式](https://flip.protein.properties/)
4. [FLIP-2 论文](https://doi.org/10.64898/2026.02.23.707496)
5. [GB1 数据集划分与泄露防护策略](./GB1-训练验证测试集划分与数据泄露防护策略.md)
