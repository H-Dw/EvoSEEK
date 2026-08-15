# Predictor 当前逻辑与外部 fitness 模型接入

## 1. 当前系统是否真的调用 Predictor 打分

是。调用链如下：

1. `CampaignRunner` 在每一轮创建一个新的 predictor；
2. predictor 使用初始实验和之前各轮已揭示的 observation 执行 `fit()`；
3. 调用 `predict(remaining)` 给所有尚未查询的候选生成 `Prediction`；
4. acquisition policy 使用 `fitness_mean`，或 `fitness_mean + beta × fitness_std`，完成最终批次选择；
5. 选中候选经过 oracle/实验后，标签才加入下一轮训练集；
6. 闭环结束时重新拟合 predictor，并在一次性打开的 final test 上计算指标。

`Prediction` 的标准输出包括：`fitness_mean`、`fitness_std`、90% 区间、OOD 分数、各模型分量和模型版本。LLM Agent 不生成数值 fitness，它只能通过假设和证据改变候选 eligibility。

## 2. 原先实际调用的模型

默认配置 `configs/model/baseline.yaml` 使用 `onehot_heterogeneous_ensemble`：

- GB1 四个位点 one-hot 加全部二阶 pairwise 特征；
- 多个 bootstrap Ridge；
- ExtraTrees 的树分组预测；
- 可选 sklearn Gaussian Process；
- 各成员均值作为 `fitness_mean`，成员分歧加 conformal radius 构成 UQ。

原系统已有 `FitnessPredictor` Protocol 和注册表，但只有上述一个内置注册项。现在注册表提供：

- `onehot_heterogeneous_ensemble`
- `kermut`
- `proteinnpt`
- `prosst`
- `pythia_ppi`

后四项是延迟加载的外部模型适配器：只有 config 选中时才加载对应插件和权重。其中
Kermut 已提供真实 backend；其余三项继续保留 backend 接口。

## 3. 通过 config 选择模型

实验配置继续通过 `model_config` 选择一个评分模型：

```yaml
model_config: configs/model/kermut.yaml
```

模型配置的公共字段为：

```yaml
name: kermut
device: cpu                 # 默认，不导入 torch，也不占用 GPU
allow_device_fallback: false
batch_size: 32
backend_factory: fitness_agents.models.backends.kermut:create_backend
checkpoint: null            # Optional local ESM-2 checkpoint
options:
  wild_type_sequence: VDGV
  feature_mode: live_esm2
  conditional_probs_path: /data/SPG1_STRSG_Wu_2016.conditional_probs.npy
  coords_path: /data/SPG1_STRSG_Wu_2016.coords.npy
  resource_positions: [39, 40, 41, 54]
```

GPU 启用方式：

```yaml
device: cuda:0
allow_device_fallback: false
```

也可以使用 `device: auto`，其顺序是 CUDA、MPS、CPU。显式请求 GPU 但不可用时默认报错；只有设置 `allow_device_fallback: true` 才会明确回退 CPU，避免一次大规模实验在不知情的情况下变得极慢。

## 4. 插件 backend 契约

外部仓库通常不是稳定的 Python 库，而且 Kermut、ProteinNPT、ProSST、Pythia-PPI 的 PyTorch、Transformers、PyG 和 CUDA 版本可能冲突。因此核心项目不直接固定这些依赖，而是由轻量 adapter 加载一个模型专属 backend：

```python
from fitness_agents.models import ExternalModelContext


class KermutBackend:
    model_version = "checkpoint-sha"

    def __init__(self, context: ExternalModelContext):
        self.device = context.device
        self.batch_size = context.batch_size
        self.options = context.options

    def fit(self, variants, observations, validation_variants=None,
            validation_observations=None):
        ...

    def predict(self, variants):
        return [
            {
                "variant_id": variant.variant_id,
                "fitness_mean": mean,
                "fitness_std": std,
                "component_scores": {"raw_kermut": mean},
            }
            for variant, mean, std in ...
        ]


def create_backend(context: ExternalModelContext):
    return KermutBackend(context)
```

`backend_factory` 必须采用 `python.module:factory_name` 格式。backend 的 `predict()` 可以返回完整 `Prediction`，也可以返回字典；adapter 会检查候选 ID 完整性、重复、顺序、有限数值、非负标准差和区间方向。

所有 backend 输出必须遵守统一方向：**`fitness_mean` 越大代表预测的 assay fitness 越高**。例如 Pythia-PPI 若输出“正值表示结合变弱”的 ΔΔG，插件必须先变换方向或完成 assay 校准，不能把原始 ΔΔG 直接送入 acquisition。

## 5. 四个模型的具体映射

### Kermut

- 已实现 `fitness_agents.models.backends.kermut:create_backend`，使用固定上游 commit 下经
  MIT 许可适配的 Kermut tokenizer、复合核、Exact GP 和优化器，不是 one-hot 或 sklearn GP
  替代品。上游当前 wheel 不导出 Python 模块，因此核心代码保留了许可证和明确来源。
- `fit()`：读取/生成 ESM-2 mean-pooled embedding 和 masked-marginal zero-shot 分数，读取
  ProteinMPNN 条件氨基酸概率与 C-alpha 坐标，用已揭示 GB1 标签拟合复合核 Exact GP。
- `predict()`：返回 likelihood posterior mean/std，并恢复到原始 assay fitness 标度。
- CPU：可以运行且是默认设置；ESM/ProteinMPNN embedding 会慢，建议预缓存。几千标签以上需要稀疏 GP。
- GPU：主要加速 embedding，设置 `device: cuda:0`。

安装真实后端：

```bash
python -m pip install -e ".[kermut]"
```

有两种特征模式：

```yaml
# 对新序列实时计算；允许开放候选空间，但完整 landscape 很慢
feature_mode: live_esm2
cache_dir: artifacts/model_cache/kermut_esm2

# 使用不可变 NPZ；适合固定 benchmark 和测试
feature_mode: precomputed
precomputed_features_path: /data/gb1_kermut_features.npz
```

预计算 NPZ 必须包含 `variant_ids` 或 `sequences`，以及 `embeddings` 和 `zero_shot`。两种模式
都需要 ProteinMPNN 的 `conditional_probs_path` 和 C-alpha `coords_path`。如果只保存四个 GB1
位点的资源，或候选表本身仅用四位字符串表示 GB1 变体，可用
`resource_positions: [39, 40, 41, 54]` 在四位点表示与完整蛋白资源之间映射。

官方 Kermut GB1 资源的典型配置为：

```yaml
conditional_probs_path: /kermut_data/conditional_probs/ProteinMPNN/SPG1_STRSG_Wu_2016.npy
coords_path: /kermut_data/structures/coords/SPG1_STRSG_Wu_2016.npy
resource_positions: [39, 40, 41, 54]
positions_are_one_indexed: true
```

固定候选池可一次性生成 feature store：

```bash
python scripts/models/build_kermut_feature_store.py \
  --public-csv data/processed/gb1_full_public.csv \
  --output models/kermut/gb1_features.npz \
  --cache-dir artifacts/model_cache/kermut_esm2 \
  --device cpu
```

该 NPZ 只含公开序列衍生特征，不写入 fitness 标签。未知序列空间则保持 `live_esm2`；系统会
按序列哈希缓存 embedding，并按 WT 位点缓存 masked-marginal 结果。CPU 为默认设备，切换到
`device: cuda:0` 即可保留同一接口使用 GPU。

### ProteinNPT

- `fit()`：把已揭示序列及 assay fitness 组成 ProteinNPT 的 target context，执行官方监督训练/适配。
- `predict()`：取目标 property token 的预测并映射到统一 fitness 尺度。
- CPU：技术上可用，但微调大模型通常不实用；建议只用于小型 smoke test。
- GPU：生产运行建议 CUDA；backend 应固定 checkpoint、MSA/context 和训练种子。

### ProSST

- 默认作为 zero-shot prior，`fit()` 可以是 no-op 或只拟合单调校准器。
- `predict()`：生成结构 token 后计算 mutant 相对 WT 的 masked-marginal/log-likelihood 差，再做方向统一。
- CPU：可运行，但对完整 GB1 landscape 批量推理较慢；应缓存 WT、结构 token 和不变 embedding。
- GPU：建议用于大候选池推理。

### Pythia-PPI

- 默认是辅助 PPI ΔΔG scorer，需要 GB1-Fc 复合物结构和明确的链/残基映射。
- `fit()`：预训练模型可 no-op；推荐用 validation 数据拟合方向和单调校准，而不是重新训练整个网络。
- `predict()`：输出校准后的、higher-is-better fitness 分数；原始 ΔΔG保留在 `component_scores`。
- CPU：通常可运行但图构建和批量推理较慢；GPU 由同一 `device` 参数启用。
- 单独将其作为主 predictor 在技术上允许，但不推荐；它没有覆盖 DMS 中表达、折叠和展示效率等因素。

## 6. 推荐启用顺序

1. 继续用 `onehot_heterogeneous_ensemble` 作为可复现基线；
2. 首先实现 Kermut backend，并在相同 split/seed/budget 上比较；
3. ProSST 作为无标签 prior，ProteinNPT 作为监督深度模型消融；
4. Pythia-PPI 先以 `component_scores` 形式完成 GB1 校准，再决定是否参与主排序；
5. 需要多模型同时融合时，新增一个注册的 composite predictor，由它分别调用这些 backend 并输出经过验证集校准的 rank fusion；不要直接平均不同量纲的原始分数。

当前实现已完成**模型选择、插件执行基础设施和真实 Kermut backend**。Kermut 核心实现固定到
上游 commit `7e9e2e62a59773f6cc8291d85e6d6006a41a6862`；ESM-2 权重、ProteinMPNN 条件概率和结构坐标
仍作为模型资源外置。ProteinNPT、ProSST、Pythia-PPI 只保留接口，避免在没有官方权重和明确
输入预处理时返回伪分数。
