# Ablation and runtime-audit tables

三折 fold-aligned 差值，均值 ± 样本标准差；括号内为正/持平/负 folds。

## Module comparisons

| Comparison | Final best-seen Δ | AULC Δ | R3 mean Δ | R3 median Δ |
|---|---:|---:|---:|---:|
| KG base − Agent only | +0.940 ± 0.524 (3/0/0) | +0.500 ± 0.151 (3/0/0) | -0.502 ± 0.479 (1/0/2) | -1.005 ± 1.207 (1/0/2) |
| 3-channel − KG base | +0.264 ± 1.068 (2/0/1) | +0.218 ± 0.341 (2/0/1) | +0.060 ± 0.414 (2/0/1) | +0.101 ± 0.924 (2/0/1) |
| KG + RAG − KG base | -0.317 ± 0.550 (0/2/1) | +0.123 ± 0.118 (2/1/0) | +0.200 ± 0.457 (2/0/1) | +0.568 ± 0.762 (2/0/1) |
| 3-channel + RAG − KG + RAG | +0.000 ± 0.000 (0/3/0) | -0.056 ± 0.153 (1/1/1) | +0.001 ± 0.197 (2/0/1) | +0.035 ± 0.157 (2/0/1) |
| 3-channel + RAG − 3-channel | -0.582 ± 0.534 (0/1/2) | -0.151 ± 0.569 (1/1/1) | +0.141 ± 0.704 (2/0/1) | +0.502 ± 1.474 (2/0/1) |
| KG + active learning − KG base | -0.266 ± 0.599 (1/1/1) | +0.152 ± 0.111 (3/0/0) | +0.725 ± 0.382 (3/0/0) | +1.472 ± 0.553 (3/0/0) |

## Feature-by-RAG interaction

| Metric | Feature effect without RAG | Feature effect with RAG | Interaction |
|---|---:|---:|---:|
| Final best-seen Δ | +0.264 ± 1.068 | +0.000 ± 0.000 | -0.264 ± 1.068 |
| AULC Δ | +0.218 ± 0.341 | -0.056 ± 0.153 | -0.275 ± 0.487 |
| R3 mean Δ | +0.060 ± 0.414 | +0.001 ± 0.197 | -0.059 ± 0.534 |
| R3 median Δ | +0.101 ± 0.924 | +0.035 ± 0.157 | -0.065 ± 0.880 |

## Three-channel execution audit

| Condition | Channel | Approved paths | Covered samples | Findings | Candidate hypotheses |
|---|---|---:|---:|---:|---:|
| kg_3features_base | conservation | 9/9 | 90 | 71 | 0 |
| kg_3features_base | physchem | 9/9 | 90 | 67 | 0 |
| kg_3features_base | structure | 9/9 | 90 | 70 | 0 |
| kg_3features_rag | conservation | 9/9 | 87 | 69 | 0 |
| kg_3features_rag | physchem | 9/9 | 87 | 68 | 0 |
| kg_3features_rag | structure | 9/9 | 87 | 68 | 0 |

## New-condition runtime boundary

| Condition | KG enabled | Hierarchical | Local RAG | Enabled channels | KG entities | KG relations | Selection evidence / fold |
|---|---:|---:|---:|---|---:|---:|---:|
| agent_only | False | False | False | none | 0–0 | 0–0 | 0/0/0 |
| kg_3features_base | True | True | False | conservation,kg,physchem,structure | 1402–1471 | 3214–3537 | 48/48/48 |
