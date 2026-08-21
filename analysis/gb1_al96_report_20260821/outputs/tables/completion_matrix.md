# Completion matrix

| 条件 | fold 0 | fold 1 | fold 2 | 正式纳入 |
|---|---:|---:|---:|---:|
| Random | 通过 | 通过 | 通过 | 3/3 |
| Kermut direct | 通过 | 通过 | 通过 | 3/3 |
| Agent only | 通过 | 通过 | 通过 | 3/3 |
| KG base | 通过 | 通过 | 通过 | 3/3 |
| KG + 3-channel | 通过 | 通过 | 通过 | 3/3 |
| KG + RAG | 通过 | 通过 | 通过 | 3/3 |
| KG + 3-channel + RAG | 通过 | 通过 | 通过 | 3/3 |
| KG + active learning | 通过 | 通过 | 通过 | 3/3 |

> `kg_3features_rag`、`kg_3features_base` 与 `agent_only` 均以新完成的三折结果正式纳入；旧目录中的失败 `kg_3features_rag` 运行仅作为被替代的审计记录。
