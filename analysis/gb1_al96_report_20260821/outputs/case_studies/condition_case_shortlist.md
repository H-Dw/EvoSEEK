# Condition-level case shortlist

| Condition | Case | Fold | Round | Variant | Mutation | Predicted fitness | Wet fitness | Acquisition | Knowledge |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| agent_only | positive | 0 | 2 | IYGC | V39I;D40Y;V54C | 3.8038 ± 0.5903 | 4.4857 | 1.7158 | 0.0000 |
| agent_only | negative | 1 | 1 | IFHA | V39I;D40F;G41H;V54A | 1.7360 ± 0.7811 | 0.0032 | 1.5103 | 0.0000 |
| kg_3features_base | positive | 2 | 2 | VWAA | D40W;G41A;V54A | 3.5489 ± 0.7451 | 6.1238 | 2.4652 | 0.7257 |
| kg_3features_base | negative | 0 | 2 | LYWC | V39L;D40Y;G41W;V54C | 3.4050 ± 0.7147 | 0.0198 | 2.5421 | 0.7924 |
| kg_3features_rag | positive | 0 | 1 | LYGV | V39L;D40Y | 4.1728 ± 0.4864 | 5.0753 | 1.4366 | 0.6717 |
| kg_3features_rag | negative | 0 | 3 | VYGY | D40Y;V54Y | 2.0767 ± 0.6858 | 0.2170 | 3.3594 | 0.6970 |
| kg_base | positive | 0 | 3 | LWAA | V39L;D40W;G41A;V54A | 1.1660 ± 0.7215 | 6.0275 | 2.1631 | 0.5523 |
| kg_base | negative | 0 | 2 | LFHC | V39L;D40F;G41H;V54C | 2.0703 ± 0.7596 | 0.0047 | 2.2654 | 0.6258 |
| kg_base_al | positive | 2 | 3 | IWGM | V39I;D40W;V54M | 2.7780 ± 0.6861 | 5.2292 | 0.8226 | 0.7587 |
| kg_base_al | negative | 1 | 2 | LYHV | V39L;D40Y;G41H | 3.4965 ± 0.5454 | 0.0065 | 0.8306 | 0.8254 |
| kg_base_rag | positive | 1 | 2 | LYGV | V39L;D40Y | 4.6663 ± 0.4042 | 5.0753 | 1.8762 | 0.8059 |
| kg_base_rag | negative | 2 | 3 | LWTC | V39L;D40W;G41T;V54C | 2.5268 ± 0.6476 | 0.0145 | 2.5143 | 0.7985 |

Positive cases maximize wet fitness within each condition. Negative cases use the predeclared low-wet/high-acquisition surprise rule.
