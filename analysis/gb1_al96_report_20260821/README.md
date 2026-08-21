# GB1 AL96 report analysis

This folder contains the complete, reproducible analysis used to update the GB1 manuscript report. Raw campaign artifacts are read only; every derived file is written under `outputs/`.

## Inputs

- `artifacts/random-fitness-direct-s42-al96-collected-20260820T102640Z`
- `artifacts/hierarchical-scientist-kg_base_kg_base_rag_kg_base_al`
- `artifacts/hierarchical-scientist-kg_3features_rag`
- `artifacts/hierarchical-scientist-kg_3features_base_agent_only`

The formal comparison includes `random`, `fitness_direct`, `agent_only`, `kg_base`, `kg_3features_base`, `kg_base_rag`, `kg_3features_rag`, and `kg_base_al`, each with folds 0–2. The three failed `kg_3features_rag` runs in the older hierarchical root are retained only as superseded audit records; the complete runs from the dedicated artifact root are used for metrics and cases.

## Environment

Required Python packages:

- Python 3.11+
- numpy
- pandas
- matplotlib
- pillow

The implementation uses only Python for plotting. SQLite access uses the Python standard library.

## Reproduce

From the repository root:

```powershell
uv run --no-project --with numpy --with pandas --with matplotlib --with pillow python analysis/gb1_al96_report_20260821/run_analysis.py
```

The command validates the run matrix, reconstructs round 0, computes fold-level and aggregate metrics, audits candidate-pool overlap, active-learning execution, all 54 three-channel child-agent paths, the `agent_only`/`kg_3features_base` runtime boundary, and feature-by-RAG interaction deltas, selects traceable Prompt/KG cases, writes Markdown tables, and exports SVG/PDF/TIFF/PNG figures.

To reproduce the focused diagnosis of rising cumulative best fitness alongside falling batch mean/median:

```powershell
python analysis/gb1_al96_report_20260821/run_anomaly_diagnostics.py
```

This analysis decomposes record events, full selected-batch distributions, dry–wet score alignment, acquisition-arm outcomes, candidate-pool overlap, and sequence-motif composition for the three KG conditions.

To audit why local RAG did not produce a stable gain, including retrieved claims, exact model-visible Prompt fields, evidence-ID projection, hypothesis changes, and base-vs-RAG candidate-pool divergence:

```powershell
python analysis/gb1_al96_report_20260821/run_rag_effect_diagnostics.py
```

To audit whether rounds add mutations to prior parents or independently select complete variants, and to separate Scientist preferences, 32-candidate pools, 16-variant selected batches, mutation depth, and position-residue novelty:

```powershell
python analysis/gb1_al96_report_20260821/run_mutation_behavior_diagnostics.py
```

To build the standalone Attachment D results-showcase module (WT and mutant
sequences, per-round Top-5, a representative wet-fitness trajectory, and an
auditable Prompt-to-outcome chain) without editing the source manuscript:

```powershell
python analysis/gb1_al96_report_20260821/build_results_showcase.py --top-k 5
python analysis/gb1_al96_report_20260821/validate_results_showcase.py
```

The generated Markdown is
`docs/GB1实验报告-附件D-结果展示-20260821.md`; supporting CSV, JSON, figure, and
hash-manifest files are written under `outputs/results_showcase/`. Provider
`reasoning_content` is deliberately excluded; the module displays only
model-visible messages, typed intermediate cards, structured responses, Critic
verdicts, acquisition receipts, and revealed outcomes.

To rerun the analysis and verify that source-data, table, and case-study files are byte-identical:

```powershell
python analysis/gb1_al96_report_20260821/verify_determinism.py
```

The anomaly tables and written diagnosis have a separate byte-level reproducibility check:

```powershell
python analysis/gb1_al96_report_20260821/verify_anomaly_determinism.py
```

The RAG-effect report, source tables, and Prompt evidence cases have a separate reproducibility check:

```powershell
python analysis/gb1_al96_report_20260821/verify_rag_effect_determinism.py
```

The closed-pool mutation-behavior report and its source tables have a separate reproducibility check:

```powershell
python analysis/gb1_al96_report_20260821/verify_mutation_behavior_determinism.py
```

After revising the manuscript, validate its key discovery values and figure links:

```powershell
python analysis/gb1_al96_report_20260821/validate_report.py
```

Build and validate the full eight-condition manuscript without overwriting either the original or six-condition draft:

```powershell
python analysis/gb1_al96_report_20260821/build_eight_strategy_report.py
uv run --no-project --with pandas python analysis/gb1_al96_report_20260821/validate_eight_strategy_report.py
```

## Statistical contract

- AL96 means 96 initial observations.
- Each campaign adds 16 queries per round for three rounds, for 48 total queries and 144 final visible observations.
- Curves and tables report three-fold mean and sample SD (`ddof=1`).
- No inferential test is performed for `n=3`.
- Candidate pools and seeds are not strictly paired across all conditions; module comparisons are descriptive fold-aligned differences.
- Prediction metrics on the isolated final set are kept separate from wet-fitness discovery metrics.

## Outputs

- `outputs/source_data/`: clean CSV files underlying every table and figure.
- `outputs/source_data/feature_channel_audit.csv`: 2 conditions × 3 folds × 3 rounds × physchem/conservation/structure execution and child-Critic evidence audit.
- `outputs/source_data/new_condition_runtime_audit.csv`: implemented runtime boundary for `agent_only` and `kg_3features_base`, including KG entity/relation and selection-evidence counts.
- `outputs/source_data/feature_rag_interaction_deltas*.csv`: fold-level and mean ± SD feature-by-RAG effects.
- `outputs/case_studies/`: selected cases, selection audit, Prompt excerpts, evidence records, and KG subgraphs.
- `outputs/tables/`: ready-to-paste performance, completion, module-delta, interaction, feature-execution, and runtime-boundary Markdown tables.
- `outputs/figures/`: editable SVG plus PDF/TIFF/PNG exports.
- `outputs/analysis_summary.json`: input fingerprints, run counts, environment, output hashes, and compact metric summary.
- `outputs/anomaly_diagnostics/`: focused anomaly report, 11 source-data tables, figure QA notes, and SVG/PDF/600-dpi TIFF/PNG exports.
- `outputs/rag_effect_diagnostics/`: retrieval/Prompt/selection-chain audit, five source-data tables, and exact model-visible Prompt evidence cases.
- `outputs/mutation_behavior_diagnostics/`: closed-pool semantics, per-round mutation-depth/novelty tables, Scientist/candidate/selected residue sets, exact 32-candidate pools, and lineage-adjacency audit.
