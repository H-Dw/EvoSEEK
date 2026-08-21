# Figure QA notes

## Figure 2: fitness trajectories

- Core conclusion: Agent-only enriches batches beyond random and Kermut-direct; Experimental Memory improves cumulative peak discovery; KG + 3 channels gives the highest peak/AULC, whereas active learning gives the strongest late-batch distribution.
- Archetype: quantitative grid with one cumulative-discovery panel and two batch-distribution panels.
- Backend: Python/matplotlib only.
- Source data: `../source_data/round_metrics_by_fold.csv` and `../source_data/round_metrics_mean_sd.csv`.
- Statistics: n=3 folds; center=mean; spread=sample s.d. (`ddof=1`); no inferential test.
- Export bundle: editable SVG, PDF, 600-dpi RGB TIFF and 300-dpi PNG.
- Visual QA: PNG inspected at final export on 2026-08-21; axes, eight-method two-row legend, fold trajectories, mean markers and ±s.d. bands are readable with no overlap.

## Figure 3: module deltas

- Core conclusion: Experimental Memory consistently improves peak/AULC over Agent-only, active learning consistently improves AULC and batch outcomes, whereas three-channel and RAG increments are fold-dependent and show no stable additive synergy.
- Archetype: quantitative grid of four fold-aligned delta panels.
- Backend: Python/matplotlib only.
- Source data: `../source_data/kg_module_fold_deltas.csv`.
- Statistics: three fold-level deltas plus their mean and observed range; no confidence interval or inferential test.
- Comparators: KG base − Agent only; 3 channels − KG base; KG + RAG − KG base; 3 channels + RAG − KG + RAG; 3 channels + RAG − 3 channels; KG + active learning − KG base.
- Export bundle: editable SVG, PDF, 600-dpi RGB TIFF and 300-dpi PNG.
- Visual QA: PNG inspected at final export on 2026-08-21; all six comparator labels remain visible, four zero references are aligned, and fold points are not occluded by mean diamonds.
