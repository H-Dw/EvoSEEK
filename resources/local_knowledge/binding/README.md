# Binding-affinity knowledge bundle

This English-only bundle contains target-independent operational knowledge for
improving protein binding affinity by directed evolution. It is intentionally
separate from `resources/local_knowledge/directed_evolution` so that binding
knowledge can be enabled, filtered, or ablated as one corpus layer.

Every retrieval unit is one `scientific-atomic-claim:v1` file under
`claims/**/*.md`. In addition to a `binding_*` knowledge type, each claim carries
`corpus_layer: binding`; the directory, knowledge type, and metadata therefore
provide three independent ways to audit or remove this layer.

The operational hierarchy covers campaign definition, affinity measurement,
display-platform selection, focused library design, equilibrium and kinetic
selection, specificity counterselection, round decisions, and developability
validation. Claims specify an action, an observable or decision criterion, and an
applicability boundary. Raw enrichment, display fluorescence, avidity, or a
computational score is never represented as an experimentally confirmed affinity.

Publication metadata is normalized in `catalog/publications.yaml`. Search and
verification records live under `research/` and are excluded from retrieval.
All claims default to `selection_eligible: false`; candidate-specific use requires
a separate leakage-safe calibration against measurements visible in the campaign.

To ablate this layer, remove the `resources/local_knowledge/binding` root from the
active knowledge configuration. To retain the root but exclude binding operations,
filter out knowledge types beginning with `binding_`.

Example queries:

- How should ligand concentrations be chosen for affinity maturation?
- How can a display assay distinguish affinity from expression and avidity?
- How should slow-off-rate variants be selected without rebinding?
- Which counterselections should accompany affinity optimization?
- Which measurements are required before an affinity-matured clone becomes the next parent?
