# Directed-evolution scientific knowledge bundle

This bundle is English-only and target-independent. Retrieval input is limited to
`claims/**/*.md`; each file contains one atomic claim under the
`scientific-atomic-claim:v1` schema. Publication metadata is normalized once in
`catalog/publications.yaml` and is not embedded as duplicate prose.

The bundle intentionally separates descriptive scientific priors from executable
campaign guidance. Files with `claim_kind: operational_guideline` state a concrete
action, the observable used to make the decision, and an applicability boundary.
Operational knowledge types cover campaign definition, assay engineering, random
and saturation mutagenesis, recombination, specificity engineering, stability and
evolvability, droplet screening, machine-learning and active-learning cycles,
continuous evolution, and round-to-round decisions.

Binding-affinity-specific operations are intentionally not stored in this bundle.
They live in the independently configurable `resources/local_knowledge/binding`
bundle so binding knowledge can be ablated without changing this general corpus.

An atomic claim is retrieved as one model-safe chunk. A retrieval match creates
`Document`, `DocumentChunk`, `Claim`, `CitationSupport`, and `Publication` records
in the knowledge graph. Raw retrieval context never contributes a fitness score.
Candidate-specific selection requires a separate validated calibration overlay.

The corpus must not contain target assay labels, hidden fitness measurements,
preferred target sequences, prompt instructions, or campaign retrieval history.

Research audit records live under `research/` as YAML and are excluded from the
retrieval glob. They preserve discovery queries, inclusion and exclusion criteria,
and verification routes without turning search notes into model-visible evidence.

Example operational queries include:

- How should an error-prone PCR mutation load be calibrated before scaling?
- How should saturation-library coverage be recalculated after pooled sequencing?
- How should positive and negative selection be coupled to evolve specificity?
- How should uncertainty be used to choose the next active-learning batch?
