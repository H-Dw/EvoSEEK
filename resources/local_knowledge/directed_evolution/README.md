# Directed-evolution scientific knowledge bundle

This bundle is English-only and target-independent. Retrieval input is limited to
`claims/**/*.md`; each file contains one atomic claim under the
`scientific-atomic-claim:v1` schema. Publication metadata is normalized once in
`catalog/publications.yaml` and is not embedded as duplicate prose.

An atomic claim is retrieved as one model-safe chunk. A retrieval match creates
`Document`, `DocumentChunk`, `Claim`, `CitationSupport`, and `Publication` records
in the knowledge graph. Raw retrieval context never contributes a fitness score.
Candidate-specific selection requires a separate validated calibration overlay.

The corpus must not contain target assay labels, hidden fitness measurements,
preferred target sequences, prompt instructions, or campaign retrieval history.
