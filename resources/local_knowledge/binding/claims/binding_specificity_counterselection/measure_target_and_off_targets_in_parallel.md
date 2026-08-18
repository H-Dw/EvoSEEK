---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:measure-target-and-off-targets-in-parallel
title: Measure target and off-target interactions in parallel
language: en
knowledge_type: binding_specificity_counterselection
corpus_layer: binding
statement: "Measure each enriched sequence against the target and off-target panel under matched concentrations and report the complete interaction profile before selecting a new parent."
subject: enriched binding sequence
predicate: should_be_profiled_against
object: matched target and off-target interaction matrix
polarity: support
claim_kind: operational_guideline
confidence: 0.95
applicability: {scope: multiplexable_binding_assays_or_display_NGS, limitation: relative_enrichment_requires_soluble_confirmation_for_absolute_affinity}
citation_support:
  - support_id: binding:citation:parallel-target-off-target
    publication_id: doi:10.1039/c9me00118b
    support_type: direct_support
    locator: abstract_and_multiplex_selection_results
    verified_against_source: true
selection_eligible: false
---
Measure each enriched sequence against the target and off-target panel under matched concentrations and report the complete interaction profile before selecting a new parent.
