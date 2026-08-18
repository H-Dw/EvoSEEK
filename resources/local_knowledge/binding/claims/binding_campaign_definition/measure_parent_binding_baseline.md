---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:measure-parent-binding-baseline
title: Measure the parent binding baseline
language: en
knowledge_type: binding_campaign_definition
corpus_layer: binding
statement: "Measure the parent as a soluble monomer across a ligand concentration series under the intended buffer and temperature before choosing affinity-selection stringency."
subject: parent binding protein
predicate: should_be_baselined_by
object: soluble monomeric concentration-series measurement
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: campaigns_with_purifiable_parent_and_ligand, limitation: membrane_context_interactions_may_require_an_additional_native_format_assay}
citation_support:
  - support_id: binding:citation:measure-parent-binding-baseline
    publication_id: doi:10.1016/j.xpro.2021.100836
    support_type: method_basis
    locator: affinity_measurement_workflow_and_concentration_series
    verified_against_source: true
selection_eligible: false
---
Measure the parent as a soluble monomer across a ligand concentration series under the intended buffer and temperature before choosing affinity-selection stringency.
