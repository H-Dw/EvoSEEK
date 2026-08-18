---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:deplete-on-target-negative-cells
title: Deplete on target-negative cells before cell panning
language: en
knowledge_type: binding_specificity_counterselection
corpus_layer: binding
statement: "Before positive panning on target-expressing cells, deplete the library on a matched target-negative cell population to remove binders to shared cell-surface components."
subject: cell-based ligand selection
predicate: should_begin_with
object: matched target-negative cell depletion
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: cell_surface_target_panning_with_a_matched_negative_population, limitation: unmatched_cell_states_can_leave_context_specific_nonspecific_binders}
citation_support:
  - support_id: binding:citation:deplete-target-negative-cells
    publication_id: doi:10.1007/978-1-4939-9853-1_17
    support_type: method_basis
    locator: target_negative_cell_depletion_and_cell_panning_sections
    verified_against_source: true
selection_eligible: false
---
Before positive panning on target-expressing cells, deplete the library on a matched target-negative cell population to remove binders to shared cell-surface components.
