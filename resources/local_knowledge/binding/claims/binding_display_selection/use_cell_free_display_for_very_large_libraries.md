---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:use-cell-free-display-for-very-large-libraries
title: Use cell-free display for very large libraries
language: en
knowledge_type: binding_display_selection
corpus_layer: binding
statement: "Use ribosome or mRNA display when the required sequence diversity exceeds practical transformation-limited cellular libraries and the protein can fold under cell-free selection conditions."
subject: transformation-limited binding library
predicate: can_be_expanded_by
object: ribosome or mRNA display
polarity: support
claim_kind: operational_guideline
confidence: 0.93
applicability: {scope: soluble_binders_compatible_with_cell_free_translation, limitation: complex_post_translational_modifications_and_membrane_context_are_not_reproduced}
citation_support:
  - support_id: binding:citation:cell-free-display-primary
    publication_id: doi:10.1073/pnas.94.10.4937
    support_type: method_basis
    locator: abstract_cell_free_selection_without_host_constraints
    verified_against_source: true
  - support_id: binding:citation:cell-free-display-recent
    publication_id: doi:10.1038/s41551-023-01093-3
    support_type: empirical_example
    locator: abstract_massively_parallel_ribosome_display_and_affinity_screening
    verified_against_source: true
selection_eligible: false
---
Use ribosome or mRNA display when the required sequence diversity exceeds practical transformation-limited cellular libraries and the protein can fold under cell-free selection conditions.
