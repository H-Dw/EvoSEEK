---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:assay-combinations-and-components-together
title: Assay mutation combinations with their components
language: en
knowledge_type: round_decision_operations
statement: "When combining substitutions from different hits, assay the combined variant and the relevant component variants in the same batch to detect context-dependent effects before fixing the combination."
subject: combined directed evolution hit
predicate: should_be_compared_with
object: relevant component variants in the same assay batch
polarity: support
claim_kind: operational_guideline
confidence: 0.91
applicability: {scope: recombination_or_combinatorial_mutation_testing, limitation: high_order_interactions_can_require_more_than_pairwise_components}
citation_support:
  - support_id: de:citation:assay-combinations-components
    publication_id: doi:10.1038/nbt1286
    support_type: direct_support
    locator: discussion_of_context_and_combinatorial_libraries
    verified_against_source: true
selection_eligible: false
---
When combining substitutions from different hits, assay the combined variant and the relevant component variants in the same batch to detect context-dependent effects before fixing the combination.
