---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:analyze-non-hit-sequence-activity-data
title: Analyze sequence-activity data beyond the top hits
language: en
knowledge_type: round_decision_operations
statement: "Do not discard all non-hit sequence-activity data after screening because a beneficial substitution can be hidden inside a low-performing combination."
subject: non-hit sequence-activity data
predicate: can_contain
object: beneficial substitutions masked by context
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: sequenced_combinatorial_libraries, limitation: repeated_observations_are_needed_for_reliable_attribution}
citation_support:
  - support_id: de:citation:analyze-non-hit-sequences
    publication_id: doi:10.1038/nbt1286
    support_type: direct_support
    locator: table_2_and_mutation_analysis
    verified_against_source: true
selection_eligible: false
---
Do not discard all non-hit sequence-activity data after screening because a beneficial substitution can be hidden inside a low-performing combination.
