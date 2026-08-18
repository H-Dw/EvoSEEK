---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:estimate-programmed-mutation-contributions
title: Estimate programmed mutation contributions from combinatorial data
language: en
knowledge_type: round_decision_operations
statement: "Use sequence-activity data from combinatorial libraries to estimate each programmed mutation's contribution before deciding which substitutions to retain, discard, or test again."
subject: programmed mutations in a combinatorial library
predicate: should_be_ranked_by
object: contribution estimates from sequence-activity data
polarity: support
claim_kind: operational_guideline
confidence: 0.93
applicability: {scope: combinatorial_libraries_with_repeated_mutation_observations, limitation: sparse_or_strongly_epistatic_data_need_nonlinear_or_followup_tests}
citation_support:
  - support_id: de:citation:estimate-mutation-contributions
    publication_id: doi:10.1038/nbt1286
    support_type: method_basis
    locator: optimization_strategy_and_mutation_analysis
    verified_against_source: true
selection_eligible: false
---
Use sequence-activity data from combinatorial libraries to estimate each programmed mutation's contribution before deciding which substitutions to retain, discard, or test again.
