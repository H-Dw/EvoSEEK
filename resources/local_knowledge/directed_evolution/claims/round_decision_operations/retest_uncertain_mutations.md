---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:retest-uncertain-mutations-in-new-library
title: Retest uncertain substitutions in a new combinatorial context
language: en
knowledge_type: round_decision_operations
statement: "Retest a substitution in a new combinatorial library when its estimated effect is uncertain instead of permanently fixing or discarding it after one observation."
subject: substitution with uncertain effect
predicate: should_be_retested_in
object: new combinatorial library context
polarity: support
claim_kind: operational_guideline
confidence: 0.92
applicability: {scope: iterative_statistical_or_model_guided_evolution, limitation: retesting_consumes_screening_capacity}
citation_support:
  - support_id: de:citation:retest-uncertain-mutations
    publication_id: doi:10.1038/nbt1286
    support_type: direct_support
    locator: optimization_strategy_and_uncertain_mutation_retesting
    verified_against_source: true
selection_eligible: false
---
Retest a substitution in a new combinatorial library when its estimated effect is uncertain instead of permanently fixing or discarding it after one observation.
