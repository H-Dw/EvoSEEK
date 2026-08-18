---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:build-focused-predicted-library
title: Rank in silico and build a focused measured library
language: en
knowledge_type: machine_learning_operations
statement: "Use the selected model to rank the enumerated combinatorial space, then build and assay a smaller predicted library rather than treating predictions as experimental fitness."
subject: machine learning guided combinatorial search
predicate: should_convert_rankings_into
object: experimentally assayed focused library
polarity: support
claim_kind: operational_guideline
confidence: 0.95
applicability: {scope: enumerable_or_sampleable_combinatorial_spaces, limitation: unmeasured_predictions_are_not_selection_eligible_evidence}
citation_support:
  - support_id: de:citation:build-focused-predicted-library
    publication_id: doi:10.1073/pnas.1901979116
    support_type: direct_support
    locator: directed_evolution_and_machine_learning_results
    verified_against_source: true
selection_eligible: false
---
Use the selected model to rank the enumerated combinatorial space, then build and assay a smaller predicted library rather than treating predictions as experimental fitness.
