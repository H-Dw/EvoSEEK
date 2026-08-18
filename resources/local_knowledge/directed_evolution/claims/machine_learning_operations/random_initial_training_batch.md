---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:random-initial-training-batch
title: Start ML guidance from a random measured batch
language: en
knowledge_type: machine_learning_operations
statement: "For a combinatorial machine-learning-guided campaign, begin with a random measured sample from the intended design space rather than training only on previously selected hits."
subject: initial machine learning training library
predicate: should_be_sampled_from
object: intended combinatorial design space
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: supervised_ml_guided_combinatorial_libraries, limitation: random_sampling_can_be_stratified_when_functional_sequences_are_extremely_rare}
citation_support:
  - support_id: de:citation:random-initial-training-batch
    publication_id: doi:10.1073/pnas.1901979116
    support_type: method_basis
    locator: directed_evolution_and_machine_learning_results
    verified_against_source: true
selection_eligible: false
---
For a combinatorial machine-learning-guided campaign, begin with a random measured sample from the intended design space rather than training only on previously selected hits.
