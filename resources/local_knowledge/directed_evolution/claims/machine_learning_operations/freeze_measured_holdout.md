---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:freeze-measured-holdout-for-model-checks
title: Freeze a measured holdout for model checks
language: en
knowledge_type: machine_learning_operations
statement: "Keep a measured holdout set outside model fitting and acquisition updates so ranking quality can be checked before model-guided candidate selection."
subject: protein fitness model evaluation
predicate: should_reserve
object: measured holdout outside fitting and acquisition
polarity: support
claim_kind: operational_guideline
confidence: 0.90
applicability: {scope: model_guided_directed_evolution_with_sufficient_measurements, limitation: reducing_training_size_can_hurt_very_low_data_campaigns}
citation_support:
  - support_id: de:citation:freeze-measured-holdout
    publication_id: doi:10.1073/pnas.1901979116
    support_type: method_basis
    locator: validation_on_empirical_fitness_landscape
    verified_against_source: true
selection_eligible: false
---
Keep a measured holdout set outside model fitting and acquisition updates so ranking quality can be checked before model-guided candidate selection.
