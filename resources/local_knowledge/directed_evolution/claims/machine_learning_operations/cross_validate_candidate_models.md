---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:cross-validate-candidate-protein-models
title: Cross-validate candidate models before ordering genes
language: en
knowledge_type: machine_learning_operations
statement: "Train multiple model classes or hyperparameter settings and choose them using cross-validation plus a measured holdout before using predictions to order genes."
subject: protein fitness prediction model
predicate: should_be_selected_by
object: cross-validation and measured holdout performance
polarity: support
claim_kind: operational_guideline
confidence: 0.92
applicability: {scope: supervised_machine_learning_guided_evolution, limitation: small_holdouts_have_high_metric_uncertainty}
citation_support:
  - support_id: de:citation:cross-validate-candidate-models
    publication_id: doi:10.1073/pnas.1901979116
    support_type: method_basis
    locator: validation_and_model_training_methods
    verified_against_source: true
selection_eligible: false
---
Train multiple model classes or hyperparameter settings and choose them using cross-validation plus a measured holdout before using predictions to order genes.
