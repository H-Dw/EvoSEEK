---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:retain-all-sequence-assay-pairs
title: Retain every measured sequence-assay pair
language: en
knowledge_type: machine_learning_operations
statement: "Retain sequence and assay data for every measured variant, including low performers, because the full labeled batch is training information for the next model."
subject: measured directed evolution batch
predicate: should_retain
object: all sequence and assay pairs
polarity: support
claim_kind: operational_guideline
confidence: 0.96
applicability: {scope: data_driven_protein_engineering, limitation: measurements_need_consistent_assay_provenance}
citation_support:
  - support_id: de:citation:retain-all-measured-variants
    publication_id: doi:10.1073/pnas.1901979116
    support_type: method_basis
    locator: directed_evolution_and_machine_learning_results
    verified_against_source: true
selection_eligible: false
---
Retain sequence and assay data for every measured variant, including low performers, because the full labeled batch is training information for the next model.
