---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:use-exact-arrayed-genes-when-affordable
title: Use exact arrayed genes when synthesis permits
language: en
knowledge_type: machine_learning_operations
statement: "When synthesis budget permits, construct exact arrayed candidate genes so planned variants are measured without codon oversampling or unknown genotype identity."
subject: active learning candidate construction
predicate: should_use_when_affordable
object: exact arrayed gene synthesis
polarity: support
claim_kind: operational_guideline
confidence: 0.91
applicability: {scope: low_to_moderate_batch_model_guided_campaigns, limitation: exact_synthesis_cost_can_exceed_degenerate_library_cost}
citation_support:
  - support_id: de:citation:use-exact-arrayed-genes
    publication_id: doi:10.1038/s41467-025-55987-8
    support_type: empirical_example
    locator: wet_lab_workflow_and_exact_gene_delivery
    verified_against_source: true
selection_eligible: false
---
When synthesis budget permits, construct exact arrayed candidate genes so planned variants are measured without codon oversampling or unknown genotype identity.
