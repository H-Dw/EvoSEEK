---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:exclude-new-sequence-liabilities
title: Exclude new sequence liabilities during maturation
language: en
knowledge_type: binding_developability_validation
corpus_layer: binding
statement: "Before advancing an affinity-matured clone, inspect introduced mutations for unwanted glycosylation, deamidation, isomerization, oxidation, cleavage, and aggregation-prone motifs and test any application-relevant liability experimentally."
subject: affinity-matured protein sequence
predicate: should_be_screened_for
object: newly introduced sequence liabilities
polarity: support
claim_kind: operational_guideline
confidence: 0.91
applicability: {scope: recombinant_binders_intended_for_manufacture_or_long_term_use, limitation: sequence_motifs_predict_risk_and_do_not_replace_stress_testing}
citation_support:
  - support_id: binding:citation:exclude-sequence-liabilities
    publication_id: doi:10.1080/19420862.2022.2115200
    support_type: direct_support
    locator: abstract_and_liability_free_CDR_design
    verified_against_source: true
selection_eligible: false
---
Before advancing an affinity-matured clone, inspect introduced mutations for unwanted glycosylation, deamidation, isomerization, oxidation, cleavage, and aggregation-prone motifs and test any application-relevant liability experimentally.
