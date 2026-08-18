---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:use-liquid-recovery-for-pooled-qc
title: Use liquid recovery for pooled library QC
language: en
knowledge_type: saturation_mutagenesis_operations
statement: "For pooled saturation-library quality control, prepare the plasmid pool from liquid recovery culture when possible to reduce colony-growth bias before sequencing."
subject: pooled saturation-library quality control
predicate: should_prepare_plasmids_from
object: liquid recovery culture
polarity: support
claim_kind: operational_guideline
confidence: 0.90
applicability: {scope: transformation_based_saturation_libraries, limitation: liquid_growth_can_still_select_against_toxic_variants}
citation_support:
  - support_id: de:citation:liquid-recovery-pooled-qc
    publication_id: doi:10.1038/srep10654
    support_type: direct_support
    locator: quick_quality_control_discussion
    verified_against_source: true
selection_eligible: false
---
For pooled saturation-library quality control, prepare the plasmid pool from liquid recovery culture when possible to reduce colony-growth bias before sequencing.
