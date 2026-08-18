---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:pilot-multiple-epcr-mutation-loads
title: Pilot more than one epPCR mutation load
language: en
knowledge_type: random_mutagenesis_operations
statement: "When the retained functional fraction is unknown, pilot multiple mutation-load conditions and scale the condition whose measured diversity and retained activity fit the available screen."
subject: epPCR mutation load
predicate: should_be_selected_by
object: pilot diversity and retained activity
polarity: support
claim_kind: operational_guideline
confidence: 0.84
applicability: {scope: new_error_prone_pcr_campaigns, limitation: optimal_load_depends_on_parent_robustness_and_assay}
citation_support:
  - support_id: de:citation:pilot-multiple-mutation-loads
    publication_id: doi:10.1186/s12859-016-0996-7
    support_type: background_support
    locator: background_and_test_library_recommendation
    verified_against_source: true
selection_eligible: false
---
When the retained functional fraction is unknown, pilot multiple mutation-load conditions and scale the condition whose measured diversity and retained activity fit the available screen.
