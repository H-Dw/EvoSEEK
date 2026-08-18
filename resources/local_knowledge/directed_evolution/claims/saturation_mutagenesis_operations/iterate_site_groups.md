---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:iterate-saturation-site-groups
title: Iterate manageable saturation site groups
language: en
knowledge_type: saturation_mutagenesis_operations
statement: "If simultaneous saturation of all selected sites exceeds capacity, mutate one site or a small site group, screen it, fix a verified improved variant, and use it as the parent for the next group."
subject: overlarge multi-site saturation plan
predicate: should_be_partitioned_into
object: iterative site-group libraries
polarity: support
claim_kind: operational_guideline
confidence: 0.92
applicability: {scope: iterative_saturation_mutagenesis, limitation: site_order_can_change_accessible_epistatic_paths}
citation_support:
  - support_id: de:citation:iterate-saturation-site-groups
    publication_id: doi:10.1038/nprot.2007.72
    support_type: method_basis
    locator: abstract_and_protocol_scope
    verified_against_source: true
selection_eligible: false
---
If simultaneous saturation of all selected sites exceeds capacity, mutate one site or a small site group, screen it, fix a verified improved variant, and use it as the parent for the next group.
