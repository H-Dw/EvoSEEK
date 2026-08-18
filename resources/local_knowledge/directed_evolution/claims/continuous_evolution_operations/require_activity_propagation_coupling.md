---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:require-activity-propagation-coupling-for-pace
title: Couple activity to propagation before using PACE
language: en
knowledge_type: continuous_evolution_operations
statement: "Use phage-assisted continuous evolution only after the desired protein activity is coupled to production of the phage propagation factor."
subject: phage-assisted continuous evolution
predicate: requires
object: desired activity coupled to phage propagation
polarity: support
claim_kind: operational_guideline
confidence: 0.96
applicability: {scope: gene_encoded_activities_expressible_in_compatible_hosts, limitation: uncoupled_properties_need_a_new_selection_circuit}
citation_support:
  - support_id: de:citation:require-activity-propagation-coupling
    publication_id: doi:10.1038/nature09929
    support_type: method_basis
    locator: abstract_and_figure_2_scope
    verified_against_source: true
selection_eligible: false
---
Use phage-assisted continuous evolution only after the desired protein activity is coupled to production of the phage propagation factor.
