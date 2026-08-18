---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:map-permissive-interface-positions
title: Map permissive interface positions experimentally
language: en
knowledge_type: binding_library_design
corpus_layer: binding
statement: "Use experimental alanine scanning or another single-substitution scan to identify interface positions that tolerate mutation before constructing a combinatorial affinity-maturation library."
subject: focused binding-interface library
predicate: should_be_preceded_by
object: experimental permissiveness scan
polarity: support
claim_kind: operational_guideline
confidence: 0.94
applicability: {scope: binders_with_identified_or_predicted_contact_regions, limitation: alanine_effects_do_not_enumerate_all_beneficial_nonalanine_substitutions}
citation_support:
  - support_id: binding:citation:map-permissive-interface-positions
    publication_id: doi:10.3389/fimmu.2017.00986
    support_type: direct_support
    locator: abstract_and_three_step_library_design
    verified_against_source: true
selection_eligible: false
---
Use experimental alanine scanning or another single-substitution scan to identify interface positions that tolerate mutation before constructing a combinatorial affinity-maturation library.
