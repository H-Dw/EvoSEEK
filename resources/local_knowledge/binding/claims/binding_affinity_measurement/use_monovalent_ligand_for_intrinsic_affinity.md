---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:use-monovalent-ligand-for-intrinsic-affinity
title: Use monovalent ligand for intrinsic affinity
language: en
knowledge_type: binding_affinity_measurement
corpus_layer: binding
statement: "Use a monovalent ligand or a validated one-to-one binding geometry when estimating intrinsic affinity because multivalent target formats can convert weak monomeric interactions into strong avidity signals."
subject: intrinsic affinity measurement
predicate: should_use
object: monovalent one-to-one binding geometry
polarity: support
claim_kind: operational_guideline
confidence: 0.96
applicability: {scope: protein_protein_binding_and_antibody_antigen_measurement, limitation: native_multivalent_function_may_require_a_separate_avidity_assay}
citation_support:
  - support_id: binding:citation:use-monovalent-ligand-bli
    publication_id: doi:10.1016/j.xpro.2021.100836
    support_type: method_basis
    locator: protocol_note_requiring_monovalent_antigen_or_saturated_sites
    verified_against_source: true
  - support_id: binding:citation:use-monovalent-ligand-selection
    publication_id: doi:10.1073/pnas.170297297
    support_type: empirical_example
    locator: abstract_and_monovalent_affinity_validation
    verified_against_source: true
selection_eligible: false
---
Use a monovalent ligand or a validated one-to-one binding geometry when estimating intrinsic affinity because multivalent target formats can convert weak monomeric interactions into strong avidity signals.
