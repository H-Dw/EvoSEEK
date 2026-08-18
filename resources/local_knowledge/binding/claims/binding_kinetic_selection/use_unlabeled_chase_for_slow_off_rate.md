---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:use-unlabeled-chase-for-slow-off-rate
title: Use an unlabeled chase for slow-off-rate selection
language: en
knowledge_type: binding_kinetic_selection
corpus_layer: binding
statement: "To enrich slow-dissociating variants, prebind labeled ligand, remove free label, and chase with excess unlabeled ligand or a sufficiently large volume so that dissociated labeled ligand cannot rebind."
subject: slow-off-rate display selection
predicate: should_use
object: labeled prebinding followed by rebinding-blocking chase
polarity: support
claim_kind: operational_guideline
confidence: 0.97
applicability: {scope: phage_yeast_or_ribosome_display_with_stable_genotype_phenotype_linkage, limitation: chase_duration_is_bounded_by_display_complex_and_protein_stability}
citation_support:
  - support_id: binding:citation:unlabeled-chase-phage
    publication_id: doi:10.1016/0022-2836(92)90639-2
    support_type: method_basis
    locator: abstract_off_rate_selection
    verified_against_source: true
  - support_id: binding:citation:unlabeled-chase-yeast
    publication_id: doi:10.1073/pnas.170297297
    support_type: direct_support
    locator: methods_and_results_kinetic_screening
    verified_against_source: true
selection_eligible: false
---
To enrich slow-dissociating variants, prebind labeled ligand, remove free label, and chase with excess unlabeled ligand or a sufficiently large volume so that dissociated labeled ligand cannot rebind.
