---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:load-monoclonal-droplets-by-poisson-dilution
title: Load predominantly monoclonal droplets
language: en
knowledge_type: droplet_screening_operations
statement: "Set cell or DNA loading to a Poisson regime in which occupied droplets are predominantly monoclonal, accepting many empty droplets to avoid mixed-genotype phenotypes."
subject: droplet library loading
predicate: should_target
object: predominantly monoclonal occupied droplets
polarity: support
claim_kind: operational_guideline
confidence: 0.93
applicability: {scope: single_variant_droplet_screening, limitation: exact_occupancy_depends_on_cell_or_dna_input_statistics}
citation_support:
  - support_id: de:citation:load-monoclonal-droplets
    publication_id: doi:10.1021/acs.chemrev.2c00910
    support_type: background_support
    locator: expression_systems_and_monoclonality_section
    verified_against_source: true
selection_eligible: false
---
Set cell or DNA loading to a Poisson regime in which occupied droplets are predominantly monoclonal, accepting many empty droplets to avoid mixed-genotype phenotypes.
