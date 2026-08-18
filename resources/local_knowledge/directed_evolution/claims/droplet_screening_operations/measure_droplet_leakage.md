---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:measure-droplet-assay-leakage
title: Measure assay leakage between droplets
language: en
knowledge_type: droplet_screening_operations
statement: "Measure substrate and product leakage between positive and negative droplets over the intended incubation time before screening a library."
subject: droplet assay chemistry
predicate: should_be_tested_for
object: inter-droplet substrate and product leakage
polarity: support
claim_kind: operational_guideline
confidence: 0.90
applicability: {scope: droplet_based_activity_screens, limitation: leakage_depends_on_oil_surfactant_and_molecular_properties}
citation_support:
  - support_id: de:citation:measure-droplet-leakage
    publication_id: doi:10.1021/acs.chemrev.2c00910
    support_type: background_support
    locator: practical_assay_compartmentalization_challenges
    verified_against_source: true
selection_eligible: false
---
Measure substrate and product leakage between positive and negative droplets over the intended incubation time before screening a library.
