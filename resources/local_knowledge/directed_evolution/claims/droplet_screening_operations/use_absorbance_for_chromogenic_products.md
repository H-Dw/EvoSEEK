---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:use-absorbance-sorting-for-chromogenic-products
title: Use absorbance sorting for faithful chromogenic products
language: en
knowledge_type: droplet_screening_operations
statement: "If the desired reaction produces a strong chromophore but lacks a faithful fluorogenic substrate, calibrate absorbance-activated droplet sorting against parent and blank droplets."
subject: chromogenic enzyme reaction
predicate: can_use
object: calibrated absorbance-activated droplet sorting
polarity: support
claim_kind: operational_guideline
confidence: 0.92
applicability: {scope: reactions_with_sufficient_absorbance_signal, limitation: absorbance_is_less_sensitive_than_fluorescence}
citation_support:
  - support_id: de:citation:use-absorbance-for-chromogenic-products
    publication_id: doi:10.1073/pnas.1606927113
    support_type: method_basis
    locator: operation_calibration_and_utility_discussion
    verified_against_source: true
selection_eligible: false
---
If the desired reaction produces a strong chromophore but lacks a faithful fluorogenic substrate, calibrate absorbance-activated droplet sorting against parent and blank droplets.
