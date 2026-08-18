---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:validate-computational-hotspots-experimentally
title: Validate computational hotspots experimentally
language: en
knowledge_type: binding_library_design
corpus_layer: binding
statement: "Use computational alanine scanning or interface-energy scores to prioritize sites, then validate those sites experimentally before allocating the focused library budget."
subject: computationally predicted interface hotspot
predicate: should_be_validated_by
object: experimental mutation and binding measurement
polarity: support
claim_kind: operational_guideline
confidence: 0.91
applicability: {scope: structure_available_protein_protein_interfaces, limitation: prediction_accuracy_varies_with_structure_quality_and_conformational_change}
citation_support:
  - support_id: binding:citation:validate-computational-hotspots
    publication_id: doi:10.1021/acschembio.9b00560
    support_type: direct_support
    locator: abstract_and_experimental_validation_results
    verified_against_source: true
selection_eligible: false
---
Use computational alanine scanning or interface-energy scores to prioritize sites, then validate those sites experimentally before allocating the focused library budget.
