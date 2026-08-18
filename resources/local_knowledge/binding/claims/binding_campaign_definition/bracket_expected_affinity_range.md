---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:bracket-expected-affinity-range
title: Bracket the expected affinity range
language: en
knowledge_type: binding_campaign_definition
corpus_layer: binding
statement: "Choose ligand concentrations that span the parent KD and the expected improved range instead of using one fixed concentration for every affinity-maturation round."
subject: ligand concentration schedule
predicate: should_span
object: parent and expected improved KD range
polarity: support
claim_kind: operational_guideline
confidence: 0.95
applicability: {scope: equilibrium_binding_screens_and_titration_sorting, limitation: concentrations_outside_the_assay_dynamic_range_do_not_resolve_affinity}
citation_support:
  - support_id: binding:citation:bracket-expected-affinity-range
    publication_id: doi:10.7554/elife.23156
    support_type: direct_support
    locator: results_and_discussion_on_multi_concentration_titration
    verified_against_source: true
selection_eligible: false
---
Choose ligand concentrations that span the parent KD and the expected improved range instead of using one fixed concentration for every affinity-maturation round.
