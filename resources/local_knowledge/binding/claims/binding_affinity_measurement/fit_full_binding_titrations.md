---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: binding:claim:fit-full-binding-titrations
title: Fit full binding titrations
language: en
knowledge_type: binding_affinity_measurement
corpus_layer: binding
statement: "Infer KD from a binding curve measured at multiple ligand concentrations rather than from enrichment or fluorescence at a single concentration."
subject: equilibrium dissociation constant
predicate: should_be_inferred_from
object: multi-concentration binding curve
polarity: support
claim_kind: operational_guideline
confidence: 0.97
applicability: {scope: display_or_soluble_equilibrium_binding_measurement, limitation: fitted_KD_requires_equilibrium_and_an_identifiable_dynamic_range}
citation_support:
  - support_id: binding:citation:fit-full-binding-titrations
    publication_id: doi:10.7554/elife.23156
    support_type: direct_support
    locator: figure_1_methods_and_discussion_on_single_concentration_confounding
    verified_against_source: true
selection_eligible: false
---
Infer KD from a binding curve measured at multiple ligand concentrations rather than from enrichment or fluorescence at a single concentration.
