---
schema_version: scientific-atomic-claim:v1
record_type: atomic_claim
claim_id: de:claim:profile-selected-variants-across-substrates
title: Profile specificity after selection
language: en
knowledge_type: specificity_engineering
statement: "After dual selection, quantify evolved variants on a panel containing the desired substrate, the parental substrate, and relevant off-targets before choosing a parent."
subject: dual-selection hit
predicate: requires_post_selection_profile_on
object: desired parental and off-target substrates
polarity: support
claim_kind: operational_guideline
confidence: 0.93
applicability: {scope: evolved_substrate_or_binding_specificity, limitation: finite_panels_do_not_cover_all_off_targets}
citation_support:
  - support_id: de:citation:profile-after-dual-selection
    publication_id: doi:10.1038/s41467-020-20650-x
    support_type: direct_support
    locator: abstract_and_downstream_pam_preference_assays
    verified_against_source: true
selection_eligible: false
---
After dual selection, quantify evolved variants on a panel containing the desired substrate, the parental substrate, and relevant off-targets before choosing a parent.
