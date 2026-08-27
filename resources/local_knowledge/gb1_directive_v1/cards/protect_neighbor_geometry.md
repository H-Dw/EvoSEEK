---
schema_version: gb1-directive-card:v1
record_type: knowledge_decision_card
language: en
status: research_candidate
human_reviewed: false
selection_eligible: false
permission: explanation_only
benchmark_overlap: none
scientific_credibility: medium_high
task_applicability: high
retrieval_similarity: runtime_only
card_id: DC-GB1-004
title: Protect fixed aromatic neighbor geometry
inputs:
  positions: [39, 41, 54]
  structural_context: neighbors_of_positions_43_45_52
feature: steric_compatibility_with_fixed_aromatics
direction: avoid_uncompensated_crowding
boundaries:
  - Fixed-neighbor sensitivity is a structural constraint, not a binding score.
  - Side-chain relaxation can rescue apparent static clashes.
uncertainty: medium
knowledge_type: structural_context
topics: [steric compatibility, aromatic packing, compensatory change]
source_spans: [SS-001, SS-004, SS-006]
logic_units: [LU-004]
---

## Use when

A candidate introduces bulky or rigid side chains near the fixed aromatic cluster around positions 43, 45, and 52, particularly together with changes at 41 or 54.

## Decision rule

Preserve space for the fixed aromatic network. Penalize simultaneous bulky changes that approach it unless another local change plausibly relieves crowding.

## Candidate action

Keep one packing-conservative candidate for every crowding hypothesis. Prefer compensated volume redistribution over uniform enlargement, and preserve a small-backbone-compatible option at position 41 when nearby aromatic packing is already tightened.

## Matched comparison

Hold interface chemistry constant and compare a crowded proposal with a volume-compensated proposal. Add a conservative reference that changes only one packing site.

## Abstain or downgrade when

Downgrade static clash judgments when a credible mutant structure supports rotamer relaxation or backbone movement. Abstain if the structure mapping is uncertain.

## Evidence basis

The GB1 crystal structure shows compact local packing, and independent stability measurements identify positions 41, 45, 52, and 54 as unusually sensitive. The direct inference is constraint preservation, not affinity prediction. The rule is falsified if modeled or measured folded candidates tolerate the crowded class without compensatory rearrangement.
