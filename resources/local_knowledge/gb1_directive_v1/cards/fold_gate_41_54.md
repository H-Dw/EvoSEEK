---
schema_version: gb1-directive-card:v1
record_type: knowledge_decision_card
language: en
status: research_candidate
human_reviewed: false
selection_eligible: false
permission: explanation_only
benchmark_overlap: none
scientific_credibility: high
task_applicability: high_for_stability_gate
retrieval_similarity: runtime_only
card_id: DC-GB1-001
title: Apply a fold gate at positions 41 and 54
inputs:
  positions: [41, 54]
  structural_context: beta_hairpin_and_core_boundary
feature: backbone_and_packing_compatibility
direction: preserve_before_optimizing_binding
boundaries:
  - This rule constrains stability risk and does not predict binding improvement.
  - Mutant repacking is not observed in the wild-type structure.
uncertainty: medium
knowledge_type: structural_context
topics: [folding, thermodynamic stability, residue burial, positions 41 and 54]
source_spans: [SS-001, SS-003, SS-004, SS-006]
logic_units: [LU-001]
---

## Use when

A proposed batch changes position 41, position 54, or both, especially when it also introduces a large side-chain volume change.

## Decision rule

Treat these positions as a folding gate: first preserve backbone compatibility at 41 and local packing at 54, then spend remaining diversity on binding hypotheses.

## Candidate action

Keep at least one small or conformationally compatible class at 41 and one size-compatible hydrophobic class at 54. Avoid making every candidate simultaneously larger or more conformationally restrictive at both sites.

## Matched comparison

Compare a pair that shares positions 39 and 40 but differs only in whether 41/54 obey the gate. Include a second pair that changes one gate position at a time.

## Abstain or downgrade when

Downgrade if a mutant structure supports compensatory repacking, or if the assay explicitly separates folded abundance from Fc binding. Do not infer affinity from this gate alone.

## Evidence basis

Independent thermodynamic measurements identify positions 41 and 54 as stability sensitive, while 1PGA shows a compact fold and direct local proximity. The rule is falsified if gate-violating matched candidates repeatedly retain fold-compatible behavior without compensation.
