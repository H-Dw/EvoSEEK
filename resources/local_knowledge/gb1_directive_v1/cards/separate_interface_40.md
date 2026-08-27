---
schema_version: gb1-directive-card:v1
record_type: knowledge_decision_card
language: en
status: research_candidate
human_reviewed: false
selection_eligible: false
permission: explanation_only
benchmark_overlap: none
scientific_credibility: medium
task_applicability: medium_high
retrieval_similarity: runtime_only
card_id: DC-GB1-003
title: Test position 40 as an interface variable
inputs:
  positions: [40]
  structural_context: Fc_facing_region
feature: interface_chemistry
direction: diversify_under_fixed_fold_context
boundaries:
  - Interface proximity does not establish a preferred residue.
  - Binding conditions can change the value of polar and electrostatic interactions.
uncertainty: medium_high
knowledge_type: measured_function
topics: [Fc binding, interface chemistry, matched comparison, position 40]
source_spans: [SS-002, SS-005]
logic_units: [LU-003]
---

## Use when

The batch needs binding-directed diversity after the 41/54 fold gate and 39/54 packing balance have been addressed.

## Decision rule

Treat position 40 as an interface hypothesis rather than a stability proxy. Compare chemistry classes while holding the fold-sensitive background constant.

## Candidate action

Allocate candidates across acidic or amide-like hydrogen-bonding, compact polar, and limited aromatic-contact hypotheses. Preserve at least two distinct chemistry classes instead of selecting one class by generic hydrophobicity.

## Matched comparison

Use the same 39/41/54 background for each position-40 class. If budget permits, repeat the class comparison on one alternative packing-compatible background to detect epistasis.

## Abstain or downgrade when

Downgrade when assay pH, ionic strength, or complex geometry is unknown. Abstain if the ranking depends only on a stability source or an unverified docking score.

## Evidence basis

The 1FCC record establishes a protein G–Fc complex, while independent thermodynamic analysis shows that folding and binding contributions must be separated. This card intentionally does not name a preferred substitution. It is falsified if chemistry-class differences vanish in matched backgrounds or reverse with assay conditions.
