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
task_applicability: high_for_early_search
retrieval_similarity: runtime_only
card_id: DC-GB1-V3-001
title: Start with binding-first diversity under a fold boundary
inputs:
  stage: early_round
  positions: [39, 40, 41, 54]
feature: binding_hypothesis_diversity_with_fold_plausibility
direction: diversify_function_before_static_overconstraint
boundaries:
  - Stability is a gate and not the binding objective.
  - Static structure cannot rank mutant affinity.
uncertainty: medium
knowledge_type: gb1_stage_early_decision_rule
topics: [early round, binding first, interface chemistry, fold boundary]
source_spans: [SS-001, SS-002, SS-003, SS-005]
logic_units: [LU-003, LU-005]
---

## Use when

Use in the first search round, before the current campaign has produced new measurements for its proposed hypotheses.

## Decision rule

Let binding-directed chemistry define the alternatives. Apply folding and packing evidence only as a boundary against jointly implausible changes, never as the objective or a residue ranking.

## Candidate action

Propose distinct interface and packing classes, keep one fold-conservative reference, and avoid spending the whole batch on one static-structure story. Prefer hypotheses that can be separated by the configured assay.

## Matched comparison

Hold the fold-sensitive background approximately constant while comparing interface-chemistry classes. Pair one ambitious functional hypothesis with a conservative background-matched reference.

## Abstain or downgrade when

Downgrade any choice supported only by stability, hydrophobicity, or a static contact. Abstain from naming a preferred class when assay conditions are not transferable.

## Evidence basis

The inherited structure and thermodynamic spans support fold plausibility and separation of folding from binding. The stage policy is an explicit inference: broad functional testing is more useful than repeating several overlapping stability cautions. Falsify it if a later blinded paired run shows slower early best-seen progress without improved uncertainty resolution.
