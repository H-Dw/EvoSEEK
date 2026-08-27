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
task_applicability: high_for_midround_update
retrieval_similarity: runtime_only
card_id: DC-GB1-V2-002
title: Update one coupling hypothesis from measured associations
inputs:
  stage: middle_round
  positions: [39, 40, 41, 54]
feature: measured_association_conditioned_coupling_test
direction: observations_first_external_boundary_second
boundaries:
  - External structure evidence cannot override current assay observations.
  - Test only one interpretable coupling contrast per matched comparison.
uncertainty: medium
knowledge_type: stage_specific_decision_rule
topics: [middle round, measured update, pairwise coupling, matched comparison]
source_spans: [SS-001, SS-003, SS-004, SS-006]
logic_units: [LU-001, LU-002, LU-004]
---

## Use when

Use after at least one current campaign round has produced measured associations and the next batch must refine rather than restart the search.

## Decision rule

Treat current measured direction and uncertainty as decision-bearing. Use the inherited packing and fold evidence only to choose one plausible pairwise explanation or a counterexample that distinguishes it.

## Candidate action

Exploit the best supported observed mechanism, add a matched alternative that changes one coupling class, and keep one uncertainty probe only when it resolves a named conflict. Do not replay every external card.

## Matched comparison

Fix the remaining background and compare a volume-balanced or fold-compatible pair against one controlled alternative. Attribute the contrast to the tested coupling, not to generic similarity.

## Abstain or downgrade when

Downgrade the external prior when measured evidence disagrees or when the proposed contrast changes multiple mechanisms. Abstain from structural certainty without a mutant structure.

## Evidence basis

Inherited spans support local proximity, packing sensitivity, and the need to preserve backbone compatibility. Giving current observations decision priority is a runtime inference. Falsify this card if matched updates repeatedly underperform a fresh, observation-agnostic search across blinded seeds.
