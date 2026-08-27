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
card_id: DC-GB1-002
title: Balance packing jointly at positions 39 and 54
inputs:
  positions: [39, 54]
  structural_context: contacting_side_chains
feature: paired_side_chain_volume_and_hydrophobicity
direction: balance_jointly
boundaries:
  - Wild-type contact geometry does not determine the best mutant pair.
  - Large-to-small and small-to-large changes may repack differently.
uncertainty: medium
knowledge_type: sequence_interaction_context
topics: [epistasis, paired packing, side-chain volume, positions 39 and 54]
source_spans: [SS-001, SS-003, SS-006]
logic_units: [LU-002]
---

## Use when

Ranking candidates that alter both positions 39 and 54, or when independent per-position scores favor two large side chains without considering their shared space.

## Decision rule

Score 39/54 as a pair: reward compatible hydrophobic packing and penalize uncompensated joint volume increase or loss of core-like contact.

## Candidate action

Retain multiple paired classes: a near-native volume pair, a redistributed-volume pair with one larger and one smaller member, and a conservative hydrophobic pair. Do not collapse the batch onto one aromaticity or size class.

## Matched comparison

Within a fixed 40/41 background, compare pairs with similar total hydrophobic character but different volume balance. Add a swap-style comparison to distinguish site identity from combined composition.

## Abstain or downgrade when

Downgrade if structural modeling predicts a backbone shift, solvent exposure, or an alternative rotamer network. Abstain from declaring a pair optimal from distance alone.

## Evidence basis

The official 1PGA coordinates place the wild-type side chains at 39 and 54 within direct heavy-atom contact range. Domain-wide stability work also warns that core predictions are less reliable than boundary predictions. The rule is falsified if volume-balanced pairs show no advantage over matched overcrowded or cavity-forming pairs across backgrounds.
