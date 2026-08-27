# Decision card schema

Each card is a Markdown file with YAML frontmatter and six required body sections.

## Required frontmatter

```yaml
schema_version: gb1-directive-card:v1
record_type: knowledge_decision_card
language: en
status: research_candidate
human_reviewed: false
selection_eligible: false
permission: explanation_only
benchmark_overlap: none
scientific_credibility: high
task_applicability: high
retrieval_similarity: runtime_only
card_id: DC-GB1-001
title: Short action-oriented title
inputs:
  positions: [41, 54]
  structural_context: fold
feature: backbone_compatibility
direction: preserve
boundaries:
  - Does not predict binding improvement by itself.
uncertainty: medium
knowledge_type: structural_context
topics: [folding, packing, matched comparison]
source_spans: [SS-001]
logic_units: [LU-001]
```

## Required body sections

- `## Use when`: observable trigger and required context.
- `## Decision rule`: one compact conditional rule.
- `## Candidate action`: residue-class or comparison guidance, not a benchmark-labelled substitution.
- `## Matched comparison`: a small comparison that isolates the hypothesis.
- `## Abstain or downgrade when`: missing context, conflict, transfer risk, or confounding.
- `## Evidence basis`: direct sources, inference boundary, and falsifier.

## Card quality rules

- Aim for 120–260 words of body text.
- One card should implement one decision.
- State whether the rule concerns folding, stability, binding, or experimental design.
- Do not encode exact preferred benchmark substitutions or numeric fitness values.
- Do not turn a stability observation into a binding prediction without an explicit inference boundary.
- Prefer a matched comparison and a falsifier over generic advice.
