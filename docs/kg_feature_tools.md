# KG feature-tool and mutation-effect design

## Evidence flow

The three feature providers return the common `Evidence` contract. They are
descriptors or priors, not direct assay-fitness labels. The structured-KG build
materializes the following typed projections while retaining a
`DERIVED_FROM -> Evidence` provenance edge:

| Channel | Tool | KG projection rooted at `Mutation` | LLM-visible fields |
| --- | --- | --- | --- |
| Physchem | `query_physchem_delta` | `HAS_PHYSCHEM_DELTA -> SubstitutionDescriptor`; residue values are `ResidueType -> HAS_DESCRIPTOR -> PhyschemPropertyValue` | site deltas, WT/mutant property values, global sequence deltas |
| Conservation | `query_evolutionary_profile` | `HAS_EVOLUTIONARY_CONTEXT -> EvolutionProfile` | site frequency, log odds, entropy/information content, coverage, effective count, MSA Neff and Neff/L, pairwise eligibility, smoothing provenance |
| Structure | `query_structure_environment` | `OCCURS_IN_ENVIRONMENT -> ResidueEnvironment` | contacts, SASA, coarse backbone state, static interaction candidates |

`query_feature_bundle` retrieves any configured subset jointly and reports a
per-channel availability map plus the child query IDs. Every operation is still
round-scoped, read-only, allow-listed, row-bounded, and auditable. Raw SQL and
oracle/final-test arguments remain forbidden.

The Scientist prompt receives a bounded allow-list of those feature fields.
Large backend/cache payloads are excluded. Consequently, activating a tool can
change an LLM hypothesis only through explicit `Evidence` included in the
round's `kg_interaction` result; it does not automatically change a fitness
score. Selection influence still requires the provider's explicit,
visible-data-only calibration and `contributes_to_selection` policy.

## Conservation provider strategy

The recommended GB1 provider uses the supplied single-chain
`non_pairing.a3m` as a weighted homolog sample. It calculates canonical
single-site frequencies after sequence-identity reweighting, records effective
counts and information content per mutable position, and evaluates a mutation
with WT-relative single-site log odds. The sum and mean per mutated position
are both retained so downstream code does not have to infer mutation-depth
normalization from one scalar.

The configured smoothing mode is `neff_scaled_uniform`: a total prior mass of
`pseudocount_weight * Neff` is distributed over q states for single columns or
q² states for pairs. This prevents pair pseudocount mass from growing by a
factor of 20 merely because the state space changed from q to q². The legacy
`pseudocount` option remains supported as an explicit per-state compatibility
mode for older configs.

Pairwise evidence is a separate, gated capability. Raw pair frequencies are
not DCA and are not labelled epistasis. The GB1 example has Neff/L about 0.27,
so `pairwise_enabled: false`; its evidence score is single-site only. If a
future alignment clears the configured depth gate, the available
`marginal_corrected_log_odds` mode reports a residual co-occurrence descriptor
with a `not_direct_coupling` warning. A genuine Potts/DCA implementation should
use a distinct provider kind rather than silently changing this descriptor's
meaning.

The config contains a machine-readable `estimated_parameters` list and
`parameter_annotations`. These distinguish the literature-precedent but
task-dependent identity threshold from useful, currently estimated values such
as smoothing weight, effective-count cutoffs, coverage/gap filters, and the
pairwise Neff/L gate. All remain uncalibrated and do not contribute directly to
selection.

## Keyword truncation audit

Set `kg_interaction.truncation_audit_enabled: true` and list literal search
terms under `truncation_audit_items`. The `query_kg_truncation_audit` operator
counts all round-visible entity and relation matches before applying
`max_rows`, returns only a bounded sample, and classifies each item as
`complete`, `truncated`, or `not_found`. Its facts and caveats are passed in the
same bounded `kg_interaction` context as the feature evidence, and a full
diagnostic report is written to
`round_XX/kg_truncation_audit.json`.

For an existing run, the same check can be repeated without invoking an LLM:

```powershell
.\.venv\Scripts\python.exe scripts/diagnostics/audit_kg_truncation.py `
  <run-dir>\structured_kg.sqlite --round-id 1 --max-rows 12 `
  --item physchem --item HAS_PHYSCHEM_DELTA
```

The report also checks whether each item is already visible in the non-audit
tool packs. Channel names normally appear in their feature evidence. Exact KG
predicate names may appear only in the audit pack because the feature tools
return typed `Evidence`, rather than dumping raw relation rows.

## Agent-loop strategies

`kg_interaction.feature_tool_strategy` supports:

- `context_only`: no feature-specific tool call (backward-compatible default).
- `independent`: one tool call per enabled channel and representative variant.
- `joint`: one `query_feature_bundle` call per representative variant.
- `independent_and_joint`: both paths, intended for audits and parity tests.

Set `feature_channels`, `feature_variant_limit`, and a sufficient
`max_tool_calls`. The config validator fails early instead of silently dropping
a requested feature channel. See
`configs/experiments/knowledge_agent_features.example.yaml` for the
deterministic offline harness and
`configs/experiments/knowledge_agent_features.deepseek.example.yaml` for the
remote Scientist/Critic path. The latter requires `DEEPSEEK_API_KEY`; both
configs pass the same bounded feature and audit packs into the Scientist
context.

## Mutation effects and epistasis

`MutationEffectEstimate` is created only when a mutation-bearing variant and
the exactly matched background variant both have visible observations:

`delta = fitness(child) - fitness(background)`.

Pairwise `EffectEstimate` is created only when all four variants in a matched
background square are observed:

`epistasis = f11 - f10 - f01 + f00`.

No marginal average or missing-background imputation is used. `ParentVariant`
is represented by the existing `Variant` entity in the semantic role targeted
by `IN_BACKGROUND`, avoiding a duplicate identity node.

## GB1 example resources

The example uses the official RCSB 1PGB mmCIF (unmutated GB1, chain A) and a
hash-pinned path under `examples/gb1`. The reference sequence comes from
`rcsb_pdb_1PGB.fasta`. The single-chain conservation provider uses
`msa/0/non_pairing.a3m` through the first-class `a3m_path` field; with the
example filters it retains 39 sequences and has Neff 15.13 (Neff/L 0.27).
Consequently, the recommended configuration enables only the single-site
profile and records pairwise evidence as disabled. `pairing.a3m` is kept for
paired-MSA workflows, while `hmmsearch.a3m` is a template-hit set and is not
used as the primary conservation alignment.

RCSB entry: <https://www.rcsb.org/structure/1PGB>.
