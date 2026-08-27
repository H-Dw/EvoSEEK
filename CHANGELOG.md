# Changelog

All notable changes to EvoSEEK are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Added a provider-neutral, two-stage Researcher agent. Phase A emits bounded
  support/counterevidence/boundary retrieval plans; Phase B requests only allow-listed
  feature projections for visible samples. Strict schemas, semantic retries, per-round
  budgets, abstention, failure receipts, and profile/schema hashes make each plan auditable.
- Added a Deep Research evidence-product pipeline covering research briefs, scoped search
  routes, accepted search hits, publications, source spans, evidence groups, atomic claims,
  logic units, decision cards, review receipts, release approvals, manifests, and native or
  legacy Local RAG exports.
- Added HMAC attestations with versioned policy, reviewer, and release keyrings. The CLI now
  supports policy inspection, planning, search, validation, approval, manifest creation, and
  deterministic runtime export without embedding key material in the repository.
- Added native typed Local RAG records (`AtomicClaim`, `LogicUnit`, and
  `KnowledgeDecisionCard`), faceted retrieval, round-specific routes, multi-query retrieval,
  and KG materialization that preserves citation entities and provenance relations.
- Added Agentic RAG, cold-start, no-RAG, and directive-RAG benchmark configurations,
  leakage-safe candidate bundles, canary/probe scripts, paired comparison scripts, and
  structural integration tests.
- Added a deterministic Batch Critic policy gate with optional semantic-risk escalation to a
  remote critic, plus audited fail-closed behavior when required semantic review is unavailable.

### Changed

- The campaign orchestrator can run a two-stage Researcher once per round, route the returned
  queries through Local RAG and feature tools, inject typed `rag_records` exactly once into the
  Scientist prompt, and persist accepted or rejected round receipts.
- Knowledge configuration now supports ordered YAML layers so feature providers, retrieval
  policy, and experiment-specific Agentic RAG settings can evolve independently.
- Local retrieval now separates relevance, scientific quality, task applicability, permission,
  boundary conditions, counterclaims, and abstention rules. Retrieval similarity never grants
  selection authority.
- New indexes use `local-knowledge-index:v7`; v6 remains readable for the legacy projection
  path. Native indexes store record facets and runtime-manifest security bindings.
- Feature tools accept bounded position/focus projections while preserving compatibility with
  legacy tool adapters when no projection is requested.
- Candidate scoring now counts only genuine non-wild-type edits toward hypothesis matches and
  excludes explanation-only evidence from selection scoring.
- Cold-start campaigns may use a preregistered coverage-exploration batch when WT-only evidence
  cannot support a directional hypothesis, instead of fabricating a residue preference.

### Security

- Added workspace access-policy enforcement from `AvoidRead.txt` and the matching editor rule.
  Denied paths are rejected before metadata access, and symlink/junction traversal is blocked.
- Production knowledge roots now require an exact `runtime-files.json` manifest bound to the
  active workspace and external-evidence policies. Arbitrary recursive corpus discovery is
  disabled for production mode.
- Released runtime projections are verified against the complete signed evidence product and
  rerendered byte-for-byte before indexing. Detached manifests, stale receipts, unknown keys,
  mixed or unknown source scope, and missing human release approval fail closed.
- Search and ingestion enforce nonviral scope, protected-identity filtering, query budgets, and
  explicit quarantine instead of treating keyword matches or retrieval rank as authorization.

### Compatibility notes

- Existing production Local RAG roots must be exported as a signed runtime release before use.
  Synthetic tests and benchmark candidates must opt into `legacy_compatible` explicitly and
  remain explanation-only.
- The Scientist prompt contract now uses `rag_records`; consumers that read the former
  `rag_claims` field must migrate to the typed record contract.

### Baseline

- Compared against GitHub `origin/main` commit
  `d3135ce5db3feede6500d629937f64b22a356df3` (`Harden live ReThink API execution`).
