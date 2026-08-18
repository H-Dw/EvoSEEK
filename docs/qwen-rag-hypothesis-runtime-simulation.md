# Qwen RAG → KG → Scientist hypothesis runtime simulation

## Scope

This audit executed the project-owned hypothesis path with:

- English atomic claims under `resources/local_knowledge/directed_evolution/claims`;
- local SQLite corpus and task overlay;
- Alibaba Model Studio `text-embedding-v4` for document/query vectors;
- Alibaba Model Studio `qwen3-rerank` for second-stage ranking;
- structured RAG-to-KG materialization;
- the exact `scientific_v1` Scientist system/user prompt;
- a real Qwen Scientist completion through the shared workspace API catalog.

The final successful compact run is stored at
`artifacts/rag_api_hypothesis_simulation_compact/`.

## Actual runtime query

The hypothesis-stage KG interaction used the same English query and anchors as
`CampaignRunner._run_kg_interaction`:

```text
general protein structure stability binding mutation physicochemical epistasis knowledge
```

The corpus prefetch used the optimization objective, assay conditions, generic protein-engineering
anchors, and enabled provider names. The target leakage guard removed or generalized protected
target-specific terms before retrieval.

## Real API result

- Corpus: 20 English atomic-claim Markdown documents.
- Stored vectors: 20 vectors, dimension 1024.
- Dense backend: `text-embedding-v4`, DashScope query/document asymmetric mode.
- Reranker: `qwen3-rerank` with an English scientific evidence instruction.
- Returned prefetch chunks: 5.
- Returned prefetch claims: 5.
- All selected chunks passed the configured dense threshold of 0.50.

| Rank | Knowledge type | Dense cosine | Reranker | Atomic fact |
|---:|---|---:|---:|---|
| 1 | history-guided combination | 0.5510 | 0.8034 | Double-mutant measurements can reveal widespread positive and negative pairwise epistasis. |
| 2 | structure context | 0.6018 | 0.6693 | Packing, hydrogen bonding and local interactions require structural context. |
| 3 | structure context | 0.5249 | 0.6460 | Hotspot selection can integrate structural, functional and evolutionary evidence. |
| 4 | substitution conservativeness | 0.5542 | 0.6451 | Site context can override a globally conservative substitution label. |
| 5 | amino-acid properties | 0.5065 | 0.5120 | Hydropathy is one descriptor, not a standalone fitness determinant. |

The explicit hypothesis-stage local query additionally retrieved background-dependent epistasis
and indirect adaptive paths. Across prefetch and interaction, seven unique local claims entered the
round-visible structured KG.

## RAG-to-KG representation

The successful run materialized 42 local-knowledge entities and 70 provenance-preserving
relations:

| Entity type | Count | Role |
|---|---:|---|
| `Document` | 7 | Source Markdown artifact, file hash and knowledge type |
| `DocumentChunk` | 7 | Exact text span plus lexical, dense, RRF and reranker scores |
| `Claim` | 7 | Atomic subject–predicate–object scientific statement |
| `Evidence` | 7 | Round-visible local-RAG evidence with applicability and warnings |
| `Publication` | 7 | Normalized DOI/title/authors/year/venue metadata |
| `CitationSupport` | 7 | Claim-to-publication support type, locator and verification status |

The relation vocabulary was:

```text
Document -HAS_CHUNK-> DocumentChunk
DocumentChunk -ASSERTS-> Claim
Claim -SUPPORTED_BY_SOURCE-> Evidence
Claim -SUPPORTED_BY_CITATION-> CitationSupport
CitationSupport -CITES_PUBLICATION-> Publication
CitationSupport -DERIVED_FROM-> DocumentChunk
Evidence -DERIVED_FROM-> DocumentChunk
```

Relations from prefetch and the explicit hypothesis query retain different `query_id` context,
even when they link the same claim and chunk. This is intentional retrieval provenance rather than
an accidental duplicate. A source-ID defect that produced `localdoc:localdoc:*` was found during
the simulation and fixed in both the local evidence service and KG adapter.

## Exact Scientist prompt composition

The runtime prompt is built from:

1. the versioned `scientific_v1` role profile and `HypothesisOutput` JSON schema;
2. sanitized round state and visible measurements only;
3. the `hypothesis_context` EvidencePack from the observation KG;
4. the `query_local_knowledge` EvidencePack containing RAG facts/evidence;
5. the same local Evidence records as allowed citable evidence IDs.

The original full-provenance prompt was 106,119 user-message characters because RAG evidence and
embedding fingerprints were duplicated. The prompt projection now keeps evidence IDs, statements,
scores, artifact locations, manifest identity, policy decisions and warnings while omitting repeated
backend fingerprints. The successful compact prompt was 53,411 user-message characters. Full
provenance remains in the audit artifacts and KG; only the LLM view is compacted.

## Actual Qwen hypothesis

The real Qwen Scientist completion proposed preferred residues `39:W`, `40:K`, `41:Y`, `54:A`.
Its central claim was that K40 and A54 are supported by visible residue-level aggregates, while the
generic RAG evidence requires the combination to be treated as background-dependent because of
epistasis. It proposed testing `WKYA` against wild type and the visible D40K benchmark.

The completion cited five local-RAG evidence IDs. All five were members of the seven IDs supplied
to the prompt; no fabricated evidence ID was accepted by the output validator.

The interpretation boundary is important:

- local RAG supplied generic mechanisms and cautions;
- visible observation KG aggregates supplied residue-specific preferences;
- RAG relevance scores were not interpreted as fitness effects;
- the LLM combined both sources into a falsifiable hypothesis.

## How this affects selection

`kg_update.enabled: true`, `materialization: retrieved_only`, `allow_remote_context: true`, and the
enabled `query_local_knowledge` operator make local claims visible to the remote Scientist.

`kg_update.contributes_to_selection` remains `false`, so the raw local Evidence has `score=0.0` and
does not enter the candidate evidence average. However, the influence is still indirect: the LLM's
`preferred_residues` becomes `hypothesis_score` in `AgentUncertaintySelector`, and that score is
weighted by `generation.hypothesis_weight`. Thus context-only RAG can affect selection through the
validated hypothesis even when it cannot act as a direct candidate fitness prior.

Direct numerical RAG contribution must remain fail-closed until a
`local-rag-selection-calibration:v1` file has `status: validated`, leakage-safe validation metadata,
and rules linked to selection-eligible source-verified claims.

## Unified API catalog

`configs/api/aliyun-qwen-workspace.yaml` now separates shared connection data from typed items:

- `scientist_llm`: OpenAI-compatible chat base URL and LLM model;
- `rag_embedding`: DashScope embedding endpoint and embedding contract;
- `rag_reranker`: qwen3-rerank endpoint and reranking contract.

All items share the same workspace origin, region and `env:DASHSCOPE_API_KEY` reference. Each item
retains its own protocol, operation path, model, limits, instruction and response contract. Thin
item-reference YAML files select the required item without duplicating the host or key reference.

## External API status

Multiple real calls completed successfully, including embedding, reranking and two Qwen Scientist
completions. A later repeat received `403 AccessDenied.Unpurchased` from the embedding model. This
is an external workspace/model entitlement state, not a request-format error. Verify model
entitlement and rotate the exposed key before the next paid run.
