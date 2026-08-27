# V3 decision log

- Used only the blinded v2 aggregate receipt and retrieval coverage metadata.
- Added no publication, mutation identity, or per-variant label.
- Preserved all v2 DecisionCard text and inherited evidence relationships.
- Assigned one distinct `knowledge_type` to each stage card so the runtime can filter before ranking.
- Kept top-k at one and preserved the same embedding, reranker, LLM, seeds, budget, truth adapter, and strict success threshold.
- Kept Kermut, OpenAlex, quarantined target-label sources, dataset mirrors, and virus-protein material excluded.
- This is the third and final preregistered knowledge version.
