# Paired benchmark protocol

## Frozen comparison

- Task: GB1 benchmark truth.
- Conditions: no local RAG versus the current versioned directive RAG bundle.
- Seeds: `11, 23, 37, 53, 71`.
- Alternate condition execution order by seed.
- Keep model providers, prompts, budgets, rounds, candidate limits, evaluators, and all non-local-knowledge settings identical.
- Candidate generation uses the existing DeepSeek/Qwen runtime. GPT-5.6 Sol does not generate candidates.
- Kermut, active-learning predictors, model fallback, and target-label retrieval are forbidden.

## Blinded feedback

Store and expose to research iteration only:

- final best-seen fitness per run;
- mean of round-level best-seen fitness as AUC proxy;
- paired aggregate deltas;
- run completion and fallback status;
- retrieval count and intended-card hit status.

Do not expose mutation identities, per-variant labels, benchmark winners, or losing substitutions to the research/Skill iteration loop.

## Predeclared success

The RAG version passes only if all are true:

1. median paired delta in final best-seen fitness is greater than zero;
2. mean paired delta in the AUC proxy is greater than zero;
3. no run uses fallback, Kermut, a generation predictor, or a quarantined source;
4. at least four of five RAG runs retrieve an intended decision card.

If a version fails, diagnose only from aggregate metrics, retrieval traces, card coverage, and runtime integrity. Create a new versioned bundle/index and repeat. Stop after three RAG versions and report the result without moving the success threshold.
