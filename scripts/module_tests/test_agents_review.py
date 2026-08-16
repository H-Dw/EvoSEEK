from __future__ import annotations

import os
from dataclasses import replace

from common import (
    ensure,
    load_config,
    parse_args,
    placeholder,
    resolve_output,
    write_legacy_benchmark,
    write_result,
)

from fitness_agents.agents.critic import CriticAgent, OpenAICriticClient, RuleBasedCriticClient
from fitness_agents.agents.llm import MockScientistLLMClient, OpenAICompatibleLLMClient
from fitness_agents.agents.remote_llm import load_project_env, resolve_secret
from fitness_agents.agents.scientist import ScientistAgent, assert_sanitized
from fitness_agents.config import CriticConfig, KnowledgeConfig, ModelConfig, TaskConfig
from fitness_agents.contracts.schemas import CampaignState, ReviewVerdict
from fitness_agents.data import load_dataset_bundle
from fitness_agents.evaluation.hypotheses import (
    DeterministicHypothesisEvaluator,
    preregister_batch_median_test,
)
from fitness_agents.knowledge import KnowledgeEngine
from fitness_agents.loop.backends import ApprovalEnforcingBackend, CsvOracleBackend
from fitness_agents.loop.review import BoundedReviewLoop
from fitness_agents.models import create_predictor
from fitness_agents.utils.progress import configure_progress_logging
from fitness_agents.validation.batch import BatchHardValidator, build_draft_batch


def _configure_remote(values: dict[str, object]) -> None:
    load_project_env()
    api_key = resolve_secret(
        values.get("api_key"),
        "FITNESS_AGENTS_LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    )
    scientist_model = resolve_secret(
        values.get("scientist_model"), "FITNESS_AGENTS_LLM_MODEL"
    ) or values.get("scientist_model")
    critic_model = resolve_secret(
        values.get("critic_model"), "FITNESS_AGENTS_CRITIC_MODEL", "FITNESS_AGENTS_LLM_MODEL"
    ) or values.get("critic_model")
    base_url = resolve_secret(
        values.get("base_url"),
        "FITNESS_AGENTS_LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "DEEPSEEK_BASE_URL",
    )
    ensure(bool(api_key), "Replace remote_llm.api_key or set DEEPSEEK_API_KEY before remote testing")
    ensure(
        bool(scientist_model) and not placeholder(scientist_model),
        "Replace remote_llm.scientist_model before remote testing",
    )
    ensure(
        bool(critic_model) and not placeholder(critic_model),
        "Replace remote_llm.critic_model before remote testing",
    )
    ensure(
        bool(base_url) and not placeholder(base_url),
        "Replace remote_llm.base_url or set FITNESS_AGENTS_LLM_BASE_URL before remote testing",
    )
    os.environ["FITNESS_AGENTS_LLM_API_KEY"] = str(api_key)
    os.environ["DEEPSEEK_API_KEY"] = str(api_key)
    os.environ["OPENAI_BASE_URL"] = str(base_url)
    os.environ["FITNESS_AGENTS_LLM_BASE_URL"] = str(base_url)
    os.environ["FITNESS_AGENTS_LLM_MODEL"] = str(scientist_model)
    os.environ["FITNESS_AGENTS_CRITIC_MODEL"] = str(critic_model)
    if values.get("reasoning_effort"):
        os.environ["FITNESS_AGENTS_LLM_REASONING_EFFORT"] = str(values["reasoning_effort"])
    if values.get("thinking"):
        os.environ["FITNESS_AGENTS_LLM_THINKING"] = str(values["thinking"])
    if values.get("max_tokens"):
        os.environ["FITNESS_AGENTS_LLM_MAX_TOKENS"] = str(values["max_tokens"])
    values["api_key"] = str(api_key)
    values["scientist_model"] = str(scientist_model)
    values["critic_model"] = str(critic_model)
    values["base_url"] = str(base_url)


def main() -> None:
    args = parse_args("configs/module_tests/agents_review.yaml", remote=True)
    config = load_config(args.config)
    output = resolve_output(config, args.output_dir)
    paths = write_legacy_benchmark(output / "input", seed=int(config["seed"]))
    bundle = load_dataset_bundle(paths["public"], paths["oracle"])
    candidates = bundle.oracle_pool[:6]

    model = create_predictor(
        ModelConfig(
            ridge_members=3,
            extra_trees_estimators=24,
            ridge_alpha=5.0,
        ),
        seed=int(config["seed"]),
    )
    model.fit(
        bundle.initial_variants,
        bundle.initial_observations,
        bundle.validation_variants,
        bundle.validation_observations,
    )
    predictions = model.predict(candidates)
    prediction_map = {item.variant_id: item for item in predictions}

    knowledge = KnowledgeEngine(
        KnowledgeConfig(),
        graph_path=output / "agent_knowledge.sqlite",
        assay_id="module_test_assay",
    )
    knowledge.update(bundle.initial_variants, bundle.initial_observations)
    evidence = knowledge.evidence_for(candidates, round_id=1)
    knowledge.record_inference_context(candidates, predictions, evidence, round_id=1)
    state = CampaignState("module-agent-review", "knowledge_agent", int(config["seed"]), round_id=1)
    scientist = ScientistAgent(
        MockScientistLLMClient(),
        knowledge_graph=knowledge.agent_tool(max_rows=6),
    )
    hypothesis = scientist.propose_hypothesis(
        state,
        bundle.initial_variants,
        bundle.initial_observations,
        [item for bundle_items in evidence.values() for item in bundle_items],
    )
    ensure(hypothesis.preferred_residues, "Scientist did not produce residue preferences")
    ensure(hypothesis.falsification_criterion, "Scientist hypothesis is not falsifiable")
    ensure(scientist.last_knowledge_query_id is not None, "Scientist did not query the safe KG")
    explanation = scientist.inspect_variant(candidates[0].variant_id, round_id=1)
    ensure(explanation["found"], "Scientist variant inspection failed")

    sanitization_guard = False
    try:
        assert_sanitized({"nested": {"oracle_path": "hidden.csv"}})
    except ValueError as error:
        sanitization_guard = "Forbidden hidden-label" in str(error)
    ensure(sanitization_guard, "Scientist hidden-label sanitizer did not fire")

    selected_ids = tuple(item.variant_id for item in candidates[: int(config["batch_budget"])])
    falsification = preregister_batch_median_test(
        hypothesis_id=hypothesis.hypothesis_id,
        round_id=1,
        target_variant_ids=selected_ids,
        visible_observations=bundle.initial_observations,
    )
    task = TaskConfig(
        task_id="module_agent_review",
        protein_id="GB1",
        assay_id="module_test_assay",
        wild_type_sites="VDGV",
        mutable_positions=[39, 40, 41, 54],
        objective="maximize",
        public_data_path=paths["public"],
        oracle_data_path=paths["oracle"],
    )
    critic_config = CriticConfig(**config["critic"])
    validator = BatchHardValidator(task, critic_config)
    review_loop = BoundedReviewLoop(
        validator=validator,
        critic=CriticAgent(RuleBasedCriticClient(), max_retries=0),
        max_revision_attempts=int(config["max_revision_attempts"]),
    )
    variants = {item.variant_id: item for item in candidates}

    def draft_builder(attempt: int, parent_id: str | None, exclusions: set[str]):
        eligible = [identifier for identifier in selected_ids if identifier not in exclusions]
        return build_draft_batch(
            round_id=1,
            review_attempt=attempt,
            candidate_ids=eligible,
            variants=variants,
            predictions=prediction_map,
            evidence=evidence,
            hypothesis_id=hypothesis.hypothesis_id,
            falsification_spec=None if attempt == 0 else falsification,
            parent_draft_batch_id=parent_id,
        )

    review = review_loop.run(
        draft_builder=draft_builder,
        variants=variants,
        predictions=prediction_map,
        evidence=evidence,
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids=set(variants),
        expected_batch_size=len(selected_ids),
    )
    ensure(
        [item.verdict for item in review.attempts] == [ReviewVerdict.REVISE, ReviewVerdict.APPROVE],
        "Bounded revise-then-approve flow was not exercised",
    )

    raw_backend = CsvOracleBackend(paths["oracle"], query_budget=len(selected_ids))
    approved_backend = ApprovalEnforcingBackend(raw_backend, review_loop.gateway)
    experiment_id = approved_backend.submit(review.approved_batch)
    revealed = approved_backend.collect(experiment_id)
    ensure(len(revealed) == len(selected_ids), "Approved experiment did not reveal the full batch")
    receipt_guard = False
    try:
        approved_backend.submit(
            replace(review.approved_batch, approval_receipt_hash="0" * 64)
        )
    except PermissionError:
        receipt_guard = True
    ensure(receipt_guard, "Tampered approval receipt was accepted")

    assessment = DeterministicHypothesisEvaluator().evaluate(
        spec=falsification,
        observations=[*bundle.initial_observations, *revealed],
        round_id=1,
    )
    ensure(assessment.criterion_results, "Hypothesis evaluator returned no criterion results")

    remote_values = config["remote_llm"]
    remote_enabled = bool(args.enable_remote or remote_values.get("enabled", False))
    remote_result: dict[str, object] = {"enabled": False, "status": "skipped"}
    if remote_enabled:
        configure_progress_logging()
        _configure_remote(remote_values)
        remote_scientist = ScientistAgent(
            OpenAICompatibleLLMClient(
                model=str(remote_values["scientist_model"]),
                base_url=str(remote_values["base_url"]),
                provider="deepseek",
                reasoning_effort=str(remote_values.get("reasoning_effort") or "high"),
                thinking=str(remote_values.get("thinking") or "enabled"),
            ),
            knowledge_graph=knowledge.agent_tool(max_rows=6),
        )
        remote_hypothesis = remote_scientist.propose_hypothesis(
            state,
            bundle.initial_variants,
            bundle.initial_observations,
            [item for bundle_items in evidence.values() for item in bundle_items],
        )
        remote_critic = CriticAgent(
            OpenAICriticClient(
                model=str(remote_values["critic_model"]),
                profile=critic_config.profile,
                temperature=critic_config.temperature,
                base_url=str(remote_values["base_url"]),
                provider="deepseek",
                reasoning_effort=str(remote_values.get("reasoning_effort") or "high"),
                thinking=str(remote_values.get("thinking") or "enabled"),
            ),
            max_retries=critic_config.max_model_retries,
            fallback=RuleBasedCriticClient(),
        )
        remote_decision = remote_critic.review(
            draft=review.draft,
            variants=variants,
            predictions=prediction_map,
            evidence=evidence,
            conflict_report=review.report,
        )
        remote_result = {
            "enabled": True,
            "status": "passed",
            "provider": "deepseek",
            "scientist_model": str(remote_values["scientist_model"]),
            "critic_model": str(remote_values["critic_model"]),
            "hypothesis_id": remote_hypothesis.hypothesis_id,
            "hypothesis_has_falsification": bool(remote_hypothesis.falsification_criterion),
            "critic_verdict": remote_decision.verdict,
        }

    knowledge.close()
    write_result(
        output,
        "agents_review",
        {
            "config": config["_config_path"],
            "scientist": {
                "hypothesis_id": hypothesis.hypothesis_id,
                "evidence_ids": hypothesis.evidence_ids,
                "knowledge_query_id": scientist.last_knowledge_query_id,
            },
            "review_verdicts": [item.verdict for item in review.attempts],
            "approved_batch_id": review.approved_batch.draft_batch_id,
            "revealed_count": len(revealed),
            "hypothesis_status": assessment.status,
            "guards": {
                "hidden_label_sanitized": sanitization_guard,
                "tampered_receipt_rejected": receipt_guard,
            },
            "remote_llm": remote_result,
        },
    )


if __name__ == "__main__":
    main()

