"""One-shot full-reference sequence design, independent of benchmark candidate pools."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from fitness_agents.active_learning import create_active_learning_module
from fitness_agents.agents.client_registry import create_role_client_bundle
from fitness_agents.agents.critic import (
    CriticAgent,
    create_batch_critic_agent,
)
from fitness_agents.agents.output_guards import RevisionConstraints
from fitness_agents.agents.scientist import ScientistAgent
from fitness_agents.config import ExperimentConfig
from fitness_agents.contracts.agent_io import RoleActivationState
from fitness_agents.contracts.batch_review import (
    BatchReviewContext,
    CandidateIntentCard,
    RevisionQuotaShortfallReceipt,
    prediction_review_card,
    soft_prior_mismatch_ids,
)
from fitness_agents.contracts.design import (
    RankedSequenceDesign,
    ResolvedDesignSpace,
    SequenceProposal,
)
from fitness_agents.contracts.schemas import (
    CampaignPhase,
    CampaignState,
    Evidence,
    FitnessObservation,
    Prediction,
    Variant,
)
from fitness_agents.data import load_open_design_initial_bundle
from fitness_agents.evaluation.hypotheses import preregister_batch_median_test
from fitness_agents.knowledge import KnowledgeEngine
from fitness_agents.models import create_predictor
from fitness_agents.mutation import (
    create_open_design_proposer,
    normalize_visible_variants,
    resolve_design_space,
)
from fitness_agents.protein_features import ProteinTaskContext
from fitness_agents.utils import (
    JsonArtifactWriter,
    bind_progress,
    reset_progress,
    seed_everything,
)
from fitness_agents.validation import ApprovalGateway, OpenDesignHardValidator, build_draft_batch

from .review import BoundedReviewLoop, RevisionConstraintInfeasible


def _flatten_evidence(evidence: dict[str, list[Evidence]], limit: int = 120) -> list[Evidence]:
    entries = [item for values in evidence.values() for item in values]
    return sorted(
        entries,
        key=lambda item: (
            item.quality_status != "unavailable",
            item.contributes_to_selection,
            item.confidence * abs(item.score),
            item.evidence_id,
        ),
        reverse=True,
    )[:limit]


def _hypothesis_prior(
    proposal: SequenceProposal, preferred_residues: dict[int, tuple[str, ...]]
) -> float:
    if not proposal.edits:
        return 0.0
    matches = [
        edit.mutant in preferred_residues.get(edit.position, ())
        for edit in proposal.edits
        if edit.position in preferred_residues
    ]
    return float(sum(matches) / len(matches)) if matches else 0.0


def _structure_constraint(evidence: Sequence[Evidence]) -> float:
    """Return a non-positive static-context penalty, never a fitness estimate."""

    risks = [
        max(0.0, -float(item.score))
        for item in evidence
        if item.channel == "structure" and item.quality_status == "ok"
    ]
    return -float(sum(risks))


def _not_evaluated_prediction(variant_id: str) -> Prediction:
    """Explicit no-posterior marker: NaN fitness fields, never a fitness estimate."""

    nan = float("nan")
    return Prediction(
        variant_id=variant_id,
        fitness_mean=nan,
        fitness_std=nan,
        interval_90=(nan, nan),
        ood_score=nan,
        component_scores={},
        model_version="none:knowledge_only_no_posterior",
    )


class OpenDesignRunner:
    """Generate, score, and export de novo full-sequence single mutants.

    This runner has no experiment backend and no candidate-pool attribute.  It consumes
    only already-visible observations, then creates its own search space from the
    configured reference sequence.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        predictor_factory: Callable[..., Any] = create_predictor,
        agent: ScientistAgent | None = None,
        critic_agent: CriticAgent | None = None,
        resolved_design_space: ResolvedDesignSpace | None = None,
        initial_variants: Sequence[Variant] | None = None,
        initial_observations: Sequence[FitnessObservation] | None = None,
    ) -> None:
        if config.designer.space != "open_design":
            raise ValueError("OpenDesignRunner requires designer.space=open_design")
        if (initial_variants is None) != (initial_observations is None):
            raise ValueError("Inject both initial_variants and initial_observations, or neither")
        self.config = config
        self.rng = seed_everything(config.seed)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        label = f"-{config.run_label}" if config.run_label else ""
        self.run_id = f"open-design-s{config.seed}{label}-{timestamp}"
        self.writer = JsonArtifactWriter(config.output_root, self.run_id)
        self.source_context = ProteinTaskContext.from_task(config.task)
        self.computation_context = self.source_context.for_open_design()
        # Compatibility alias: callers that need sequence features still receive the
        # full computation context. Mutation authority lives only in design_space.
        self.task_context = self.computation_context
        self.design_space = resolved_design_space or resolve_design_space(
            self.computation_context, config.designer
        )
        if (
            self.design_space.reference_sequence != self.computation_context.full_sequence
            or self.design_space.computation_positions
            != tuple(self.computation_context.mutable_positions)
            or self.design_space.position_to_sequence_index
            != self.computation_context.position_to_sequence_index
            or self.design_space.position_policy != config.designer.position_policy
            or self.design_space.policy_include_positions
            != tuple(config.designer.include_positions)
            or self.design_space.policy_exclude_positions
            != tuple(config.designer.exclude_positions)
            or self.design_space.allowed_residues
            != tuple(config.designer.allowed_residues)
            or self.design_space.proposer != config.designer.proposer
            or self.design_space.mutation_depth != config.designer.mutation_depth
        ):
            raise ValueError(
                "Injected resolved design space does not match the trusted open-design config"
            )
        self.resolved_positions = self.design_space.allowed_mutation_positions
        self.proposer = create_open_design_proposer(
            config.designer, self.computation_context, self.design_space
        )
        # The posterior is fitted only in active-learning mode; with
        # active_learning disabled the runner ranks generated sequences from
        # knowledge evidence alone and needs no labeled observations at all.
        self.posterior_enabled = bool(config.active_learning.enabled)
        if initial_variants is None:
            has_measurement_source = (
                config.task.split_root is not None
                or config.task.initial_observations_path is not None
                or (
                    config.task.public_data_path is not None
                    and config.task.oracle_data_path is not None
                )
            )
            if has_measurement_source:
                initial = load_open_design_initial_bundle(
                    split_root=config.task.split_root,
                    fold_index=config.task.fold_index,
                    public_path=config.task.public_data_path,
                    oracle_path=config.task.oracle_data_path,
                    initial_path=config.task.initial_observations_path,
                )
                raw_variants = initial.variants
                raw_observations = initial.observations
                self.initial_data_source = initial.source
            else:
                raw_variants = []
                raw_observations = []
                self.initial_data_source = "none_configured"
        else:
            raw_variants = list(initial_variants)
            raw_observations = list(initial_observations or ())
            self.initial_data_source = "injected_visible_observations"
        self.observed_variants, self.observations = normalize_visible_variants(
            raw_variants,
            raw_observations,
            source_context=self.source_context,
            open_context=self.task_context,
        )
        if self.posterior_enabled and len(self.observed_variants) < 4:
            raise ValueError(
                "Open-design posterior requires at least four visible observations; "
                "provide initial observations or set active_learning.enabled=false "
                "for knowledge-only ranking without labels"
            )
        self.predictor_factory = predictor_factory
        self.active_learning = create_active_learning_module(
            config.active_learning,
            fallback_model=config.model,
            predictor_factory=predictor_factory,
            seed=config.seed,
        )
        self.knowledge = KnowledgeEngine(
            config.knowledge,
            graph_path=self.writer.run_dir / "knowledge_graph.sqlite",
            assay_id=config.task.assay_id,
            protein_id=config.task.protein_id,
            validation_config=config.validation,
            structured_graph_path=self.writer.run_dir / "structured_kg.sqlite",
            task_context=self.task_context,
            protein_name=config.task.protein_name,
            protein_aliases=config.task.protein_aliases,
            protein_accessions=config.task.protein_accessions,
            local_knowledge_enabled=config.knowledge_enabled,
            snapshot_mode=config.structured_kg_snapshot_mode,
        )
        role_clients = create_role_client_bundle(
            config.llm.provider,
            rethink_mode=config.validation.rethink_mode,
            profile=config.llm.profile,
            model=config.llm.model,
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            reasoning_effort=config.llm.reasoning_effort,
            thinking=config.llm.thinking,
        )
        graph_tool = (
            self.knowledge.agent_tool()
            if config.knowledge_enabled and config.knowledge.kg
            else None
        )
        self.agent = agent or ScientistAgent(
            role_clients.scientist,
            task_context=self.task_context,
            objective=config.task.objective,
            knowledge_graph=graph_tool,
            design_space="open_design",
            position_policy=config.designer.position_policy,
            max_preferred_positions=config.designer.max_preferred_positions,
            allowed_mutation_positions=self.resolved_positions,
        )
        if critic_agent is None:
            critic_agent = create_batch_critic_agent(config.critic)
        self.critic_agent = critic_agent
        self.approval_gateway = ApprovalGateway()
        self.hard_validator = OpenDesignHardValidator(
            self.design_space,
            config.critic,
            mutation_depth=config.designer.mutation_depth,
        )
        self.review_loop = BoundedReviewLoop(
            validator=self.hard_validator,
            critic=self.critic_agent,
            max_revision_attempts=config.critic.max_revision_attempts,
            gateway=self.approval_gateway,
        )
        self.state = CampaignState(
            run_id=self.run_id,
            mode=config.mode,
            seed=config.seed,
            round_id=1,
            phase=CampaignPhase.INITIALIZED,
            observed=list(self.observations),
            revealed_variant_ids={item.variant_id for item in self.observations},
        )

    def _config_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.config.mode,
            "seed": self.config.seed,
            "designer": asdict(self.config.designer),
            "selection_driver": self.config.generation.selection_driver,
            "active_learning": asdict(self.config.active_learning),
            "posterior_enabled": self.posterior_enabled,
            "reference_sequence": self.task_context.full_sequence,
            "reference_length": len(self.task_context.full_sequence),
            "numbering_scheme": self.task_context.numbering_scheme,
            "initial_data_source": self.initial_data_source,
            "candidate_source": "generated_from_reference",
            "candidate_pool_consulted": False,
            "resolved_design_space": self.design_space.public_dict(),
        }

    def _scientist_activation_state(self, evidence: Sequence[Evidence]) -> RoleActivationState:
        available_channels = {
            item.channel for item in evidence if item.quality_status != "unavailable"
        }
        unavailable_channels = {
            item.channel for item in evidence if item.quality_status == "unavailable"
        }
        kg_configured = getattr(self.agent, "knowledge_graph", None) is not None
        selection_driver = self.config.generation.selection_driver
        return RoleActivationState(
            role="scientist",
            design_space="open_design",
            candidate_source="generated_from_reference",
            candidate_pool_consulted=False,
            position_policy=self.config.designer.position_policy,
            selection_driver=selection_driver,
            active_learning_enabled=bool(
                selection_driver == "active_learning" and self.config.active_learning.enabled
            ),
            fitness_predictors_used_for_generation=selection_driver
            in {"active_learning", "predictor"},
            rag_configured=bool(
                self.config.knowledge_enabled and self.config.knowledge.local_knowledge.enabled
            ),
            rag_context_visible=False,
            rag_retrieval_performed=False,
            rag_evidence_present=False,
            kg_configured=kg_configured,
            kg_interaction_enabled=False,
            configured_kg_tools=(("hypothesis_context",) if kg_configured else ()),
            available_evidence_channels=tuple(sorted(available_channels)),
            unavailable_evidence_channels=tuple(sorted(unavailable_channels)),
        )

    def run(self) -> dict[str, Any]:
        token = bind_progress(self.writer)
        try:
            return self._run()
        except Exception as error:
            with contextlib.suppress(Exception):
                self.writer.report(
                    "open_design_failed",
                    message="open design failed",
                    phase="failed",
                    round_id=1,
                    error_type=type(error).__name__,
                    error_message=str(error)[:500],
                )
            raise
        finally:
            reset_progress(token)
            with contextlib.suppress(Exception):
                self.knowledge.close()

    def _run(self) -> dict[str, Any]:
        self.writer.write_json("config.json", self._config_record())
        self.knowledge.update(self.observed_variants, self.observations)
        observed_sequences = {item.sequence for item in self.observed_variants}
        raw_proposals = self.proposer.propose()
        proposals = [item for item in raw_proposals if item.sequence not in observed_sequences]
        if not proposals:
            raise RuntimeError("Open-design proposer produced no unmeasured sequences")
        candidates = [item.to_variant() for item in proposals]
        self.writer.report(
            "open_design_proposed",
            message=f"generated {len(candidates)} full-sequence candidates",
            phase=CampaignPhase.PROPOSED.value,
            round_id=1,
            n_candidates=len(candidates),
        )
        self.writer.write_json(
            "design_space.json",
            {
                "reference_sequence": self.task_context.full_sequence,
                "reference_length": len(self.task_context.full_sequence),
                **self.design_space.public_dict(),
                # Retain the old field as an explicit alias for artifact consumers.
                "positions": list(self.resolved_positions),
                "generated_count_before_observed_filter": (len(raw_proposals)),
                "unmeasured_proposal_count": len(proposals),
                "candidate_source": "generated_from_reference",
                "candidate_pool_consulted": False,
            },
        )
        self.writer.write_csv("proposals.csv", [item.public_dict() for item in proposals])

        evidence = self.knowledge.evidence_for(candidates, round_id=1)
        raw_knowledge_scores = (
            self.knowledge.scores(evidence) if self.config.knowledge_enabled else {}
        )
        scientist_evidence = _flatten_evidence(
            evidence, limit=self.config.scientist_prompt_evidence_limit
        )
        hypothesis = self.agent.propose_hypothesis(
            self.state,
            self.observed_variants,
            self.observations,
            scientist_evidence,
            activation_state=self._scientist_activation_state(scientist_evidence),
        )
        self.state.hypotheses.append(hypothesis)
        self.writer.write_json("hypothesis.json", hypothesis)

        hypothesis_scores = {
            proposal.proposal_id: _hypothesis_prior(proposal, hypothesis.preferred_residues)
            for proposal in proposals
        }
        structure_scores = {
            variant.variant_id: _structure_constraint(evidence.get(variant.variant_id, ()))
            for variant in candidates
        }
        acquisition_knowledge = {
            variant.variant_id: (
                raw_knowledge_scores.get(variant.variant_id, 0.0)
                + self.config.designer.hypothesis_prior_weight
                * hypothesis_scores[variant.variant_id]
                + self.config.designer.structure_constraint_weight
                * structure_scores[variant.variant_id]
            )
            for variant in candidates
        }

        expected_batch_size = min(self.config.budget_per_round, len(candidates))
        if self.posterior_enabled:
            self.writer.report(
                "open_design_posterior_fit_started",
                message="fitting visible-label posterior over generated sequences",
                phase=CampaignPhase.MODEL_FIT.value,
                round_id=1,
                n_train=len(self.observed_variants),
                n_candidates=len(candidates),
            )
            posterior = self.active_learning.fit_predict(
                self.observed_variants,
                self.observations,
                candidates,
            )
            hybrid_scores = self.active_learning.score(posterior, acquisition_knowledge)
            selection = self.active_learning.select(
                candidates,
                hybrid_scores,
                self.config.budget_per_round,
                knowledge_scores=acquisition_knowledge,
            )
            prediction_by_id = {item.variant_id: item for item in posterior.predictions}
            score_by_id = hybrid_scores.by_id()
            composite_by_id = {
                variant_id: score.composite for variant_id, score in score_by_id.items()
            }
            arm_by_id = {
                variant_id: arm for arm, ids in selection.selected_by_arm.items() for variant_id in ids
            }
            initial_ids = tuple(selection.selected_ids)
        else:
            # Knowledge-only mode: no posterior is fitted, fitness fields stay
            # NaN, and ranking uses the knowledge/hypothesis/structure composite.
            posterior = None
            self.writer.report(
                "open_design_knowledge_only",
                message=(
                    "active_learning disabled; ranking generated sequences from "
                    "knowledge evidence without a fitted posterior"
                ),
                phase=CampaignPhase.MODEL_FIT.value,
                round_id=1,
                n_train=len(self.observed_variants),
                n_candidates=len(candidates),
            )
            prediction_by_id = {
                variant.variant_id: _not_evaluated_prediction(variant.variant_id)
                for variant in candidates
            }
            composite_by_id = dict(acquisition_knowledge)
            arm_by_id = {}
            initial_ids = tuple(
                item.variant_id
                for item in sorted(
                    candidates,
                    key=lambda item: (-acquisition_knowledge[item.variant_id], item.variant_id),
                )[:expected_batch_size]
            )
        ranked = [
            RankedSequenceDesign(
                proposal=proposal,
                fitness_mean=prediction_by_id[proposal.proposal_id].fitness_mean,
                fitness_std=prediction_by_id[proposal.proposal_id].fitness_std,
                interval_90=prediction_by_id[proposal.proposal_id].interval_90,
                ood_score=prediction_by_id[proposal.proposal_id].ood_score,
                acquisition_score=composite_by_id[proposal.proposal_id],
                knowledge_score=raw_knowledge_scores.get(proposal.proposal_id, 0.0),
                hypothesis_prior=hypothesis_scores[proposal.proposal_id],
                structure_constraint=structure_scores[proposal.proposal_id],
                acquisition_arm=arm_by_id.get(proposal.proposal_id),
            )
            for proposal in proposals
        ]
        if self.posterior_enabled:
            ranked.sort(
                key=lambda item: (
                    item.acquisition_score,
                    item.fitness_mean,
                    item.fitness_std,
                    item.proposal.proposal_id,
                ),
                reverse=True,
            )
        else:
            # fitness fields are NaN here; order by the knowledge composite only.
            ranked.sort(
                key=lambda item: (item.acquisition_score, item.proposal.proposal_id),
                reverse=True,
            )
        ranked_by_id = {item.proposal.proposal_id: item for item in ranked}
        candidate_by_id = {item.variant_id: item for item in candidates}
        ranked_ids = tuple(item.proposal.proposal_id for item in ranked)
        review_context_holder: dict[str, BatchReviewContext] = {}
        prediction_cards = {
            variant_id: prediction_review_card(
                prediction,
                source_kind=(
                    "active_posterior" if self.posterior_enabled else "not_fitted"
                ),
                decision_eligible=self.posterior_enabled,
                calibration_status=(
                    "calibrated"
                    if posterior is not None and posterior.calibration.status == "calibrated"
                    else ("uncalibrated" if self.posterior_enabled else "not_applicable")
                ),
            )
            for variant_id, prediction in prediction_by_id.items()
        }
        wild_type_by_position = {
            position: self.design_space.reference_sequence[index]
            for position, index in self.design_space.position_to_sequence_index.items()
        }

        def draft_builder(
            review_attempt: int,
            parent_draft_batch_id: str | None,
            exclusions: set[str],
            constraints: RevisionConstraints | None = None,
            revision_feedback: Any | None = None,
        ):
            ordered = tuple(dict.fromkeys((*initial_ids, *ranked_ids)))
            eligible_ids = tuple(item for item in ordered if item not in exclusions)
            before_residue_filter = len(eligible_ids)
            if constraints is not None and constraints.has_residue_constraints:
                eligible_ids = tuple(
                    candidate_id
                    for candidate_id in eligible_ids
                    if not constraints.variant_violations(
                        candidate_by_id[candidate_id],
                        arm="fallback",
                        position_to_index=(
                            self.design_space.position_to_sequence_index
                        ),
                        wild_type_by_position=wild_type_by_position,
                    )
                )
            candidate_ids = eligible_ids[:expected_batch_size]
            if review_attempt > 0 and len(candidate_ids) < expected_batch_size:
                raise RevisionConstraintInfeasible(
                    RevisionQuotaShortfallReceipt(
                        required_batch_size=expected_batch_size,
                        eligible_before_filter=len(ordered),
                        eligible_after_filter=len(eligible_ids),
                        selected_count=len(candidate_ids),
                        shortfall=expected_batch_size - len(candidate_ids),
                        quota_shortfalls={
                            "batch_total": expected_batch_size - len(candidate_ids)
                        },
                        excluded_candidate_count=(
                            len(exclusions)
                            + before_residue_filter
                            - len(eligible_ids)
                        ),
                        constraints_id="RC01",
                    )
                )
            review_context_holder["value"] = BatchReviewContext(
                prediction_status_by_id={
                    candidate_id: prediction_cards[candidate_id]
                    for candidate_id in candidate_ids
                },
                candidate_intent_by_id={
                    candidate_id: CandidateIntentCard(
                        candidate_id=candidate_id,
                        arm="fallback",
                    )
                    for candidate_id in candidate_ids
                },
                soft_prior_mismatch_ids=soft_prior_mismatch_ids(
                    candidate_ids=candidate_ids,
                    variants_by_id=candidate_by_id,
                    hypothesis=hypothesis,
                    position_to_index=self.design_space.position_to_sequence_index,
                ),
                review_controls=self.config.critic.review_controls,
                review_diversity=self.config.critic.review_diversity,
                revision_feedback=revision_feedback,
            )
            falsification_spec = preregister_batch_median_test(
                hypothesis=hypothesis,
                round_id=1,
                target_variant_ids=candidate_ids,
                visible_observations=self.observations,
            )
            return build_draft_batch(
                round_id=1,
                review_attempt=review_attempt,
                candidate_ids=candidate_ids,
                variants=candidate_by_id,
                predictions=prediction_by_id,
                evidence=evidence,
                hypothesis_id=hypothesis.hypothesis_id,
                falsification_spec=falsification_spec,
                parent_draft_batch_id=parent_draft_batch_id,
                rationale_claims={
                    item: (
                        "Selected from generated full sequences by calibrated posterior, "
                        "uncertainty, Scientist soft prior, and available structural context."
                        if self.posterior_enabled
                        else "Selected from generated full sequences by knowledge "
                        "evidence, Scientist soft prior, and available structural "
                        "context; no posterior was fitted."
                    )
                    for item in candidate_ids
                },
            )

        def record_review_start(draft: Any, report: Any) -> None:
            self.state.phase = CampaignPhase.HARD_VALIDATED
            self.writer.write_json(f"review/draft_attempt_{draft.review_attempt}.json", draft)
            self.writer.write_json(
                f"review/hard_validation_attempt_{draft.review_attempt}.json", report
            )
            self.writer.report(
                "open_design_hard_validated",
                message=(f"hard validation completed with {len(report.hard_conflicts)} blockers"),
                phase=CampaignPhase.HARD_VALIDATED.value,
                round_id=1,
                attempt=draft.review_attempt,
                resolved_positions=list(self.resolved_positions),
                hard_conflicts=len(report.hard_conflicts),
            )

        def record_review(draft: Any, report: Any, decision: Any) -> None:
            self.state.critique_decisions.append(decision)
            self.writer.write_json(f"review/critique_attempt_{draft.review_attempt}.json", decision)
            self.writer.report(
                "open_design_critic_completed",
                message=f"critic returned {decision.verdict.value}",
                phase=CampaignPhase.CRITIQUE_REQUESTED.value,
                round_id=1,
                attempt=draft.review_attempt,
                verdict=decision.verdict.value,
            )

        try:
            review = self.review_loop.run(
                draft_builder=draft_builder,
                variants=candidate_by_id,
                predictions=prediction_by_id,
                evidence=evidence,
                revealed_ids={item.variant_id for item in self.observed_variants},
                pending_ids=set(),
                allowed_ids=set(candidate_by_id),
                expected_batch_size=expected_batch_size,
                context_evidence=_flatten_evidence(evidence),
                hypothesis=hypothesis,
                position_to_index=self.design_space.position_to_sequence_index,
                review_context_provider=lambda _draft: review_context_holder["value"],
                on_attempt_start=record_review_start,
                on_attempt=record_review,
            )
        except RevisionConstraintInfeasible as error:
            self.writer.write_json(
                "review/revision_constraint_infeasible.json",
                error.receipt.model_dump(mode="json"),
            )
            self.writer.report(
                "revision_constraint_infeasible",
                message=error.receipt.code,
                phase=CampaignPhase.ROUND_ABORTED.value,
                round_id=1,
                receipt=error.receipt.model_dump(mode="json"),
            )
            raise
        self.approval_gateway.verify(review.approved_batch)
        self.state.phase = CampaignPhase.APPROVED
        self.writer.write_json("review/approved_batch.json", review.approved_batch)
        selected_ids = tuple(review.approved_batch.candidate_ids)
        selected = [ranked_by_id[item] for item in selected_ids]
        self.writer.write_json(
            "posterior.json",
            {
                "calibration": (
                    posterior.calibration
                    if posterior is not None
                    else {"status": "not_fitted_no_visible_labels"}
                ),
                "predictions": (posterior.predictions if posterior is not None else []),
            },
        )
        self.writer.write_json(
            "knowledge/evidence.json",
            {key: list(value) for key, value in evidence.items()},
        )
        self.writer.write_csv("ranked_candidates.csv", [item.public_dict() for item in ranked])
        self.writer.write_json(
            "selected_candidates.json", [item.public_dict() for item in selected]
        )
        self.writer.write_csv("selected_candidates.csv", [item.public_dict() for item in selected])
        self.writer.write_text(
            "selected_candidates.fasta",
            "".join(
                f">{item.proposal.proposal_id} {item.proposal.mutation_notation}\n"
                f"{item.proposal.sequence}\n"
                for item in selected
            ),
        )
        structure_available = any(
            item.channel == "structure" and item.quality_status == "ok"
            for values in evidence.values()
            for item in values
        )
        summary = {
            "run_id": self.run_id,
            "design_space": "open_design",
            "position_policy": self.config.designer.position_policy,
            "reference_length": len(self.task_context.full_sequence),
            "open_position_count": len(self.resolved_positions),
            "resolved_positions": list(self.resolved_positions),
            "generated_candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_variant_ids": list(selected_ids),
            "candidate_source": "generated_from_reference",
            "candidate_pool_consulted": False,
            "posterior_enabled": self.posterior_enabled,
            "posterior_calibration_status": (
                posterior.calibration.status
                if posterior is not None
                else "not_fitted_no_visible_labels"
            ),
            "knowledge_used": self.config.knowledge_enabled,
            "structure_constraint_available": structure_available,
            "structure_constraint_semantics": "static_context_penalty_not_fitness",
            "scientist_preference_semantics": "sparse_soft_prior_not_search_filter",
            "initial_data_source": self.initial_data_source,
            "hard_validation_report_id": review.report.report_id,
            "hard_validation_conflicts": len(review.report.conflicts),
            "hard_validation_blockers": len(review.report.hard_conflicts),
            "critic_verdict": review.decision.verdict.value,
            "approval_id": review.approved_batch.approval_id,
            "run_dir": str(self.writer.run_dir),
        }
        self.state.phase = CampaignPhase.FINALIZED
        self.writer.write_json("state.json", self.state.as_dict())
        self.writer.write_json("summary.json", summary)
        self.writer.report(
            "open_design_finalized",
            message="open design finalized",
            phase=CampaignPhase.FINALIZED.value,
            round_id=1,
            n_candidates=len(candidates),
        )
        return summary


def run_open_design(config: ExperimentConfig) -> dict[str, Any]:
    return OpenDesignRunner(config).run()
