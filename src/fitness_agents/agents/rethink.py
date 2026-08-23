from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from functools import partial
from statistics import mean
from typing import Any

from fitness_agents.agents.remote_llm import create_openai_client, resolve_base_url, resolve_model
from fitness_agents.contracts.agent_io import (
    AgentTraceContext,
    HypothesisReflectionContextInput,
    ReThinkObservationCard,
    RoundEvidenceDigest,
)
from fitness_agents.contracts.schemas import HypothesisReflection
from fitness_agents.utils.progress import report_event, report_llm_id_bridge

from .output_contracts import (
    HypothesisDimensionGroupOutput,
    HypothesisReflectionOutput,
)
from .profile_loader import load_role_profile
from .short_ids import FieldIdPolicy, RequestScopedIdBridge
from .structured_completion import complete_structured
from .transports import OpenAICompatibleChatTransport

RETHINK_SCHEMA: dict[str, Any] = HypothesisReflectionOutput.model_json_schema()

RETHINK_DIMENSIONS = (
    "measured_function",
    "edit_level_direction",
    "sequence_interaction_context",
    "structural_context",
    "evolutionary_context",
    "physicochemical_context",
    "feasibility_developability",
    "uncertainty_domain_shift",
)
RETHINK_DIMENSION_GROUPS = {
    "outcome_and_edit": ("measured_function", "edit_level_direction"),
    "sequence_and_physchem": ("sequence_interaction_context", "physicochemical_context"),
    "structure_and_evolution": ("structural_context", "evolutionary_context"),
    "execution_and_uncertainty": (
        "feasibility_developability",
        "uncertainty_domain_shift",
    ),
}

_PROVIDER_GATE_LOCK = threading.Lock()
_PROVIDER_GATES: dict[tuple[str, int], threading.BoundedSemaphore] = {}


def _provider_gate(provider: str, concurrency_limit: int) -> threading.BoundedSemaphore:
    key = (provider, concurrency_limit)
    with _PROVIDER_GATE_LOCK:
        gate = _PROVIDER_GATES.get(key)
        if gate is None:
            gate = threading.BoundedSemaphore(concurrency_limit)
            _PROVIDER_GATES[key] = gate
        return gate


class LLMAttemptBudget:
    """Thread-safe hard cap over provider attempts for one hypothesis reflection."""

    def __init__(
        self,
        *,
        limit: int,
        reserve: int,
        concurrency_limit: int,
        provider: str = "rethink",
    ) -> None:
        self.limit = int(limit)
        self.reserve = int(reserve)
        self._lock = threading.Lock()
        self._consumed = 0
        self._by_stage: dict[str, int] = {}
        self._semaphore = _provider_gate(provider, concurrency_limit)
        self._active = 0
        self._peak_inflight = 0
        self._accepted = 0
        self._failed = 0
        self._cancelled = 0
        self._tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @property
    def usable_baseline(self) -> int:
        return self.limit - self.reserve

    def consume(self, metadata: dict[str, Any]) -> None:
        stage = str(metadata.get("completion_stage") or "single")
        with self._lock:
            if self._consumed >= self.limit:
                raise RuntimeError(
                    "ReThink provider attempt budget exhausted before request dispatch; "
                    f"limit={self.limit}, consumed={self._consumed}"
                )
            self._consumed += 1
            self._by_stage[stage] = self._by_stage.get(stage, 0) + 1
        self._semaphore.acquire()
        with self._lock:
            self._active += 1
            self._peak_inflight = max(self._peak_inflight, self._active)

    def release(self, metadata: dict[str, Any]) -> None:
        with self._lock:
            self._active -= 1
            if self._active < 0:
                raise RuntimeError("ReThink provider attempt budget release underflow")
            outcome = str(metadata.get("outcome") or "failed")
            if outcome == "accepted":
                self._accepted += 1
            elif outcome == "cancelled":
                self._cancelled += 1
            else:
                self._failed += 1
            usage = metadata.get("usage") or {}
            for name in self._tokens:
                value = usage.get(name)
                if isinstance(value, int):
                    self._tokens[name] += value
        self._semaphore.release()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "limit": self.limit,
                "reserve": self.reserve,
                "consumed": self._consumed,
                "remaining": self.limit - self._consumed,
                "by_stage": dict(self._by_stage),
                "accepted": self._accepted,
                "failed": self._failed,
                "cancelled": self._cancelled,
                "tokens": dict(self._tokens),
                "inflight": self._active,
                "peak_inflight": self._peak_inflight,
            }


def build_round_evidence_digest(
    observations: Sequence[ReThinkObservationCard | dict[str, Any]],
    *,
    visible_baseline: float,
    optimization_direction: str,
    criterion_receipts: Sequence[dict[str, Any]] = (),
) -> RoundEvidenceDigest:
    """Build a deterministic hypothesis-level digest without generated prose."""

    typed = tuple(ReThinkObservationCard.model_validate(item) for item in observations)
    favorable = (
        (lambda value: value > visible_baseline)
        if optimization_direction == "higher_is_better"
        else (lambda value: value < visible_baseline)
    )
    arm_summaries = []
    disagreements = []
    evidence_ids: list[str] = []
    for arm in (
        "hypothesis_target",
        "evidence_prior",
        "coverage_exploration",
        "matched_control",
        "fallback",
    ):
        selected = tuple(item for item in typed if item.intent_arm == arm)
        if not selected:
            continue
        wet_values = [item.wet_value for item in selected]
        disagreement_count = 0
        for item in selected:
            evidence_ids.extend(item.evidence_ids)
            if not item.dry_validations:
                continue
            dry_mean = mean(entry.value for entry in item.dry_validations)
            if favorable(dry_mean) != favorable(item.wet_value):
                disagreement_count += 1
                disagreements.append(
                    {
                        "variant_id": item.variant_id,
                        "wet_value": item.wet_value,
                        "dry_mean": dry_mean,
                        "residual": item.wet_value - dry_mean,
                        "max_ood_score": max(entry.ood_score for entry in item.dry_validations),
                        "model_versions": tuple(
                            dict.fromkeys(entry.model_version for entry in item.dry_validations)
                        ),
                    }
                )
        arm_summaries.append(
            {
                "arm": arm,
                "sample_count": len(selected),
                "variant_ids": tuple(item.variant_id for item in selected),
                "wet_mean": mean(wet_values),
                "wet_min": min(wet_values),
                "wet_max": max(wet_values),
                "favorable_count": sum(favorable(value) for value in wet_values),
                "dry_wet_disagreement_count": disagreement_count,
            }
        )
    return RoundEvidenceDigest.model_validate(
        {
            "observation_count": len(typed),
            "observations": typed,
            "arm_summaries": arm_summaries,
            "dry_wet_disagreements": disagreements,
            "criterion_receipts": tuple(criterion_receipts),
            "evidence_ids": tuple(dict.fromkeys(evidence_ids)),
        }
    )


def _rethink_bridge(
    context: HypothesisReflectionContextInput,
    *,
    scope_id: str,
) -> RequestScopedIdBridge:
    observation_ids = list(context.expected_variant_ids)
    evidence_ids = list(context.round_evidence_digest.evidence_ids)
    hypothesis_ids: list[str] = []
    assessment_ids: list[str] = []
    spec_ids: list[str] = []
    criterion_ids: list[str] = []
    for item in context.round_evidence_digest.observations:
        evidence_ids.extend(item.evidence_ids)
        if item.matched_to:
            observation_ids.append(item.matched_to)
    if context.approved_hypothesis is not None:
        hypothesis_ids.append(context.approved_hypothesis.hypothesis_id)
        evidence_ids.extend(context.approved_hypothesis.evidence_ids)
    evidence_ids.extend(context.final_critic_decision.cited_evidence_ids)
    if context.hypothesis_assessment is not None:
        assessment_ids.append(context.hypothesis_assessment.assessment_id)
        hypothesis_ids.append(context.hypothesis_assessment.hypothesis_id)
        spec_ids.append(context.hypothesis_assessment.falsification_spec_id)
        criterion_ids.extend(context.hypothesis_assessment.decisive_criterion_ids)
        criterion_ids.extend(context.hypothesis_assessment.unresolved_criterion_ids)
        observation_ids.extend(context.hypothesis_assessment.observation_ids)
    if context.falsification_spec is not None:
        spec_ids.append(context.falsification_spec.spec_id)
        hypothesis_ids.append(context.falsification_spec.hypothesis_id)
        for criterion in context.falsification_spec.criteria:
            criterion_ids.append(criterion.criterion_id)
            observation_ids.extend(criterion.target_variant_ids)
            observation_ids.extend(criterion.comparator_variant_ids)
    return RequestScopedIdBridge.build(
        scope_id=scope_id,
        role="rethink",
        schema_name="HypothesisDimensionGroupOutput",
        namespace_values={
            "S": tuple(observation_ids),
            "E": tuple(evidence_ids),
            "H": tuple(hypothesis_ids),
            "A": tuple(assessment_ids),
            "P": tuple(spec_ids),
            "T": tuple(criterion_ids),
        },
        field_policies={
            "approved_hypothesis.hypothesis_id": FieldIdPolicy("H", "exact"),
            "approved_hypothesis.evidence_ids[]": FieldIdPolicy("E", "normalize"),
            "hypothesis_assessment.assessment_id": FieldIdPolicy("A", "exact"),
            "hypothesis_assessment.hypothesis_id": FieldIdPolicy("H", "exact"),
            "hypothesis_assessment.falsification_spec_id": FieldIdPolicy("P", "exact"),
            "hypothesis_assessment.criterion_results[].criterion_id": FieldIdPolicy(
                "T", "normalize"
            ),
            "hypothesis_assessment.observation_ids[]": FieldIdPolicy("S", "normalize"),
            "hypothesis_assessment.criterion_results[].observation_ids[]": FieldIdPolicy(
                "S", "normalize"
            ),
            "hypothesis_assessment.decisive_criterion_ids[]": FieldIdPolicy(
                "T", "normalize"
            ),
            "hypothesis_assessment.unresolved_criterion_ids[]": FieldIdPolicy(
                "T", "normalize"
            ),
            "falsification_spec.spec_id": FieldIdPolicy("P", "exact"),
            "falsification_spec.hypothesis_id": FieldIdPolicy("H", "exact"),
            "falsification_spec.criteria[].criterion_id": FieldIdPolicy(
                "T", "normalize"
            ),
            "falsification_spec.criteria[].target_variant_ids[]": FieldIdPolicy(
                "S", "normalize"
            ),
            "falsification_spec.criteria[].comparator_variant_ids[]": FieldIdPolicy(
                "S", "normalize"
            ),
            "round_evidence_digest.observations[].variant_id": FieldIdPolicy(
                "S", "unique_near"
            ),
            "round_evidence_digest.observations[].matched_to": FieldIdPolicy("S", "normalize"),
            "round_evidence_digest.observations[].evidence_ids[]": FieldIdPolicy(
                "E", "normalize"
            ),
            "round_evidence_digest.arm_summaries[].variant_ids[]": FieldIdPolicy(
                "S", "normalize"
            ),
            "round_evidence_digest.dry_wet_disagreements[].variant_id": FieldIdPolicy(
                "S", "unique_near"
            ),
            "round_evidence_digest.criterion_receipts[].criterion_id": FieldIdPolicy(
                "T", "normalize"
            ),
            "round_evidence_digest.criterion_receipts[].observation_ids[]": FieldIdPolicy(
                "S", "normalize"
            ),
            "round_evidence_digest.evidence_ids[]": FieldIdPolicy("E", "normalize"),
            "hypothesis_id": FieldIdPolicy("H", "exact"),
            "assessment_id": FieldIdPolicy("A", "exact"),
            "supporting_observation_ids[]": FieldIdPolicy("S", "normalize"),
            "supporting_evidence_ids[]": FieldIdPolicy("E", "normalize"),
        },
    )


def _group_prompt_context(
    context: HypothesisReflectionContextInput,
    *,
    group_name: str,
    dimensions: tuple[str, str],
) -> dict[str, Any]:
    payload = context.model_dump(mode="json")
    source = payload["round_evidence_digest"]
    observations = source["observations"]
    if group_name == "outcome_and_edit":
        digest = {
            "observation_count": source["observation_count"],
            "observations": [
                {key: item[key] for key in (
                    "variant_id", "mutation_notation", "wet_value", "intent_arm",
                    "matched_to", "falsification_role",
                )}
                for item in observations
            ],
            "arm_summaries": source["arm_summaries"],
            "criterion_receipts": source["criterion_receipts"],
        }
    elif group_name == "sequence_and_physchem":
        digest = {
            "observation_count": source["observation_count"],
            "observations": [
                {key: item[key] for key in (
                    "variant_id", "mutation_notation", "wet_value", "intent_arm",
                    "matched_to", "evidence_ids",
                )}
                for item in observations
            ],
            "arm_summaries": source["arm_summaries"],
            "evidence_ids": source["evidence_ids"],
        }
    elif group_name == "structure_and_evolution":
        digest = {
            "observation_count": source["observation_count"],
            "observations": [
                {
                    "variant_id": item["variant_id"],
                    "mutation_notation": item["mutation_notation"],
                    "evidence_ids": item["evidence_ids"],
                }
                for item in observations
            ],
            "evidence_ids": source["evidence_ids"],
        }
    else:
        digest = {
            "observation_count": source["observation_count"],
            "observations": [
                {key: item[key] for key in (
                    "variant_id", "wet_value", "dry_validations", "intent_arm", "matched_to",
                )}
                for item in observations
            ],
            "arm_summaries": source["arm_summaries"],
            "dry_wet_disagreements": source["dry_wet_disagreements"],
            "criterion_receipts": source["criterion_receipts"],
        }
    payload["round_evidence_digest"] = digest
    payload["output_scope"] = {
        "level": "hypothesis_assessment",
        "dimension_group": group_name,
        "required_dimensions": list(dimensions),
        "assessment_status_runtime_owned": True,
    }
    return payload


def _relation_from_status(status: str) -> str:
    return {
        "SUPPORTED": "positive",
        "CONTRADICTED": "negative",
        "INCONCLUSIVE": "unresolved",
    }[status]


def _fallback_group(
    *,
    hypothesis_id: str,
    assessment_id: str,
    group_name: str,
    dimensions: tuple[str, str],
) -> HypothesisDimensionGroupOutput:
    return HypothesisDimensionGroupOutput(
        hypothesis_id=hypothesis_id,
        assessment_id=assessment_id,
        group_name=group_name,
        dimension_assessments=[
            {
                "dimension": dimension,
                "evidence_status": "missing",
                "relation_to_hypothesis": "unresolved",
                "finding_code": "remote_group_unavailable",
                "finding": "The remote group did not produce a validated hypothesis-level result.",
                "implication": "Keep this dimension unresolved and out of selection evidence.",
                "quality_status": "deterministic_fallback",
            }
            for dimension in dimensions
        ],
        unresolved_questions=[f"The {group_name} reflection remains unavailable."],
        recommended_actions=["collect_missing_measurement"],
        group_advice="Repeat this bounded group assessment without changing the deterministic status.",
    )


class MockHypothesisReThinkClient:
    provider_name = "mock_rethink"

    def __init__(self) -> None:
        self.last_dimension_groups: tuple[HypothesisDimensionGroupOutput, ...] = ()

    def reflect_hypothesis(
        self, *, context: HypothesisReflectionContextInput
    ) -> HypothesisReflection | None:
        typed = HypothesisReflectionContextInput.model_validate(context)
        self.last_dimension_groups = ()
        if typed.approved_hypothesis is None or typed.hypothesis_assessment is None:
            return None
        status = typed.hypothesis_assessment.status
        relation = _relation_from_status(status)
        groups = []
        for group_name, dimensions in RETHINK_DIMENSION_GROUPS.items():
            groups.append(
                HypothesisDimensionGroupOutput(
                    hypothesis_id=typed.approved_hypothesis.hypothesis_id,
                    assessment_id=typed.hypothesis_assessment.assessment_id,
                    group_name=group_name,
                    dimension_assessments=[
                        {
                            "dimension": dimension,
                            "evidence_status": "measured",
                            "relation_to_hypothesis": relation,
                            "finding_code": f"deterministic_{status.casefold()}",
                            "finding": f"The deterministic assessment status is {status}.",
                            "implication": "Treat the result as hypothesis-level and round-bounded.",
                            "quality_status": "deterministic_fallback",
                        }
                        for dimension in dimensions
                    ],
                    retained_claims=(
                        ["Retain the tested hypothesis only within its observed scope."]
                        if status == "SUPPORTED" else []
                    ),
                    invalidated_assumptions=(
                        ["Do not reuse the contradicted hypothesis without revision."]
                        if status == "CONTRADICTED" else []
                    ),
                    unresolved_questions=(
                        ["The deterministic assessment remains inconclusive."]
                        if status == "INCONCLUSIVE" else []
                    ),
                    recommended_actions=[
                        "downweight_rationale"
                        if status == "CONTRADICTED"
                        else "retain_uncertainty_aware_exploration"
                    ],
                    supporting_observation_ids=list(
                        typed.hypothesis_assessment.observation_ids[:12]
                    ),
                    group_advice="Use the deterministic assessment and preserve alternatives.",
                )
            )
        self.last_dimension_groups = tuple(groups)
        return HypothesisReflectionOutput(
            hypothesis_id=typed.approved_hypothesis.hypothesis_id,
            assessment_id=typed.hypothesis_assessment.assessment_id,
            assessment_status=status,
            dimension_groups=groups,
        ).to_reflection(
            round_id=typed.round_id,
            provider=self.provider_name,
            quality_status="deterministic_fallback",
        )

class NativeHypothesisReThinkClient:
    provider_name = "openai_compatible_rethink"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 32768,
        render_max_tokens: int | None = 32768,
        reasoning_effort: str | None = None,
        thinking: str | None = "disabled",
        api_key: str | None = None,
        profile: str = "scientific_v1",
        max_transport_retries: int = 2,
        max_truncation_retries: int = 1,
        max_syntax_retries: int = 1,
        max_schema_retries: int = 2,
        max_semantic_retries: int = 1,
        max_unknown_evidence_retries: int = 1,
        retry_backoff_seconds: float = 1.0,
        request_timeout_seconds: float = 120.0,
        allow_unknown_evidence_stripping: bool = False,
        max_input_chars: int | None = None,
        max_parallel_batches: int = 4,
        max_calls_per_round: int = 32,
        call_reserve: int = 16,
        parallel_dimension_groups: bool = True,
    ) -> None:
        self.model = resolve_model(model, provider=provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.render_max_tokens = render_max_tokens
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.max_transport_retries = max_transport_retries
        self.max_truncation_retries = max_truncation_retries
        self.max_syntax_retries = max_syntax_retries
        self.max_schema_retries = max_schema_retries
        self.max_semantic_retries = max_semantic_retries
        self.max_unknown_evidence_retries = max_unknown_evidence_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.allow_unknown_evidence_stripping = allow_unknown_evidence_stripping
        self.max_input_chars = max_input_chars
        if max_parallel_batches < 1:
            raise ValueError("ReThink max_parallel_batches must be positive")
        self.max_parallel_batches = min(max_parallel_batches, len(RETHINK_DIMENSION_GROUPS))
        if max_calls_per_round < 1 or not 0 <= call_reserve < max_calls_per_round:
            raise ValueError("ReThink call budget and reserve are invalid")
        self.max_calls_per_round = max_calls_per_round
        self.call_reserve = call_reserve
        self.parallel_dimension_groups = parallel_dimension_groups
        self._budget_lock = threading.Lock()
        self._attempt_budget: LLMAttemptBudget | None = None
        self._bridge_lock = threading.Lock()
        self._bridge_sequence = 0
        self.last_dimension_groups: tuple[HypothesisDimensionGroupOutput, ...] = ()
        role_profile = load_role_profile("rethink", profile)
        self.profile_name = profile
        self.profile_version = role_profile.metadata.get("version")
        self.profile = role_profile.instructions
        self.client = create_openai_client(
            api_key=api_key,
            base_url=base_url,
            provider=provider,
            request_timeout_seconds=request_timeout_seconds,
        )
        self.transport = OpenAICompatibleChatTransport(self.client)

    def _new_bridge_scope(self) -> str:
        with self._bridge_lock:
            self._bridge_sequence += 1
            return f"IDB{self._bridge_sequence:06d}"

    def _consume_attempt(self, metadata: dict[str, Any]) -> None:
        if self._attempt_budget is not None:
            self._attempt_budget.consume(metadata)

    def _release_attempt(self, metadata: dict[str, Any]) -> None:
        if self._attempt_budget is not None:
            self._attempt_budget.release(metadata)

    def _reset_attempt_budget(self) -> LLMAttemptBudget:
        budget = LLMAttemptBudget(
            limit=self.max_calls_per_round,
            reserve=self.call_reserve,
            concurrency_limit=self.max_parallel_batches,
            provider=self.provider_name,
        )
        with self._budget_lock:
            self._attempt_budget = budget
        return budget

    def _reflect_dimension_group(
        self,
        *,
        context: HypothesisReflectionContextInput,
        group_name: str,
        dimensions: tuple[str, str],
    ) -> HypothesisDimensionGroupOutput:
        assert context.approved_hypothesis is not None
        assert context.hypothesis_assessment is not None
        bridge = _rethink_bridge(context, scope_id=self._new_bridge_scope())
        alias_context = HypothesisReflectionContextInput.model_validate(
            bridge.encode_projection(context.model_dump(mode="python"))
        )
        report_llm_id_bridge(round_id=context.round_id, **bridge.audit_payload())

        def validate_output(value: dict[str, Any]) -> dict[str, Any]:
            decoded = bridge.decode_and_validate(value)
            output = HypothesisDimensionGroupOutput.model_validate(decoded)
            actual = {item.dimension for item in output.dimension_assessments}
            if (
                output.hypothesis_id != context.approved_hypothesis.hypothesis_id
                or output.assessment_id != context.hypothesis_assessment.assessment_id
                or output.group_name != group_name
                or actual != frozenset(dimensions)
            ):
                raise ValueError("ReThink dimension group does not match its hypothesis request")
            return value

        output = complete_structured(
            client=self.client,
            transport=self.transport,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        self.profile
                        + "\nAnalyze only the two named dimensions for the frozen hypothesis and "
                        "round-level evidence digest. The deterministic assessment status is "
                        "runtime-owned and cannot be changed. Return "
                        "HypothesisDimensionGroupOutput JSON without hidden reasoning. Required "
                        "dimensions: "
                        + json.dumps(dimensions)
                        + ". Schema: "
                        + json.dumps(
                            HypothesisDimensionGroupOutput.model_json_schema(), ensure_ascii=False
                        )
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        _group_prompt_context(
                            alias_context, group_name=group_name, dimensions=dimensions
                        ),
                        ensure_ascii=False,
                    ),
                },
            ],
            output_type=HypothesisDimensionGroupOutput,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            render_max_tokens=self.render_max_tokens,
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
            retries=0,
            transport_retries=self.max_transport_retries,
            truncation_retries=self.max_truncation_retries,
            syntax_retries=self.max_syntax_retries,
            schema_retries=self.max_schema_retries,
            semantic_retries=self.max_semantic_retries,
            unknown_evidence_retries=self.max_unknown_evidence_retries,
            empty_retries=max(1, self.max_syntax_retries),
            retry_backoff_seconds=self.retry_backoff_seconds,
            allow_unknown_evidence_stripping=self.allow_unknown_evidence_stripping,
            max_input_chars=self.max_input_chars,
            separate_json_render=True,
            contextual_validator=validate_output,
            trace_context={
                **AgentTraceContext(
                    run_id=context.run_id,
                    round_id=context.round_id,
                    role="rethink",
                    request_id=(
                        f"rethink:{context.run_id}:r{context.round_id}:"
                        f"{context.hypothesis_assessment.assessment_id}:{group_name}"
                    ),
                ).model_dump(mode="json"),
                "profile": self.profile_name,
                "profile_version": self.profile_version,
                "schema_name": "HypothesisDimensionGroupOutput",
                "rethink_scope": "hypothesis_dimension_group",
                "rethink_dimension_group": group_name,
                "id_bridge_scope": bridge.scope_id,
            },
            attempt_guard=self._consume_attempt,
            attempt_release=self._release_attempt,
        )
        decoded = bridge.decode_and_validate(
            output.model_dump(mode="json"), record_receipts=False
        )
        report_llm_id_bridge(round_id=context.round_id, **bridge.audit_payload())
        return HypothesisDimensionGroupOutput.model_validate(decoded)

    def reflect_hypothesis(
        self, *, context: HypothesisReflectionContextInput
    ) -> HypothesisReflection | None:
        typed = HypothesisReflectionContextInput.model_validate(context)
        self.last_dimension_groups = ()
        if typed.approved_hypothesis is None or typed.hypothesis_assessment is None:
            report_event(
                "rethink_not_applicable",
                message="ReThink skipped because no assessed hypothesis is available",
                round_id=typed.round_id,
            )
            return None
        budget = self._reset_attempt_budget()
        calls_per_completion = 1 if self.thinking == "disabled" else 2
        planned_calls = len(RETHINK_DIMENSION_GROUPS) * calls_per_completion
        if planned_calls > budget.usable_baseline:
            raise ValueError(
                "ReThink dimension call plan exceeds the call budget after retry reserve; "
                f"planned={planned_calls}, usable={budget.usable_baseline}"
            )
        groups: dict[str, HypothesisDimensionGroupOutput] = {}

        def execute(group_name: str, dimensions: tuple[str, str]) -> None:
            try:
                groups[group_name] = self._reflect_dimension_group(
                    context=typed, group_name=group_name, dimensions=dimensions
                )
            except Exception as error:  # noqa: BLE001 - isolate one advisory group
                groups[group_name] = _fallback_group(
                    hypothesis_id=typed.approved_hypothesis.hypothesis_id,
                    assessment_id=typed.hypothesis_assessment.assessment_id,
                    group_name=group_name,
                    dimensions=dimensions,
                )
                report_event(
                    "rethink_dimension_group_degraded",
                    message="ReThink dimension group degraded to a typed fallback",
                    round_id=typed.round_id,
                    hypothesis_id=typed.approved_hypothesis.hypothesis_id,
                    group_name=group_name,
                    error_type=type(error).__name__,
                    error=str(error),
                )

        if self.parallel_dimension_groups:
            with ThreadPoolExecutor(max_workers=self.max_parallel_batches) as executor:
                futures = {
                    executor.submit(
                        copy_context().run, partial(execute, group_name, dimensions)
                    ): group_name
                    for group_name, dimensions in RETHINK_DIMENSION_GROUPS.items()
                }
                for future in as_completed(futures):
                    future.result()
        else:
            for group_name, dimensions in RETHINK_DIMENSION_GROUPS.items():
                execute(group_name, dimensions)

        ordered_groups = [groups[name] for name in RETHINK_DIMENSION_GROUPS]
        self.last_dimension_groups = tuple(ordered_groups)
        degraded = any(
            entry.quality_status == "deterministic_fallback"
            for group in ordered_groups
            for entry in group.dimension_assessments
        )
        reflection = HypothesisReflectionOutput(
            hypothesis_id=typed.approved_hypothesis.hypothesis_id,
            assessment_id=typed.hypothesis_assessment.assessment_id,
            assessment_status=typed.hypothesis_assessment.status,
            dimension_groups=ordered_groups,
        ).to_reflection(
            round_id=typed.round_id,
            provider=self.provider_name,
            quality_status="deterministic_fallback" if degraded else "model",
        )
        report_event(
            "rethink_attempt_budget_completed",
            message="Hypothesis-level ReThink provider attempt budget completed",
            round_id=typed.round_id,
            logical_group_calls=len(RETHINK_DIMENSION_GROUPS),
            **budget.snapshot(),
        )
        return reflection

OpenAICompatibleHypothesisReThinkClient = NativeHypothesisReThinkClient


def create_hypothesis_rethink_client(provider: str, **kwargs: Any):
    if "runtime" in kwargs:
        runtime = str(kwargs.pop("runtime"))
        if runtime != "chat_completions":
            raise ValueError(f"Removed Agents SDK runtime is not supported: {runtime!r}")
    if provider == "mock":
        return MockHypothesisReThinkClient()
    if provider in {"openai", "openai_compatible", "deepseek"}:
        if provider == "deepseek":
            kwargs.setdefault("provider", "deepseek")
            kwargs.setdefault(
                "base_url", resolve_base_url(kwargs.get("base_url"), provider="deepseek")
            )
            kwargs.setdefault("model", resolve_model(kwargs.get("model"), provider="deepseek"))
        return NativeHypothesisReThinkClient(**kwargs)
    raise ValueError(f"Unknown ReThink provider {provider!r}")


# Preserve the original candidate-level public API. New integrations should use
# the explicitly named hypothesis-level classes/factory above.
from .rethink_sample import (  # noqa: F401
    MockReThinkClient,
    NativeReThinkClient,
    OpenAICompatibleReThinkClient,
    create_sample_rethink_client,
)


def create_rethink_client(provider: str, **kwargs: Any):
    return create_sample_rethink_client(provider, **kwargs)
