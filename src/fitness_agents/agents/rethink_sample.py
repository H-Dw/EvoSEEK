from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import replace
from functools import partial
from typing import Any

from fitness_agents.agents.adaptive_batch import AdaptiveBatchWork, adaptive_batch_submit
from fitness_agents.agents.remote_llm import (
    RemoteLLMCompletionError,
    create_openai_client,
    resolve_base_url,
    resolve_model,
)
from fitness_agents.contracts.rethink_sample_io import (
    AgentTraceContext,
    ReThinkContextInput,
    ReThinkReflection,
)
from fitness_agents.utils.progress import report_event, report_llm_id_bridge

from .profile_loader import load_role_profile
from .rethink_sample_output_contracts import (
    ReThinkDimensionGroupOutput,
    ReThinkItemsOutput,
    ReThinkOutput,
)
from .short_ids import FieldIdPolicy, RequestScopedIdBridge
from .structured_completion import complete_structured
from .transports import OpenAICompatibleChatTransport

RETHINK_SCHEMA: dict[str, Any] = ReThinkOutput.model_json_schema()

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
    "sequence_and_physchem": (
        "sequence_interaction_context",
        "physicochemical_context",
    ),
    "structure_and_evolution": ("structural_context", "evolutionary_context"),
    "execution_and_uncertainty": (
        "feasibility_developability",
        "uncertainty_domain_shift",
    ),
}

_PROVIDER_GATE_LOCK = threading.Lock()
_PROVIDER_GATES: dict[tuple[str, int], threading.BoundedSemaphore] = {}


def _provider_gate(provider: str, concurrency_limit: int) -> threading.BoundedSemaphore:
    """Share a provider concurrency gate across ReThink clients in this process."""

    key = (provider, concurrency_limit)
    with _PROVIDER_GATE_LOCK:
        gate = _PROVIDER_GATES.get(key)
        if gate is None:
            gate = threading.BoundedSemaphore(concurrency_limit)
            _PROVIDER_GATES[key] = gate
        return gate


class LLMAttemptBudget:
    """Thread-safe hard cap over actual provider attempts for one round."""

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


def _rethink_bridge(
    context: ReThinkContextInput,
    *,
    scope_id: str,
    schema_name: str,
) -> RequestScopedIdBridge:
    payload = context.model_dump(mode="json")
    sample_ids: list[str] = [item.variant_id for item in context.candidates]
    evidence_ids: list[str] = []
    hypothesis_ids: list[str] = []
    decision_ids = [context.final_critic_decision.decision_id]
    assessment_ids: list[str] = []
    spec_ids: list[str] = []
    criterion_ids: list[str] = []
    for item in context.candidates:
        evidence_ids.extend(item.evidence_ids)
        if item.matched_to:
            sample_ids.append(item.matched_to)
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
    if context.falsification_spec is not None:
        spec_ids.append(context.falsification_spec.spec_id)
        hypothesis_ids.append(context.falsification_spec.hypothesis_id)
        for criterion in context.falsification_spec.criteria:
            criterion_ids.append(criterion.criterion_id)
            sample_ids.extend(criterion.target_variant_ids)
            sample_ids.extend(criterion.comparator_variant_ids)
    del payload
    return RequestScopedIdBridge.build(
        scope_id=scope_id,
        role="rethink",
        schema_name=schema_name,
        namespace_values={
            "S": tuple(sample_ids),
            "E": tuple(evidence_ids),
            "H": tuple(hypothesis_ids),
            "D": tuple(decision_ids),
            "A": tuple(assessment_ids),
            "P": tuple(spec_ids),
            "T": tuple(criterion_ids),
        },
        field_policies={
            "candidates[].variant_id": FieldIdPolicy("S", "unique_near"),
            "candidates[].matched_to": FieldIdPolicy("S", "normalize"),
            "candidates[].evidence_ids[]": FieldIdPolicy("E", "normalize"),
            "approved_hypothesis.hypothesis_id": FieldIdPolicy("H", "exact"),
            "approved_hypothesis.evidence_ids[]": FieldIdPolicy("E", "normalize"),
            "final_critic_decision.decision_id": FieldIdPolicy("D", "exact"),
            "final_critic_decision.cited_evidence_ids[]": FieldIdPolicy("E", "normalize"),
            "hypothesis_assessment.assessment_id": FieldIdPolicy("A", "exact"),
            "hypothesis_assessment.hypothesis_id": FieldIdPolicy("H", "exact"),
            "hypothesis_assessment.falsification_spec_id": FieldIdPolicy("P", "exact"),
            "hypothesis_assessment.decisive_criterion_ids[]": FieldIdPolicy("T", "exact"),
            "hypothesis_assessment.unresolved_criterion_ids[]": FieldIdPolicy("T", "exact"),
            "falsification_spec.spec_id": FieldIdPolicy("P", "exact"),
            "falsification_spec.hypothesis_id": FieldIdPolicy("H", "exact"),
            "falsification_spec.criteria[].criterion_id": FieldIdPolicy("T", "exact"),
            "falsification_spec.criteria[].target_variant_ids[]": FieldIdPolicy("S", "unique_near"),
            "falsification_spec.criteria[].comparator_variant_ids[]": FieldIdPolicy("S", "unique_near"),
            "reflections[].variant_id": FieldIdPolicy("S", "unique_near"),
            "variant_id": FieldIdPolicy("S", "unique_near"),
        },
    )


def _bounded_join(values: list[str] | tuple[str, ...], *, limit: int) -> str:
    selected: list[str] = []
    length = 0
    for value in values:
        cleaned = " ".join(str(value).split())
        if not cleaned:
            continue
        addition = len(cleaned) + (1 if selected else 0)
        if length + addition > limit:
            break
        selected.append(cleaned)
        length += addition
    return " ".join(selected) or "No additional bounded advice was available."


def _prioritized_group_advice(
    groups: dict[str, ReThinkDimensionGroupOutput],
) -> tuple[str, tuple[dict[str, str], ...]]:
    """Keep all group advice for audit and select at most two next-step priorities."""

    relation_weight = {"unresolved": 4, "negative": 3, "mixed": 2, "positive": 1}
    ranked: list[tuple[int, int, str, str]] = []
    persisted: list[dict[str, str]] = []
    for order, group_name in enumerate(RETHINK_DIMENSION_GROUPS):
        group = groups[group_name]
        persisted.append({"group": group_name, "advice": group.group_advice})
        priority = max(
            relation_weight[item.relation_to_sample_rationale]
            + (2 if item.quality_status == "deterministic_fallback" else 0)
            for item in group.dimension_assessments
        )
        ranked.append((priority, -order, group_name, group.group_advice))
    ranked.sort(reverse=True)
    selected = [f"{name}: {advice}" for _, _, name, advice in ranked[:2]]
    return _bounded_join(selected, limit=2000), tuple(persisted)


def _relation_for_candidate(
    *,
    intent_arm: str,
    wet_support: bool,
    dry_agrees: bool,
) -> str:
    """Relate the observation to the sample rationale, not the batch hypothesis."""

    if intent_arm in {"hypothesis_target", "evidence_prior"}:
        if not dry_agrees:
            return "mixed"
        return "support" if wet_support else "conflict"
    if intent_arm in {"coverage_exploration", "matched_control", "fallback"}:
        return "inconclusive" if dry_agrees else "mixed"
    return "inconclusive"


def _dimension_assessments(*, wet_support: bool, dry_agrees: bool) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for dimension in RETHINK_DIMENSIONS:
        if dimension == "measured_function":
            finding = "The revealed value exceeded the visible baseline." if wet_support else "The revealed value did not exceed the visible baseline."
            status = "measured"
        elif dimension == "uncertainty_domain_shift":
            finding = "Dry and wet directions agree." if dry_agrees else "Dry and wet directions disagree."
            status = "mixed"
        else:
            finding = "No dimension-specific evidence was supplied in this request."
            status = "missing"
        output.append(
            {
                "dimension": dimension,
                "evidence_status": status,
                "finding": finding,
                "implication": "Keep this dimension bounded to the visible round evidence.",
            }
        )
    return output


def _compact_rethink_prompt(context: ReThinkContextInput) -> dict[str, Any]:
    """Remove repeated comparator universes while preserving executable assessment semantics."""

    payload = context.model_dump(mode="json")
    spec = payload.get("falsification_spec")
    if isinstance(spec, dict):
        candidate_ids = {item["variant_id"] for item in payload.get("candidates", ())}
        compact_criteria: list[dict[str, Any]] = []
        for raw in spec.get("criteria", ()):
            criterion = dict(raw)
            targets = tuple(criterion.pop("target_variant_ids", ()))
            comparators = tuple(criterion.pop("comparator_variant_ids", ()))
            criterion["target_scope"] = (
                "current_request_candidates"
                if set(targets).intersection(candidate_ids)
                else "outside_current_request"
            )
            criterion["target_count"] = len(targets)
            criterion["comparator_scope"] = "registered_visible_baseline"
            criterion["comparator_count"] = len(comparators)
            compact_criteria.append(criterion)
        spec["criteria"] = compact_criteria
    payload["output_scope"] = {
        "level": "sample",
        "required_sample_ids": [item["variant_id"] for item in payload["candidates"]],
        "batch_assessment": "runtime_owned_not_model_output",
        "required_dimensions": list(RETHINK_DIMENSIONS),
    }
    return payload


def _parse_reflections(
    payload: dict[str, Any], *, run_id: str, round_id: int, provider: str
) -> tuple[ReThinkReflection, ...]:
    return ReThinkOutput.model_validate(payload).to_reflections(
        run_id=run_id,
        round_id=round_id,
        provider=provider,
    )


def _validate_rethink_items(
    payload: dict[str, Any], *, expected_variant_ids: frozenset[str]
) -> dict[str, Any]:
    output = ReThinkItemsOutput.model_validate(payload)
    actual = {item.variant_id for item in output.reflections}
    missing = sorted(expected_variant_ids.difference(actual))
    unexpected = sorted(actual.difference(expected_variant_ids))
    if missing or unexpected:
        raise ValueError(
            f"ReThink sample coverage mismatch; missing={missing}, unexpected={unexpected}"
        )
    return output.model_dump(mode="json")


class MockReThinkClient:
    """Deterministic offline reflection with the same structured output as the remote role."""

    provider_name = "mock_rethink"

    def reflect_round(self, *, context: ReThinkContextInput) -> tuple[ReThinkReflection, ...]:
        typed_context = ReThinkContextInput.model_validate(context)
        context = typed_context.model_dump(mode="json")
        baseline = typed_context.visible_baseline
        items: list[dict[str, Any]] = []
        for item in context.get("candidates", ()):
            wet = float(item["wet_value"])
            dry = [float(entry["value"]) for entry in item.get("dry_validations", ())]
            wet_support = wet > baseline
            dry_mean = sum(dry) / len(dry) if dry else None
            dry_agrees = dry_mean is None or (dry_mean > baseline) == wet_support
            verdict = _relation_for_candidate(
                intent_arm=str(item.get("intent_arm", "fallback")),
                wet_support=wet_support,
                dry_agrees=dry_agrees,
            )
            positives = []
            negatives = []
            if wet_support:
                positives.append("Wet validation exceeded the pre-round visible baseline.")
            else:
                negatives.append("Wet validation did not exceed the pre-round visible baseline.")
            if dry_mean is not None and dry_agrees:
                positives.append("Dry validation agreed with the wet direction.")
            elif dry_mean is not None:
                negatives.append("Dry and wet validation directions disagreed.")
            items.append(
                {
                    "variant_id": item["variant_id"],
                    "candidate_relation": verdict,
                    "summary": (
                        f"Recommendation reason is {verdict} by wet/dry directional checks; "
                        f"wet={wet:.4f}, baseline={baseline:.4f}."
                    ),
                    "positive_findings": positives,
                    "negative_findings": negatives,
                    "revised_reason": (
                        "Treat the original selection rationale as round-specific evidence. "
                        "Do not infer a universal residue effect from this fallback review."
                    ),
                    "next_round_advice": (
                        "Retain related mutations with uncertainty-aware exploration."
                        if wet_support
                        else "Down-weight this rationale and test matched alternatives."
                    ),
                    "next_round_action": (
                        "retain_uncertainty_aware_exploration"
                        if wet_support
                        else "test_matched_alternative"
                    ),
                    "dimension_assessments": _dimension_assessments(
                        wet_support=wet_support, dry_agrees=dry_agrees
                    ),
                }
            )
        return _parse_reflections(
            {
                "reflections": items,
                "batch_assessment": {
                    "assessment_id": (
                        typed_context.hypothesis_assessment.assessment_id
                        if typed_context.hypothesis_assessment is not None
                        else None
                    ),
                    "status": (
                        typed_context.hypothesis_assessment.status
                        if typed_context.hypothesis_assessment is not None
                        else "NOT_APPLICABLE"
                    ),
                    "commentary": (
                        "Candidate-level rationale relations are advisory; the runtime assessment "
                        "is the authoritative batch-level hypothesis result."
                    ),
                    "next_round_advice": "Use the deterministic assessment and candidate relations separately.",
                },
            },
            run_id=str(context["run_id"]),
            round_id=int(context["round_id"]),
            provider=self.provider_name,
        )


class NativeReThinkClient:
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
        profile: str = "sample_v1",
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
        reasoning_batch_size: int = 1,
        max_parallel_batches: int = 8,
        max_calls_per_round: int = 256,
        call_reserve: int = 96,
        dimension_parallel: bool = True,
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
        if not 1 <= reasoning_batch_size <= 8:
            raise ValueError("ReThink reasoning_batch_size must be between 1 and 8")
        if max_parallel_batches < 1:
            raise ValueError("ReThink max_parallel_batches must be positive")
        self.reasoning_batch_size = reasoning_batch_size
        self.max_parallel_batches = max_parallel_batches
        if max_calls_per_round < 1 or not 0 <= call_reserve < max_calls_per_round:
            raise ValueError("ReThink call budget and reserve are invalid")
        self.max_calls_per_round = max_calls_per_round
        self.call_reserve = call_reserve
        self.dimension_parallel = dimension_parallel
        self._budget_lock = threading.Lock()
        self._attempt_budget: LLMAttemptBudget | None = None
        self._bridge_lock = threading.Lock()
        self._bridge_sequence = 0
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
        if not hasattr(self, "_bridge_lock"):
            self._bridge_lock = threading.Lock()
            self._bridge_sequence = 0
        with self._bridge_lock:
            self._bridge_sequence += 1
            return f"IDB{self._bridge_sequence:06d}"

    def _consume_attempt(self, metadata: dict[str, Any]) -> None:
        budget = self._attempt_budget
        if budget is not None:
            budget.consume(metadata)

    def _release_attempt(self, metadata: dict[str, Any]) -> None:
        budget = self._attempt_budget
        if budget is not None:
            budget.release(metadata)

    def _reset_attempt_budget(self) -> LLMAttemptBudget:
        budget = LLMAttemptBudget(
            limit=getattr(self, "max_calls_per_round", 256),
            reserve=getattr(self, "call_reserve", 96),
            concurrency_limit=getattr(self, "max_parallel_batches", 8),
            provider=getattr(self, "provider_name", "rethink"),
        )
        if not hasattr(self, "_budget_lock"):
            self._budget_lock = threading.Lock()
        with self._budget_lock:
            self._attempt_budget = budget
        return budget

    @staticmethod
    def _is_splittable_output_failure(error: Exception) -> bool:
        return isinstance(error, RemoteLLMCompletionError) and (
            error.error_code in {"OUTPUT_TRUNCATED", "OUTPUT_JSON_INVALID"}
        )

    def _reflect_batch(
        self,
        *,
        context: ReThinkContextInput,
        batch_id: str,
        split_depth: int,
    ) -> tuple[ReThinkReflection, ...]:
        bridge = _rethink_bridge(
            context,
            scope_id=self._new_bridge_scope(),
            schema_name="ReThinkItemsOutput",
        )
        alias_context = ReThinkContextInput.model_validate(
            bridge.encode_projection(context.model_dump(mode="json"))
        )
        expected_aliases = alias_context.expected_variant_ids
        trace_context = AgentTraceContext(
            run_id=context.run_id,
            round_id=context.round_id,
            role="rethink",
            request_id=(
                f"rethink:{context.run_id}:r{context.round_id}:{batch_id}"
            ),
        )
        report_llm_id_bridge(
            round_id=context.round_id,
            **bridge.audit_payload(),
        )

        def validate_output(value: dict[str, Any]) -> dict[str, Any]:
            decoded = bridge.decode_and_validate(value)
            _validate_rethink_items(
                decoded,
                expected_variant_ids=context.expected_variant_ids,
            )
            return value

        output = complete_structured(
            client=self.client,
            transport=self.transport,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        self.profile + "\nReturn JSON matching ReThinkItemsOutput. "
                        "Use only the request-local S labels in candidates; local code maps them "
                        "back to canonical sample records. "
                        "Return exactly one sample-level reflection and all eight dimension "
                        "assessments per candidate. Do not return a batch assessment; the runtime "
                        "owns and injects it. "
                        + json.dumps(ReThinkItemsOutput.model_json_schema(), ensure_ascii=False)
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(_compact_rethink_prompt(alias_context), ensure_ascii=False),
                },
            ],
            output_type=ReThinkItemsOutput,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            render_max_tokens=getattr(self, "render_max_tokens", self.max_tokens),
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
            retries=0,
            transport_retries=getattr(self, "max_transport_retries", 2),
            truncation_retries=getattr(self, "max_truncation_retries", 1),
            syntax_retries=getattr(self, "max_syntax_retries", 1),
            schema_retries=getattr(self, "max_schema_retries", 2),
            semantic_retries=getattr(self, "max_semantic_retries", 1),
            unknown_evidence_retries=getattr(
                self, "max_unknown_evidence_retries", 1
            ),
            retry_backoff_seconds=getattr(self, "retry_backoff_seconds", 0.0),
            allow_unknown_evidence_stripping=getattr(
                self, "allow_unknown_evidence_stripping", False
            ),
            max_input_chars=getattr(self, "max_input_chars", None),
            separate_json_render=True,
            repair_hints={
                "reflections[].variant_id": tuple(expected_aliases)
            },
            contextual_validator=validate_output,
            reasoning_truncation_retries=0,
            preserve_reasoning_on_retry=True,
            trace_context={
                **trace_context.model_dump(mode="json"),
                "profile": self.profile_name,
                "profile_version": getattr(self, "profile_version", None),
                "schema_name": "ReThinkOutput",
                "retry_scope": f"rethink:{batch_id}",
                "rethink_batch_id": batch_id,
                "rethink_batch_size": len(context.candidates),
                "rethink_split_depth": split_depth,
                "id_bridge_scope": bridge.scope_id,
            },
            attempt_guard=self._consume_attempt,
            attempt_release=self._release_attempt,
        )
        assessment = context.hypothesis_assessment
        complete_output = ReThinkOutput(
            reflections=output.reflections,
            batch_assessment={
                "assessment_id": assessment.assessment_id if assessment is not None else None,
                "status": assessment.status if assessment is not None else "NOT_APPLICABLE",
                "commentary": "Runtime-owned deterministic batch assessment; sample reflections are advisory.",
                "next_round_advice": "Use sample-level dimensions without overriding the deterministic batch result.",
            },
        )
        canonical_payload = bridge.decode_and_validate(
            complete_output.model_dump(mode="json"),
            record_receipts=False,
        )
        report_llm_id_bridge(
            round_id=context.round_id,
            **bridge.audit_payload(),
        )
        reflections = _parse_reflections(
            canonical_payload,
            run_id=context.run_id,
            round_id=context.round_id,
            provider=self.provider_name,
        )
        return reflections

    def reflect_round(self, *, context: ReThinkContextInput) -> tuple[ReThinkReflection, ...]:
        validated_context = ReThinkContextInput.model_validate(context)
        candidates = tuple(validated_context.candidates)
        if not candidates:
            return ()
        budget = self._reset_attempt_budget()
        if getattr(self, "dimension_parallel", False):
            output = self._reflect_round_dimensions(validated_context)
            report_event(
                "rethink_attempt_budget_completed",
                message="ReThink provider attempt budget completed",
                round_id=validated_context.round_id,
                **budget.snapshot(),
            )
            return output
        batch_size = getattr(self, "reasoning_batch_size", 1)
        planned_calls = (len(candidates) + batch_size - 1) // batch_size
        max_calls = getattr(self, "max_calls_per_round", 256)
        reserve = getattr(self, "call_reserve", 96)
        usable_calls = max_calls - reserve
        if planned_calls > usable_calls:
            raise ValueError(
                "ReThink logical call plan exceeds the call budget after retry reserve; "
                f"planned={planned_calls}, usable={usable_calls}"
            )
        by_variant_id: dict[str, ReThinkReflection] = {}
        batches = adaptive_batch_submit(
            candidates,
            item_id=lambda item: item.variant_id,
            submit_batch=lambda work: self._reflect_batch_work(
                context=validated_context,
                work=work,
            ),
            initial_batch_size=getattr(self, "reasoning_batch_size", 1),
            max_parallel_batches=getattr(self, "max_parallel_batches", 8),
            should_split_failure=self._is_splittable_output_failure,
            role="rethink",
            round_id=validated_context.round_id,
            event_reporter=report_event,
        )
        for batch in batches:
            for reflection in batch.output:
                if reflection.variant_id in by_variant_id:
                    raise ValueError(
                        "Adaptive ReThink batches returned duplicate variant_id "
                        f"{reflection.variant_id!r}"
                    )
                by_variant_id[reflection.variant_id] = reflection
        expected_ids = tuple(item.variant_id for item in candidates)
        missing = sorted(set(expected_ids).difference(by_variant_id))
        unexpected = sorted(set(by_variant_id).difference(expected_ids))
        if missing or unexpected:
            raise ValueError(
                "Adaptive ReThink batch coverage mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        output = tuple(
            replace(by_variant_id[item], reflection_id=f"R{validated_context.round_id:02d}-{index:02d}")
            for index, item in enumerate(expected_ids, start=1)
        )
        report_event(
            "rethink_attempt_budget_completed",
            message="ReThink provider attempt budget completed",
            round_id=validated_context.round_id,
            **budget.snapshot(),
        )
        return output

    def _reflect_dimension_group(
        self,
        *,
        context: ReThinkContextInput,
        group_name: str,
        dimensions: tuple[str, str],
    ) -> ReThinkDimensionGroupOutput:
        candidate = context.candidates[0]
        bridge = _rethink_bridge(
            context,
            scope_id=self._new_bridge_scope(),
            schema_name="ReThinkDimensionGroupOutput",
        )
        alias_context = ReThinkContextInput.model_validate(
            bridge.encode_projection(context.model_dump(mode="python"))
        )
        alias_variant_id = bridge.namespaces["S"].encode(
            candidate.variant_id, strict=True
        )
        report_llm_id_bridge(
            round_id=context.round_id,
            **bridge.audit_payload(),
        )

        def validate_output(value: dict[str, Any]) -> dict[str, Any]:
            decoded = bridge.decode_and_validate(value)
            self._validate_dimension_group(
                decoded,
                expected_variant_id=candidate.variant_id,
                expected_dimensions=frozenset(dimensions),
            )
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
                        + "\nAnalyze only the two named dimensions for exactly one sample. "
                        "Return ReThinkDimensionGroupOutput JSON. Do not return a batch verdict "
                        "or hidden reasoning. Required dimensions: "
                        + json.dumps(dimensions)
                        + ". Schema: "
                        + json.dumps(
                            ReThinkDimensionGroupOutput.model_json_schema(),
                            ensure_ascii=False,
                        )
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            **_compact_rethink_prompt(alias_context),
                            "dimension_group": group_name,
                            "required_dimensions": list(dimensions),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            output_type=ReThinkDimensionGroupOutput,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            render_max_tokens=getattr(self, "render_max_tokens", self.max_tokens),
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
            retries=0,
            transport_retries=min(self.max_transport_retries, 1),
            truncation_retries=0,
            syntax_retries=0,
            schema_retries=0,
            semantic_retries=0,
            unknown_evidence_retries=0,
            empty_retries=0,
            retry_backoff_seconds=self.retry_backoff_seconds,
            max_input_chars=self.max_input_chars,
            separate_json_render=True,
            contextual_validator=validate_output,
            repair_hints={"variant_id": (alias_variant_id,)},
            trace_context={
                "run_id": context.run_id,
                "round_id": context.round_id,
                "role": "rethink",
                "profile": self.profile_name,
                "profile_version": self.profile_version,
                "request_id": (
                    f"rethink:{context.run_id}:r{context.round_id}:"
                    f"{alias_variant_id}:{group_name}"
                ),
                "rethink_scope": "sample_dimension_group",
                "rethink_dimension_group": group_name,
                "id_bridge_scope": bridge.scope_id,
            },
            attempt_guard=self._consume_attempt,
            attempt_release=self._release_attempt,
        )
        decoded = bridge.decode_and_validate(
            output.model_dump(mode="json"),
            record_receipts=False,
        )
        report_llm_id_bridge(
            round_id=context.round_id,
            **bridge.audit_payload(),
        )
        return ReThinkDimensionGroupOutput.model_validate(decoded)

    @staticmethod
    def _validate_dimension_group(
        payload: dict[str, Any],
        *,
        expected_variant_id: str,
        expected_dimensions: frozenset[str],
    ) -> dict[str, Any]:
        output = ReThinkDimensionGroupOutput.model_validate(payload)
        actual = {item.dimension for item in output.dimension_assessments}
        if output.variant_id != expected_variant_id or actual != expected_dimensions:
            raise ValueError("ReThink dimension group does not match its sample/group request")
        return output.model_dump(mode="json")

    @staticmethod
    def _fallback_dimension_group(
        *,
        candidate_id: str,
        group_name: str,
        dimensions: tuple[str, str],
        error: Exception,
    ) -> ReThinkDimensionGroupOutput:
        del error
        return ReThinkDimensionGroupOutput(
            variant_id=candidate_id,
            dimension_assessments=[
                {
                    "dimension": dimension,
                    "evidence_status": "missing",
                    "relation_to_sample_rationale": "unresolved",
                    "finding_code": "remote_group_unavailable",
                    "finding": (
                        "The remote ReThink group did not produce a validated result for "
                        "this sample; no dimension-specific conclusion is available."
                    ),
                    "implication": (
                        "Keep this dimension unresolved and ineligible for selection until a "
                        "validated sample-level review is available."
                    ),
                    "quality_status": "deterministic_fallback",
                }
                for dimension in dimensions
            ],
            group_advice=(
                f"Repeat the {group_name} review with the same request-local evidence scope; "
                "do not infer a direction from this fallback."
            ),
        )

    def _reflect_round_dimensions(
        self, context: ReThinkContextInput
    ) -> tuple[ReThinkReflection, ...]:
        candidates = tuple(context.candidates)
        calls_per_completion = 1 if self.thinking == "disabled" else 2
        planned_calls = (
            len(candidates) * len(RETHINK_DIMENSION_GROUPS) * calls_per_completion
        )
        usable_calls = self.max_calls_per_round - self.call_reserve
        if planned_calls > usable_calls:
            raise ValueError(
                "ReThink dimension call plan exceeds the call budget after retry reserve; "
                f"planned={planned_calls}, usable={usable_calls}"
            )
        grouped: dict[str, dict[str, ReThinkDimensionGroupOutput]] = {
            item.variant_id: {} for item in candidates
        }
        with ThreadPoolExecutor(max_workers=self.max_parallel_batches) as executor:
            futures = {}
            for candidate in candidates:
                sample_context = context.model_copy(update={"candidates": [candidate]})
                for group_name, dimensions in RETHINK_DIMENSION_GROUPS.items():
                    future = executor.submit(
                        copy_context().run,
                        partial(
                            self._reflect_dimension_group,
                            context=sample_context,
                            group_name=group_name,
                            dimensions=dimensions,
                        ),
                    )
                    futures[future] = (candidate.variant_id, group_name, dimensions)
            for future in as_completed(futures):
                variant_id, group_name, dimensions = futures[future]
                try:
                    grouped[variant_id][group_name] = future.result()
                except Exception as error:  # noqa: BLE001 - isolate one advisory group
                    grouped[variant_id][group_name] = self._fallback_dimension_group(
                        candidate_id=variant_id,
                        group_name=group_name,
                        dimensions=dimensions,
                        error=error,
                    )
                    report_event(
                        "rethink_dimension_group_degraded",
                        message="ReThink dimension group degraded to a typed fallback",
                        round_id=context.round_id,
                        variant_id=variant_id,
                        group_name=group_name,
                        error_type=type(error).__name__,
                        error=str(error),
                    )

        assessment = context.hypothesis_assessment
        output: list[ReThinkReflection] = []
        for index, candidate in enumerate(candidates, start=1):
            groups = grouped[candidate.variant_id]
            dimensions = tuple(
                item.model_dump(mode="json")
                for group_name in RETHINK_DIMENSION_GROUPS
                for item in groups[group_name].dimension_assessments
            )
            wet_support = (
                candidate.wet_value > context.visible_baseline
                if context.measurement_contract.optimization_direction == "higher_is_better"
                else candidate.wet_value < context.visible_baseline
            )
            dry_values = [item.value for item in candidate.dry_validations]
            dry_agrees = True
            if dry_values:
                dry_mean = sum(dry_values) / len(dry_values)
                dry_support = (
                    dry_mean > context.visible_baseline
                    if context.measurement_contract.optimization_direction == "higher_is_better"
                    else dry_mean < context.visible_baseline
                )
                dry_agrees = dry_support == wet_support
            relation = _relation_for_candidate(
                intent_arm=candidate.intent_arm,
                wet_support=wet_support,
                dry_agrees=dry_agrees,
            )
            advice, group_advice = _prioritized_group_advice(groups)
            degraded = any(
                item.get("quality_status") == "deterministic_fallback"
                for item in dimensions
            )
            output.append(
                ReThinkReflection(
                    reflection_id=f"R{context.round_id:02d}-{index:02d}",
                    variant_id=candidate.variant_id,
                    round_id=context.round_id,
                    verdict=relation,
                    summary=(
                        f"Revealed value {candidate.wet_value:.4g} versus visible baseline "
                        f"{context.visible_baseline:.4g}; eight dimensions reviewed."
                    ),
                    positive_findings=tuple(
                        item["finding"]
                        for item in dimensions
                        if item.get("relation_to_sample_rationale") == "positive"
                    )[:8],
                    negative_findings=tuple(
                        item["finding"]
                        for item in dimensions
                        if item.get("relation_to_sample_rationale") == "negative"
                    )[:8],
                    revised_reason=_bounded_join(
                        [
                            "Treat the original selection rationale as round- and sample-specific.",
                            candidate.critic_explanation,
                            *candidate.critic_suggestions[:2],
                        ],
                        limit=2000,
                    ),
                    next_round_advice=advice,
                    next_round_action=(
                        "retain_uncertainty_aware_exploration"
                        if wet_support and dry_agrees
                        else "test_matched_alternative"
                    ),
                    provider=self.provider_name,
                    assessment_id=assessment.assessment_id if assessment else None,
                    assessment_status=assessment.status if assessment else "NOT_APPLICABLE",
                    assessment_commentary=(
                        "Runtime-owned deterministic batch assessment; sample dimensions are advisory."
                    ),
                    quality_status=(
                        "deterministic_fallback" if degraded else "model"
                    ),
                    dimension_assessments=dimensions,
                    dimension_group_advice=group_advice,
                )
            )
        return tuple(output)

    def _reflect_batch_work(
        self,
        *,
        context: ReThinkContextInput,
        work: AdaptiveBatchWork[Any],
    ) -> tuple[ReThinkReflection, ...]:
        batch_context = context.model_copy(update={"candidates": list(work.items)})
        return self._reflect_batch(
            context=batch_context,
            batch_id=work.batch_id,
            split_depth=work.split_depth,
        )


OpenAICompatibleReThinkClient = NativeReThinkClient


def create_sample_rethink_client(provider: str, **kwargs: Any):
    if "runtime" in kwargs:
        runtime = str(kwargs.pop("runtime"))
        if runtime != "chat_completions":
            raise ValueError(f"Removed Agents SDK runtime is not supported: {runtime!r}")
    if provider == "mock":
        return MockReThinkClient()
    if provider in {"openai", "openai_compatible", "deepseek"}:
        if kwargs.get("profile") in {None, "scientific_v1"}:
            kwargs["profile"] = "sample_v1"
        if provider == "deepseek":
            kwargs.setdefault("provider", "deepseek")
            kwargs.setdefault(
                "base_url", resolve_base_url(kwargs.get("base_url"), provider="deepseek")
            )
            kwargs.setdefault("model", resolve_model(kwargs.get("model"), provider="deepseek"))
        return NativeReThinkClient(**kwargs)
    raise ValueError(f"Unknown ReThink provider {provider!r}")
