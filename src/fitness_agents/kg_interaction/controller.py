from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fitness_agents.plugin_registry import PluginRegistry

from .ablation import InteractionAblationConfig
from .contracts import (
    EvidencePack,
    InteractionResult,
    KGQueryContext,
    KGQueryPlan,
    KGQueryStep,
    QueryIntent,
)
from .operators import QueryOperator

_FORBIDDEN_ARGUMENT_KEYS = {
    "cypher",
    "final_test",
    "final_test_ids",
    "normalized_fitness",
    "oracle",
    "oracle_data_path",
    "oracle_path",
    "raw_fitness",
    "sparql",
    "sql",
}


@dataclass(frozen=True)
class EvidenceSufficiencyPolicy:
    min_fact_count: int = 1
    min_supporting_paths: int = 0
    require_counterevidence: bool = False

    def is_sufficient(self, packs: tuple[EvidencePack, ...]) -> bool:
        fact_count = sum(pack.fact_count for pack in packs)
        path_count = sum(len(pack.supporting_paths) for pack in packs)
        has_counterevidence = any(pack.has_counterevidence for pack in packs)
        return (
            fact_count >= self.min_fact_count
            and path_count >= self.min_supporting_paths
            and (not self.require_counterevidence or has_counterevidence)
        )


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key).lower())
            keys.update(_walk_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_walk_keys(item))
    return keys


def _variant_ids(step: KGQueryStep) -> frozenset[str]:
    values: set[str] = set()
    if "variant_id" in step.arguments:
        values.add(str(step.arguments["variant_id"]))
    values.update(str(item) for item in step.arguments.get("variant_ids", ()))
    return frozenset(values)


class KGInteractionController:
    """Execute a bounded, allow-listed KG plan and return auditable evidence packs."""

    def __init__(
        self,
        operators: PluginRegistry[QueryOperator],
        *,
        config: InteractionAblationConfig | None = None,
        sufficiency: EvidenceSufficiencyPolicy | None = None,
    ) -> None:
        self.operators = operators
        self.config = config or InteractionAblationConfig()
        self.sufficiency = sufficiency or EvidenceSufficiencyPolicy(
            require_counterevidence=self.config.use_counterevidence
        )

    def _validate_step(self, step: KGQueryStep, context: KGQueryContext) -> None:
        forbidden = _walk_keys(step.arguments).intersection(_FORBIDDEN_ARGUMENT_KEYS)
        if forbidden:
            raise ValueError(
                f"Step {step.step_id!r} contains forbidden arguments: {sorted(forbidden)}"
            )
        requested = _variant_ids(step)
        if context.allowed_variant_ids is not None:
            outside = requested.difference(context.allowed_variant_ids)
            if outside:
                raise ValueError(
                    f"Step {step.step_id!r} requests out-of-scope variants: {sorted(outside)}"
                )
        if "limit" in step.arguments:
            limit = int(step.arguments["limit"])
            if limit < 1 or limit > context.max_rows:
                raise ValueError(
                    f"Step {step.step_id!r} limit must be between 1 and {context.max_rows}"
                )
        if len(step.arguments.get("variant_ids", ())) > context.max_rows:
            raise ValueError(
                f"Step {step.step_id!r} requests more than {context.max_rows} variants"
            )

    @staticmethod
    def _validate_pack(pack: EvidencePack, context: KGQueryContext) -> None:
        for field_name in (
            "facts",
            "predictions",
            "evidence",
            "supporting_paths",
            "counterevidence",
            "directional_signals",
            "caveats",
            "provenance",
        ):
            if len(getattr(pack, field_name)) > context.max_rows:
                raise ValueError(
                    f"Operator {pack.operator!r} returned more than {context.max_rows} "
                    f"rows in {field_name}"
                )

    def execute(self, plan: KGQueryPlan, context: KGQueryContext) -> InteractionResult:
        packs: list[EvidencePack] = []
        executed: list[str] = []
        skipped: list[tuple[str, str]] = []
        call_budget = min(plan.max_tool_calls, self.config.max_tool_calls)

        for step in plan.steps:
            if len(executed) >= call_budget:
                skipped.append((step.step_id, "tool_call_budget_exhausted"))
                continue
            if not self.config.operator_enabled(step.operator):
                skipped.append((step.step_id, "operator_ablation"))
                continue
            if step.intent is QueryIntent.COUNTEREVIDENCE and not self.config.use_counterevidence:
                skipped.append((step.step_id, "counterevidence_ablation"))
                continue
            missing = [item for item in step.depends_on if item not in executed]
            if missing:
                skipped.append((step.step_id, f"missing_dependencies:{','.join(missing)}"))
                continue

            self._validate_step(step, context)
            operator = self.operators.get(step.operator)
            pack = operator.execute(step, context)
            if pack.as_of_round != context.round_id:
                raise ValueError(
                    f"Operator {step.operator!r} returned as_of_round={pack.as_of_round}, "
                    f"expected {context.round_id}"
                )
            self._validate_pack(pack, context)
            packs.append(pack)
            executed.append(step.step_id)
            if self.config.stop_when_sufficient and self.sufficiency.is_sufficient(tuple(packs)):
                return InteractionResult(
                    plan_id=plan.plan_id,
                    packs=tuple(packs),
                    executed_steps=tuple(executed),
                    skipped_steps=tuple(skipped),
                    stop_reason="evidence_sufficient",
                )

        stop_reason = (
            "evidence_sufficient"
            if self.sufficiency.is_sufficient(tuple(packs))
            else "plan_or_budget_exhausted"
        )
        return InteractionResult(
            plan_id=plan.plan_id,
            packs=tuple(packs),
            executed_steps=tuple(executed),
            skipped_steps=tuple(skipped),
            stop_reason=stop_reason,
        )
