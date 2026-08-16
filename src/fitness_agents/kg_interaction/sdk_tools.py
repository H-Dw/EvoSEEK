from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fitness_agents.utils.progress import report_event

from .contracts import (
    EvidencePack,
    InteractionResult,
    KGQueryContext,
    KGQueryPlan,
    KGQueryStep,
    QueryIntent,
)
from .controller import KGInteractionController


class KGToolSession:
    """Round-scoped SDK facade; every query still passes through KGInteractionController."""

    def __init__(
        self,
        controller: KGInteractionController,
        context: KGQueryContext,
        *,
        plan_id: str,
        max_tool_calls: int,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        self.controller = controller
        self.context = context
        self.plan_id = plan_id
        self.max_tool_calls = max_tool_calls
        self._packs: list[EvidencePack] = []
        self._executed_steps: list[str] = []
        self._skipped_steps: list[tuple[str, str]] = []
        self._call_attempts = 0

    @property
    def query_ids(self) -> tuple[str, ...]:
        return tuple(pack.query_id for pack in self._packs)

    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_tool_calls - self._call_attempts)

    def call(
        self,
        operator: str,
        intent: QueryIntent,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if self.remaining_calls < 1:
            raise RuntimeError("KG query budget exhausted for this round")
        self._call_attempts += 1
        step_id = f"sdk_tool_{self._call_attempts:02d}"
        try:
            result = self.controller.execute(
                KGQueryPlan(
                    plan_id=f"{self.plan_id}:{step_id}",
                    objective="Answer one bounded, read-only Scientist KG query.",
                    steps=(KGQueryStep(step_id, operator, intent, arguments),),
                    max_tool_calls=1,
                ),
                self.context,
            )
        except Exception as error:
            self._skipped_steps.append((step_id, f"rejected:{type(error).__name__}"))
            raise
        if not result.packs:
            reason = result.skipped_steps[0][1] if result.skipped_steps else result.stop_reason
            self._skipped_steps.append((step_id, reason))
            raise RuntimeError(f"KG query {operator!r} was not executed: {reason}")
        pack = result.packs[0]
        self._packs.append(pack)
        self._executed_steps.append(step_id)
        variant_ids = []
        if "variant_id" in arguments:
            variant_ids.append(str(arguments["variant_id"]))
        variant_ids.extend(str(item) for item in arguments.get("variant_ids", ()))
        report_event(
            "sdk_kg_tool_completed",
            message=f"SDK KG tool {operator} completed",
            run_id=self.context.run_id,
            round_id=self.context.round_id,
            variant_ids=variant_ids,
            operator=operator,
            query_id=pack.query_id,
            remaining_calls=self.remaining_calls,
        )
        return asdict(pack)

    def result(self) -> InteractionResult:
        return InteractionResult(
            plan_id=self.plan_id,
            packs=tuple(self._packs),
            executed_steps=tuple(self._executed_steps),
            skipped_steps=tuple(self._skipped_steps),
            stop_reason=(
                "no_tool_requested"
                if self._call_attempts == 0
                else (
                    "tool_call_budget_exhausted"
                    if self.remaining_calls == 0
                    else "agent_completed"
                )
            ),
        )
