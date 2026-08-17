"""Bounded native tool loop; it owns no campaign or scientific state."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from fitness_agents.kg_interaction.tool_runtime import RoundScopedToolExecutor

from .tool_contracts import AgentStep, ToolSpec


class LocalAgentLoop:
    def __init__(self, *, max_turns: int = 6) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns

    def run(
        self,
        *,
        next_step: Callable[[Sequence[dict[str, Any]], Sequence[ToolSpec]], AgentStep],
        messages: Sequence[dict[str, Any]],
        tools: Sequence[ToolSpec],
        executor: RoundScopedToolExecutor,
    ) -> dict[str, Any]:
        allowed = {tool.name: tool for tool in tools}
        transcript = list(messages)
        for _ in range(self.max_turns):
            step = next_step(tuple(transcript), tuple(tools))
            if step.final_output is not None:
                return step.final_output
            assert step.tool_call is not None
            try:
                spec = allowed[step.tool_call.name]
            except KeyError as error:
                raise ValueError(f"Tool is not allow-listed: {step.tool_call.name!r}") from error
            result = executor.call(spec.name, spec.intent, step.tool_call.arguments)
            transcript.append(
                {"role": "tool", "name": spec.name, "content": result}
            )
        raise RuntimeError("Local agent loop exhausted max_turns without final output")
