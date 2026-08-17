"""Local, provider-neutral tool declarations and model actions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fitness_agents.kg_interaction import QueryIntent


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    description: str
    intent: QueryIntent
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tool_call: ToolCall | None = None
    final_output: dict[str, Any] | None = None

    @model_validator(mode="after")
    def exactly_one_action(self) -> AgentStep:
        if (self.tool_call is None) == (self.final_output is None):
            raise ValueError("Agent step requires exactly one tool_call or final_output")
        return self
