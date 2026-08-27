from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from fitness_agents.safety import discover_workspace_access_policy

from .contracts import EvidenceProductBundle, ResearchBrief, ScopeAssertion


def _read_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path).absolute()
    access_policy = discover_workspace_access_policy(source)
    access_policy.require_allowed(source)
    if source.is_symlink() or bool(getattr(source, "is_junction", lambda: False)()):
        raise ValueError("Structured Deep Research input cannot be a symlink or junction")
    resolved_source = source.resolve()
    access_policy.require_allowed(resolved_source)
    text = resolved_source.read_text(encoding="utf-8")
    payload = (
        json.loads(text)
        if resolved_source.suffix.casefold() == ".json"
        else yaml.safe_load(text)
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Structured input must be a mapping: {source}")
    return {str(key): value for key, value in payload.items()}


def _validate_json_mode(model: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
    return model.model_validate_json(json.dumps(payload, ensure_ascii=False, default=str))


def load_research_brief(path: str | Path) -> ResearchBrief:
    return _validate_json_mode(ResearchBrief, _read_mapping(path))  # type: ignore[return-value]


def load_evidence_product(path: str | Path) -> EvidenceProductBundle:
    payload = _read_mapping(path)
    schema_version = payload.get("schema_version")
    if schema_version == "scientific-evidence-product:v1":
        raise ValueError(
            "Evidence product v1 lacks canonical ScopeAssertion/SearchHit provenance; "
            "rebuild reviews, approvals, and manifest as v2"
        )
    return _validate_json_mode(  # type: ignore[return-value]
        EvidenceProductBundle,
        payload,
    )


def load_scope_assertions(path: str | Path | None) -> dict[str, ScopeAssertion]:
    if path is None:
        return {}
    payload = _read_mapping(path)
    if payload.get("schema_version") != "external-scope-assertions:v1":
        raise ValueError("Unsupported scope-assertion collection schema")
    raw_assertions = payload.get("assertions", [])
    if not isinstance(raw_assertions, list):
        raise TypeError("Scope assertion file requires an assertions list")
    assertions: dict[str, ScopeAssertion] = {}
    assertion_ids: set[str] = set()
    for raw in raw_assertions:
        assertion = ScopeAssertion.model_validate_json(
            json.dumps(raw, ensure_ascii=False, default=str)
        )
        if assertion.artifact_id in assertions:
            raise ValueError(f"Duplicate scope assertion for {assertion.artifact_id}")
        if assertion.scope_assertion_id in assertion_ids:
            raise ValueError(
                f"Duplicate scope assertion ID: {assertion.scope_assertion_id}"
            )
        assertions[assertion.artifact_id] = assertion
        assertion_ids.add(assertion.scope_assertion_id)
    return assertions


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    access_policy = discover_workspace_access_policy(target)
    access_policy.require_allowed(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
