#!/usr/bin/env python3
"""Replay hypothesis prompts and critic-REVISE retries against a live LLM.

Offline default is skip. Pass ``--enable-remote`` (and a DeepSeek key) to call the API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from common import (
    ensure,
    jsonable,
    load_config,
    parse_args,
    placeholder,
    resolve_output,
    write_result,
)

from fitness_agents.agents.llm import (
    HYPOTHESIS_SCHEMA,
    OpenAICompatibleLLMClient,
    build_scientist_hypothesis_messages,
)
from fitness_agents.agents.output_contracts import validate_hypothesis_payload
from fitness_agents.agents.output_guards import classify_output_failure, json_salvage
from fitness_agents.agents.profile_loader import load_role_profile
from fitness_agents.agents.remote_llm import (
    complete_json,
    create_openai_client,
    extract_json_object,
    load_project_env,
    resolve_secret,
)
from fitness_agents.contracts.agent_io import ScientistContextInput
from fitness_agents.contracts.schemas import Evidence

FAILED_RUN = (
    Path(__file__).resolve().parents[2]
    / "artifacts/reasoning-routes-20260818T201253Z/runs/"
    "knowledge_agent-s11-f02-GB1-reasoning-rag_kg_physchem-f02-20260819T040426219460Z"
)


def _configure_remote(values: dict[str, object]) -> dict[str, str]:
    load_project_env()
    api_key = resolve_secret(
        values.get("api_key"),
        "FITNESS_AGENTS_LLM_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    )
    model = resolve_secret(values.get("scientist_model"), "FITNESS_AGENTS_LLM_MODEL") or values.get(
        "scientist_model"
    )
    base_url = resolve_secret(
        values.get("base_url"),
        "FITNESS_AGENTS_LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "DEEPSEEK_BASE_URL",
    )
    ensure(bool(api_key), "Set DEEPSEEK_API_KEY before remote testing")
    ensure(bool(model) and not placeholder(model), "Replace remote_llm.scientist_model")
    ensure(bool(base_url) and not placeholder(base_url), "Replace remote_llm.base_url")
    os.environ["FITNESS_AGENTS_LLM_API_KEY"] = str(api_key)
    os.environ["DEEPSEEK_API_KEY"] = str(api_key)
    os.environ["OPENAI_BASE_URL"] = str(base_url)
    os.environ["FITNESS_AGENTS_LLM_BASE_URL"] = str(base_url)
    os.environ["FITNESS_AGENTS_LLM_MODEL"] = str(model)
    return {
        "api_key": str(api_key),
        "model": str(model),
        "base_url": str(base_url),
    }


def _allowed_evidence_ids(
    context: ScientistContextInput, evidence: list[Evidence]
) -> frozenset[str]:
    pack_ids = {
        str(item["evidence_id"])
        for pack in (context.kg_interaction or {}).get("packs", ())
        for item in pack.get("evidence", ())
        if isinstance(item, dict) and item.get("evidence_id")
    }
    return frozenset({item.evidence_id for item in evidence} | pack_ids)


def _write_report(output: Path, payload: dict[str, Any]) -> Path:
    path = output / "llm_output_stability_report.json"
    path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _synthetic_context() -> tuple[ScientistContextInput, list[Evidence]]:
    context = ScientistContextInput.model_validate(
        {
            "run_id": "live-stability",
            "mode": "knowledge_agent",
            "round_id": 3,
            "expected_hypothesis_id": "hyp:live-stability:r3",
            "task": "maximize visible fitness",
            "protein_id": "GB1",
            "objective": "maximize",
            "mutable_positions": [39, 40, 41, 54],
            "wild_type_sites": "VDGV",
            "protein_context_id": "ctx:live",
            "visible_observations": [],
            "previous_hypothesis_id": "hyp:live-stability:r2",
            "previous_hypothesis_assessment": None,
        }
    )
    evidence = [
        Evidence(
            evidence_id="ev:live:1",
            variant_id="context:protein",
            channel="physchem",
            statement="Hydrophobic substitutions at the mutable sites are visible in this round.",
            score=0.4,
            source_id="live:source",
            confidence=0.7,
            round_id=3,
        )
    ]
    return context, evidence


def _messages_from_failed_run() -> tuple[list[dict[str, str]], frozenset[str], tuple[int, ...]] | None:
    if not FAILED_RUN.is_dir():
        return None
    config_path = FAILED_RUN / "config.json"
    if not config_path.is_file():
        return None
    context, evidence = _synthetic_context()
    kg_path = FAILED_RUN / "round_03" / "kg_interaction.json"
    if kg_path.is_file():
        interaction = json.loads(kg_path.read_text(encoding="utf-8"))
        dumped = context.model_dump(mode="json")
        dumped["kg_interaction"] = interaction
        context = ScientistContextInput.model_validate(dumped)
    profile = load_role_profile("scientist", "scientific_v1").instructions
    messages = build_scientist_hypothesis_messages(
        profile=profile,
        sanitized_context=context,
        evidence=evidence,
        output_schema=HYPOTHESIS_SCHEMA,
    )
    return messages, _allowed_evidence_ids(context, evidence), (39, 40, 41, 54)


def _score_completion(content: str, finish_reason: str | None, allowed: frozenset[str]) -> dict[str, Any]:
    raw_ok = False
    salvage_ok = False
    schema_ok = False
    truncated = str(finish_reason or "").lower() in {"length", "max_tokens", "max_output_tokens"}
    error = None
    try:
        parsed = json.loads(content)
        raw_ok = isinstance(parsed, dict)
    except json.JSONDecodeError as exc:
        error = str(exc)
        parsed = json_salvage(content)
        salvage_ok = parsed is not None
    else:
        salvage_ok = True
        parsed = parsed if isinstance(parsed, dict) else None
    if parsed is None:
        try:
            parsed = extract_json_object(content)
            salvage_ok = True
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            error = str(exc)
    if parsed is not None:
        try:
            validate_hypothesis_payload(
                parsed,
                allowed_evidence_ids=allowed,
                expected_positions=(39, 40, 41, 54),
                on_unknown_evidence="strip",
            )
            schema_ok = True
        except Exception as exc:  # noqa: BLE001 - live eval records any contract miss
            error = f"{type(exc).__name__}: {exc}"
            failure = classify_output_failure(exc, finish_reason=finish_reason, content=content)
            truncated = truncated or failure.kind == "truncated"
    return {
        "raw_json_ok": raw_ok,
        "salvage_or_extract_ok": salvage_ok,
        "schema_ok": schema_ok,
        "truncated": truncated,
        "finish_reason": finish_reason,
        "content_chars": len(content or ""),
        "error": error,
    }


def _run_hypothesis_replay(remote: dict[str, str], *, repeats: int) -> dict[str, Any]:
    packed = _messages_from_failed_run()
    if packed is None:
        context, evidence = _synthetic_context()
        from fitness_agents.agents.profile_loader import load_role_profile

        messages = build_scientist_hypothesis_messages(
            profile=load_role_profile("scientist", "scientific_v1").instructions,
            sanitized_context=context,
            evidence=evidence,
            output_schema=HYPOTHESIS_SCHEMA,
        )
        allowed = _allowed_evidence_ids(context, evidence)
        source = "synthetic"
    else:
        messages, allowed, _positions = packed
        source = "failed_run_round_03"
    client = create_openai_client(api_key=remote["api_key"], base_url=remote["base_url"])
    trials = []
    for thinking in ("enabled", "disabled"):
        for index in range(repeats):
            captured: dict[str, Any] = {}

            class _Capture:
                base_url = remote["base_url"]

                def create_chat_completion(self, **kwargs):
                    captured["kwargs"] = kwargs
                    response = client.chat.completions.create(**kwargs)
                    captured["finish_reason"] = response.choices[0].finish_reason
                    message = response.choices[0].message
                    captured["content"] = getattr(message, "content", None) or ""
                    captured["usage"] = getattr(response, "usage", None)
                    return response

            try:
                payload = complete_json(
                    client=client,
                    transport=_Capture(),  # type: ignore[arg-type]
                    model=remote["model"],
                    messages=messages,
                    max_tokens=int(os.environ.get("FITNESS_AGENTS_LLM_MAX_TOKENS", "16384")),
                    thinking=thinking,
                    retries=2,
                    validator=lambda value: validate_hypothesis_payload(
                        value,
                        allowed_evidence_ids=allowed,
                        expected_positions=(39, 40, 41, 54),
                    ),
                )
                contract_ok = True
                error = None
            except Exception as exc:  # noqa: BLE001 - live eval
                payload = None
                contract_ok = False
                error = f"{type(exc).__name__}: {exc}"
            usage = captured.get("usage")
            usage_payload = {}
            if usage is not None:
                details = getattr(usage, "completion_tokens_details", None)
                usage_payload = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "reasoning_tokens": getattr(details, "reasoning_tokens", None)
                    if details is not None
                    else None,
                }
            score = _score_completion(
                str(captured.get("content") or ""),
                captured.get("finish_reason"),
                allowed,
            )
            trials.append(
                {
                    "thinking": thinking,
                    "index": index,
                    "guarded_contract_ok": contract_ok,
                    "payload_keys": sorted(payload) if isinstance(payload, dict) else [],
                    "error": error,
                    "usage": usage_payload,
                    **score,
                }
            )
    n = len(trials)
    return {
        "source": source,
        "n": n,
        "raw_json_ok": sum(item["raw_json_ok"] for item in trials),
        "salvage_or_extract_ok": sum(item["salvage_or_extract_ok"] for item in trials),
        "schema_ok": sum(item["schema_ok"] for item in trials),
        "guarded_contract_ok": sum(item["guarded_contract_ok"] for item in trials),
        "truncated": sum(item["truncated"] for item in trials),
        "trials": trials,
    }


def _run_revise_reproposal(remote: dict[str, str]) -> dict[str, Any]:
    scientist = OpenAICompatibleLLMClient(
        model=remote["model"],
        base_url=remote["base_url"],
        api_key=remote["api_key"],
        thinking="disabled",
        max_tokens=8192,
    )
    context, evidence = _synthetic_context()
    first = scientist.generate_hypothesis(
        sanitized_context=context,
        evidence=evidence,
        output_schema=HYPOTHESIS_SCHEMA,
    )
    revision_context = ScientistContextInput.model_validate(
        {
            **context.model_dump(mode="json"),
            "expected_hypothesis_id": "hyp:live-stability:r3:a1",
            "previous_hypothesis_id": first.hypothesis_id,
            "critic_revision": {
                "verdict": "REVISE",
                "summary": "Increase diversity and do not repeat the same residue map.",
                "required_changes": [
                    {
                        "action": "REGENERATE_WITH_CONSTRAINTS",
                        "rationale": "Hypothesis restates a crowded batch.",
                    }
                ],
                "rejected_hypothesis_id": first.hypothesis_id,
                "rejected_statement": first.statement,
                "rejected_preferred_residues": {
                    str(site): list(residues) for site, residues in first.preferred_residues.items()
                },
            },
        }
    )
    second = scientist.generate_hypothesis(
        sanitized_context=revision_context,
        evidence=evidence,
        output_schema=HYPOTHESIS_SCHEMA,
    )
    changed = (
        second.preferred_residues != first.preferred_residues or second.statement != first.statement
    )
    visible = {item.evidence_id for item in evidence}
    unknown = [item for item in second.evidence_ids if item not in visible]
    return {
        "first_id": first.hypothesis_id,
        "second_id": second.hypothesis_id,
        "parent_matches": second.parent_hypothesis_id == first.hypothesis_id,
        "changed_statement_or_residues": changed,
        "unknown_evidence_ids": unknown,
        "second_statement": second.statement,
    }


def main() -> None:
    args = parse_args("configs/module_tests/agents_review.yaml", remote=True)
    config = load_config(args.config)
    output = resolve_output(config, args.output_dir)
    remote_cfg = dict(config.get("remote_llm") or {})
    if not args.enable_remote and not remote_cfg.get("enabled"):
        write_result(
            output,
            "llm_output_stability",
            {"skipped": True, "reason": "Pass --enable-remote to call the live LLM API"},
        )
        return
    remote = _configure_remote(remote_cfg)
    replay = _run_hypothesis_replay(remote, repeats=5)
    _write_report(output, {"hypothesis_replay": replay, "revise_reproposal": {"pending": True}})
    try:
        revise = _run_revise_reproposal(remote)
    except Exception as exc:  # noqa: BLE001 - persist replay stats even if REVISE eval fails
        revise = {"error": f"{type(exc).__name__}: {exc}"}
        _write_report(output, {"hypothesis_replay": replay, "revise_reproposal": revise})
        raise
    report = {"hypothesis_replay": replay, "revise_reproposal": revise}
    _write_report(output, report)
    ensure(
        replay["guarded_contract_ok"] >= 1 or replay["schema_ok"] >= 1,
        "Live hypothesis replay produced no valid JSON contract",
    )
    ensure(revise["parent_matches"], "Revised hypothesis did not point at the rejected parent")
    ensure(not revise["unknown_evidence_ids"], "Revised hypothesis cited invisible evidence_ids")
    write_result(output, "llm_output_stability", report)


if __name__ == "__main__":
    main()
