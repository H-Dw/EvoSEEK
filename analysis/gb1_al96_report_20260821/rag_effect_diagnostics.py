"""Artifact-grounded diagnostics for the incremental effect of local RAG."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cases import build_candidate_table
from config import EXPECTED_ROUNDS, REPO_ROOT
from io_artifacts import RunArtifact, read_json


BASE_CONDITION = "kg_base"
RAG_CONDITION = "kg_base_rag"
MUTABLE_POSITIONS = ("39", "40", "41", "54")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _short_evidence_ids(values: Any) -> list[str]:
    return sorted(
        {
            str(value)
            for value in (values or [])
            if re.fullmatch(r"E\d+", str(value))
        }
    )


def _evidence_universe_entries(prompt_payload: dict[str, Any]) -> list[dict[str, Any]]:
    universe = prompt_payload.get("evidence_universe") or []
    if isinstance(universe, dict):
        return list(universe.get("entries", []))
    if isinstance(universe, list) and universe and isinstance(universe[0], dict):
        return list(universe[0].get("entries", []))
    return []


def _conversation(round_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    conversation_dir = round_dir / "llm" / "scientist" / "conversations"
    paths = sorted(conversation_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No Scientist conversation in {conversation_dir}")
    candidates = [(path, read_json(path)) for path in paths]
    accepted = [
        item
        for item in candidates
        if item[1].get("disposition") == "accepted"
        and item[1].get("conversation_stage") == "reasoning_draft"
    ]
    path, conversation = (accepted or candidates)[-1]
    user_messages = [
        item for item in conversation.get("messages", []) if item.get("role") == "user"
    ]
    if not user_messages:
        raise ValueError(f"Scientist conversation lacks user input: {path}")
    prompt_payload = json.loads(user_messages[-1]["content"])
    response_payload = json.loads(conversation["response_content"])
    return path, conversation, prompt_payload, response_payload


def _run_map(runs: list[RunArtifact], condition: str) -> dict[int, RunArtifact]:
    mapping = {
        run.fold: run
        for run in runs
        if run.eligible and run.condition == condition
    }
    if set(mapping) != {0, 1, 2}:
        raise ValueError(f"Expected folds 0-2 for {condition}, found {sorted(mapping)}")
    return mapping


def _claim_actionability(claim_id: str, statement: str) -> str:
    text = f"{claim_id} {statement}".lower()
    if "epistas" in text or "combine-validated" in text:
        return "interpretation_or_future_combination"
    if "stability" in text or "expression" in text or "soluble" in text:
        return "requires_unavailable_stability_assay"
    if "hotspot" in text or "structural-core" in text or "local-geometry" in text:
        return "requires_structure_or_site_validation"
    if "library" in text or "reduced-alphabet" in text or "permissive" in text:
        return "library_design_not_current_candidate_score"
    return "generic_prior_not_candidate_score"


def _target_specific(statement: str) -> bool:
    return bool(
        re.search(
            r"\bGB1\b|\b(?:V39|D40|G41|V54)\b|position[- ]?(?:39|40|41|54)\b",
            statement,
            flags=re.IGNORECASE,
        )
    )


def _system_rag_excerpt(messages: list[dict[str, Any]]) -> str:
    system = "\n".join(
        str(item.get("content", ""))
        for item in messages
        if item.get("role") == "system"
    )
    patterns = (
        r"### 3\.3 RAG route.*?(?=### 3\.4 KG route)",
        r"RAG claims are canonical full atomic cards.*?(?=\n\n)",
    )
    excerpts = []
    for pattern in patterns:
        match = re.search(pattern, system, flags=re.DOTALL)
        if match:
            excerpts.append(match.group(0).strip())
    return "\n\n".join(excerpts)


def _prompt_claim_maps(prompt_payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    claims = {
        str(item.get("claim_id")): item
        for item in prompt_payload.get("rag_claims", [])
        if item.get("claim_id")
    }
    labels: dict[str, set[str]] = {}
    for claim_id, item in claims.items():
        for evidence_id in _short_evidence_ids(item.get("evidence_ids")):
            labels.setdefault(evidence_id, set()).add(claim_id)
    return claims, labels


def build_prompt_audit(runs: list[RunArtifact]) -> pd.DataFrame:
    """Summarize Scientist Prompt size, visible RAG claims, and citations."""

    rows: list[dict[str, Any]] = []
    for condition in (BASE_CONDITION, RAG_CONDITION):
        for fold, run in _run_map(runs, condition).items():
            for round_id in EXPECTED_ROUNDS:
                round_dir = run.path / f"round_{round_id:02d}"
                path, conversation, prompt, response = _conversation(round_dir)
                prompt_claims, label_to_claims = _prompt_claim_maps(prompt)
                rag_labels = set(label_to_claims)
                response_ids = set(_short_evidence_ids(response.get("evidence_ids")))
                cited_rag_ids = sorted(rag_labels & response_ids)
                mismatched_claims = sum(
                    "claim_text_mismatch_across_paths" in (item.get("warnings") or [])
                    for item in prompt_claims.values()
                )
                duplicate_labels = {
                    label: sorted(claim_ids)
                    for label, claim_ids in label_to_claims.items()
                    if len(claim_ids) > 1
                }
                universe_entries = _evidence_universe_entries(prompt)
                usage = conversation.get("usage") or {}
                preferred_residues = response.get("preferred_residues", {})
                preferred_combination_count = int(
                    np.prod(
                        [
                            max(len(preferred_residues.get(position, [])), 1)
                            for position in MUTABLE_POSITIONS
                        ]
                    )
                )
                rows.append(
                    {
                        "condition": condition,
                        "fold": fold,
                        "round_id": round_id,
                        "run_id": run.run_id,
                        "conversation_path": _repo_relative(path),
                        "prompt_chars": sum(
                            len(str(item.get("content", "")))
                            for item in conversation.get("messages", [])
                        ),
                        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                        "visible_observation_count": len(
                            (prompt.get("context") or {}).get("visible_observations", [])
                        ),
                        "prompt_evidence_count": len(prompt.get("evidence", [])),
                        "evidence_universe_count": len(universe_entries),
                        "prompt_rag_claim_count": len(prompt_claims),
                        "prompt_rag_short_label_count": len(rag_labels),
                        "rag_claims_with_mismatch_warning": mismatched_claims,
                        "ambiguous_rag_short_label_count": len(duplicate_labels),
                        "ambiguous_rag_short_labels_json": _json_dumps(duplicate_labels),
                        "response_evidence_count": len(response_ids),
                        "response_rag_citation_count": len(cited_rag_ids),
                        "response_rag_evidence_ids_json": _json_dumps(cited_rag_ids),
                        "response_used_rag": bool(cited_rag_ids),
                        "claim_modality": response.get("claim_modality"),
                        "preferred_combination_count": preferred_combination_count,
                        "preferred_residues_json": _json_dumps(
                            preferred_residues
                        ),
                        "response_statement": response.get("statement"),
                        "response_expected_outcome": response.get("expected_outcome"),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["condition", "fold", "round_id"]
    ).reset_index(drop=True)


def build_retrieval_claim_audit(runs: list[RunArtifact]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit retrieved claims and their transformed Prompt cards."""

    retrieval_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for fold, run in _run_map(runs, RAG_CONDITION).items():
        for round_id in EXPECTED_ROUNDS:
            round_dir = run.path / f"round_{round_id:02d}"
            retrieval_path = round_dir / "local_rag_retrieval.json"
            retrieval = read_json(retrieval_path)
            conversation_path, conversation, prompt, response = _conversation(round_dir)
            prompt_claims, label_to_claims = _prompt_claim_maps(prompt)
            response_ids = set(_short_evidence_ids(response.get("evidence_ids")))
            retrieved_claim_ids = {
                str(item.get("claim_id")) for item in retrieval.get("claims", [])
            }
            chunk_to_claim: dict[str, str] = {}
            chunk_to_type: dict[str, str] = {}
            for chunk in retrieval.get("chunks", []):
                chunk_id = str(chunk.get("chunk_id"))
                claim_id = str(
                    ((chunk.get("provenance") or {}).get("metadata") or {}).get(
                        "claim_id", ""
                    )
                )
                chunk_to_claim[chunk_id] = claim_id
                chunk_to_type[chunk_id] = str(chunk.get("knowledge_type", ""))
            for rank, claim in enumerate(retrieval.get("claims", []), start=1):
                claim_id = str(claim.get("claim_id"))
                prompt_claim = prompt_claims.get(claim_id, {})
                labels = _short_evidence_ids(prompt_claim.get("evidence_ids"))
                supports = claim.get("citation_support") or []
                statement = str(claim.get("statement", ""))
                retrieval_rows.append(
                    {
                        "fold": fold,
                        "round_id": round_id,
                        "run_id": run.run_id,
                        "retrieval_path": _repo_relative(retrieval_path),
                        "conversation_path": _repo_relative(conversation_path),
                        "query_id": retrieval.get("query_id"),
                        "original_query_hash": retrieval.get("original_query_hash"),
                        "sanitized_query": retrieval.get("sanitized_query"),
                        "retrieval_rank": rank,
                        "claim_id": claim_id,
                        "statement": statement,
                        "claim_kind": claim.get("claim_kind"),
                        "polarity": claim.get("polarity"),
                        "confidence": float(claim.get("confidence", np.nan)),
                        "selection_eligible": bool(claim.get("selection_eligible")),
                        "target_specific": _target_specific(statement),
                        "actionability_class": _claim_actionability(claim_id, statement),
                        "applicability_scope": (claim.get("applicability") or {}).get(
                            "scope"
                        ),
                        "applicability_limitation": (
                            claim.get("applicability") or {}
                        ).get("limitation"),
                        "citation_support_count": len(supports),
                        "verified_citation_support_count": sum(
                            bool(item.get("verified_against_source")) for item in supports
                        ),
                        "prompt_visible": bool(prompt_claim),
                        "prompt_short_evidence_ids_json": _json_dumps(labels),
                        "scientist_cited": bool(set(labels) & response_ids),
                    }
                )
            for prompt_rank, claim in enumerate(prompt.get("rag_claims", []), start=1):
                claim_id = str(claim.get("claim_id"))
                labels = _short_evidence_ids(claim.get("evidence_ids"))
                source_claim_ids = []
                mismatched_source_count = 0
                for source_ref in claim.get("source_refs", []) or []:
                    chunk_id = str(source_ref).removeprefix("source:local_rag:")
                    source_claim = chunk_to_claim.get(chunk_id)
                    if source_claim:
                        source_claim_ids.append(source_claim)
                        mismatched_source_count += source_claim != claim_id
                statement = str(claim.get("statement", ""))
                prompt_rows.append(
                    {
                        "fold": fold,
                        "round_id": round_id,
                        "run_id": run.run_id,
                        "prompt_rank": prompt_rank,
                        "claim_id": claim_id,
                        "statement": statement,
                        "retrieved_this_round": claim_id in retrieved_claim_ids,
                        "target_specific": _target_specific(statement),
                        "selection_eligible": bool(claim.get("selection_eligible")),
                        "contributes_to_selection": bool(
                            claim.get("contributes_to_selection")
                        ),
                        "quality_status": claim.get("quality_status"),
                        "confidence": float(claim.get("confidence", np.nan)),
                        "actionability_class": _claim_actionability(claim_id, statement),
                        "short_evidence_ids_json": _json_dumps(labels),
                        "scientist_cited": bool(set(labels) & response_ids),
                        "warning_count": len(claim.get("warnings") or []),
                        "claim_text_mismatch_warning": (
                            "claim_text_mismatch_across_paths"
                            in (claim.get("warnings") or [])
                        ),
                        "source_ref_count": len(claim.get("source_refs") or []),
                        "mapped_source_claim_ids_json": _json_dumps(
                            sorted(set(source_claim_ids))
                        ),
                        "mismatched_mapped_source_count": mismatched_source_count,
                        "ambiguous_short_label": any(
                            len(label_to_claims.get(label, set())) > 1 for label in labels
                        ),
                    }
                )
    return (
        pd.DataFrame(retrieval_rows).sort_values(
            ["fold", "round_id", "retrieval_rank"]
        ).reset_index(drop=True),
        pd.DataFrame(prompt_rows).sort_values(
            ["fold", "round_id", "prompt_rank"]
        ).reset_index(drop=True),
    )


def _preference_similarity(left_json: str, right_json: str) -> float:
    left = json.loads(left_json)
    right = json.loads(right_json)
    values = []
    for position in MUTABLE_POSITIONS:
        left_set = set(left.get(position, []))
        right_set = set(right.get(position, []))
        union = left_set | right_set
        values.append(len(left_set & right_set) / len(union) if union else 1.0)
    return float(np.mean(values))


def _arm_map(round_dir: Path) -> dict[str, str]:
    receipt = read_json(round_dir / "agent_quota_acquisition.json")
    mapping: dict[str, str] = {}
    for arm, ids in (receipt.get("selected_by_arm") or {}).items():
        for variant_id in ids or []:
            mapping[str(variant_id)] = str(arm)
    for variant_id in receipt.get("fallback_ids") or []:
        mapping.setdefault(str(variant_id), "fallback_fill")
    return mapping


def build_matched_impact(
    runs: list[RunArtifact], prompt_audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare base and RAG within fold/round and audit RAG-selected candidates."""

    candidates = build_candidate_table(runs)
    base_runs = _run_map(runs, BASE_CONDITION)
    rag_runs = _run_map(runs, RAG_CONDITION)
    impact_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    prompt_index = prompt_audit.set_index(["condition", "fold", "round_id"])
    for fold in sorted(base_runs):
        for round_id in EXPECTED_ROUNDS:
            base_run = base_runs[fold]
            rag_run = rag_runs[fold]
            base_dir = base_run.path / f"round_{round_id:02d}"
            rag_dir = rag_run.path / f"round_{round_id:02d}"
            base_pool_receipt = read_json(base_dir / "candidate_pool_receipt.json")
            rag_pool_receipt = read_json(rag_dir / "candidate_pool_receipt.json")
            base_pool = set(map(str, base_pool_receipt["candidate_ids"]))
            rag_pool = set(map(str, rag_pool_receipt["candidate_ids"]))
            base_group = candidates[
                (candidates["condition"] == BASE_CONDITION)
                & (candidates["fold"] == fold)
                & (candidates["round_id"] == round_id)
            ].copy()
            rag_group = candidates[
                (candidates["condition"] == RAG_CONDITION)
                & (candidates["fold"] == fold)
                & (candidates["round_id"] == round_id)
            ].copy()
            base_selected = set(base_group["variant_id"].astype(str))
            rag_selected = set(rag_group["variant_id"].astype(str))
            base_prompt = prompt_index.loc[(BASE_CONDITION, fold, round_id)]
            rag_prompt = prompt_index.loc[(RAG_CONDITION, fold, round_id)]
            base_statement = str(base_prompt["response_statement"])
            rag_statement = str(rag_prompt["response_statement"])
            base_arm = _arm_map(base_dir)
            rag_arm = _arm_map(rag_dir)
            for _, item in rag_group.iterrows():
                variant = str(item.get("variant", ""))
                preferred = json.loads(str(rag_prompt["preferred_residues_json"]))
                matches = sum(
                    len(variant) == 4 and variant[index] in set(preferred.get(position, []))
                    for index, position in enumerate(MUTABLE_POSITIONS)
                )
                row = item.to_dict()
                row.update(
                    {
                        "selection_arm": rag_arm.get(str(item["variant_id"]), "unknown"),
                        "preferred_position_matches": matches,
                        "all_positions_match": matches == 4,
                        "also_selected_by_base": str(item["variant_id"]) in base_selected,
                        "rag_claim_citation_count": int(
                            rag_prompt["response_rag_citation_count"]
                        ),
                    }
                )
                candidate_rows.append(row)
            base_core = base_group[
                base_group["variant_id"].astype(str).map(base_arm).eq("hypothesis_target")
            ]
            rag_core = rag_group[
                rag_group["variant_id"].astype(str).map(rag_arm).eq("hypothesis_target")
            ]
            pool_union = base_pool | rag_pool
            selected_union = base_selected | rag_selected
            impact_rows.append(
                {
                    "fold": fold,
                    "round_id": round_id,
                    "candidate_pool_overlap_count": len(base_pool & rag_pool),
                    "candidate_pool_jaccard": (
                        len(base_pool & rag_pool) / len(pool_union) if pool_union else 1.0
                    ),
                    "selected_overlap_count": len(base_selected & rag_selected),
                    "same_pool_seed": base_pool_receipt.get("seed")
                    == rag_pool_receipt.get("seed"),
                    "same_sampling_namespace": base_pool_receipt.get(
                        "sampling_namespace"
                    )
                    == rag_pool_receipt.get("sampling_namespace"),
                    "same_sampling_strategy": base_pool_receipt.get(
                        "sampling_strategy"
                    )
                    == rag_pool_receipt.get("sampling_strategy"),
                    "selected_jaccard": (
                        len(base_selected & rag_selected) / len(selected_union)
                        if selected_union
                        else 1.0
                    ),
                    "base_batch_mean": float(base_group["wet_fitness"].mean()),
                    "rag_batch_mean": float(rag_group["wet_fitness"].mean()),
                    "rag_minus_base_batch_mean": float(
                        rag_group["wet_fitness"].mean()
                        - base_group["wet_fitness"].mean()
                    ),
                    "base_batch_median": float(base_group["wet_fitness"].median()),
                    "rag_batch_median": float(rag_group["wet_fitness"].median()),
                    "rag_minus_base_batch_median": float(
                        rag_group["wet_fitness"].median()
                        - base_group["wet_fitness"].median()
                    ),
                    "base_batch_best": float(base_group["wet_fitness"].max()),
                    "rag_batch_best": float(rag_group["wet_fitness"].max()),
                    "rag_minus_base_batch_best": float(
                        rag_group["wet_fitness"].max()
                        - base_group["wet_fitness"].max()
                    ),
                    "base_unique_selected_wet_mean": float(
                        base_group[
                            ~base_group["variant_id"].astype(str).isin(rag_selected)
                        ]["wet_fitness"].mean()
                    ),
                    "rag_unique_selected_wet_mean": float(
                        rag_group[
                            ~rag_group["variant_id"].astype(str).isin(base_selected)
                        ]["wet_fitness"].mean()
                    ),
                    "base_hypothesis_arm_wet_mean": float(
                        base_core["wet_fitness"].mean()
                    ),
                    "rag_hypothesis_arm_wet_mean": float(
                        rag_core["wet_fitness"].mean()
                    ),
                    "base_prompt_tokens": int(base_prompt["prompt_tokens"]),
                    "rag_prompt_tokens": int(rag_prompt["prompt_tokens"]),
                    "rag_prompt_token_increment": int(
                        rag_prompt["prompt_tokens"] - base_prompt["prompt_tokens"]
                    ),
                    "preferred_residue_similarity": _preference_similarity(
                        str(base_prompt["preferred_residues_json"]),
                        str(rag_prompt["preferred_residues_json"]),
                    ),
                    "statement_similarity": difflib.SequenceMatcher(
                        None, base_statement, rag_statement
                    ).ratio(),
                    "rag_claim_citation_count": int(
                        rag_prompt["response_rag_citation_count"]
                    ),
                    "rag_claims_with_mismatch_warning": int(
                        rag_prompt["rag_claims_with_mismatch_warning"]
                    ),
                }
            )
    return (
        pd.DataFrame(impact_rows).sort_values(["fold", "round_id"]).reset_index(
            drop=True
        ),
        pd.DataFrame(candidate_rows).sort_values(
            ["fold", "round_id", "selection_order"]
        ).reset_index(drop=True),
    )


def build_prompt_cases(
    runs: list[RunArtifact], matched_impact: pd.DataFrame
) -> list[dict[str, Any]]:
    """Extract exact RAG/Prompt fields for the strongest positive and negative rounds."""

    cases = []
    selections = {
        "largest_negative_batch_delta": matched_impact.sort_values(
            ["rag_minus_base_batch_mean", "fold", "round_id"]
        ).iloc[0],
        "largest_positive_batch_delta": matched_impact.sort_values(
            ["rag_minus_base_batch_mean", "fold", "round_id"], ascending=[False, True, True]
        ).iloc[0],
    }
    base_runs = _run_map(runs, BASE_CONDITION)
    rag_runs = _run_map(runs, RAG_CONDITION)
    for case_id, metric_row in selections.items():
        fold = int(metric_row["fold"])
        round_id = int(metric_row["round_id"])
        base_dir = base_runs[fold].path / f"round_{round_id:02d}"
        rag_dir = rag_runs[fold].path / f"round_{round_id:02d}"
        rag_retrieval = read_json(rag_dir / "local_rag_retrieval.json")
        rag_path, rag_conversation, rag_prompt, rag_response = _conversation(rag_dir)
        base_path, _, _, base_response = _conversation(base_dir)
        _, rag_label_to_claims = _prompt_claim_maps(rag_prompt)
        rag_response_ids = set(_short_evidence_ids(rag_response.get("evidence_ids")))
        cases.append(
            {
                "case_id": case_id,
                "fold": fold,
                "round_id": round_id,
                "metric_comparison": {
                    key: (
                        int(value)
                        if key in {"fold", "round_id"}
                        else float(value)
                        if isinstance(value, (np.integer, np.floating))
                        else value
                    )
                    for key, value in metric_row.to_dict().items()
                },
                "rag_retrieval_path": _repo_relative(
                    rag_dir / "local_rag_retrieval.json"
                ),
                "rag_conversation_path": _repo_relative(rag_path),
                "base_conversation_path": _repo_relative(base_path),
                "retrieval_query": rag_retrieval.get("sanitized_query"),
                "retrieved_claims": [
                    {
                        "claim_id": item.get("claim_id"),
                        "statement": item.get("statement"),
                        "confidence": item.get("confidence"),
                        "selection_eligible": item.get("selection_eligible"),
                        "applicability": item.get("applicability"),
                    }
                    for item in rag_retrieval.get("claims", [])
                ],
                "llm_input_prompt_excerpt": {
                    "system_rag_contract": _system_rag_excerpt(
                        rag_conversation.get("messages", [])
                    ),
                    "activation_state": (
                        rag_prompt.get("context") or {}
                    ).get("activation_state"),
                    "rag_claims": rag_prompt.get("rag_claims", []),
                    "evidence_universe_rag_entries": [
                        item
                        for item in _evidence_universe_entries(rag_prompt)
                        if "kg_pack:query_local_knowledge" in (item.get("origins") or [])
                    ],
                },
                "llm_output_comparison": {
                    "kg_base": base_response,
                    "kg_base_rag": rag_response,
                },
                "rag_cited_evidence_ids": sorted(
                    set(rag_label_to_claims) & rag_response_ids
                ),
                "reasoning_content_policy": (
                    "The conversation artifact contains provider reasoning_content, "
                    "but this derived case exports only model-visible input and final output."
                ),
            }
        )
    return cases
