#!/usr/bin/env python3
"""Validate a GB1 directive research-candidate bundle without opening source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REQUIRED_FRONTMATTER: dict[str, Any] = {
    "schema_version": "gb1-directive-card:v1",
    "record_type": "knowledge_decision_card",
    "language": "en",
    "status": "research_candidate",
    "human_reviewed": False,
    "selection_eligible": False,
    "permission": "explanation_only",
    "benchmark_overlap": "none",
    "retrieval_similarity": "runtime_only",
}
REQUIRED_LISTS = ("inputs", "boundaries", "source_spans", "logic_units")
REQUIRED_SCALARS = (
    "card_id",
    "title",
    "feature",
    "direction",
    "uncertainty",
    "scientific_credibility",
    "task_applicability",
    "knowledge_type",
)
REQUIRED_SECTIONS = (
    "Use when",
    "Decision rule",
    "Candidate action",
    "Matched comparison",
    "Abstain or downgrade when",
    "Evidence basis",
)
QUARANTINED_MARKERS = (
    "10.7554/elife.16965",
    "10.1016/j.cub.2014.09.072",
    "10.1073/pnas.1901979116",
    "proteingym",
    "huggingface",
    "hugging face",
    "flip benchmark",
)
OUT_OF_SCOPE_TERMS = (
    "viral protein",
    "virus protein",
    "capsid",
    "virion",
    "bacteriophage",
)
INSTRUCTION_MARKERS = (
    "ignore previous instructions",
    "system prompt",
    "developer message",
    "jailbreak",
)
EXACT_SUBSTITUTION = re.compile(r"\b(?:V39|D40|G41|V54)[ACDEFGHIKLMNPQRSTVWY]\b", re.IGNORECASE)
NUMERIC_FITNESS = re.compile(r"\b(?:fitness|score|label)\s*(?:=|:|is)\s*-?\d+(?:\.\d+)?\b", re.IGNORECASE)
CONTROLLED_FACETS = {
    "record_type",
    "knowledge_type",
    "question_leaf_id",
    "decision_slot",
    "task_route",
    "feature_channel",
    "required_input",
    "permission",
    "expected_direction",
    "stage",
    "evidence_role",
}
NATIVE_RECORD_TYPES = {
    "atomic_claim",
    "logic_unit",
    "knowledge_decision_card",
}
CONTROLLED_FEATURE_CHANNELS = {"physchem", "conservation", "structure"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_card(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("missing YAML frontmatter")
    try:
        raw_frontmatter, body = text[4:].split("\n---\n", 1)
    except ValueError as exc:
        raise ValueError("unterminated YAML frontmatter") from exc
    metadata = yaml.safe_load(raw_frontmatter)
    if not isinstance(metadata, dict):
        raise TypeError("frontmatter must be a mapping")
    return metadata, body


def validate_card(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        metadata, body = parse_card(path)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        return [f"{path.name}: {exc}"]

    for key, expected in REQUIRED_FRONTMATTER.items():
        if metadata.get(key) != expected:
            errors.append(f"{path.name}: {key!r} must equal {expected!r}")
    for key in REQUIRED_SCALARS:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            errors.append(f"{path.name}: {key!r} must be a non-empty string")
    if not isinstance(metadata.get("inputs"), dict) or not metadata["inputs"]:
        errors.append(f"{path.name}: 'inputs' must be a non-empty mapping")
    if not isinstance(metadata.get("topics"), list) or not metadata["topics"]:
        errors.append(f"{path.name}: 'topics' must be a non-empty list")
    for key in REQUIRED_LISTS[1:]:
        if not isinstance(metadata.get(key), list) or not metadata[key]:
            errors.append(f"{path.name}: {key!r} must be a non-empty list")

    for section in REQUIRED_SECTIONS:
        if f"## {section}" not in body:
            errors.append(f"{path.name}: missing section '## {section}'")
    word_count = len(re.findall(r"\b[\w'-]+\b", body))
    if not 120 <= word_count <= 260:
        errors.append(f"{path.name}: body has {word_count} words; expected 120-260")

    lowered = path.read_text(encoding="utf-8").lower()
    for marker in QUARANTINED_MARKERS:
        if marker in lowered:
            errors.append(f"{path.name}: contains quarantined-source marker")
    for term in OUT_OF_SCOPE_TERMS:
        if term in lowered:
            errors.append(f"{path.name}: contains out-of-scope source vocabulary")
    for marker in INSTRUCTION_MARKERS:
        if marker in lowered:
            errors.append(f"{path.name}: contains instruction-like source text")
    if EXACT_SUBSTITUTION.search(body):
        errors.append(f"{path.name}: contains an exact benchmark-site substitution")
    if NUMERIC_FITNESS.search(body):
        errors.append(f"{path.name}: contains a numeric fitness label")
    return errors


def validate_native_record(path: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    try:
        metadata, body = parse_card(path)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        return [f"{path.name}: {exc}"], {}
    record_type = str(metadata.get("record_type", ""))
    if record_type not in NATIVE_RECORD_TYPES:
        errors.append(f"{path.name}: unsupported native record_type {record_type!r}")
    for key, expected in {
        "language": "en",
        "status": "research_candidate",
        "human_reviewed": False,
        "selection_eligible": False,
        "permission": "explanation_only",
    }.items():
        if metadata.get(key) != expected:
            errors.append(f"{path.name}: {key!r} must equal {expected!r}")
    for key in ("record_id", "knowledge_type", "retrieval_text"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            errors.append(f"{path.name}: {key!r} must be a non-empty string")
    if str(metadata.get("retrieval_text", "")).strip() != body.strip():
        errors.append(f"{path.name}: Markdown body must equal retrieval_text")
    facets = metadata.get("facets") or {
        name: metadata[name] for name in CONTROLLED_FACETS if name in metadata
    }
    if not isinstance(facets, dict) or not facets:
        errors.append(f"{path.name}: facets must be a non-empty mapping")
        facets = {}
    facets = {
        str(name): (
            list(values) if isinstance(values, (list, tuple)) else [str(values)]
        )
        for name, values in facets.items()
    }
    metadata["facets"] = facets
    unknown_facets = set(facets).difference(CONTROLLED_FACETS)
    if unknown_facets:
        errors.append(f"{path.name}: unknown facets {sorted(unknown_facets)}")
    for name, values in facets.items():
        if not values:
            errors.append(f"{path.name}: facet {name!r} must be a non-empty list")
    channels = {str(item) for item in facets.get("feature_channel", ())}
    if channels.difference(CONTROLLED_FEATURE_CHANNELS):
        errors.append(f"{path.name}: contains an unknown feature channel")
    if record_type in {"logic_unit", "knowledge_decision_card"}:
        for key in ("boundary_conditions", "abstain_if"):
            if not isinstance(metadata.get(key), list) or not metadata[key]:
                errors.append(f"{path.name}: {key!r} must be a non-empty list")
        if not facets.get("feature_channel"):
            errors.append(f"{path.name}: feature_channel routing is required")
        if not facets.get("required_input"):
            errors.append(f"{path.name}: required_input routing is required")
    if record_type == "logic_unit":
        if not isinstance(metadata.get("falsifiers"), list) or not metadata["falsifiers"]:
            errors.append(f"{path.name}: LogicUnit falsifiers are required")
        if not isinstance(metadata.get("feature_focus"), list) or not metadata["feature_focus"]:
            errors.append(f"{path.name}: LogicUnit feature_focus mapping is required")
    if record_type == "knowledge_decision_card" and (
        not isinstance(metadata.get("required_inputs"), list)
        or not metadata["required_inputs"]
    ):
        errors.append(f"{path.name}: DecisionCard required_inputs are required")

    lowered = path.read_text(encoding="utf-8").lower()
    if "openalex" in lowered:
        errors.append(f"{path.name}: OpenAlex is prohibited for this bundle")
    for term in OUT_OF_SCOPE_TERMS:
        if term in lowered:
            errors.append(f"{path.name}: contains out-of-scope source vocabulary")
    if EXACT_SUBSTITUTION.search(body):
        errors.append(f"{path.name}: contains an exact benchmark-site substitution")
    if NUMERIC_FITNESS.search(body):
        errors.append(f"{path.name}: contains a numeric fitness label")
    return errors, metadata


def validate_bundle_text(root: Path) -> list[str]:
    """Apply source-scope and benchmark-leakage checks to every bundle artifact."""

    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".yaml", ".yml", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(root)}: cannot read text: {exc}")
            continue
        lowered = text.lower()
        if "openalex" in lowered:
            errors.append(f"{path.relative_to(root)}: prohibited metadata provider reference")
        for marker in QUARANTINED_MARKERS:
            if marker in lowered:
                errors.append(f"{path.relative_to(root)}: contains quarantined-source marker")
        for term in OUT_OF_SCOPE_TERMS:
            if term in lowered:
                errors.append(f"{path.relative_to(root)}: contains out-of-scope source vocabulary")
        if EXACT_SUBSTITUTION.search(text):
            errors.append(f"{path.relative_to(root)}: contains an exact benchmark-site substitution")
        if NUMERIC_FITNESS.search(text):
            errors.append(f"{path.relative_to(root)}: contains a numeric fitness label")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    root = args.bundle_root.resolve()
    cards_dir = root / "cards"
    cards = sorted(cards_dir.glob("*.md")) if cards_dir.is_dir() else []
    records_dir = root / "records"
    records = sorted(records_dir.rglob("*.md")) if records_dir.is_dir() else []
    errors: list[str] = []
    errors.extend(validate_bundle_text(root))
    if not cards and not records:
        errors.append("bundle contains neither cards/*.md nor records/**/*.md files")
    for card in cards:
        errors.extend(validate_card(card))
    native_metadata: list[dict[str, Any]] = []
    for record in records:
        record_errors, metadata = validate_native_record(record)
        errors.extend(record_errors)
        if metadata:
            native_metadata.append(metadata)
    if records:
        present_types = {str(item.get("record_type")) for item in native_metadata}
        missing_types = NATIVE_RECORD_TYPES.difference(present_types)
        if missing_types:
            errors.append(f"native bundle is missing record types: {sorted(missing_types)}")
        evidence_roles = {
            str(role)
            for item in native_metadata
            for role in (item.get("facets", {}).get("evidence_role", ()))
        }
        if "support" not in evidence_roles:
            errors.append("native bundle lacks support evidence_role coverage")
        if not evidence_roles.intersection({"counterevidence", "boundary"}):
            errors.append("native bundle lacks counterevidence/boundary coverage")

    audit_dir = root / "audit"
    for relative in (
        "research-brief.yaml",
        "search-runs.yaml",
        "source-ledger.yaml",
        "gap-matrix.yaml",
        "decision-log.md",
    ):
        if not (audit_dir / relative).is_file():
            errors.append(f"missing audit/{relative}")

    receipt_path = (args.receipt or root / "validation-receipt.json").resolve()
    receipt = {
        "schema_version": "gb1-directive-validation:v1",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(root),
        "status": "passed" if not errors else "failed",
        "research_status": "research_candidate",
        "human_reviewed": False,
        "selection_eligible": False,
        "permission": "explanation_only",
        "cards": [
            {"path": str(card.relative_to(root)), "sha256": sha256(card)} for card in cards
        ],
        "native_records": [
            {"path": str(record.relative_to(root)), "sha256": sha256(record)}
            for record in records
        ],
        "errors": errors,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
