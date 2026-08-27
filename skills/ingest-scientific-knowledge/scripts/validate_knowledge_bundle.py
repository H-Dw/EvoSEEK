from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

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
NATIVE_TYPES = {"atomic_claim", "logic_unit", "knowledge_decision_card"}


def _front_matter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"missing YAML front matter: {path.name}")
    raw, _body = text[4:].split("\n---\n", 1)
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"front matter must be a mapping: {path.name}")
    return {str(key): value for key, value in payload.items()}


def _native_report(bundle_root: Path, embedding_model: Path | None) -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository_root / "src"))
    from fitness_agents.deep_research.legacy_validator import model_token_counter
    from fitness_agents.local_knowledge.runtime_manifest import load_runtime_file_manifest
    from fitness_agents.safety import discover_workspace_access_policy

    errors: list[str] = []
    warnings: list[str] = []
    root = bundle_root.resolve()
    access_policy = discover_workspace_access_policy(root)
    try:
        manifest = load_runtime_file_manifest(root, access_policy=access_policy)
    except Exception as error:  # noqa: BLE001 - validator converts failures to a report
        return {
            "schema_version": "native-runtime-bundle-validation:v1",
            "valid": False,
            "errors": [f"{type(error).__name__}: {error}"],
            "warnings": [],
        }
    if manifest is None:
        return {
            "schema_version": "native-runtime-bundle-validation:v1",
            "valid": False,
            "errors": ["missing runtime-files.json"],
            "warnings": [],
        }
    type_counts: Counter[str] = Counter()
    facet_counts: Counter[str] = Counter()
    knowledge_types: Counter[str] = Counter()
    evidence_roles: set[str] = set()
    token_counter, model_limit = model_token_counter(embedding_model)
    token_counts: list[int] = []
    for entry in manifest.entries:
        if entry.record_type not in NATIVE_TYPES:
            continue
        type_counts[entry.record_type] += 1
        path = root / entry.relative_path
        try:
            metadata = _front_matter(path)
        except (OSError, UnicodeError, ValueError, TypeError, yaml.YAMLError) as error:
            errors.append(str(error))
            continue
        if metadata.get("record_type") != entry.record_type:
            errors.append(f"record type mismatch: {entry.relative_path}")
        if metadata.get("selection_eligible") is not False:
            errors.append(f"selection_eligible must remain false: {entry.relative_path}")
        if str(metadata.get("source_release_id", "")) != manifest.source_release_id:
            errors.append(f"source release mismatch: {entry.relative_path}")
        if not str(metadata.get("permission", "")):
            errors.append(f"permission is required: {entry.relative_path}")
        knowledge_type = str(metadata.get("knowledge_type", ""))
        if knowledge_type:
            knowledge_types[knowledge_type] += 1
        present_facets = {
            name: metadata[name] for name in CONTROLLED_FACETS if name in metadata
        }
        for name, raw_values in present_facets.items():
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            for value in values:
                facet_counts[f"{name}={value}"] += 1
            if name == "evidence_role":
                evidence_roles.update(str(value) for value in values)
        if entry.record_type in {"logic_unit", "knowledge_decision_card"}:
            if not metadata.get("boundary_conditions") or not metadata.get("abstain_if"):
                errors.append(f"boundary/abstention is required: {entry.relative_path}")
            if not metadata.get("feature_channel") or not metadata.get("required_input"):
                errors.append(f"feature routing is required: {entry.relative_path}")
        if entry.record_type == "knowledge_decision_card":
            payload = metadata.get("record_payload")
            if not isinstance(payload, dict) or not payload.get("required_inputs"):
                errors.append(f"DecisionCard required inputs are missing: {entry.relative_path}")
        if token_counter is not None:
            retrieval_text = str(metadata.get("retrieval_text", ""))
            count = int(token_counter(retrieval_text))
            token_counts.append(count)
            if model_limit is not None and count > model_limit:
                errors.append(f"record exceeds embedding token limit: {entry.relative_path}")
    missing_types = NATIVE_TYPES.difference(type_counts)
    if missing_types:
        errors.append(f"missing native record types: {sorted(missing_types)}")
    if "support" not in evidence_roles:
        errors.append("native runtime lacks support evidence_role coverage")
    if not evidence_roles.intersection({"counterevidence", "boundary"}):
        errors.append("native runtime lacks counterevidence/boundary coverage")
    return {
        "schema_version": "native-runtime-bundle-validation:v1",
        "valid": not errors,
        "source_release_id": manifest.source_release_id,
        "runtime_manifest_sha256": manifest.manifest_sha256,
        "record_type_counts": dict(sorted(type_counts.items())),
        "facet_counts": dict(sorted(facet_counts.items())),
        "knowledge_types": dict(sorted(knowledge_types.items())),
        "model_token_check": {
            "performed": token_counter is not None,
            "model_path": str(embedding_model.resolve()) if embedding_model else None,
            "model_max_tokens": model_limit,
            "maximum_record_tokens": max(token_counts, default=None),
        },
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate native v2 or legacy v1 manifest-driven knowledge runtimes"
    )
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--embedding-model", type=Path)
    args = parser.parse_args()
    manifest_path = args.bundle_root / "runtime-files.json"
    try:
        schema = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "schema_version"
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        schema = None
    repository_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository_root / "src"))
    if schema == "local-rag-runtime-files:v2":
        report = _native_report(args.bundle_root, args.embedding_model)
    else:
        from fitness_agents.deep_research.legacy_validator import (
            validate_legacy_runtime_bundle,
        )

        report = validate_legacy_runtime_bundle(
            args.bundle_root,
            embedding_model=args.embedding_model,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
