from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from fitness_agents.safety import discover_workspace_access_policy

from .canonical import stable_id
from .contracts import EvidenceProductBundle
from .pipeline import validate_evidence_product
from .policy import ExternalEvidenceScopePolicy


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    prefix = normalized[:60] or "claim"
    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"


def _scientific_confidence(bundle: EvidenceProductBundle, claim_id: str) -> float:
    qualities = [
        logic.scientific_quality
        for logic in bundle.logic_units
        if claim_id in logic.premise_claim_ids
    ]
    if not qualities:
        return 0.0
    values = [
        quality.source_credibility * (1.0 - quality.uncertainty)
        for quality in qualities
    ]
    return round(min(values), 6)


def _selection_eligible(bundle: EvidenceProductBundle, claim_id: str) -> bool:
    del bundle, claim_id
    # The v1 runtime format cannot preserve calibration/benchmark/card closure.
    # Never downgrade a canonical permission into a claim-level selection boolean.
    return False


def export_legacy_local_rag_bundle(
    bundle: EvidenceProductBundle,
    output_root: str | Path,
    *,
    active_policy: ExternalEvidenceScopePolicy,
    trusted_reviewer_keys: Mapping[str, tuple[str, bytes]],
    trusted_release_approval_keys: Mapping[str, tuple[str, bytes]],
) -> dict[str, Any]:
    """Stage, validate, and atomically publish a legacy compatibility view."""

    report = validate_evidence_product(
        bundle,
        active_policy=active_policy,
        trusted_reviewer_keys=trusted_reviewer_keys,
        trusted_release_approval_keys=trusted_release_approval_keys,
    )
    if not report.release_ready:
        raise ValueError("Only a valid released evidence product can be exported")
    target_root = Path(output_root).absolute()
    access_policy = discover_workspace_access_policy(target_root)
    access_policy.require_allowed(target_root)
    if target_root.exists():
        raise FileExistsError(f"Refusing to overwrite export path: {target_root}")
    target_parent = target_root.parent
    access_policy.require_allowed(target_parent)
    target_parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{target_root.name or 'runtime'}.building-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=target_parent) as staging:
        staging_root = Path(staging)
        receipt = _write_legacy_local_rag_bundle(
            bundle,
            staging_root,
            active_policy=active_policy,
            trusted_reviewer_keys=trusted_reviewer_keys,
            trusted_release_approval_keys=trusted_release_approval_keys,
        )
        from .legacy_validator import validate_legacy_runtime_bundle

        validation = validate_legacy_runtime_bundle(
            staging_root,
            active_policy=active_policy,
            trusted_reviewer_keys=trusted_reviewer_keys,
            trusted_release_approval_keys=trusted_release_approval_keys,
        )
        if not validation["valid"]:
            raise ValueError(
                "Staged legacy export failed self-validation: "
                + "; ".join(validation["errors"])
            )
        staging_root.replace(target_root)
        return receipt


def _write_legacy_local_rag_bundle(
    bundle: EvidenceProductBundle,
    output_root: str | Path,
    *,
    active_policy: ExternalEvidenceScopePolicy,
    trusted_reviewer_keys: Mapping[str, tuple[str, bytes]],
    trusted_release_approval_keys: Mapping[str, tuple[str, bytes]],
) -> dict[str, Any]:
    report = validate_evidence_product(
        bundle,
        active_policy=active_policy,
        trusted_reviewer_keys=trusted_reviewer_keys,
        trusted_release_approval_keys=trusted_release_approval_keys,
    )
    if not report.release_ready:
        raise ValueError("Only a valid released evidence product can be exported")
    root = Path(output_root)
    access_policy = discover_workspace_access_policy(root)
    access_policy.require_allowed(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty export directory: {root}")

    publications_by_id = {item.publication_id: item for item in bundle.publications}
    spans_by_id = {item.source_span_id: item for item in bundle.source_spans}
    groups_by_id = {item.evidence_group_id: item for item in bundle.evidence_groups}
    release_records = {
        item.record_id: item for item in bundle.release_manifest.records
    }
    export_claims = tuple(
        claim for claim in bundle.atomic_claims if claim.claim_status == "supported"
    )
    if not export_claims:
        raise ValueError("Legacy export requires at least one supported AtomicClaim")
    planned_targets: dict[str, Path] = {}
    for claim in export_claims:
        release_record = release_records.get(claim.claim_id)
        if release_record is None or release_record.record_type != "atomic_claim":
            raise ValueError(
                f"AtomicClaim is absent from the release manifest: {claim.claim_id}"
            )
        target = root / "claims" / claim.knowledge_type / f"{_slug(claim.claim_id)}.md"
        if not target.absolute().is_relative_to(root.absolute()):
            raise ValueError("Legacy export target escapes the output root")
        target_key = str(target.absolute()).casefold()
        if target_key in planned_targets:
            raise ValueError("Legacy export claim paths collide")
        planned_targets[target_key] = target
        for group_id in claim.evidence_group_ids:
            group = groups_by_id[group_id]
            if group.stance != "supports":
                continue
            for span_id in group.source_span_ids:
                publication = publications_by_id[spans_by_id[span_id].publication_id]
                if not publication.doi:
                    raise ValueError(
                        f"Legacy export requires a DOI publication: {publication.publication_id}"
                    )

    claims_root = root / "claims"
    catalog_root = root / "catalog"
    claims_root.mkdir(parents=True, exist_ok=True)
    catalog_root.mkdir(parents=True, exist_ok=True)
    publication_ids_used: set[str] = set()
    written: list[dict[str, Any]] = []

    for claim in sorted(export_claims, key=lambda item: item.claim_id):
        supports: list[dict[str, Any]] = []
        for group_id in claim.evidence_group_ids:
            group = groups_by_id[group_id]
            if group.stance != "supports":
                continue
            support_type = {
                "supports": "direct_support",
                "refutes": "limiting",
                "limits": "limiting",
                "unknown": "background_support",
            }[group.stance]
            for span_id in group.source_span_ids:
                span = spans_by_id[span_id]
                publication = publications_by_id[span.publication_id]
                publication_ids_used.add(publication.publication_id)
                supports.append(
                    {
                        "support_id": stable_id(
                            "citation", claim.claim_id, span.source_span_id
                        ),
                        "publication_id": publication.publication_id,
                        "support_type": support_type,
                        "locator": span.locator,
                        "verified_against_source": True,
                    }
                )
        metadata = {
            "schema_version": "scientific-atomic-claim:v1",
            "record_type": "atomic_claim",
            "claim_id": claim.claim_id,
            "title": claim.statement[:100],
            "language": "en",
            "knowledge_type": claim.knowledge_type,
            "statement": claim.statement,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "object": claim.object,
            "polarity": "support",
            "claim_kind": claim.claim_kind,
            "confidence": _scientific_confidence(bundle, claim.claim_id),
            "applicability": claim.applicability,
            "citation_support": supports,
            "selection_eligible": _selection_eligible(bundle, claim.claim_id),
            "source_release_id": bundle.release_manifest.release_id,
            "source_record_hash": next(
                record.content_sha256
                for record in bundle.release_manifest.records
                if record.record_id == claim.claim_id
            ),
        }
        target = claims_root / claim.knowledge_type / f"{_slug(claim.claim_id)}.md"
        access_policy.require_allowed(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        front_matter = yaml.safe_dump(
            metadata,
            allow_unicode=True,
            sort_keys=False,
        ).strip()
        target.write_bytes(
            f"---\n{front_matter}\n---\n{claim.statement}\n".encode()
        )
        written.append(_file_receipt(root, target, "atomic_claim", claim.claim_id))

    catalog = {
        "schema_version": "scientific-publications:v1",
        "generated_from": bundle.release_manifest.release_id,
        "verified_on": bundle.release_manifest.created_at.date().isoformat(),
        "publications": [
            {
                "publication_id": publication.publication_id,
                "title": publication.title,
                "authors": list(publication.authors),
                "year": publication.year,
                "venue": publication.venue,
                "doi": publication.doi,
                "url": publication.canonical_url,
                "publication_type": publication.publication_type,
                "verification": {
                    "metadata_source": "deep-research-evidence-product",
                    "metadata_verified": publication.metadata_verified,
                    "full_text_verified": publication.full_text_status == "verified",
                    "source_release_id": bundle.release_manifest.release_id,
                },
            }
            for publication in sorted(
                (
                    publications_by_id[publication_id]
                    for publication_id in publication_ids_used
                ),
                key=lambda item: item.publication_id,
            )
        ],
    }
    catalog_path = catalog_root / "publications.yaml"
    access_policy.require_allowed(catalog_path)
    catalog_path.write_bytes(
        yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False).encode("utf-8")
    )
    written.append(_file_receipt(root, catalog_path, "publication_catalog", "catalog"))

    release_path = root / "evidence-release.json"
    access_policy.require_allowed(release_path)
    release_path.write_bytes(
        (
            json.dumps(
                bundle.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    written.append(
        _file_receipt(
            root,
            release_path,
            "evidence_release",
            bundle.release_manifest.release_id,
        )
    )
    runtime_manifest = {
        "schema_version": "local-rag-runtime-files:v1",
        "source_release_id": bundle.release_manifest.release_id,
        "policy_hash": bundle.release_manifest.policy_hash,
        "workspace_access_policy_hash": access_policy.policy_hash,
        "files": sorted(written, key=lambda item: item["relative_path"]),
    }
    runtime_manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            runtime_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = root / "runtime-files.json"
    access_policy.require_allowed(manifest_path)
    manifest_path.write_text(
        json.dumps(runtime_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return runtime_manifest


def _native_record_documents(
    bundle: EvidenceProductBundle,
) -> tuple[tuple[str, str, str, bytes], ...]:
    """Render deterministic native AtomicClaim/LogicUnit/DecisionCard documents."""

    if bundle.release_manifest is None:
        raise ValueError("Native projection requires a ReleaseManifest")
    release_records = {item.record_id: item for item in bundle.release_manifest.records}
    questions = {
        item.question_leaf_id: item for item in bundle.research_brief.question_tree
    }
    claims = {item.claim_id: item for item in bundle.atomic_claims}
    logics = {item.logic_unit_id: item for item in bundle.logic_units}
    cards_by_logic: dict[str, list[Any]] = {}
    for card in bundle.decision_cards:
        for logic_id in card.logic_unit_ids:
            cards_by_logic.setdefault(logic_id, []).append(card)
    logics_by_claim: dict[str, list[Any]] = {}
    for logic in bundle.logic_units:
        for claim_id in (*logic.premise_claim_ids, *logic.counterclaim_ids):
            logics_by_claim.setdefault(claim_id, []).append(logic)

    def record_hash(record_id: str, record_type: str) -> str:
        record = release_records.get(record_id)
        if record is None or record.record_type != record_type:
            raise ValueError(
                f"Native runtime record is absent from release manifest: {record_type}/{record_id}"
            )
        return record.content_sha256

    def unique(values: Any) -> list[str]:
        return list(dict.fromkeys(str(item) for item in values if str(item)))

    def feature_channels(required_inputs: Any, candidate_feature: Any = None) -> list[str]:
        text = " ".join(
            [*(str(item) for item in required_inputs), str(candidate_feature or "")]
        ).casefold()
        markers = {
            "physchem": ("physchem", "chemical", "charge", "hydropathy", "descriptor"),
            "conservation": ("evolution", "msa", "conservation", "log_odds"),
            "structure": ("structure", "contact", "interface", "solvent", "backbone"),
        }
        return [
            channel
            for channel, channel_markers in markers.items()
            if any(marker in text for marker in channel_markers)
        ]

    documents: list[tuple[str, str, str, bytes]] = []

    def add_document(
        *,
        record_type: str,
        record_id: str,
        knowledge_type: str,
        retrieval_text: str,
        metadata: dict[str, Any],
    ) -> None:
        full = {
            "schema_version": {
                "atomic_claim": "scientific-atomic-claim-runtime:v2",
                "logic_unit": "scientific-logic-unit-runtime:v1",
                "knowledge_decision_card": "knowledge-decision-card-runtime:v1",
            }[record_type],
            "record_type": record_type,
            "record_id": record_id,
            "knowledge_type": knowledge_type,
            "retrieval_text": retrieval_text,
            "language": "en",
            "selection_eligible": False,
            "source_release_id": bundle.release_manifest.release_id,
            "source_record_hash": record_hash(record_id, record_type),
            **metadata,
        }
        if record_type == "atomic_claim":
            full.setdefault("claim_id", record_id)
            full.setdefault("statement", retrieval_text)
        relative = (
            Path("records") / record_type / f"{_slug(record_id)}.md"
        ).as_posix()
        front_matter = yaml.safe_dump(
            full, allow_unicode=True, sort_keys=False
        ).strip()
        raw = f"---\n{front_matter}\n---\n{retrieval_text}\n".encode()
        documents.append((relative, record_type, record_id, raw))

    for claim in sorted(bundle.atomic_claims, key=lambda item: item.claim_id):
        if claim.claim_status != "supported":
            continue
        linked_logics = logics_by_claim.get(claim.claim_id, [])
        linked_cards = [
            card
            for logic in linked_logics
            for card in cards_by_logic.get(logic.logic_unit_id, [])
        ]
        question_ids = unique(logic.question_leaf_id for logic in linked_logics)
        permissions = unique(card.permission.value for card in linked_cards)
        add_document(
            record_type="atomic_claim",
            record_id=claim.claim_id,
            knowledge_type=claim.knowledge_type,
            retrieval_text=claim.statement,
            metadata={
                "claim_id": claim.claim_id,
                "statement": claim.statement,
                "subject": claim.subject,
                "predicate": claim.predicate,
                "object": claim.object,
                "polarity": "support",
                "claim_kind": claim.claim_kind,
                "applicability": claim.applicability,
                "permission": permissions[0] if len(set(permissions)) == 1 else "explanation_only",
                "scientific_quality": {
                    "logic_units": [
                        logic.scientific_quality.model_dump(mode="json")
                        for logic in linked_logics
                    ]
                },
                "task_applicability": {
                    "logic_units": [
                        logic.task_applicability.model_dump(mode="json")
                        for logic in linked_logics
                    ]
                },
                "boundary_conditions": unique(
                    boundary
                    for logic in linked_logics
                    for boundary in logic.task_applicability.boundary_conditions
                ),
                "counterclaims": unique(
                    item for logic in linked_logics for item in logic.counterclaim_ids
                ),
                "abstain_if": unique(
                    item for logic in linked_logics for item in logic.abstain_if
                ),
                "question_leaf_id": question_ids,
                "decision_slot": unique(
                    questions[item].decision_slot for item in question_ids if item in questions
                ),
                "task_route": unique(logic.task_route for logic in linked_logics),
                "feature_channel": unique(
                    channel
                    for card in linked_cards
                    for channel in feature_channels(
                        card.required_inputs, card.candidate_feature
                    )
                ),
                "required_input": unique(
                    item for card in linked_cards for item in card.required_inputs
                ),
                "expected_direction": unique(
                    card.expected_direction for card in linked_cards
                ),
                "evidence_role": ["support"],
                "record_payload": claim.model_dump(mode="json"),
            },
        )

    for logic in sorted(bundle.logic_units, key=lambda item: item.logic_unit_id):
        linked_cards = cards_by_logic.get(logic.logic_unit_id, [])
        knowledge_types = unique(
            claims[item].knowledge_type
            for item in (*logic.premise_claim_ids, *logic.counterclaim_ids)
            if item in claims
        )
        add_document(
            record_type="logic_unit",
            record_id=logic.logic_unit_id,
            knowledge_type=knowledge_types[0] if len(knowledge_types) == 1 else "scientific_reasoning",
            retrieval_text=logic.retrieval_text,
            metadata={
                "logic_unit_id": logic.logic_unit_id,
                "permission": (
                    linked_cards[0].permission.value
                    if linked_cards
                    and len({card.permission.value for card in linked_cards}) == 1
                    else "explanation_only"
                ),
                "scientific_quality": logic.scientific_quality.model_dump(mode="json"),
                "task_applicability": logic.task_applicability.model_dump(mode="json"),
                "boundary_conditions": list(
                    logic.task_applicability.boundary_conditions
                ),
                "counterclaims": list(logic.counterclaim_ids),
                "abstain_if": list(logic.abstain_if),
                "question_leaf_id": [logic.question_leaf_id],
                "decision_slot": (
                    [questions[logic.question_leaf_id].decision_slot]
                    if logic.question_leaf_id in questions
                    else []
                ),
                "task_route": [logic.task_route],
                "feature_channel": unique(
                    channel
                    for card in linked_cards
                    for channel in feature_channels(
                        card.required_inputs, card.candidate_feature
                    )
                ),
                "required_input": unique(
                    item for card in linked_cards for item in card.required_inputs
                ),
                "expected_direction": unique(
                    card.expected_direction for card in linked_cards
                ),
                "evidence_role": [
                    "boundary" if logic.operator in {"constraint", "abstain"} else "support"
                ],
                "record_payload": logic.model_dump(mode="json"),
            },
        )

    for card in sorted(bundle.decision_cards, key=lambda item: item.decision_card_id):
        linked_logics = [logics[item] for item in card.logic_unit_ids if item in logics]
        retrieval_text = " ".join(
            part
            for part in (
                f"Decision route: {card.task_route}.",
                "Required inputs: " + "; ".join(card.required_inputs) + ".",
                f"Candidate feature: {card.candidate_feature}." if card.candidate_feature else "",
                "Boundaries: " + "; ".join(card.boundary_conditions) + ".",
                "Abstain when: " + "; ".join(card.abstain_if) + ".",
            )
            if part
        )
        knowledge_types = unique(
            claims[claim_id].knowledge_type
            for logic in linked_logics
            for claim_id in logic.premise_claim_ids
            if claim_id in claims
        )
        add_document(
            record_type="knowledge_decision_card",
            record_id=card.decision_card_id,
            knowledge_type=knowledge_types[0] if len(knowledge_types) == 1 else "decision_guidance",
            retrieval_text=retrieval_text,
            metadata={
                "decision_card_id": card.decision_card_id,
                "permission": card.permission.value,
                "scientific_quality": {
                    "logic_units": [
                        logic.scientific_quality.model_dump(mode="json")
                        for logic in linked_logics
                    ]
                },
                "task_applicability": {
                    "logic_units": [
                        logic.task_applicability.model_dump(mode="json")
                        for logic in linked_logics
                    ]
                },
                "boundary_conditions": list(card.boundary_conditions),
                "counterclaims": unique(
                    item for logic in linked_logics for item in logic.counterclaim_ids
                ),
                "abstain_if": list(card.abstain_if),
                "question_leaf_id": [card.question_leaf_id],
                "decision_slot": (
                    [questions[card.question_leaf_id].decision_slot]
                    if card.question_leaf_id in questions
                    else []
                ),
                "task_route": [card.task_route],
                "feature_channel": feature_channels(
                    card.required_inputs, card.candidate_feature
                ),
                "required_input": list(card.required_inputs),
                "expected_direction": [card.expected_direction],
                "evidence_role": ["boundary"],
                "record_payload": card.model_dump(mode="json"),
            },
        )
    return tuple(documents)


def expected_native_projection_files(
    bundle: EvidenceProductBundle,
) -> dict[str, tuple[str, str, bytes]]:
    rendered = {
        relative: (record_type, record_id, raw)
        for relative, record_type, record_id, raw in _native_record_documents(bundle)
    }
    rendered["evidence-release.json"] = (
        "evidence_release",
        bundle.release_manifest.release_id,
        (
            json.dumps(
                bundle.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return rendered


def export_native_local_rag_bundle(
    bundle: EvidenceProductBundle,
    output_root: str | Path,
    *,
    active_policy: ExternalEvidenceScopePolicy,
    trusted_reviewer_keys: Mapping[str, tuple[str, bytes]],
    trusted_release_approval_keys: Mapping[str, tuple[str, bytes]],
) -> dict[str, Any]:
    """Atomically publish the native three-record runtime projection."""

    report = validate_evidence_product(
        bundle,
        active_policy=active_policy,
        trusted_reviewer_keys=trusted_reviewer_keys,
        trusted_release_approval_keys=trusted_release_approval_keys,
    )
    if not report.release_ready:
        raise ValueError("Only a valid released evidence product can be exported")
    target_root = Path(output_root).absolute()
    access_policy = discover_workspace_access_policy(target_root)
    access_policy.require_allowed(target_root)
    if target_root.exists():
        raise FileExistsError(f"Refusing to overwrite export path: {target_root}")
    target_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target_root.name or 'native-runtime'}.building-",
        dir=target_root.parent,
    ) as staging:
        staging_root = Path(staging)
        rendered = expected_native_projection_files(bundle)
        written: list[dict[str, Any]] = []
        for relative, (record_type, record_id, raw) in rendered.items():
            target = staging_root / Path(relative)
            access_policy.require_allowed(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            written.append(_file_receipt(staging_root, target, record_type, record_id))
        runtime_manifest = {
            "schema_version": "local-rag-runtime-files:v2",
            "source_release_id": bundle.release_manifest.release_id,
            "policy_hash": bundle.release_manifest.policy_hash,
            "workspace_access_policy_hash": access_policy.policy_hash,
            "projection": "native_evidence_records_v1",
            "files": sorted(written, key=lambda item: item["relative_path"]),
        }
        runtime_manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                runtime_manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        (staging_root / "runtime-files.json").write_text(
            json.dumps(runtime_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        from fitness_agents.local_knowledge.runtime_manifest import (
            load_runtime_file_manifest,
        )

        load_runtime_file_manifest(
            staging_root,
            access_policy=access_policy,
            expected_external_policy_hash=active_policy.policy_hash,
            active_policy=active_policy,
            trusted_reviewer_keys=trusted_reviewer_keys,
            trusted_release_approval_keys=trusted_release_approval_keys,
        )
        staging_root.replace(target_root)
        return runtime_manifest


def expected_legacy_projection_files(
    bundle: EvidenceProductBundle,
) -> dict[str, tuple[str, str, bytes]]:
    """Render the exact deterministic v1 projection authenticated by Bundle v2.

    Runtime loading recomputes these bytes from the signed canonical bundle. This
    prevents an attacker from changing a projected claim/catalog and merely
    recomputing the ordinary file hashes in ``runtime-files.json``.
    """

    if bundle.release_manifest is None:
        raise ValueError("Legacy projection requires a ReleaseManifest")
    publications_by_id = {
        item.publication_id: item for item in bundle.publications
    }
    spans_by_id = {item.source_span_id: item for item in bundle.source_spans}
    groups_by_id = {
        item.evidence_group_id: item for item in bundle.evidence_groups
    }
    release_records = {
        item.record_id: item for item in bundle.release_manifest.records
    }
    export_claims = tuple(
        claim for claim in bundle.atomic_claims if claim.claim_status == "supported"
    )
    if not export_claims:
        raise ValueError("Legacy export requires at least one supported AtomicClaim")

    rendered: dict[str, tuple[str, str, bytes]] = {}
    publication_ids_used: set[str] = set()
    for claim in sorted(export_claims, key=lambda item: item.claim_id):
        release_record = release_records.get(claim.claim_id)
        if release_record is None or release_record.record_type != "atomic_claim":
            raise ValueError(
                f"AtomicClaim is absent from the release manifest: {claim.claim_id}"
            )
        supports: list[dict[str, Any]] = []
        for group_id in claim.evidence_group_ids:
            group = groups_by_id[group_id]
            if group.stance != "supports":
                continue
            for span_id in group.source_span_ids:
                span = spans_by_id[span_id]
                publication = publications_by_id[span.publication_id]
                if not publication.doi:
                    raise ValueError(
                        "Legacy export requires a DOI publication: "
                        f"{publication.publication_id}"
                    )
                publication_ids_used.add(publication.publication_id)
                supports.append(
                    {
                        "support_id": stable_id(
                            "citation",
                            claim.claim_id,
                            span.source_span_id,
                        ),
                        "publication_id": publication.publication_id,
                        "support_type": "direct_support",
                        "locator": span.locator,
                        "verified_against_source": True,
                    }
                )
        metadata = {
            "schema_version": "scientific-atomic-claim:v1",
            "record_type": "atomic_claim",
            "claim_id": claim.claim_id,
            "title": claim.statement[:100],
            "language": "en",
            "knowledge_type": claim.knowledge_type,
            "statement": claim.statement,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "object": claim.object,
            "polarity": "support",
            "claim_kind": claim.claim_kind,
            "confidence": _scientific_confidence(bundle, claim.claim_id),
            "applicability": claim.applicability,
            "citation_support": supports,
            "selection_eligible": _selection_eligible(bundle, claim.claim_id),
            "source_release_id": bundle.release_manifest.release_id,
            "source_record_hash": release_record.content_sha256,
        }
        relative = (
            Path("claims")
            / claim.knowledge_type
            / f"{_slug(claim.claim_id)}.md"
        ).as_posix()
        if relative in rendered:
            raise ValueError("Legacy export claim paths collide")
        front_matter = yaml.safe_dump(
            metadata,
            allow_unicode=True,
            sort_keys=False,
        ).strip()
        rendered[relative] = (
            "atomic_claim",
            claim.claim_id,
            f"---\n{front_matter}\n---\n{claim.statement}\n".encode(),
        )

    catalog = {
        "schema_version": "scientific-publications:v1",
        "generated_from": bundle.release_manifest.release_id,
        "verified_on": bundle.release_manifest.created_at.date().isoformat(),
        "publications": [
            {
                "publication_id": publication.publication_id,
                "title": publication.title,
                "authors": list(publication.authors),
                "year": publication.year,
                "venue": publication.venue,
                "doi": publication.doi,
                "url": publication.canonical_url,
                "publication_type": publication.publication_type,
                "verification": {
                    "metadata_source": "deep-research-evidence-product",
                    "metadata_verified": publication.metadata_verified,
                    "full_text_verified": publication.full_text_status == "verified",
                    "source_release_id": bundle.release_manifest.release_id,
                },
            }
            for publication in sorted(
                (
                    publications_by_id[publication_id]
                    for publication_id in publication_ids_used
                ),
                key=lambda item: item.publication_id,
            )
        ],
    }
    rendered["catalog/publications.yaml"] = (
        "publication_catalog",
        "catalog",
        yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False).encode("utf-8"),
    )
    rendered["evidence-release.json"] = (
        "evidence_release",
        bundle.release_manifest.release_id,
        (
            json.dumps(
                bundle.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return rendered


def _file_receipt(
    root: Path,
    path: Path,
    record_type: str,
    record_id: str,
) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "record_type": record_type,
        "record_id": record_id,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
