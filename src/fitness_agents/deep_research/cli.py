from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fitness_agents.safety import discover_workspace_access_policy

from .contracts import QuestionLeaf, ResearchBrief
from .export import export_legacy_local_rag_bundle, export_native_local_rag_bundle
from .io import (
    load_evidence_product,
    load_research_brief,
    load_scope_assertions,
    write_json,
)
from .pipeline import (
    DeepSearchEngine,
    DeepSearchPlanner,
    build_release_manifest,
    issue_release_approval,
    validate_evidence_product,
)
from .providers import CrossrefSearchProvider, OpenAlexSearchProvider
from .trust import (
    RELEASE_KEYRING_ENV,
    REVIEW_KEYRING_ENV,
    load_active_policy_from_environment,
    load_signer_keyring_from_environment,
)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Release timestamp must include a timezone")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, discover, and validate quality-first external evidence"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    policy = subparsers.add_parser("policy", help="Print the active evidence-scope policy")
    policy.add_argument("--output", type=Path)

    init = subparsers.add_parser("init", help="Create a policy-bound ResearchBrief")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--brief-id", required=True)
    init.add_argument("--question", required=True)
    init.add_argument(
        "--decision-use",
        choices=("mechanism_explanation", "candidate_ranking", "campaign_strategy"),
        required=True,
    )
    init.add_argument("--leaf-id", default="leaf:primary")
    init.add_argument("--leaf-question", required=True)
    init.add_argument("--counterevidence-question", required=True)

    plan = subparsers.add_parser("plan", help="Create bounded, policy-checked search queries")
    plan.add_argument("brief", type=Path)
    plan.add_argument("--output", type=Path)

    search = subparsers.add_parser(
        "search",
        help="Run metadata-only discovery; unknown sources remain quarantined",
    )
    search.add_argument("brief", type=Path)
    search.add_argument("--output", type=Path, required=True)
    search.add_argument(
        "--providers",
        default="crossref,openalex",
        help="Comma-separated subset of crossref,openalex",
    )
    search.add_argument("--scope-assertions", type=Path)
    search.add_argument("--mailto")

    validate = subparsers.add_parser(
        "validate",
        help="Validate a complete evidence product and release manifest",
    )
    validate.add_argument("bundle", type=Path)
    validate.add_argument("--output", type=Path)

    approve = subparsers.add_parser(
        "approve",
        help="Record a human release approval bound to the complete evidence product",
    )
    approve.add_argument("bundle", type=Path)
    approve.add_argument("--output", type=Path, required=True)
    approve.add_argument("--approval-id", required=True)
    approve.add_argument("--reviewer-id", required=True)
    approve.add_argument("--method-version", required=True)
    approve.add_argument("--approval-artifact", type=Path, required=True)
    approve.add_argument("--approval-artifact-id", required=True)
    approve.add_argument("--release-version", required=True)
    approve.add_argument(
        "--release-created-at",
        help=(
            "Exact timezone-aware ISO-8601 timestamp for the intended release; "
            "defaults to approval time"
        ),
    )
    approve.add_argument("--parent-release-id")

    manifest = subparsers.add_parser(
        "manifest",
        help="Attach a content-addressed candidate or released manifest",
    )
    manifest.add_argument("bundle", type=Path)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--release-version", required=True)
    manifest.add_argument("--released", action="store_true")

    export = subparsers.add_parser(
        "export-legacy",
        help="Export a valid released product to the current local-RAG bundle format",
    )
    export.add_argument("bundle", type=Path)
    export.add_argument("--output-root", type=Path, required=True)
    export_native = subparsers.add_parser(
        "export-native",
        help="Export a valid released product to native evidence runtime records",
    )
    export_native.add_argument("bundle", type=Path)
    export_native.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Validation must still be able to emit a structured schema error before any
    # trust material is available. Signing operations fail closed when they try to
    # issue or verify a receipt with this unkeyed policy.
    policy = load_active_policy_from_environment(signing_required=False)
    review_signers = load_signer_keyring_from_environment(REVIEW_KEYRING_ENV)
    release_signers = load_signer_keyring_from_environment(RELEASE_KEYRING_ENV)
    reviewer_keys = review_signers
    release_keys = release_signers
    if args.action == "policy":
        payload = {
            "schema_version": "external-evidence-scope-manifest:v1",
            "policy_version": "external-evidence-scope:v1",
            "policy_hash": policy.policy_hash,
            "default_unknown_action": policy.default_unknown_action,
            "full_text_requires_verified_scope_assertion": True,
            "scope_assertion_requires_excluded_subject_absence": True,
            "receipt_attestation_required": True,
            "receipt_signer_key_id": policy.signing_key_id,
            "hmac_non_repudiation": False,
        }
        if args.output is not None:
            write_json(args.output, payload)
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.action == "init":
        decision = policy.inspect_query(
            f"{args.question} {args.leaf_question} {args.counterevidence_question}"
        )
        if decision.decision != "allowed":
            raise PermissionError(decision.reason)
        brief = ResearchBrief(
            brief_id=args.brief_id,
            research_question=args.question,
            decision_use=args.decision_use,
            question_tree=(
                QuestionLeaf(
                    question_leaf_id=args.leaf_id,
                    decision_slot="mechanism",
                    question=args.leaf_question,
                    counterevidence_question=args.counterevidence_question,
                    closure_rule=(
                        "Verified primary evidence, a mixed finding, or an explicit evidence gap."
                    ),
                ),
            ),
            inclusion_criteria=(
                "English primary research with verifiable source spans.",
                "Generic or independently verified nonviral protein scope.",
            ),
            exclusion_criteria=(
                "Virus-protein, mixed, unknown, retracted, or benchmark-quarantined sources.",
            ),
            policy_hash=policy.policy_hash,
            stop_conditions=(
                "All high-priority question leaves are supported, mixed, or no-evidence.",
                "Counterevidence and boundary searches are complete.",
                "No new independent LogicUnit is found within the remaining budget.",
            ),
        )
        write_json(args.output, brief)
        return 0

    if args.action == "plan":
        brief = load_research_brief(args.brief)
        plan = DeepSearchPlanner(policy).plan(brief)
        payload = {
            "schema_version": "deep-search-plan:v1",
            "brief_id": brief.brief_id,
            "policy_hash": policy.policy_hash,
            "queries": [item.model_dump(mode="json") for item in plan],
        }
        if args.output is not None:
            write_json(args.output, payload)
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.action == "search":
        provider_names = tuple(
            item.strip().casefold() for item in args.providers.split(",") if item.strip()
        )
        unknown = sorted(set(provider_names).difference({"crossref", "openalex"}))
        if unknown:
            raise ValueError(f"Unsupported Deep Search providers: {unknown}")
        providers = []
        if "crossref" in provider_names:
            providers.append(CrossrefSearchProvider(mailto=args.mailto))
        if "openalex" in provider_names:
            providers.append(
                OpenAlexSearchProvider(
                    api_key=os.environ.get("OPENALEX_API_KEY"),
                    mailto=args.mailto,
                )
            )
        discovery = DeepSearchEngine(
            providers,
            policy=policy,
            scope_assertions=load_scope_assertions(args.scope_assertions),
        ).discover(load_research_brief(args.brief))
        write_json(args.output, discovery)
        return 0

    if args.action == "validate":
        try:
            bundle = load_evidence_product(args.bundle)
        except (OSError, TypeError, ValueError) as validation_error:
            payload = {
                "schema_version": "deep-research-validation-error:v1",
                "valid": False,
                "release_ready": False,
                "issues": [
                    {
                        "severity": "error",
                        "code": "bundle.schema_invalid",
                        "record_id": None,
                        "message": str(validation_error),
                    }
                ],
            }
            if args.output is not None:
                write_json(args.output, payload)
            else:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 1
        report = validate_evidence_product(
            bundle,
            active_policy=policy,
            trusted_reviewer_keys=reviewer_keys,
            trusted_release_approval_keys=release_keys,
        )
        if args.output is not None:
            write_json(args.output, report)
        else:
            print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0 if report.valid else 1

    if args.action == "approve":
        bundle = load_evidence_product(args.bundle).model_copy(
            update={"release_manifest": None}
        )
        report = validate_evidence_product(
            bundle,
            active_policy=policy,
            trusted_reviewer_keys=reviewer_keys,
            trusted_release_approval_keys=release_keys,
        )
        if not report.valid:
            print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 1
        if any(
            item.approval_id == args.approval_id
            for item in bundle.release_approvals
        ):
            raise ValueError(f"Duplicate release approval ID: {args.approval_id}")
        artifact = args.approval_artifact.absolute()
        access_policy = discover_workspace_access_policy(artifact)
        access_policy.require_allowed(artifact)
        if artifact.is_symlink() or bool(
            getattr(artifact, "is_junction", lambda: False)()
        ):
            raise ValueError("Release approval artifact must not be a symlink or junction")
        resolved_artifact = artifact.resolve()
        access_policy.require_allowed(resolved_artifact)
        if not resolved_artifact.is_file():
            raise FileNotFoundError(
                f"Release approval artifact does not exist: {resolved_artifact}"
            )
        artifact_raw = resolved_artifact.read_bytes()
        signer = release_signers.get(args.reviewer_id)
        if signer is None:
            raise ValueError(
                f"Reviewer {args.reviewer_id!r} is absent from {RELEASE_KEYRING_ENV}"
            )
        signing_key_id, signing_key = signer
        approved_at = datetime.now(timezone.utc)
        target_created_at = (
            _parse_timestamp(args.release_created_at)
            if args.release_created_at
            else approved_at
        )
        approval = issue_release_approval(
            bundle,
            approval_id=args.approval_id,
            reviewer_id=args.reviewer_id,
            method_version=args.method_version,
            approval_artifact_id=args.approval_artifact_id,
            approval_artifact_sha256=hashlib.sha256(artifact_raw).hexdigest(),
            approved_at=approved_at,
            release_version=args.release_version,
            created_at=target_created_at,
            parent_release_id=args.parent_release_id,
            signing_key_id=signing_key_id,
            signing_key=signing_key,
        )
        write_json(
            args.output,
            bundle.model_copy(
                update={
                    "release_approvals": (*bundle.release_approvals, approval),
                }
            ),
        )
        return 0

    if args.action == "manifest":
        bundle = load_evidence_product(args.bundle)
        status = "released" if args.released else "candidate"
        release_created_at = None
        parent_release_id = None
        if status == "released":
            if not bundle.release_approvals:
                raise ValueError(
                    "Released manifest requires a release-intent-bound approval"
                )
            intents = {
                (
                    item.target_release_version,
                    item.target_created_at,
                    item.target_parent_release_id,
                )
                for item in bundle.release_approvals
            }
            if len(intents) != 1:
                raise ValueError("Release approvals do not share one exact release intent")
            approved_version, release_created_at, parent_release_id = intents.pop()
            if approved_version != args.release_version:
                raise ValueError(
                    "Manifest release version differs from the approved release intent"
                )
        manifest = build_release_manifest(
            bundle.model_copy(update={"release_manifest": None}),
            release_version=args.release_version,
            status=status,
            created_at=release_created_at,
            parent_release_id=parent_release_id,
        )
        updated = bundle.model_copy(update={"release_manifest": manifest})
        report = validate_evidence_product(
            updated,
            active_policy=policy,
            trusted_reviewer_keys=reviewer_keys,
            trusted_release_approval_keys=release_keys,
        )
        if not report.valid:
            print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 1
        write_json(args.output, updated)
        return 0

    if args.action == "export-legacy":
        receipt = export_legacy_local_rag_bundle(
            load_evidence_product(args.bundle),
            args.output_root,
            active_policy=policy,
            trusted_reviewer_keys=reviewer_keys,
            trusted_release_approval_keys=release_keys,
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.action == "export-native":
        receipt = export_native_local_rag_bundle(
            load_evidence_product(args.bundle),
            args.output_root,
            active_policy=policy,
            trusted_reviewer_keys=reviewer_keys,
            trusted_release_approval_keys=release_keys,
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"Unhandled action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
