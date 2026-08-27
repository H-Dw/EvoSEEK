"""Replay the fold-2 candidate-ranking boundary without hidden fitness labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fitness_agents.contracts.schemas import CampaignState, Hypothesis, Variant
from fitness_agents.data import load_campaign_fold_bundle
from fitness_agents.mutation.generators import KnowledgeCandidateGenerator
from fitness_agents.mutation.uncertainty import reserve_hypothesis_negative_controls

DEFAULT_AGENTIC_RUN = (
    PROJECT_ROOT
    / "artifacts/gf2/r/a/2/knowledge_agent-s11-f02-A2-20260827T104748704142Z"
)
DEFAULT_NO_RAG_RUN = (
    PROJECT_ROOT
    / "artifacts/gf2/r/n/2/knowledge_agent-s11-f02-N2-20260827T101705106303Z"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/analysis/gb1-fold2-agentic-guard-20260827.json"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agentic-run", type=Path, default=DEFAULT_AGENTIC_RUN)
    parser.add_argument("--no-rag-run", type=Path, default=DEFAULT_NO_RAG_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hypothesis(path: Path) -> Hypothesis:
    raw = dict(_json(path)["main_hypothesis"])
    allowed = {item.name for item in fields(Hypothesis)}
    value = {key: item for key, item in raw.items() if key in allowed}
    value["preferred_residues"] = {
        int(position): tuple(residues)
        for position, residues in value.get("preferred_residues", {}).items()
    }
    value["hard_residue_constraints"] = {
        int(position): tuple(residues)
        for position, residues in value.get("hard_residue_constraints", {}).items()
    }
    value["preference_strength_by_position"] = {
        int(position): str(strength)
        for position, strength in value.get(
            "preference_strength_by_position", {}
        ).items()
    }
    value["evidence_ids"] = tuple(value.get("evidence_ids", ()))
    return Hypothesis(**value)


def _depth_counts(variants: list[Variant]) -> dict[str, int]:
    return {
        str(depth): count
        for depth, count in sorted(Counter(item.mutation_count for item in variants).items())
    }


def _selected_with_policy(
    remaining: list[Variant],
    *,
    hypothesis: Hypothesis,
    state: CampaignState,
    namespace: str,
    position_to_index: dict[int, int],
    wild_type_by_position: dict[int, str] | None,
    candidate_limit: int,
) -> list[Variant]:
    generated = KnowledgeCandidateGenerator(
        position_to_index,
        sampling_namespace=namespace,
        wild_type_by_position=wild_type_by_position,
    ).generate(
        remaining,
        state,
        hypothesis,
        evidence={},
        limit=candidate_limit,
    )
    return reserve_hypothesis_negative_controls(
        generated,
        remaining,
        hypothesis=hypothesis,
        position_to_index=position_to_index,
        wild_type_by_position=wild_type_by_position,
        strong_threshold=0.75,
        required_controls=2,
        candidate_limit=candidate_limit,
        reserve_multiplier=2,
    )


def _round_replay(
    run_dir: Path,
    *,
    round_id: int,
    catalog: list[Variant],
    observed: list[dict[str, Any]],
    position_to_index: dict[int, int],
    wild_type_by_position: dict[int, str],
) -> dict[str, Any]:
    round_dir = run_dir / f"round_{round_id:02d}"
    receipt = _json(round_dir / "candidate_pool_receipt.json")
    hypothesis = _hypothesis(round_dir / "hypothesis_pipeline.json")
    revealed = {
        str(item["variant_id"])
        for item in observed
        if int(item["round_revealed"]) < round_id
    }
    remaining = [item for item in catalog if item.variant_id not in revealed]
    state = CampaignState(
        run_id="fold2-public-replay",
        mode="knowledge_agent",
        seed=int(receipt["seed"]),
        round_id=round_id,
    )
    common = {
        "hypothesis": hypothesis,
        "state": state,
        "namespace": str(receipt["sampling_namespace"]),
        "position_to_index": position_to_index,
        "candidate_limit": int(receipt["planned_candidate_count"]),
    }
    legacy = _selected_with_policy(
        remaining,
        wild_type_by_position=None,
        **common,
    )
    fixed = _selected_with_policy(
        remaining,
        wild_type_by_position=wild_type_by_position,
        **common,
    )
    legacy_ids = [item.variant_id for item in legacy]
    historical_ids = list(receipt["candidate_ids"])
    if legacy_ids != historical_ids:
        raise AssertionError(
            f"round {round_id} public replay does not reproduce the historical pool"
        )
    max_depth = max((item.mutation_count for item in remaining), default=0)
    legacy_depth = _depth_counts(legacy)
    fixed_depth = _depth_counts(fixed)
    return {
        "round": round_id,
        "legacy_reproduced_exactly": True,
        "historical_depth_counts": legacy_depth,
        "counterfactual_depth_counts": fixed_depth,
        "max_depth": max_depth,
        "max_depth_count_before": legacy_depth.get(str(max_depth), 0),
        "max_depth_count_after": fixed_depth.get(str(max_depth), 0),
        "max_depth_count_delta": (
            fixed_depth.get(str(max_depth), 0) - legacy_depth.get(str(max_depth), 0)
        ),
        "candidate_count": len(fixed),
    }


def main() -> int:
    args = arguments()
    agentic_run = args.agentic_run.resolve()
    no_rag_run = args.no_rag_run.resolve()
    output = args.output.resolve()
    config = _json(agentic_run / "config.json")
    source = config["data_source"]
    protein = config["protein_context"]
    bundle = load_campaign_fold_bundle(
        Path(source["split_root"]),
        int(source["fold_index"]),
    )
    positions = tuple(int(item) for item in protein["mutable_positions"])
    wild_type = tuple(str(protein["wild_type_sites"]))
    position_to_index = {position: index for index, position in enumerate(positions)}
    wild_type_by_position = dict(zip(positions, wild_type, strict=True))
    observed = list(_json(agentic_run / "state.json")["observed"])
    replays = [
        _round_replay(
            agentic_run,
            round_id=round_id,
            catalog=bundle.public_candidates,
            observed=observed,
            position_to_index=position_to_index,
            wild_type_by_position=wild_type_by_position,
        )
        for round_id in (2, 3)
    ]
    agentic_summary = _json(agentic_run / "summary.json")
    no_rag_summary = _json(no_rag_run / "summary.json")
    payload = {
        "schema_version": "gb1-fold2-agentic-guard-analysis:v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verification_status": "VERIFIED_CONTROL_FLOW",
        "source_scope": "public candidate structures plus already revealed wet outcomes",
        "hidden_oracle_labels_used_for_replay": False,
        "historical_performance": {
            "agentic_final_best": max(
                item["best_seen_fitness"]
                for item in agentic_summary["round_metrics"]
            ),
            "no_rag_final_best": max(
                item["best_seen_fitness"]
                for item in no_rag_summary["round_metrics"]
            ),
        },
        "counterfactual_replay": replays,
        "guard_assessment": {
            "legacy_candidate_pools_reproduced": all(
                item["legacy_reproduced_exactly"] for item in replays
            ),
            "max_depth_representation_non_decreasing": all(
                item["max_depth_count_delta"] >= 0 for item in replays
            ),
            "fitness_improvement_proven": False,
            "fitness_verdict": (
                "A deterministic candidate-ranking failure is blocked; a new live paired "
                "API run is required to estimate fitness impact."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
