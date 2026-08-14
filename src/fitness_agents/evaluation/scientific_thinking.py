from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_state(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "state.json").read_text(encoding="utf-8"))


def _round_sets(state: dict[str, Any]) -> dict[int, set[str]]:
    output: dict[int, set[str]] = {}
    for record in state.get("selections", []):
        output.setdefault(int(record["round_id"]), set()).add(record["variant_id"])
    return output


def _mean_jaccard_change(reference: dict[int, set[str]], intervention: dict[int, set[str]]) -> float:
    changes = []
    for round_id in sorted(set(reference).intersection(intervention)):
        left, right = reference[round_id], intervention[round_id]
        union = left | right
        changes.append(1.0 - len(left & right) / len(union) if union else 0.0)
    return sum(changes) / len(changes) if changes else 0.0


class ScientificThinkingEvaluator:
    """Behavioral/counterfactual rubric; never judges whether prose merely sounds scientific."""

    def evaluate(
        self,
        *,
        reference_dir: str | Path,
        knowledge_ablation_dir: str | Path,
        score_shuffle_dir: str | Path,
        evidence_deletion_dir: str | Path,
    ) -> dict[str, Any]:
        states = {
            "reference": _load_state(Path(reference_dir)),
            "knowledge_ablation": _load_state(Path(knowledge_ablation_dir)),
            "score_shuffle": _load_state(Path(score_shuffle_dir)),
            "evidence_deletion": _load_state(Path(evidence_deletion_dir)),
        }
        reference = states["reference"]
        hypotheses = reference.get("hypotheses", [])
        selections = reference.get("selections", [])
        falsifiable = [
            item
            for item in hypotheses
            if item.get("expected_outcome") and item.get("falsification_criterion")
        ]
        evidence_linked = [item for item in selections if item.get("evidence_ids")]
        rank_complete = [
            item
            for item in selections
            if all(
                isinstance(item.get(key), int) and item[key] >= 1
                for key in ("model_rank_all", "acquisition_rank_all", "eligible_rank")
            )
            and item.get("total_candidates", 0) >= item.get("eligible_candidates", 0)
        ]
        updated = [item for item in hypotheses[1:] if item.get("parent_hypothesis_id")]
        reference_sets = _round_sets(reference)
        sensitivities = {
            name: _mean_jaccard_change(reference_sets, _round_sets(state))
            for name, state in states.items()
            if name != "reference"
        }
        shuffled_selections = states["score_shuffle"].get("selections", [])
        tagged_shuffle = [
            item for item in shuffled_selections if "score_shuffle" in item.get("intervention_tags", [])
        ]
        metrics = {
            "hypotheses_per_round": len(hypotheses) / max(len(reference_sets), 1),
            "falsifiable_hypothesis_fraction": len(falsifiable) / max(len(hypotheses), 1),
            "hypothesis_update_fraction": len(updated) / max(len(hypotheses) - 1, 1),
            "selection_evidence_link_rate": len(evidence_linked) / max(len(selections), 1),
            "global_rank_tracking_completeness": len(rank_complete) / max(len(selections), 1),
            "score_shuffle_intervention_disclosure": len(tagged_shuffle)
            / max(len(shuffled_selections), 1),
            "knowledge_ablation_selection_change": sensitivities["knowledge_ablation"],
            "score_shuffle_selection_change": sensitivities["score_shuffle"],
            "evidence_deletion_selection_change": sensitivities["evidence_deletion"],
        }
        gates = {
            "forms_falsifiable_hypotheses": metrics["falsifiable_hypothesis_fraction"] >= 0.95,
            "updates_hypotheses_across_rounds": (
                len(reference_sets) <= 1 or metrics["hypothesis_update_fraction"] >= 0.5
            ),
            "cites_traceable_evidence": metrics["selection_evidence_link_rate"] >= 0.8,
            "tracks_global_candidate_rank": metrics["global_rank_tracking_completeness"] == 1.0,
            "responds_to_knowledge_intervention": (
                metrics["knowledge_ablation_selection_change"] > 0.0
            ),
            "responds_to_score_intervention": metrics["score_shuffle_selection_change"] > 0.0,
            "responds_to_evidence_intervention": (
                metrics["evidence_deletion_selection_change"] > 0.0
            ),
            "discloses_score_shuffle": metrics["score_shuffle_intervention_disclosure"] == 1.0,
        }
        return {
            "verdict": "scientist_like_behavior_supported" if all(gates.values()) else "not_supported",
            "interpretation": (
                "A positive verdict supports hypothesis/evidence/counterfactual behavior in this "
                "benchmark. It does not establish human-level scientific understanding."
            ),
            "metrics": metrics,
            "gates": gates,
            "run_dirs": {
                "reference": str(Path(reference_dir)),
                "knowledge_ablation": str(Path(knowledge_ablation_dir)),
                "score_shuffle": str(Path(score_shuffle_dir)),
                "evidence_deletion": str(Path(evidence_deletion_dir)),
            },
        }

