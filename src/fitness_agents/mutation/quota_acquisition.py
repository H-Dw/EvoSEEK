from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from fitness_agents.agents.output_guards import RevisionConstraints
from fitness_agents.config import AgentQuotaAllocationConfig
from fitness_agents.contracts.batch_review import (
    CandidateIntentCard,
    ControlFeasibilityReceipt,
    control_feasibility_receipt,
)
from fitness_agents.contracts.schemas import DesignScore, Variant

ARMS = (
    "hypothesis_target",
    "evidence_prior",
    "coverage_exploration",
    "matched_control",
)


def _sequence_similarity(left: str, right: str) -> float:
    length = max(len(left), len(right), 1)
    distance = sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))
    return 1.0 - distance / length


def _sequence_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


@dataclass(frozen=True)
class AgentQuotaSelection:
    plugin: str
    selected_ids: tuple[str, ...]
    quotas: dict[str, int]
    selected_by_arm: dict[str, tuple[str, ...]]
    matched_control_pairs: dict[str, str]
    shortfalls: dict[str, int]
    fallback_ids: tuple[str, ...]
    strong_hypothesis_candidate_count: int
    matched_control_candidate_count: int
    control_feasibility: ControlFeasibilityReceipt
    eligible_before_filter: int
    eligible_after_filter: int
    constraint_excluded_ids: tuple[str, ...]

    def arm_by_id(self) -> dict[str, str]:
        output = {
            variant_id: arm
            for arm, variant_ids in self.selected_by_arm.items()
            for variant_id in variant_ids
        }
        output.update({variant_id: "fallback" for variant_id in self.fallback_ids})
        return output

    def intent_by_id(self) -> dict[str, CandidateIntentCard]:
        arms = self.arm_by_id()
        return {
            variant_id: CandidateIntentCard(
                candidate_id=variant_id,
                arm=arm,
                matched_to=self.matched_control_pairs.get(variant_id),
                allow_hypothesis_mismatch=(arm == "matched_control"),
            )
            for variant_id, arm in arms.items()
        }


class AgentQuotaBatchAcquisition:
    """Allocate one Agent-UQ batch across hypothesis, evidence, coverage and controls."""

    name = "agent_uq_quota_v1"

    def __init__(self, config: AgentQuotaAllocationConfig) -> None:
        self.config = config

    @staticmethod
    def _diversity_adjusted(
        variant_id: str,
        *,
        base_value: float,
        selected: Sequence[str],
        by_id: dict[str, Variant],
        diversity_lambda: float,
    ) -> float:
        if not selected or diversity_lambda <= 0:
            return base_value
        nearest_similarity = max(
            _sequence_similarity(by_id[variant_id].variant, by_id[chosen].variant)
            for chosen in selected
        )
        return base_value - diversity_lambda * nearest_similarity

    def select(
        self,
        variants: Sequence[Variant],
        scores: Sequence[DesignScore],
        budget: int,
        *,
        diversity_lambda: float,
        constraints: RevisionConstraints | None = None,
        position_to_index: dict[int, int] | None = None,
        wild_type_by_position: dict[int, str] | None = None,
        quota_overrides: Mapping[str, int] | None = None,
    ) -> AgentQuotaSelection:
        if budget < 0:
            raise ValueError("Agent quota acquisition budget must be non-negative")
        by_id = {item.variant_id: item for item in variants}
        original_ids = set(by_id)
        eligible_before_filter = len(by_id)
        resolved_position_to_index = position_to_index or {}

        def arm_eligible(variant_id: str, arm: str) -> bool:
            if constraints is None or not constraints.has_residue_constraints:
                return True
            if not resolved_position_to_index:
                raise ValueError(
                    "position_to_index is required for residue-constrained acquisition"
                )
            return not constraints.variant_violations(
                by_id[variant_id],
                arm=arm,
                position_to_index=resolved_position_to_index,
                wild_type_by_position=wild_type_by_position,
            )

        eligible_in_any_arm = {
            variant_id
            for variant_id in by_id
            if any(
                arm_eligible(variant_id, arm)
                for arm in (*ARMS, "fallback")
            )
        }
        if constraints is not None and constraints.reduce_mutation_depth and by_id:
            max_depth = max(item.mutation_count for item in by_id.values())
            depth_cap = max(0, max_depth - 1)
            reduced = {
                variant_id: variant
                for variant_id, variant in by_id.items()
                if variant.mutation_count <= depth_cap
            }
            if reduced:
                by_id = reduced
                variants = [item for item in variants if item.variant_id in by_id]
        eligible_after_constraints = set(by_id).intersection(eligible_in_any_arm)
        constraint_excluded_ids = tuple(
            sorted(original_ids.difference(eligible_after_constraints))
        )
        score_by_id = {item.variant_id: item for item in scores if item.variant_id in by_id}
        missing = set(by_id).difference(score_by_id)
        if missing:
            raise ValueError(f"Agent quota scores omit {len(missing)} candidates")

        target = min(budget, len(by_id))
        quotas = dict(self.config.quotas())
        if constraints is not None:
            if constraints.require_controls:
                quotas["matched_control"] = max(
                    quotas["matched_control"],
                    constraints.required_control_count or 2,
                )
            if constraints.add_exploration:
                quotas["coverage_exploration"] = quotas["coverage_exploration"] + 2
            if constraints.increase_diversity:
                diversity_lambda = max(diversity_lambda, diversity_lambda + 0.15, 0.25)
        # Round-scoped overrides (e.g. mutation-order-restricted rounds run
        # without matched controls) win over constraint escalation: a critic
        # cannot demand controls that the restricted design space lacks.
        if quota_overrides:
            for arm, value in quota_overrides.items():
                if arm not in ARMS:
                    raise ValueError(f"Unknown quota arm in overrides: {arm!r}")
                if value < 0:
                    raise ValueError("Quota overrides must be non-negative")
                quotas[arm] = int(value)
        selected: list[str] = []
        selected_by_arm: dict[str, list[str]] = {arm: [] for arm in ARMS}
        matched_control_pairs: dict[str, str] = {}
        available = set(by_id)
        minimum_batch_distance = (
            constraints.minimum_batch_distance
            if constraints is not None and constraints.increase_diversity
            else None
        )

        strong = {
            variant_id
            for variant_id, score in score_by_id.items()
            if score.hypothesis_score >= self.config.strong_hypothesis_threshold
        }
        controls = {
            variant_id
            for variant_id, score in score_by_id.items()
            if 0.0 < score.hypothesis_score < self.config.strong_hypothesis_threshold
        }
        if not controls:
            controls = {
                variant_id
                for variant_id, score in score_by_id.items()
                if score.hypothesis_score < self.config.strong_hypothesis_threshold
            }

        def take_ranked(
            arm: str,
            candidates: set[str],
            base: Callable[[DesignScore], float],
        ) -> None:
            quota = min(quotas[arm], target - len(selected))
            for _ in range(quota):
                eligible = candidates.intersection(available)
                eligible = {
                    variant_id
                    for variant_id in eligible
                    if arm_eligible(variant_id, arm)
                }
                if not eligible:
                    break
                if minimum_batch_distance is not None and selected:
                    distance_eligible = {
                        variant_id
                        for variant_id in eligible
                        if min(
                            _sequence_distance(
                                by_id[variant_id].variant,
                                by_id[chosen].variant,
                            )
                            for chosen in selected
                        )
                        >= minimum_batch_distance
                    }
                    if distance_eligible:
                        eligible = distance_eligible
                choice = max(
                    eligible,
                    key=lambda variant_id: (
                        self._diversity_adjusted(
                            variant_id,
                            base_value=base(score_by_id[variant_id]),
                            selected=selected,
                            by_id=by_id,
                            diversity_lambda=diversity_lambda,
                        ),
                        variant_id,
                    ),
                )
                selected.append(choice)
                selected_by_arm[arm].append(choice)
                available.remove(choice)

        take_ranked(
            "hypothesis_target",
            strong,
            lambda item: 2.0 * item.hypothesis_score + item.utility,
        )

        control_quota = min(quotas["matched_control"], target - len(selected))
        target_ids = tuple(selected_by_arm["hypothesis_target"])
        for _ in range(control_quota):
            eligible = {
                variant_id
                for variant_id in controls.intersection(available)
                if arm_eligible(variant_id, "matched_control")
            }
            if not eligible or not target_ids:
                break

            def control_key(variant_id: str) -> tuple[float, float, float, float, str]:
                variant = by_id[variant_id]
                same_depth = max(
                    float(variant.mutation_count == by_id[target_id].mutation_count)
                    for target_id in target_ids
                )
                nearest_distance = min(
                    1.0 - _sequence_similarity(variant.variant, by_id[target_id].variant)
                    for target_id in target_ids
                )
                prefer_wt = float(
                    constraints is not None
                    and constraints.require_controls
                    and variant.mutation_count <= 1
                )
                return (
                    prefer_wt,
                    same_depth,
                    -nearest_distance,
                    score_by_id[variant_id].hypothesis_score,
                    variant_id,
                )

            choice = max(eligible, key=control_key)
            anchor = max(
                target_ids,
                key=lambda target_id: (
                    float(
                        by_id[choice].mutation_count
                        == by_id[target_id].mutation_count
                    ),
                    _sequence_similarity(
                        by_id[choice].variant,
                        by_id[target_id].variant,
                    ),
                    target_id,
                ),
            )
            selected.append(choice)
            selected_by_arm["matched_control"].append(choice)
            matched_control_pairs[choice] = anchor
            available.remove(choice)

        evidence_candidates = {
            variant_id
            for variant_id, score in score_by_id.items()
            if max(score.evidence_score, score.prior_score) > 0.0
        }
        take_ranked(
            "evidence_prior",
            evidence_candidates,
            lambda item: max(item.evidence_score, item.prior_score) + 0.05 * item.utility,
        )
        take_ranked(
            "coverage_exploration",
            set(score_by_id),
            lambda item: item.uncertainty + 0.05 * item.utility,
        )

        shortfalls = {
            arm: max(0, quotas[arm] - len(selected_by_arm[arm])) for arm in ARMS
        }
        fallback_ids: list[str] = []
        fallback_available = {
            variant_id
            for variant_id in available
            if arm_eligible(variant_id, "fallback")
        }
        while fallback_available and len(selected) < target:
            choice = max(
                fallback_available,
                key=lambda variant_id: (
                    self._diversity_adjusted(
                        variant_id,
                        base_value=score_by_id[variant_id].utility,
                        selected=selected,
                        by_id=by_id,
                        diversity_lambda=diversity_lambda,
                    ),
                    variant_id,
                ),
            )
            selected.append(choice)
            fallback_ids.append(choice)
            available.remove(choice)
            fallback_available.remove(choice)

        return AgentQuotaSelection(
            plugin=self.name,
            selected_ids=tuple(selected),
            quotas=quotas,
            selected_by_arm={arm: tuple(selected_by_arm[arm]) for arm in ARMS},
            matched_control_pairs=matched_control_pairs,
            shortfalls=shortfalls,
            fallback_ids=tuple(fallback_ids),
            strong_hypothesis_candidate_count=len(strong),
            matched_control_candidate_count=len(controls),
            control_feasibility=control_feasibility_receipt(
                requested_controls=quotas["matched_control"],
                available_control_ids=tuple(controls),
                selected_control_ids=tuple(selected_by_arm["matched_control"]),
            ),
            eligible_before_filter=eligible_before_filter,
            eligible_after_filter=len(eligible_after_constraints),
            constraint_excluded_ids=constraint_excluded_ids,
        )
