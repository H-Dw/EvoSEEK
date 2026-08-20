"""Fail-closed parsing of user text into an EvolutionIntent."""

from __future__ import annotations

import re

from fitness_agents.contracts.interaction import EvolutionIntent, UserConstraintIntent

_SEQUENCE = re.compile(r"(?<![A-Za-z])([ACDEFGHIKLMNPQRSTVWY]{4,})(?![A-Za-z])")
_POSITION_LIST = re.compile(
    r"(?:位置|位点|positions?|sites?)\s*[:：]?\s*((?:\d+[\s,，、;；-]*)+)",
    re.IGNORECASE,
)
_BUDGET = re.compile(r"(?:预算|选择|输出|batch(?:\s+size)?)\D{0,8}(\d+)", re.IGNORECASE)
_DEPTH = re.compile(r"(?:(\d+)\s*[点位]|depth\s*[:=]?\s*(\d+))", re.IGNORECASE)
_ROUNDS = re.compile(r"(?:轮次|轮数|rounds?)\s*[:=]?\s*(\d+)", re.IGNORECASE)


def normalize_sequence_text(value: str) -> str:
    """Read plain or FASTA text without allowing prose to rewrite residues."""

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    sequence_lines = [line for line in lines if not line.startswith(">")]
    return "".join(sequence_lines).replace(" ", "").upper()


def _positions(prompt: str) -> tuple[int, ...]:
    match = _POSITION_LIST.search(prompt)
    if match is None:
        return ()
    return tuple(dict.fromkeys(int(item) for item in re.findall(r"\d+", match.group(1))))


class DeterministicEvolutionIntentParser:
    """Parse the bounded portion of a design request without shell/config interpolation."""

    def parse(
        self,
        prompt: str,
        *,
        sequence_text: str | None = None,
        configured_reference: str | None = None,
        configured_objective: str | None = None,
    ) -> EvolutionIntent:
        raw_prompt = (prompt or "").strip()
        sequence: str | None = None
        source = None
        if sequence_text and sequence_text.strip():
            sequence = normalize_sequence_text(sequence_text)
            source = "attachment"
        else:
            candidates = _SEQUENCE.findall(raw_prompt)
            if candidates:
                sequence = max(candidates, key=len).upper()
                source = "message"
            elif configured_reference:
                sequence = configured_reference.upper()
                source = "configured"

        listed_positions = _positions(raw_prompt)
        lower = raw_prompt.lower()
        if listed_positions and (
            "除" in raw_prompt
            or "except" in lower
            or "exclude" in lower
            or "排除" in raw_prompt
        ):
            constraints = UserConstraintIntent(
                position_policy="all_except",
                exclude_positions=listed_positions,
            )
        elif listed_positions:
            constraints = UserConstraintIntent(
                position_policy="include",
                include_positions=listed_positions,
            )
        else:
            constraints = UserConstraintIntent(position_policy="all")

        budget_match = _BUDGET.search(raw_prompt)
        depth_match = _DEPTH.search(raw_prompt)
        rounds_match = _ROUNDS.search(raw_prompt)
        requested_depth = (
            int(next(item for item in depth_match.groups() if item))
            if depth_match
            else 1
        )
        requested_rounds = int(rounds_match.group(1)) if rounds_match else 1
        requested_budget = int(budget_match.group(1)) if budget_match else None
        direction = (
            "minimize"
            if any(item in lower for item in ("降低", "减少", "minimize", "decrease"))
            else "maximize"
        )
        objective = raw_prompt
        if sequence:
            objective = objective.replace(sequence, "[reference sequence]")
        objective = objective.strip(" ：:,，") or (configured_objective or "")
        missing = []
        if not sequence:
            missing.append("reference_sequence")
        if not objective:
            missing.append("objective_text")
        policy_text = {
            "all": "全部位置",
            "include": f"指定位置 {list(constraints.include_positions)}",
            "all_except": f"除 {list(constraints.exclude_positions)} 外的全部位置",
        }[constraints.position_policy]
        summary = (
            f"使用 open_design，对{policy_text}执行 {requested_depth} 点替换；"
            f"方向={direction}，预算={requested_budget or '可信配置默认值'}。"
        )
        return EvolutionIntent(
            objective_text=objective or None,
            desired_direction=direction,
            sequence_source=source,
            reference_sequence=sequence,
            reference_id="REF01" if sequence else None,
            requested_depth=requested_depth,
            requested_rounds=requested_rounds,
            requested_budget=requested_budget,
            constraints=constraints,
            missing_fields=tuple(missing),
            confirmation_summary=summary,
        )
