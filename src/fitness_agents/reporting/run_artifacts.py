from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from html import escape
from typing import Any

from fitness_agents.config import OutputConfig
from fitness_agents.contracts.schemas import (
    CampaignState,
    ValidationRecord,
    Variant,
)
from fitness_agents.utils.artifacts import JsonArtifactWriter


def _fitness_svg(round_metrics: Sequence[Mapping[str, float]]) -> str:
    width, height = 760, 360
    margin = 55
    if not round_metrics:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            '<text x="30" y="50">No completed rounds</text></svg>'
        )
    values = [
        float(item[key])
        for item in round_metrics
        for key in ("best_seen_fitness", "batch_best_fitness", "batch_mean_fitness")
    ]
    low, high = min(values), max(values)
    if high - low < 1e-12:
        high = low + 1.0

    def x(index: int) -> float:
        return margin + index * (width - 2 * margin) / max(len(round_metrics) - 1, 1)

    def y(value: float) -> float:
        return height - margin - (value - low) * (height - 2 * margin) / (high - low)

    series = (
        ("best_seen_fitness", "#1f77b4", "Best seen"),
        ("batch_best_fitness", "#2ca02c", "Batch best"),
        ("batch_mean_fitness", "#ff7f0e", "Batch mean"),
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#555"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#555"/>',
        '<text x="380" y="25" text-anchor="middle" font-size="16">Fitness progress by round</text>',
    ]
    for index, item in enumerate(round_metrics):
        lines.append(
            f'<text x="{x(index):.1f}" y="{height-margin+22}" text-anchor="middle" font-size="11">R{int(item["round_id"])}</text>'
        )
    for series_index, (key, color, label) in enumerate(series):
        points = " ".join(
            f'{x(index):.1f},{y(float(item[key])):.1f}'
            for index, item in enumerate(round_metrics)
        )
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        lines.append(
            f'<text x="{width-margin-120}" y="{margin+18*series_index}" fill="{color}" font-size="12">{escape(label)}</text>'
        )
    lines.append(
        f'<text x="8" y="{margin}" font-size="10">{high:.3f}</text><text x="8" y="{height-margin}" font-size="10">{low:.3f}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_campaign_outputs(
    writer: JsonArtifactWriter,
    *,
    output: OutputConfig,
    wild_type: Variant | None,
    state: CampaignState,
    variants: Mapping[str, Variant],
    round_metrics: Sequence[Mapping[str, float]],
    validation_records: Sequence[ValidationRecord],
) -> dict[str, str]:
    enabled = set(output.artifacts)
    paths: dict[str, str] = {}
    if wild_type is not None and "json" in enabled:
        paths["wild_type"] = str(writer.write_json("wild_type.json", wild_type))

    validation_by_round_variant: dict[tuple[int, str], list[ValidationRecord]] = defaultdict(list)
    for record in validation_records:
        validation_by_round_variant[(record.round_id, record.variant_id)].append(record)

    top_rows: list[dict[str, Any]] = []
    for round_id in sorted({item.round_id for item in state.selections}):
        selections = sorted(
            (item for item in state.selections if item.round_id == round_id),
            key=lambda item: item.selection_order,
        )[: output.top_k]
        rows = []
        for item in selections:
            variant = variants[item.variant_id]
            validations = validation_by_round_variant.get((round_id, item.variant_id), [])
            dry = [record for record in validations if record.validation_type == "dry"]
            wet = [record for record in validations if record.validation_type == "wet"]
            row = {
                **asdict(item),
                "variant": variant.variant,
                "sequence": variant.sequence,
                "mutation_notation": variant.mutation_notation,
                "mutation_count": variant.mutation_count,
                "dry_validation": [record.value for record in dry],
                "wet_validation": [record.value for record in wet],
                "rethink_verdict": wet[0].reflection_verdict if wet else None,
                "rethink_summary": wet[0].reflection_summary if wet else "",
                "assessment_id": wet[0].assessment_id if wet else None,
            }
            rows.append(row)
            top_rows.append(row)
        if "json" in enabled:
            writer.write_json(f"round_{round_id:02d}/top_k.json", rows)
        if "csv" in enabled:
            writer.write_csv(f"round_{round_id:02d}/top_k.csv", rows)

    if "csv" in enabled:
        paths["round_metrics"] = str(writer.write_csv("round_metrics.csv", round_metrics))
        paths["top_k_all_rounds"] = str(writer.write_csv("top_k_all_rounds.csv", top_rows))
    if "svg" in enabled:
        paths["fitness_curve"] = str(
            writer.write_text("fitness_progress.svg", _fitness_svg(round_metrics))
        )

    if "markdown" in enabled or "reasoning" in enabled:
        lines = [
            "# Campaign reasoning and validation report",
            "",
            f"Run: `{state.run_id}`",
            "",
            "## Wild type",
            "",
            (
                f"- Variant `{wild_type.variant}`; sequence `{wild_type.sequence}`; "
                f"notation `{wild_type.mutation_notation}`."
                if wild_type is not None
                else "- No explicit mutation_count=0 wild type was present in the visible data."
            ),
            "",
        ]
        for round_id in sorted({item.round_id for item in state.selections}):
            hypothesis = next(
                (item for item in state.hypotheses if item.hypothesis_id.endswith(f":r{round_id}")),
                None,
            )
            lines.extend([f"## Round {round_id}", ""])
            if hypothesis is not None:
                lines.extend(
                    [
                        f"Hypothesis: {hypothesis.statement}",
                        "",
                        f"Falsification: {hypothesis.falsification_criterion}",
                        "",
                    ]
                )
            lines.extend(
                [
                    "| Rank | Mutation | Driver | Agent score | UQ | Dry | Wet | ReThink |",
                    "|---:|---|---|---:|---:|---|---|---|",
                ]
            )
            for row in (item for item in top_rows if int(item["round_id"]) == round_id):
                lines.append(
                    "| {selection_order} | `{mutation_notation}` | {selection_driver} | "
                    "{design_score:.4f} | {design_uncertainty:.4f} | {dry} | {wet} | {rethink} |".format(
                        **row,
                        dry=", ".join(f"{value:.4f}" for value in row["dry_validation"]),
                        wet=", ".join(f"{value:.4f}" for value in row["wet_validation"]),
                        rethink=row["rethink_verdict"] or "",
                    )
                )
            lines.append("")
            for reflection in (
                item for item in state.rethink_reflections if item.round_id == round_id
            ):
                lines.append(
                    f"- `{variants[reflection.variant_id].mutation_notation}`: "
                    f"**{reflection.verdict}** — {reflection.summary}"
                )
            if any(
                item.round_id == round_id for item in state.rethink_reflections
            ):
                lines.append("")
            for reflection in (
                item for item in state.hypothesis_reflections if item.round_id == round_id
            ):
                lines.extend(
                    [
                        (
                            f"Hypothesis reflection ({reflection.assessment_status}): "
                            f"{reflection.summary}"
                        ),
                        "",
                        "- Retained: " + "; ".join(reflection.retained_claims),
                        "- Invalidated: " + "; ".join(reflection.invalidated_assumptions),
                        "- Unresolved: " + "; ".join(reflection.unresolved_questions),
                        "- Next actions: " + "; ".join(reflection.recommended_actions),
                    ]
                )
            lines.append("")
        paths["reasoning"] = str(writer.write_text("reasoning.md", "\n".join(lines) + "\n"))
    return paths
