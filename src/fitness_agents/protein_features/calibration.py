from __future__ import annotations

from dataclasses import replace

import numpy as np

from fitness_agents.config import KnowledgeProviderConfig
from fitness_agents.contracts.schemas import Evidence, FitnessObservation


def calibrate_visible_evidence(
    evidence: dict[str, list[Evidence]],
    observations: dict[str, FitnessObservation],
    provider_configs: dict[str, KnowledgeProviderConfig],
) -> dict[str, list[Evidence]]:
    """Fit channel-local linear calibration using only already visible measurements."""

    by_channel: dict[str, list[tuple[str, Evidence]]] = {}
    for variant_id, items in evidence.items():
        if variant_id not in observations:
            continue
        for item in items:
            config = provider_configs.get(item.channel)
            if config is not None and config.calibration == "visible_linear":
                by_channel.setdefault(item.channel, []).append((variant_id, item))

    models: dict[str, dict[str, float]] = {}
    for channel, rows in by_channel.items():
        config = provider_configs[channel]
        if len(rows) < config.minimum_calibration_samples:
            continue
        x = np.asarray([float(item.score) for _variant_id, item in rows])
        y = np.asarray([float(observations[variant_id].fitness) for variant_id, _item in rows])
        if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        residual_std = float(np.sqrt(np.mean((fitted - y) ** 2)))
        correlation = float(np.corrcoef(x, y)[0, 1])
        models[channel] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "target_center": float(np.mean(y)),
            "target_scale": max(float(np.std(y)), 1e-12),
            "residual_std": residual_std,
            "correlation": correlation,
            "sample_count": float(len(rows)),
        }

    calibrated: dict[str, list[Evidence]] = {}
    for variant_id, items in evidence.items():
        output = []
        for item in items:
            config = provider_configs.get(item.channel)
            model = models.get(item.channel)
            if config is None or model is None:
                output.append(item)
                continue
            predicted = model["slope"] * float(item.score) + model["intercept"]
            normalized = (predicted - model["target_center"]) / model["target_scale"]
            calibrated_score = float(np.tanh(normalized))
            provenance = {
                **item.provenance,
                "calibration": {
                    "kind": "visible_linear",
                    **model,
                    "label_scope": "already_visible_measurements_only",
                },
            }
            output.append(
                replace(
                    item,
                    score=calibrated_score,
                    confidence=abs(model["correlation"]),
                    uncertainty=model["residual_std"],
                    calibrated_score=calibrated_score,
                    calibrated=True,
                    contributes_to_selection=(
                        config.contributes_to_selection and item.quality_status == "ok"
                    ),
                    provenance=provenance,
                )
            )
        calibrated[variant_id] = output
    return calibrated
