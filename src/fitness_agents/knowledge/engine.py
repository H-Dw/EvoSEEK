from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fitness_agents.config import KnowledgeConfig
from fitness_agents.contracts.schemas import Evidence, FitnessObservation, Prediction, Variant
from fitness_agents.utils.progress import heartbeat

from .graph import ObservationKnowledgeGraph
from .tool import AgentKnowledgeGraphTool


@dataclass(frozen=True)
class _CallableEvidenceProvider:
    channel: str
    function: object

    def evaluate(self, variant: Variant, *, round_id: int, **kwargs) -> Evidence:
        return self.function(variant, round_id, **kwargs)


# Kyte-Doolittle hydropathy, approximate side-chain volume, charge, polarity group.
AA_PROPERTIES: dict[str, tuple[float, float, float, float]] = {
    "A": (1.8, 88.6, 0.0, 0.0), "C": (2.5, 108.5, 0.0, 0.3),
    "D": (-3.5, 111.1, -1.0, 1.0), "E": (-3.5, 138.4, -1.0, 1.0),
    "F": (2.8, 189.9, 0.0, 0.0), "G": (-0.4, 60.1, 0.0, 0.2),
    "H": (-3.2, 153.2, 0.1, 0.8), "I": (4.5, 166.7, 0.0, 0.0),
    "K": (-3.9, 168.6, 1.0, 1.0), "L": (3.8, 166.7, 0.0, 0.0),
    "M": (1.9, 162.9, 0.0, 0.2), "N": (-3.5, 114.1, 0.0, 1.0),
    "P": (-1.6, 112.7, 0.0, 0.3), "Q": (-3.5, 143.8, 0.0, 1.0),
    "R": (-4.5, 173.4, 1.0, 1.0), "S": (-0.8, 89.0, 0.0, 1.0),
    "T": (-0.7, 116.1, 0.0, 1.0), "V": (4.2, 140.0, 0.0, 0.0),
    "W": (-0.9, 227.8, 0.0, 0.2), "Y": (-1.3, 193.6, 0.0, 0.7),
}


def _evaluate_provider(
    provider: object,
    variant: Variant,
    *,
    round_id: int,
    residue_statistics: dict[tuple[int, str], tuple[float, int]],
) -> Evidence:
    evaluate = provider.evaluate
    try:
        return evaluate(
            variant,
            round_id=round_id,
            residue_statistics=residue_statistics,
        )
    except TypeError:
        return evaluate(variant, round_id=round_id)


def _evidence_id(channel: str, variant_id: str, round_id: int, statement: str) -> str:
    digest = hashlib.sha256(
        f"{channel}|{variant_id}|{round_id}|{statement}".encode()
    ).hexdigest()[:16]
    return f"ev:{channel}:{digest}"


class KnowledgeEngine:
    """Composable evidence channels; every channel can be independently disabled."""

    def __init__(
        self,
        config: KnowledgeConfig,
        *,
        graph_path: str | Path,
        assay_id: str,
    ) -> None:
        self.config = config
        self.graph = ObservationKnowledgeGraph(graph_path, assay_id=assay_id)
        self._observed_variants: dict[str, Variant] = {}
        self._observations: dict[str, FitnessObservation] = {}
        self.providers: dict[str, object] = {}
        for enabled, channel, function in (
            (config.physchem, "physchem", self._physchem),
            (config.conservation, "conservation", self._conservation),
            (config.structure, "structure", self._structure),
            (config.kg, "kg", self._kg),
        ):
            if enabled:
                self.register_provider(_CallableEvidenceProvider(channel, function))

    def register_provider(self, provider: object) -> None:
        channel = getattr(provider, "channel", "")
        if not channel or channel in self.providers:
            raise ValueError(f"Evidence provider channel must be new and non-empty: {channel!r}")
        if not callable(getattr(provider, "evaluate", None)):
            raise TypeError("Evidence provider must implement evaluate(variant, round_id=...)")
        self.providers[channel] = provider

    def update(
        self,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
    ) -> None:
        for variant in variants:
            self._observed_variants[variant.variant_id] = variant
        for observation in observations:
            self._observations[observation.variant_id] = observation
        self.graph.add_observations(variants, observations)

    def _physchem(self, variant: Variant, round_id: int, **_kwargs) -> Evidence:
        distances = []
        for wt, mutant in zip("VDGV", variant.variant, strict=True):
            if wt == mutant:
                continue
            wt_props = np.asarray(AA_PROPERTIES[wt])
            mut_props = np.asarray(AA_PROPERTIES[mutant])
            scale = np.asarray([9.0, 170.0, 2.0, 1.0])
            distances.append(float(np.linalg.norm((wt_props - mut_props) / scale)))
        radicality = float(np.mean(distances)) if distances else 0.0
        score = 1.0 - min(radicality, 2.0) / 2.0
        statement = f"mean physicochemical conservativeness={score:.3f}"
        return Evidence(
            evidence_id=_evidence_id("physchem", variant.variant_id, round_id, statement),
            variant_id=variant.variant_id,
            channel="physchem",
            statement=statement,
            score=score,
            source_id="aa_properties:v1",
            confidence=0.65,
            round_id=round_id,
        )

    def _conservation(self, variant: Variant, round_id: int, **_kwargs) -> Evidence:
        values = []
        for index, (wt, mutant) in enumerate(zip("VDGV", variant.variant, strict=True)):
            if wt == mutant:
                continue
            position = (39, 40, 41, 54)[index]
            profile = self.config.site_profiles.get(position, {})
            tolerated = set(profile.get("tolerated", [wt]))
            values.append(0.5 if mutant in tolerated else -0.5)
        score = float(np.mean(values)) if values else 0.5
        statement = f"soft conservation compatibility={score:.3f}; not a hard exclusion"
        return Evidence(
            evidence_id=_evidence_id("conservation", variant.variant_id, round_id, statement),
            variant_id=variant.variant_id,
            channel="conservation",
            statement=statement,
            score=score,
            source_id="gb1_site_profile:v1",
            confidence=0.55,
            round_id=round_id,
        )

    def _structure(self, variant: Variant, round_id: int, **_kwargs) -> Evidence:
        penalties = []
        for index, (wt, mutant) in enumerate(zip("VDGV", variant.variant, strict=True)):
            if wt == mutant:
                continue
            position = (39, 40, 41, 54)[index]
            penalties.append(float(self.config.site_profiles.get(position, {}).get("structure_risk", 0.5)))
        score = 1.0 - float(np.mean(penalties)) if penalties else 0.5
        statement = f"precomputed structural tolerance={score:.3f}; no affinity claim"
        return Evidence(
            evidence_id=_evidence_id("structure", variant.variant_id, round_id, statement),
            variant_id=variant.variant_id,
            channel="structure",
            statement=statement,
            score=score,
            source_id="5LDE_site_risk:v1",
            confidence=0.45,
            round_id=round_id,
        )

    def _kg(
        self,
        variant: Variant,
        round_id: int,
        residue_statistics: dict[tuple[int, str], tuple[float, int]] | None = None,
        **_kwargs,
    ) -> Evidence:
        statistics = (
            residue_statistics if residue_statistics is not None else self.graph.residue_statistics()
        )
        global_values = [observation.fitness for observation in self._observations.values()]
        global_mean = float(np.mean(global_values)) if global_values else 0.0
        effects = []
        support = 0
        for index, residue in enumerate(variant.variant):
            position = (39, 40, 41, 54)[index]
            if (position, residue) in statistics:
                mean, count = statistics[(position, residue)]
                shrinkage = count / (count + 3.0)
                effects.append(shrinkage * (mean - global_mean))
                support += count
        raw_score = float(np.mean(effects)) if effects else 0.0
        score = float(np.tanh(raw_score))
        statement = f"context-bound residue observation score={score:.3f}; support={support}"
        return Evidence(
            evidence_id=_evidence_id("kg", variant.variant_id, round_id, statement),
            variant_id=variant.variant_id,
            channel="kg",
            statement=statement,
            score=score,
            source_id="observation_graph:current_run",
            confidence=min(0.85, 0.25 + 0.03 * support),
            round_id=round_id,
            evidence_type="measured_aggregate",
        )

    def evidence_for(
        self,
        variants: Sequence[Variant],
        *,
        round_id: int,
        delete_evidence: bool = False,
    ) -> dict[str, list[Evidence]]:
        output: dict[str, list[Evidence]] = defaultdict(list)
        if delete_evidence:
            return dict(output)
        residue_statistics = (
            self.graph.residue_statistics() if "kg" in self.providers else {}
        )
        total = len(variants)
        for index, variant in enumerate(variants, start=1):
            for provider in self.providers.values():
                output[variant.variant_id].append(
                    _evaluate_provider(
                        provider,
                        variant,
                        round_id=round_id,
                        residue_statistics=residue_statistics,
                    )
                )
            if total >= 256 and (index == total or index % 256 == 0):
                heartbeat(
                    f"knowledge.evidence {index}/{total}",
                    completed=index,
                    total=total,
                )
        return dict(output)

    def record_inference_context(
        self,
        variants: Sequence[Variant],
        predictions: Sequence[Prediction],
        evidence: dict[str, list[Evidence]],
        *,
        round_id: int,
        intervention_tags: Sequence[str] = (),
    ) -> None:
        """Persist typed model/evidence entities without upgrading them to measurements."""
        self.graph.add_predictions(
            variants,
            predictions,
            round_id=round_id,
            intervention_tags=intervention_tags,
        )
        self.graph.add_evidence([item for bundle in evidence.values() for item in bundle])

    def agent_tool(self, *, max_rows: int = 12) -> AgentKnowledgeGraphTool:
        return AgentKnowledgeGraphTool(self.graph, max_rows=max_rows)

    @staticmethod
    def scores(evidence: dict[str, list[Evidence]]) -> dict[str, float]:
        return {
            variant_id: float(
                np.average(
                    [item.score for item in items],
                    weights=[max(item.confidence, 1e-6) for item in items],
                )
            )
            for variant_id, items in evidence.items()
            if items
        }

    def close(self) -> None:
        self.graph.close()
