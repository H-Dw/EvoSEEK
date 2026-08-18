from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from fitness_agents.config import KnowledgeConfig, KnowledgeProviderConfig, ValidationConfig
from fitness_agents.contracts.schemas import (
    Evidence,
    FitnessObservation,
    Hypothesis,
    Prediction,
    ReThinkReflection,
    ValidationRecord,
    Variant,
)
from fitness_agents.kg_knowledge import (
    BuildContext,
    CampaignObservationAdapter,
    InferenceKnowledgeAdapter,
    KnowledgeGraphBuilder,
    LocalRAGKnowledgeAdapter,
    SQLiteGraphSink,
    ValidationKnowledgeAdapter,
)
from fitness_agents.local_knowledge import LocalKnowledgeBase, RetrievalResult
from fitness_agents.plugin_registry import PluginRegistry
from fitness_agents.protein_features import (
    MSAProfileProvider,
    PhyschemDescriptorProvider,
    ProteinTaskContext,
    StaticStructureProvider,
)
from fitness_agents.protein_features.calibration import calibrate_visible_evidence
from fitness_agents.utils.progress import heartbeat

from .graph import ObservationKnowledgeGraph
from .tool import AgentKnowledgeGraphTool


@dataclass(frozen=True)
class _CallableEvidenceProvider:
    channel: str
    function: object

    def evaluate(self, variant: Variant, *, round_id: int, **kwargs) -> Evidence:
        return self.function(variant, round_id, **kwargs)


@dataclass(frozen=True)
class _UnavailableEvidenceProvider:
    channel: str
    reason: str
    context: ProteinTaskContext
    parameter_set_id: str

    def evaluate(self, variant: Variant, *, round_id: int, **_kwargs: Any) -> Evidence:
        statement = f"{self.channel} evidence unavailable: {self.reason}"
        identity = json.dumps(
            {
                "channel": self.channel,
                "variant_id": variant.variant_id,
                "round_id": round_id,
                "reason": self.reason,
                "context_id": self.context.context_id,
            },
            sort_keys=True,
        )
        return Evidence(
            evidence_id=(
                f"ev:{self.channel}:"
                f"{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
            ),
            variant_id=variant.variant_id,
            channel=self.channel,
            statement=statement,
            score=0.0,
            source_id=f"unavailable:{self.channel}",
            confidence=0.0,
            round_id=round_id,
            evidence_type="unavailable",
            quality_status="unavailable",
            applicability="unknown",
            contributes_to_selection=False,
            warnings=(self.reason,),
            provenance={
                "provider": type(self).__name__,
                "provider_version": "v1",
                "context_id": self.context.context_id,
                "parameter_set_id": self.parameter_set_id,
            },
        )


# Legacy GB1 test descriptor. Production providers load named values from a versioned resource.
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
        protein_id: str = "unknown",
        validation_config: ValidationConfig | None = None,
        structured_graph_path: str | Path | None = None,
        task_context: ProteinTaskContext | None = None,
        protein_name: str | None = None,
        protein_aliases: tuple[str, ...] = (),
        protein_accessions: tuple[str, ...] = (),
        local_knowledge_enabled: bool = True,
    ) -> None:
        self.config = config
        self.validation_config = validation_config or ValidationConfig()
        self.protein_id = protein_id
        if task_context is None:
            positions = tuple(sorted(config.site_profiles))
            if not positions:
                raise ValueError(
                    "KnowledgeEngine requires task_context unless legacy site_profiles define positions"
                )
            wild_type = "".join(
                str(config.site_profiles[position].get("wild_type", ""))
                for position in positions
            )
            if len(wild_type) != len(positions):
                raise ValueError("Legacy site_profiles must define wild_type for every position")
            task_context = ProteinTaskContext.from_task(
                SimpleNamespace(
                    task_id="legacy_knowledge_context",
                    protein_id=protein_id,
                    assay_id=assay_id,
                    wild_type_sites=wild_type,
                    mutable_positions=list(positions),
                    reference_sequence=None,
                    reference_sequence_path=None,
                    sequence_position_offset=1,
                    numbering_scheme="legacy_site_profile",
                    assay_conditions={},
                    structure_resources=(),
                )
            )
        self.task_context = task_context
        self.provider_configs: dict[str, KnowledgeProviderConfig] = {}
        self.provider_status: dict[str, dict[str, str]] = {}
        self.graph = ObservationKnowledgeGraph(
            graph_path,
            assay_id=assay_id,
            recency_decay=self.validation_config.recency_decay,
            wild_type_code=task_context.wild_type_code,
            mutable_positions=task_context.mutable_positions,
        )
        self._observed_variants: dict[str, Variant] = {}
        self._observations: dict[str, FitnessObservation] = {}
        self._validation_records: dict[str, ValidationRecord] = {}
        self._reflections: dict[str, ReThinkReflection] = {}
        self._local_retrieval_results: dict[int, list[RetrievalResult]] = defaultdict(list)
        self._local_evidence: dict[int, list[Evidence]] = defaultdict(list)
        self.providers: dict[str, object] = {}
        sink_path = structured_graph_path or Path(graph_path).with_name("structured_kg.sqlite")
        self.structured_sink = SQLiteGraphSink(sink_path)
        self.local_knowledge: LocalKnowledgeBase | None = None
        self.local_knowledge_build_report = None
        if config.local_knowledge.enabled and local_knowledge_enabled:
            self.local_knowledge = LocalKnowledgeBase(
                config.local_knowledge,
                index_path=(
                    config.local_knowledge.corpus_index_path
                    or config.local_knowledge.index_path
                    or Path(graph_path).with_name("local_knowledge.sqlite")
                ),
                overlay_path=config.local_knowledge.retrieval_overlay_path,
                protein_id=protein_id,
                protein_name=protein_name,
                protein_aliases=protein_aliases,
                protein_accessions=protein_accessions,
                reference_sequence=task_context.full_sequence,
            )
            self.local_knowledge_build_report = self.local_knowledge.refresh()
        adapters = PluginRegistry("knowledge_adapter")
        adapters.register("campaign_observations", CampaignObservationAdapter())
        adapters.register("inference_records", InferenceKnowledgeAdapter())
        if (
            self.local_knowledge is not None
            and config.local_knowledge.kg_update.enabled
        ):
            adapters.register(
                "local_rag",
                LocalRAGKnowledgeAdapter(
                    self.local_knowledge.guard,
                    publication_catalog=self.local_knowledge.publication_catalog,
                ),
            )
        adapters.register("validation_records", ValidationKnowledgeAdapter())
        self.structured_builder = KnowledgeGraphBuilder(
            adapters,
            sinks=(self.structured_sink,),
            strict=True,
        )
        cache_dir = Path(graph_path).parent / "feature_cache"
        factories = {
            "physchem": ("aaindex_delta", self._create_physchem_provider),
            "conservation": ("msa_profile", lambda item: MSAProfileProvider(
                self.task_context,
                item,
                parameter_set_id=self.config.parameter_set_id,
                cache_dir=cache_dir,
            )),
            "structure": ("static_structure", lambda item: StaticStructureProvider(
                self.task_context,
                item,
                parameter_set_id=self.config.parameter_set_id,
            )),
            "kg": ("observation_association", lambda _item: _CallableEvidenceProvider(
                "kg", self._kg
            )),
        }
        for channel, (default_kind, factory) in factories.items():
            provider_config = config.provider(channel, default_kind)
            self.provider_configs[channel] = provider_config
            if not bool(getattr(config, channel)) or not provider_config.enabled:
                self.provider_status[channel] = {"status": "disabled", "kind": provider_config.kind}
                continue
            try:
                provider = self._provider_from_kind(channel, provider_config, factory)
                self.register_provider(provider)
                self.provider_status[channel] = {"status": "ready", "kind": provider_config.kind}
            except (FileNotFoundError, ValueError, RuntimeError) as error:
                if provider_config.missing_policy == "fail":
                    raise
                reason = f"{type(error).__name__}: {error}"
                self.register_provider(
                    _UnavailableEvidenceProvider(
                        channel,
                        reason,
                        self.task_context,
                        self.config.parameter_set_id,
                    )
                )
                self.provider_status[channel] = {
                    "status": "unavailable",
                    "kind": provider_config.kind,
                    "reason": reason,
                }

    def _create_physchem_provider(
        self, provider_config: KnowledgeProviderConfig
    ) -> PhyschemDescriptorProvider:
        return PhyschemDescriptorProvider(
            self.task_context,
            provider_config,
            parameter_set_id=self.config.parameter_set_id,
        )

    def _provider_from_kind(
        self,
        channel: str,
        provider_config: KnowledgeProviderConfig,
        factory: Any,
    ) -> object:
        legacy = {
            ("physchem", "legacy_physchem"): self._physchem,
            ("conservation", "legacy_site_profile"): self._conservation,
            ("structure", "legacy_site_risk"): self._structure,
        }
        if (channel, provider_config.kind) in legacy:
            return _CallableEvidenceProvider(channel, legacy[(channel, provider_config.kind)])
        expected = {
            "physchem": "aaindex_delta",
            "conservation": "msa_profile",
            "structure": "static_structure",
            "kg": "observation_association",
        }[channel]
        if provider_config.kind != expected:
            raise ValueError(
                f"Unsupported {channel} provider kind {provider_config.kind!r}; expected {expected!r}"
            )
        return factory(provider_config)

    def _parameter(self, name: str) -> float:
        spec = self.config.parameters.get(name)
        if spec is None:
            raise ValueError(
                f"Knowledge parameter {name!r} must be explicitly configured and versioned"
            )
        return float(spec.value)

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

    def record_validation(
        self,
        records: Sequence[ValidationRecord],
        reflections: Sequence[ReThinkReflection],
    ) -> None:
        self.graph.add_validation_records(records)
        self._validation_records.update({item.record_id: item for item in records})
        self._reflections.update({item.reflection_id: item for item in reflections})

    def validation_prior_scores(
        self,
        variants: Sequence[Variant],
        *,
        round_id: int,
    ) -> dict[str, float]:
        statistics = self.graph.validation_prior_statistics(round_id=round_id)
        if not statistics:
            return {item.variant_id: 0.0 for item in variants}
        global_mean = float(np.average(
            [item[0] for item in statistics.values()],
            weights=[max(item[1], 1e-8) for item in statistics.values()],
        ))
        output: dict[str, float] = {}
        for variant in variants:
            effects = []
            for index, residue in enumerate(variant.variant):
                position = self.task_context.mutable_positions[index]
                if (position, residue) not in statistics:
                    continue
                mean, weight, _wet_count, _dry_count = statistics[(position, residue)]
                effects.append((weight / (weight + 1.0)) * (mean - global_mean))
            output[variant.variant_id] = float(np.tanh(np.mean(effects))) if effects else 0.0
        return output

    def sync_structured_kg(
        self,
        *,
        run_id: str,
        round_id: int,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
        predictions: Sequence[Prediction] = (),
        evidence: Sequence[Evidence] = (),
        hypotheses: Sequence[Hypothesis] = (),
        local_retrieval_results: Sequence[RetrievalResult] | None = None,
    ):
        """Build, validate, and persist the external structured KG during the campaign."""

        current_local_results = tuple(
            self._local_retrieval_results.get(round_id, ())
            if local_retrieval_results is None
            else local_retrieval_results
        )
        return self.structured_builder.build(
            BuildContext(
                run_id=run_id,
                round_id=round_id,
                protein_id=self.protein_id,
                assay_id=self.graph.assay_id,
                resources={
                    "variants": tuple(variants),
                    "observations": tuple(observations),
                    "predictions": tuple(predictions),
                    "evidence": tuple(evidence),
                    "hypotheses": tuple(hypotheses),
                    "validation_records": tuple(self._validation_records.values()),
                    "reflections": tuple(self._reflections.values()),
                    "local_retrieval_results": current_local_results,
                },
            )
        )

    def prefetch_local_knowledge(
        self,
        *,
        round_id: int,
        objective: str,
        assay_conditions: dict[str, Any] | None = None,
        anchors: Sequence[str] = (),
        candidates: Sequence[Variant] = (),
    ) -> tuple[RetrievalResult | None, tuple[Evidence, ...]]:
        if self.local_knowledge is None:
            return None, ()
        result = self.local_knowledge.prefetch_round(
            round_id=round_id,
            objective=objective,
            assay_conditions=assay_conditions,
            anchors=anchors,
        )
        evidence = self.local_knowledge.evidence_from_result(
            result, candidates=candidates
        )
        if result.query_id not in {
            item.query_id for item in self._local_retrieval_results[round_id]
        }:
            self._local_retrieval_results[round_id].append(result)
        known_evidence = {item.evidence_id for item in self._local_evidence[round_id]}
        self._local_evidence[round_id].extend(
            item for item in evidence if item.evidence_id not in known_evidence
        )
        return result, evidence

    def local_evidence(self, *, round_id: int) -> tuple[Evidence, ...]:
        return tuple(self._local_evidence.get(round_id, ()))

    def retrieve_local_knowledge(
        self,
        *,
        query: str,
        intent: str,
        round_id: int,
        anchors: Sequence[str] = (),
        top_k: int | None = None,
        knowledge_types: Sequence[str] = (),
        stage: bool = True,
    ) -> tuple[RetrievalResult, tuple[Evidence, ...]]:
        if self.local_knowledge is None:
            raise RuntimeError("Local knowledge is not configured")
        result = self.local_knowledge.retrieve(
            query=query,
            intent=intent,
            round_id=round_id,
            anchors=anchors,
            top_k=top_k,
            knowledge_types=knowledge_types,
        )
        evidence = self.local_knowledge.evidence_from_result(result)
        if stage and bool(result.policy_decision.get("allowed", False)):
            if result.query_id not in {
                item.query_id for item in self._local_retrieval_results[round_id]
            }:
                self._local_retrieval_results[round_id].append(result)
            known_evidence = {item.evidence_id for item in self._local_evidence[round_id]}
            self._local_evidence[round_id].extend(
                item for item in evidence if item.evidence_id not in known_evidence
            )
        return result, evidence

    def query_structured_claims(
        self, *, query: str, round_id: int, limit: int = 12
    ) -> tuple[dict[str, Any], ...]:
        return self.structured_sink.query_claims(
            query=query, round_id=round_id, limit=limit
        )

    def _physchem(self, variant: Variant, round_id: int, **_kwargs) -> Evidence:
        distances = []
        for wt, mutant in zip(
            self.task_context.wild_type_residues, variant.variant, strict=True
        ):
            if wt == mutant:
                continue
            wt_props = np.asarray(AA_PROPERTIES[wt])
            mut_props = np.asarray(AA_PROPERTIES[mutant])
            scale = np.asarray([9.0, 170.0, 2.0, 1.0])
            distances.append(float(np.linalg.norm((wt_props - mut_props) / scale)))
        radicality = float(np.mean(distances)) if distances else 0.0
        score = 1.0 - min(radicality, 2.0) / 2.0
        statement = (
            f"[UNVALIDATED LEGACY HEURISTIC] mean physicochemical "
            f"conservativeness={score:.3f}"
        )
        return Evidence(
            evidence_id=_evidence_id("physchem", variant.variant_id, round_id, statement),
            variant_id=variant.variant_id,
            channel="physchem",
            statement=statement,
            score=score,
            source_id="aa_properties:v1",
            confidence=0.65,
            round_id=round_id,
            evidence_type="legacy_computed",
            quality_status="degraded",
            applicability="unknown",
            contributes_to_selection=self.config.legacy_contributes_to_selection,
            warnings=("legacy_unvalidated_heuristic",),
            provenance={
                "provider": "legacy_physchem",
                "provider_version": "v1",
                "context_id": self.task_context.context_id,
                "parameter_set_id": self.config.parameter_set_id,
            },
        )

    def _conservation(self, variant: Variant, round_id: int, **_kwargs) -> Evidence:
        values = []
        for index, (wt, mutant) in enumerate(
            zip(self.task_context.wild_type_residues, variant.variant, strict=True)
        ):
            if wt == mutant:
                continue
            position = self.task_context.mutable_positions[index]
            profile = self.config.site_profiles.get(position, {})
            tolerated = set(profile.get("tolerated", [wt]))
            values.append(0.5 if mutant in tolerated else -0.5)
        score = float(np.mean(values)) if values else 0.5
        statement = (
            f"[UNVALIDATED LEGACY HEURISTIC] site-profile compatibility={score:.3f}; "
            "this is not MSA conservation"
        )
        return Evidence(
            evidence_id=_evidence_id("conservation", variant.variant_id, round_id, statement),
            variant_id=variant.variant_id,
            channel="conservation",
            statement=statement,
            score=score,
            source_id="gb1_site_profile:v1",
            confidence=0.55,
            round_id=round_id,
            evidence_type="legacy_computed",
            quality_status="degraded",
            applicability="unknown",
            contributes_to_selection=self.config.legacy_contributes_to_selection,
            warnings=("legacy_site_profile_not_msa",),
            provenance={
                "provider": "legacy_site_profile",
                "provider_version": "v1",
                "context_id": self.task_context.context_id,
                "parameter_set_id": self.config.parameter_set_id,
            },
        )

    def _structure(self, variant: Variant, round_id: int, **_kwargs) -> Evidence:
        penalties = []
        for index, (wt, mutant) in enumerate(
            zip(self.task_context.wild_type_residues, variant.variant, strict=True)
        ):
            if wt == mutant:
                continue
            position = self.task_context.mutable_positions[index]
            penalties.append(float(self.config.site_profiles.get(position, {}).get("structure_risk", 0.5)))
        score = 1.0 - float(np.mean(penalties)) if penalties else 0.5
        statement = (
            f"[UNVALIDATED LEGACY HEURISTIC] configured site-risk tolerance={score:.3f}; "
            "no coordinates were analyzed and no affinity claim is made"
        )
        return Evidence(
            evidence_id=_evidence_id("structure", variant.variant_id, round_id, statement),
            variant_id=variant.variant_id,
            channel="structure",
            statement=statement,
            score=score,
            source_id="5LDE_site_risk:v1",
            confidence=0.45,
            round_id=round_id,
            evidence_type="legacy_computed",
            quality_status="degraded",
            applicability="unknown",
            contributes_to_selection=self.config.legacy_contributes_to_selection,
            warnings=("legacy_site_risk_not_three_dimensional_analysis",),
            provenance={
                "provider": "legacy_site_risk",
                "provider_version": "v1",
                "context_id": self.task_context.context_id,
                "parameter_set_id": self.config.parameter_set_id,
            },
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
            position = self.task_context.mutable_positions[index]
            if (position, residue) in statistics:
                mean, count = statistics[(position, residue)]
                pseudocount = self._parameter("kg.shrinkage_pseudocount")
                shrinkage = count / (count + pseudocount)
                effects.append(shrinkage * (mean - global_mean))
                support += count
        raw_score = float(np.mean(effects)) if effects else 0.0
        score = float(np.tanh(raw_score))
        statement = f"context-bound residue observation score={score:.3f}; support={support}"
        confidence_base = self._parameter("kg.confidence_base")
        support_gain = self._parameter("kg.support_gain")
        confidence_cap = self._parameter("kg.confidence_cap")
        return Evidence(
            evidence_id=_evidence_id("kg", variant.variant_id, round_id, statement),
            variant_id=variant.variant_id,
            channel="kg",
            statement=statement,
            score=score,
            source_id="observation_graph:current_run",
            confidence=min(confidence_cap, confidence_base + support_gain * support),
            round_id=round_id,
            evidence_type="measured_aggregate",
            raw_features={
                "raw_association_score": raw_score,
                "support": support,
                "global_visible_mean": global_mean,
            },
            quality_status="ok" if support else "degraded",
            applicability="partial",
            calibrated=False,
            contributes_to_selection=self.provider_configs["kg"].contributes_to_selection,
            warnings=(
                "descriptive_association_not_causal",
                "additive_residue_summary_may_miss_epistasis",
            ),
            provenance={
                "provider": "observation_association",
                "provider_version": "v2",
                "context_id": self.task_context.context_id,
                "parameter_set_id": self.config.parameter_set_id,
                "visibility": "already_visible_observations_only",
            },
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
            interval = self.config.evidence_heartbeat_interval
            if total >= interval and (index == total or index % interval == 0):
                heartbeat(
                    f"knowledge.evidence {index}/{total}",
                    completed=index,
                    total=total,
                )
        calibration_input = dict(output)
        calibration_channels = {
            channel
            for channel, provider_config in self.provider_configs.items()
            if provider_config.calibration == "visible_linear" and channel in self.providers
        }
        candidate_ids = tuple(calibration_input)
        if calibration_channels:
            for observed_variant in self._observed_variants.values():
                if observed_variant.variant_id in calibration_input:
                    continue
                calibration_input[observed_variant.variant_id] = [
                    _evaluate_provider(
                        self.providers[channel],
                        observed_variant,
                        round_id=round_id,
                        residue_statistics=residue_statistics,
                    )
                    for channel in sorted(calibration_channels)
                ]
        calibrated = calibrate_visible_evidence(
            calibration_input,
            self._observations,
            self.provider_configs,
        )
        return {
            variant_id: calibrated.get(variant_id, [])
            for variant_id in candidate_ids
        }

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
                    [item.score for item in items if item.contributes_to_selection],
                    weights=[
                        max(item.confidence, 1e-6)
                        for item in items
                        if item.contributes_to_selection
                    ],
                )
            )
            for variant_id, items in evidence.items()
            if any(item.contributes_to_selection for item in items)
        }

    def close(self) -> None:
        if self.local_knowledge is not None:
            self.local_knowledge.close()
        self.structured_sink.close()
        self.graph.close()
