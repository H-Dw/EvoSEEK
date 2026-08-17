from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fitness_agents.config import KnowledgeProviderConfig
from fitness_agents.contracts.schemas import Evidence, Variant

from .context import ProteinTaskContext


class PhyschemDescriptorProvider:
    """Named, source-backed mutation descriptors; not an assay fitness predictor."""

    channel = "physchem"

    def __init__(
        self,
        context: ProteinTaskContext,
        config: KnowledgeProviderConfig,
        *,
        parameter_set_id: str,
    ) -> None:
        if config.resource_path is None:
            raise ValueError("aaindex_delta provider requires resource_path")
        self.context = context
        self.config = config
        self.parameter_set_id = parameter_set_id
        self.resource_path = Path(config.resource_path)
        raw = yaml.safe_load(self.resource_path.read_text(encoding="utf-8")) or {}
        self.source_id = str(raw.get("source_id", self.resource_path.stem))
        self.resource_sha256 = hashlib.sha256(self.resource_path.read_bytes()).hexdigest()
        properties = raw.get("properties", {})
        if not properties:
            raise ValueError("Physchem property resource has no properties")
        self.properties: dict[str, dict[str, float]] = {}
        self.scales: dict[str, float] = {}
        self.accessions: dict[str, str] = {}
        for name, entry in properties.items():
            values = {str(key): float(value) for key, value in entry["values"].items()}
            if set(values) != set("ACDEFGHIKLMNPQRSTVWY"):
                raise ValueError(f"Property {name!r} must define all 20 canonical residues")
            self.properties[str(name)] = values
            span = max(values.values()) - min(values.values())
            self.scales[str(name)] = span if span > 0 else 1.0
            self.accessions[str(name)] = str(entry.get("accession", "unspecified"))

    def evaluate(self, variant: Variant, *, round_id: int, **_kwargs: Any) -> Evidence:
        site_features: dict[str, Any] = {}
        normalized_changes: list[float] = []
        special_flags: list[str] = []
        for position, wild_type, mutant in zip(
            self.context.mutable_positions,
            self.context.wild_type_residues,
            variant.variant,
            strict=True,
        ):
            if wild_type == mutant:
                continue
            deltas = {
                name: table[mutant] - table[wild_type]
                for name, table in self.properties.items()
            }
            site_features[str(position)] = {
                "mutation": f"{wild_type}{position}{mutant}",
                "deltas": deltas,
            }
            normalized_changes.extend(
                abs(deltas[name]) / self.scales[name] for name in sorted(deltas)
            )
            for residue, label in ((wild_type, "from"), (mutant, "to")):
                if residue in {"G", "P", "C"}:
                    special_flags.append(f"{label}_{residue}:{position}")

        mean_change = float(np.mean(normalized_changes)) if normalized_changes else 0.0
        conservativeness = 1.0 / (1.0 + mean_change)
        mutant_sequence = self.context.full_sequence_for_variant(variant.variant)
        wild_type_sequence = self.context.full_sequence

        def sequence_mean(property_name: str, sequence: str) -> float | None:
            values = self.properties.get(property_name)
            if values is None:
                return None
            return float(np.mean([values[residue] for residue in sequence]))

        global_sequence_deltas: dict[str, float] = {}
        if "residue_mass" in self.properties:
            global_sequence_deltas["molecular_weight_delta_da"] = float(
                sum(self.properties["residue_mass"][item] for item in mutant_sequence)
                - sum(self.properties["residue_mass"][item] for item in wild_type_sequence)
            )
        for property_name, output_name in (
            ("hydropathy", "mean_hydropathy_delta"),
            ("nominal_charge", "mean_nominal_charge_delta"),
        ):
            mutant_mean = sequence_mean(property_name, mutant_sequence)
            wild_type_mean = sequence_mean(property_name, wild_type_sequence)
            if mutant_mean is not None and wild_type_mean is not None:
                global_sequence_deltas[output_name] = mutant_mean - wild_type_mean
        aromatic = {"F", "W", "Y"}
        global_sequence_deltas["aromatic_fraction_delta"] = (
            sum(item in aromatic for item in mutant_sequence)
            - sum(item in aromatic for item in wild_type_sequence)
        ) / len(wild_type_sequence)
        raw_features = {
            "sites": site_features,
            "mean_normalized_absolute_delta": mean_change,
            "special_flags": sorted(set(special_flags)),
            "property_accessions": self.accessions,
            "assay_pH": self.context.assay_conditions.pH,
            "global_sequence_deltas": global_sequence_deltas,
            "global_sequence_method": "same_length_sequence_property_delta_v1",
        }
        warnings = ["descriptor_only_not_fitness", "nominal_charge_not_local_pka"]
        if self.context.assay_conditions.pH is None:
            warnings.append("assay_pH_unknown_charge_is_nominal")
        statement = (
            f"named physicochemical descriptor conservativeness={conservativeness:.3f}; "
            "descriptor only, not an assay-fitness claim"
        )
        identity = json.dumps(
            {
                "channel": self.channel,
                "variant_id": variant.variant_id,
                "round_id": round_id,
                "context_id": self.context.context_id,
                "resource_sha256": self.resource_sha256,
                "parameter_set_id": self.parameter_set_id,
                "raw_features": raw_features,
            },
            sort_keys=True,
        )
        return Evidence(
            evidence_id=f"ev:physchem:{hashlib.sha256(identity.encode()).hexdigest()[:16]}",
            variant_id=variant.variant_id,
            channel=self.channel,
            statement=statement,
            score=conservativeness,
            source_id=self.source_id,
            confidence=0.0,
            round_id=round_id,
            evidence_type="sequence_descriptor",
            raw_features=raw_features,
            quality_status="ok",
            applicability="in_domain",
            calibrated_score=None,
            calibrated=False,
            contributes_to_selection=False,
            warnings=tuple(warnings),
            provenance={
                "provider": type(self).__name__,
                "provider_version": "v1",
                "resource_path": str(self.resource_path),
                "resource_sha256": self.resource_sha256,
                "parameter_set_id": self.parameter_set_id,
                "context_id": self.context.context_id,
            },
        )
