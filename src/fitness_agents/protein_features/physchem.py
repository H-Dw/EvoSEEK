from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fitness_agents.config import KnowledgeProviderConfig
from fitness_agents.contracts.schemas import Evidence, Variant

from .context import ProteinTaskContext
from .substitution_store import CANONICAL_RESIDUES, compact_static_evidence_id


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
        self._substitutions = self._build_substitutions()
        self._sequence_length = max(len(self.context.full_sequence), 1)

    def _build_substitutions(self) -> dict[tuple[int, str, str], dict[str, Any]]:
        aromatic = {"F", "W", "Y"}
        output: dict[tuple[int, str, str], dict[str, Any]] = {}
        for position, wild_type in zip(
            self.context.mutable_positions, self.context.wild_type_residues, strict=True
        ):
            wild_type_values = {
                name: table[wild_type] for name, table in self.properties.items()
            }
            for mutant in CANONICAL_RESIDUES:
                if mutant == wild_type:
                    continue
                deltas = {
                    name: table[mutant] - table[wild_type]
                    for name, table in self.properties.items()
                }
                special_flags = []
                for residue, label in ((wild_type, "from"), (mutant, "to")):
                    if residue in {"G", "P", "C"}:
                        special_flags.append(f"{label}_{residue}:{position}")
                output[(position, wild_type, mutant)] = {
                    "mutation": f"{wild_type}{position}{mutant}",
                    "deltas": deltas,
                    "wild_type_values": wild_type_values,
                    "mutant_values": {
                        name: table[mutant] for name, table in self.properties.items()
                    },
                    "special_flags": special_flags,
                    "normalized_absolute_deltas": [
                        abs(deltas[name]) / self.scales[name] for name in sorted(deltas)
                    ],
                    "aromatic_count_delta": int(mutant in aromatic) - int(wild_type in aromatic),
                }
        return output

    def site_table(self) -> dict[str, Any]:
        positions: dict[str, Any] = {}
        for position, wild_type in zip(
            self.context.mutable_positions, self.context.wild_type_residues, strict=True
        ):
            positions[str(position)] = {
                "wild_type": wild_type,
                "substitutions": {
                    mutant: copy.deepcopy(self._substitutions[(position, wild_type, mutant)])
                    for mutant in CANONICAL_RESIDUES
                    if mutant != wild_type
                },
            }
        return {
            "channel": self.channel,
            "source_id": self.source_id,
            "resource_sha256": self.resource_sha256,
            "parameter_set_id": self.parameter_set_id,
            "property_accessions": self.accessions,
            "assay_pH": self.context.assay_conditions.pH,
            "positions": positions,
        }

    def evaluate(self, variant: Variant, *, round_id: int, **_kwargs: Any) -> Evidence:
        site_features: dict[str, Any] = {}
        normalized_changes: list[float] = []
        special_flags: list[str] = []
        mass_delta = 0.0
        hydropathy_delta = 0.0
        charge_delta = 0.0
        aromatic_count_delta = 0
        has_mass = "residue_mass" in self.properties
        has_hydropathy = "hydropathy" in self.properties
        has_charge = "nominal_charge" in self.properties
        for position, wild_type, mutant in zip(
            self.context.mutable_positions,
            self.context.wild_type_residues,
            variant.variant,
            strict=True,
        ):
            if wild_type == mutant:
                continue
            row = copy.deepcopy(self._substitutions[(position, wild_type, mutant)])
            site_features[str(position)] = {
                "mutation": row["mutation"],
                "deltas": row["deltas"],
                "wild_type_values": row["wild_type_values"],
                "mutant_values": row["mutant_values"],
            }
            normalized_changes.extend(row["normalized_absolute_deltas"])
            special_flags.extend(row["special_flags"])
            deltas = row["deltas"]
            if has_mass:
                mass_delta += float(deltas["residue_mass"])
            if has_hydropathy:
                hydropathy_delta += float(deltas["hydropathy"])
            if has_charge:
                charge_delta += float(deltas["nominal_charge"])
            aromatic_count_delta += int(row["aromatic_count_delta"])

        mean_change = float(np.mean(normalized_changes)) if normalized_changes else 0.0
        conservativeness = 1.0 / (1.0 + mean_change)
        global_sequence_deltas: dict[str, float] = {
            "aromatic_fraction_delta": aromatic_count_delta / self._sequence_length,
        }
        if has_mass:
            global_sequence_deltas["molecular_weight_delta_da"] = mass_delta
        if has_hydropathy:
            global_sequence_deltas["mean_hydropathy_delta"] = (
                hydropathy_delta / self._sequence_length
            )
        if has_charge:
            global_sequence_deltas["mean_nominal_charge_delta"] = (
                charge_delta / self._sequence_length
            )
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
        return Evidence(
            evidence_id=compact_static_evidence_id(
                self.channel,
                variant.variant_id,
            ),
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
