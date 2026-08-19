from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _read_fasta(path: Path) -> str:
    lines = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith(">"):
                continue
            lines.append(line)
    sequence = "".join(lines).upper()
    if not sequence:
        raise ValueError(f"Reference FASTA is empty: {path}")
    return sequence


@dataclass(frozen=True)
class AssayConditions:
    pH: float | None = None
    temperature_c: float | None = None
    ionic_strength_mM: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> AssayConditions:
        raw = dict(value or {})
        known = {
            key: raw.pop(key, None)
            for key in ("pH", "temperature_c", "ionic_strength_mM")
        }
        return cls(**known, extras=raw)


@dataclass(frozen=True)
class StructureResource:
    resource_id: str
    path: Path
    format: str = "pdb"
    chain: str | None = None
    partner_chains: tuple[str, ...] = ()
    assembly_id: str | None = None
    sha256: str | None = None
    residue_map: dict[int, tuple[str, int, str]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> StructureResource:
        mapping: dict[int, tuple[str, int, str]] = {}
        for key, value in dict(raw.get("residue_map", {})).items():
            if isinstance(value, dict):
                mapping[int(key)] = (
                    str(value.get("chain", raw.get("chain") or "")),
                    int(value.get("residue", key)),
                    str(value.get("insertion_code", "")),
                )
            else:
                mapping[int(key)] = (
                    str(raw.get("chain") or ""),
                    int(value),
                    "",
                )
        return cls(
            resource_id=str(raw["resource_id"]),
            path=Path(raw["path"]),
            format=str(raw.get("format", Path(raw["path"]).suffix.lstrip(".") or "pdb")),
            chain=str(raw["chain"]) if raw.get("chain") is not None else None,
            partner_chains=tuple(str(item) for item in raw.get("partner_chains", ())),
            assembly_id=(str(raw["assembly_id"]) if raw.get("assembly_id") is not None else None),
            sha256=str(raw["sha256"]) if raw.get("sha256") else None,
            residue_map=mapping,
        )

    def validate(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"Structure resource does not exist: {self.path}")
        if self.sha256:
            actual = hashlib.sha256(self.path.read_bytes()).hexdigest()
            if actual != self.sha256:
                raise ValueError(
                    f"Structure resource hash mismatch for {self.resource_id}: {actual}"
                )


@dataclass(frozen=True)
class ProteinTaskContext:
    task_id: str
    protein_id: str
    assay_id: str
    full_sequence: str
    mutable_positions: tuple[int, ...]
    wild_type_residues: tuple[str, ...]
    position_to_variant_index: dict[int, int]
    position_to_sequence_index: dict[int, int]
    numbering_scheme: str = "task"
    sequence_mode: str = "full_length"
    assay_conditions: AssayConditions = field(default_factory=AssayConditions)
    structure_resources: tuple[StructureResource, ...] = ()

    @classmethod
    def from_task(cls, task: Any) -> ProteinTaskContext:
        positions = tuple(int(item) for item in task.mutable_positions)
        wild_type = tuple(str(task.wild_type_sites).upper())
        if len(positions) != len(wild_type):
            raise ValueError("mutable_positions and wild_type_sites must have equal length")
        if len(set(positions)) != len(positions):
            raise ValueError("mutable_positions must be unique")
        if any(item not in CANONICAL_AA for item in wild_type):
            raise ValueError("wild_type_sites contains non-canonical amino acids")

        reference_path = getattr(task, "reference_sequence_path", None)
        explicit_sequence = getattr(task, "reference_sequence", None)
        if explicit_sequence and reference_path:
            raise ValueError("Configure reference_sequence or reference_sequence_path, not both")
        if reference_path:
            full_sequence = _read_fasta(Path(reference_path))
        elif explicit_sequence:
            full_sequence = str(explicit_sequence).replace(" ", "").upper()
        else:
            full_sequence = "".join(wild_type)

        if any(item not in CANONICAL_AA for item in full_sequence):
            raise ValueError("Reference sequence contains non-canonical amino acids")

        if len(full_sequence) == len(wild_type) and max(positions, default=0) > len(full_sequence):
            sequence_mode = "compact_sites"
            position_to_sequence_index = {
                position: index for index, position in enumerate(positions)
            }
        else:
            sequence_mode = "full_length"
            offset = int(getattr(task, "sequence_position_offset", 1))
            position_to_sequence_index = {
                position: position - offset for position in positions
            }
            for position, index in position_to_sequence_index.items():
                if not 0 <= index < len(full_sequence):
                    raise ValueError(
                        f"Mutable position {position} cannot be mapped to reference sequence"
                    )
                expected = wild_type[positions.index(position)]
                if full_sequence[index] != expected:
                    raise ValueError(
                        f"Reference residue mismatch at task position {position}: "
                        f"expected {expected}, found {full_sequence[index]}"
                    )

        resources = tuple(
            item if isinstance(item, StructureResource) else StructureResource.from_mapping(item)
            for item in getattr(task, "structure_resources", ())
        )
        return cls(
            task_id=str(task.task_id),
            protein_id=str(task.protein_id),
            assay_id=str(task.assay_id),
            full_sequence=full_sequence,
            mutable_positions=positions,
            wild_type_residues=wild_type,
            position_to_variant_index={
                position: index for index, position in enumerate(positions)
            },
            position_to_sequence_index=position_to_sequence_index,
            numbering_scheme=str(getattr(task, "numbering_scheme", "task")),
            sequence_mode=sequence_mode,
            assay_conditions=AssayConditions.from_mapping(
                getattr(task, "assay_conditions", None)
            ),
            structure_resources=resources,
        )

    def for_open_design(
        self, positions: tuple[int, ...] | None = None
    ) -> ProteinTaskContext:
        """Return a context whose compact code is the complete reference sequence."""

        if self.sequence_mode != "full_length":
            raise ValueError("open_design requires a complete full-length reference sequence")
        offset = min(self.position_to_sequence_index, default=1)
        # Recover the configured task offset from any valid position/index pair.
        if self.position_to_sequence_index:
            first_position = next(iter(self.position_to_sequence_index))
            offset = first_position - self.position_to_sequence_index[first_position]
        requested = positions or tuple(
            range(offset, offset + len(self.full_sequence))
        )
        requested = tuple(int(item) for item in requested)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("open_design positions must be non-empty and unique")
        mapping = {position: position - offset for position in requested}
        invalid = [position for position, index in mapping.items() if not 0 <= index < len(self.full_sequence)]
        if invalid:
            raise ValueError(f"open_design positions are outside the reference sequence: {invalid}")
        return ProteinTaskContext(
            task_id=self.task_id,
            protein_id=self.protein_id,
            assay_id=self.assay_id,
            full_sequence=self.full_sequence,
            mutable_positions=requested,
            wild_type_residues=tuple(self.full_sequence[mapping[item]] for item in requested),
            position_to_variant_index={
                position: index for index, position in enumerate(requested)
            },
            position_to_sequence_index=mapping,
            numbering_scheme=self.numbering_scheme,
            sequence_mode="full_length",
            assay_conditions=self.assay_conditions,
            structure_resources=self.structure_resources,
        )

    @property
    def wild_type_code(self) -> str:
        return "".join(self.wild_type_residues)

    @property
    def context_id(self) -> str:
        payload = "|".join(
            (
                self.task_id,
                self.protein_id,
                self.assay_id,
                self.full_sequence,
                ",".join(map(str, self.mutable_positions)),
                self.numbering_scheme,
            )
        )
        return f"protein-context:{hashlib.sha256(payload.encode()).hexdigest()[:16]}"

    def full_sequence_for_variant(self, compact_variant: str) -> str:
        if len(compact_variant) != len(self.mutable_positions):
            raise ValueError("Variant length does not match mutable_positions")
        if self.sequence_mode == "compact_sites":
            return compact_variant
        output = list(self.full_sequence)
        for position, residue in zip(self.mutable_positions, compact_variant, strict=True):
            output[self.position_to_sequence_index[position]] = residue
        return "".join(output)

    def task_description(self, objective: str) -> str:
        positions = ",".join(str(item) for item in self.mutable_positions)
        return (
            f"{objective} assay fitness for protein {self.protein_id} "
            f"at configured mutable positions {positions}"
        )
