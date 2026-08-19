from __future__ import annotations

import hashlib
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fitness_agents.config import KnowledgeProviderConfig
from fitness_agents.contracts.schemas import Evidence, Variant

from .context import ProteinTaskContext, StructureResource
from .substitution_store import compact_static_evidence_id, compact_structure_site

VDW_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
POSITIVE_ATOMS = {("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2"), ("HIS", "ND1"), ("HIS", "NE2")}
NEGATIVE_ATOMS = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}


@dataclass(frozen=True)
class Atom:
    name: str
    element: str
    coordinate: np.ndarray


@dataclass(frozen=True)
class Residue:
    name: str
    chain: str
    number: int
    insertion_code: str
    atoms: tuple[Atom, ...]

    @property
    def key(self) -> tuple[str, int, str]:
        return self.chain, self.number, self.insertion_code


def _parse_pdb(path: Path) -> tuple[Residue, ...]:
    records: dict[tuple[str, int, str, str], list[Atom]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line.startswith("HETATM") and line[17:20].strip() not in {"MSE"}:
            continue
        alternate = line[16:17].strip()
        if alternate not in {"", "A"}:
            continue
        try:
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain = line[21:22].strip()
            number = int(line[22:26])
            insertion_code = line[26:27].strip()
            coordinate = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
        except (ValueError, IndexError) as error:
            raise ValueError(f"Malformed PDB atom line in {path}: {line!r}") from error
        element = line[76:78].strip().upper() or atom_name[0].upper()
        key = (chain, number, insertion_code, residue_name)
        records.setdefault(key, []).append(Atom(atom_name, element, coordinate))
    if not records:
        raise ValueError(f"No protein atoms found in structure: {path}")
    return tuple(
        Residue(name, chain, number, insertion_code, tuple(atoms))
        for (chain, number, insertion_code, name), atoms in records.items()
    )


def _parse_mmcif(path: Path) -> tuple[Residue, ...]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: dict[tuple[str, int, str, str], list[Atom]] = {}
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        headers = []
        while index < len(lines) and lines[index].strip().startswith("_"):
            headers.append(lines[index].strip())
            index += 1
        if not headers or not all(item.startswith("_atom_site.") for item in headers):
            continue
        header_index = {name: offset for offset, name in enumerate(headers)}

        def field(
            row: list[str],
            *names: str,
            default: str = "",
            _header_index: dict[str, int] = header_index,
        ) -> str:
            for name in names:
                if name in _header_index:
                    value = row[_header_index[name]]
                    return "" if value in {".", "?"} else value
            return default

        tokens: list[str] = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("#"):
                index += 1
                if stripped.startswith("#"):
                    break
                continue
            if stripped == "loop_" or stripped.startswith(("_", "data_")):
                break
            tokens.extend(shlex.split(stripped, posix=True))
            index += 1
            while len(tokens) >= len(headers):
                row, tokens = tokens[: len(headers)], tokens[len(headers) :]
                group = field(row, "_atom_site.group_PDB")
                residue_name = field(row, "_atom_site.label_comp_id", "_atom_site.auth_comp_id")
                if group not in {"ATOM", "HETATM"} or (
                    group == "HETATM" and residue_name != "MSE"
                ):
                    continue
                alternate = field(row, "_atom_site.label_alt_id")
                if alternate not in {"", "A"}:
                    continue
                try:
                    atom_name = field(row, "_atom_site.auth_atom_id", "_atom_site.label_atom_id")
                    chain = field(row, "_atom_site.auth_asym_id", "_atom_site.label_asym_id")
                    number = int(
                        field(row, "_atom_site.auth_seq_id", "_atom_site.label_seq_id")
                    )
                    insertion_code = field(row, "_atom_site.pdbx_PDB_ins_code")
                    coordinate = np.asarray(
                        [
                            float(field(row, "_atom_site.Cartn_x")),
                            float(field(row, "_atom_site.Cartn_y")),
                            float(field(row, "_atom_site.Cartn_z")),
                        ],
                        dtype=float,
                    )
                except (ValueError, IndexError) as error:
                    raise ValueError(f"Malformed mmCIF atom row in {path}: {row!r}") from error
                element = field(row, "_atom_site.type_symbol", default=atom_name[0]).upper()
                key = (chain, number, insertion_code, residue_name)
                records.setdefault(key, []).append(Atom(atom_name, element, coordinate))
        break
    if not records:
        raise ValueError(f"No protein atom_site loop found in structure: {path}")
    return tuple(
        Residue(name, chain, number, insertion_code, tuple(atoms))
        for (chain, number, insertion_code, name), atoms in records.items()
    )


def _minimum_distance(left: Residue, right: Residue) -> float:
    left_coordinates = np.stack([atom.coordinate for atom in left.atoms])
    right_coordinates = np.stack([atom.coordinate for atom in right.atoms])
    differences = left_coordinates[:, None, :] - right_coordinates[None, :, :]
    return float(np.min(np.linalg.norm(differences, axis=2)))


def _atom(residue: Residue, name: str) -> np.ndarray | None:
    for item in residue.atoms:
        if item.name == name:
            return item.coordinate
    return None


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    first = b - a
    second = c - b
    third = d - c
    first_normal = np.cross(first, second)
    second_normal = np.cross(second, third)
    if np.linalg.norm(first_normal) == 0 or np.linalg.norm(second_normal) == 0:
        return float("nan")
    first_normal /= np.linalg.norm(first_normal)
    second_normal /= np.linalg.norm(second_normal)
    direction = second / max(np.linalg.norm(second), 1e-12)
    return float(math.degrees(math.atan2(np.dot(np.cross(first_normal, second_normal), direction), np.dot(first_normal, second_normal))))


def _secondary_structure(phi: float | None, psi: float | None) -> str:
    if phi is None or psi is None or math.isnan(phi) or math.isnan(psi):
        return "unknown"
    if -100 <= phi <= -30 and -80 <= psi <= -5:
        return "alpha_like"
    if phi <= -80 and (psi >= 70 or psi <= -120):
        return "beta_like"
    return "coil_like"


def _sphere_points(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / count)
    theta = math.pi * (1.0 + 5.0**0.5) * indices
    return np.stack(
        (np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)),
        axis=1,
    )


def _residue_sasa(
    target: Residue,
    residues: tuple[Residue, ...],
    *,
    probe_radius: float,
    sphere_point_count: int,
) -> float:
    all_atoms = tuple(atom for residue in residues for atom in residue.atoms if atom.element != "H")
    points = _sphere_points(sphere_point_count)
    area = 0.0
    for atom in target.atoms:
        if atom.element == "H":
            continue
        radius = VDW_RADII.get(atom.element, 1.70) + probe_radius
        samples = atom.coordinate + radius * points
        exposed = np.ones(sphere_point_count, dtype=bool)
        for neighbor in all_atoms:
            if neighbor is atom:
                continue
            neighbor_radius = VDW_RADII.get(neighbor.element, 1.70) + probe_radius
            distances = np.linalg.norm(samples - neighbor.coordinate, axis=1)
            exposed &= distances >= neighbor_radius
            if not np.any(exposed):
                break
        area += float(np.count_nonzero(exposed) / sphere_point_count) * 4.0 * math.pi * radius**2
    return area


class StaticStructureProvider:
    """Coordinate-backed static environment evidence; it does not model mutant relaxation."""

    channel = "structure"

    def __init__(
        self,
        context: ProteinTaskContext,
        config: KnowledgeProviderConfig,
        *,
        parameter_set_id: str,
    ) -> None:
        self.context = context
        self.config = config
        self.parameter_set_id = parameter_set_id
        resource_id = config.options.get("resource_id")
        resources = [
            item for item in context.structure_resources
            if resource_id is None or item.resource_id == resource_id
        ]
        if not resources:
            raise ValueError("static_structure provider has no configured structure resource")
        self.resource: StructureResource = resources[0]
        self.resource.validate()
        structure_format = self.resource.format.lower()
        if structure_format in {"pdb", "ent"}:
            self.residues = _parse_pdb(self.resource.path)
        elif structure_format in {"cif", "mmcif"}:
            self.residues = _parse_mmcif(self.resource.path)
        else:
            raise ValueError(f"Unsupported native structure format: {structure_format}")
        self.residue_lookup = {item.key: item for item in self.residues}
        required = {
            "contact_cutoff_angstrom",
            "interface_cutoff_angstrom",
            "hbond_cutoff_angstrom",
            "salt_bridge_cutoff_angstrom",
            "sasa_probe_radius_angstrom",
            "sasa_sphere_points",
            "dense_contact_count",
            "clash_distance_fraction",
            "disulfide_sg_cutoff_angstrom",
        }
        missing = sorted(required.difference(config.options))
        if missing:
            raise ValueError(f"static_structure options are required: {missing}")
        self.contact_cutoff = float(config.options["contact_cutoff_angstrom"])
        self.interface_cutoff = float(config.options["interface_cutoff_angstrom"])
        self.hbond_cutoff = float(config.options["hbond_cutoff_angstrom"])
        self.salt_bridge_cutoff = float(config.options["salt_bridge_cutoff_angstrom"])
        self.probe_radius = float(config.options["sasa_probe_radius_angstrom"])
        self.sphere_point_count = int(config.options["sasa_sphere_points"])
        self.clash_distance_fraction = float(config.options["clash_distance_fraction"])
        self.disulfide_sg_cutoff = float(config.options["disulfide_sg_cutoff_angstrom"])
        mapping_errors = []
        for position, expected in zip(
            context.mutable_positions, context.wild_type_residues, strict=True
        ):
            residue = self.residue_lookup.get(self._mapped_key(position))
            if residue is None:
                mapping_errors.append(f"{position}:missing")
            elif THREE_TO_ONE.get(residue.name) != expected:
                mapping_errors.append(
                    f"{position}:expected_{expected}_found_{residue.name}"
                )
        if mapping_errors:
            raise ValueError(
                "Structure residue mapping failed closed: " + ",".join(mapping_errors)
            )
        self.features = self._prepare_features()
        self.resource_sha256 = hashlib.sha256(self.resource.path.read_bytes()).hexdigest()

    def site_table(self) -> dict[str, Any]:
        positions: dict[str, Any] = {}
        for position, wild_type in zip(
            self.context.mutable_positions, self.context.wild_type_residues, strict=True
        ):
            feature = compact_structure_site(self.features[position])
            positions[str(position)] = {"wild_type": wild_type, **feature}
        return {
            "channel": self.channel,
            "resource_id": self.resource.resource_id,
            "resource_sha256": self.resource_sha256,
            "parameter_set_id": self.parameter_set_id,
            "positions": positions,
        }

    def _mapped_key(self, position: int) -> tuple[str, int, str]:
        if position in self.resource.residue_map:
            return self.resource.residue_map[position]
        return self.resource.chain or "", position, ""

    def _prepare_features(self) -> dict[int, dict[str, Any]]:
        output: dict[int, dict[str, Any]] = {}
        by_chain: dict[str, list[Residue]] = {}
        for residue in self.residues:
            by_chain.setdefault(residue.chain, []).append(residue)
        for chain_residues in by_chain.values():
            chain_residues.sort(key=lambda item: (item.number, item.insertion_code))

        for position in self.context.mutable_positions:
            key = self._mapped_key(position)
            residue = self.residue_lookup.get(key)
            if residue is None:
                output[position] = {
                    "status": "unavailable",
                    "warning": f"residue_mapping_not_found:{key}",
                }
                continue
            contacts = []
            interface_contacts = []
            for neighbor in self.residues:
                if neighbor.key == residue.key:
                    continue
                distance = _minimum_distance(residue, neighbor)
                if distance <= self.contact_cutoff:
                    contacts.append(
                        {"chain": neighbor.chain, "residue": neighbor.number, "distance": distance}
                    )
                if neighbor.chain != residue.chain and distance <= self.interface_cutoff:
                    interface_contacts.append(
                        {"chain": neighbor.chain, "residue": neighbor.number, "distance": distance}
                    )
            chain_items = by_chain[residue.chain]
            chain_index = chain_items.index(residue)
            previous = chain_items[chain_index - 1] if chain_index > 0 else None
            following = chain_items[chain_index + 1] if chain_index + 1 < len(chain_items) else None
            phi = None
            psi = None
            if previous is not None:
                c_previous, n, ca, c = (
                    _atom(previous, "C"), _atom(residue, "N"), _atom(residue, "CA"), _atom(residue, "C")
                )
                if all(item is not None for item in (c_previous, n, ca, c)):
                    phi = _dihedral(c_previous, n, ca, c)
            if following is not None:
                n, ca, c, n_following = (
                    _atom(residue, "N"), _atom(residue, "CA"), _atom(residue, "C"), _atom(following, "N")
                )
                if all(item is not None for item in (n, ca, c, n_following)):
                    psi = _dihedral(n, ca, c, n_following)
            hydrogen_bond_candidates = 0
            salt_bridge_candidates = 0
            clash_candidates = 0
            disulfide_candidates = 0
            for atom in residue.atoms:
                for neighbor in self.residues:
                    if neighbor.key == residue.key:
                        continue
                    for other in neighbor.atoms:
                        distance = float(np.linalg.norm(atom.coordinate - other.coordinate))
                        if {atom.element, other.element} == {"N", "O"} and distance <= self.hbond_cutoff:
                            hydrogen_bond_candidates += 1
                        charged_pair = (
                            ((residue.name, atom.name) in POSITIVE_ATOMS and (neighbor.name, other.name) in NEGATIVE_ATOMS)
                            or ((residue.name, atom.name) in NEGATIVE_ATOMS and (neighbor.name, other.name) in POSITIVE_ATOMS)
                        )
                        if charged_pair and distance <= self.salt_bridge_cutoff:
                            salt_bridge_candidates += 1
                        if (
                            atom.element != "H"
                            and other.element != "H"
                            and not (
                                residue.chain == neighbor.chain
                                and abs(residue.number - neighbor.number) <= 1
                            )
                            and distance
                            < self.clash_distance_fraction
                            * (
                                VDW_RADII.get(atom.element, 1.70)
                                + VDW_RADII.get(other.element, 1.70)
                            )
                        ):
                            clash_candidates += 1
                        if (
                            residue.name == "CYS"
                            and neighbor.name == "CYS"
                            and atom.name == "SG"
                            and other.name == "SG"
                            and distance <= self.disulfide_sg_cutoff
                        ):
                            disulfide_candidates += 1
            sasa = _residue_sasa(
                residue,
                self.residues,
                probe_radius=self.probe_radius,
                sphere_point_count=self.sphere_point_count,
            )
            missing_backbone_atoms = [
                atom_name
                for atom_name in ("N", "CA", "C", "O")
                if _atom(residue, atom_name) is None
            ]
            output[position] = {
                "status": "ok",
                "structure_chain": residue.chain,
                "structure_residue": residue.number,
                "structure_residue_name": residue.name,
                "insertion_code": residue.insertion_code,
                "sasa_angstrom2": sasa,
                "relative_sasa": sasa / MAX_ASA.get(residue.name, max(sasa, 1.0)),
                "maximum_asa_reference": "Tien2013_extended_tripeptide",
                "contact_count": len(contacts),
                "closest_contacts": sorted(contacts, key=lambda item: item["distance"])[:8],
                "interface_contact_count": len(interface_contacts),
                "interface_contacts": sorted(interface_contacts, key=lambda item: item["distance"])[:8],
                "phi_degrees": phi,
                "psi_degrees": psi,
                "secondary_structure": _secondary_structure(phi, psi),
                "secondary_structure_method": "coarse_phi_psi_v1",
                "hydrogen_bond_candidate_count": hydrogen_bond_candidates,
                "salt_bridge_candidate_count": salt_bridge_candidates,
                "disulfide_candidate_count": disulfide_candidates,
                "clash_candidate_count": clash_candidates,
                "missing_backbone_atoms": missing_backbone_atoms,
            }
        return output

    def evaluate(self, variant: Variant, *, round_id: int, **_kwargs: Any) -> Evidence:
        sites = {}
        missing = []
        for position, wild_type, mutant in zip(
            self.context.mutable_positions,
            self.context.wild_type_residues,
            variant.variant,
            strict=True,
        ):
            if wild_type == mutant:
                continue
            feature = compact_structure_site(self.features[position])
            feature["mutation"] = f"{wild_type}{position}{mutant}"
            feature["mutant_side_chain_not_modelled"] = True
            sites[str(position)] = feature
            if feature.get("status") != "ok":
                missing.append(position)
        quality = "degraded" if missing else "ok"
        static_risk_count = sum(
            int(item.get("contact_count", 0) >= int(self.config.options["dense_contact_count"]))
            + int(item.get("salt_bridge_candidate_count", 0) > 0)
            for item in sites.values()
            if item.get("status") == "ok"
        )
        raw_score = -float(static_risk_count)
        statement = (
            f"coordinate-backed static environment from {self.resource.resource_id} for "
            f"{len(sites)} mutated sites; "
            f"static context flags={static_risk_count}; mutant side chains were not relaxed; "
            "no folding or affinity claim"
        )
        warnings = ["mutant_side_chains_not_modelled", "static_structure_not_fitness"]
        if missing:
            warnings.append(f"unmapped_positions:{','.join(map(str, missing))}")
        raw_features = {
            "sites": sites,
            "static_context_flag_count": static_risk_count,
            "resource_id": self.resource.resource_id,
        }
        return Evidence(
            evidence_id=compact_static_evidence_id(
                self.channel,
                variant.variant_id,
                self.parameter_set_id,
                self.resource_sha256,
            ),
            variant_id=variant.variant_id,
            channel=self.channel,
            statement=statement,
            score=raw_score,
            source_id=f"structure:{self.resource.resource_id}",
            confidence=0.0,
            round_id=round_id,
            evidence_type="static_structure_descriptor",
            raw_features=raw_features,
            quality_status=quality,
            applicability="partial",
            calibrated=False,
            contributes_to_selection=False,
            warnings=tuple(warnings),
            provenance={
                "provider": type(self).__name__,
                "provider_version": "v1",
                "resource_id": self.resource.resource_id,
                "resource_path": str(self.resource.path),
                "resource_sha256": self.resource_sha256,
                "parameter_set_id": self.parameter_set_id,
                "context_id": self.context.context_id,
            },
        )


# TODO(phase6): add inverse-folding, FoldX and Rosetta adapters behind the same
# Evidence contract. They must remain opt-in, bounded and external to campaign truth.
