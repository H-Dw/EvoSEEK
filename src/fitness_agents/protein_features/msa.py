from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fitness_agents.config import KnowledgeProviderConfig
from fitness_agents.contracts.schemas import Evidence, Variant

from .context import CANONICAL_AA, ProteinTaskContext


def _read_alignment(path: Path) -> tuple[str, ...]:
    sequences: list[str] = []
    current: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
                continue
            current.append(line)
    if current:
        sequences.append("".join(current))
    if not sequences:
        raise ValueError(f"MSA contains no sequences: {path}")

    # A3M lower-case characters are insertions relative to the query and are removed.
    cleaned = tuple(
        "".join(character for character in sequence if not character.islower()).upper()
        for sequence in sequences
    )
    length = len(cleaned[0])
    if any(len(sequence) != length for sequence in cleaned):
        raise ValueError("MSA rows do not have a common aligned length")
    return cleaned


def _identity(left: str, right: str) -> float:
    comparable = [(a, b) for a, b in zip(left, right, strict=True) if a != "-" and b != "-"]
    if not comparable:
        return 0.0
    return sum(a == b for a, b in comparable) / len(comparable)


@dataclass(frozen=True)
class MSAProfile:
    query: str
    query_index_to_column: tuple[int, ...]
    frequencies: tuple[dict[str, float], ...]
    pair_frequencies: dict[tuple[int, int], dict[str, float]]
    coverage: tuple[float, ...]
    gap_fraction: tuple[float, ...]
    entropy: tuple[float, ...]
    sequence_count: int
    neff: float
    resource_sha256: str
    settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_index_to_column": list(self.query_index_to_column),
            "frequencies": list(self.frequencies),
            "pair_frequencies": {
                f"{left},{right}": values
                for (left, right), values in self.pair_frequencies.items()
            },
            "coverage": list(self.coverage),
            "gap_fraction": list(self.gap_fraction),
            "entropy": list(self.entropy),
            "sequence_count": self.sequence_count,
            "neff": self.neff,
            "resource_sha256": self.resource_sha256,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MSAProfile:
        return cls(
            query=str(raw["query"]),
            query_index_to_column=tuple(int(item) for item in raw["query_index_to_column"]),
            frequencies=tuple(dict(item) for item in raw["frequencies"]),
            pair_frequencies={
                tuple(int(item) for item in key.split(",")): dict(values)
                for key, values in raw.get("pair_frequencies", {}).items()
            },
            coverage=tuple(float(item) for item in raw["coverage"]),
            gap_fraction=tuple(float(item) for item in raw["gap_fraction"]),
            entropy=tuple(float(item) for item in raw["entropy"]),
            sequence_count=int(raw["sequence_count"]),
            neff=float(raw["neff"]),
            resource_sha256=str(raw["resource_sha256"]),
            settings=dict(raw["settings"]),
        )


def _run_mmseqs_search(
    context: ProteinTaskContext,
    config: KnowledgeProviderConfig,
    cache_dir: Path,
) -> Path:
    executable = str(config.options.get("mmseqs_executable", "mmseqs"))
    database = config.options.get("mmseqs_database")
    if not database:
        raise ValueError("msa_profile requires resource_path or options.mmseqs_database")
    cache_dir.mkdir(parents=True, exist_ok=True)
    query_path = cache_dir / "query.fasta"
    result_path = cache_dir / "hits.tsv"
    temporary_path = cache_dir / "mmseqs_tmp"
    alignment_path = cache_dir / "search_alignment.fasta"
    query_path.write_text(
        f">{context.protein_id}\n{context.full_sequence}\n", encoding="utf-8"
    )
    command = [
        executable,
        "easy-search",
        str(query_path),
        str(database),
        str(result_path),
        str(temporary_path),
        "--format-output",
        "target,evalue,qcov,tcov,fident,qstart,qend,qaln,taln",
    ]
    for item in config.options.get("mmseqs_args", ()):
        command.append(str(item))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(config.options.get("timeout_seconds", 1800)),
    )
    (cache_dir / "mmseqs.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (cache_dir / "mmseqs.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"MMseqs2 search failed with exit code {completed.returncode}")

    rows = [f">query\n{context.full_sequence}"]
    for index, raw in enumerate(result_path.read_text(encoding="utf-8").splitlines()):
        fields = raw.split("\t")
        if len(fields) != 9:
            continue
        target, _evalue, _qcov, _tcov, _fident, qstart, qend, qaln, taln = fields
        start = max(int(qstart) - 1, 0)
        end = min(int(qend), len(context.full_sequence))
        if end <= start or len(qaln) != len(taln):
            continue
        projected = ["-"] * len(context.full_sequence)
        query_index = start
        for query_character, target_character in zip(qaln, taln, strict=True):
            if query_character == "-":
                continue
            if query_index >= len(projected):
                break
            projected[query_index] = target_character
            query_index += 1
        rows.append(f">{target or f'hit_{index}'}\n{''.join(projected)}")
    alignment_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return alignment_path


def build_profile(
    alignment_path: Path,
    *,
    mutable_columns: tuple[int, ...],
    identity_threshold: float,
    pseudocount: float,
    minimum_sequence_coverage: float,
    maximum_sequence_gap_fraction: float,
) -> MSAProfile:
    unfiltered = _read_alignment(alignment_path)
    query_non_gap = max(sum(character != "-" for character in unfiltered[0]), 1)
    sequences = tuple(
        sequence
        for index, sequence in enumerate(unfiltered)
        if index == 0
        or (
            sum(
                query_character != "-" and sequence_character in CANONICAL_AA
                for query_character, sequence_character in zip(
                    unfiltered[0], sequence, strict=True
                )
            )
            / query_non_gap
            >= minimum_sequence_coverage
            and sequence.count("-") / len(sequence) <= maximum_sequence_gap_fraction
        )
    )
    query = sequences[0].replace("-", "")
    aligned_query = sequences[0]
    weights = []
    for left in sequences:
        neighbors = sum(
            _identity(left, right) >= identity_threshold for right in sequences
        )
        weights.append(1.0 / max(neighbors, 1))
    neff = float(sum(weights))
    total_weight = max(neff, 1e-12)
    frequencies: list[dict[str, float]] = []
    coverage: list[float] = []
    gaps: list[float] = []
    entropies: list[float] = []
    alphabet = tuple(sorted(CANONICAL_AA))
    for column in range(len(aligned_query)):
        counts = {residue: pseudocount for residue in alphabet}
        observed_weight = 0.0
        for sequence, weight in zip(sequences, weights, strict=True):
            residue = sequence[column]
            if residue in CANONICAL_AA:
                counts[residue] += weight
                observed_weight += weight
        denominator = sum(counts.values())
        profile = {residue: counts[residue] / denominator for residue in alphabet}
        frequencies.append(profile)
        column_coverage = observed_weight / total_weight
        coverage.append(column_coverage)
        gaps.append(1.0 - column_coverage)
        entropies.append(
            -sum(value * math.log(value + 1e-12) for value in profile.values())
        )

    query_index_to_column = tuple(
        column for column, character in enumerate(aligned_query) if character != "-"
    )
    aligned_mutable_columns = tuple(query_index_to_column[index] for index in mutable_columns)
    pair_frequencies: dict[tuple[int, int], dict[str, float]] = {}
    for left_index, left in enumerate(aligned_mutable_columns):
        for right in aligned_mutable_columns[left_index + 1 :]:
            counts: dict[str, float] = {}
            for sequence, weight in zip(sequences, weights, strict=True):
                pair = f"{sequence[left]}{sequence[right]}"
                if all(item in CANONICAL_AA for item in pair):
                    counts[pair] = counts.get(pair, 0.0) + weight
            denominator = sum(counts.values()) + pseudocount * len(alphabet) ** 2
            pair_frequencies[(left, right)] = {
                f"{a}{b}": (counts.get(f"{a}{b}", 0.0) + pseudocount) / denominator
                for a in alphabet
                for b in alphabet
            }

    return MSAProfile(
        query=query,
        query_index_to_column=query_index_to_column,
        frequencies=tuple(frequencies),
        pair_frequencies=pair_frequencies,
        coverage=tuple(coverage),
        gap_fraction=tuple(gaps),
        entropy=tuple(entropies),
        sequence_count=len(sequences),
        neff=neff,
        resource_sha256=hashlib.sha256(alignment_path.read_bytes()).hexdigest(),
        settings={
            "identity_threshold": identity_threshold,
            "pseudocount": pseudocount,
            "minimum_sequence_coverage": minimum_sequence_coverage,
            "maximum_sequence_gap_fraction": maximum_sequence_gap_fraction,
            "unfiltered_sequence_count": len(unfiltered),
        },
    )


class MSAProfileProvider:
    channel = "conservation"

    def __init__(
        self,
        context: ProteinTaskContext,
        config: KnowledgeProviderConfig,
        *,
        parameter_set_id: str,
        cache_dir: Path,
    ) -> None:
        if context.sequence_mode != "full_length":
            raise ValueError("MSA analysis requires a full-length/domain reference sequence")
        self.context = context
        self.config = config
        self.parameter_set_id = parameter_set_id
        required = {
            "identity_threshold",
            "pseudocount",
            "minimum_neff",
            "minimum_sequence_coverage",
            "maximum_sequence_gap_fraction",
        }
        missing = sorted(required.difference(config.options))
        if missing:
            raise ValueError(f"msa_profile options are required: {missing}")
        identity_threshold = float(config.options["identity_threshold"])
        pseudocount = float(config.options["pseudocount"])
        minimum_sequence_coverage = float(config.options["minimum_sequence_coverage"])
        maximum_sequence_gap_fraction = float(
            config.options["maximum_sequence_gap_fraction"]
        )
        if not 0 < identity_threshold <= 1 or pseudocount <= 0:
            raise ValueError("MSA identity_threshold and pseudocount are outside valid ranges")
        if not 0 <= minimum_sequence_coverage <= 1:
            raise ValueError("minimum_sequence_coverage must be in [0, 1]")
        if not 0 <= maximum_sequence_gap_fraction <= 1:
            raise ValueError("maximum_sequence_gap_fraction must be in [0, 1]")
        resource_sha256 = None
        if config.resource_path is not None:
            resource_sha256 = hashlib.sha256(Path(config.resource_path).read_bytes()).hexdigest()
        settings = {
            "reference_sha256": hashlib.sha256(context.full_sequence.encode()).hexdigest(),
            "resource_path": str(config.resource_path) if config.resource_path else None,
            "resource_sha256": resource_sha256,
            "options": config.options,
        }
        cache_key = hashlib.sha256(
            json.dumps(settings, sort_keys=True, default=str).encode()
        ).hexdigest()[:20]
        provider_cache = cache_dir / "msa" / cache_key
        provider_cache.mkdir(parents=True, exist_ok=True)
        profile_path = provider_cache / "profile.json"
        if profile_path.is_file():
            self.profile = MSAProfile.from_dict(
                json.loads(profile_path.read_text(encoding="utf-8"))
            )
            self.cache_status = "hit"
        else:
            alignment_path = (
                Path(config.resource_path)
                if config.resource_path is not None
                else _run_mmseqs_search(context, config, provider_cache)
            )
            columns = tuple(
                context.position_to_sequence_index[position]
                for position in context.mutable_positions
            )
            self.profile = build_profile(
                alignment_path,
                mutable_columns=columns,
                identity_threshold=identity_threshold,
                pseudocount=pseudocount,
                minimum_sequence_coverage=minimum_sequence_coverage,
                maximum_sequence_gap_fraction=maximum_sequence_gap_fraction,
            )
            if self.profile.query != context.full_sequence:
                raise ValueError(
                    "MSA query sequence does not match the configured reference sequence"
                )
            profile_path.write_text(
                json.dumps(self.profile.to_dict(), sort_keys=True, indent=2),
                encoding="utf-8",
            )
            self.cache_status = "miss"
        self.profile_path = profile_path
        self.manifest_path = provider_cache / "prepare_manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "provider": type(self).__name__,
                    "provider_version": "v1",
                    "cache_key": cache_key,
                    "cache_status": self.cache_status,
                    "profile_path": str(profile_path),
                    "profile_resource_sha256": self.profile.resource_sha256,
                    "parameter_set_id": parameter_set_id,
                    "settings": settings,
                    "profile_settings": self.profile.settings,
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )

    def evaluate(self, variant: Variant, *, round_id: int, **_kwargs: Any) -> Evidence:
        epsilon = 1e-12
        single_terms: list[float] = []
        site_features: dict[str, Any] = {}
        variant_by_position = dict(
            zip(self.context.mutable_positions, variant.variant, strict=True)
        )
        for position, wild_type in zip(
            self.context.mutable_positions, self.context.wild_type_residues, strict=True
        ):
            mutant = variant_by_position[position]
            sequence_index = self.context.position_to_sequence_index[position]
            column = self.profile.query_index_to_column[sequence_index]
            profile = self.profile.frequencies[column]
            log_odds = math.log((profile[mutant] + epsilon) / (profile[wild_type] + epsilon))
            single_terms.append(log_odds)
            site_features[str(position)] = {
                "column": column,
                "wild_type_frequency": profile[wild_type],
                "mutant_frequency": profile[mutant],
                "log_odds_vs_wild_type": log_odds,
                "entropy": self.profile.entropy[column],
                "coverage": self.profile.coverage[column],
                "gap_fraction": self.profile.gap_fraction[column],
            }
        pair_terms: list[float] = []
        for (left_column, right_column), profile in self.profile.pair_frequencies.items():
            left_position = next(
                position
                for position, sequence_index in self.context.position_to_sequence_index.items()
                if self.profile.query_index_to_column[sequence_index] == left_column
            )
            right_position = next(
                position
                for position, sequence_index in self.context.position_to_sequence_index.items()
                if self.profile.query_index_to_column[sequence_index] == right_column
            )
            mutant_pair = variant_by_position[left_position] + variant_by_position[right_position]
            wild_type_pair = (
                self.context.wild_type_residues[
                    self.context.position_to_variant_index[left_position]
                ]
                + self.context.wild_type_residues[
                    self.context.position_to_variant_index[right_position]
                ]
            )
            pair_terms.append(
                math.log((profile[mutant_pair] + epsilon) / (profile[wild_type_pair] + epsilon))
            )
        independent_score = float(sum(single_terms))
        pairwise_score = float(sum(pair_terms))
        raw_score = independent_score + pairwise_score
        minimum_neff = float(self.config.options["minimum_neff"])
        quality = "ok" if self.profile.neff >= minimum_neff else "degraded"
        warnings = [] if quality == "ok" else ["msa_neff_below_configured_minimum"]
        statement = (
            f"MSA log-odds={raw_score:.3f} (independent={independent_score:.3f}, "
            f"pairwise-frequency={pairwise_score:.3f}); Neff={self.profile.neff:.2f}; "
            "evolutionary prior, not assay fitness"
        )
        raw_features = {
            "sites": site_features,
            "independent_log_odds": independent_score,
            "pairwise_frequency_log_odds": pairwise_score,
            "sequence_count": self.profile.sequence_count,
            "neff": self.profile.neff,
            "cache_status": self.cache_status,
        }
        identity = json.dumps(
            {
                "variant_id": variant.variant_id,
                "round_id": round_id,
                "context_id": self.context.context_id,
                "profile_sha256": self.profile.resource_sha256,
                "raw_features": raw_features,
            },
            sort_keys=True,
        )
        return Evidence(
            evidence_id=f"ev:conservation:{hashlib.sha256(identity.encode()).hexdigest()[:16]}",
            variant_id=variant.variant_id,
            channel=self.channel,
            statement=statement,
            score=raw_score,
            source_id=f"msa_profile:{self.profile.resource_sha256[:16]}",
            confidence=0.0,
            round_id=round_id,
            evidence_type="evolutionary_profile",
            raw_features=raw_features,
            quality_status=quality,
            applicability="in_domain" if quality == "ok" else "partial",
            calibrated=False,
            contributes_to_selection=False,
            warnings=tuple(warnings),
            provenance={
                "provider": type(self).__name__,
                "provider_version": "v1",
                "profile_path": str(self.profile_path),
                "prepare_manifest_path": str(self.manifest_path),
                "resource_sha256": self.profile.resource_sha256,
                "parameter_set_id": self.parameter_set_id,
                "context_id": self.context.context_id,
            },
        )
