from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fitness_agents.contracts.schemas import FitnessObservation, Variant
from fitness_agents.mutation.notation import (
    InvalidMutationNotation,
    edits_from_tokens,
    format_canonical,
    parse_mutation_notation,
)

PUBLIC_REQUIRED = {
    "variant_id",
    "variant",
    "sequence",
    "mutation_notation",
    "mutation_count",
    "split_role",
}
ORACLE_REQUIRED = {"variant_id", "fitness", "split_role"}


@dataclass(frozen=True)
class DatasetBundle:
    initial_variants: list[Variant]
    initial_observations: list[FitnessObservation]
    validation_variants: list[Variant]
    validation_observations: list[FitnessObservation]
    oracle_pool: list[Variant]
    final_test: list[Variant]

    @property
    def public_candidates(self) -> list[Variant]:
        return [*self.oracle_pool]


@dataclass(frozen=True)
class FoldBundle:
    """Manifest-driven data view with capability-scoped tables."""

    root: Path
    fold_index: int
    consumer_role: str
    manifest: dict[str, object]
    fold_manifest: dict[str, object]
    observed: pd.DataFrame | None = None
    candidates: pd.DataFrame | None = None
    validation: pd.DataFrame | None = None
    queryable_labels: pd.DataFrame | None = None
    final_inputs: pd.DataFrame | None = None
    final_labels: pd.DataFrame | None = None
    quarantine: pd.DataFrame | None = None


def _row_to_variant(row: object) -> Variant:
    return Variant(
        variant_id=str(row.variant_id),
        variant=str(row.variant),
        sequence=str(row.sequence),
        mutation_notation=str(row.mutation_notation),
        mutation_count=int(row.mutation_count),
        split_role=str(row.split_role),
    )


def _cell(row: object, name: str) -> str | None:
    if not hasattr(row, name):
        return None
    value = getattr(row, name)
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    return text


def _mutation_notation_from_row(row: object) -> str:
    tokens_text = _cell(row, "mutation_tokens")
    if tokens_text is not None:
        parsed = json.loads(tokens_text)
        return format_canonical(edits_from_tokens(parsed))
    notation = _cell(row, "mutation_notation")
    if notation is not None:
        return format_canonical(parse_mutation_notation(notation))
    variant = _cell(row, "variant")
    if variant is None:
        return "WT"
    try:
        return format_canonical(parse_mutation_notation(variant))
    except InvalidMutationNotation:
        return variant


def variants_from_fold_frame(frame: pd.DataFrame, split_role: str) -> list[Variant]:
    required = {"variant_id", "variant", "sequence", "mutation_count"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Fold variant table missing columns: {sorted(missing)}")
    return [
        Variant(
            variant_id=str(row.variant_id),
            variant=str(row.variant),
            sequence=str(row.sequence),
            mutation_notation=_mutation_notation_from_row(row),
            mutation_count=int(row.mutation_count),
            split_role=split_role,
        )
        for row in frame.itertuples(index=False)
    ]


def _observations_from_fold_frame(
    frame: pd.DataFrame, variants: list[Variant]
) -> list[FitnessObservation]:
    if "target" not in frame:
        raise ValueError("Observed fold table is missing target")
    target_by_id = frame.set_index("variant_id")["target"].to_dict()
    return [
        FitnessObservation(
            variant_id=variant.variant_id,
            fitness=float(target_by_id[variant.variant_id]),
            split_role=variant.split_role,
            round_revealed=0,
            source="fold_initial_observed",
        )
        for variant in variants
    ]


def _visible_observations(
    variants: list[Variant], labels: pd.DataFrame, round_revealed: int
) -> list[FitnessObservation]:
    label_map = labels.set_index("variant_id")["fitness"].to_dict()
    return [
        FitnessObservation(
            variant_id=variant.variant_id,
            fitness=float(label_map[variant.variant_id]),
            split_role=variant.split_role,
            round_revealed=round_revealed,
            source="benchmark_initial" if round_revealed == 0 else "validation",
        )
        for variant in variants
    ]


def load_dataset_bundle(public_path: str | Path, oracle_path: str | Path) -> DatasetBundle:
    public = pd.read_csv(public_path)
    labels = pd.read_csv(oracle_path)
    if missing := PUBLIC_REQUIRED.difference(public.columns):
        raise ValueError(f"Public dataset missing columns: {sorted(missing)}")
    if hidden := {"fitness", "raw_fitness", "normalized_fitness"}.intersection(public.columns):
        raise ValueError(f"Public candidate table contains hidden labels: {sorted(hidden)}")
    if missing := ORACLE_REQUIRED.difference(labels.columns):
        raise ValueError(f"Oracle dataset missing columns: {sorted(missing)}")
    if public["variant_id"].duplicated().any() or labels["variant_id"].duplicated().any():
        raise ValueError("Variant IDs must be unique in public and oracle tables")
    if set(public["variant_id"]) != set(labels["variant_id"]):
        raise ValueError("Public and oracle tables have different variant IDs")
    if public.groupby("variant_id")["split_role"].nunique().max() != 1:
        raise ValueError("A variant cannot belong to more than one split")

    by_role: dict[str, list[Variant]] = {}
    for role, frame in public.groupby("split_role", sort=False):
        by_role[str(role)] = [_row_to_variant(row) for row in frame.itertuples(index=False)]
    required_roles = {"initial_observed", "validation", "oracle_pool", "final_test"}
    if missing := required_roles.difference(by_role):
        raise ValueError(f"Missing split roles: {sorted(missing)}")
    initial = by_role["initial_observed"]
    validation = by_role["validation"]
    return DatasetBundle(
        initial_variants=initial,
        initial_observations=_visible_observations(initial, labels, 0),
        validation_variants=validation,
        validation_observations=_visible_observations(validation, labels, -1),
        oracle_pool=by_role["oracle_pool"],
        final_test=by_role["final_test"],
    )


def _verify_fold_file(fold_root: Path, record: dict[str, object]) -> Path:
    path = fold_root / str(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Manifest output is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != record["sha256"]:
        raise ValueError(f"Manifest hash mismatch for {path}")
    return path


def load_fold_bundle(
    root: str | Path, fold_index: int, consumer_role: str = "agent"
) -> FoldBundle:
    """Load only the files authorized for one consumer role.

    The evaluator deliberately does not inherit oracle labels, and the oracle does not inherit
    evaluator labels. A caller needing multiple capabilities must create isolated processes/views.
    """

    allowed = {
        "agent",
        "controller",
        "oracle",
        "evaluator_inputs",
        "evaluator",
        "auditor",
    }
    if consumer_role not in allowed:
        raise ValueError(f"Unknown consumer_role {consumer_role!r}; expected one of {sorted(allowed)}")
    split_root = Path(root)
    manifest = json.loads((split_root / "manifest.public.json").read_text(encoding="utf-8"))
    fold_root = split_root / f"fold_{fold_index:02d}"
    fold_manifest_path = fold_root / "fold_manifest.json"
    fold_relative = fold_manifest_path.relative_to(split_root).as_posix()
    root_record = next(
        (
            item
            for item in manifest.get("fold_manifests", [])
            if str(item["path"]) == fold_relative
        ),
        None,
    )
    if root_record is None:
        raise ValueError(f"Root manifest does not authorize {fold_relative}")
    actual_fold_hash = hashlib.sha256(fold_manifest_path.read_bytes()).hexdigest()
    if actual_fold_hash != root_record["sha256"]:
        raise ValueError("Fold manifest hash does not match root manifest")
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    if int(fold_manifest["fold_index"]) != fold_index:
        raise ValueError("Fold manifest index does not match requested fold")
    records = {str(item["path"]): item for item in fold_manifest["files"]}

    def read(relative: str) -> pd.DataFrame:
        return pd.read_csv(_verify_fold_file(fold_root, records[relative]))

    observed = candidates = validation = queryable = final_inputs = final_labels = quarantine = None
    if consumer_role in {"agent", "controller", "auditor"}:
        observed = read("agent/initial_or_train_observed.csv.gz")
        candidates = read("agent/candidate_pool.csv.gz")
    if consumer_role in {"controller", "auditor"}:
        validation = read("controller/benchmark_validation.csv.gz")
    if consumer_role in {"oracle", "auditor"}:
        queryable = read("oracle/queryable_labels.csv.gz")
    if consumer_role in {"evaluator_inputs", "evaluator", "auditor"}:
        final_inputs = read("evaluator/final_test_inputs.csv.gz")
    if consumer_role in {"evaluator", "auditor"}:
        final_labels = read("evaluator/final_test_labels.csv.gz")
    if consumer_role == "auditor":
        quarantine = read("quarantine/excluded_variants.csv.gz")
    return FoldBundle(
        root=split_root,
        fold_index=fold_index,
        consumer_role=consumer_role,
        manifest=manifest,
        fold_manifest=fold_manifest,
        observed=observed,
        candidates=candidates,
        validation=validation,
        queryable_labels=queryable,
        final_inputs=final_inputs,
        final_labels=final_labels,
        quarantine=quarantine,
    )


def load_campaign_fold_bundle(root: str | Path, fold_index: int) -> DatasetBundle:
    """Convert an AL fold's agent view into the legacy runtime contract without hidden labels."""

    view = load_fold_bundle(root, fold_index, "agent")
    if view.observed is None or view.candidates is None:
        raise AssertionError("Agent fold view is incomplete")
    initial = variants_from_fold_frame(view.observed, "initial_observed")
    candidates = variants_from_fold_frame(view.candidates, "candidate_pool")
    if not initial:
        raise ValueError("Closed-loop fold has no initial observations")
    if not candidates:
        raise ValueError(
            "Closed-loop campaign requires a non-empty candidate pool; static OOD folds are not queryable"
        )
    return DatasetBundle(
        initial_variants=initial,
        initial_observations=_observations_from_fold_frame(view.observed, initial),
        validation_variants=[],
        validation_observations=[],
        oracle_pool=candidates,
        final_test=[],
    )


def load_fold_final_variants(root: str | Path, fold_index: int) -> list[Variant]:
    view = load_fold_bundle(root, fold_index, "evaluator_inputs")
    if view.final_inputs is None:
        raise AssertionError("Evaluator fold view is incomplete")
    variants = variants_from_fold_frame(view.final_inputs, "final_test")
    if not variants:
        raise ValueError("Fold final test is empty")
    return variants
