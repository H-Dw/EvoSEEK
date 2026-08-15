from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fitness_agents.contracts.schemas import (
    Evidence,
    FitnessObservation,
    Prediction,
    Variant,
)
from fitness_agents.data.gb1 import canonical_mutation_notation, variant_id

DEFAULT_RESIDUES_BY_SITE = (
    ("V", "A", "C"),
    ("D", "E", "W"),
    ("G", "A", "S"),
    ("V", "I", "F"),
)


def parse_args(default_config: str, *, remote: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config, help="Module-test YAML path")
    parser.add_argument("--output-dir", help="Override config output_dir")
    if remote:
        parser.add_argument(
            "--enable-remote",
            action="store_true",
            help="Use the real LLM API block after placeholders have been replaced",
        )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    with candidate.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Config must be a YAML mapping: {candidate}")
    value["_config_path"] = str(candidate.resolve())
    return value


def resolve_output(config: dict[str, Any], override: str | None) -> Path:
    value = Path(override or str(config["output_dir"]))
    output = value if value.is_absolute() else REPO_ROOT / value
    output.mkdir(parents=True, exist_ok=True)
    return output.resolve()


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


def write_result(output_dir: Path, module: str, payload: dict[str, Any]) -> Path:
    result = {
        "module": module,
        "status": "passed",
        **payload,
    }
    path = output_dir / "result.json"
    path.write_text(
        json.dumps(jsonable(result), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "module": module, "result": str(path)}))
    return path


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def placeholder(value: object) -> bool:
    return isinstance(value, str) and value.startswith("REPLACE_WITH_")


def fitness_for(code: str) -> float:
    additive = (
        {"V": 0.15, "A": 0.35, "C": -0.15}.get(code[0], -0.25)
        + {"D": 0.05, "E": 0.25, "W": 0.45}.get(code[1], -0.20)
        + {"G": 0.20, "A": 0.30, "S": 0.10}.get(code[2], -0.20)
        + {"V": 0.10, "I": 0.30, "F": 0.40}.get(code[3], -0.20)
    )
    epistasis = 0.55 if code[0] == "A" and code[1] == "W" else 0.0
    epistasis -= 0.35 if code[2] == "S" and code[3] == "F" else 0.0
    return float(additive + epistasis)


def make_variant(code: str, *, split_role: str = "oracle_pool") -> Variant:
    return Variant(
        variant_id=variant_id(code, assay_id="module_test_assay"),
        variant=code,
        sequence=code,
        mutation_notation=canonical_mutation_notation(code),
        mutation_count=sum(left != right for left, right in zip(code, "VDGV", strict=True)),
        split_role=split_role,
    )


def variant_grid(
    residues_by_site: tuple[tuple[str, ...], ...] = DEFAULT_RESIDUES_BY_SITE,
    *,
    split_role: str = "oracle_pool",
) -> list[Variant]:
    return [make_variant("".join(code), split_role=split_role) for code in product(*residues_by_site)]


def make_observations(
    variants: list[Variant],
    *,
    round_revealed: int = 0,
    source: str = "module_test",
) -> list[FitnessObservation]:
    return [
        FitnessObservation(
            variant_id=item.variant_id,
            fitness=fitness_for(item.variant),
            split_role=item.split_role,
            round_revealed=round_revealed,
            source=source,
        )
        for item in variants
    ]


def make_predictions(variants: list[Variant], *, model_version: str = "module-test:v1") -> list[Prediction]:
    predictions = []
    for index, item in enumerate(variants):
        mean = fitness_for(item.variant) + 0.03 * ((index % 3) - 1)
        std = 0.08 + 0.04 * (item.mutation_count / 4.0)
        predictions.append(
            Prediction(
                variant_id=item.variant_id,
                fitness_mean=mean,
                fitness_std=std,
                interval_90=(mean - 1.645 * std, mean + 1.645 * std),
                ood_score=item.mutation_count / 4.0,
                component_scores={"additive": mean - 0.05, "pairwise": mean + 0.05},
                model_version=model_version,
            )
        )
    return predictions


def make_evidence(variants: list[Variant], *, round_id: int = 1) -> dict[str, list[Evidence]]:
    output: dict[str, list[Evidence]] = {}
    for index, item in enumerate(variants):
        score = 0.7 if index % 2 == 0 else -0.4
        output[item.variant_id] = [
            Evidence(
                evidence_id=f"ev:module:{index}:support",
                variant_id=item.variant_id,
                channel="module_support",
                statement="Deterministic module-test evidence.",
                score=score,
                source_id="module-test:source-a",
                confidence=0.8,
                round_id=round_id,
            ),
            Evidence(
                evidence_id=f"ev:module:{index}:context",
                variant_id=item.variant_id,
                channel="module_context",
                statement="Independent context channel.",
                score=-0.2 if index == 1 else 0.2,
                source_id="module-test:source-b",
                confidence=0.6,
                round_id=round_id,
            ),
        ]
    return output


def write_legacy_benchmark(output_dir: Path, *, seed: int) -> dict[str, Path]:
    variants = variant_grid()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(variants)).tolist()
    wt_index = next(index for index, item in enumerate(variants) if item.variant == "VDGV")
    order.remove(wt_index)
    order.insert(0, wt_index)
    role_sizes = {
        "initial_observed": 16,
        "validation": 8,
        "final_test": 12,
        "oracle_pool": len(variants) - 36,
    }
    role_by_index: dict[int, str] = {}
    cursor = 0
    for role, size in role_sizes.items():
        for index in order[cursor : cursor + size]:
            role_by_index[index] = role
        cursor += size

    public_rows = []
    oracle_rows = []
    for index, variant in enumerate(variants):
        role = role_by_index[index]
        public_rows.append(
            {
                "variant_id": variant.variant_id,
                "variant": variant.variant,
                "sequence": variant.sequence,
                "mutation_notation": variant.mutation_notation,
                "mutation_count": variant.mutation_count,
                "split_role": role,
            }
        )
        oracle_rows.append(
            {
                "variant_id": variant.variant_id,
                "fitness": fitness_for(variant.variant),
                "split_role": role,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    public_path = output_dir / "synthetic_public.csv"
    oracle_path = output_dir / "synthetic_oracle.csv"
    pd.DataFrame(public_rows).to_csv(public_path, index=False)
    pd.DataFrame(oracle_rows).to_csv(oracle_path, index=False)
    return {"public": public_path, "oracle": oracle_path}

