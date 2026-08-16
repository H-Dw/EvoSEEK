from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fitness_agents.config import project_root
from fitness_agents.contracts.schemas import FitnessObservation, Variant
from fitness_agents.models.external import ExternalModelContext
from fitness_agents.utils.progress import TimedHeartbeat, emit_batch_progress, heartbeat

from .kermut_features import KermutFeatureError, create_kermut_feature_source

OFFICIAL_KERMUT_COMMIT = "7e9e2e62a59773f6cc8291d85e6d6006a41a6862"


class KermutDependencyError(RuntimeError):
    """Raised when the optional official Kermut runtime is unavailable."""


def _load_runtime() -> dict[str, Any]:
    try:
        import gpytorch
        import torch
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.priors import HalfCauchyPrior

        from fitness_agents.models.backends.kermut_core import KermutGP, Tokenizer, optimize_gp
    except ImportError as error:
        raise KermutDependencyError(
            "The built-in Kermut backend requires torch and gpytorch, plus fair-esm for live "
            "features. "
            "Install `fitness-agents[kermut]`."
        ) from error
    return {
        "gpytorch": gpytorch,
        "torch": torch,
        "GaussianLikelihood": GaussianLikelihood,
        "HalfCauchyPrior": HalfCauchyPrior,
        "Tokenizer": Tokenizer,
        "KermutGP": KermutGP,
        "optimize_gp": optimize_gp,
    }


class KermutBackend:
    """Assay-specific backend using the official Kermut composite-kernel Exact GP.

    This adapter retains the published model components: ESM-2 mean-pooled embeddings, an ESM-2
    zero-shot linear mean, ProteinMPNN conditional distributions, C-alpha distances, and the
    weighted sequence/structure covariance. It does not silently replace missing components with
    one-hot features.
    """

    model_version = f"official-main-{OFFICIAL_KERMUT_COMMIT[:12]}"

    def __init__(self, context: ExternalModelContext) -> None:
        self.context = context
        self.options = dict(context.options)
        self.runtime = _load_runtime()
        self.torch = self.runtime["torch"]
        self.device = self.torch.device(context.device)
        missing_resources = [
            name
            for name in ("conditional_probs_path", "coords_path")
            if not self.options.get(name)
        ]
        if missing_resources:
            raise KermutFeatureError(
                "Kermut requires configured structure resources before ESM-2 is loaded: "
                + ", ".join(f"options.{name}" for name in missing_resources)
            )
        self.feature_source = create_kermut_feature_source(
            device=context.device,
            batch_size=context.batch_size,
            checkpoint=context.checkpoint,
            options=self.options,
        )
        self.gp = None
        self.likelihood = None
        self.wild_type_sequence = ""
        self._wt_fitness: float | None = None
        self._target_mean = 0.0
        self._target_scale = 1.0
        self._train_codes: list[str] = []
        self._calibration_radius = 0.0

    @staticmethod
    def _align_targets(
        variants: Sequence[Variant], observations: Sequence[FitnessObservation]
    ) -> np.ndarray:
        by_id = {observation.variant_id: observation.fitness for observation in observations}
        missing = [variant.variant_id for variant in variants if variant.variant_id not in by_id]
        if missing:
            raise ValueError(f"Kermut is missing observations for {len(missing)} variants")
        return np.asarray([by_id[variant.variant_id] for variant in variants], dtype=np.float32)

    def _find_wild_type(
        self, variants: Sequence[Variant], observations: Sequence[FitnessObservation]
    ) -> str:
        configured = self.options.get("wild_type_sequence")
        candidates = [variant for variant in variants if variant.mutation_count == 0]
        if configured:
            wild_type = str(configured)
            if candidates and any(variant.sequence != wild_type for variant in candidates):
                raise ValueError("Configured Kermut wild_type_sequence disagrees with observed WT")
        elif candidates:
            sequences = {variant.sequence for variant in candidates}
            if len(sequences) != 1:
                raise ValueError("Observed Kermut wild-type variants have inconsistent sequences")
            wild_type = sequences.pop()
        else:
            raise ValueError(
                "Kermut needs an observed mutation_count=0 variant or options.wild_type_sequence"
            )
        target_by_id = {item.variant_id: item.fitness for item in observations}
        for variant in candidates:
            if variant.variant_id in target_by_id:
                self._wt_fitness = float(target_by_id[variant.variant_id])
                break
        return wild_type

    @staticmethod
    def _resolve_path(path_value: Any, name: str) -> Path:
        if not path_value:
            raise KermutFeatureError(f"Kermut requires options.{name}_path")
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            path = project_root() / path
        return path

    @staticmethod
    def _load_array(path_value: Any, name: str) -> np.ndarray:
        path = KermutBackend._resolve_path(path_value, name)
        if not path.is_file():
            raise FileNotFoundError(f"Kermut {name} file does not exist: {path}")
        value = np.load(path, allow_pickle=False)
        return np.asarray(value, dtype=np.float32)

    def _expand_site_resource(
        self, value: np.ndarray, sequence_length: int, width: int, name: str
    ) -> np.ndarray:
        if value.ndim != 2 or value.shape[1] != width:
            raise KermutFeatureError(f"Kermut {name} must have shape (L, {width})")
        if value.shape[0] == sequence_length:
            return value
        positions = self.options.get("resource_positions")
        if not positions:
            raise KermutFeatureError(
                f"Kermut {name} has {value.shape[0]} rows for a {sequence_length}-residue "
                "sequence; provide options.resource_positions"
            )
        one_indexed = bool(self.options.get("positions_are_one_indexed", True))
        resource_indices = np.asarray(
            [int(position) - (1 if one_indexed else 0) for position in positions],
            dtype=np.int64,
        )
        if len(set(resource_indices.tolist())) != len(resource_indices):
            raise KermutFeatureError("Kermut resource_positions must be unique")

        # GB1 tables often encode only the four mutable residues while upstream structure files
        # retain the full protein. Select those rows into the compact four-position sequence.
        if len(resource_indices) == sequence_length and (
            (resource_indices >= 0) & (resource_indices < value.shape[0])
        ).all():
            return value[resource_indices]

        # Conversely, an open-space backend can use full protein sequences with site-only resource
        # files. Scatter the supplied rows into their positions and fill unused rows safely.
        if len(resource_indices) != value.shape[0] or not (
            (resource_indices >= 0) & (resource_indices < sequence_length)
        ).all():
            raise KermutFeatureError(
                f"Kermut cannot map {value.shape[0]} {name} rows to sequence length "
                f"{sequence_length} using resource_positions={list(positions)}"
            )
        if name == "conditional_probs":
            expanded = np.full((sequence_length, width), 1.0 / width, dtype=np.float32)
        else:
            expanded = np.zeros((sequence_length, width), dtype=np.float32)
        for source_index, target in enumerate(resource_indices):
            expanded[target] = value[source_index]
        return expanded

    def _active_model_rows(self, sequence_length: int) -> np.ndarray:
        positions = self.options.get("resource_positions")
        if not positions:
            return np.arange(sequence_length)
        offset = 1 if bool(self.options.get("positions_are_one_indexed", True)) else 0
        rows = np.asarray([int(position) - offset for position in positions], dtype=np.int64)
        if len(rows) == sequence_length and not ((rows >= 0) & (rows < sequence_length)).all():
            # The sequence itself is a compact mutable-site representation such as GB1 "VDGV".
            return np.arange(sequence_length)
        if ((rows >= 0) & (rows < sequence_length)).all():
            return rows
        raise KermutFeatureError(
            "Kermut resource_positions cannot be mapped to the model sequence representation"
        )

    def _structure_inputs(self, sequence_length: int) -> tuple[Any, Any]:
        conditional = self._load_array(
            self.options.get("conditional_probs_path"), "conditional_probs"
        )
        coords = self._load_array(self.options.get("coords_path"), "coords")
        conditional = self._expand_site_resource(
            conditional, sequence_length, 20, "conditional_probs"
        )
        coords = self._expand_site_resource(coords, sequence_length, 3, "coords")
        if self.options.get("resource_positions"):
            active_rows = self._active_model_rows(sequence_length)
            if not np.isfinite(conditional[active_rows]).all():
                raise KermutFeatureError(
                    "Kermut conditional_probs contains non-finite values at active sites"
                )
            if not np.isfinite(coords[active_rows]).all():
                raise KermutFeatureError("Kermut coords contains non-finite values at active sites")
            conditional = np.nan_to_num(conditional, nan=1.0 / 20, posinf=1.0, neginf=1e-12)
            coords = np.nan_to_num(coords)
        elif not np.isfinite(conditional).all() or not np.isfinite(coords).all():
            raise KermutFeatureError(
                "Kermut structure resources contain non-finite values; provide resource_positions "
                "when only selected mutation sites are valid"
            )
        conditional = np.clip(conditional, 1e-12, None)
        return (
            self.torch.tensor(conditional, dtype=self.torch.float32, device=self.device),
            self.torch.tensor(coords, dtype=self.torch.float32, device=self.device),
        )

    def _inputs_for(self, variants: Sequence[Variant]) -> tuple[Any, Any, Any, np.ndarray]:
        tokenizer = self.runtime["Tokenizer"]()
        sequences = [variant.sequence for variant in variants]
        tokens = tokenizer(sequences).to(self.device)
        embeddings, zero_shot = self.feature_source.encode(variants, self.wild_type_sequence)
        x_embeddings = self.torch.tensor(
            embeddings, dtype=self.torch.float32, device=self.device
        )
        x_zero = self.torch.tensor(zero_shot, dtype=self.torch.float32, device=self.device)
        return tokens, x_embeddings, x_zero, zero_shot

    def fit(
        self,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
        validation_variants: Sequence[Variant] | None = None,
        validation_observations: Sequence[FitnessObservation] | None = None,
    ) -> None:
        self.wild_type_sequence = self._find_wild_type(variants, observations)
        if any(len(variant.sequence) != len(self.wild_type_sequence) for variant in variants):
            raise ValueError("Kermut currently supports substitution variants of a fixed length")

        # The official structure kernel is defined over mutation events and has no event for WT.
        # Keep WT as an exact observed prediction but fit the GP on actual mutants.
        train_variants = [variant for variant in variants if variant.mutation_count > 0]
        if len(train_variants) < 4:
            raise ValueError("Kermut requires at least four observed non-WT variants")
        y = self._align_targets(train_variants, observations)
        self._target_mean = float(y.mean())
        # torch.std() in the official implementation uses the sample standard deviation.
        target_std = float(y.std(ddof=1))
        self._target_scale = target_std if target_std > 1e-8 else 1.0
        y_standardized = (y - self._target_mean) / self._target_scale

        self.torch.manual_seed(self.context.seed)
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(self.context.seed)
        heartbeat(
            f"Kermut encoding {len(train_variants)} training variants on {self.device.type}",
            n_train=len(train_variants),
            device=str(self.device),
        )
        with TimedHeartbeat("Kermut ESM encode"):
            x_tokens, x_embeddings, x_zero, _ = self._inputs_for(train_variants)
        y_tensor = self.torch.tensor(
            y_standardized, dtype=self.torch.float32, device=self.device
        )
        conditional_probs, coords = self._structure_inputs(len(self.wild_type_sequence))
        wt_tokens = self.runtime["Tokenizer"]()(self.wild_type_sequence).to(self.device)
        noise_prior = self.runtime["HalfCauchyPrior"](
            scale=float(self.options.get("noise_prior_scale", 0.1))
        )
        likelihood = self.runtime["GaussianLikelihood"](noise_prior=noise_prior).to(self.device)
        gp = self.runtime["KermutGP"](
            (x_tokens, x_embeddings, x_zero),
            y_tensor,
            likelihood,
            use_zero_shot_mean=bool(self.options.get("use_zero_shot_mean", True)),
            composition=str(self.options.get("composition", "weighted_sum")),
            wild_type=wt_tokens,
            conditional_probs=conditional_probs,
            coords=coords,
            h_lengthscale=float(self.options.get("h_lengthscale", 1.0)),
            d_lengthscale=float(self.options.get("d_lengthscale", 1.0)),
            p_lengthscale=float(self.options.get("p_lengthscale", 1.0)),
        ).to(self.device)
        n_steps = int(self.options.get("n_steps", 150))
        if n_steps < 1:
            raise ValueError("Kermut options.n_steps must be at least 1")
        heartbeat(
            f"Kermut optimizing GP for {n_steps} steps",
            n_train=len(train_variants),
            n_steps=n_steps,
        )
        with TimedHeartbeat("Kermut GP optimize"):
            gp, likelihood = self.runtime["optimize_gp"](
                gp,
                likelihood,
                (x_tokens, x_embeddings, x_zero),
                y_tensor,
                learning_rate=float(self.options.get("learning_rate", 0.1)),
                n_steps=n_steps,
            )
        self.gp = gp
        self.likelihood = likelihood
        self._train_codes = [variant.variant for variant in train_variants]
        self._calibration_radius = 0.0

        if (
            bool(self.options.get("use_validation_conformal", False))
            and validation_variants
            and validation_observations
        ):
            validation_targets = self._align_targets(
                validation_variants, validation_observations
            )
            means, _, _, _ = self._predict_numeric(validation_variants)
            residuals = np.abs(validation_targets - means)
            alpha = float(self.options.get("conformal_alpha", 0.1))
            quantile = min(
                1.0,
                math.ceil((len(residuals) + 1) * (1 - alpha)) / len(residuals),
            )
            self._calibration_radius = float(
                np.quantile(residuals, quantile, method="higher")
            )

    def _predict_numeric(
        self, variants: Sequence[Variant]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.gp is None or self.likelihood is None:
            raise RuntimeError("Kermut must be fitted before prediction")
        if not variants:
            empty = np.asarray([], dtype=np.float32)
            return empty, empty, empty, empty
        x_tokens, x_embeddings, x_zero, zero_shot = self._inputs_for(variants)
        self.gp.eval()
        self.likelihood.eval()
        with self.torch.no_grad(), self.runtime["gpytorch"].settings.fast_pred_var():
            distribution = self.likelihood(self.gp(x_tokens, x_embeddings, x_zero))
            standardized_mean = distribution.mean.detach().cpu().numpy()
            standardized_var = distribution.variance.clamp_min(0).detach().cpu().numpy()
        means = standardized_mean * self._target_scale + self._target_mean
        stds = np.sqrt(standardized_var) * abs(self._target_scale)
        if self._calibration_radius > 0:
            stds = np.sqrt(stds**2 + (self._calibration_radius / 1.645) ** 2)
        return means, stds, standardized_mean, zero_shot

    def _ood_score(self, variant: Variant) -> float:
        if not self._train_codes:
            return 1.0
        distances = [
            sum(a != b for a, b in zip(variant.variant, code, strict=True))
            for code in self._train_codes
        ]
        return min(1.0, min(distances) / max(1, len(variant.variant)))

    def predict(self, variants: Sequence[Variant]) -> list[dict[str, Any]]:
        if not variants:
            return []
        results: dict[str, dict[str, Any]] = {}
        mutants = [variant for variant in variants if variant.mutation_count > 0]
        n_batches = (len(mutants) + self.context.batch_size - 1) // self.context.batch_size if mutants else 0
        for batch_index, start in enumerate(range(0, len(mutants), self.context.batch_size), start=1):
            batch = mutants[start : start + self.context.batch_size]
            means, stds, standardized_means, zero_shot = self._predict_numeric(batch)
            for index, variant in enumerate(batch):
                mean = float(means[index])
                std = float(stds[index])
                results[variant.variant_id] = {
                    "variant_id": variant.variant_id,
                    "fitness_mean": mean,
                    "fitness_std": std,
                    "interval_90": (mean - 1.645 * std, mean + 1.645 * std),
                    "ood_score": self._ood_score(variant),
                    "component_scores": {
                        "kermut_gp_standardized": float(standardized_means[index]),
                        "esm2_zero_shot": float(zero_shot[index]),
                    },
                    "model_version": self.model_version,
                }
            if n_batches > 1:
                emit_batch_progress(
                    "kermut.predict",
                    completed=batch_index,
                    total=n_batches,
                    items_done=len(results),
                    items_total=len(mutants),
                )

        for variant in variants:
            if variant.mutation_count != 0:
                continue
            mean = self._wt_fitness if self._wt_fitness is not None else self._target_mean
            results[variant.variant_id] = {
                "variant_id": variant.variant_id,
                "fitness_mean": float(mean),
                "fitness_std": 0.0,
                "interval_90": (float(mean), float(mean)),
                "ood_score": 0.0,
                "component_scores": {"observed_wild_type": float(mean)},
                "model_version": self.model_version,
                "is_measured": self._wt_fitness is not None,
            }
        return [results[variant.variant_id] for variant in variants]


def create_backend(context: ExternalModelContext) -> KermutBackend:
    return KermutBackend(context)
