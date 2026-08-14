from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge

from fitness_agents.contracts.interfaces import FeatureProvider
from fitness_agents.contracts.schemas import FitnessObservation, Prediction, Variant
from fitness_agents.features.gb1 import hamming_distance


@dataclass
class _FittedModels:
    ridge: list[Ridge]
    extra_trees: ExtraTreesRegressor
    gaussian_process: GaussianProcessRegressor | None


class OneHotHeterogeneousEnsemble:
    """Bootstrap Ridge + ExtraTrees groups, with optional small-data GP.

    The implementation follows ALDE's useful separation of predictor uncertainty and acquisition,
    while keeping the dependency footprint CPU-friendly. It intentionally contains no PLM+RF path.
    """

    def __init__(
        self,
        feature_provider: FeatureProvider,
        *,
        ridge_members: int = 5,
        extra_trees_estimators: int = 160,
        ridge_alpha: float = 10.0,
        bootstrap_fraction: float = 0.85,
        conformal_alpha: float = 0.10,
        include_gaussian_process: bool = False,
        seed: int = 0,
    ) -> None:
        self.feature_provider = feature_provider
        self.ridge_members = ridge_members
        self.extra_trees_estimators = extra_trees_estimators
        self.ridge_alpha = ridge_alpha
        self.bootstrap_fraction = bootstrap_fraction
        self.conformal_alpha = conformal_alpha
        self.include_gaussian_process = include_gaussian_process
        self.seed = seed
        self.model_version = f"onehot-ensemble-seed{seed}"
        self._models: _FittedModels | None = None
        self._calibration_radius = 0.0
        self._train_codes: list[str] = []

    @staticmethod
    def _align_targets(
        variants: Sequence[Variant], observations: Sequence[FitnessObservation]
    ) -> np.ndarray:
        target_map = {observation.variant_id: observation.fitness for observation in observations}
        missing = [variant.variant_id for variant in variants if variant.variant_id not in target_map]
        if missing:
            raise ValueError(f"Missing observations for {len(missing)} variants")
        return np.asarray([target_map[variant.variant_id] for variant in variants], dtype=float)

    def fit(
        self,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
        validation_variants: Sequence[Variant] | None = None,
        validation_observations: Sequence[FitnessObservation] | None = None,
    ) -> OneHotHeterogeneousEnsemble:
        if len(variants) < 4:
            raise ValueError("At least four visible observations are required")
        self.feature_provider.fit(variants)
        x_train = self.feature_provider.transform(variants)
        y_train = self._align_targets(variants, observations)
        rng = np.random.default_rng(self.seed + len(variants))
        sample_size = max(4, round(len(variants) * self.bootstrap_fraction))
        ridge_models: list[Ridge] = []
        for member in range(self.ridge_members):
            indices = rng.choice(len(variants), size=sample_size, replace=True)
            model = Ridge(alpha=self.ridge_alpha)
            model.fit(x_train[indices], y_train[indices])
            ridge_models.append(model)

        forest = ExtraTreesRegressor(
            n_estimators=self.extra_trees_estimators,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            max_samples=self.bootstrap_fraction,
            random_state=self.seed + len(variants) * 17,
            n_jobs=-1,
        )
        forest.fit(x_train, y_train)

        gp = None
        if self.include_gaussian_process:
            kernel = ConstantKernel(1.0) * RBF(length_scale=2.0) + WhiteKernel(noise_level=0.05)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=self.seed,
                n_restarts_optimizer=0,
            )
            gp.fit(x_train, y_train)
        self._models = _FittedModels(ridge=ridge_models, extra_trees=forest, gaussian_process=gp)
        self._train_codes = [variant.variant for variant in variants]

        self._calibration_radius = 0.0
        if validation_variants and validation_observations:
            validation_targets = self._align_targets(validation_variants, validation_observations)
            validation_members, _ = self._member_predictions(validation_variants)
            residuals = np.abs(validation_targets - validation_members.mean(axis=1))
            if len(residuals):
                quantile = min(1.0, np.ceil((len(residuals) + 1) * (1 - self.conformal_alpha)) / len(residuals))
                self._calibration_radius = float(np.quantile(residuals, quantile, method="higher"))
        return self

    def _member_predictions(
        self, variants: Sequence[Variant]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if self._models is None:
            raise RuntimeError("Predictor must be fitted before prediction")
        features = self.feature_provider.transform(variants)
        ridge_matrix = np.column_stack([model.predict(features) for model in self._models.ridge])
        tree_predictions = np.column_stack(
            [tree.predict(features) for tree in self._models.extra_trees.estimators_]
        )
        tree_groups = np.array_split(tree_predictions, min(5, tree_predictions.shape[1]), axis=1)
        forest_matrix = np.column_stack([group.mean(axis=1) for group in tree_groups])
        members = [ridge_matrix, forest_matrix]
        components = {
            "ridge": ridge_matrix.mean(axis=1),
            "extra_trees": tree_predictions.mean(axis=1),
        }
        if self._models.gaussian_process is not None:
            gp_mean, _ = self._models.gaussian_process.predict(features, return_std=True)
            members.append(gp_mean[:, None])
            components["gaussian_process"] = gp_mean
        return np.column_stack(members), components

    def predict(self, variants: Sequence[Variant]) -> list[Prediction]:
        if not variants:
            return []
        calibrated_sigma = max(self._calibration_radius / 1.645, 1e-8)
        predictions: list[Prediction] = []
        max_train_depth = max(
            sum(a != b for a, b in zip(code, "VDGV", strict=True)) for code in self._train_codes
        )
        batch_size = 8192
        for start in range(0, len(variants), batch_size):
            batch = variants[start : start + batch_size]
            member_matrix, components = self._member_predictions(batch)
            means = member_matrix.mean(axis=1)
            epistemic = (
                member_matrix.std(axis=1, ddof=1)
                if member_matrix.shape[1] > 1
                else np.zeros(len(batch))
            )
            stds = np.sqrt(epistemic**2 + calibrated_sigma**2)
            for index, variant in enumerate(batch):
                nearest = min(hamming_distance(variant.variant, code) for code in self._train_codes)
                depth_novelty = max(0, variant.mutation_count - max_train_depth)
                ood = min(1.0, nearest / 4.0 + 0.15 * depth_novelty)
                radius = max(self._calibration_radius, 1.645 * float(stds[index]))
                predictions.append(
                    Prediction(
                        variant_id=variant.variant_id,
                        fitness_mean=float(means[index]),
                        fitness_std=float(stds[index]),
                        interval_90=(float(means[index] - radius), float(means[index] + radius)),
                        ood_score=float(ood),
                        component_scores={
                            name: float(values[index]) for name, values in components.items()
                        },
                        model_version=self.model_version,
                    )
                )
        return predictions
