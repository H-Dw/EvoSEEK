from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fitness_agents.contracts.schemas import Variant


class KermutFeatureError(RuntimeError):
    """Raised when Kermut sequence features cannot be produced safely."""


class PrecomputedKermutFeatures:
    """Read ESM-2 embeddings and zero-shot scores from a compact NPZ feature store.

    Required arrays are ``embeddings`` (N x D), ``zero_shot`` (N), and one key array named
    ``variant_ids`` or ``sequences``. This format is intentionally independent of campaign splits;
    the same immutable store can serve fitting and newly proposed candidates.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Kermut feature store does not exist: {self.path}")
        with np.load(self.path, allow_pickle=False) as payload:
            if "variant_ids" in payload:
                self.key_kind = "variant_id"
                keys = payload["variant_ids"]
            elif "sequences" in payload:
                self.key_kind = "sequence"
                keys = payload["sequences"]
            else:
                raise KermutFeatureError(
                    f"{self.path} must contain either variant_ids or sequences"
                )
            if "embeddings" not in payload or "zero_shot" not in payload:
                raise KermutFeatureError(
                    f"{self.path} must contain embeddings and zero_shot arrays"
                )
            self.keys = np.asarray(keys).astype(str)
            self.embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            self.zero_shot = np.asarray(payload["zero_shot"], dtype=np.float32).reshape(-1)

        if self.embeddings.ndim != 2:
            raise KermutFeatureError("Kermut embeddings must have shape (N, D)")
        if len(self.keys) != len(self.embeddings) or len(self.keys) != len(self.zero_shot):
            raise KermutFeatureError("Kermut feature-store arrays have inconsistent lengths")
        if len(set(self.keys)) != len(self.keys):
            raise KermutFeatureError("Kermut feature-store keys must be unique")
        if not np.isfinite(self.embeddings).all() or not np.isfinite(self.zero_shot).all():
            raise KermutFeatureError("Kermut feature store contains non-finite values")
        self._index = {key: index for index, key in enumerate(self.keys)}

    def encode(
        self, variants: Sequence[Variant], wild_type_sequence: str
    ) -> tuple[np.ndarray, np.ndarray]:
        del wild_type_sequence
        keys = [
            variant.variant_id if self.key_kind == "variant_id" else variant.sequence
            for variant in variants
        ]
        missing = [key for key in keys if key not in self._index]
        if missing:
            raise KermutFeatureError(
                f"Kermut feature store is missing {len(missing)} requested candidates; "
                f"examples={missing[:3]}"
            )
        indices = [self._index[key] for key in keys]
        return self.embeddings[indices], self.zero_shot[indices]


class LiveESM2KermutFeatures:
    """Generate the exact ESM-2 inputs used by Kermut, with an on-disk sequence cache."""

    def __init__(
        self,
        *,
        device: str,
        batch_size: int,
        checkpoint: str | None,
        options: dict[str, Any],
    ) -> None:
        try:
            import torch
            from esm import pretrained
        except ImportError as error:
            raise KermutFeatureError(
                "Live Kermut features require torch and fair-esm. Install the project with "
                "the 'kermut' optional dependency or use feature_mode: precomputed."
            ) from error

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model_name = str(options.get("esm_model", "esm2_t33_650M_UR50D"))
        model_path = checkpoint or options.get("esm_model_path")
        if model_path:
            path = Path(str(model_path))
            if not path.is_file():
                raise FileNotFoundError(f"ESM-2 checkpoint does not exist: {path}")
            self.model, self.alphabet = pretrained.load_model_and_alphabet_local(path)
        else:
            try:
                loader = getattr(pretrained, self.model_name)
            except AttributeError as error:
                raise KermutFeatureError(
                    f"fair-esm has no pretrained loader {self.model_name!r}"
                ) from error
            self.model, self.alphabet = loader()
        self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.representation_layer = int(options.get("esm_representation_layer", 33))
        cache_value = options.get("cache_dir")
        self.cache_dir = Path(str(cache_value)) if cache_value else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_memory: dict[str, np.ndarray] = {}
        self._position_memory: dict[tuple[str, int], np.ndarray] = {}

    @staticmethod
    def _sequence_hash(sequence: str) -> str:
        return hashlib.sha256(sequence.encode("ascii")).hexdigest()

    def _cache_path(self, kind: str, sequence: str, suffix: str = "") -> Path | None:
        if self.cache_dir is None:
            return None
        digest = self._sequence_hash(sequence)
        directory = self.cache_dir / kind / digest[:2]
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{digest}{suffix}.npy"

    def _load_embedding_cache(self, sequence: str) -> np.ndarray | None:
        if sequence in self._embedding_memory:
            return self._embedding_memory[sequence]
        path = self._cache_path("embeddings", sequence)
        if path is not None and path.is_file():
            value = np.load(path, allow_pickle=False).astype(np.float32)
            self._embedding_memory[sequence] = value
            return value
        return None

    def _save_embedding_cache(self, sequence: str, value: np.ndarray) -> None:
        result = np.asarray(value, dtype=np.float32)
        self._embedding_memory[sequence] = result
        path = self._cache_path("embeddings", sequence)
        if path is not None:
            np.save(path, result, allow_pickle=False)

    def _embed_missing(self, sequences: Sequence[str]) -> None:
        if any(len(sequence) > 1022 for sequence in sequences):
            raise KermutFeatureError(
                "ESM-2 Kermut embeddings require sequences of at most 1022 residues; "
                "silent truncation is disabled because it can hide mutated positions."
            )
        for start in range(0, len(sequences), self.batch_size):
            batch_sequences = list(sequences[start : start + self.batch_size])
            data = [(f"variant-{index}", sequence) for index, sequence in enumerate(batch_sequences)]
            _, strings, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)
            with self.torch.no_grad():
                output = self.model(
                    tokens,
                    repr_layers=[self.representation_layer],
                    return_contacts=False,
                )
            representations = output["representations"][self.representation_layer].detach().cpu()
            for index, sequence in enumerate(strings):
                pooled = representations[index, 1 : len(sequence) + 1].mean(dim=0).numpy()
                self._save_embedding_cache(sequence, pooled)

    def _embedding_matrix(self, sequences: Sequence[str]) -> np.ndarray:
        unique_missing = []
        seen = set()
        for sequence in sequences:
            if sequence not in seen and self._load_embedding_cache(sequence) is None:
                unique_missing.append(sequence)
                seen.add(sequence)
        if unique_missing:
            self._embed_missing(unique_missing)
        return np.stack([self._load_embedding_cache(sequence) for sequence in sequences]).astype(
            np.float32
        )

    def _masked_position_log_probs(self, wild_type: str, position: int) -> np.ndarray:
        key = (wild_type, position)
        if key in self._position_memory:
            return self._position_memory[key]
        path = self._cache_path("masked_log_probs", wild_type, suffix=f"-p{position}")
        if path is not None and path.is_file():
            value = np.load(path, allow_pickle=False).astype(np.float32)
            self._position_memory[key] = value
            return value

        _, _, tokens = self.batch_converter([("wild-type", wild_type)])
        token_position = position + 1  # BOS offset
        tokens[0, token_position] = self.alphabet.mask_idx
        with self.torch.no_grad():
            logits = self.model(tokens.to(self.device))["logits"]
            value = (
                self.torch.log_softmax(logits[0, token_position], dim=-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        self._position_memory[key] = value
        if path is not None:
            np.save(path, value, allow_pickle=False)
        return value

    def _zero_shot_scores(self, sequences: Sequence[str], wild_type: str) -> np.ndarray:
        scores = np.zeros(len(sequences), dtype=np.float32)
        for row, sequence in enumerate(sequences):
            if len(sequence) != len(wild_type):
                raise KermutFeatureError("Kermut currently supports substitutions, not indels")
            for position, (wt, mutant) in enumerate(zip(wild_type, sequence, strict=True)):
                if wt == mutant:
                    continue
                log_probs = self._masked_position_log_probs(wild_type, position)
                scores[row] += float(
                    log_probs[self.alphabet.get_idx(mutant)]
                    - log_probs[self.alphabet.get_idx(wt)]
                )
        return scores

    def encode(
        self, variants: Sequence[Variant], wild_type_sequence: str
    ) -> tuple[np.ndarray, np.ndarray]:
        sequences = [variant.sequence for variant in variants]
        return self._embedding_matrix(sequences), self._zero_shot_scores(
            sequences, wild_type_sequence
        )


def create_kermut_feature_source(
    *,
    device: str,
    batch_size: int,
    checkpoint: str | None,
    options: dict[str, Any],
):
    mode = str(options.get("feature_mode", "live_esm2")).lower()
    if mode == "precomputed":
        path = options.get("precomputed_features_path")
        if not path:
            raise KermutFeatureError(
                "feature_mode: precomputed requires options.precomputed_features_path"
            )
        return PrecomputedKermutFeatures(str(path))
    if mode == "live_esm2":
        return LiveESM2KermutFeatures(
            device=device,
            batch_size=batch_size,
            checkpoint=checkpoint,
            options=options,
        )
    raise KermutFeatureError(
        f"Unknown Kermut feature_mode {mode!r}; expected precomputed or live_esm2"
    )
