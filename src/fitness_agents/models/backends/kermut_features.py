from __future__ import annotations

import hashlib
import urllib.error
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fitness_agents.config import project_root
from fitness_agents.contracts.schemas import Variant
from fitness_agents.utils.progress import emit_batch_progress

# Official fair-esm hub cache location for ESM-2 650M. Experiment configs should
# point ModelConfig.checkpoint here instead of relying on the hub download API.
DEFAULT_ESM2_CHECKPOINT = "~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt"


class KermutFeatureError(RuntimeError):
    """Raised when Kermut sequence features cannot be produced safely."""


def resolve_esm_checkpoint_path(model_path: str | Path) -> Path:
    path = Path(str(model_path)).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    if not path.is_file():
        raise FileNotFoundError(f"ESM-2 checkpoint does not exist: {path}")
    return path


_FAIR_ESM_CPU_PATCH_ATTR = "_fitness_agents_cpu_loader_patched"
DEFAULT_CHECKPOINT_MAP_LOCATION = "cpu"


def torch_load_trusted_esm(
    torch_module: Any,
    path: Path,
    *,
    map_location: str | None = None,
) -> Any:
    """Load an official fair-esm ``.pt`` file onto CPU.

    Facebook ESM checkpoints pickle ``argparse.Namespace``. PyTorch 2.6+ defaults
    ``torch.load(..., weights_only=True)``, which rejects that type and is what
    ``esm.pretrained.load_model_and_alphabet_local`` hits. These files are local
    checkpoints from a trusted source, so this helper always uses
    ``weights_only=False`` and ``map_location='cpu'``. Callers then ``.to(device)``.
    """
    import argparse

    target = DEFAULT_CHECKPOINT_MAP_LOCATION if map_location is None else map_location
    serialization = getattr(torch_module, "serialization", None)
    add_safe_globals = getattr(serialization, "add_safe_globals", None)
    if callable(add_safe_globals):
        add_safe_globals([argparse.Namespace])

    try:
        return torch_module.load(str(path), map_location=target, weights_only=False)
    except TypeError:
        return torch_module.load(str(path), map_location=target)


def _needs_esm_regression_weights(pretrained: Any, model_name: str) -> bool:
    checker = getattr(pretrained, "_has_regression_weights", None)
    if callable(checker):
        return bool(checker(model_name))
    return "esm1v" not in model_name and "esm_if" not in model_name


def load_fair_esm_local(pretrained: Any, torch_module: Any, path: Path):
    """Load a local ESM-2 checkpoint without fair-esm's PyTorch 2.6-incompatible helper."""
    model_data = torch_load_trusted_esm(torch_module, path)
    model_name = path.stem
    regression_data = None
    if _needs_esm_regression_weights(pretrained, model_name):
        regression_path = path.with_name(f"{path.stem}-contact-regression.pt")
        if not regression_path.is_file():
            hub_fallback = Path(torch_module.hub.get_dir()) / "checkpoints" / regression_path.name
            if hub_fallback.is_file():
                regression_path = hub_fallback
        if not regression_path.is_file():
            raise FileNotFoundError(
                "ESM contact-regression weights were not found next to the checkpoint "
                f"(expected {path.stem}-contact-regression.pt beside {path})"
            )
        regression_data = torch_load_trusted_esm(torch_module, regression_path)
    return pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data)


def patch_fair_esm_cpu_loaders(pretrained: Any, torch_module: Any) -> None:
    """Make fair-esm local and hub loaders unpickle official checkpoints on CPU.

    Hub downloads still work on PyTorch 2.6+ because ``load_state_dict_from_url``
    defaults to ``weights_only=False``. The local helper does not, so campaigns
    that point ``checkpoint`` at a cached ``.pt`` file crash unless patched.
    """
    if getattr(pretrained, _FAIR_ESM_CPU_PATCH_ATTR, False):
        return

    def load_model_and_alphabet_local(model_location):
        return load_fair_esm_local(
            pretrained, torch_module, resolve_esm_checkpoint_path(model_location)
        )

    def load_hub_workaround(url):
        kwargs = {"progress": False, "map_location": DEFAULT_CHECKPOINT_MAP_LOCATION}
        try:
            try:
                return torch_module.hub.load_state_dict_from_url(
                    url, weights_only=False, **kwargs
                )
            except TypeError:
                return torch_module.hub.load_state_dict_from_url(url, **kwargs)
        except RuntimeError:
            cached = Path(torch_module.hub.get_dir()) / "checkpoints" / Path(url).name
            return torch_load_trusted_esm(torch_module, cached)
        except urllib.error.HTTPError as error:
            raise KermutFeatureError(
                f"Could not load {url}, check if you specified a correct model name?"
            ) from error

    pretrained.load_model_and_alphabet_local = load_model_and_alphabet_local
    pretrained.load_hub_workaround = load_hub_workaround
    setattr(pretrained, _FAIR_ESM_CPU_PATCH_ATTR, True)


class PrecomputedKermutFeatures:
    """Read ESM-2 embeddings and zero-shot scores from a compact NPZ feature store.

    Required arrays are ``embeddings`` (N x D), ``zero_shot`` (N), and one key array named
    ``variant_ids`` or ``sequences``. This format is intentionally independent of campaign splits;
    the same immutable store can serve fitting and newly proposed candidates.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        if not self.path.is_absolute():
            self.path = project_root() / self.path
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

    def lookup(self, variant: Variant) -> tuple[np.ndarray, float] | None:
        key = variant.variant_id if self.key_kind == "variant_id" else variant.sequence
        index = self._index.get(key)
        if index is None:
            return None
        return self.embeddings[index], float(self.zero_shot[index])

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
    """ESM-2 inputs for Kermut: offline store lookup, npy cache, then live encode.

    ``precomputed_features_path`` is an optional NPZ library (variant_id or sequence keys).
    Hits never load fair-esm. Misses use ``cache_dir`` then a lazy CPU-loaded ESM-2 model.
    """

    def __init__(
        self,
        *,
        device: str,
        batch_size: int,
        checkpoint: str | None,
        options: dict[str, Any],
    ) -> None:
        self.device_name = device
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.options = dict(options)
        self.model_name = str(options.get("esm_model", "esm2_t33_650M_UR50D"))
        self.representation_layer = int(options.get("esm_representation_layer", 33))
        store_path = options.get("precomputed_features_path")
        self._store = PrecomputedKermutFeatures(str(store_path)) if store_path else None
        cache_value = options.get("cache_dir")
        cache_path = Path(str(cache_value)).expanduser() if cache_value else None
        if cache_path is not None and not cache_path.is_absolute():
            cache_path = project_root() / cache_path
        self.cache_dir = cache_path
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.torch = None
        self.device = None
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self._embedding_memory: dict[str, np.ndarray] = {}
        self._zero_shot_memory: dict[str, float] = {}
        self._position_memory: dict[tuple[str, int], np.ndarray] = {}

    def _ensure_runtime(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from esm import pretrained
        except ImportError as error:
            raise KermutFeatureError(
                "Live Kermut features require torch and fair-esm. Install the project with "
                "the 'kermut' optional dependency, or provide options.precomputed_features_path "
                "covering every requested sequence."
            ) from error

        self.torch = torch
        self.device = torch.device(self.device_name)
        patch_fair_esm_cpu_loaders(pretrained, torch)
        model_path = self.checkpoint or self.options.get("esm_model_path")
        if model_path:
            path = resolve_esm_checkpoint_path(str(model_path))
            self.model, self.alphabet = load_fair_esm_local(pretrained, torch, path)
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

    def _load_zero_shot_cache(self, sequence: str) -> float | None:
        if sequence in self._zero_shot_memory:
            return self._zero_shot_memory[sequence]
        path = self._cache_path("zero_shot", sequence)
        if path is not None and path.is_file():
            value = float(np.load(path, allow_pickle=False).astype(np.float32).reshape(-1)[0])
            self._zero_shot_memory[sequence] = value
            return value
        return None

    def _save_zero_shot_cache(self, sequence: str, value: float) -> None:
        result = float(value)
        self._zero_shot_memory[sequence] = result
        path = self._cache_path("zero_shot", sequence)
        if path is not None:
            np.save(path, np.asarray(result, dtype=np.float32), allow_pickle=False)

    def _embed_missing(self, sequences: Sequence[str]) -> None:
        self._ensure_runtime()
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
            completed = min(start + self.batch_size, len(sequences))
            total_batches = (len(sequences) + self.batch_size - 1) // self.batch_size
            batch_index = start // self.batch_size + 1
            if total_batches > 1:
                emit_batch_progress(
                    "kermut.esm_embed",
                    completed=batch_index,
                    total=total_batches,
                    items_done=completed,
                    items_total=len(sequences),
                )

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
        self._ensure_runtime()
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
        n_variants = len(variants)
        embeddings: list[np.ndarray | None] = [None] * n_variants
        zero_shot = np.full(n_variants, np.nan, dtype=np.float32)
        missing_embed: list[str] = []
        missing_zero: list[int] = []
        seen_missing: set[str] = set()

        for index, variant in enumerate(variants):
            if self._store is not None:
                hit = self._store.lookup(variant)
                if hit is not None:
                    embeddings[index] = np.asarray(hit[0], dtype=np.float32)
                    zero_shot[index] = hit[1]
                    self._embedding_memory[variant.sequence] = embeddings[index]
                    self._zero_shot_memory[variant.sequence] = hit[1]
                    continue
            cached_embedding = self._load_embedding_cache(variant.sequence)
            if cached_embedding is not None:
                embeddings[index] = cached_embedding
            elif variant.sequence not in seen_missing:
                missing_embed.append(variant.sequence)
                seen_missing.add(variant.sequence)
            cached_zero = self._load_zero_shot_cache(variant.sequence)
            if cached_zero is not None:
                zero_shot[index] = cached_zero
            else:
                missing_zero.append(index)

        if missing_embed:
            self._embed_missing(missing_embed)
            for index, variant in enumerate(variants):
                if embeddings[index] is None:
                    embeddings[index] = self._load_embedding_cache(variant.sequence)
        if any(item is None for item in embeddings):
            raise KermutFeatureError("Kermut ESM-2 embeddings could not be materialized")
        if missing_zero:
            live_sequences = [variants[index].sequence for index in missing_zero]
            live_scores = self._zero_shot_scores(live_sequences, wild_type_sequence)
            for offset, index in enumerate(missing_zero):
                zero_shot[index] = live_scores[offset]
                self._save_zero_shot_cache(variants[index].sequence, float(live_scores[offset]))
        return np.stack(embeddings).astype(np.float32), zero_shot


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
