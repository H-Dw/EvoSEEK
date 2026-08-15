from __future__ import annotations

import importlib


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested accelerator is unavailable."""


def _torch_module():
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def resolve_device(requested: str, *, allow_fallback: bool = False) -> str:
    """Resolve a model device without importing torch for the default CPU path.

    The core package intentionally does not depend on torch. External predictor plugins receive
    the returned device string and own the actual tensor/model placement.
    """

    device = requested.strip().lower()
    if device == "gpu":
        device = "cuda"
    if device == "cpu":
        return "cpu"
    if device not in {"auto", "cuda", "mps"} and not device.startswith("cuda:"):
        raise ValueError(
            "device must be one of cpu, auto, gpu, cuda, cuda:N, or mps; "
            f"received {requested!r}"
        )

    torch = _torch_module()
    if torch is None:
        if allow_fallback:
            return "cpu"
        raise DeviceUnavailableError(
            f"Requested device {requested!r}, but torch is not installed. "
            "Install the selected predictor backend or set device: cpu."
        )

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if device == "auto":
        if cuda_available:
            return "cuda:0"
        if mps_available:
            return "mps"
        return "cpu"
    if device.startswith("cuda") and cuda_available:
        if device == "cuda":
            return "cuda:0"
        try:
            index = int(device.split(":", 1)[1])
        except ValueError as error:
            raise ValueError(f"Invalid CUDA device {requested!r}") from error
        if index < 0 or index >= int(torch.cuda.device_count()):
            if allow_fallback:
                return "cpu"
            raise DeviceUnavailableError(
                f"Requested CUDA index {index}, available device count={torch.cuda.device_count()}"
            )
        return device
    if device == "mps" and mps_available:
        return "mps"
    if allow_fallback:
        return "cpu"
    raise DeviceUnavailableError(
        f"Requested device {requested!r} is unavailable; set allow_device_fallback: true "
        "to fall back to CPU explicitly."
    )
