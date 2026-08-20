"""Assign one physical GPU to each parallel Kermut campaign process.

Kermut YAML keeps ``device: cuda:0``. Parallel workers must not all see the same
visible device set, or every process loads ESM-2 onto physical GPU 0. The
scheduler isolates workers with ``CUDA_VISIBLE_DEVICES`` so each child treats
its assigned card as ``cuda:0``.
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Mapping, Sequence
from queue import Queue
from typing import Any

NONE_ALIASES = frozenset({"", "none", "off", "disable", "cpu"})


class CudaDevicePool:
    """Thread-safe idle-GPU queue. Concurrent jobs never share a card."""

    def __init__(self, devices: Sequence[str]) -> None:
        normalized = tuple(devices)
        if not normalized:
            raise ValueError("CUDA device pool is empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("CUDA device list contains duplicates")
        self.devices = normalized
        self._idle: Queue[str] = Queue()
        for item in normalized:
            self._idle.put(item)
        self._lock = threading.Lock()
        self._in_use: set[str] = set()

    def acquire(self) -> str:
        device = self._idle.get()
        with self._lock:
            self._in_use.add(device)
        return device

    def release(self, device: str) -> None:
        with self._lock:
            self._in_use.discard(device)
        self._idle.put(device)


def normalize_cuda_device_id(value: str) -> str:
    token = value.strip()
    if token.lower().startswith("cuda:"):
        token = token.split(":", 1)[1].strip()
    if not token.isdigit():
        raise ValueError(f"Invalid CUDA device id {value!r}")
    return str(int(token))


def parse_cuda_devices_arg(value: str) -> str | tuple[str, ...]:
    """Return ``'none'``, ``'auto'``, or an explicit device-id tuple."""

    text = value.strip()
    lowered = text.lower()
    if lowered in NONE_ALIASES:
        return "none"
    if lowered == "auto":
        return "auto"
    devices = tuple(
        normalize_cuda_device_id(item) for item in text.split(",") if item.strip()
    )
    if not devices:
        raise ValueError("CUDA device list is empty")
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA device list contains duplicates")
    return devices


def discover_cuda_device_ids() -> list[str]:
    """Physical GPU indices visible to this process.

    Parent ``CUDA_VISIBLE_DEVICES`` wins so a cgroup/job allocation is respected.
    Otherwise query ``nvidia-smi``, then torch.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        text = visible.strip()
        if text in {"", "-1"}:
            return []
        return [normalize_cuda_device_id(item) for item in text.split(",") if item.strip()]
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, TimeoutError, OSError, subprocess.TimeoutExpired):
        completed = None
    if completed is not None and completed.returncode == 0:
        ids = [
            normalize_cuda_device_id(line)
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
        if ids:
            return ids
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [str(index) for index in range(int(torch.cuda.device_count()))]


def resolve_cuda_device_pool(
    spec: str | tuple[str, ...],
    *,
    max_parallel: int,
    enforce_capacity: bool = True,
) -> tuple[str, ...] | None:
    """Return the GPU pool, or None to inherit the parent environment.

    ``auto`` with no visible GPUs disables isolation instead of failing, so CPU
    dry-runs and CI stay usable. An explicit list, or ``auto`` that found cards
    when ``enforce_capacity`` is true, must cover ``max_parallel`` workers.
    """

    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")
    if spec == "none":
        return None
    if spec == "auto":
        devices = tuple(discover_cuda_device_ids())
        if not devices:
            return None
    else:
        devices = spec
    if enforce_capacity and max_parallel > len(devices):
        raise ValueError(
            f"--max-parallel {max_parallel} exceeds {len(devices)} CUDA device(s) "
            f"{list(devices)}. Pass matching --cuda-devices (for example 0,1,2,3), "
            "lower --max-parallel, or use --cuda-devices none."
        )
    return devices


def environment_with_cuda_device(
    device: str | None,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    if device is None:
        return env
    env["CUDA_VISIBLE_DEVICES"] = device
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def cuda_assignment_record(
    *,
    policy: str,
    devices: Sequence[str] | None,
    max_parallel: int,
) -> dict[str, Any]:
    enabled = devices is not None
    return {
        "policy": policy,
        "enabled": enabled,
        "devices": list(devices) if devices is not None else [],
        "max_parallel": max_parallel,
        "isolation": "CUDA_VISIBLE_DEVICES",
        "worker_kermut_device": "cuda:0",
    }
