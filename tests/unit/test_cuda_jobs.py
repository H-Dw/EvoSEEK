from __future__ import annotations

import threading
import time

import pytest

from fitness_agents.utils.cuda_jobs import (
    CudaDevicePool,
    discover_cuda_device_ids,
    environment_with_cuda_device,
    parse_cuda_devices_arg,
    resolve_cuda_device_pool,
)


def test_parse_cuda_devices_arg_accepts_aliases_and_cuda_prefix() -> None:
    assert parse_cuda_devices_arg("none") == "none"
    assert parse_cuda_devices_arg("AUTO") == "auto"
    assert parse_cuda_devices_arg("0,1,2,3") == ("0", "1", "2", "3")
    assert parse_cuda_devices_arg("cuda:0,cuda:3") == ("0", "3")


def test_parse_cuda_devices_arg_rejects_duplicates_and_junk() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        parse_cuda_devices_arg("0,0")
    with pytest.raises(ValueError, match="Invalid CUDA device"):
        parse_cuda_devices_arg("gpu0")


def test_resolve_auto_without_gpus_disables_isolation(monkeypatch) -> None:
    monkeypatch.setattr(
        "fitness_agents.utils.cuda_jobs.discover_cuda_device_ids", lambda: []
    )
    assert resolve_cuda_device_pool("auto", max_parallel=4) is None


def test_resolve_explicit_pool_must_cover_parallelism() -> None:
    assert resolve_cuda_device_pool(("0", "1", "2", "3"), max_parallel=4) == (
        "0",
        "1",
        "2",
        "3",
    )
    with pytest.raises(ValueError, match="exceeds 2 CUDA"):
        resolve_cuda_device_pool(("0", "1"), max_parallel=4)


def test_discover_prefers_parent_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert discover_cuda_device_ids() == ["2", "3"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert discover_cuda_device_ids() == []


def test_environment_isolates_one_physical_gpu() -> None:
    env = environment_with_cuda_device("2", base_env={"PATH": "/bin"})
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["PATH"] == "/bin"


def test_pool_never_lends_the_same_gpu_to_two_callers() -> None:
    pool = CudaDevicePool(("0", "1", "2", "3"))
    borrowed: set[str] = set()
    max_borrowed = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal max_borrowed
        device = pool.acquire()
        with lock:
            assert device not in borrowed
            borrowed.add(device)
            max_borrowed = max(max_borrowed, len(borrowed))
        time.sleep(0.05)
        with lock:
            borrowed.remove(device)
        pool.release(device)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max_borrowed == 4
    assert borrowed == set()
