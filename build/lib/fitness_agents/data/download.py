"""Registry-driven dataset download engine.

Replaces per-dataset hardcoded curl scripts with a single engine configured by
``configs/data/datasets.yaml`` and ``configs/data/profiles/*.yaml``.

Guarantees:
- pinned upstream versions (commit hashes / release tags, never a moving branch
  as the primary reference);
- ordered mirror fallback per file;
- resume of interrupted downloads via HTTP Range on ``.partial`` files;
- atomic rename to the final name only after checksum verification;
- SHA256 verification when pinned, trust-on-first-use recording otherwise;
- archive member whitelists with path-traversal guards;
- a per-dataset ``download_manifest.json`` capturing version, URLs, checksums,
  license and extraction results;
- ``--force``, ``--offline`` and ``--verify-only`` semantics.
"""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fitness_agents.config import project_root, read_yaml

REGISTRY_PATH = "configs/data/datasets.yaml"
PROFILE_DIR = "configs/data/profiles"
MANIFEST_NAME = "download_manifest.json"
USER_AGENT = "fitness-agents-downloader/1.0"


class DownloadError(RuntimeError):
    """Raised when a required file cannot be fetched or verified."""


@dataclass
class FileResult:
    name: str
    status: str  # downloaded | reused | verified | skipped_optional | failed
    path: str | None = None
    url_used: str | None = None
    sha256: str | None = None
    sha256_pinned: bool = False
    size_bytes: int = 0
    extracted_members: list[str] = field(default_factory=list)
    message: str | None = None


@dataclass
class DatasetResult:
    dataset_id: str
    status: str  # ok | partial | failed | verified
    dest: str
    files: list[FileResult] = field(default_factory=list)
    message: str | None = None


# --------------------------------------------------------------------- utils


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------- registry


def load_registry(root: Path | None = None) -> dict[str, dict]:
    base = root or project_root()
    registry = read_yaml(REGISTRY_PATH, base)
    datasets = registry.get("datasets") or {}
    if not datasets:
        raise DownloadError(f"No datasets declared in {REGISTRY_PATH}")
    return datasets


def resolve_profile(profile: str, root: Path | None = None) -> list[str]:
    """Resolve a profile name to an ordered, de-duplicated dataset id list."""
    base = root or project_root()
    path = Path(profile)
    if not path.suffix:
        path = base / PROFILE_DIR / f"{profile}.yaml"
    if not path.exists():
        raise DownloadError(f"Unknown profile: {profile} (looked for {path})")
    cfg = read_yaml(path, base)
    ordered: list[str] = []

    def _add(ids: list[str]) -> None:
        for dataset_id in ids:
            if dataset_id not in ordered:
                ordered.append(dataset_id)

    for include in cfg.get("includes") or []:
        _add(resolve_profile(include, base))
    _add(cfg.get("datasets") or [])
    if not ordered:
        raise DownloadError(f"Profile {profile} resolved to zero datasets")
    return ordered


def load_mvp_assays(dataset_cfg: dict, root: Path) -> list[str]:
    list_path = dataset_cfg.get("mvp_assay_list")
    if not list_path:
        return []
    path = Path(list_path)
    if not path.is_absolute():
        path = root / path
    assays = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            assays.append(line)
    return assays


def _member_patterns(file_cfg: dict, dataset_cfg: dict, root: Path) -> list[str]:
    patterns = list(file_cfg.get("extract_members") or [])
    if "@mvp_assays" in patterns:
        patterns.remove("@mvp_assays")
        for assay in load_mvp_assays(dataset_cfg, root):
            patterns.append(f"{assay}.csv")
            patterns.append(f"{assay}.*")
    return patterns


def _matches(member: str, patterns: list[str]) -> bool:
    base = member.rsplit("/", 1)[-1]
    return any(
        fnmatch.fnmatch(member, pat) or fnmatch.fnmatch(base, pat) for pat in patterns
    )


# ----------------------------------------------------------------- download


def _fetch(url: str, partial: Path, retries: int, timeout: int) -> int:
    """Download ``url`` into ``partial`` with resume and retries. Returns size."""
    headers = {"User-Agent": USER_AGENT}
    attempt = 0
    while True:
        attempt += 1
        try:
            resume_from = partial.stat().st_size if partial.exists() else 0
            request_headers = dict(headers)
            if resume_from:
                request_headers["Range"] = f"bytes={resume_from}-"
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                mode = "ab" if (resume_from and response.status == 206) else "wb"
                if mode == "wb" and resume_from:
                    resume_from = 0  # server ignored Range; restart cleanly
                with partial.open(mode) as out:
                    shutil.copyfileobj(response, out, length=1024 * 256)
            return partial.stat().st_size
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt > retries:
                raise DownloadError(f"failed after {retries} attempts: {exc}") from exc
            time.sleep(min(2**attempt, 30))


def download_file(
    file_cfg: dict,
    dest_dir: Path,
    *,
    force: bool = False,
    offline: bool = False,
    retries: int = 3,
    timeout: int = 60,
) -> FileResult:
    name = file_cfg["name"]
    target = dest_dir / file_cfg.get("target", name)
    pinned = file_cfg.get("sha256")
    partial = target.with_name(target.name + ".partial")

    if target.exists() and not force:
        if pinned:
            actual = sha256_file(target)
            if actual != pinned:
                target.unlink()
            else:
                return FileResult(name, "reused", str(target), sha256=actual,
                                  sha256_pinned=True, size_bytes=target.stat().st_size)
        else:
            return FileResult(name, "reused", str(target),
                              sha256=sha256_file(target), sha256_pinned=False,
                              size_bytes=target.stat().st_size)

    if offline:
        raise DownloadError(f"{name}: not present locally and --offline is set")

    dest_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in file_cfg.get("urls") or []:
        try:
            size = _fetch(url, partial, retries=retries, timeout=timeout)
        except DownloadError as exc:
            errors.append(f"{url}: {exc}")
            continue
        min_bytes = int(file_cfg.get("min_bytes") or 0)
        if size < min_bytes:
            errors.append(f"{url}: size {size} below expected minimum {min_bytes}")
            partial.unlink(missing_ok=True)
            continue
        actual = sha256_file(partial)
        if pinned and actual != pinned:
            errors.append(f"{url}: sha256 mismatch ({actual})")
            partial.unlink(missing_ok=True)
            continue
        partial.replace(target)  # atomic rename after verification
        return FileResult(
            name, "downloaded", str(target), url_used=url, sha256=actual,
            sha256_pinned=bool(pinned), size_bytes=size,
            message=None if pinned else "TOFU: pin this sha256 in datasets.yaml",
        )
    partial.unlink(missing_ok=True)
    raise DownloadError(f"{name}: all sources exhausted; " + "; ".join(errors))


# ---------------------------------------------------------------- extraction


def _safe_members(archive: zipfile.ZipFile, patterns: list[str]) -> list[str]:
    names = [n for n in archive.namelist() if not n.endswith("/")]
    if patterns:
        names = [n for n in names if _matches(n, patterns)]
    for name in names:
        normalized = Path(name)
        if normalized.is_absolute() or ".." in normalized.parts:
            raise DownloadError(f"unsafe archive member rejected: {name}")
    if patterns and not names:
        raise DownloadError(f"member whitelist {patterns} matched nothing")
    return names


def extract_file(file_cfg: dict, dataset_cfg: dict, root: Path, dest_dir: Path) -> list[str]:
    """Extract whitelisted members; returns extracted member names (may be empty)."""
    if not file_cfg.get("extract"):
        return []
    archive_type = file_cfg.get("archive_type", "none")
    target = dest_dir / file_cfg.get("target", file_cfg["name"])
    patterns = _member_patterns(file_cfg, dataset_cfg, root)

    if archive_type == "zip":
        with zipfile.ZipFile(target) as archive:
            members = _safe_members(archive, patterns)
            for member in members:
                # Flatten single-file archives; keep relative layout otherwise.
                out_name = Path(member).name if len(members) == 1 else member
                out_path = dest_dir / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        if not file_cfg.get("keep_archive", True):
            target.unlink()
        return members
    if archive_type == "gz":
        out_path = dest_dir / target.stem  # strip .gz
        with gzip.open(target, "rb") as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        if not file_cfg.get("keep_archive", True):
            target.unlink()
        return [out_path.name]
    if archive_type == "none":
        return []
    raise DownloadError(f"unsupported archive_type: {archive_type}")


# ------------------------------------------------------------------ manifest


def _write_manifest(dest_dir: Path, dataset_id: str, cfg: dict, files: list[FileResult]) -> Path:
    manifest = {
        "dataset_id": dataset_id,
        "version": cfg.get("version"),
        "source_type": cfg.get("source_type"),
        "license": cfg.get("license"),
        "citation": cfg.get("citation"),
        "adapter": cfg.get("adapter"),
        "fitness_direction": cfg.get("fitness_direction", "maximize"),
        "downloaded_at": _utc_now(),
        "files": [
            {
                "name": f.name,
                "path": f.path,
                "url_used": f.url_used,
                "sha256": f.sha256,
                "sha256_pinned": f.sha256_pinned,
                "size_bytes": f.size_bytes,
                "extracted_members": f.extracted_members,
                "status": f.status,
            }
            for f in files
            if f.status not in {"failed"}
        ],
    }
    path = dest_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# -------------------------------------------------------------------- driver


def download_dataset(
    dataset_id: str,
    cfg: dict,
    root: Path,
    *,
    force: bool = False,
    offline: bool = False,
    verify_only: bool = False,
    retries: int = 3,
    timeout: int = 60,
) -> DatasetResult:
    dest = Path(cfg["dest"])
    if not dest.is_absolute():
        dest = root / dest
    result = DatasetResult(dataset_id, "ok", str(dest))

    if verify_only:
        return verify_dataset(dataset_id, cfg, root)

    for file_cfg in cfg.get("files") or []:
        optional = bool(file_cfg.get("optional"))
        try:
            file_result = download_file(
                file_cfg, dest, force=force, offline=offline,
                retries=retries, timeout=timeout,
            )
            if file_result.status in {"downloaded", "reused"}:
                file_result.extracted_members = extract_file(
                    file_cfg, cfg, root, dest
                )
            result.files.append(file_result)
        except DownloadError as exc:
            if optional:
                result.files.append(FileResult(file_cfg["name"], "skipped_optional",
                                               message=str(exc)))
            else:
                result.files.append(FileResult(file_cfg["name"], "failed",
                                               message=str(exc)))
                result.status = "failed"
                result.message = str(exc)
                break
    if result.status != "failed" and any(f.status == "skipped_optional" for f in result.files):
        result.status = "partial"
    if result.status != "failed":
        dest.mkdir(parents=True, exist_ok=True)
        _write_manifest(dest, dataset_id, cfg, result.files)
    return result


def verify_dataset(dataset_id: str, cfg: dict, root: Path) -> DatasetResult:
    """Offline verification of an already-downloaded dataset."""
    dest = Path(cfg["dest"])
    if not dest.is_absolute():
        dest = root / dest
    result = DatasetResult(dataset_id, "verified", str(dest))
    for file_cfg in cfg.get("files") or []:
        target = dest / file_cfg.get("target", file_cfg["name"])
        optional = bool(file_cfg.get("optional"))
        if not target.exists():
            status = "skipped_optional" if optional else "failed"
            result.files.append(FileResult(file_cfg["name"], status,
                                           message="missing"))
            if not optional:
                result.status = "failed"
            continue
        actual = sha256_file(target)
        pinned = file_cfg.get("sha256")
        if pinned and actual != pinned:
            result.files.append(FileResult(file_cfg["name"], "failed", str(target),
                                           sha256=actual, sha256_pinned=True,
                                           message="checksum mismatch"))
            result.status = "failed"
        else:
            result.files.append(FileResult(file_cfg["name"], "verified", str(target),
                                           sha256=actual, sha256_pinned=bool(pinned),
                                           size_bytes=target.stat().st_size))
    return result


def run(
    dataset_ids: list[str],
    root: Path | None = None,
    *,
    force: bool = False,
    offline: bool = False,
    verify_only: bool = False,
    retries: int = 3,
    timeout: int = 60,
) -> list[DatasetResult]:
    base = root or project_root()
    registry = load_registry(base)
    unknown = [d for d in dataset_ids if d not in registry]
    if unknown:
        raise DownloadError(
            f"unknown dataset ids: {unknown}; registered: {sorted(registry)}"
        )
    results = []
    for dataset_id in dataset_ids:
        results.append(
            download_dataset(
                dataset_id, registry[dataset_id], base,
                force=force, offline=offline, verify_only=verify_only,
                retries=retries, timeout=timeout,
            )
        )
    return results
