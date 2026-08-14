"""Manual end-to-end test for the registry download engine (offline, synthetic data).

Run: python tests/_manual_download_engine_test.py
"""
import gzip as gzmod
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitness_agents.data.download import download_dataset

tmp = Path(tempfile.mkdtemp(prefix="dltest_"))
src = tmp / "upstream"
src.mkdir()
zip_path = src / "pack.zip"
with zipfile.ZipFile(zip_path, "w") as z:
    z.writestr("nested/data.csv", "variant,fitness\nA1B,1.0\nC2D,0.5\n")
    z.writestr("nested/junk.txt", "ignore me")
digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
url = zip_path.as_uri()

base = {"version": "test-1", "source_type": "test", "license": "x", "citation": "y",
        "adapter": "t", "dest": str(tmp / "out")}
cfg_tofu = dict(base, files=[{"name": "pack.zip", "urls": [url, "file:///nonexistent"],
    "sha256": None, "archive_type": "zip", "extract": True, "extract_members": ["*.csv"]}])

# 1. TOFU download + whitelist extraction + manifest
r = download_dataset("t", cfg_tofu, tmp)
assert r.status == "ok", r
f = r.files[0]
assert f.status == "downloaded" and not f.sha256_pinned and f.url_used == url
assert f.extracted_members == ["nested/data.csv"], f.extracted_members
assert (tmp / "out" / "data.csv").exists()  # single matched member is flattened
assert not (tmp / "out" / "nested" / "junk.txt").exists()
manifest = json.loads((tmp / "out" / "download_manifest.json").read_text())
assert manifest["files"][0]["sha256"] == digest
print("1 TOFU download + whitelist extract + manifest: OK")

# 2. second run reuses verified file
r2 = download_dataset("t", cfg_tofu, tmp)
assert r2.files[0].status == "reused"
print("2 reuse: OK")

# 3. pinned checksum passes
cfg_pin = dict(base, files=[dict(cfg_tofu["files"][0], sha256=digest)])
r3 = download_dataset("t", cfg_pin, tmp)
assert r3.files[0].status == "reused" and r3.files[0].sha256_pinned
print("3 pinned checksum reuse: OK")

# 4. corrupted file -> mirror fallback re-download
(tmp / "out" / "pack.zip").write_bytes(b"corrupt")
cfg_bad_first = dict(base, files=[dict(cfg_tofu["files"][0], sha256=digest,
    urls=["file:///nonexistent", url])])
r4 = download_dataset("t", cfg_bad_first, tmp)
assert r4.files[0].status == "downloaded" and r4.files[0].url_used == url
assert hashlib.sha256((tmp / "out" / "pack.zip").read_bytes()).hexdigest() == digest
print("4 mirror fallback + re-download after corruption: OK")

# 5. offline semantics
cfg_off = dict(base, dest=str(tmp / "out2"), files=[dict(cfg_tofu["files"][0])])
r5 = download_dataset("t", cfg_off, tmp, offline=True)
assert r5.status == "failed"
cfg_opt = dict(base, dest=str(tmp / "out3"), files=[dict(cfg_tofu["files"][0], optional=True)])
r6 = download_dataset("t", cfg_opt, tmp, offline=True)
assert r6.status == "partial" and r6.files[0].status == "skipped_optional"
print("5 offline semantics: OK")

# 6. whitelist matching nothing -> error
cfg_none = dict(base, dest=str(tmp / "out4"),
    files=[dict(cfg_tofu["files"][0], extract_members=["*.doesnotexist"])])
r7 = download_dataset("t", cfg_none, tmp)
assert r7.status == "failed"
print("6 whitelist-miss guard: OK")

# 7. path traversal member rejected
evil = src / "evil.zip"
with zipfile.ZipFile(evil, "w") as z:
    z.writestr("../escape.csv", "x")
cfg_evil = dict(base, dest=str(tmp / "out5"),
    files=[{"name": "evil.zip", "urls": [evil.as_uri()], "archive_type": "zip", "extract": True}])
r8 = download_dataset("t", cfg_evil, tmp)
assert r8.status == "failed" and "unsafe" in (r8.message or "")
print("7 path-traversal guard: OK")

# 8. gz single-file extraction
gz_path = src / "table.csv.gz"
with gzmod.open(gz_path, "wb") as h:
    h.write(b"a,b\n1,2\n")
cfg_gz = dict(base, dest=str(tmp / "out6"),
    files=[{"name": "table.csv.gz", "urls": [gz_path.as_uri()], "archive_type": "gz", "extract": True}])
r9 = download_dataset("t", cfg_gz, tmp)
assert (tmp / "out6" / "table.csv").exists()
print("8 gz extraction: OK")

print("ALL ENGINE TESTS PASSED")
