import yaml

from fitness_agents.config import project_root


def test_flip_download_uses_current_repository_and_checksum_fallbacks():
    root = project_root()
    script = (project_root() / "scripts/data/download_flip_gb1.sh").read_text(encoding="utf-8")
    registry = yaml.safe_load((root / "configs/data/datasets.yaml").read_text(encoding="utf-8"))
    gb1 = registry["datasets"]["flip_gb1"]
    archive = gb1["files"][0]

    assert "download_profile.py" in script
    assert "--dataset flip_gb1" in script
    assert gb1["version"] == "62cace8735f5610e2743cf06ce0f944b37fffaa6"
    assert archive["sha256"] == (
        "85692d808dcd3ae54fa2ac31f4e590858d4582369b6c7b05df299b9b6c383bff"
    )
    assert any("J-SNACKKB/FLIP" in url for url in archive["urls"])
    assert all("facebookresearch/FLIP" not in url for url in archive["urls"])


def test_mvp_assay_list_excludes_out_of_scope_receptor_binding_assays():
    assays = (project_root() / "configs/data/proteingym_mvp_assays.txt").read_text(
        encoding="utf-8"
    )
    assert "SPIKE_SARS2" not in assays
    assert "SARS-CoV-2" not in assays


def test_readme_documents_current_flip_owner():
    readme = (project_root() / "README.md").read_text(encoding="utf-8")
    assert "J-SNACKKB/FLIP" in readme
