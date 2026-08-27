from __future__ import annotations

import os
from pathlib import Path

import pytest

from fitness_agents.config import LocalKnowledgeRootConfig
from fitness_agents.local_knowledge.parsers import discover_local_files
from fitness_agents.safety import WorkspaceAccessPolicy


def test_workspace_policy_unions_machine_and_editor_denies(tmp_path: Path) -> None:
    denied = "knowledge/blocked.md"
    (tmp_path / "AvoidRead.txt").write_text(denied + "\n", encoding="utf-8")
    rule = tmp_path / ".cursor" / "rules" / "avoid-biosafety-content.mdc"
    rule.parent.mkdir(parents=True)
    rule.write_text(
        "Do not access:\n"
        f"- `{denied}`\n"
        "All other content, including `knowledge/allowed.md`, is allowed.\n",
        encoding="utf-8",
    )

    policy = WorkspaceAccessPolicy.load(tmp_path)

    assert policy.decide(tmp_path / denied).allowed is False
    assert policy.decide(tmp_path / "knowledge" / "allowed.md").allowed is True
    assert policy.public_manifest()["denied_path_count"] == 1
    assert len(policy.policy_hash) == 64


def test_denied_local_knowledge_path_is_skipped_before_stat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()
    allowed = root / "allowed.md"
    blocked = root / "blocked.md"
    allowed.write_text("Neutral synthetic evidence.", encoding="utf-8")
    blocked.write_text("This file must never be inspected.", encoding="utf-8")
    (tmp_path / "AvoidRead.txt").write_text("knowledge/blocked.md\n", encoding="utf-8")

    original_stat = Path.stat

    def audited_stat(path: Path, *args, **kwargs):
        if path.absolute() == blocked.absolute():
            raise AssertionError("denied path reached Path.stat")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", audited_stat)
    events: list[dict[str, str]] = []
    files = discover_local_files(
        (
            LocalKnowledgeRootConfig(
                path=root,
                root_id="SYNTHETIC",
                runtime_manifest_mode="legacy_compatible",
                include=("**/*.md",),
                exclude=(),
            ),
        ),
        follow_symlinks=False,
        policy_events=events,
    )

    assert [item.path.name for item in files] == ["allowed.md"]
    assert len(events) == 1
    assert events[0]["event"] == "path_denied_before_stat"
    assert events[0]["relative_path"] == "blocked.md"
    assert len(events[0]["policy_hash"]) == 64


def test_production_root_without_exact_manifest_never_falls_back_to_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()
    (tmp_path / "AvoidRead.txt").write_text(
        "unrelated/blocked.md\n",
        encoding="utf-8",
    )

    def forbidden_walk(*args, **kwargs):
        raise AssertionError("production discovery attempted a recursive walk")

    monkeypatch.setattr(os, "walk", forbidden_walk)

    with pytest.raises(FileNotFoundError, match="runtime-files.json"):
        discover_local_files(
            (
                LocalKnowledgeRootConfig(
                    path=root,
                    root_id="SYNTHETIC",
                    access_policy_mode="required",
                    runtime_manifest_mode="required",
                ),
            ),
            follow_symlinks=False,
        )


def test_production_root_without_workspace_policy_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()

    with pytest.raises(PermissionError, match="workspace access policy"):
        discover_local_files(
            (
                LocalKnowledgeRootConfig(
                    path=root,
                    root_id="SYNTHETIC",
                    access_policy_mode="required",
                    runtime_manifest_mode="required",
                ),
            ),
            follow_symlinks=False,
        )
