"""Shared fail-closed safety policies for filesystem and external evidence access."""

from .access_policy import (
    AccessDecision,
    WorkspaceAccessPolicy,
    discover_workspace_access_policy,
)

__all__ = [
    "AccessDecision",
    "WorkspaceAccessPolicy",
    "discover_workspace_access_policy",
]
