from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SplitRequest:
    strategy: str
    n_folds: int = 5
    seed: int = 20260815
    public_salt: str | None = None
    protocol_version: str = "v1"
    options: dict[str, Any] = field(default_factory=dict)
    allow_label_dependent_membership: bool = False

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")


@dataclass(frozen=True)
class FoldSplit:
    fold_index: int
    assignments: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitResult:
    strategy: str
    folds: tuple[FoldSplit, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

