from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

HGVS_SHORT = re.compile(r"^([A-Z])(\d+)([A-Z*])$")
COMPONENT_PREFIXED = re.compile(r"^([A-Za-z0-9_]+):([A-Z])(\d+)([A-Z*])$")


class InvalidMutationNotation(ValueError):
    """Raised when a mutation string cannot be parsed into discrete edits."""


@dataclass(frozen=True)
class MutationEdit:
    wt: str
    position: int
    mutant: str
    component: str | None = None
    backbone: str | None = None

    @property
    def hgvs_short(self) -> str:
        return f"{self.wt}{self.position}{self.mutant}"

    @property
    def identity(self) -> tuple[int, str, str]:
        return (self.position, self.wt, self.mutant)


def edits_from_tokens(tokens: Sequence[object]) -> tuple[MutationEdit, ...]:
    edits: list[MutationEdit] = []
    for value in tokens:
        item = json.loads(value) if isinstance(value, str) else value
        if not isinstance(item, (list, tuple)) or len(item) != 5:
            raise InvalidMutationNotation(f"Mutation token must be a 5-field record: {value!r}")
        backbone, component, position, wild_type, mutant = item
        edits.append(
            MutationEdit(
                wt=str(wild_type),
                position=int(position),
                mutant=str(mutant),
                component=str(component) if component not in (None, "") else None,
                backbone=str(backbone) if backbone not in (None, "") else None,
            )
        )
    return tuple(edits)


def edits_from_site_code(
    code: str,
    *,
    wild_type: str,
    positions: Iterable[int],
    component: str | None = None,
    backbone: str | None = None,
) -> tuple[MutationEdit, ...]:
    position_list = tuple(int(item) for item in positions)
    if len(code) != len(wild_type):
        raise InvalidMutationNotation(
            f"Expected {len(wild_type)} residues, received {code!r}"
        )
    if len(position_list) != len(wild_type):
        raise InvalidMutationNotation("Mutable-position mapping does not match WT")
    return tuple(
        MutationEdit(
            wt=wt,
            position=position,
            mutant=mutant,
            component=component,
            backbone=backbone,
        )
        for wt, position, mutant in zip(wild_type, position_list, code, strict=True)
        if wt != mutant
    )


def format_canonical(edits: Sequence[MutationEdit]) -> str:
    if not edits:
        return "WT"
    ordered = sorted(
        edits,
        key=lambda item: ((item.component or ""), item.position, item.wt, item.mutant),
    )
    components = {item.component for item in ordered if item.component}
    include_component = len(components) > 1
    parts: list[str] = []
    for edit in ordered:
        token = edit.hgvs_short
        if include_component and edit.component:
            token = f"{edit.component}:{token}"
        parts.append(token)
    return ";".join(parts)


def _parse_one_token(token: str) -> MutationEdit | None:
    match = HGVS_SHORT.fullmatch(token)
    if match:
        wild_type, raw_position, mutant = match.groups()
        return MutationEdit(wt=wild_type, position=int(raw_position), mutant=mutant)
    match = COMPONENT_PREFIXED.fullmatch(token)
    if match:
        component, wild_type, raw_position, mutant = match.groups()
        if HGVS_SHORT.fullmatch(component):
            return None
        return MutationEdit(
            wt=wild_type,
            position=int(raw_position),
            mutant=mutant,
            component=component,
        )
    return None


def _parse_proteingym(raw: str) -> tuple[MutationEdit, ...] | None:
    parts = [item.strip() for item in raw.split(":") if item.strip()]
    if len(parts) < 2:
        return None
    edits: list[MutationEdit] = []
    for part in parts:
        match = HGVS_SHORT.fullmatch(part)
        if match is None:
            return None
        wild_type, raw_position, mutant = match.groups()
        edits.append(MutationEdit(wt=wild_type, position=int(raw_position), mutant=mutant))
    return tuple(edits)


def parse_mutation_notation(text: str) -> tuple[MutationEdit, ...]:
    raw = (text or "").strip()
    if not raw or raw.upper() == "WT":
        return ()
    if raw[0] in "[{":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise InvalidMutationNotation(f"Malformed mutation notation: {raw!r}") from error
        if isinstance(parsed, list):
            return edits_from_tokens(parsed)
    if ";" in raw:
        edits = []
        for part in (item.strip() for item in raw.split(";")):
            if not part:
                continue
            edit = _parse_one_token(part)
            if edit is None:
                raise InvalidMutationNotation(f"Malformed mutation token: {part!r}")
            edits.append(edit)
        return tuple(edits)
    single = _parse_one_token(raw)
    if single is not None:
        return (single,)
    proteingym = _parse_proteingym(raw)
    if proteingym is not None:
        return proteingym
    raise InvalidMutationNotation(f"Malformed mutation notation: {raw!r}")
