from __future__ import annotations

from fitness_agents.data.specs import DatasetSpec


def split_component_sequences(value: str, spec: DatasetSpec) -> tuple[str, ...]:
    sequence = str(value).strip().upper()
    expected = len(spec.components)
    if expected == 1:
        return (sequence,)
    parts = tuple(part.strip().upper() for part in sequence.split(spec.component_separator))
    if len(parts) != expected:
        raise ValueError(
            f"Expected {expected} components separated by {spec.component_separator!r}; "
            f"received {value!r}"
        )
    # FLIP-2 paired releases can encode a single-component mutant with an empty side of
    # `component_a:component_b`. An empty side means the configured reference component;
    # it must never be inferred from other rows.
    return tuple(
        part if part else component.reference
        for part, component in zip(parts, spec.components, strict=True)
    )
