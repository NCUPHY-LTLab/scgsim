"""Pure owner projections for declared surfaces and legacy interfaces.

Each projection preserves source order because ownership order is part of the
semantic record consumed by geometry adapters.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast


def unique_ids(values: Iterable[object]) -> tuple[str, ...]:
    """Convert identifiers to text while keeping only their first occurrence."""
    result: list[str] = []
    for value in values:
        text = str(value)
        if text not in result:
            result.append(text)
    return tuple(result)


def surface_declared_owner_ids(raw: object, fallback: str) -> tuple[str, ...]:
    """Read declared owners, treating a scalar string as one identifier."""
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, Sequence):
        return tuple(str(value) for value in raw)
    return (fallback,)


def interface_surface_owner_ids(
    explicit_owner_ids: object | None,
    *,
    route_a_sheet_owner_ids: Sequence[object] | None,
    fallback_owner_ids: Sequence[object],
) -> tuple[str, ...]:
    """Choose explicit owners, Route A sheet owners, or the declared fallback."""
    if explicit_owner_ids is not None:
        if isinstance(explicit_owner_ids, str):
            return (explicit_owner_ids,)
        # Legacy declarations accept any iterable after preserving scalar text.
        return unique_ids(cast(Iterable[object], explicit_owner_ids))
    if route_a_sheet_owner_ids is not None:
        return unique_ids(route_a_sheet_owner_ids)
    return tuple(str(value) for value in fallback_owner_ids)


def surface_physical_owner_ids(
    explicit_owner_ids: object | None,
    declared_owner_ids: Sequence[object],
) -> tuple[str, ...]:
    """Resolve physical owners while preserving explicit list order."""
    if isinstance(explicit_owner_ids, str):
        return (explicit_owner_ids,)
    if isinstance(explicit_owner_ids, Sequence):
        return tuple(str(value) for value in explicit_owner_ids)
    return tuple(str(value) for value in declared_owner_ids)


def physical_group_owner_ids(
    owner_ids: Sequence[object],
    resolved_group_ids: dict[str, str],
) -> tuple[str, ...]:
    """Map semantic owners to resolved groups and deduplicate in source order."""
    return unique_ids(
        resolved_group_ids.get(str(owner), str(owner)) for owner in owner_ids
    )
