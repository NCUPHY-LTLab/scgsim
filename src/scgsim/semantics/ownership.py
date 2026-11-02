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


def project_legacy_interface_record_owners(
    owners: tuple[str, ...],
    boundary_volume_ids: object,
    *,
    surface_id: str,
) -> tuple[str, ...]:
    """Project oversized legacy ownership onto its explicit boundary pair.

    Existing records with at most two owners retain their order unchanged. A
    larger record must provide the two boundary owners rather than guessing.
    """
    if len(owners) <= 2:
        return owners
    # Legacy records supply this field as an iterable, not a scalar.
    raw_boundary_ids = (
        ()
        if isinstance(boundary_volume_ids, str)
        else cast(Iterable[object], boundary_volume_ids)
    )
    boundary_ids = tuple(str(value) for value in raw_boundary_ids)
    if (
        len(boundary_ids) != 2
        or len(set(boundary_ids)) != 2
        or not set(boundary_ids).issubset(owners)
    ):
        raise ValueError(
            f"{surface_id} structured interface requires exactly two "
            "distinct boundary_volume_ids from its owners"
        )
    return (boundary_ids[0], boundary_ids[1])


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
