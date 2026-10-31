"""Distinct pure owner projections used by SGB adapters."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def unique_ids(values: Iterable[object]) -> tuple[str, ...]:
    """Stringify ids and drop repeats, keeping first-seen order."""
    result: list[str] = []
    for value in values:
        text = str(value)
        if text not in result:
            result.append(text)
    return tuple(result)


def surface_declared_owner_ids(raw: object, fallback: str) -> tuple[str, ...]:
    """Read the owners a surface declares; fall back to the given id when none are declared."""
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
    """Reduce an old-style interface record with more than two owners to its two boundary volumes."""
    if len(owners) <= 2:
        return owners
    boundary_ids = (
        ()
        if isinstance(boundary_volume_ids, str)
        else tuple(str(value) for value in boundary_volume_ids)  # type: ignore[arg-type]
    )
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
    """Pick interface owners: explicit ids first, then Route A sheet owners, then the fallback."""
    if explicit_owner_ids is not None:
        if isinstance(explicit_owner_ids, str):
            return (explicit_owner_ids,)
        return unique_ids(explicit_owner_ids)  # type: ignore[arg-type]
    if route_a_sheet_owner_ids is not None:
        return unique_ids(route_a_sheet_owner_ids)
    return tuple(str(value) for value in fallback_owner_ids)


def surface_physical_owner_ids(
    explicit_owner_ids: object | None,
    declared_owner_ids: Sequence[object],
) -> tuple[str, ...]:
    """Pick the owners of a surface's physical group: explicit ids if given, else the declared ones."""
    if isinstance(explicit_owner_ids, str):
        return (explicit_owner_ids,)
    if isinstance(explicit_owner_ids, Sequence):
        return tuple(str(value) for value in explicit_owner_ids)
    return tuple(str(value) for value in declared_owner_ids)


def physical_group_owner_ids(
    owner_ids: Sequence[object],
    resolved_group_ids: dict[str, str],
) -> tuple[str, ...]:
    """Map owners to their resolved physical-group ids and drop repeats."""
    return unique_ids(
        resolved_group_ids.get(str(owner), str(owner)) for owner in owner_ids
    )
