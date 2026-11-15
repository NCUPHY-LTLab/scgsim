"""Resolve the face metal a contact pad attaches to: same Z range, finite overlap, one net."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, cast

from scgsim.geometry.grid.booleans import polygon_boolean
from scgsim.geometry.grid.planar_loops import LoopCoordinates
from scgsim.geometry.grid.scope import UNGRIDDED
from scgsim.geometry.models.input import SemanticEntitySpec
from scgsim.refusal import Declared

BOOLEAN_AREA_EPS_UM2 = 1e-8

BOOLEAN_RELATIVE_AREA_EPS = 1e-12


def _pad(pad: SemanticEntitySpec) -> Declared:
    """A contact pad, whose attachment its stack layer declares."""
    return Declared(
        f"contact pad {pad.semantic_id!r}",
        "its stack layer: attached_face_metal_semantic_id, Z and Net",
    )


class _PolygonWithArea(Protocol):
    """Polygon shape shared by native and grid Boolean results."""

    @property
    def points(self) -> LoopCoordinates:
        """Return the polygon boundary walk used for area checks."""
        ...

    def area(self) -> float:
        """Return the polygon area in square micrometres."""
        ...


def _declared_face_metals(
    pad: SemanticEntitySpec, entities: Sequence[SemanticEntitySpec]
) -> tuple[SemanticEntitySpec, ...]:
    """The face-metal Entities the pad's attached id names, by id or by semantic group."""
    attached_id = pad.attached_face_metal_semantic_id
    candidates = tuple(
        candidate
        for candidate in entities
        if attached_id
        in {
            candidate.semantic_id,
            str(candidate.metadata.get("semantic_group_id", "")),
        }
    )
    if not candidates:
        raise ValueError(
            _pad(pad).refusal("its attached face metal id names no Entity")
        )
    if any(candidate.part_role != "face_metal" for candidate in candidates):
        raise ValueError(
            _pad(pad).refusal("its attachment must be a face_metal Entity")
        )
    return candidates


def resolve_contact_pad_attachment(
    pad: SemanticEntitySpec,
    entities: Sequence[SemanticEntitySpec],
    *,
    finite_overlap: Callable[[SemanticEntitySpec, SemanticEntitySpec], bool]
    | None = None,
    defer_overlap: bool = False,
) -> SemanticEntitySpec | None:
    """Resolve attachment, or check declarations before native curved overlap exists."""
    candidates = _declared_face_metals(pad, entities)
    matches = tuple(
        candidate
        for candidate in candidates
        if _same_entity_z_range(pad, candidate)
        and (
            defer_overlap
            or (finite_overlap or entities_have_finite_overlap)(pad, candidate)
        )
    )
    if not matches:
        raise ValueError(
            _pad(pad).refusal(
                "it needs a finite overlap, at the same Z, with its attached face_metal"
            )
        )
    if defer_overlap:
        # Actual curved overlap selects the attachment and proves uniqueness
        # later. Here only authored identity, role, Z and Net can be checked.
        if not pad.net_id or not any(
            candidate.net_id == pad.net_id for candidate in matches
        ):
            raise ValueError(
                _pad(pad).refusal(
                    "the contact_pad and its attached M1 require one equal net"
                )
            )
        return None
    if len(matches) != 1:
        raise ValueError(
            _pad(pad).refusal(
                "its attachment is ambiguous: it overlaps several face metals at its Z"
            )
        )
    attached = matches[0]
    if not pad.net_id or not attached.net_id or attached.net_id != pad.net_id:
        raise ValueError(
            _pad(pad).refusal(
                "the contact_pad and its attached M1 require one equal net"
            )
        )
    return attached


def _same_entity_z_range(first: SemanticEntitySpec, second: SemanticEntitySpec) -> bool:
    """Return whether two entities share both Z range endpoints."""
    first_min, first_max = _entity_z_range(first)
    second_min, second_max = _entity_z_range(second)
    return abs(first_min - second_min) <= 1e-9 and abs(first_max - second_max) <= 1e-9


def _entity_z_range(entity: SemanticEntitySpec) -> tuple[float, float]:
    """Return an entity's base Z and base Z plus thickness."""
    z_min = float(entity.geometry.get("z_min_um", entity.geometry.get("z_um", 0.0)))
    return z_min, z_min + float(entity.geometry.get("thickness_um", 0.0))


def entities_have_finite_overlap(
    first: SemanticEntitySpec, second: SemanticEntitySpec
) -> bool:
    """Check Boolean overlap area against absolute and relative noise floors."""
    overlap = polygon_boolean(
        UNGRIDDED,
        _entity_occupied_region(first),
        _entity_occupied_region(second),
        "and",
        precision=1e-9,
    )
    return any(
        _polygon_has_finite_area(cast(_PolygonWithArea, polygon))
        for polygon in overlap or ()
    )


def _polygon_has_finite_area(polygon: _PolygonWithArea) -> bool:
    """Apply the existing absolute and relative overlap-area thresholds."""
    return abs(_polygon_area(polygon.points)) > max(
        BOOLEAN_AREA_EPS_UM2,
        abs(polygon.area()) * BOOLEAN_RELATIVE_AREA_EPS,
    )


def _entity_occupied_region(
    entity: SemanticEntitySpec,
) -> tuple[_PolygonWithArea, ...]:
    """Build the entity's native outer polygon with authored holes removed."""
    import gdstk  # noqa: PLC0415 -- gdstk is only needed for native curved overlap validation

    outer_loop = entity.geometry.get("outer_loop")
    if outer_loop is None:
        return ()
    outer = cast(_PolygonWithArea, gdstk.Polygon(outer_loop))
    holes = tuple(
        cast(_PolygonWithArea, gdstk.Polygon(loop))
        for loop in entity.geometry.get("hole_loops", ())
    )
    if not holes:
        return (outer,)
    return tuple(
        cast(_PolygonWithArea, polygon)
        for polygon in polygon_boolean(
            UNGRIDDED, (outer,), holes, "not", precision=1e-9
        )
        or ()
    )


def _polygon_area(points: Iterable[Sequence[float]]) -> float:
    """Compute signed shoelace area for a closed boundary walk."""
    ring = tuple((float(point[0]), float(point[1])) for point in points)
    return 0.5 * sum(
        start[0] * end[1] - start[1] * end[0]
        for start, end in zip(ring, (*ring[1:], ring[0]), strict=True)
    )
