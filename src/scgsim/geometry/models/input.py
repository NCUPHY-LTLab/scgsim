"""Adapter-normalized input records for route-aware geometry construction.

These immutable records preserve source facts and provenance before compiler
planning; compiled geometry records are defined in their own modules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.models.common import (
    ConductorPartRoleLiteral,
    ConductorRepresentationLiteral,
    PolygonRing,
    RouteLiteral,
)
from scgsim.geometry.models.lumped import LumpedSupport
from scgsim.geometry.models.regions import PortSheetRegionRecord


@dataclass(frozen=True)
class LayoutPolygonSpec:
    """Adapter-normalized polygon with stable frontend provenance.

    ``drawn_rings`` holds, for each ring of the polygon (the exterior, then each hole in
    the order of ``holes``), its vertices in the order and direction the GDS draws them:
    the polygon's walk from its own first vertex, carried through the occurrence
    transform and placed on the model grid as the source stage places it, in assembly
    micrometres.  Reconstruction indices name these vertices.  The source stage fills it
    where it reads the GDS, for every polygon it reads; an entry is None where that ring
    is not one drawn GDS polygon (fused from several walks, or a drawn vertex dropped).
    It stays empty when nothing filled it, as on a polygon built by hand.
    """

    polygon_id: str
    layer: str
    exterior: PolygonRing
    holes: tuple[PolygonRing, ...] = ()
    object_name: str | None = None
    net_name: str | None = None
    port_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    drawn_rings: tuple[PolygonRing | None, ...] = ()


@dataclass(frozen=True)
class SourceCurveSpec:
    """Component-authored XY geometry; control points are not topology vertices.

    A circular arc uses start, on-arc, and end points. B-spline knots are
    distinct values with corresponding multiplicities; an optional interval
    trims the native source parameter domain without changing its geometry.
    """

    kind: str
    points_um: tuple[tuple[float, float], ...]
    degree: int | None = None
    weights: tuple[float, ...] = ()
    knots: tuple[float, ...] = ()
    multiplicities: tuple[int, ...] = ()
    parameter_interval: tuple[float, float] | None = None


@dataclass(frozen=True)
class BoundaryCurveChainSpec:
    """Ordered source boundary bound to one Entity-owned source polygon.

    The selector chooses a polygon in that Entity, not an arbitrary surface.
    Hole indices address that selected polygon's recorded hole order.
    """

    boundary_id: str
    entity_id: str
    selector_point_um: tuple[float, float]
    curves: tuple[SourceCurveSpec, ...]
    role: str = "outer"
    hole_index: int | None = None
    source_occurrence_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GdsBoundaryReconstructionSpec:
    """Explicit caller selection of a polygon boundary reconstruction.

    Segment index ``i`` is the edge from vertex ``i`` to vertex ``i + 1`` of the
    selected ring as drawn in the GDS: the order and direction in which the GDS
    polygon walks it, from its own first vertex, carried through the occurrence
    transform, as the selected polygon's ``LayoutPolygonSpec.drawn_rings`` holds it.
    Anchor indices name drawn vertices in the same order.  Circular
    reconstruction uses three caller-selected vertex indices. Spline
    reconstruction uses the recorded boundary vertices, or explicit selected
    indices. No polygon-only input is implicitly reconstructed.
    """

    boundary_id: str
    entity_id: str
    selector_point_um: tuple[float, float]
    kind: str
    role: str = "outer"
    hole_index: int | None = None
    segment_indices: tuple[int, ...] = ()
    closed: bool = False
    anchor_indices: tuple[int, ...] = ()
    degree: int | None = None
    parameter_interval: tuple[float, float] | None = None
    source_occurrence_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# A refusal lists at most this many drawn vertices, then counts the rest.
_LISTED_VERTICES = 5


def ring_without_drawn_order(what: str, ring_place: str) -> str:
    """The refusal of a record whose ring has no drawn ring recorded for it, at the ring.

    ``what`` names the record, its Entity and its layer; ``ring_place`` is where the
    selected ring lies, in micrometres.
    """
    return (
        f"{what}: the ring it selects has no drawn GDS ring recorded for it: the ring at "
        f"{ring_place}. Its indices name no drawn edge: the ring is not one drawn GDS "
        "polygon (several fused together, or vertices the source stage dropped), or its "
        "polygon was not read from the GDS by the source stage. Fix the layout or the "
        "declaration: draw the ring as one GDS polygon, or reconstruct a ring that is."
    )


def ring_lost_drawn_order(what: str, findings: Sequence[str]) -> str:
    """The refusal of a ring that no longer holds its drawn vertices once each, in order.

    ``findings`` say, for each drawn vertex at fault, where it lies and what the ring does
    with it (:func:`drawn_vertices_off_ring` finds them).
    """
    shown = "; ".join(findings[:_LISTED_VERTICES])
    rest = len(findings) - _LISTED_VERTICES
    more = f"; and {rest} more drawn vertices" if rest > 0 else ""
    return (
        f"{what}: the ring it selects no longer holds each vertex of its drawn ring once, "
        f"in drawn order: {shown}{more}. Its drawn segments cannot be found on it. Fix the "
        "layout: draw the ring's vertices at least one grid step apart, so the source "
        "stage keeps each one."
    )


@dataclass(frozen=True)
class RingChain:
    """A reconstruction's chain and anchors as indices of the ring one stage holds."""

    segments: tuple[int, ...]
    anchors: tuple[int, ...]


def drawn_ring_of(polygon: LayoutPolygonSpec, ring_index: int) -> PolygonRing | None:
    """The drawn order of one ring of a polygon (0 the exterior, 1 + k hole k), or None.

    This is the one rule for which vertex order a ring's reconstruction indices name.  A
    polygon the source stage read from the GDS carries it in ``drawn_rings``; a None entry
    there means the ring is not one drawn GDS polygon.  A polygon without ``drawn_rings``
    that the source grid never emitted, as an input built by hand, holds its rings as they
    were given, and they are its drawn order.  A polygon the grid emitted without
    ``drawn_rings`` has lost its drawn order.  A ring no polygon holds, as an Entity's
    outline loop, is given as drawn.
    """
    rings = (polygon.exterior, *polygon.holes)
    if not 0 <= ring_index < len(rings):
        return None
    if polygon.drawn_rings:
        return (
            polygon.drawn_rings[ring_index]
            if ring_index < len(polygon.drawn_rings)
            else None
        )
    # The source grid stage records under this key every polygon it rounds and nodes.
    if "unquantized_source_geometry" in polygon.metadata:
        return None
    return rings[ring_index]


def drawn_chain_on_ring[Key](  # noqa: PLR0913 -- the declared chain, its closure, and the two rings it joins
    segments: Sequence[int],
    anchors: Sequence[int],
    closed: bool,
    drawn: Sequence[Key],
    ring: Sequence[Key],
) -> RingChain | None:
    """Carry drawn segments and anchors onto a ring that holds every drawn vertex once.

    ``drawn`` and ``ring`` are vertex keys on one lattice.  Drawn segment ``i`` joins drawn
    vertices ``i`` and ``i + 1``; on the ring it is the run of segments between those two
    vertices, which holds any crossing the grid noded into the drawn segment.  The ring may
    start at another vertex and may run the other way, as a mirrored occurrence or a
    re-oriented ring does.  The chain comes back in the ring's own order, as the binding
    stages walk it, and an open chain's anchors are reordered with it so they still start
    and end it.  None when the ring lacks a drawn vertex, holds one twice, or meets the
    drawn vertices in another order.
    """
    places = _places_on_ring(drawn, ring)
    if places is None:
        return None
    count = len(ring)
    step = _walk_direction(places, count)
    if step is None:
        return None

    def run(i: int) -> tuple[int, ...]:
        start, end = places[i], places[(i + 1) % len(places)]
        if step < 0:
            start, end = end, start
        return tuple((start + k) % count for k in range((end - start) % count))

    order = segments if step > 0 else tuple(reversed(segments))
    ring_anchors = tuple(places[a] for a in anchors)
    if step < 0 and not closed:
        ring_anchors = ring_anchors[::-1]
    return RingChain(tuple(j for i in order for j in run(i)), ring_anchors)


def drawn_vertices_off_ring[Key](
    drawn: Sequence[Key], ring: Sequence[Key]
) -> tuple[tuple[int, str], ...]:
    """The drawn vertices that keep ``drawn_chain_on_ring`` from finding the chain, and why.

    Each is a drawn vertex index with what the ring does with it: it lacks the vertex,
    holds it more than once, or meets it out of drawn order (the next drawn vertex along
    the ring is not its neighbour in the drawing, either way round).
    """
    where: dict[Key, list[int]] = {}
    for index, key in enumerate(ring):
        where.setdefault(key, []).append(index)
    held = [len(where.get(key, ())) for key in drawn]
    if any(count != 1 for count in held):
        return tuple(
            (index, "is missing from it" if count == 0 else f"is held {count} times")
            for index, count in enumerate(held)
            if count != 1
        )
    places = [where[key][0] for key in drawn]
    count = len(drawn)
    along = sorted(range(count), key=places.__getitem__)
    after = {along[k]: along[(k + 1) % count] for k in range(count)}
    # Read the ring in whichever direction keeps more of the drawn order.
    breaks = min(
        (
            [i for i in range(count) if after[i] != (i + step) % count]
            for step in (1, -1)
        ),
        key=len,
    )
    return tuple((index, "is out of drawn order") for index in breaks)


def _places_on_ring[Key](
    drawn: Sequence[Key], ring: Sequence[Key]
) -> tuple[int, ...] | None:
    where: dict[Key, list[int]] = {}
    for index, key in enumerate(ring):
        where.setdefault(key, []).append(index)
    places = [where.get(key, []) for key in drawn]
    if any(len(found) != 1 for found in places):
        return None
    return tuple(found[0] for found in places)


def _walk_direction(places: Sequence[int], count: int) -> int | None:
    # The ring meets the drawn vertices in drawn order exactly when the forward gaps
    # between consecutive drawn vertices add up to one turn of the ring.
    pairs = list(zip(places, (*places[1:], places[0]), strict=True))
    if sum((after - before) % count for before, after in pairs) == count:
        return 1
    if sum((before - after) % count for before, after in pairs) == count:
        return -1
    return None


@dataclass(frozen=True)
class VacuumRegionSpec:
    """Cross-layer vacuum padding specification for auto envelope derivation."""

    x_minus_um: float
    x_plus_um: float
    y_minus_um: float
    y_plus_um: float
    z_minus_um: float
    z_plus_um: float

    @classmethod
    def from_padding(
        cls,
        value: float | Sequence[float] | Mapping[str, Any],
    ) -> VacuumRegionSpec:
        """Normalize scalar, 3-sequence, or sparse six-face vacuum padding."""
        if isinstance(value, bool) or value is None:
            raise TypeError(
                "vacuum padding must be a number, 3-sequence, or six-face mapping."
            )

        if isinstance(value, (int, float)):
            base = _non_negative_float(value, "padding")
            return cls(base, base, base, base, base, base)

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 3:
                raise ValueError(
                    "vacuum 3-sequence padding must contain exactly 3 items for x/y/z."
                )
            x_padding, y_padding, z_padding = (
                _non_negative_float(value[0], "x_padding"),
                _non_negative_float(value[1], "y_padding"),
                _non_negative_float(value[2], "z_padding"),
            )
            return cls(
                x_padding,
                x_padding,
                y_padding,
                y_padding,
                z_padding,
                z_padding,
            )

        if not isinstance(value, Mapping):
            raise TypeError("vacuum padding must be a number, 3-sequence, or mapping.")

        required = {
            "x_minus_um",
            "x_plus_um",
            "y_minus_um",
            "y_plus_um",
            "z_minus_um",
            "z_plus_um",
        }
        keys = set(value.keys())
        if not keys <= required:
            extras = tuple(sorted(keys - required))
            raise ValueError(
                f"vacuum face padding mapping has unsupported keys: {extras!r}."
            )

        face_map = {
            key: _non_negative_float(value.get(key, 0.0), key) for key in required
        }
        return cls(**{key: value for key, value in face_map.items()})


def _non_negative_float(value: Any, field: str) -> float:
    if value is None:
        raise ValueError(f"vacuum padding field {field!r} is required.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"vacuum padding field {field!r} must be numeric.")
    text = float(value)
    if not math.isfinite(text):
        raise ValueError(f"vacuum padding field {field!r} must be finite.")
    if text < 0.0:
        raise ValueError(f"vacuum padding field {field!r} must be non-negative.")
    return text


@dataclass(frozen=True)
class SemanticEntitySpec:
    """Stable semantic object before route-aware construction planning.

    `priority`: the HIGHER value wins between overlapping solution regions of
    the same material kind; a dielectric region wins over a vacuum region
    before priority is compared. PDK `mesh_order` (lower wins) is converted by
    negation in source normalization.
    """

    semantic_id: str
    role: str
    material_id: str
    material_kind: str
    priority: int
    geometry_kind: str
    part_role: ConductorPartRoleLiteral | None = None
    attached_face_metal_semantic_id: str | None = None
    net_id: str | None = None
    polygon_ids: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    host_void_semantic_id: str | None = None
    requires_construction_body: bool = False
    route_representations: Mapping[
        RouteLiteral,
        ConductorRepresentationLiteral,
    ] = field(default_factory=dict)
    geometry: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryBuildInput:
    """Adapter boundary for route-aware semantic geometry construction.

    `port_sheet_regions` are pre-planning 2D region-layer records for Palace
    lumped-port sheets. They are not backend topology and must not become live
    surfaces without an explicit local-fragment lowering step.
    """

    polygons: tuple[LayoutPolygonSpec, ...]
    entities: tuple[SemanticEntitySpec, ...]
    solution_regions: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    port_sheet_regions: tuple[PortSheetRegionRecord, ...] = ()
    boundary_curves: tuple[BoundaryCurveChainSpec, ...] = ()
    boundary_reconstruction: tuple[GdsBoundaryReconstructionSpec, ...] = ()
    lumped_supports: tuple[LumpedSupport, ...] = ()
