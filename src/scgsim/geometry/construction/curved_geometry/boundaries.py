"""Bind authored boundaries and explicit reconstruction to original source rings.

The junction partition binds a lumped port's terminal conductors here before SGB's own
binding (``sgb.prisms.curves``) runs, so for those conductors a refusal here is the one the
layout author reads.  It names the record, its Entity and the ring's layer, says where in
micrometres the ring, or the indices at fault, lie, and that the declaration must change.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import cast

from scgsim.geometry.models.common import Coordinate, PolygonRing
from scgsim.geometry.models.grid import (
    format_extent,
    format_point,
    geometry_grid_context,
)
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
    SourceCurveSpec,
    drawn_chain_on_ring,
    drawn_ring_of,
    drawn_vertices_off_ring,
    reconstruction_chain_refusal,
    ring_lost_drawn_order,
    ring_without_drawn_order,
)
from scgsim.geometry.models.junctions import is_junction_end_piece
from scgsim.geometry.primitives.grid_regions import region_key
from scgsim.geometry.source.curves import (
    BoundaryKey,
    SourceRings,
    boundary_binding,
    select_boundary,
)

BoundaryChains = dict[BoundaryKey, tuple[SourceCurveSpec, ...]]
ReconstructionSelection = tuple[tuple[int, ...], tuple[SourceCurveSpec, ...], str]


def source_z_range(entity: SemanticEntitySpec) -> tuple[float, float]:
    """Read the source-authored lower and upper Z coordinates."""
    geometry = entity.geometry
    lower = float(geometry.get("z_min_um", geometry.get("z_um", 0.0)))
    return lower, float(
        geometry.get("z_max_um", lower + float(geometry.get("thickness_um", 0.0)))
    )


def source_rings(build_input: GeometryBuildInput) -> SourceRings:
    """Use adapter-normalized Entity-owned logical polygons, including holes."""
    polygons = {p.polygon_id: p for p in build_input.polygons}
    result: SourceRings = {}
    for entity in build_input.entities:
        if entity.polygon_ids:
            result[entity.semantic_id] = [
                (pid, polygons[pid].exterior, polygons[pid].holes)
                for pid in entity.polygon_ids
            ]
        else:
            outer = entity.geometry.get("outer_loop")
            if entity.metadata.get("is_auto_vacuum_region"):
                outer = entity.metadata.get("auto_vacuum_envelope_outer_loop", outer)
            if outer is None:
                bounds = entity.geometry.get(
                    "domain_bounds_um", entity.geometry.get("bounds_um")
                )
                if bounds is None:
                    raise ValueError(
                        f"{entity.semantic_id} has no source planar footprint"
                    )
                outer = (
                    (bounds["x_min_um"], bounds["y_min_um"]),
                    (bounds["x_max_um"], bounds["y_min_um"]),
                    (bounds["x_max_um"], bounds["y_max_um"]),
                    (bounds["x_min_um"], bounds["y_max_um"]),
                )
            result[entity.semantic_id] = [
                (
                    entity.semantic_id,
                    outer,
                    ()
                    if entity.metadata.get("is_auto_vacuum_region")
                    else entity.geometry.get("hole_loops", ()),
                )
            ]
    return result


def circle_through_points(points: Sequence[Coordinate]) -> tuple[Coordinate, float]:
    """Solve the circle through three supplied arc anchors."""
    a, b, c = points
    ax, ay = a
    bx, by = b
    cx, cy = c
    determinant = 2 * _twice_area(a, b, c)
    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / determinant
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / determinant
    return (ux, uy), math.hypot(ax - ux, ay - uy)


def _twice_area(a: Coordinate, b: Coordinate, c: Coordinate) -> float:
    # Twice the signed area of the triangle: zero exactly when no circle runs through
    # the three points, so circle_through_points would divide by zero.
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    return ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)


@dataclass(frozen=True)
class _BindingPlaces:
    """What the binding's refusals call a ring, and the grid they place it on."""

    grid_size_um: float | None
    # A ring's polygon is one its Entity owns; an outline ring, keyed by the Entity's own
    # semantic ID, has none, whatever polygon happens to carry that ID.
    owned: Mapping[tuple[str, str], LayoutPolygonSpec]
    # The layers of each Entity's polygons, joined, as SGB's selector refusal names them.
    entity_layers: Mapping[str, str]

    @classmethod
    def read(cls, build_input: GeometryBuildInput) -> _BindingPlaces:
        """The grid, the owned polygons and the Entity layers of an input."""
        by_id = {polygon.polygon_id: polygon for polygon in build_input.polygons}
        context = geometry_grid_context(build_input)
        return cls(
            None if context is None else context.grid_size_um,
            {
                (entity.semantic_id, polygon_id): by_id[polygon_id]
                for entity in build_input.entities
                for polygon_id in entity.polygon_ids
            },
            {
                entity.semantic_id: ", ".join(
                    sorted({by_id[p].layer for p in entity.polygon_ids if p in by_id})
                )
                for entity in build_input.entities
            },
        )

    def polygon(self, key: BoundaryKey) -> LayoutPolygonSpec | None:
        """The polygon whose ring a key names, or None for an Entity's outline ring."""
        return self.owned.get((key[0], key[1]))

    def claimants(self, key: BoundaryKey, names: Sequence[str]) -> str:
        """The records that claim one ring, with its Entity and layer."""
        polygon = self.polygon(key)
        on = "" if polygon is None else f" on layer {polygon.layer}"
        return f"{' and '.join(names)} (Entity {key[0]!r}{on})"

    def claim(self, key: BoundaryKey, ring: PolygonRing, names: Sequence[str]) -> str:
        """The records that claim one ring, and where the ring lies."""
        return (
            f"{self.claimants(key, names)} select the ring at "
            f"{format_extent(ring, self.grid_size_um)}"
        )


def _named(
    record: GdsBoundaryReconstructionSpec, polygon: LayoutPolygonSpec | None
) -> str:
    """The record, its Entity and the selected polygon's layer, as refusals name them."""
    on = "" if polygon is None else f", layer {polygon.layer}"
    return f"{record.boundary_id} (Entity {record.entity_id!r}{on})"


def _drawn_ring(
    record: GdsBoundaryReconstructionSpec,
    key: BoundaryKey,
    ring: PolygonRing,
    polygon: LayoutPolygonSpec | None,
    grid_size_um: float | None,
) -> PolygonRing:
    # The ring the record's indices name, by the one rule drawn_ring_of states; a ring no
    # polygon holds (an Entity's outline loop) is given as drawn.
    if polygon is None:
        return ring
    _, _, role, hole_index = key
    drawn = drawn_ring_of(polygon, 0 if role == "outer" else 1 + cast(int, hole_index))
    if drawn is None:
        raise ValueError(
            ring_without_drawn_order(
                _named(record, polygon), format_extent(ring, grid_size_um)
            )
        )
    return drawn


def _reconstruct(  # noqa: PLR0913 -- the record, its two rings, how refusals name it, and the grid
    record: GdsBoundaryReconstructionSpec,
    ring: PolygonRing,
    drawn: PolygonRing,
    what: str,
    grid_size_um: float | None,
):
    # The record's indices name the drawn ring; they are checked there, then carried onto
    # the ring this stage holds, which may start elsewhere, run the other way and hold
    # crossings the grid noded in.
    refusal = reconstruction_chain_refusal(
        record,
        what,
        len(drawn),
        lambda i: format_point(*drawn[i], grid_size_um),
        format_extent(drawn, grid_size_um),
    )
    if refusal is not None:
        raise ValueError(refusal)
    segments = tuple(record.segment_indices)
    drawn_anchors = tuple(record.anchor_indices) or (
        *segments,
        (segments[-1] + 1) % len(drawn),
    )
    drawn_keys = [region_key(point, grid_size_um) for point in drawn]
    ring_keys = [region_key(point, grid_size_um) for point in ring]
    chain = drawn_chain_on_ring(
        segments, drawn_anchors, record.closed, drawn_keys, ring_keys
    )
    if chain is None:
        raise ValueError(
            ring_lost_drawn_order(
                what,
                [
                    f"drawn vertex {i} at {format_point(*drawn[i], grid_size_um)} {fault}"
                    for i, fault in drawn_vertices_off_ring(drawn_keys, ring_keys)
                ],
            )
        )
    segments, anchors = chain.segments, chain.anchors
    count = len(ring)
    points = tuple(ring[i] for i in anchors)
    if record.kind == "circular_arc" and _twice_area(*points) == 0:
        raise ValueError(_no_circle(what, drawn_anchors, drawn, grid_size_um))
    if record.kind == "circular_arc" and record.closed:
        center, radius = circle_through_points(points)
        start = points[0]
        angle = math.atan2(start[1] - center[1], start[0] - center[0])
        area = sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(ring, (*ring[1:], ring[0]), strict=False)
        )
        direction = 1 if area > 0 else -1

        def position(t):
            return (
                center[0] + radius * math.cos(t),
                center[1] + radius * math.sin(t),
            )

        opposite = position(angle + direction * math.pi)
        curves = (
            SourceCurveSpec(
                "circular_arc",
                (start, position(angle + direction * math.pi / 2), opposite),
            ),
            SourceCurveSpec(
                "circular_arc",
                (opposite, position(angle + direction * 3 * math.pi / 2), start),
            ),
        )
    else:
        curves = (
            SourceCurveSpec(
                record.kind,
                points,
                degree=record.degree,
                parameter_interval=record.parameter_interval,
            ),
        )
    return (
        segments,
        curves,
        {
            "requested": asdict(record),
            "drawn_ring": drawn,
            "original_ring_um": ring,
            "selected_segments": [
                {"index": i, "start_um": ring[i], "end_um": ring[(i + 1) % count]}
                for i in segments
            ],
            "selected_anchor_indices": anchors,
            "selected_anchor_points_um": points,
        },
    )


def _no_circle(
    what: str,
    anchors: Sequence[int],
    drawn: PolygonRing,
    grid_size_um: float | None,
) -> str:
    """The refusal of a circular reconstruction whose three anchors define no circle."""
    at = ", ".join(
        f"drawn vertex {a} at {format_point(*drawn[a], grid_size_um)}" for a in anchors
    )
    how = (
        "two of them are one vertex"
        if len(set(anchors)) < len(anchors)
        else "they lie on one line"
    )
    return (
        f"{what}: the three arc points are collinear, so they define no circle: its "
        f"anchors are {at}; {how}. Fix the declaration: choose three anchors of its "
        "chain that do not lie on one line."
    )


def source_boundaries(
    build_input: GeometryBuildInput, rings: SourceRings
) -> tuple[BoundaryChains, list[dict[str, object]]]:
    """Bind authored and reconstructed curves in original source-ring order."""
    places = _BindingPlaces.read(build_input)
    end_pieces = frozenset(
        polygon.polygon_id
        for polygon in build_input.polygons
        if is_junction_end_piece(polygon)
    )

    def select(
        record: BoundaryCurveChainSpec | GdsBoundaryReconstructionSpec,
    ) -> tuple[BoundaryKey, PolygonRing]:
        return select_boundary(
            record,
            rings,
            end_pieces,
            layer=places.entity_layers.get(record.entity_id),
            grid_size_um=places.grid_size_um,
        )

    authored: dict[BoundaryKey, BoundaryCurveChainSpec] = {}
    provenance: list[dict[str, object]] = []
    for raw in build_input.boundary_curves:
        record = boundary_binding(raw)
        key, ring = select(record)
        if key in authored:
            names = (authored[key].boundary_id, record.boundary_id)
            raise ValueError(
                f"multiple complete authored boundaries claim {key}: "
                f"{places.claim(key, ring, names)}. Fix the declaration: give each ring "
                "one complete boundary record."
            )
        authored[key] = record
        provenance.append(
            {
                "source": asdict(record),
                "original_ring_um": ring,
                "operation": "authored",
            }
        )
    reconstructed: defaultdict[BoundaryKey, list[ReconstructionSelection]] = (
        defaultdict(list)
    )
    for raw in build_input.boundary_reconstruction:
        record = boundary_binding(raw, reconstruction=True)
        key, ring = select(record)
        polygon = places.polygon(key)
        drawn = _drawn_ring(record, key, ring, polygon, places.grid_size_um)
        segments, curves, operation = _reconstruct(
            record, ring, drawn, _named(record, polygon), places.grid_size_um
        )
        reconstructed[key].append((segments, curves, record.boundary_id))
        provenance.append({"operation": "explicit_gds_reconstruction", **operation})
    return _ring_chains(rings, authored, reconstructed, places), provenance


def _ring_chains(
    rings: SourceRings,
    authored: Mapping[BoundaryKey, BoundaryCurveChainSpec],
    reconstructed: Mapping[BoundaryKey, Sequence[ReconstructionSelection]],
    places: _BindingPlaces,
) -> BoundaryChains:
    """Each source ring as its authored curves, or its segments with reconstructions put in."""
    result: BoundaryChains = {}
    for entity_id, records in rings.items():
        for pid, outer, holes in records:
            for role, index, ring in [
                ("outer", None, outer),
                *[("hole", i, h) for i, h in enumerate(holes)],
            ]:
                key = (entity_id, pid, role, index)
                if key in authored:
                    if key in reconstructed:
                        names = (
                            authored[key].boundary_id,
                            *(bid for _, _, bid in reconstructed[key]),
                        )
                        raise ValueError(
                            f"authored and reconstructed boundaries both claim {key}: "
                            f"{places.claim(key, ring, names)}. Fix the declaration: "
                            "describe the ring by one authored boundary or by "
                            "reconstruction records, not both."
                        )
                    result[key] = authored[key].curves
                    continue
                result[key] = _replace_segments(
                    key, ring, reconstructed.get(key, ()), places
                )
    return result


def _replace_segments(
    key: BoundaryKey,
    ring: PolygonRing,
    operations: Sequence[ReconstructionSelection],
    places: _BindingPlaces,
) -> tuple[SourceCurveSpec, ...]:
    """Replace selected source chains while retaining unselected straight segments."""
    claimed = {}
    starts = {}
    for segments, curves, bid in operations:
        shared = [i for i in segments if i in claimed]
        if shared:
            names = tuple(dict.fromkeys((*(claimed[i] for i in shared), bid)))
            raise ValueError(
                f"overlapping reconstruction segment claims on {key}: "
                f"{places.claimants(key, names)} both replace the ring "
                f"{_runs(shared, ring, places.grid_size_um)}. Fix the declaration: give "
                "each drawn segment of the ring to one reconstruction record."
            )
        for i in segments:
            claimed[i] = bid
        starts[segments[0]] = curves
    # Traverse from an operation start when it crosses original ring index zero.
    first = next((segments[0] for segments, _, _ in operations if 0 in segments), 0)
    indices = [(first + i) % len(ring) for i in range(len(ring))]
    curves = []
    for i in indices:
        if i in starts:
            curves.extend(starts[i])
        elif i not in claimed:
            curves.append(
                SourceCurveSpec(
                    "line_segment",
                    (
                        cast(Coordinate, tuple(ring[i])),
                        cast(Coordinate, tuple(ring[(i + 1) % len(ring)])),
                    ),
                )
            )
    return tuple(curves)


def _runs(indices: Sequence[int], ring: PolygonRing, grid_size_um: float | None) -> str:
    """Where the ring's segments at these indices run, those that follow each other as one.

    The ring may run the other way from the drawing, so a run is named by its two ends.
    """
    count = len(ring)
    runs: list[list[int]] = []
    for i in indices:
        if runs and i == (runs[-1][-1] + 1) % count:
            runs[-1].append(i)
        else:
            runs.append([i])
    return ", ".join(
        f"between {format_point(*ring[run[0]], grid_size_um)} and "
        f"{format_point(*ring[(run[-1] + 1) % count], grid_size_um)}"
        for run in runs
    )
