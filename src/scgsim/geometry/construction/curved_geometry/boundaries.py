"""Bind authored boundaries and explicit reconstruction to original source rings."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from typing import cast

from scgsim.geometry.models.common import Coordinate, PolygonRing
from scgsim.geometry.models.grid import geometry_grid_context
from scgsim.geometry.models.input import (
    RING_LOST_DRAWN_ORDER,
    RING_WITHOUT_DRAWN_ORDER,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
    SourceCurveSpec,
    carried_drawn_ring,
    drawn_chain_on_ring,
    emitted_by_source_grid,
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
    determinant = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
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


def _drawn_ring(
    record: GdsBoundaryReconstructionSpec,
    ring: PolygonRing,
    polygon: LayoutPolygonSpec | None,
) -> PolygonRing:
    # The ring the record's indices name: the one the source stage recorded, or the ring
    # itself when no source grid stage has emitted it.
    carried = carried_drawn_ring(record)
    if carried is not None:
        return carried
    if emitted_by_source_grid(polygon):
        raise ValueError(f"{record.boundary_id}: {RING_WITHOUT_DRAWN_ORDER}")
    return ring


def _reconstruct(
    record: GdsBoundaryReconstructionSpec,
    ring: PolygonRing,
    drawn: PolygonRing,
    grid_size_um: float | None,
):
    # The record's indices name the drawn ring; they are checked there, then carried onto
    # the ring this stage holds, which may start elsewhere, run the other way and hold
    # crossings the grid noded in.
    count = len(drawn)
    segments = tuple(record.segment_indices)
    if not segments or any(
        isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count
        for i in segments
    ):
        raise ValueError(
            f"{record.boundary_id} requires explicit original segment indices"
        )
    if len(set(segments)) != len(segments) or any(
        b != (a + 1) % count for a, b in zip(segments, segments[1:], strict=False)
    ):
        raise ValueError(
            f"{record.boundary_id} segments must form one ordered source chain"
        )
    if record.closed != (
        len(segments) == count and (segments[-1] + 1) % count == segments[0]
    ):
        raise ValueError(
            f"{record.boundary_id} closure differs from selected source chain"
        )
    vertex_indices = (*segments, (segments[-1] + 1) % count)
    anchors = tuple(record.anchor_indices) or vertex_indices
    if any(i not in vertex_indices for i in anchors):
        raise ValueError(
            f"{record.boundary_id} anchors must belong to selected source chain"
        )
    if not record.closed and (
        anchors[0] != vertex_indices[0] or anchors[-1] != vertex_indices[-1]
    ):
        raise ValueError(
            f"{record.boundary_id} reconstruction must retain original chain endpoints"
        )
    chain = drawn_chain_on_ring(
        segments,
        anchors,
        record.closed,
        [region_key(point, grid_size_um) for point in drawn],
        [region_key(point, grid_size_um) for point in ring],
    )
    if chain is None:
        raise ValueError(f"{record.boundary_id}: {RING_LOST_DRAWN_ORDER}")
    segments, anchors = chain.segments, chain.anchors
    count = len(ring)
    points = tuple(ring[i] for i in anchors)
    if record.kind == "circular_arc" and record.closed:
        if len(points) != 3:
            raise ValueError(
                "closed circular reconstruction requires three explicit distinct anchors"
            )
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
            "drawn_ring_um": drawn,
            "original_ring_um": ring,
            "selected_segments": [
                {"index": i, "start_um": ring[i], "end_um": ring[(i + 1) % count]}
                for i in segments
            ],
            "selected_anchor_indices": anchors,
            "selected_anchor_points_um": points,
        },
    )


def source_boundaries(
    build_input: GeometryBuildInput, rings: SourceRings
) -> tuple[BoundaryChains, list[dict[str, object]]]:
    """Bind authored and reconstructed curves in original source-ring order."""
    authored: BoundaryChains = {}
    reconstructed: defaultdict[BoundaryKey, list[ReconstructionSelection]] = (
        defaultdict(list)
    )
    provenance: list[dict[str, object]] = []
    end_pieces = frozenset(
        polygon.polygon_id
        for polygon in build_input.polygons
        if is_junction_end_piece(polygon)
    )
    for raw in build_input.boundary_curves:
        record = boundary_binding(raw)
        key, ring = select_boundary(record, rings, end_pieces)
        if key in authored:
            raise ValueError(f"multiple complete authored boundaries claim {key}")
        authored[key] = record.curves
        provenance.append(
            {
                "source": asdict(record),
                "original_ring_um": ring,
                "operation": "authored",
            }
        )
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    context = geometry_grid_context(build_input)
    grid_size_um = None if context is None else context.grid_size_um
    for raw in build_input.boundary_reconstruction:
        record = boundary_binding(raw, reconstruction=True)
        key, ring = select_boundary(record, rings, end_pieces)
        drawn = _drawn_ring(record, ring, polygons.get(key[1]))
        segments, curves, operation = _reconstruct(record, ring, drawn, grid_size_um)
        reconstructed[key].append((segments, curves, record.boundary_id))
        provenance.append({"operation": "explicit_gds_reconstruction", **operation})
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
                        raise ValueError(
                            f"authored and reconstructed boundaries both claim {key}"
                        )
                    result[key] = authored[key]
                    continue
                result[key] = _replace_segments(key, ring, reconstructed.get(key, ()))
    return result, provenance


def _replace_segments(
    key: BoundaryKey, ring: PolygonRing, operations: Sequence[ReconstructionSelection]
) -> tuple[SourceCurveSpec, ...]:
    """Replace selected source chains while retaining unselected straight segments."""
    claimed = {}
    starts = {}
    for segments, curves, bid in operations:
        if any(i in claimed for i in segments):
            raise ValueError(f"overlapping reconstruction segment claims on {key}")
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
