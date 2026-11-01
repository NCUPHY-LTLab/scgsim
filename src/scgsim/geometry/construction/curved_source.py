"""Read the authored source rings and boundary curves of a curved build.

Owns the pure-Python half of curved construction: Entity-owned logical polygon
rings, selection of the ring a boundary record binds to, explicit GDS (Graphic
Design System layout format) arc reconstruction, and the per-ring curve chains
handed to the native kernel. Does not talk to gmsh and does not name any
native entity.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict

from scgsim.geometry.models.input import SourceCurveSpec
from scgsim.geometry.source.curves import boundary_binding


def _source_rings(build_input):
    """Use adapter-normalized Entity-owned logical polygons, including holes."""
    polygons = {p.polygon_id: p for p in build_input.polygons}
    result = {}
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


def _select_boundary(record, rings):
    """Find the one ring a boundary record's selector point falls inside."""
    import gdstk

    candidates = []
    for pid, outer, holes in rings.get(record.entity_id, ()):
        region = [gdstk.Polygon(outer)]
        if holes:
            region = (
                gdstk.boolean(region, [gdstk.Polygon(h) for h in holes], "not") or []
            )
        if gdstk.inside([record.selector_point_um], region)[0]:
            candidates.append((pid, outer, holes))
    if len(candidates) != 1:
        raise ValueError(
            f"{record.boundary_id} selector must bind one Entity-owned logical polygon"
        )
    pid, outer, holes = candidates[0]
    ring = outer if record.role == "outer" else holes[record.hole_index]
    return (record.entity_id, pid, record.role, record.hole_index), tuple(
        tuple(float(x) for x in xy) for xy in ring
    )


def _circle_points(points):
    """Center and radius of the circle through three points (circumcircle)."""
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


def _validated_chain(record, ring):
    """Check a reconstruction record against its ring; return segments, anchors, points."""
    count = len(ring)
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
    points = tuple(ring[i] for i in anchors)
    if not record.closed and (
        anchors[0] != vertex_indices[0] or anchors[-1] != vertex_indices[-1]
    ):
        raise ValueError(
            f"{record.boundary_id} reconstruction must retain original chain endpoints"
        )
    return segments, anchors, points


def _reconstruct(record, ring):
    """Validate an explicit reconstruction record and turn it into curve specs.

    Returns the claimed segment indices, the curves replacing them, and the
    provenance of the operation.
    """
    count = len(ring)
    segments, anchors, points = _validated_chain(record, ring)
    if record.kind == "circular_arc" and record.closed:
        if len(points) != 3:
            raise ValueError(
                "closed circular reconstruction requires three explicit distinct anchors"
            )
        center, radius = _circle_points(points)
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
            "original_ring_um": ring,
            "selected_segments": [
                {"index": i, "start_um": ring[i], "end_um": ring[(i + 1) % count]}
                for i in segments
            ],
            "selected_anchor_indices": anchors,
            "selected_anchor_points_um": points,
        },
    )


def _boundaries(build_input, rings):
    """Return one ordered curve chain per ring, plus the provenance of each claim."""
    authored, reconstructed, provenance = _boundary_claims(build_input, rings)
    result = {}
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
                result[key] = _chain_with_reconstruction(
                    key, ring, reconstructed.get(key, ())
                )
    return result, provenance


def _boundary_claims(build_input, rings):
    """Bind authored curves and reconstruction records to the rings they select."""
    authored = {}
    reconstructed = defaultdict(list)
    provenance = []
    for raw in build_input.boundary_curves:
        record = boundary_binding(raw)
        key, ring = _select_boundary(record, rings)
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
    for raw in build_input.boundary_reconstruction:
        record = boundary_binding(raw, reconstruction=True)
        key, ring = _select_boundary(record, rings)
        segments, curves, operation = _reconstruct(record, ring)
        reconstructed[key].append((segments, curves, record.boundary_id))
        provenance.append({"operation": "explicit_gds_reconstruction", **operation})
    return authored, reconstructed, provenance


def _chain_with_reconstruction(key, ring, operations):
    """Walk a ring, replacing claimed segment runs by their reconstructed curves."""
    claimed = {}
    starts = {}
    for segments, curves, bid in operations:
        if any(i in claimed for i in segments):
            raise ValueError(f"overlapping reconstruction segment claims on {key}")
        for i in segments:
            claimed[i] = bid
        starts[segments[0]] = curves
    # Traverse from an operation start when it crosses original ring index zero,
    # so a reconstructed chain that wraps the ring seam stays contiguous.
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
                    (tuple(ring[i]), tuple(ring[(i + 1) % len(ring)])),
                )
            )
    return tuple(curves)
