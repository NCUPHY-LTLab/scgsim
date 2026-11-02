"""Compare native junction boundaries, contacts and planar topology."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from itertools import pairwise
from typing import Any

from scgsim.aedt.preparation.junction_partition import (
    closed_contact,
    cross,
    inside_ring,
    section,
)
from scgsim.geometry.primitives.planar_loops import (
    cancel_reversed_planar_edges,
    simple_planar_loops_from_edges,
)

# Native CAD readback uses an absolute micrometre tolerance, never relative scaling.
NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM = 1e-5


def native_coordinates_close(left, right) -> bool:
    """Compare coordinates using the existing absolute native-readback resolution."""
    return len(left) == len(right) and all(
        math.isfinite(float(a))
        and math.isfinite(float(b))
        and abs(float(a) - float(b)) <= NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM
        for a, b in zip(left, right, strict=False)
    )


def native_junction_corners(polygon, direction_xy, matches):
    """Order matched native corners A-low, B-low, B-high, A-high.

    The shared terminal-line validator establishes a rectangle in the declared
    axial direction. Rank its source corners along that same axis and its
    transverse (-dy, dx) axis; ring start and winding carry no terminal meaning.
    Only the ordering changes: each returned coordinate is the original matched
    native value, including its readback deviation from the source.
    """
    dx, dy = (Fraction(str(float(value))) for value in direction_xy)
    source_points = [
        tuple(Fraction(str(float(value))) for value in point)
        for point in polygon["exterior"]
    ]
    along = sorted(source_points, key=lambda point: point[0] * dx + point[1] * dy)
    terminal_a = sorted(along[:2], key=lambda point: -point[0] * dy + point[1] * dx)
    terminal_b = sorted(along[2:], key=lambda point: -point[0] * dy + point[1] * dx)
    return [
        matches[point]
        for point in (terminal_a[0], terminal_b[0], terminal_b[1], terminal_a[1])
    ]


def native_straight_ring(ring):
    """Comparison copy with whole monotone straight chains, never moved nodes."""
    points = [tuple(Fraction(str(float(v))) for v in point) for point in ring]
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or any(
        a == b for a, b in zip(points, (*points[1:], points[0]), strict=False)
    ):
        raise RuntimeError(
            "native junction boundary is unresolved at readback resolution"
        )
    chains = [[a, b] for a, b in zip(points, (*points[1:], points[0]), strict=False)]
    changed = True
    while changed and len(points) > 3:
        changed = False
        for index in range(len(points)):
            chain = chains[index - 1] + chains[index][1:]
            parameters = [
                _native_edge_projection(p, chain[0], chain[-1])[0] for p in chain
            ]
            if parameters == sorted(parameters) and all(
                native_on_segment(p, chain[0], chain[-1]) for p in chain
            ):
                chains[index - 1] = chain
                del chains[index]
                del points[index]
                changed = True
                break
    if any(
        native_coordinates_close(a, b)
        for a, b in zip(points, (*points[1:], points[0]), strict=False)
    ):
        raise RuntimeError(
            "native junction boundary is unresolved at readback resolution"
        )
    return points, chains


def native_polygon_correspondence(
    loops, polygon, name, deviations, *, subdivisions=True
):
    """Match boundary corner cycles, retaining raw native loops for contacts.

    Both raw chains must cover each other at the existing coordinate policy;
    independent straight-chain reduction alone could hide opposing deviations.
    The central rectangle disables reduction and retains its four-edge contract.
    """
    source_rings = (polygon["exterior"], *polygon["holes"])
    actual_rings = [ring for rings in loops for ring in rings]
    if len(loops) != 1 or len(actual_rings) != len(source_rings):
        raise RuntimeError(
            f"native junction conductor shape/holes differ from source: {name!r}"
        )

    def comparison(rings):
        corners, chains = [], {}
        for ring in rings:
            points = [tuple(Fraction(str(float(v))) for v in p) for p in ring]
            if subdivisions:
                points, raw_chains = native_straight_ring(points)
            else:
                raw_chains = [
                    [a, b]
                    for a, b in zip(points, (*points[1:], points[0]), strict=False)
                ]
            corners.extend(points)
            chains.update({tuple(sorted((c[0], c[-1]))): c for c in raw_chains})
        return corners, chains

    expected, expected_chains = comparison(source_rings)
    actual, actual_chains = comparison(actual_rings)
    matches = {}
    for point in actual:
        candidates = [p for p in expected if native_coordinates_close(point, p)]
        if len(candidates) != 1 or candidates[0] in matches:
            raise RuntimeError(
                f"native junction vertex correspondence is missing or ambiguous: {name!r}"
            )
        matches[candidates[0]] = point
        deviations.append(
            max(abs(float(a - b)) for a, b in zip(point, candidates[0], strict=False))
        )
    if len(actual) != len(expected) or set(matches) != set(expected):
        raise RuntimeError(f"native junction vertex topology differs: {name!r}")
    if len(actual_chains) != len(actual) or len(expected_chains) != len(expected):
        raise RuntimeError(f"native junction boundary has repeated edges: {name!r}")
    reverse = {point: wanted for wanted, point in matches.items()}
    for edge, chain in actual_chains.items():
        source_chain = expected_chains.get(tuple(sorted(reverse[p] for p in edge)))
        if source_chain is None:
            raise RuntimeError(
                f"native junction conductor shape/holes differ from source: {name!r}"
            )
        for nodes, other in ((chain, source_chain), (source_chain, chain)):
            for point in nodes:
                segments = list(pairwise(other))
                if not any(native_on_segment(point, a, b) for a, b in segments):
                    raise RuntimeError(
                        f"native junction boundary chain differs from source: {name!r}"
                    )
                residuals = []
                for a, b in segments:
                    parameter, projected = _native_edge_projection(point, a, b)
                    nearest = a if parameter < 0 else b if parameter > 1 else projected
                    residuals.append(
                        max(
                            abs(float(p - q))
                            for p, q in zip(point, nearest, strict=False)
                        )
                    )
                deviations.append(min(residuals))
    return matches


def _native_edge_projection(point, start, end):
    delta = tuple(b - a for a, b in zip(start, end, strict=False))
    norm = sum(v * v for v in delta)
    if not norm:
        raise RuntimeError("native junction boundary has a collapsed edge")
    parameter = (
        sum((p - a) * v for p, a, v in zip(point, start, delta, strict=False)) / norm
    )
    projected = tuple(a + parameter * v for a, v in zip(start, delta, strict=False))
    return parameter, projected


def native_on_segment(point, start, end) -> bool:
    """Compare a point to the nearest bounded point on one native edge."""
    parameter, projected = _native_edge_projection(point, start, end)
    nearest = start if parameter < 0 else end if parameter > 1 else projected
    return native_coordinates_close(point, nearest)


def native_edge_section(bodies, start, end, *, boundary_only=False):
    """Sections of actual footprints, with only bounded boundary comparisons."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = dx * dx + dy * dy
    if not norm or native_coordinates_close(start, end):
        raise RuntimeError("native junction edge is unresolved at readback resolution")
    intervals, contacts = [], set()

    def clamp(value):
        point = (start[0] + value * dx, start[1] + value * dy)
        if native_coordinates_close(point, start):
            return Fraction(0)
        if native_coordinates_close(point, end):
            return Fraction(1)
        return value

    for rings in bodies:
        if not boundary_only:
            transformed = tuple(
                tuple(
                    (
                        ((p[0] - start[0]) * dx + (p[1] - start[1]) * dy) / norm,
                        dx * (p[1] - start[1]) - dy * (p[0] - start[0]),
                    )
                    for p in ring
                )
                for ring in rings
            )
            spans, points = section(transformed, 1, 0, 0, 1)
            intervals.extend((clamp(a), clamp(b)) for a, b in spans)
            contacts.update(clamp(v) for v in points)
        for ring in rings:
            for a, b in zip(ring, (*ring[1:], ring[0]), strict=False):
                ta, pa = _native_edge_projection(a, start, end)
                tb, pb = _native_edge_projection(b, start, end)
                near_a = native_coordinates_close(a, pa)
                near_b = native_coordinates_close(b, pb)
                if near_a and 0 <= clamp(ta) <= 1:
                    contacts.add(clamp(ta))
                if near_b and 0 <= clamp(tb) <= 1:
                    contacts.add(clamp(tb))
                if near_a and near_b:
                    lo, hi = (
                        max(Fraction(0), clamp(min(ta, tb))),
                        min(Fraction(1), clamp(max(ta, tb))),
                    )
                    if lo < hi:
                        intervals.append((lo, hi))
    spans = [(a, b) for a, b in intervals if a < b]
    contacts.update(a for a, b in intervals if a == b)
    return spans, contacts


def native_edge_covered(bodies, start, end) -> bool:
    """Require native boundary intervals to cover the full designated edge."""
    intervals, _ = native_edge_section(bodies, start, end, boundary_only=True)
    reached = Fraction(0)
    for lo, hi in sorted(intervals):
        left = tuple(a + lo * (b - a) for a, b in zip(start, end, strict=False))
        right = tuple(a + reached * (b - a) for a, b in zip(start, end, strict=False))
        if lo > reached and not native_coordinates_close(left, right):
            return False
        reached = max(reached, hi)
    return reached == 1


def native_boundary_contact(left, right) -> bool:
    """Require a resolved positive span; tiny raw subdivisions cannot prove one."""
    for rings in left:
        for ring in rings:
            for start, end in zip(ring, (*ring[1:], ring[0]), strict=False):
                if start != end and native_coordinates_close(start, end):
                    continue
                spans, _ = native_edge_section(right, start, end, boundary_only=True)
                for lo, hi in spans:
                    a = tuple(
                        p + lo * (q - p) for p, q in zip(start, end, strict=False)
                    )
                    b = tuple(
                        p + hi * (q - p) for p, q in zip(start, end, strict=False)
                    )
                    if not native_coordinates_close(a, b):
                        return True
    return False


def native_closed_contact(left, right) -> bool:
    """Detect closed-set footprint contact without moving native vertices."""
    return closed_contact(left, right) or any(
        native_on_segment(point, a, b)
        for regions, others in ((left, right), (right, left))
        for rings in regions
        for ring in rings
        for point in ring
        for other in others
        for boundary in other
        for a, b in zip(boundary, (*boundary[1:], boundary[0]), strict=False)
    )


def native_strict_inside(point, rings) -> bool:
    """Test strict footprint interior after excluding bounded edge contact."""
    return (
        inside_ring(point, rings[0]) == 1
        and all(inside_ring(point, hole) == -1 for hole in rings[1:])
        and not any(
            native_on_segment(point, a, b)
            for ring in rings
            for a, b in zip(ring, (*ring[1:], ring[0]), strict=False)
        )
    )


def native_cross_sign(a, b, point):
    """Read the oriented side of a native edge under the coordinate policy."""
    _, projected = _native_edge_projection(point, a, b)
    value = cross(a, b, point)
    return 0 if native_coordinates_close(point, projected) else (1 if value > 0 else -1)


def native_z_may_contact(left, right) -> bool:
    """Exclude forbidden contact only when actual Z ranges are separated."""
    low, high = max(left[0], right[0]), min(left[1], right[1])
    return low <= high or native_coordinates_close((low,), (high,))


def native_z_contact(left, left_solid, right, right_solid) -> bool:
    """Required contact compatibility; sheets need full consistent planes."""
    if not left_solid and not right_solid:
        return all(native_coordinates_close((a,), (b,)) for a in left for b in right)
    if left_solid and right_solid:
        return native_z_may_contact(left, right)
    solid, sheet = (left, right) if left_solid else (right, left)
    return all(
        (z >= solid[0] or native_coordinates_close((z,), (solid[0],)))
        and (z <= solid[1] or native_coordinates_close((z,), (solid[1],)))
        for z in sheet
    )


def native_face_edges(face, name):
    """Read straight edges once; failed observations stay in the private receipt."""
    edges = []
    edge_observations = []
    for edge in face.edges:
        vertices = [vertex.position for vertex in edge.vertices]
        if len(vertices) != 2 or any(p is None or len(p) != 3 for p in vertices):
            raise RuntimeError(f"native junction boundary is not polygonal: {name!r}")
        midpoint = edge.midpoint
        expected_midpoint = [
            (float(a) + float(b)) / 2 for a, b in zip(*vertices, strict=False)
        ]
        length = edge.length
        chord = math.dist(*vertices)
        try:
            midpoint_values = [float(v) for v in midpoint]
        except (TypeError, ValueError, OverflowError):
            midpoint_values = []
        midpoint_valid = len(midpoint_values) == 3 and all(
            math.isfinite(v) for v in midpoint_values
        )
        midpoint_matches = midpoint_valid and native_coordinates_close(
            midpoint_values, expected_midpoint
        )
        try:
            length_valid = (
                not isinstance(length, bool)
                and isinstance(length, (int, float))
                and math.isfinite(float(length))
            )
        except OverflowError:
            length_valid = False
        length_matches = length_valid and native_coordinates_close((length,), (chord,))
        if not midpoint_matches or not length_matches:

            def observed(value):
                if isinstance(value, float) and not math.isfinite(value):
                    return str(value)
                if isinstance(value, (list, tuple)):
                    return [observed(v) for v in value]
                if value is None or isinstance(value, (str, bool, int, float)):
                    return value
                return repr(value)

            residuals = (
                [
                    a - b
                    for a, b in zip(midpoint_values, expected_midpoint, strict=False)
                ]
                if midpoint_valid
                else None
            )
            diagnostic = {
                "object_name": name,
                "edge_id": getattr(edge, "__dict__", {}).get("id"),
                "face_id": getattr(face, "__dict__", {}).get(
                    "_id", getattr(face, "__dict__", {}).get("id")
                ),
                "model_units": "um",
                "absolute_tolerance_um": NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
                "relative_tolerance": 0.0,
                "endpoints_um": vertices,
                "observed_midpoint_um": midpoint,
                "midpoint_type": type(midpoint).__name__,
                "expected_midpoint_um": expected_midpoint,
                "native_length_um": length,
                "native_length_type": type(length).__name__,
                "chord_length_um": chord,
                "midpoint_residuals_um": residuals,
                "length_residual_um": float(length) - chord if length_valid else None,
                "failed_conditions": [
                    key
                    for key, failed in (
                        ("midpoint_invalid", not midpoint_valid),
                        ("midpoint_mismatch", midpoint_valid and not midpoint_matches),
                        ("native_length_invalid", not length_valid),
                        ("native_length_mismatch", length_valid and not length_matches),
                    )
                    if failed
                ],
            }
            raise RuntimeError(
                f"native junction edge is curved or its linear readback differs: {name!r}"
                + "; edge_diagnostic="
                + json.dumps(
                    {key: observed(value) for key, value in diagnostic.items()},
                    allow_nan=False,
                )
            )
        a, b = [tuple(Fraction(str(float(v))) for v in p) for p in vertices]
        edges.append((a, b))
        edge_observations.append(
            {
                "edge_id": getattr(edge, "__dict__", {}).get("id"),
                "vertices_um": [list(map(float, point)) for point in vertices],
                "midpoint_um": list(map(float, midpoint)),
                "native_length_um": float(length),
                "chord_length_um": chord,
                "maximum_midpoint_coordinate_deviation_um": max(
                    abs(float(a) - float(b))
                    for a, b in zip(midpoint, expected_midpoint, strict=False)
                ),
            }
        )
    return edges, edge_observations


def native_planar_loops(obj: Any, z_um: float, observations=None) -> list[tuple]:
    """Read actual planar boundaries at absolute 1e-5 um coordinate resolution.

    Independent vertex/midpoint/length getters need not serialize identically.
    Midpoint and actual native length jointly check straight edges; ambiguity
    below this comparison resolution is not an exact CAD contact guarantee.
    """

    loops = []
    for face in obj.faces:
        positions = [vertex.position for vertex in face.vertices]
        if not positions or any(p is None or len(p) != 3 for p in positions):
            raise RuntimeError(
                f"native junction face vertices unavailable: {obj.name!r}"
            )
        if any(not native_coordinates_close((p[2],), (z_um,)) for p in positions):
            continue
        if not native_coordinates_close(
            (min(float(p[2]) for p in positions),),
            (max(float(p[2]) for p in positions),),
        ):
            raise RuntimeError(
                "native junction face has inconsistent actual Z coordinates"
            )
        xyz_edges, edge_observations = native_face_edges(face, obj.name)
        edges = [(a[:2], b[:2]) for a, b in xyz_edges]
        if observations is not None:
            observations.append(
                {
                    "object_name": obj.name,
                    "face_id": getattr(face, "__dict__", {}).get(
                        "_id", getattr(face, "__dict__", {}).get("id")
                    ),
                    "requested_plane_z_um": z_um,
                    "maximum_plane_coordinate_deviation_um": max(
                        abs(float(point[2]) - z_um) for point in positions
                    ),
                    "face_vertices_um": [
                        list(map(float, point)) for point in positions
                    ],
                    "edges": edge_observations,
                }
            )
        # Edge direction is not a native face-loop ordering contract.
        face_loops = simple_planar_loops_from_edges(cancel_reversed_planar_edges(edges))

        def area(ring):
            return abs(
                sum(
                    a[0] * b[1] - b[0] * a[1]
                    for a, b in zip(ring, (*ring[1:], ring[0]), strict=False)
                )
            )

        outer = max(face_loops, key=area)
        loops.append((outer, *(ring for ring in face_loops if ring is not outer)))
    if not loops:
        raise RuntimeError(
            f"native junction conductor has no planar face at source Z: {obj.name!r}"
        )
    return loops
