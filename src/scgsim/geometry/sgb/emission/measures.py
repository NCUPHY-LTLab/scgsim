"""The areas and volumes the plan promises, in micrometres (design brief section 11).

Owns the planned measure of an emitted face (arcs realised: chord polygon plus signed circular
segments, or height times bottom-boundary arc length for a wall) and of a volume (the exact signed
volume of its shells, cylinder bulges included).  The load-side check (``verify``), the OCCT checks
(``occt``) and the sidecar compare the loaded or re-read geometry against these numbers, so a wrongly
oriented shell or a misplaced vertex shows up as a difference rather than as a plausible model.

Numbers are rounded to ``SIGNIFICANT_DIGITS`` for the sidecar: arc segment areas come from the
platform's ``asin``, which is not correctly rounded, and twelve digits are far below the 1e-9
relative tolerance of every comparison.
"""

from __future__ import annotations

import math

from scgsim.geometry.sgb.assembly.records import TopoFace, Topology, TopoVolume

SIGNIFICANT_DIGITS = 12


def rounded(value: float) -> float:
    """``value`` with twelve significant digits (the precision the sidecar promises)."""
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def planned_face_area_um2(topology: Topology, face: TopoFace) -> float:
    """Area (um^2) of a planned face with its arcs realised (what OCCT should report for the emitted surface)."""
    g = topology.grid_um
    if face.kind == "horizontal":
        twice = float(face.twice_xy_area)
        for loop in face.loops:
            for eid, sign in loop:
                curve = topology.edges[eid].curve
                if curve is not None:
                    twice += (
                        curve.signed_segment_twice_area * sign
                    )  # the loop keeps the face on its left
        return twice / 2 * g * g
    # A wall is vertical: height times the length of its bottom boundary (arcs by arc length).
    z_lo, z_hi = face.z_range
    length = 0.0
    for eid, _ in face.loops[0]:
        edge = topology.edges[eid]
        if edge.kind != "horizontal" or topology.vertices[edge.v0].z != z_lo:
            continue
        a, b = topology.vertices[edge.v0], topology.vertices[edge.v1]
        chord = math.hypot(b.x - a.x, b.y - a.y)
        if edge.curve is None:
            length += chord
        else:
            length += (
                edge.curve.radius
                * 2
                * math.asin(min(1.0, chord / (2 * edge.curve.radius)))
            )
    return length * g * float(z_hi - z_lo)


def planned_volume_um3(topology: Topology, volume: TopoVolume) -> float:
    """Volume (um^3) of a planned volume: the signed volume of all its shells (a void shell subtracts)."""
    return (
        float(sum(shell.signed_volume for shell in volume.shells)) * topology.grid_um**2
    )
