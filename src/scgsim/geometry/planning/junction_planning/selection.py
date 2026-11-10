"""Select terminal-owned end pieces on the authored corridor and grid.

Besides the topology of each end piece, this module owns the steps an end piece leaves on its
terminal: where the terminal's built outline meets a sheet side (``built_outline_meetings``)
and how long the edge is that the end piece adds along that side up to the cut
(``side_steps``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from decimal import ROUND_FLOOR, Decimal
from fractions import Fraction

from scgsim.geometry.construction.curved_geometry.boundaries import (
    circle_through_points,
    source_boundaries,
    source_rings,
)
from scgsim.geometry.models.grid import geometry_grid_context
from scgsim.geometry.models.input import GeometryBuildInput, SourceCurveSpec
from scgsim.geometry.planning.junction_planning.boolean_geometry import (
    boolean,
    build_polygon_region,
    read_boolean_rings,
    rectangle,
)
from scgsim.geometry.planning.junction_planning.core import PRECISION_UM
from scgsim.geometry.planning.junction_planning.exact_geometry import (
    closed_contact,
    covered,
    positive_boundary_contact,
    section,
)
from scgsim.geometry.planning.junction_planning.local_frame import JunctionLocalFrame
from scgsim.geometry.source.curves import boundary_binding


@dataclass(frozen=True)
class EndSection:
    """Carry one selected end cut, authored rectangle extent and terminal side."""

    cut: int
    span: int
    width: int
    left: bool


def contacts(polygons, span, width):
    """Find extrema of area, segment and point contact with closed authored S."""
    region = boolean(
        build_polygon_region(polygons),
        build_polygon_region(rectangle(0, 0, span, width)),
        "and",
    )
    positions: list[int | Fraction] = [
        round(x / PRECISION_UM) for p in region for x, _ in p.points
    ]
    for rings in polygons:
        for axis, c, low, high in (
            (0, 0, 0, width),
            (0, span, 0, width),
            (1, 0, 0, span),
            (1, width, 0, span),
        ):
            intervals, points = section(rings, axis, c, low, high)
            if axis == 0 and (intervals or points):
                positions.append(c)
            elif axis == 1:
                positions.extend(points)
                positions.extend((v for pair in intervals for v in pair))
    return (min(positions), max(positions)) if positions else None


def bound_decimal(bound):
    """Convert a rational corridor bound into its existing decimal grid value."""
    rational = Fraction(bound)
    return (
        Decimal(rational.numerator)
        / Decimal(rational.denominator)
        * Decimal(str(PRECISION_UM))
    )


def candidate_indices(bound, step, *, left):
    """Read the existing ordered candidate grid indices beside one bound."""
    q = bound_decimal(bound) / Decimal(str(step))
    floor = int(q.to_integral_value(rounding=ROUND_FLOOR))
    aligned = q == floor
    if left:
        return ([floor] if aligned else []) + [floor + 1]
    return ([floor] if aligned else []) + [floor if not aligned else floor - 1]


def grid_cut(index, step):
    """Resolve the requested and realized position of one grid index."""
    requested = float(Decimal(index) * Decimal(str(step)))
    return (requested, round(requested / PRECISION_UM))


def end_candidate(source, own, foreign, section: EndSection):
    """Cut the end piece from ``source`` beyond the cut and check it against exact contact conditions.

    ``source`` is the drawn sheet where SGB rebuilds the terminal with exact arcs: the end
    piece is then the sheet from its own end to the cut and overlaps its terminal where the
    sheet lands, so it reaches the arc also where the arc lies beyond the drawn polygon.  For a
    terminal SGB builds as drawn, ``source`` is the sheet's free part, and the end piece is what
    the sheet adds to the terminal; joined to the terminal, the two are the same region.  It is
    accepted when it is one piece, meets its own terminal over an area or a length, touches no
    other net and with its terminal covers the cut edge.
    """
    clip = (
        rectangle(0, 0, section.cut, section.width)
        if section.left
        else rectangle(section.cut, 0, section.span, section.width)
    )
    region = (
        boolean(source, build_polygon_region(clip), "and")
        if (section.cut > 0 if section.left else section.cut < section.span)
        else []
    )
    try:
        end = read_boolean_rings(region)
    except ValueError as exc:
        return (None, {"valid": False, "reason": str(exc)})
    if len(end) > 1:
        return (
            None,
            {
                "valid": False,
                "reason": "nonempty end has disconnected components",
                "components": len(end),
            },
        )
    if end and (not own_contact(end, own)):
        return (
            None,
            {"valid": False, "reason": "end lacks positive own-arm contact"},
        )
    if closed_contact(end, foreign):
        return (
            None,
            {"valid": False, "reason": "end contacts wrong terminal or third net"},
        )
    if not covered([*own, *end], 0, section.cut, 0, section.width):
        return (
            None,
            {"valid": False, "reason": "designated end edge is not wholly covered"},
        )
    return (end, {"valid": True, "empty_residual": not end, "components": len(end)})


def own_contact(end, own) -> bool:
    """Whether an end piece meets its terminal over a positive area or along a positive length."""
    return positive_boundary_contact(end, own) or bool(
        boolean(build_polygon_region(end), build_polygon_region(own), "and")
    )


@dataclass(frozen=True)
class SideMeeting:
    """One place where a terminal's built outline meets a sheet side line.

    ``position`` is along the junction direction in local Boolean units.  ``along_side`` marks
    an outline edge that lies on the side line and ends here, so an end piece reaching on from
    this point continues that edge straight instead of turning a corner.
    """

    position: int
    along_side: bool


@dataclass(frozen=True)
class BuiltOutline:
    """The outline SGB builds for a terminal, as the end-piece cut reads it.

    ``meetings`` holds, per sheet side line (``0`` and ``width``), where the outline meets
    it.  ``rebuilt_arcs`` says whether SGB builds any of the terminal's edges as an exact arc
    instead of its drawn polygon.
    """

    meetings: dict[int, tuple[SideMeeting, ...]]
    rebuilt_arcs: bool


def built_outline(
    build_input: GeometryBuildInput,
    entity_ids: Iterable[str],
    frame: JunctionLocalFrame,
    width: int,
) -> BuiltOutline:
    """The outline SGB builds for these conductors, where it meets each sheet side line.

    A ring with an authored boundary or an arc reconstruction is built from its exact lines and
    circular arcs, bound here as the construction path binds them; any other ring is its drawn
    polyline.  So a round terminal meets a side where its arc does, not where its polyline
    chord does.  SGB rounds a crossing to the model grid point whose cell holds it, so a meeting
    is taken at that grid point.
    """
    context = geometry_grid_context(build_input)
    grid_um = None if context is None else context.grid_size_um
    selected = set(entity_ids)
    curves = tuple(
        item
        for item in build_input.boundary_curves
        if boundary_binding(item).entity_id in selected
    )
    reconstructions = tuple(
        item
        for item in build_input.boundary_reconstruction
        if boundary_binding(item, reconstruction=True).entity_id in selected
    )
    terminals = replace(
        build_input,
        entities=tuple(e for e in build_input.entities if e.semantic_id in selected),
        boundary_curves=curves,
        boundary_reconstruction=reconstructions,
    )
    chains, _ = source_boundaries(terminals, source_rings(terminals))
    meetings: dict[int, list[SideMeeting]] = {0: [], width: []}
    for chain in chains.values():
        for curve in chain:
            for side, found in meetings.items():
                found.extend(_curve_meetings(curve, frame, side, grid_um))
    return BuiltOutline(
        {side: tuple(found) for side, found in meetings.items()},
        any(
            curve.kind != "line_segment" for chain in chains.values() for curve in chain
        ),
    )


def _curve_meetings(
    curve: SourceCurveSpec, frame: JunctionLocalFrame, side: int, grid_um: float | None
) -> list[SideMeeting]:
    """The meetings of one line or circular arc with the side line ``v = side``."""
    if curve.kind == "line_segment":
        (u0, v0), (u1, v1) = (_local_units(point, frame) for point in curve.points_um)
        if v0 == v1 == side:
            return [SideMeeting(u0, True), SideMeeting(u1, True)]
        if v0 == v1 or (v0 - side) * (v1 - side) > 0:
            return []
        crossing = u0 + Fraction(side - v0, v1 - v0) * (u1 - u0)
        return [_crossing(float(crossing) * PRECISION_UM, side, frame, grid_um)]
    if curve.kind == "circular_arc":
        start, through, end = (_local_um(point, frame) for point in curve.points_um)
        (cu, cv), radius = circle_through_points((start, through, end))
        offset = side * PRECISION_UM - cv
        if abs(offset) > radius:
            return []
        half = math.sqrt(radius * radius - offset * offset)
        return [
            _crossing(u, side, frame, grid_um)
            for u in sorted({cu - half, cu + half})
            if _on_arc((u, side * PRECISION_UM), start, through, end)
        ]
    raise ValueError(
        f"junction terminal outline curve {curve.kind!r} is not supported; "
        "a terminal is built from lines and circular arcs"
    )


def _local_um(point, frame: JunctionLocalFrame) -> tuple[float, float]:
    """A model point in the junction's local frame, in micrometres."""
    x, y = point
    return (
        x * frame.dx + y * frame.dy - frame.origin[0],
        x * frame.px + y * frame.py - frame.origin[1],
    )


def _crossing(
    u_um: float, side: int, frame: JunctionLocalFrame, grid_um: float | None
) -> SideMeeting:
    """A crossing of the side line at ``u_um``, moved to the model grid point SGB rounds it to."""
    if grid_um is not None:
        # Back to model XY, where SGB's grid lies; its snap rounding rounds half up.
        u, v = u_um + frame.origin[0], side * PRECISION_UM + frame.origin[1]
        x = math.floor((u * frame.dx + v * frame.px) / grid_um + 0.5) * grid_um
        y = math.floor((u * frame.dy + v * frame.py) / grid_um + 0.5) * grid_um
        u_um = _local_um((x, y), frame)[0]
    return SideMeeting(round(u_um / PRECISION_UM), False)


def _local_units(point, frame: JunctionLocalFrame) -> tuple[int, int]:
    """A model point in the local frame, rounded to Boolean units as polygons are."""
    u, v = _local_um(point, frame)
    return (round(u / PRECISION_UM), round(v / PRECISION_UM))


def _on_arc(point, start, through, end) -> bool:
    """Whether a point of the arc's circle lies on the arc from ``start`` through ``through`` to ``end``."""
    # The chord from start to end splits the circle in two; the arc is the half holding ``through``.
    if math.dist(point, start) <= PRECISION_UM or math.dist(point, end) <= PRECISION_UM:
        return True

    def side_of_chord(p) -> float:
        return (end[0] - start[0]) * (p[1] - start[1]) - (end[1] - start[1]) * (
            p[0] - start[0]
        )

    return side_of_chord(point) * side_of_chord(through) > 0


def side_steps(
    meetings: Mapping[int, tuple[SideMeeting, ...]],
    cut: int,
    span: int,
    *,
    left: bool,
) -> tuple[int, ...]:
    """The edge an end piece adds along each sheet side, from where the terminal meets it to the cut.

    The end piece runs along a side from the terminal's nearest meeting with that side, or
    from the sheet corner when the terminal meets the side nowhere between, to the cut.  An end
    piece that continues an edge the terminal already runs along the side adds no corner, so
    that side counts as zero, as does a cut on a straight edge.  Lengths are Boolean units.
    """
    steps = []
    for found in meetings.values():
        if left:
            reached = [m for m in found if 0 <= m.position <= cut]
            nearest = max((m.position for m in reached), default=0)
        else:
            reached = [m for m in found if cut <= m.position <= span]
            nearest = min((m.position for m in reached), default=span)
        continues = any(m.along_side and m.position == nearest for m in reached)
        steps.append(0 if continues else abs(cut - nearest))
    return tuple(steps)
