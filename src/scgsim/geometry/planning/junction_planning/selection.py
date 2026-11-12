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
from scgsim.geometry.models.grid import format_point, geometry_grid_context
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
from scgsim.geometry.primitives.grid_regions import grid_change_message
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
    positions = [u for u, _ in contact_points(polygons, span, width)]
    return (min(positions), max(positions)) if positions else None


def contact_points(polygons, span, width) -> list[tuple]:
    """The local points ``(u, v)`` that bound the polygons' area, segment and point contact with closed S.

    ``contacts`` reads their extent along the junction; a refusal locates the contact by them.
    """
    region = boolean(
        build_polygon_region(polygons),
        build_polygon_region(rectangle(0, 0, span, width)),
        "and",
    )
    found: list[tuple] = [
        (round(x / PRECISION_UM), round(y / PRECISION_UM))
        for p in region
        for x, y in p.points
    ]
    for rings in polygons:
        for axis, c, low, high in (
            (0, 0, 0, width),
            (0, span, 0, width),
            (1, 0, 0, span),
            (1, width, 0, span),
        ):
            intervals, points = section(rings, axis, c, low, high)
            across = [*points, *(v for pair in intervals for v in pair)]
            found.extend((c, v) if axis == 0 else (v, c) for v in across)
    return found


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
    region = end_region(source, section)
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


def end_region(source, section: EndSection):
    """The part of ``source`` beyond the cut, as the native Boolean gives it: empty for a cut at the sheet's end."""
    if not (section.cut > 0 if section.left else section.cut < section.span):
        return []
    clip = (
        rectangle(0, 0, section.cut, section.width)
        if section.left
        else rectangle(section.cut, 0, section.span, section.width)
    )
    return boolean(source, build_polygon_region(clip), "and")


# Why ``end_candidate`` refuses an end piece, in the words of a layout refusal.
END_REFUSALS = {
    "nonempty end has disconnected components": "falls apart into separate pieces",
    "end lacks positive own-arm contact": (
        "meets its conductor neither along an edge nor over an area"
    ),
    "end contacts wrong terminal or third net": "touches a conductor of another Net",
    "designated end edge is not wholly covered": (
        "leaves part of the cut covered neither by it nor by its conductor"
    ),
}


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
    it.  ``rebuilt_arcs`` says whether the sheet lands on an edge SGB builds as an exact arc
    instead of its drawn polygon: an arc of the terminal that meets the sheet's rectangle.  An
    arc elsewhere on the terminal, such as a rounded corner away from the port, leaves a
    landing on a straight edge as drawn.  ``arc_reach`` is the lowest and highest position
    along the junction direction, in Boolean units, of those arcs inside the sheet's
    rectangle, or ``None``: an exact arc can reach farther into the sheet than the drawn
    polygon's chords, and the cut must lie beyond it.
    """

    meetings: dict[int, tuple[SideMeeting, ...]]
    rebuilt_arcs: bool
    arc_reach: tuple[int, int] | None


def built_outline(
    build_input: GeometryBuildInput,
    entity_ids: Iterable[str],
    frame: JunctionLocalFrame,
    span: int,
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
    layers = {p.polygon_id: p.layer for p in build_input.polygons}
    meetings: dict[int, list[SideMeeting]] = {0: [], width: []}
    for (entity_id, polygon_id, *_), chain in chains.items():
        drawn = _DrawnRing(entity_id, layers.get(polygon_id), grid_um)
        for curve in chain:
            for side, found in meetings.items():
                found.extend(_curve_meetings(curve, frame, side, drawn))
    arcs = [
        curve
        for chain in chains.values()
        for curve in chain
        if curve.kind == "circular_arc"
    ]
    # SGB rounds an arc's crossings to the grid, so an arc within one grid step of the sheet
    # may touch it once built.
    near = PRECISION_UM if grid_um is None else grid_um
    sheet = (0.0, span * PRECISION_UM, 0.0, width * PRECISION_UM)
    widened = (-near, sheet[1] + near, -near, sheet[3] + near)
    inside = [u for curve in arcs for u in _arc_positions(curve, frame, sheet)]
    return BuiltOutline(
        {side: tuple(found) for side, found in meetings.items()},
        any(_arc_positions(curve, frame, widened) for curve in arcs),
        (
            (round(min(inside) / PRECISION_UM), round(max(inside) / PRECISION_UM))
            if inside
            else None
        ),
    )


def _arc_positions(
    curve: SourceCurveSpec,
    frame: JunctionLocalFrame,
    box: tuple[float, float, float, float],
) -> list[float]:
    """Where a circular arc inside the box ``(low u, high u, low v, high v)`` reaches farthest along u.

    These are the arc's ends inside the box, its crossings with the box's four edges and its
    points farthest along u, each where it lies on the arc and in the box, in micrometres.  The
    arc is continuous and moves monotonically along u between those points, so the list is
    empty exactly when the arc misses the box, and its extremes are the arc's reach inside it.
    """
    low_u, high_u, low_v, high_v = box
    start, through, end = (_local_um(point, frame) for point in curve.points_um)
    (cu, cv), radius = circle_through_points((start, through, end))

    def in_box(u: float, v: float) -> bool:
        return low_u <= u <= high_u and low_v <= v <= high_v

    found = [u for u, v in (start, end) if in_box(u, v)]
    found.extend(
        u
        for u in (cu - radius, cu + radius)
        if in_box(u, cv) and _on_arc((u, cv), start, through, end)
    )
    for at_u, value in ((True, low_u), (True, high_u), (False, low_v), (False, high_v)):
        offset = value - (cu if at_u else cv)
        if abs(offset) > radius:
            continue
        half = math.sqrt(radius * radius - offset * offset)
        centre = cv if at_u else cu
        for across in (centre - half, centre + half):
            point = (value, across) if at_u else (across, value)
            if in_box(*point) and _on_arc(point, start, through, end):
                found.append(point[0])
    return found


@dataclass(frozen=True)
class _DrawnRing:
    """The terminal ring a curve outlines: its Entity and source layer, and the model grid, for a refusal."""

    entity_id: str
    layer: str | None
    grid_um: float | None


def _curve_meetings(
    curve: SourceCurveSpec, frame: JunctionLocalFrame, side: int, drawn: _DrawnRing
) -> list[SideMeeting]:
    """The meetings of one line or circular arc with the side line ``v = side``."""
    grid_um = drawn.grid_um
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
    (x0, y0), (x1, y1) = curve.points_um[0], curve.points_um[-1]
    raise ValueError(
        grid_change_message(
            f"junction terminal outline curve {curve.kind!r} is not supported; "
            f"a terminal is built from lines and circular arcs: {drawn.entity_id}",
            [] if drawn.layer is None else [drawn.layer],
            grid_um,
            [
                f"its outline runs the curve from {format_point(x0, y0, grid_um)} "
                f"to {format_point(x1, y1, grid_um)}"
            ],
            "Fix the layout: draw the curve as lines and circular arcs.",
        )
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
        nearest, continues = _nearest_meeting(found, cut, span, left=left)
        steps.append(0 if continues else abs(cut - nearest))
    return tuple(steps)


def nearest_meetings(
    meetings: Mapping[int, tuple[SideMeeting, ...]],
    cut: int,
    span: int,
    *,
    left: bool,
) -> tuple[tuple[int, int], ...]:
    """Per sheet side line, where the end piece's edge along it starts: ``(position, side line)``.

    The positions ``side_steps`` measures from, for a refusal to locate.
    """
    return tuple(
        (_nearest_meeting(found, cut, span, left=left)[0], line)
        for line, found in meetings.items()
    )


def _nearest_meeting(found, cut, span, *, left) -> tuple[int, bool]:
    """The terminal's meeting with one side line nearest the cut on its side, or the sheet corner, and whether an edge along the side ends there."""
    if left:
        reached = [m for m in found if 0 <= m.position <= cut]
        nearest = max((m.position for m in reached), default=0)
    else:
        reached = [m for m in found if cut <= m.position <= span]
        nearest = min((m.position for m in reached), default=span)
    return nearest, any(m.along_side and m.position == nearest for m in reached)
