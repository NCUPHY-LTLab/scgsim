"""Select terminal-owned end pieces on the authored corridor and grid.

Besides the topology of each end piece, this module owns the steps an end piece leaves on its
terminal: where the terminal's built outline meets a sheet side (``built_outline``)
and how long the edge is that the end piece adds along that side up to the cut
(``side_steps``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from fractions import Fraction
from typing import NamedTuple

from scgsim.geometry.exact.boundaries import bound_ring_primitives
from scgsim.geometry.exact.carriers import ArcCarrier, LineEdge, RingPrimitive
from scgsim.geometry.exact.footprints import conductor_loops
from scgsim.geometry.exact.grid import Grid
from scgsim.geometry.grid.scope import GridScope
from scgsim.geometry.junctions.boolean_geometry import (
    boolean,
    build_polygon_region,
    read_boolean_rings,
    rectangle,
)
from scgsim.geometry.junctions.core import PRECISION_UM
from scgsim.geometry.junctions.exact_geometry import (
    closed_contact,
    covered,
    positive_boundary_contact,
    section,
)
from scgsim.geometry.junctions.local_frame import JunctionLocalFrame
from scgsim.geometry.models.grid import geometry_grid_context
from scgsim.geometry.models.input import GeometryBuildInput


@dataclass(frozen=True)
class EndSection:
    """Carry one selected end cut, authored rectangle extent and terminal side."""

    cut: int
    span: int
    width: int
    left: bool


def contacts(scope: GridScope, polygons, span, width):
    """Find extrema of area, segment and point contact with closed authored S."""
    positions = [u for u, _ in contact_points(scope, polygons, span, width)]
    return (min(positions), max(positions)) if positions else None


def contact_points(scope: GridScope, polygons, span, width) -> list[tuple]:
    """The local points ``(u, v)`` that bound the polygons' area, segment and point contact with closed S.

    ``contacts`` reads their extent along the junction; a refusal locates the contact by them.
    """
    region = boolean(
        scope,
        build_polygon_region(scope, polygons),
        build_polygon_region(scope, rectangle(0, 0, span, width)),
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


def end_candidate(scope: GridScope, source, own, foreign, section: EndSection):
    """Cut the end piece from ``source`` beyond the cut and check it against exact contact conditions.

    ``source`` is the drawn sheet where SGB rebuilds the terminal with exact arcs: the end
    piece is then the sheet from its own end to the cut and overlaps its terminal where the
    sheet lands, so it reaches the arc also where the arc lies beyond the drawn polygon.  For a
    terminal SGB builds as drawn, ``source`` is the sheet's free part, and the end piece is what
    the sheet adds to the terminal; joined to the terminal, the two are the same region.  It is
    accepted when it is one piece, meets its own terminal over an area or a length, touches no
    other net and with its terminal covers the cut edge.
    """
    region = end_region(scope, source, section)
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
    if end and (not own_contact(scope, end, own)):
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


def end_region(scope: GridScope, source, section: EndSection):
    """The part of ``source`` beyond the cut, as the native Boolean gives it: empty for a cut at the sheet's end."""
    if not (section.cut > 0 if section.left else section.cut < section.span):
        return []
    clip = (
        rectangle(0, 0, section.cut, section.width)
        if section.left
        else rectangle(section.cut, 0, section.span, section.width)
    )
    return boolean(scope, source, build_polygon_region(scope, clip), "and")


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


def own_contact(scope: GridScope, end, own) -> bool:
    """Whether an end piece meets its terminal over a positive area or along a positive length."""
    return positive_boundary_contact(end, own) or bool(
        boolean(
            scope,
            build_polygon_region(scope, end),
            build_polygon_region(scope, own),
            "and",
        )
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

    A ring with an authored boundary or an arc reconstruction is built from the lines and
    circular arcs SGB's own binding gives it (``geometry.exact.boundaries``), so a record on a
    terminal is bound, and refused, by the one rule the full build uses; any other ring is its
    drawn polyline.  So a round terminal meets a side where its arc does, not where its
    polyline chord does.  SGB rounds a crossing to the model grid point whose cell holds it,
    so a meeting is taken at that grid point.
    """
    grid_um = geometry_grid_context(build_input).grid_size_um
    lines, arcs = _outline_edges(build_input, set(entity_ids), frame)
    meetings: dict[int, list[SideMeeting]] = {0: [], width: []}
    for side, found in meetings.items():
        for line in lines:
            found.extend(_line_meetings(line, frame, side, grid_um))
        for arc in arcs:
            found.extend(_arc_meetings(arc, frame, side, grid_um))
    sheet = (0.0, span * PRECISION_UM, 0.0, width * PRECISION_UM)
    # SGB rounds an arc's crossings to the grid, so an arc within one grid step of the sheet
    # may touch it once built.
    widened = (-grid_um, sheet[1] + grid_um, -grid_um, sheet[3] + grid_um)
    inside = [u for arc in arcs for u in _arc_positions(arc, sheet)]
    return BuiltOutline(
        {side: tuple(found) for side, found in meetings.items()},
        any(_arc_positions(arc, widened) for arc in arcs),
        (
            (round(min(inside) / PRECISION_UM), round(max(inside) / PRECISION_UM))
            if inside
            else None
        ),
    )


Line = tuple[tuple[float, float], tuple[float, float]]
"""A straight outline edge: its two end points in model XY, in micrometres."""


@dataclass(frozen=True)
class _Arc:
    """A circular arc of a terminal's built outline, in the junction's local frame, in micrometres.

    ``sweep`` is the signed angle from ``start`` to ``end`` about ``centre``: positive when the
    arc runs counter-clockwise in the local frame.  A full circle has ``start`` equal to
    ``end`` and a sweep of a whole turn.
    """

    centre: tuple[float, float]
    radius: float
    start: tuple[float, float]
    end: tuple[float, float]
    sweep: float

    def holds(self, point: tuple[float, float]) -> bool:
        """Whether a point of the arc's circle lies on the arc."""
        if (
            math.dist(point, self.start) <= PRECISION_UM
            or math.dist(point, self.end) <= PRECISION_UM
        ):
            return True
        turned = _angle(point, self.centre) - _angle(self.start, self.centre)
        return (turned if self.sweep > 0 else -turned) % math.tau <= abs(self.sweep)


def _angle(point: tuple[float, float], centre: tuple[float, float]) -> float:
    return math.atan2(point[1] - centre[1], point[0] - centre[0])


class _OutlineEdges(NamedTuple):
    """A conductor outline's straight edges, in model XY, and its circular arcs, in the local frame."""

    lines: list[Line]
    arcs: list[_Arc]


def _outline_edges(
    build_input: GeometryBuildInput, selected: set[str], frame: JunctionLocalFrame
) -> _OutlineEdges:
    """The straight edges, in model XY, and the circular arcs, in the local frame, of these conductors' outline.

    A ring a boundary record binds is read from SGB's binding; every other ring from its drawn
    vertices, which needs no grid.
    """
    records = (*build_input.boundary_curves, *build_input.boundary_reconstruction)
    grid, bound = (
        bound_ring_primitives(build_input, selected)
        if any(record.entity_id in selected for record in records)
        else (None, {})
    )
    lines: list[Line] = []
    arcs: list[_Arc] = []
    for entity in build_input.entities:
        if entity.semantic_id not in selected:
            continue
        loops, _ = conductor_loops(entity, build_input)
        for polygon, (outer, holes) in enumerate(loops):
            for ring_index, ring in enumerate((outer, *holes)):
                primitives = bound.get((entity.semantic_id, polygon, ring_index))
                if grid is None or primitives is None:
                    lines.extend(_drawn_edges(ring))
                    continue
                lines.extend(
                    (_um_point(p.p, grid), _um_point(p.q, grid))
                    for p in primitives
                    if isinstance(p, LineEdge)
                )
                arcs.extend(_bound_arcs(primitives, grid, frame))
    return _OutlineEdges(lines, arcs)


def _drawn_edges(ring) -> list[Line]:
    """The edges of a drawn ring in micrometres, the last one closing it."""
    points = [(float(x), float(y)) for x, y in ring]
    return list(zip(points, (*points[1:], points[0]), strict=True))


def _bound_arcs(
    primitives: Iterable[RingPrimitive], grid: Grid, frame: JunctionLocalFrame
) -> list[_Arc]:
    """The source arcs of a bound ring, each joined from the pieces SGB cuts it into.

    SGB cuts a source arc into pieces of at most a quarter turn at lattice points, and a
    piece's inner ends are rounded onto the lattice; so an arc is read from its own two ends,
    its exact circle and the sum of its pieces' sweeps, never from an inner end.
    """
    pieces: dict[str, list[ArcCarrier]] = {}
    for primitive in primitives:
        if isinstance(primitive, ArcCarrier):
            pieces.setdefault(primitive.root_id, []).append(primitive)
    # A mirroring frame turns the arc's sense of travel round.
    keeps_sense = frame.dx * frame.py - frame.dy * frame.px > 0
    arcs = []
    for chain in pieces.values():
        first = chain[0]
        turn = float(sum(abs(piece.sweep()[1]) for piece in chain))
        arcs.append(
            _Arc(
                _local_um(_um_point(first.centre, grid), frame),
                float(first.radius * Fraction(repr(grid.size_um))),
                _local_um(_um_point(first.start, grid), frame),
                _local_um(_um_point(chain[-1].end, grid), frame),
                turn if first.ccw == keeps_sense else -turn,
            )
        )
    return arcs


def _um_point(point, grid: Grid) -> tuple[float, float]:
    """A grid-unit point in micrometres: the float nearest its exact value."""
    size = Fraction(repr(grid.size_um))
    return (float(Fraction(point[0]) * size), float(Fraction(point[1]) * size))


def _arc_positions(arc: _Arc, box: tuple[float, float, float, float]) -> list[float]:
    """Where a circular arc inside the box ``(low u, high u, low v, high v)`` reaches farthest along u.

    These are the arc's ends inside the box, its crossings with the box's four edges and its
    points farthest along u, each where it lies on the arc and in the box, in micrometres.  The
    arc is continuous and moves monotonically along u between those points, so the list is
    empty exactly when the arc misses the box, and its extremes are the arc's reach inside it.
    """
    low_u, high_u, low_v, high_v = box
    (cu, cv), radius = arc.centre, arc.radius

    def in_box(u: float, v: float) -> bool:
        return low_u <= u <= high_u and low_v <= v <= high_v

    found = [u for u, v in (arc.start, arc.end) if in_box(u, v)]
    found.extend(
        u for u in (cu - radius, cu + radius) if in_box(u, cv) and arc.holds((u, cv))
    )
    for at_u, value in ((True, low_u), (True, high_u), (False, low_v), (False, high_v)):
        offset = value - (cu if at_u else cv)
        if abs(offset) > radius:
            continue
        half = math.sqrt(radius * radius - offset * offset)
        centre = cv if at_u else cu
        for across in (centre - half, centre + half):
            point = (value, across) if at_u else (across, value)
            if in_box(*point) and arc.holds(point):
                found.append(point[0])
    return found


def _line_meetings(
    line: Line, frame: JunctionLocalFrame, side: int, grid_um: float | None
) -> list[SideMeeting]:
    """The meetings of one straight outline edge with the side line ``v = side``."""
    (u0, v0), (u1, v1) = (_local_units(point, frame) for point in line)
    if v0 == v1 == side:
        return [SideMeeting(u0, True), SideMeeting(u1, True)]
    if v0 == v1 or (v0 - side) * (v1 - side) > 0:
        return []
    crossing = u0 + Fraction(side - v0, v1 - v0) * (u1 - u0)
    return [_crossing(float(crossing) * PRECISION_UM, side, frame, grid_um)]


def _arc_meetings(
    arc: _Arc, frame: JunctionLocalFrame, side: int, grid_um: float | None
) -> list[SideMeeting]:
    """The meetings of one circular arc with the side line ``v = side``."""
    (cu, cv), radius = arc.centre, arc.radius
    offset = side * PRECISION_UM - cv
    if abs(offset) > radius:
        return []
    half = math.sqrt(radius * radius - offset * offset)
    return [
        _crossing(u, side, frame, grid_um)
        for u in sorted({cu - half, cu + half})
        if arc.holds((u, side * PRECISION_UM))
    ]


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


class _NearestMeeting(NamedTuple):
    """Where a terminal meets a side line nearest the cut, and whether an edge along the side ends there."""

    position: int
    along_side: bool


def _nearest_meeting(found, cut, span, *, left) -> _NearestMeeting:
    """The terminal's meeting with one side line nearest the cut on its side, or the sheet corner, and whether an edge along the side ends there."""
    if left:
        reached = [m for m in found if 0 <= m.position <= cut]
        nearest = max((m.position for m in reached), default=0)
    else:
        reached = [m for m in found if cut <= m.position <= span]
        nearest = min((m.position for m in reached), default=span)
    return _NearestMeeting(
        nearest, any(m.along_side and m.position == nearest for m in reached)
    )
