"""Read a drawn junction sheet as a rectangle whose corners are grid points, in any direction.

On a grid of step g, a rectangle whose four corners are grid points (a lattice rectangle) has
integer side vectors: its side along the port is k v, for v = (p, q) that side divided by the
gcd of its components (a primitive integer vector), and its other side is m (-q, p).  Cut lines
every |v| grid steps along v from a corner then end on grid points on both sides.  Along x or
y, v is a unit vector and the cut lines are the grid lines, as before any other direction was
supported.

This module owns how a drawn sheet becomes such a rectangle (``lattice_rectangle``): its four
corners, the shift onto the nearest lattice rectangle when the drawn corners are not one, the
cut positions a partition may choose from (``CutLattice``) and the sheet the partition models,
straight between its two cuts (``LatticeRectangle.modelled_sheet``).  Choosing the cuts is
``partition``'s.

The only tolerance is the grid's snap bound: the grid rounds a point to the grid point whose
cell holds it, so a rounded point lies within half a grid diagonal, (sqrt 2 / 2) g, of where
it was drawn, the bound Supported Geometry states.  Every comparison against it is exact.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from itertools import product
from typing import Any

from scgsim.geometry.models.grid import format_point
from scgsim.geometry.planning.junction_planning.core import PRECISION_UM
from scgsim.geometry.planning.junction_planning.exact_geometry import cross
from scgsim.geometry.planning.junction_planning.local_frame import JunctionLocalFrame
from scgsim.geometry.planning.junction_planning.selection import (
    bound_decimal,
    candidate_indices,
    grid_cut,
)
from scgsim.geometry.planning.junction_planning.terminal import rectangle_along
from scgsim.geometry.primitives.grid_regions import grid_change_message

type GridPoint = tuple[Fraction, Fraction]
type PointUm = tuple[float, float]

# The snap bound squared, in grid steps squared: (sqrt 2 / 2)^2.
SNAP_BOUND_SQUARED_STEPS = Fraction(1, 2)

# The long sides of a lattice rectangle by its corner labels (0 = A ... 3 = D): A-B is side 0,
# along the lattice line through A, and C-D is side 1, through D.
_LONG_SIDES = {frozenset((0, 1)): 0, frozenset((2, 3)): 1}


@dataclass(frozen=True)
class LatticeRectangle:
    """A drawn sheet read as a rectangle with grid-point corners, and how far each corner moved.

    ``corners`` and ``drawn`` are in grid steps, in the order A, B, C, D: A starts the port
    on its low side (the right of the port direction), B = A + k v ends it, C = B + m w and
    D = A + m w, with ``direction`` v = (p, q) along the port and w = (-q, p).  ``drawn`` are
    the corners as drawn, before the grid rounded them.  ``sheet`` is the sheet's ring on the
    grid as the partition receives it, and ``sheet_corners`` the ring positions of its corners
    A, B, C and D.
    """

    corners: tuple[GridPoint, GridPoint, GridPoint, GridPoint]
    drawn: tuple[GridPoint, GridPoint, GridPoint, GridPoint]
    direction: tuple[int, int]
    length_steps: int
    width_steps: int
    grid_um: float
    sheet: tuple[GridPoint, ...]
    sheet_corners: tuple[int, int, int, int]

    @property
    def unit(self) -> bool:
        """Whether the port runs along x or y, where every grid step along it is a cut position."""
        p, q = self.direction
        return p * p + q * q == 1

    @property
    def spacing_um(self) -> float:
        """The distance between neighbouring cut positions along the port, |v| g."""
        p, q = self.direction
        return math.hypot(p, q) * self.grid_um

    @property
    def corners_um(self) -> tuple[PointUm, ...]:
        """The four corners in micrometres, in the order A, B, C, D."""
        return tuple(self._um(point) for point in self.corners)

    def cut_ends_um(self, index: int) -> tuple[PointUm, PointUm]:
        """The two ends of cut ``index``: ``index`` |v| grid steps along the port from A and from D."""
        a, d = self._cut_end(0, index), self._cut_end(1, index)
        return (self._um(a), self._um(d))

    def record(self) -> dict[str, Any]:
        """The lattice as the partition record and the port metadata carry it."""
        grid = Fraction(str(self.grid_um))
        return {
            "direction_grid": list(self.direction),
            "grid_um": self.grid_um,
            "cut_spacing_um": self.spacing_um,
            "length_steps": self.length_steps,
            "width_steps": self.width_steps,
            "corners_grid": [[int(x), int(y)] for x, y in self.corners],
            "corners_um": [list(point) for point in self.corners_um],
            "drawn_corners_um": [list(self._um(point)) for point in self.drawn],
            "corner_shifts_um": [
                [float((c[0] - d[0]) * grid), float((c[1] - d[1]) * grid)]
                for c, d in zip(self.corners, self.drawn, strict=True)
            ],
        }

    def cut_record(self, index: int) -> dict[str, Any]:
        """Where cut ``index`` lies: its distance from corner A along the port and its two ends."""
        return {
            "index": index,
            "from_corner_a_um": index * self.spacing_um,
            "ends_um": [list(end) for end in self.cut_ends_um(index)],
        }

    def modelled_sheet(self, cuts: tuple[int, int]) -> tuple[PointUm, ...] | None:
        """The sheet with both long sides on their lattice lines between the cuts, or ``None``.

        The grid rounds the point where a conductor outline crosses a long side to a grid
        point, which on a side that is not along x or y can lie off the side, by up to the snap
        bound; the edge from there to the next vertex then crosses the whole gap askew.  A
        shifted corner tilts its sides the same way.  Here each long side runs along its
        lattice line from cut to cut, through the two cut ends, and keeps its vertices beyond
        the cuts, so the central rectangle is exact and every vertex the grid shares with a
        conductor stays where both have it.  ``None`` when every vertex of the long sides
        already lies on its lattice line.
        """
        sides = list(self._ring_sides())
        if all(
            self._offset(point, side) == 0
            for vertices, following, side in sides
            if side is not None
            for point in (*vertices, following)
        ):
            return None
        result: list[GridPoint] = []
        for vertices, following, side in sides:
            if side is None:
                result.extend(vertices)
            else:
                result.extend(self._straightened(vertices, following, side, cuts))
        return tuple(self._um(point) for point in result)

    def _ring_sides(self) -> Iterator[tuple[list[GridPoint], GridPoint, int | None]]:
        """The ring's four sides in ring order: the vertices from one corner up to the next, that next corner, and the long side it is."""
        count = len(self.sheet)
        start_a = self.sheet_corners[0]
        labels = sorted(
            range(4), key=lambda n: (self.sheet_corners[n] - start_a) % count
        )
        for k, label in enumerate(labels):
            following = labels[(k + 1) % 4]
            start, finish = self.sheet_corners[label], self.sheet_corners[following]
            vertices = [
                self.sheet[(start + n) % count] for n in range((finish - start) % count)
            ]
            side = _LONG_SIDES.get(frozenset((label, following)))
            yield vertices, self.sheet[finish], side

    def _straightened(self, vertices, following, side, cuts) -> list[GridPoint]:
        """One long side, ``vertices`` up to ``following``, along its lattice line between the cuts."""
        first, last = cuts
        increasing = self._along(following) > self._along(vertices[0])
        if increasing:
            before = [p for p in vertices if self._along(p) < first]
            after = [p for p in vertices if self._along(p) > last]
            ends = [self._cut_end(side, first), self._cut_end(side, last)]
        else:
            before = [p for p in vertices if self._along(p) > last]
            after = [p for p in vertices if self._along(p) < first]
            ends = [self._cut_end(side, last), self._cut_end(side, first)]
        return [*before, *ends, *after]

    def _cut_end(self, side: int, index: int) -> GridPoint:
        """The end of cut ``index`` on long side 0 (from A) or 1 (from D)."""
        p, q = self.direction
        corner = self.corners[0 if side == 0 else 3]
        return (corner[0] + index * p, corner[1] + index * q)

    def _offset(self, point: GridPoint, side: int) -> Fraction:
        """How far ``point`` lies off the lattice line of a long side, times |v|."""
        p, q = self.direction
        corner = self.corners[0 if side == 0 else 3]
        return (point[0] - corner[0]) * -q + (point[1] - corner[1]) * p

    def _along(self, point: GridPoint) -> Fraction:
        """How many cut spacings ``point`` lies along the port from the start corner A."""
        p, q = self.direction
        a = self.corners[0]
        return ((point[0] - a[0]) * p + (point[1] - a[1]) * q) / (p * p + q * q)

    def _um(self, point: GridPoint) -> PointUm:
        grid = Fraction(str(self.grid_um))
        return (float(point[0] * grid), float(point[1] * grid))


def drawn_exterior(sheet: Any) -> Sequence[Sequence[float]]:
    """The outer ring of a sheet (a polygon or a port sheet region) as drawn, before the grid rounded it.

    The grid realization keeps each rounded polygon's drawn rings in its metadata.  A sheet the
    partition already cut is its central rectangle, emitted on grid points; the drawn rings it
    still carries are the whole sheet's, so it is read as it is, like a sheet the grid never
    rounded.
    """
    metadata = sheet.metadata
    drawn = metadata.get("unquantized_source_geometry")
    if drawn is None or "rectangular_junction_partition" in metadata:
        return sheet.exterior
    return drawn["exterior"]


def lattice_rectangle(  # noqa: PLR0913 -- the sheet on the grid, as drawn, and how a refusal names it
    exterior: Sequence[Sequence[float]],
    grid_um: float,
    direction_xy: tuple[float, float],
    *,
    drawn: Sequence[Sequence[float]],
    port: str,
    layer: str,
) -> LatticeRectangle:
    """Read a sheet as its lattice rectangle, oriented along ``direction_xy``.

    ``exterior`` is the sheet on the grid, ``drawn`` the same sheet as drawn (``drawn_exterior``).

    1. The four corners: a vertex is dropped when it lies within the snap bound of the line
       between the two corners of its side, as a vertex the grid adds where an outline crosses
       a side does.  On the grid, each corner is the one vertex within the snap bound of its
       drawn corner.
    2. The lattice rectangle: the nearest one to the drawn corners, by the sum of squared
       corner shifts, whose corners each lie within the snap bound of the drawn ones.  For
       corners drawn on grid points that is the drawn rectangle itself, or none.  It must run
       along the port direction, as ``junction_terminal_line`` checks a sheet.

    ``port`` and ``layer`` name the sheet in a refusal, which says what must change.
    """
    sheet = _grid_points(exterior, grid_um)
    drawn_points = _grid_points(drawn, grid_um)
    found = _sheet_corners(drawn_points)
    if found is None:
        raise ValueError(_not_a_rectangle(port, layer, drawn_points, grid_um))
    drawn_corners = [drawn_points[n] for n in found]
    positions = _rounded_corners(sheet, drawn_corners)
    if positions is None:
        raise ValueError(_rounded_off_its_sides(port, layer, sheet, grid_um))
    if not _forward(positions, len(sheet)):
        # The ring runs the other way round than the drawn one: read the corners in ring order.
        drawn_corners.reverse()
        positions.reverse()
    lattice = _nearest_lattice_rectangle(drawn_corners)
    if lattice is None:
        raise ValueError(
            _no_lattice_rectangle(port, layer, drawn_corners, grid_um, None)
        )
    rectangle = _oriented(
        lattice, drawn_corners, sheet, positions, direction_xy, grid_um
    )
    if not rectangle_along(rectangle.corners_um, direction_xy):
        crosswise = (rectangle.direction, direction_xy)
        raise ValueError(
            _no_lattice_rectangle(port, layer, drawn_corners, grid_um, crosswise)
        )
    return rectangle


def _grid_points(ring, grid_um) -> list[GridPoint]:
    """The ring's vertices in grid steps, exact, without repeated consecutive points."""
    grid = Fraction(str(grid_um))
    points = [
        (Fraction(str(float(x))) / grid, Fraction(str(float(y))) / grid)
        for x, y in ring
    ]
    return [point for n, point in enumerate(points) if point != points[n - 1]]


def _sheet_corners(points: list[GridPoint]) -> list[int] | None:
    """The indices of the sheet's four corners in ``points``, or ``None`` when it has no four.

    Vertices are dropped nearest first while one lies within the snap bound of the line
    through its two remaining neighbours; the four that remain are then checked against the
    rule itself: every dropped vertex lies within the snap bound of the line between the two
    corners of its side.
    """
    kept = list(range(len(points)))
    while len(kept) > 4:
        nearest = min(
            (
                (ratio, n)
                for n in range(len(kept))
                if (ratio := _offset_ratio(points, kept, n)) is not None
            ),
            default=None,
        )
        if nearest is None:
            return None
        del kept[nearest[1]]
    if len(kept) != 4:
        return None
    for n, start in enumerate(kept):
        finish = kept[(n + 1) % 4]
        if any(
            not _within_snap_bound(points[start], points[finish], points[m])
            for m in _between(start, finish, len(points))
        ):
            return None
    return kept


def _offset_ratio(points, kept, n) -> Fraction | None:
    """The squared offset of kept vertex ``n`` from the line through its neighbours, over the snap bound's.

    ``None`` when the vertex lies beyond the snap bound, so it is a corner.
    """
    before, at, after = (points[kept[(n + k) % len(kept)]] for k in (-1, 0, 1))
    length = _squared(before, after)
    if length == 0:
        return None
    ratio = cross(before, after, at) ** 2 / (length * SNAP_BOUND_SQUARED_STEPS)
    return ratio if ratio <= 1 else None


def _rounded_corners(sheet, drawn_corners) -> list[int] | None:
    """The ring positions of the drawn corners in the sheet on the grid, in their order, or ``None``.

    Each corner is the vertex nearest its drawn corner, within the snap bound of it and nearer
    than any other, and every other vertex lies within the snap bound of the drawn side it is
    between.  For a sheet the grid never rounded, the corners are the drawn ones.
    """
    count = len(sheet)
    positions = []
    for corner in drawn_corners:
        distances = sorted(
            (_squared(point, corner), n) for n, point in enumerate(sheet)
        )
        (nearest, position), (runner_up, _) = distances[:2]
        if nearest > SNAP_BOUND_SQUARED_STEPS or runner_up == nearest:
            return None
        positions.append(position)
    if _forward(positions, count):
        order = [0, 1, 2, 3]
    elif _forward(positions[::-1], count):
        order = [3, 2, 1, 0]
    else:
        return None
    for n, m in zip(order, (*order[1:], order[0]), strict=True):
        line = (drawn_corners[n], drawn_corners[m])
        if any(
            not _within_snap_bound(*line, sheet[k])
            for k in _between(positions[n], positions[m], count)
        ):
            return None
    return positions


def _forward(positions: Sequence[int], count: int) -> bool:
    """Whether four ring positions follow each other in ring order."""
    return (
        sum((positions[(n + 1) % 4] - positions[n]) % count for n in range(4)) == count
    )


def _between(start: int, finish: int, count: int) -> list[int]:
    """The ring positions strictly after ``start`` and before ``finish``, in ring order."""
    return [(start + n) % count for n in range(1, (finish - start) % count)]


def _within_snap_bound(start: GridPoint, finish: GridPoint, point: GridPoint) -> bool:
    return cross(start, finish, point) ** 2 <= SNAP_BOUND_SQUARED_STEPS * _squared(
        start, finish
    )


def _squared(a: GridPoint, b: GridPoint) -> Fraction:
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


def _nearest_lattice_rectangle(drawn) -> tuple[GridPoint, ...] | None:
    """The rectangle with grid-point corners nearest the drawn corners, each within the snap bound."""
    choices = [_grid_points_near(corner) for corner in drawn]
    best = None
    for corners in product(*choices):
        if not _is_rectangle(corners):
            continue
        shift = sum(_squared(c, d) for c, d in zip(corners, drawn, strict=True))
        if best is None or (shift, corners) < best:
            best = (shift, corners)
    return None if best is None else best[1]


def _grid_points_near(point: GridPoint) -> list[GridPoint]:
    """The grid points within the snap bound of ``point``: itself alone when it is one."""
    xs = {math.floor(point[0]), math.ceil(point[0])}
    ys = {math.floor(point[1]), math.ceil(point[1])}
    return sorted(
        (Fraction(x), Fraction(y))
        for x, y in product(xs, ys)
        if _squared(point, (Fraction(x), Fraction(y))) <= SNAP_BOUND_SQUARED_STEPS
    )


def _is_rectangle(corners) -> bool:
    """Whether four grid points, in ring order, are the corners of a rectangle."""
    a, b, c, d = corners
    ab = (b[0] - a[0], b[1] - a[1])
    ad = (d[0] - a[0], d[1] - a[1])
    return (
        (a[0] + c[0], a[1] + c[1]) == (b[0] + d[0], b[1] + d[1])
        and ab[0] * ad[0] + ab[1] * ad[1] == 0
        and ab != (0, 0)
        and ad != (0, 0)
    )


def _oriented(  # noqa: PLR0913 -- the lattice, the drawn corners and the sheet they belong to
    lattice, drawn, sheet, positions, direction_xy, grid_um
) -> LatticeRectangle:
    """Order the corners A, B, C, D along the port direction and find v, k and m."""
    dx, dy = direction_xy
    sides = [
        (
            lattice[(n + 1) % 4][0] - lattice[n][0],
            lattice[(n + 1) % 4][1] - lattice[n][1],
        )
        for n in range(4)
    ]
    # The side most nearly along the port direction, by the cosine of the angle between them.
    along = max(
        sides,
        key=lambda s: (float(s[0]) * dx + float(s[1]) * dy) / math.hypot(*s),
    )
    steps = math.gcd(int(along[0]), int(along[1]))
    p, q = int(along[0]) // steps, int(along[1]) // steps

    def position(point):
        return (point[0] * p + point[1] * q, -point[0] * q + point[1] * p)

    start = min(range(4), key=lambda n: position(lattice[n]))
    order = [(start + k) % 4 for k in range(4)]
    a = lattice[order[0]]
    if (lattice[order[1]][0] - a[0], lattice[order[1]][1] - a[1]) != (
        steps * p,
        steps * q,
    ):
        order = [order[0], order[3], order[2], order[1]]
    d = lattice[order[3]]
    width = ((d[0] - a[0]) * -q + (d[1] - a[1]) * p) / (p * p + q * q)
    return LatticeRectangle(
        tuple(lattice[n] for n in order),
        tuple(drawn[n] for n in order),
        (p, q),
        steps,
        int(width),
        grid_um,
        tuple(sheet),
        tuple(positions[n] for n in order),
    )


def _format_points(points, grid_um) -> str:
    grid = Fraction(str(grid_um))
    return ", ".join(
        format_point(float(x * grid), float(y * grid), None) for x, y in points
    )


def _snap_bound_um(grid_um: float) -> str:
    return f"{math.sqrt(2) / 2 * grid_um:.3g} um"


def _not_a_rectangle(port, layer, points, grid_um) -> str:
    return grid_change_message(
        f"{port} is not a rectangle",
        [layer],
        grid_um,
        [
            "no four of its drawn vertices hold every other one within half a grid "
            f"diagonal ({_snap_bound_um(grid_um)}) of the sides between them; it is drawn "
            f"with the vertices {_format_points(points, grid_um)}"
        ],
        "Fix the layout: draw the sheet as a rectangle.",
    )


def _rounded_off_its_sides(port, layer, sheet, grid_um) -> str:
    return grid_change_message(
        f"{port} lost its rectangle on the grid",
        [layer],
        grid_um,
        [
            "on the grid it has the vertices "
            f"{_format_points(sheet, grid_um)}, and not each of its drawn corners has one "
            f"vertex within half a grid diagonal ({_snap_bound_um(grid_um)}) of it with the "
            "others within that bound of its drawn sides"
        ],
        "This is an SCGSim defect, not a layout rule: report it to SCGSim with the layout.",
    )


def _no_lattice_rectangle(port, layer, drawn, grid_um, crosswise) -> str:
    """The refusal of a sheet with no lattice rectangle near its drawn corners along its port.

    ``crosswise`` is ``None`` when no lattice rectangle lies that near, else the nearest one's
    direction v and the port direction it does not run along.
    """
    bound = f"half a grid diagonal ({_snap_bound_um(grid_um)})"
    if crosswise is None:
        finding = (
            "no rectangle with its corners on grid points has each corner within "
            f"{bound} of them"
        )
    else:
        (p, q), (dx, dy) = crosswise
        finding = (
            f"the nearest rectangle with its corners on grid points, each within {bound} "
            f"of them, runs along v = ({p}, {q}), not along the port direction "
            f"({dx:.6g}, {dy:.6g})"
        )
    return grid_change_message(
        f"{port} has no rectangle with its corners on the grid near its drawn corners",
        [layer],
        grid_um,
        [f"its corners are drawn at {_format_points(drawn, grid_um)}, and {finding}"],
        "Fix the layout: draw the sheet with its four corners on the grid, along x or y "
        "or along a direction of whole grid steps such as 45 degrees, or place a sheet "
        "drawn along x or y through an instance turned by a quarter turn.",
    )


@dataclass(frozen=True)
class CutLattice:
    """The positions a junction may be cut at: every |v| grid steps along the port from corner A.

    A position is given along the port from the local origin, in micrometres as requested and
    in the local Boolean units of ``frame``.  Along x or y these are the grid steps the
    partition always took, computed as before.
    """

    rectangle: LatticeRectangle
    frame: JunctionLocalFrame
    step: float

    def position(self, index: int) -> tuple[float, int]:
        """The requested distance of cut ``index`` from corner A, and the local unit it lies at."""
        if self.rectangle.unit:
            return grid_cut(index, self.step)
        (x, y), _ = self.rectangle.cut_ends_um(index)
        u = x * self.frame.dx + y * self.frame.dy - self.frame.origin[0]
        return (index * self.rectangle.spacing_um, round(u / PRECISION_UM))

    def candidates(self, bound, *, left: bool) -> list[int]:
        """The first cut positions beside ``bound``: on it, if one is, then the next one into the gap."""
        if self.rectangle.unit:
            return candidate_indices(bound, self.step, left=left)
        spacing = self.rectangle.spacing_um / PRECISION_UM
        index = math.floor(float(bound) / spacing) + (-1 if left else 2)
        toward = 1 if left else -1
        while (self.position(index)[1] - bound) * toward < 0:
            index += toward
        if self.position(index)[1] == bound:
            return [index, index + toward]
        return [index]

    def beyond(self, index: int, bound, *, left: bool) -> bool:
        """Whether cut ``index`` lies strictly inside the gap past ``bound``."""
        if self.rectangle.unit:
            requested = Decimal(index) * Decimal(str(self.step))
            limit = bound_decimal(bound)
            return requested > limit if left else requested < limit
        cut = self.position(index)[1]
        return cut > bound if left else cut < bound
