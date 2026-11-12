"""Conductors cut by the die edge: where the die edge lies on a level, and whether a cut can be continued.

Owns two questions about the die's outer edge, a dicing line.  SGB treats a conductor that the die edge cuts as
continuing past it (``SupportRegion.die_cuts``), which is sound only where nothing of the die lies beside the
continuation.

* **Where the die edge lies** (:func:`cells_beyond_the_die`).  A level cell lies beyond the die when no substrate
  and no metal meet the plane there and such cells join it to the outside of the model.  The rim of a vacuum
  bore, trench or cavity that the die encloses is no dicing line.
* **Whether a cut may be continued** (:func:`check_die_cuts`).  Only a straight origin is continued, so the
  build is refused where an arc origin ends on the die edge (``SupportRegion.cut_arcs``).  Each continuation runs
  from the cut vertex to a point twice the largest margin beyond the die edge's line.  The build is refused
  where the die outline comes back within the largest margin of a continuation on the far side of that line, as
  beside a notch, or where two continuations of one support cross: a band there would come from a conductor
  that is not drawn, and SCGSim does not model it.  Nothing is extended further and no cap is put back.

Every test is exact, in fractions of the lattice.  A curved stretch of the die outline is enclosed by the circle
on its chord, which holds an arc of at most a half turn, so a cut beside one is refused rather than cleared.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from scgsim.geometry.sgb.derivation.layer import Layer
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.records.carriers import (
    ArcCarrier,
    RingPrimitive,
    round_to_lattice,
)
from scgsim.geometry.sgb.records.labels import LevelLabel
from scgsim.geometry.sgb.records.support import ArcCut, DieCut, SupportRegion

Exact = tuple[Fraction, Fraction]


def cells_beyond_the_die(layer: Layer) -> frozenset[int]:
    """The cells of a level beyond the die's outer edge: off the die, and joined to the unbounded face through such cells.

    A cell is off the die when no substrate and no metal meet the plane there (:func:`_is_off_the_die`).  Those
    beside the die's outer edge reach the unbounded face (face 0, outside the model) across edges between off-die
    cells; a vacuum bore, trench or cavity that the die encloses does not, whatever lies in it.  The search follows
    the arrangement's edges, so it is exact.
    """
    arr = layer.arr
    neighbours: list[list[int]] = [[] for _ in arr.faces]
    for edge in range(len(arr.edges)):
        left, right = arr.he_face[2 * edge], arr.he_face[2 * edge + 1]
        neighbours[left].append(right)
        neighbours[right].append(left)
    beyond = {0}
    queue: deque[int] = deque([0])
    while queue:
        for other in neighbours[queue.popleft()]:
            if other not in beyond and _is_off_the_die(layer.labels[other]):
                beyond.add(other)
                queue.append(other)
    return frozenset(beyond)


def _is_off_the_die(label: LevelLabel | None) -> bool:
    """Whether a level cell holds no substrate and no metal: outside the model, or only vacuum, or nothing, on each side."""
    if label is None:
        return True
    if label.sheet is not None:
        return False
    return all(
        domain is None or domain.kind == "vacuum"
        for domain in (label.below, label.above)
    )


def check_die_cuts(  # noqa: PLR0913 - one support, the plane its die edge was read on, and what names it
    request_id: str,
    region: SupportRegion,
    plane: Layer,
    beyond: frozenset[int],
    conductor: str | None,
    grid_um: float,
) -> None:
    """Refuse a support cut by the die edge along an arc, or whose continuations past the die edge the die outline comes back to, or that cross.

    ``plane`` is the level the die edge was read on and ``beyond`` its cells beyond the die.  ``conductor`` names
    the cut conductor of a conductor face; for a solution interface it is ``None`` and each cut is named by the
    conductor beside it on the plane.  ``grid_um`` only scales the locations in the message.
    """
    if region.cut_arcs:
        arc = region.cut_arcs[0]
        raise InputError(
            f"inset request {request_id!r}: the die edge cuts {_name(conductor, plane, region, arc)!r} at "
            f"{_at(arc.vertex, grid_um)} um, where its gap edge is an exact arc. A conductor cut by the die edge "
            "is continued straight past it, which an arc cannot be; SCGSim does not model it. Let a straight "
            "edge of the conductor meet the die edge"
        )
    cuts = region.die_cuts
    if not cuts:
        return
    reach = region.margins[-1]
    outline = _die_outline(plane, beyond)
    for cut in cuts:
        if any(_comes_back(cut, edge, reach) for edge in outline):
            raise InputError(
                f"inset request {request_id!r}: the die outline comes back within the band margin "
                f"({reach * grid_um:g} um) of the continuation of {_name(conductor, plane, region, cut)!r}, cut "
                f"by the die edge at {_at(cut.vertex, grid_um)} um. A conductor cut by the die edge is continued "
                "straight past it, and the die beside that continuation would take bands from a conductor that "
                "is not drawn; SCGSim does not model it. Keep the die outline farther than the margin from it"
            )
    for k, first in enumerate(cuts):
        for second in cuts[k + 1 :]:
            if _segments_meet(
                _exact(first.vertex),
                _exact(first.end),
                _exact(second.vertex),
                _exact(second.end),
            ):
                raise InputError(
                    f"inset request {request_id!r}: the continuations of "
                    f"{_name(conductor, plane, region, first)!r} and {_name(conductor, plane, region, second)!r}, "
                    f"cut by the die edge at {_at(first.vertex, grid_um)} and {_at(second.vertex, grid_um)} um, "
                    "cross past the die edge. A conductor cut by the die edge is continued straight past it, and "
                    "two that cross there give the bands between them no single distance; SCGSim does not model it"
                )


def _at(point: Point, grid_um: float) -> str:
    return f"({point[0] * grid_um:.4f}, {point[1] * grid_um:.4f})"


# --------------------------------------------------------------------------- the die outline


@dataclass(frozen=True)
class _OutlineEdge:
    """An edge of the die outline on the plane: its lattice end points, and whether it is an arc."""

    a: Point
    b: Point
    curved: bool


def _die_outline(plane: Layer, beyond: frozenset[int]) -> list[_OutlineEdge]:
    """The edges of the plane between a cell beyond the die and one that is not."""
    arr = plane.arr
    return [
        _OutlineEdge(edge.a, edge.b, plane.edge_curve[index] is not None)
        for index, edge in enumerate(arr.edges)
        if (arr.he_face[2 * index] in beyond) != (arr.he_face[2 * index + 1] in beyond)
    ]


def _comes_back(cut: DieCut, edge: _OutlineEdge, reach: int) -> bool:
    """Whether a stretch of the die outline past the die edge's line at the cut lies within ``reach`` of the continuation.

    The die edge's line at the cut bounds what the continuation may touch: on the die's side of it lies the cut
    itself, the die the conductor came from.  An outline stretch on that line or on the die's side is no return.
    """
    vertex, end = _exact(cut.vertex), _exact(cut.end)
    a, b = _exact(edge.a), _exact(edge.b)
    if edge.curved:
        return _arc_comes_back(cut, a, b, vertex, end, reach)
    side_a, side_b = _side(cut, a), _side(cut, b)
    if side_a >= 0 and side_b >= 0:
        return False
    if side_a > 0:
        a = _along(a, b, side_a / (side_a - side_b))
    elif side_b > 0:
        b = _along(b, a, side_b / (side_b - side_a))
    return _segment_distance_squared(a, b, vertex, end) <= reach * reach


def _arc_comes_back(  # noqa: PLR0913 - the cut, the arc's chord, the continuation and the margin
    cut: DieCut, a: Exact, b: Exact, vertex: Exact, end: Exact, reach: int
) -> bool:
    """The same for a curved outline stretch, enclosed by the circle on its chord ``a``-``b``.

    An arc of at most a half turn lies in the disk whose diameter is its chord, so the disk stands for it: no
    return when the disk lies on the die's side of the die edge's line, else a return when it comes within
    ``reach`` of the continuation.
    """
    centre = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    radius_squared = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) / 4
    dx, dy = cut.die_direction
    side = _side(cut, centre)
    if side > 0 and side * side >= radius_squared * (dx * dx + dy * dy):
        return False
    # distance <= reach + radius, squared without square roots
    excess = (
        _point_segment_squared(centre, vertex, end) - reach * reach - radius_squared
    )
    return excess <= 0 or excess * excess <= 4 * reach * reach * radius_squared


def _side(cut: DieCut, point: Exact) -> Fraction:
    """Positive on the die's side of the die edge's line at the cut, zero on it, negative past it."""
    dx, dy = cut.die_direction
    return dx * (point[1] - cut.vertex[1]) - dy * (point[0] - cut.vertex[0])


# --------------------------------------------------------------------------- naming the cut conductor


def _name(
    conductor: str | None, plane: Layer, region: SupportRegion, cut: DieCut | ArcCut
) -> str:
    """The cut conductor: given for a conductor face, else the conductor beside the cut edge on the plane."""
    if conductor is not None:
        return conductor
    prim = region.rings[cut.ring][cut.primitive]
    arr = plane.arr
    for index, edge in enumerate(arr.edges):
        if cut.vertex not in (edge.a, edge.b):
            continue
        other = edge.b if edge.a == cut.vertex else edge.a
        if not _runs_back_along(prim, cut, other, plane.edge_curve[index] is not None):
            continue
        for half_edge in (2 * index, 2 * index + 1):
            owner = _conductor_owner(plane.labels[arr.he_face[half_edge]])
            if owner is not None:
                return owner
    return "a conductor"


def _runs_back_along(
    prim: RingPrimitive, cut: DieCut | ArcCut, other: Point, curved: bool
) -> bool:
    """Whether the plane edge from the cut vertex to ``other`` runs back along the cut origin.

    A straight origin runs along it in the same direction.  An arc origin of a solution interface is one curved
    edge of the plane, its own level, so that edge ends where the arc does.
    """
    if isinstance(prim, ArcCarrier):
        inner = round_to_lattice(prim.start if cut.at_end else prim.end)
        return curved and other == inner
    inner = prim.p if cut.at_end else prim.q
    toward = (inner[0] - cut.vertex[0], inner[1] - cut.vertex[1])
    run = (other[0] - cut.vertex[0], other[1] - cut.vertex[1])
    parallel = run[0] * toward[1] - run[1] * toward[0] == 0
    return parallel and run[0] * toward[0] + run[1] * toward[1] > 0


def _conductor_owner(label: LevelLabel | None) -> str | None:
    if label is None:
        return None
    if label.sheet is not None and label.sheet.kind == "conductor":
        return label.sheet.name
    for domain in (label.below, label.above):
        if domain is not None and domain.kind == "conductor":
            return domain.name
    return None


# --------------------------------------------------------------------------- exact plane geometry


def _exact(point: Point) -> Exact:
    return Fraction(point[0]), Fraction(point[1])


def _along(start: Exact, end: Exact, t: Fraction) -> Exact:
    return start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])


def _orientation(o: Exact, a: Exact, b: Exact) -> Fraction:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _within_box(a: Exact, b: Exact, p: Exact) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[
        1
    ] <= max(a[1], b[1])


def _segments_meet(a: Exact, b: Exact, c: Exact, d: Exact) -> bool:
    """Whether two closed segments share a point."""
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > 0 > o2) or (o1 < 0 < o2)) and ((o3 > 0 > o4) or (o3 < 0 < o4)):
        return True
    return (
        (o1 == 0 and _within_box(a, b, c))
        or (o2 == 0 and _within_box(a, b, d))
        or (o3 == 0 and _within_box(c, d, a))
        or (o4 == 0 and _within_box(c, d, b))
    )


def _point_segment_squared(p: Exact, a: Exact, b: Exact) -> Fraction:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_squared = dx * dx + dy * dy
    t = Fraction(0)
    if length_squared:
        t = min(
            Fraction(1),
            max(
                Fraction(0), ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_squared
            ),
        )
    x, y = a[0] + t * dx, a[1] + t * dy
    return (p[0] - x) ** 2 + (p[1] - y) ** 2


def _segment_distance_squared(a: Exact, b: Exact, c: Exact, d: Exact) -> Fraction:
    """The squared distance between two closed segments."""
    if _segments_meet(a, b, c, d):
        return Fraction(0)
    return min(
        _point_segment_squared(a, c, d),
        _point_segment_squared(b, c, d),
        _point_segment_squared(c, a, b),
        _point_segment_squared(d, a, b),
    )
