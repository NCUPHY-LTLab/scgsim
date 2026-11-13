"""The region ``R_m`` of one margin and its required seams, from the candidate offsets (design brief section 6).

Owns turning the candidate offset curves of one margin into the cells of a small arrangement and deciding,
per cell, whether it lies in the support and on which side of the margin:

1. **Arrangement.**  The candidates join the support boundary (every ring, origins and the rest) and the port
   footprint boundaries in one snap-rounded arrangement (``arrangement.dcel``).  Every candidate is an edge, so
   no cell crosses the level set ``dist = m`` wherever the candidates contain it.
   Where a gap in the candidates leaves one cell on both sides of the margin, the margin is refused.
2. **Support.**  A cell lies in the support when the support boundary has it on its interior side; the
   classification follows the boundary's orientation from cell to cell, so it is exact (no sample point).
3. **Side.**  One sample point of the cell whose distance to the origins differs from ``m`` by more than the
   slack of a cut curve decides whether the cell lies in ``R_m``.  A cell too thin for any such sample is as
   exact as the cut itself and takes the side of its interior point (``dist >= m``).
4. **Required seams.**  A seam is an edge of the arrangement that carries a candidate and has the support on
   both sides, with one side in ``R_m`` and the other outside it.  An edge between two cells on one side -- the
   two coincident offsets on the centre line of a gap exactly ``2m`` wide, a convex corner's overlap, a cap
   outside its run -- is no seam.  Edges inside a port's active interior are never seams, so a port stays one
   face; a seam along a port's boundary is no cut of its interior and stays.
5. **Clipping.**  A seam chain may end only on the support boundary or on a port boundary.  That is how a band
   reaching the die edge or a port is accounted for; an end anywhere else would be a lost boundary and is refused.

Positive-area cells are the only cells an arrangement has, so a band region that is only a line or a point
(the gap exactly ``2m`` wide) has no cell and no seam: a legitimately empty band.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from scgsim.geometry.sgb.arrangement.dcel import (
    Arrangement,
    build_arrangement,
    interior_point,
    sample_points,
)
from scgsim.geometry.sgb.errors import ValidationError
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import NodedEdge, Segment, snap_round
from scgsim.geometry.sgb.exact.plane import Point, RepPoint, point_in_ring
from scgsim.geometry.sgb.exact.carriers import (
    CarrierRole,
    EdgeCarrier,
    LineEdge,
    piece_carriers,
)
from scgsim.geometry.sgb.exact.prisms import Footprint
from scgsim.geometry.sgb.exact.support import DISTANCE_SLACK, SupportRegion

SUPPORT_BOUNDARY = "~support"
"""Prefix of the carrier ids of the support boundary (written, never read: the role is ``SUPPORT_BOUNDARY``)."""
PORT_BOUNDARY = "~port"
"""Prefix of the carrier ids of a port footprint boundary (written, never read: the role is ``PORT_BOUNDARY``)."""
CLIP_ROLES = (CarrierRole.SUPPORT_BOUNDARY, CarrierRole.PORT_BOUNDARY)
"""The roles of the boundaries a seam chain may end on."""


@dataclass(frozen=True)
class PortClip:
    """A lumped port as the seams see it: its footprint and the footprints of the hosts it lands on."""

    footprint: Footprint
    landings: tuple[Footprint, ...] = ()

    def holds(self, point: RepPoint) -> bool:
        """Whether a point off every boundary lies in the port's active interior (off its hosts)."""
        return _inside(self.footprint, point) and not any(
            _inside(host, point) for host in self.landings
        )


def _inside(footprint: Footprint, point: RepPoint) -> bool:
    return sum(point_in_ring(ring, point) for ring in footprint.rings()) % 2 == 1


@dataclass(frozen=True)
class RequiredEdge:
    """A noded seam edge (candidate carriers only) and whether ``R_m`` lies on the left of ``a -> b``."""

    edge: NodedEdge
    inside_left: bool


@dataclass(frozen=True)
class MarginCells:
    """What one margin's arrangement establishes: the seams of ``R_m`` and the cycles of its cells."""

    required: list[RequiredEdge]
    region_rings: tuple[tuple[Point, ...], ...]


def margin_cells(  # noqa: PLR0913 - one margin of one support, with its ports and grid
    family_id: str,
    candidates: list[Segment],
    region: SupportRegion,
    margin: int,
    ports: tuple[PortClip, ...],
    grid_um: float,
) -> MarginCells:
    """Build the arrangement of one margin's candidates and read ``R_m`` and its required seams from its cells."""
    directions: dict[EdgeCarrier, tuple[Point, Point]] = {}
    segments = [
        *candidates,
        *_support_segments(region, directions),
        *_port_boundary_segments(ports),
    ]
    arr = build_arrangement(snap_round(segments).edges)
    in_support = _cells_in_support(arr, directions)
    _check_curve_closed(arr, in_support, region, margin, family_id, grid_um)
    inside = {
        face: _in_region(arr, face, region, margin)
        for face, held in enumerate(in_support)
        if held
    }
    required = _required_edges(arr, inside, ports)
    _check_seam_ends(arr, required, grid_um)
    rings = tuple(
        tuple(arr.cycle_points(cycle))
        for face, held in sorted(inside.items())
        if held
        for cycle in arr.face_cycles(face)
    )
    return MarginCells(required, rings)


def _support_segments(
    region: SupportRegion, directions: dict[EdgeCarrier, tuple[Point, Point]]
) -> list[Segment]:
    """Every boundary primitive of the support, as noding input, recording each piece's direction of travel."""
    segments = []
    for ri, ring in enumerate(region.rings):
        for pi, prim in enumerate(ring):
            points = (
                [prim.p, prim.q]
                if isinstance(prim, LineEdge)
                else prim.witness_points()
            )
            pieces = piece_carriers(
                f"{SUPPORT_BOUNDARY}{ri}.{pi}",
                len(points) - 1,
                CarrierRole.SUPPORT_BOUNDARY,
            )
            for carrier, p, q in zip(pieces, points, points[1:], strict=False):
                directions[carrier] = (p, q)
                segments.append(Segment(p, q, frozenset([carrier])))
    return segments


def _port_boundary_segments(ports: tuple[PortClip, ...]) -> list[Segment]:
    """The port footprint boundaries, as noding input only: they split the candidates where they cross them."""
    return [
        Segment(
            p,
            q,
            frozenset(
                [EdgeCarrier(f"{PORT_BOUNDARY}{k}.{r}.{j}", CarrierRole.PORT_BOUNDARY)]
            ),
        )
        for k, clip in enumerate(ports)
        for r, ring in enumerate(clip.footprint.rings())
        for j, (p, q) in enumerate(zip(ring, (*ring[1:], ring[0]), strict=True))
    ]


def _cells_in_support(
    arr: Arrangement, directions: dict[EdgeCarrier, tuple[Point, Point]]
) -> list[bool]:
    """Per face, whether it lies in the support: seeded across the boundary, then spread across every other edge.

    The support's interior lies on the left of each boundary ring, and snap rounding keeps a boundary piece
    running the way of its source segment, so the two faces of a boundary edge are known exactly.  Every other
    edge has the same answer on both sides.
    """
    state: list[bool | None] = [None] * len(arr.faces)
    state[0] = False
    queue: deque[int] = deque([0])
    neighbours: list[list[int]] = [[] for _ in arr.faces]
    for ei, edge in enumerate(arr.edges):
        left, right = arr.he_face[2 * ei], arr.he_face[2 * ei + 1]
        along = [
            _dot(edge, directions[c])
            for c in edge.carriers
            if c.role is CarrierRole.SUPPORT_BOUNDARY
        ]
        if not along:
            neighbours[left].append(right)
            neighbours[right].append(left)
            continue
        for face, held in (
            (left, any(d > 0 for d in along)),
            (right, any(d < 0 for d in along)),
        ):
            if state[face] is None:
                state[face] = held
                queue.append(face)
            elif state[face] != held:
                raise _inconsistent_support(arr, face)
    while queue:
        face = queue.popleft()
        for other in neighbours[face]:
            if state[other] is None:
                state[other] = state[face]
                queue.append(other)
            elif state[other] != state[face]:
                raise _inconsistent_support(arr, other)
    return [bool(s) for s in state]


def _dot(edge: NodedEdge, direction: tuple[Point, Point]) -> int:
    (px, py), (qx, qy) = direction
    return (edge.b[0] - edge.a[0]) * (qx - px) + (edge.b[1] - edge.a[1]) * (qy - py)


def _inconsistent_support(arr: Arrangement, face: int) -> ValidationError:
    return ValidationError(
        f"inset cell at {arr.face_anchor(face)} (grid units) lies both inside and outside its support "
        "region: the support boundary is not a set of oriented rings"
    )


def _check_curve_closed(  # noqa: PLR0913 - one margin's cells, its support and what names it
    arr: Arrangement,
    in_support: list[bool],
    region: SupportRegion,
    margin: int,
    family_id: str,
    grid_um: float,
) -> None:
    """Refuse a margin whose iso-distance curve has a gap, so that one cell holds both sides of the margin.

    A candidate edge whose midpoint lies at the margin (within the slack of a cut) and that has the same cell on
    both sides is a loose end of a curve piece.  Two loose ends are legitimate: the overlap of two offsets past a
    convex crossing, which lies just outside ``R_m`` (it is nearer the next origin), and a dangling stub of one
    side.  Beside a gap the curve stops short of the next piece, so the same cell lies clearly inside ``R_m`` on
    one side of the loose end and clearly outside it on the other: two probes, one slack beyond the cut's own
    uncertainty on each side of the edge, find that.  Two coincident offsets on the centre line of a gap ``2m``
    wide separate two cells and are never loose ends.
    """
    for ei, edge in enumerate(arr.edges):
        face = arr.he_face[2 * ei]
        if face != arr.he_face[2 * ei + 1] or not in_support[face]:
            continue
        if all(c.role is not CarrierRole.CURVE for c in edge.carriers):
            continue  # no candidate lies here
        x, y = (edge.a[0] + edge.b[0]) / 2, (edge.a[1] + edge.b[1]) / 2
        if abs(region.distance((x, y)) - margin) > DISTANCE_SLACK:
            continue
        if _straddles(arr, face, edge, region, margin):
            raise ValidationError(
                f"inset family {family_id!r}: the inset curve at m = {margin * grid_um:g} um is not "
                f"closed near ({x * grid_um:.4f}, {y * grid_um:.4f}) um: two offsets of the support "
                "boundary neither meet nor cross there, so the band regions cannot be told apart. "
                "This is an SCGSim defect, not a layout rule: report it to SCGSim with the layout "
                "(see 'When something fails', docs/supported-scope, in the SCGSim documentation)"
            )


PROBE_DENOMINATOR = 1000  # probe points are rational with this denominator (a thousandth of a grid unit)


def _straddles(
    arr: Arrangement, face: int, edge: NodedEdge, region: SupportRegion, margin: int
) -> bool:
    """Whether the cell holds a point clearly inside ``R_m`` on one side of a loose end and clearly outside on the other."""
    dx, dy = edge.b[0] - edge.a[0], edge.b[1] - edge.a[1]
    length = math.hypot(dx, dy)
    reach = 2 * DISTANCE_SLACK / length
    mx, my = (edge.a[0] + edge.b[0]) / 2, (edge.a[1] + edge.b[1]) / 2
    rings = [arr.cycle_points(c) for c in arr.face_cycles(face)]
    sides = set()
    for sign in (1, -1):
        px, py = mx - sign * dy * reach, my + sign * dx * reach
        probe = (
            round(px * PROBE_DENOMINATOR),
            round(py * PROBE_DENOMINATOR),
            PROBE_DENOMINATOR,
        )
        if sum(point_in_ring(ring, probe) for ring in rings) % 2 == 0:
            return (
                False  # the probe left the cell: the curve separates the two sides here
            )
        distance = region.distance((px, py))
        if abs(distance - margin) <= DISTANCE_SLACK:
            return False
        sides.add(distance > margin)
    return len(sides) == 2


def _in_region(arr: Arrangement, face: int, region: SupportRegion, margin: int) -> bool:
    """Whether a cell of the support lies in ``R_m``: the first sample clear of the margin decides."""
    for x, y, den in sample_points(arr, face):
        distance = region.distance((x / den, y / den))
        if abs(distance - margin) > DISTANCE_SLACK:
            return distance > margin
    # No sample clear of the margin: the cell is as thin as the slack of a cut, so either side is as exact
    # as the cut itself.
    x, y, den = interior_point(arr, face)
    return region.distance((x / den, y / den)) >= margin


def _required_edges(
    arr: Arrangement, inside: dict[int, bool], ports: tuple[PortClip, ...]
) -> list[RequiredEdge]:
    """Edges that carry a candidate and separate a cell of ``R_m`` from a cell of the support outside it."""
    required = []
    for ei, edge in enumerate(arr.edges):
        left, right = arr.he_face[2 * ei], arr.he_face[2 * ei + 1]
        if left not in inside or right not in inside or inside[left] == inside[right]:
            continue
        candidates = frozenset(c for c in edge.carriers if c.role is CarrierRole.CURVE)
        if not candidates:
            continue  # a port boundary between two cells too thin to tell apart: no offset lies here
        on_port = any(c.role is CarrierRole.PORT_BOUNDARY for c in edge.carriers)
        if not on_port and _in_a_port(edge, ports):
            continue
        required.append(
            RequiredEdge(NodedEdge(edge.a, edge.b, candidates), inside[left])
        )
    return required


def _in_a_port(edge: NodedEdge, ports: tuple[PortClip, ...]) -> bool:
    """Whether the midpoint of an edge off every port boundary lies in the active interior of a port."""
    middle = (edge.a[0] + edge.b[0], edge.a[1] + edge.b[1], 2)
    return any(clip.holds(middle) for clip in ports)


def _check_seam_ends(
    arr: Arrangement, required: list[RequiredEdge], grid_um: float
) -> None:
    """A seam chain ends only where it is clipped: on the support boundary or on a port boundary."""
    degree: dict[Point, int] = {}
    for seam in required:
        for v in (seam.edge.a, seam.edge.b):
            degree[v] = degree.get(v, 0) + 1
    open_ends = {v for v, count in degree.items() if count % 2}
    if not open_ends:
        return
    clipping = {
        v
        for edge in arr.edges
        if any(c.role in CLIP_ROLES for c in edge.carriers)
        for v in (edge.a, edge.b)
    }
    lost = sorted(open_ends - clipping)
    if lost:
        x, y = lost[0]
        raise ValidationError(
            f"a band seam ends at ({x * grid_um:.4f}, {y * grid_um:.4f}) um, on neither the support "
            "boundary nor a port boundary: part of the band boundary was lost"
        )
