"""Surface-EPR inset bands (design brief sections 5 and 6): the first cut-family generator.

Owns generating, from a *support region* ``R`` (rings of exact line and arc primitives), the boundary
curves of ``R_m = R (-) D(m)`` (Minkowski erosion by a disk) for the margins ``m_1 < ... < m_K``.  The bands
are ``B_0 = R \\ R_{m_1}``, ``B_k = R_{m_k} \\ R_{m_{k+1}}``, ``B_K = R_{m_K}``; "at least ``m_k`` from the
edge" is the union of the bands ``>= k``.

Construction (no polygonised disk):

* every line is offset by ``m`` toward the region interior (a line);
* every convex arc (centre on the interior side) becomes the concentric arc of radius ``rho - m`` (it
  vanishes to a sharp corner when ``m >= rho``); every concave arc (a hole) the concentric arc of radius ``r + m``;
* at every reflex vertex the offset curves leave a gap bridged by an arc of radius ``m`` centred at the vertex;
  at convex vertices the offset lines overlap and simply cross -- convex corners stay sharp;
* these *candidate* curves are noded together by snap rounding, and an edge is kept iff its midpoint is at
  distance ``>= m`` from the region boundary (the ``dist >= m`` cells of the design brief); dangling stubs go.

Distance origins: only the region's gap edges (``SupportRegion.origins``; see ``cutfamilies.requests``).  The
offset is built from those primitives alone: a line or arc per origin, a bridging arc at every reflex vertex
between two consecutive origins, and a half circle round each end of a run of origins.  The rest of the
boundary (the die edge, an edge shared with another conductor) only clips the candidates where they cross it,
and a curve may end on it.  Contact caps, other bands and tile seams are never origins -- they are not
inputs of this module.  Ports are never origins either: a port footprint on the support's level is no origin
and moves no band, but it keeps the band seams out of itself (below).  Topology changes (a neck that splits, a
band that vanishes) are ordinary outcomes: the bands that survive are the ones the labels record.

Ports.  A lumped-port sheet lies on the support's level and must stay one face, so no band seam may enter its
interior.  The footprints of the ports on the level (the authored region whole) join the noding as clip
boundaries only: the candidates are cut where they cross a port boundary, the pieces whose midpoint lies in a
port's interior are dropped, and a curve may end on the port boundary like on the free boundary.  The part of a
port that lands on a host conductor is the host's surface, not the port (labelling), so a piece inside the
landing is kept: a port's *active* interior is its footprint minus the footprints of its hosts (:class:`PortClip`).
A seam never crosses a host boundary: it lies in the support region, which is either wholly the host's surface
or wholly outside it.  Pieces
along the port boundary or outside it are unchanged, and the distance origins never saw the port, so every band
keeps its geometry outside the port.  Only the seams of this family are clipped: a refinement curve is a user
request and is never clipped here.

The offset of an oblique edge is not representable on the grid (its normal is irrational): curve end points
are snapped to the lattice like every other vertex, so the carriers are exact up to the snap bound.  Every
angle and normal is computed in fixed-precision decimals, so the end points -- and the ids built from them --
are identical on every platform.

Members keep their generating primitive in the lineage: ``<support>/ring<r>/edge<k>``, for straight members
and arcs alike; reflex arcs ``<support>/ring<r>/corner<k>``, end caps ``<support>/ring<r>/capstart<k>`` and ``capend<k>``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from scgsim.geometry.sgb.noding.snap_rounding import NodedEdge, Segment, snap_round
from scgsim.geometry.sgb.plane import Point, point_in_ring
from scgsim.geometry.sgb.records import circle_math
from scgsim.geometry.sgb.records.carriers import (
    ArcCarrier,
    Lineage,
    LineCarrier,
    LineEdge,
    RingPrimitive,
)
from scgsim.geometry.sgb.records.families import CurveFamilyRecord
from scgsim.geometry.sgb.records.prisms import Footprint
from scgsim.geometry.sgb.records.support import DISTANCE_SLACK, SupportRegion

Vec = tuple[Decimal, Decimal]
KEEP_TOLERANCE = DISTANCE_SLACK  # a kept edge may lie this much nearer the origins than m (snap drift)
PARALLEL_EPSILON = Decimal(
    "1e-9"
)  # tangents closer to parallel than this are a straight or convex junction
PIECE_SLACK = Decimal("1e-9")
FREE_BOUNDARY = "~free"  # carrier id prefix of the boundary that is not an origin
PORT_BOUNDARY = "~port"  # carrier id prefix of a port footprint boundary (a clip boundary, never an origin)


@dataclass(frozen=True)
class PortClip:
    """A lumped port as the seams see it: its footprint and the footprints of the hosts it lands on."""

    footprint: Footprint
    landings: tuple[Footprint, ...] = ()

    def holds(self, point: tuple[int, int, int]) -> bool:
        """Whether a point off every boundary lies in the port's active interior (off its hosts)."""
        return _inside(self.footprint, point) and not any(
            _inside(host, point) for host in self.landings
        )


def _inside(footprint: Footprint, point: tuple[int, int, int]) -> bool:
    return sum(point_in_ring(ring, point) for ring in footprint.rings()) % 2 == 1


AddArc = Callable[[str, tuple[int, int], Fraction, bool, Point, Point], None]


def inset_family(
    family_id: str,
    region: SupportRegion,
    target_levels: tuple[Fraction, ...],
    ports: tuple[PortClip, ...] = (),
) -> CurveFamilyRecord:
    """The ``required`` cut family holding the boundaries of ``R_m`` for every margin of the region.

    ``ports`` are the lumped ports on the region's level: no curve has a piece in their active interior, so a port stays one face where only band seams would cut it.
    """
    lines: list[LineCarrier] = []
    arcs: list[ArcCarrier] = []
    for margin in region.margins:
        margin_lines, margin_arcs = _boundary_of_eroded(
            family_id, region, margin, ports
        )
        lines.extend(margin_lines)
        arcs.extend(margin_arcs)
    return CurveFamilyRecord(
        family_id,
        "inset_band",
        tuple(lines),
        target_levels,
        "required",
        tuple(arcs),
        region,
    )


# --------------------------------------------------------------------------- decimal geometry of primitives


def _ends(prim: RingPrimitive) -> tuple[Vec, Vec]:
    if isinstance(prim, LineEdge):
        return (Decimal(prim.p[0]), Decimal(prim.p[1])), (
            Decimal(prim.q[0]),
            Decimal(prim.q[1]),
        )
    return (
        (circle_math.to_decimal(prim.start[0]), circle_math.to_decimal(prim.start[1])),
        (circle_math.to_decimal(prim.end[0]), circle_math.to_decimal(prim.end[1])),
    )


def _unit_tangent(prim: RingPrimitive, *, at_end: bool) -> Vec:
    """Unit tangent of a primitive at its end or start, in the direction of travel."""
    if isinstance(prim, LineEdge):
        dx, dy = Decimal(prim.q[0] - prim.p[0]), Decimal(prim.q[1] - prim.p[1])
        length = circle_math.sqrt(dx * dx + dy * dy)
        return dx / length, dy / length
    x, y = _ends(prim)[1 if at_end else 0]
    vx = x - circle_math.to_decimal(prim.centre[0])
    vy = y - circle_math.to_decimal(prim.centre[1])
    length = circle_math.sqrt(vx * vx + vy * vy)
    ux, uy = vx / length, vy / length
    return (-uy, ux) if prim.ccw else (uy, -ux)


def _left_normal(prim: RingPrimitive, *, at_end: bool) -> Vec:
    tx, ty = _unit_tangent(prim, at_end=at_end)
    return -ty, tx


def _offset_point(prim: RingPrimitive, margin: int, *, at_end: bool) -> Point:
    """End point of the offset of ``prim`` by ``margin`` toward the interior, rounded to the lattice."""
    x, y = _ends(prim)[1 if at_end else 0]
    nx, ny = _left_normal(prim, at_end=at_end)
    return (
        circle_math.round_half_up(x + margin * nx),
        circle_math.round_half_up(y + margin * ny),
    )


# --------------------------------------------------------------------------- candidate curves


class _Candidates:
    """The candidate offset curves of one margin: lines and arcs, each with the id it will carry."""

    def __init__(self, family_id: str, margin: int) -> None:
        self.family_id = family_id
        self.margin = margin
        # A member's id is ``<support>/ring<r>/edge<k>@m<margin>``: the generating primitive plus the
        # margin.  Its lineage keeps the primitive alone, for straight members and arcs alike.
        self.tag = f"@m{margin}"
        self.lines: list[tuple[str, Point, Point]] = []
        self.arcs: dict[str, ArcCarrier] = {}

    def add_line(self, name: str, p: Point, q: Point) -> None:
        """A straight candidate (nothing when its end points coincide after snapping)."""
        if p != q:
            self.lines.append((f"{self.family_id}/{name}{self.tag}", p, q))

    def add_arc(  # noqa: PLR0913 - an arc is its centre, radius, sense and two end points
        self,
        name: str,
        centre: tuple[Fraction, Fraction],
        radius: Fraction,
        ccw: bool,
        start: Point,
        end: Point,
    ) -> None:
        """An arc candidate (nothing when its end points coincide after snapping)."""
        if start == end:
            return
        arc = ArcCarrier(
            f"{self.family_id}/{name}{self.tag}",
            centre,
            radius,
            ccw,
            (Fraction(start[0]), Fraction(start[1])),
            (Fraction(end[0]), Fraction(end[1])),
            Lineage(
                "family_member",
                family_id=self.family_id,
                member=f"{self.family_id}/{name}",
            ),
        )
        self.arcs[arc.id] = arc


def _generate_candidates(
    family_id: str, region: SupportRegion, margin: int
) -> _Candidates:
    candidates = _Candidates(family_id, margin)
    for ri, ring in enumerate(region.rings):
        count = len(ring)
        for pi, prim in enumerate(ring):
            if not region.is_origin(ri, pi):
                continue
            name = f"ring{ri}/edge{pi}"
            start = _offset_point(prim, margin, at_end=False)
            end = _offset_point(prim, margin, at_end=True)
            if isinstance(prim, LineEdge):
                candidates.add_line(name, start, end)
            else:
                radius = prim.radius - margin if prim.ccw else prim.radius + margin
                if radius >= 1:
                    candidates.add_arc(name, prim.centre, radius, prim.ccw, start, end)
            following = (pi + 1) % count
            if region.is_origin(ri, following):
                _corner_arcs(
                    f"ring{ri}/corner{pi}", prim, ring[following], margin, candidates
                )
            else:
                _cap_arcs(f"ring{ri}/capend{pi}", prim, margin, candidates, at_end=True)
            if not region.is_origin(ri, (pi - 1) % count):
                _cap_arcs(
                    f"ring{ri}/capstart{pi}", prim, margin, candidates, at_end=False
                )
    return candidates


def _cap_arcs(
    name: str,
    prim: RingPrimitive,
    margin: int,
    candidates: _Candidates,
    *,
    at_end: bool,
) -> None:
    """Half circle of radius ``margin`` about an end of a run of origins, round the end away from the run.

    The offset of the run stops at the end point; the points at distance ``margin`` from it close the curve
    round that end.  The half circle starts at the offset end point on the interior side and turns the
    other way at the run's start than at its end.  The part of it outside the region, or nearer to another
    origin, is dropped when the candidates are selected.
    """
    vertex = _ends(prim)[1 if at_end else 0]
    centre_point = (
        circle_math.round_half_up(vertex[0]),
        circle_math.round_half_up(vertex[1]),
    )
    nx, ny = _left_normal(prim, at_end=at_end)
    rotation = circle_math.Rotation(circle_math.HALF_PI, ccw=not at_end)
    centre = (Fraction(centre_point[0]), Fraction(centre_point[1]))
    vector: Vec = (margin * nx, margin * ny)
    previous = _offset_point(prim, margin, at_end=at_end)
    for k in range(2):
        vector = rotation.apply(*vector)
        end = (
            circle_math.round_half_up(vertex[0] + vector[0]),
            circle_math.round_half_up(vertex[1] + vector[1]),
        )
        candidates.add_arc(
            f"{name}.{k}", centre, Fraction(margin), not at_end, previous, end
        )
        previous = end


def _corner_arcs(
    name: str,
    prim: RingPrimitive,
    following: RingPrimitive,
    margin: int,
    candidates: _Candidates,
) -> None:
    """Arc of radius ``margin`` about a reflex vertex joining the offsets of two consecutive primitives."""
    t0 = _unit_tangent(prim, at_end=True)
    t1 = _unit_tangent(following, at_end=False)
    cross = t0[0] * t1[1] - t0[1] * t1[0]
    dot = t0[0] * t1[0] + t0[1] * t1[1]
    if cross >= -PARALLEL_EPSILON:
        return  # straight or tangent junction, or a convex corner: the offsets meet or overlap
    turn = circle_math.atan2(-cross, dot)  # right turn angle in (0, pi)
    pieces = max(1, circle_math.ceil_div(turn - PIECE_SLACK, circle_math.HALF_PI))
    vertex = _ends(prim)[1]
    centre_point = (
        circle_math.round_half_up(vertex[0]),
        circle_math.round_half_up(vertex[1]),
    )
    nx, ny = _left_normal(prim, at_end=True)
    rotation = circle_math.Rotation(turn / pieces, ccw=False)
    vector: Vec = (margin * nx, margin * ny)
    first_offset = _offset_point(prim, margin, at_end=True)
    last_offset = _offset_point(following, margin, at_end=False)
    previous = first_offset
    for k in range(pieces):
        vector = rotation.apply(*vector)
        end = (
            circle_math.round_half_up(vertex[0] + vector[0]),
            circle_math.round_half_up(vertex[1] + vector[1]),
        )
        if k == pieces - 1:
            end = last_offset
        piece_name = name if pieces == 1 else f"{name}.{k}"
        candidates.add_arc(
            piece_name,
            (Fraction(centre_point[0]), Fraction(centre_point[1])),
            Fraction(margin),
            False,
            previous,
            end,
        )
        previous = end


# --------------------------------------------------------------------------- noding and selection


def _boundary_of_eroded(
    family_id: str, region: SupportRegion, margin: int, ports: tuple[PortClip, ...]
) -> tuple[list[LineCarrier], list[ArcCarrier]]:
    candidates = _generate_candidates(family_id, region, margin)
    segments = [Segment(p, q, frozenset([cid])) for cid, p, q in candidates.lines]
    arc_pieces: dict[str, tuple[ArcCarrier, list[Point]]] = {}
    for arc in candidates.arcs.values():
        points = arc.witness_points()
        arc_pieces[arc.id] = (arc, points)
        segments.extend(
            Segment(p, q, frozenset([f"{arc.id}#{k}"]))
            for k, (p, q) in enumerate(zip(points, points[1:], strict=False))
        )
    segments.extend(_free_boundary_segments(region))
    segments.extend(_port_boundary_segments(ports))
    noded = [
        e
        for e in snap_round(segments).edges
        if not any(c.startswith(FREE_BOUNDARY) for c in e.carriers)
    ]
    clipped, port_ends = _clipped_to_ports(noded, ports)
    kept = _kept_edges(clipped, region, margin, port_ends)
    return _lines_of(family_id, kept), _arcs_of(kept, arc_pieces)


def _free_boundary_segments(region: SupportRegion) -> list[Segment]:
    """The boundary that is not an origin, as noding input only: it splits the candidates where they cross it."""
    segments = []
    for k, prim in enumerate(region.free_boundary):
        if isinstance(prim, LineEdge):
            segments.append(Segment(prim.p, prim.q, frozenset([f"{FREE_BOUNDARY}{k}"])))
        else:
            points = prim.witness_points()
            segments.extend(
                Segment(p, q, frozenset([f"{FREE_BOUNDARY}{k}#{j}"]))
                for j, (p, q) in enumerate(zip(points, points[1:], strict=False))
            )
    return segments


def _port_boundary_segments(ports: tuple[PortClip, ...]) -> list[Segment]:
    """The port footprint boundaries, as noding input only: they split the candidates where they cross them."""
    return [
        Segment(p, q, frozenset([f"{PORT_BOUNDARY}{k}.{r}.{j}"]))
        for k, clip in enumerate(ports)
        for r, ring in enumerate(clip.footprint.rings())
        for j, (p, q) in enumerate(zip(ring, (*ring[1:], ring[0]), strict=True))
    ]


def _clipped_to_ports(
    edges: list[NodedEdge], ports: tuple[PortClip, ...]
) -> tuple[list[NodedEdge], set[Point]]:
    """The edges without the pieces inside a port, and the vertices on a port boundary.

    The noding split every candidate at the port boundaries, so a noded edge lies on a boundary (it
    carries its carrier id) or wholly inside or outside every port, and its midpoint decides.  Edges
    that lie on a port boundary only are the clip boundary itself and are dropped; an edge a candidate shares
    with a port boundary stays, as a seam along the port is no cut of its interior.
    """
    if not ports:
        return edges, set()
    clipped: list[NodedEdge] = []
    port_ends: set[Point] = set()
    for e in edges:
        on_boundary = any(c.startswith(PORT_BOUNDARY) for c in e.carriers)
        if on_boundary:
            port_ends.update((e.a, e.b))
        elif _in_a_port(e, ports):
            continue
        carriers = frozenset(c for c in e.carriers if not c.startswith(PORT_BOUNDARY))
        if carriers:
            clipped.append(NodedEdge(e.a, e.b, carriers))
    return clipped, port_ends


def _in_a_port(edge: NodedEdge, ports: tuple[PortClip, ...]) -> bool:
    """Whether the midpoint of an edge off every port boundary lies in the active interior of a port."""
    middle = (edge.a[0] + edge.b[0], edge.a[1] + edge.b[1], 2)
    return any(clip.holds(middle) for clip in ports)


def _kept_edges(
    edges: Iterable[NodedEdge],
    region: SupportRegion,
    margin: int,
    port_ends: set[Point],
) -> list[NodedEdge]:
    """Edges whose midpoint is at least ``margin`` from the distance origins, with dangling stubs removed.

    A curve may end on the part of the boundary that is not an origin (a band reaching the die edge) or on
    a port boundary (a seam stopping at a port), so an end point there is not dangling.
    """
    kept = []
    for e in edges:
        middle = (e.a[0] + e.b[0], e.a[1] + e.b[1], 2)
        if (
            region.contains(middle)
            and region.distance((middle[0] / 2, middle[1] / 2))
            >= margin - KEEP_TOLERANCE
        ):
            kept.append(e)
    while True:
        degree: dict[Point, int] = {}
        for e in kept:
            degree[e.a] = degree.get(e.a, 0) + 1
            degree[e.b] = degree.get(e.b, 0) + 1
        pruned = [
            e
            for e in kept
            if all(
                degree[v] > 1 or v in port_ends or _on_free_boundary(region, v)
                for v in (e.a, e.b)
            )
        ]
        if len(pruned) == len(kept):
            return kept
        kept = pruned


def _on_free_boundary(region: SupportRegion, vertex: Point) -> bool:
    """Whether a curve end lies on the boundary that is not an origin (up to the snap drift)."""
    return (
        region.distance_to_free_boundary((float(vertex[0]), float(vertex[1])))
        <= KEEP_TOLERANCE
    )


def _lines_of(family_id: str, kept: list[NodedEdge]) -> list[LineCarrier]:
    """Straight members: the kept edges that carry no arc witness chord."""
    lines: list[LineCarrier] = []
    pieces_of_line: dict[str, int] = {}
    for e in kept:
        if any("#" in c for c in e.carriers):
            continue
        source = sorted(e.carriers)[
            0
        ]  # the offset line (one primitive of the support) this piece lies on
        n = pieces_of_line.get(source, 0)
        pieces_of_line[source] = n + 1
        member = source.split("@", 1)[0]
        lines.append(
            LineCarrier(
                f"{source}.{n}",
                e.a,
                e.b,
                Lineage("family_member", family_id=family_id, member=member),
            )
        )
    return lines


def _arcs_of(
    kept: list[NodedEdge], arc_pieces: dict[str, tuple[ArcCarrier, list[Point]]]
) -> list[ArcCarrier]:
    kept_pieces: dict[str, set[int]] = {}
    kept_points: dict[
        tuple[str, int], set[Point]
    ] = {}  # end points of the kept (possibly trimmed) part of each chord
    for e in kept:
        for c in e.carriers:
            base, _, k = c.partition("#")
            if k:
                kept_pieces.setdefault(base, set()).add(int(k))
                kept_points.setdefault((base, int(k)), set()).update((e.a, e.b))
    return _arc_runs(kept_pieces, arc_pieces, kept_points)


def _along(chord: tuple[Point, Point], point: Point) -> int:
    """Position of ``point`` along the chord (dot product with its direction)."""
    (px, py), (qx, qy) = chord
    return (point[0] - px) * (qx - px) + (point[1] - py) * (qy - py)


def _arc_runs(
    kept_pieces: dict[str, set[int]],
    arc_pieces: dict[str, tuple[ArcCarrier, list[Point]]],
    kept_points: dict[tuple[str, int], set[Point]],
) -> list[ArcCarrier]:
    """Maximal runs of kept witness chords of each candidate arc, as sub-arcs through their end points.

    A chord trimmed where another carrier crosses it (the crossing lies inside the chord) ends the run at the
    crossing, not at the chord's far end; otherwise the arc would stick out past the crossing as a dangling stub.
    """
    out: list[ArcCarrier] = []
    for arc_id in sorted(kept_pieces):
        arc, points = arc_pieces[arc_id]
        indices = sorted(kept_pieces[arc_id])
        runs, start = [], indices[0]
        for previous, current in zip(indices, indices[1:], strict=False):
            if current != previous + 1:
                runs.append((start, previous))
                start = current
        runs.append((start, indices[-1]))
        for i, (k0, k1) in enumerate(runs):
            a, b = points[k0], points[k1 + 1]
            first, last = kept_points.get((arc_id, k0)), kept_points.get((arc_id, k1))
            if first:
                a = min(
                    first, key=lambda pt, k=k0: _along((points[k], points[k + 1]), pt)
                )
            if last:
                b = max(
                    last, key=lambda pt, k=k1: _along((points[k], points[k + 1]), pt)
                )
            if a == b:
                continue
            out.append(
                ArcCarrier(
                    f"{arc_id}.{i}",
                    arc.centre,
                    arc.radius,
                    arc.ccw,
                    (Fraction(a[0]), Fraction(a[1])),
                    (Fraction(b[0]), Fraction(b[1])),
                    arc.lineage,
                    arc.root_id,
                )
            )
    return out
