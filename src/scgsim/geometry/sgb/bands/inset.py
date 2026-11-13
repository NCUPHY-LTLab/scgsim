"""Surface-EPR inset bands (design brief sections 5 and 6): the first cut-family generator.

Owns generating, from a *support region* ``R`` (rings of exact line and arc primitives), the boundary
curves of ``R_m = R (-) D(m)`` (Minkowski erosion by a disk) for the margins ``m_1 < ... < m_K``, and the
regions themselves.  The bands are ``B_0 = R \\ R_{m_1}``, ``B_k = R_{m_k} \\ R_{m_{k+1}}``, ``B_K = R_{m_K}``;
"at least ``m_k`` from the edge" is the union of the bands ``>= k``.

Construction (no polygonised disk), per margin:

1. **Candidates.**  Every line is offset by ``m`` toward the region interior (a line); every convex arc
   (centre on the interior side) becomes the concentric arc of radius ``rho - m`` (it vanishes to a sharp corner
   when ``m >= rho``); every concave arc (a hole) the concentric arc of radius ``r + m``; at every reflex vertex
   the offset curves leave a gap bridged by an arc of radius ``m`` centred at the vertex; at convex vertices the
   offset lines overlap and cross -- convex corners stay sharp.  At a convex vertex between two lines both
   offsets end at their crossing, one shared lattice point, when it lies inside both (Issue #138): the overlap
   past it is never band boundary, and near a straight junction snapping would fuse it into a seam piece a few
   grid steps long.  A reflex vertex between two lines whose arc would be shorter than
   :data:`SHORT_ARC_GRID_STEPS` gets no arc: both offsets end at their intersection instead, which lies
   ``m (1 / cos(theta / 2) - 1)`` beyond the arc (``theta`` the turn) -- under one grid step, or the arc stays.
   An edge of a few grid steps is exact geometry that the mesher cannot use (Issue #125).
2. **Region.**  The candidates, the support boundary and the port boundaries form a small arrangement; its cells
   lie in ``R_m`` or outside it, one sample point per cell deciding (``bands.inset_regions``).
3. **Required seams.**  Only the candidate edges between a cell of ``R_m`` and a cell outside it become
   carriers, each recording the side band ``k`` lies on (:class:`BandSeam`).  The overlap at a convex corner,
   the part of a cap outside its run and two coincident offsets with the same band on both sides are no seams.

The regions and seams travel with the family (``CurveFamilyRecord.regions``): labelling locates a face in them
and the Imprint check proves that the level arrangement kept every seam (``arrangement.labelling.bands``).  Neither proves
that the candidates contain the whole level set ``dist = m``.  One failure is refused: where two neighbouring
offsets neither meet nor cross, the curve has a gap and one cell lies on both sides of the margin
(``inset_regions``).  Two lines whose offsets are ended at their crossing or intersection always meet; the
refusal remains for the junctions that are not (an arc beside a small turn, or a convex junction whose crossing
lies outside an offset).  A piece the candidates miss without leaving such a gap is not detected.

Distance origins: only the region's gap edges (``SupportRegion.origins``; see ``bands.requests``).  The
offset is built from those primitives alone: a line or arc per origin, a bridging arc at every reflex vertex
between two consecutive origins, and a half circle round each end of a run of origins.  The rest of the
boundary (the die edge, an edge shared with another conductor) only clips the candidates where they cross it,
and a seam may end on it.  The die edge is a dicing line, so a run of origins that ends on it belongs to a
conductor the dicing cut, which continues past it (Issue #138): where the run ends in a line, that line is
continued past the die edge (``SupportRegion.continued_origins``), its offsets run straight on until the die
edge clips them, and no half circle is drawn round that end.  Otherwise the part of the half circle inside the
die would be a short arc, the margin times the angle's deviation from a right angle, at every oblique cut.  A
run that ends on the die edge in an arc never reaches this module: the support is refused when it is resolved.
Contact caps and other bands are never origins -- they are not inputs of this module.  Ports are
never origins either: a port footprint on the support's level is no origin
and moves no band, but it keeps the band seams out of itself (below).  Topology changes (a neck that splits, a
band that vanishes) are ordinary outcomes: the bands that survive are the ones the regions hold.

Ports.  A lumped-port sheet lies on the support's level and must stay one face, so no band seam may enter its
interior.  The footprints of the ports on the level (the authored region whole) join the arrangement as clip
boundaries only: a seam is cut where it crosses a port boundary, no edge in a port's interior is a seam, and a
seam may end on the port boundary like on the free boundary.  The part of a port that lands on a host conductor
is the host's surface, not the port (labelling), so a seam inside the landing stays: a port's *active* interior
is its footprint minus the footprints of its hosts (:class:`PortClip`).  A seam never crosses a host boundary:
it lies in the support region, which is either wholly the host's surface or wholly outside it.  The distance
origins never saw the port, so every band keeps its geometry outside the port.  Only the seams of this family
are clipped: a refinement curve is a user request and is never clipped here.

The offset of an oblique edge is not representable on the grid (its normal is irrational): curve end points
are snapped to the lattice like every other vertex, so the carriers are exact up to the snap bound.  Every
angle and normal is computed in fixed-precision decimals, so the end points -- and the ids built from them --
are identical on every platform.

Each member's lineage records where on the support it comes from and its margin as a ``BandMember``: the
ring, the primitive (or the junction after it), the role (``edge`` for the offset of a primitive, straight or
arc; ``corner`` for a reflex arc; ``capstart`` and ``capend`` for an end cap) and the margin.  Its id and member
name, ``<support>/ring<r>/<role><k>``, are written from those and are never read back.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction

from scgsim.geometry.sgb.arrangement.noding.snap_rounding import (
    Segment,
    crossing_pixel,
    proper_crossing,
)
from scgsim.geometry.sgb.bands.inset_regions import (
    PortClip,
    RequiredEdge,
    margin_cells,
)
from scgsim.geometry.sgb.exact import circle_math
from scgsim.geometry.sgb.exact.band_regions import (
    BandRegions,
    BandSeam,
    MarginRegion,
)
from scgsim.geometry.sgb.exact.carriers import (
    ArcCarrier,
    BandMember,
    BandMemberRole,
    EdgeCarrier,
    Lineage,
    LineCarrier,
    LineEdge,
    RingPrimitive,
)
from scgsim.geometry.sgb.exact.families import CurveFamilyRecord
from scgsim.geometry.sgb.exact.plane import Point
from scgsim.geometry.sgb.exact.support import SupportRegion

Vec = tuple[Decimal, Decimal]
PARALLEL_EPSILON = Decimal(
    "1e-9"
)  # tangents closer to parallel than this are a straight or convex junction
PIECE_SLACK = Decimal("1e-9")
# A reflex arc shorter than this many grid steps (margin times turn) is replaced by the intersection of its two
# offsets, when both are lines and the intersection lies under one grid step beyond the arc.  The value is the
# Human's choice on Issue #125, derived as 0.5 um times a 2.8125 degree turn (one vertex of a 128-point circle).
SHORT_ARC_GRID_STEPS = 25


def inset_family(
    family_id: str,
    region: SupportRegion,
    target_levels: tuple[Fraction, ...],
    ports: tuple[PortClip, ...] = (),
    grid_um: float = 1.0,
) -> CurveFamilyRecord:
    """The ``required`` cut family of the seams between the bands of the region, with its regions ``R_m``.

    ``ports`` are the lumped ports on the region's level: no seam has a piece in their active interior, so a
    port stays one face where only band seams would cut it.  ``grid_um`` only scales the locations in errors.
    """
    lines: list[LineCarrier] = []
    arcs: list[ArcCarrier] = []
    seams: list[BandSeam] = []
    regions: list[MarginRegion] = []
    for band, margin in enumerate(region.margins, start=1):
        found = _margin_seams(family_id, region, band, margin, ports, grid_um)
        lines.extend(found.lines)
        arcs.extend(found.arcs)
        seams.extend(found.seams)
        regions.append(found.region)
    return CurveFamilyRecord(
        family_id,
        "inset_band",
        tuple(lines),
        target_levels,
        "required",
        tuple(arcs),
        region,
        BandRegions(tuple(regions), tuple(seams)),
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


def _offset_point(
    prim: RingPrimitive, margin: int, *, at_end: bool, vertex: Vec | None = None
) -> Point:
    """End point of the offset of ``prim`` by ``margin`` toward the interior, rounded to the lattice.

    ``vertex`` replaces the primitive's own end point: the end of a line continued past the die edge, offset
    along the line's own normal.
    """
    x, y = vertex or _ends(prim)[1 if at_end else 0]
    nx, ny = _left_normal(prim, at_end=at_end)
    return (
        circle_math.round_half_up(x + margin * nx),
        circle_math.round_half_up(y + margin * ny),
    )


# --------------------------------------------------------------------------- candidate curves


class _Candidates:
    """The candidate offset curves of one margin: lines and arcs, each with the id and lineage it will carry."""

    def __init__(self, family_id: str, margin: int) -> None:
        self.family_id = family_id
        self.margin = margin
        self.lines: list[tuple[str, Point, Point]] = []
        self.line_lineages: dict[str, Lineage] = {}
        self.arcs: dict[str, ArcCarrier] = {}

    def member(self, ring: int, index: int, role: BandMemberRole) -> BandMember:
        """The place on the support a candidate of this margin comes from."""
        return BandMember(ring, index, role, self.margin)

    def add_line(self, member: BandMember, p: Point, q: Point) -> None:
        """A straight candidate (nothing when its end points coincide after snapping)."""
        if p != q:
            candidate_id, lineage = self._named(member, None)
            self.lines.append((candidate_id, p, q))
            self.line_lineages[candidate_id] = lineage

    def add_arc(  # noqa: PLR0913 - an arc is its place, centre, radius, sense, two end points and piece
        self,
        member: BandMember,
        centre: tuple[Fraction, Fraction],
        radius: Fraction,
        ccw: bool,
        start: Point,
        end: Point,
        piece: int | None = None,
    ) -> None:
        """An arc candidate, ``piece`` of a cap or corner cut into several (nothing when its end points coincide)."""
        if start == end:
            return
        candidate_id, lineage = self._named(member, piece)
        arc = ArcCarrier(
            candidate_id,
            centre,
            radius,
            ccw,
            (Fraction(start[0]), Fraction(start[1])),
            (Fraction(end[0]), Fraction(end[1])),
            lineage,
        )
        self.arcs[arc.id] = arc

    def _named(self, member: BandMember, piece: int | None) -> tuple[str, Lineage]:
        """A candidate's id ``<support>/ring<r>/<role><k>[.<piece>]@m<margin>`` and its lineage.

        The lineage keeps the member name (the id without the margin) and, as :class:`BandMember`, the place, the
        margin and the piece; the id and the name are written from the member, and nothing reads them back.
        """
        suffix = "" if piece is None else f".{piece}"
        name = f"{self.family_id}/ring{member.ring}/{member.role}{member.index}{suffix}"
        lineage = Lineage(
            "family_member",
            family_id=self.family_id,
            member=name,
            band_member=replace(member, piece=piece),
        )
        return f"{name}@m{self.margin}", lineage


def _generate_candidates(
    family_id: str, region: SupportRegion, margin: int
) -> _Candidates:
    candidates = _Candidates(family_id, margin)
    # the offsets, normals and turns are decimal steps written with operators
    with circle_math.working_precision():
        for ri, ring in enumerate(region.rings):
            count = len(ring)
            joins = _corner_joins(region, ri, margin)
            for pi, prim in enumerate(ring):
                if not region.is_origin(ri, pi):
                    continue
                offset = candidates.member(ri, pi, "edge")
                # The ends the die edge cuts are those of the line continued past it; joins lie at the others.
                first, last = _ends(region.origin(ri, pi))
                cut_first, cut_last = first != _ends(prim)[0], last != _ends(prim)[1]
                start = joins.get((pi - 1) % count) or _offset_point(
                    prim, margin, at_end=False, vertex=first
                )
                end = joins.get(pi) or _offset_point(
                    prim, margin, at_end=True, vertex=last
                )
                if isinstance(prim, LineEdge):
                    candidates.add_line(offset, start, end)
                else:
                    radius = prim.radius - margin if prim.ccw else prim.radius + margin
                    if radius >= 1:
                        candidates.add_arc(
                            offset, prim.centre, radius, prim.ccw, start, end
                        )
                following = (pi + 1) % count
                if not region.is_origin(ri, following):
                    if not cut_last:
                        _cap_arcs(
                            candidates.member(ri, pi, "capend"),
                            prim,
                            margin,
                            candidates,
                            at_end=True,
                        )
                elif (
                    pi not in joins
                ):  # a joined junction has its two offsets meet: no arc
                    _corner_arcs(
                        candidates.member(ri, pi, "corner"),
                        prim,
                        ring[following],
                        margin,
                        candidates,
                    )
                if not region.is_origin(ri, (pi - 1) % count) and not cut_first:
                    _cap_arcs(
                        candidates.member(ri, pi, "capstart"),
                        prim,
                        margin,
                        candidates,
                        at_end=False,
                    )
    return candidates


def _corner_joins(
    region: SupportRegion, ring_index: int, margin: int
) -> dict[int, Point]:
    """Where the offsets of two consecutive origin lines end at their intersection: junction index -> that point.

    Junction ``k`` lies between primitives ``k`` and ``k + 1`` of the ring.  Two kinds qualify:

    * a reflex junction whose arc (``margin`` times the turn) is shorter than :data:`SHORT_ARC_GRID_STEPS`
      and whose intersection lies less than one grid step beyond the arc (Issue #125);
    * a convex junction whose offsets cross inside both of them (:func:`_convex_joins`).
    """
    ring = region.rings[ring_index]
    joins: dict[int, Point] = {}
    convex: dict[int, _ConvexJunction] = {}
    for k, prim in enumerate(ring):
        following = (k + 1) % len(ring)
        if not (
            region.is_origin(ring_index, k) and region.is_origin(ring_index, following)
        ):
            continue
        after = ring[following]
        if not (isinstance(prim, LineEdge) and isinstance(after, LineEdge)):
            continue
        junction = _line_junction(prim, after)
        if junction.cross > PARALLEL_EPSILON:
            convex_junction = _convex_junction(junction, margin)
            if convex_junction is not None:
                convex[k] = convex_junction
            continue
        join = _short_corner_join(junction, margin)
        if join is not None:
            joins[k] = join
    joins.update(_convex_joins(ring, convex, margin))
    return joins


@dataclass(frozen=True)
class _LineJunction:
    """The vertex between two lines, their unit normals and the cross and dot products of their tangents."""

    vertex: Vec
    n0: Vec
    n1: Vec
    cross: Decimal  # sine of the turn: positive at a convex junction, negative at a reflex one
    dot: Decimal  # cosine of the turn

    def offsets_meet(self, margin: int) -> Point:
        """The intersection of the two offset lines, rounded to the lattice.

        It lies on the bisector of the two normals at ``margin / cos(turn / 2)`` from the vertex.
        """
        scale = margin / (1 + self.dot)
        return (
            circle_math.round_half_up(
                self.vertex[0] + scale * (self.n0[0] + self.n1[0])
            ),
            circle_math.round_half_up(
                self.vertex[1] + scale * (self.n0[1] + self.n1[1])
            ),
        )


def _line_junction(prim: LineEdge, following: LineEdge) -> _LineJunction:
    t0 = _unit_tangent(prim, at_end=True)
    t1 = _unit_tangent(following, at_end=False)
    return _LineJunction(
        _ends(prim)[1],
        _left_normal(prim, at_end=True),
        _left_normal(following, at_end=False),
        t0[0] * t1[1] - t0[1] * t1[0],
        t0[0] * t1[0] + t0[1] * t1[1],
    )


def _short_corner_join(junction: _LineJunction, margin: int) -> Point | None:
    """The intersection of the two offsets at a short reflex junction of two lines, or ``None`` to keep the arc."""
    if junction.cross >= -PARALLEL_EPSILON:
        return None  # straight or tangent: there is no arc to replace
    turn = circle_math.atan2(-junction.cross, junction.dot)
    if margin * turn >= SHORT_ARC_GRID_STEPS:
        return None
    cos_half = circle_math.sqrt((1 + junction.dot) / 2)
    if margin / cos_half - margin >= 1:
        return None  # the intersection would lie a grid step or more beyond the arc
    return junction.offsets_meet(margin)


@dataclass(frozen=True)
class _ConvexJunction:
    """A convex junction of two lines at one margin: how far each offset runs past their crossing.

    ``trim`` is the length of offset past the crossing on each of the two offsets, ``margin * tan(turn / 2)``.
    ``reach`` is how far along each line, from the vertex, the feet of those two overlap pieces lie on the other
    line: at most ``margin * max(sin(turn), tan(turn / 2))``.
    """

    junction: _LineJunction
    trim: Decimal
    reach: Decimal


def _convex_junction(junction: _LineJunction, margin: int) -> _ConvexJunction | None:
    """The trim and reach of a convex junction, or ``None`` where the lines fold back (a 180 degree turn)."""
    if 1 + junction.dot <= 0:
        return None
    trim = margin * junction.cross / (1 + junction.dot)
    return _ConvexJunction(junction, trim, max(trim, margin * junction.cross))


def _convex_joins(
    ring: tuple[RingPrimitive, ...] | list[RingPrimitive],
    convex: dict[int, _ConvexJunction],
    margin: int,
) -> dict[int, Point]:
    """The convex junctions whose two offsets end at their crossing, with that crossing on the lattice.

    At a convex junction of two lines the offsets cross ``trim`` before their own ends.  Past the crossing each
    offset runs nearer than ``margin`` to the other line, so that overlap piece is never part of the boundary of
    ``R_m`` whatever else the support holds; the boundary turns from one offset to the other at the crossing.
    Ending both offsets there drops only those two pieces and adds no point: the curve is the same set.

    That holds when the overlap pieces lie within ``margin`` of the other *segment*, not only of its line:
    their feet must fall on it, so each line must be at least ``reach`` long, and its offset must keep a part
    between the crossings at its two ends.  A junction that fails either keeps the offsets as they are, and the
    cells decide as before.  (At a near-straight junction the overlap pieces are under a grid step from
    each other, so snapping fuses them into a seam piece a few grid steps long: Issue #138.)
    """
    trims: dict[int, Decimal] = {}  # primitive index -> trim at its two ends together
    count = len(ring)
    for k, convex_junction in convex.items():
        for pi in (k, (k + 1) % count):
            trims[pi] = trims.get(pi, Decimal(0)) + convex_junction.trim
    joins: dict[int, Point] = {}
    for k, convex_junction in convex.items():
        sides = (k, (k + 1) % count)
        if all(
            _line_length(ring[pi]) > max(trims[pi], convex_junction.reach)
            for pi in sides
        ):
            joins[k] = _crossing_on_lattice(
                ring[k], ring[(k + 1) % count], convex_junction, margin
            )
    return joins


def _crossing_on_lattice(
    prim: RingPrimitive,
    following: RingPrimitive,
    convex_junction: _ConvexJunction,
    margin: int,
) -> Point:
    """The lattice point at which both offsets of a joined convex junction end.

    Where the two offsets, with their end points on the lattice, cross properly, the noding has always put
    their corner at the pixel of that crossing (``snap_rounding.crossing_pixel``); the join keeps that point.
    Where they do not -- near a straight junction they touch or overlap instead -- the exact crossing is rounded.
    """
    crossing = proper_crossing(
        _offset_point(prim, margin, at_end=False),
        _offset_point(prim, margin, at_end=True),
        _offset_point(following, margin, at_end=False),
        _offset_point(following, margin, at_end=True),
    )
    if crossing is not None:
        return crossing_pixel(crossing)
    return convex_junction.junction.offsets_meet(margin)


def _line_length(prim: RingPrimitive) -> Decimal:
    (px, py), (qx, qy) = _ends(prim)
    return circle_math.sqrt((qx - px) ** 2 + (qy - py) ** 2)


def _cap_arcs(
    member: BandMember,
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
            member, centre, Fraction(margin), not at_end, previous, end, piece=k
        )
        previous = end


def _corner_arcs(
    member: BandMember,
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
        candidates.add_arc(
            member,
            (Fraction(centre_point[0]), Fraction(centre_point[1])),
            Fraction(margin),
            False,
            previous,
            end,
            piece=None if pieces == 1 else k,
        )
        previous = end


# --------------------------------------------------------------------------- regions and seams


@dataclass(frozen=True)
class _MarginSeams:
    """The carriers one margin contributes, the seams they are, and its region ``R_m``."""

    lines: list[LineCarrier]
    arcs: list[ArcCarrier]
    seams: list[BandSeam]
    region: MarginRegion


def _margin_seams(  # noqa: PLR0913 - one margin of one support, with its ports and grid
    family_id: str,
    region: SupportRegion,
    band: int,
    margin: int,
    ports: tuple[PortClip, ...],
    grid_um: float,
) -> _MarginSeams:
    """The required seams of ``R_m`` as carriers, each recording the side band ``band`` lies on."""
    candidates = _generate_candidates(family_id, region, margin)
    segments = [
        Segment(p, q, frozenset([EdgeCarrier(cid)])) for cid, p, q in candidates.lines
    ]
    arc_pieces: dict[str, tuple[ArcCarrier, list[Point]]] = {}
    for arc in candidates.arcs.values():
        points = arc.witness_points()
        arc_pieces[arc.id] = (arc, points)
        segments.extend(
            Segment(p, q, frozenset([piece]))
            for piece, p, q in zip(arc.pieces(), points, points[1:], strict=False)
        )
    cells = margin_cells(family_id, segments, region, margin, ports, grid_um)
    lines, line_seams = _lines_of(cells.required, candidates.line_lineages, band)
    arcs, arc_seams = _arcs_of(cells.required, arc_pieces, band)
    return _MarginSeams(
        lines, arcs, line_seams + arc_seams, MarginRegion(margin, cells.region_rings)
    )


def _lines_of(
    required: list[RequiredEdge], lineages: dict[str, Lineage], band: int
) -> tuple[list[LineCarrier], list[BandSeam]]:
    """Straight members: the seams that carry no piece of an arc, each with the side of ``R_m``.

    ``lineages`` is the lineage of every straight candidate, by candidate id; a seam piece keeps its candidate's.
    """
    lines: list[LineCarrier] = []
    seams: list[BandSeam] = []
    pieces_of_line: dict[str, int] = {}
    for seam in required:
        e = seam.edge
        if any(c.piece is not None for c in e.carriers):
            continue
        # Coincident offsets of one margin from several primitives: the carrier is named after one of them
        # and keeps its lineage; which side is R_m was read from the cells, never from that name.
        source = min(e.carriers).id
        n = pieces_of_line.get(source, 0)
        pieces_of_line[source] = n + 1
        carrier = LineCarrier(f"{source}.{n}", e.a, e.b, lineages[source])
        lines.append(carrier)
        seams.append(BandSeam(carrier.id, band, seam.inside_left))
    return lines, seams


def _arcs_of(
    required: list[RequiredEdge],
    arc_pieces: dict[str, tuple[ArcCarrier, list[Point]]],
    band: int,
) -> tuple[list[ArcCarrier], list[BandSeam]]:
    """Arc members: maximal runs of seam chords of each candidate arc with ``R_m`` on one side of the arc.

    Where two arcs meet at a shallow angle, snap rounding can make a few edges common to chords of both.
    Such an edge is kept for one of them, the first by id: kept for both, each run would end at the far end
    of the common edges, so the two sub-arcs would overlap there; drawn again as their own chords in the
    level arrangement they part, and the stretch of one beyond its meeting with the other is left inside
    a single face.
    """
    kept_side: dict[str, dict[int, bool]] = {}
    kept_points: dict[
        tuple[str, int], set[Point]
    ] = {}  # end points of the seam (possibly trimmed) part of each chord
    for seam in required:
        e = seam.edge
        arcs = {c.curve for c in e.carriers if c.piece is not None}
        if not arcs:
            continue
        owner = min(arcs)
        for c in e.carriers:
            if c.piece is None or c.curve != owner:
                continue
            base, chord = c.curve, c.piece
            points = arc_pieces[base][1]
            along = _along((points[chord], points[chord + 1]), e.b) >= _along(
                (points[chord], points[chord + 1]), e.a
            )
            kept_side.setdefault(base, {}).setdefault(
                chord, seam.inside_left if along else not seam.inside_left
            )
            kept_points.setdefault((base, chord), set()).update((e.a, e.b))
    return _arc_runs(kept_side, arc_pieces, kept_points, band)


def _along(chord: tuple[Point, Point], point: Point) -> int:
    """Position of ``point`` along the chord (dot product with its direction)."""
    (px, py), (qx, qy) = chord
    return (point[0] - px) * (qx - px) + (point[1] - py) * (qy - py)


def _arc_runs(
    kept_side: dict[str, dict[int, bool]],
    arc_pieces: dict[str, tuple[ArcCarrier, list[Point]]],
    kept_points: dict[tuple[str, int], set[Point]],
    band: int,
) -> tuple[list[ArcCarrier], list[BandSeam]]:
    """Maximal runs of seam chords of each candidate arc, as sub-arcs through their end points.

    A run is consecutive chords with ``R_m`` on the same side of the arc.  A chord trimmed where another
    carrier crosses it (the crossing lies inside the chord) ends the run at the crossing, not at the chord's
    far end; otherwise the arc would stick out past the crossing into cells of one side.
    """
    out: list[ArcCarrier] = []
    seams: list[BandSeam] = []
    for arc_id in sorted(kept_side):
        arc, points = arc_pieces[arc_id]
        sides = kept_side[arc_id]
        indices = sorted(sides)
        runs, start = [], indices[0]
        for previous, current in zip(indices, indices[1:], strict=False):
            if current != previous + 1 or sides[current] != sides[previous]:
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
            carrier = ArcCarrier(
                f"{arc_id}.{i}",
                arc.centre,
                arc.radius,
                arc.ccw,
                (Fraction(a[0]), Fraction(a[1])),
                (Fraction(b[0]), Fraction(b[1])),
                arc.lineage,
                arc.root_id,
            )
            out.append(carrier)
            seams.append(BandSeam(carrier.id, band, sides[k0]))
    return out, seams
