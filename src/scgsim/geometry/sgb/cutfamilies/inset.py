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
   offset lines overlap and simply cross -- convex corners stay sharp.
2. **Region.**  The candidates, the support boundary and the port boundaries form a small arrangement; its cells
   lie in ``R_m`` or outside it, one sample point per cell deciding (``cutfamilies.inset_regions``).
3. **Required seams.**  Only the candidate edges between a cell of ``R_m`` and a cell outside it become
   carriers, each recording the side band ``k`` lies on (:class:`BandSeam`).  The overlap at a convex corner,
   the part of a cap outside its run and two coincident offsets with the same band on both sides are no seams.

The regions and seams travel with the family (``CurveFamilyRecord.regions``): labelling locates a face in them
and the Imprint check proves that the level arrangement kept every seam (``labelling.bands``).  Neither proves
that the candidates contain the whole level set ``dist = m``.  One failure is refused: where two neighbouring
offsets neither meet nor cross (an outline whose vertices turn by small angles, such as a polygon circle with
more than 128 points per full circle), the curve has a gap and one cell lies on both sides of the margin
(``inset_regions``).  A piece the candidates miss without leaving such a gap is not detected.

Distance origins: only the region's gap edges (``SupportRegion.origins``; see ``cutfamilies.requests``).  The
offset is built from those primitives alone: a line or arc per origin, a bridging arc at every reflex vertex
between two consecutive origins, and a half circle round each end of a run of origins.  The rest of the
boundary (the die edge, an edge shared with another conductor) only clips the candidates where they cross it,
and a seam may end on it.  Contact caps, other bands and tile seams are never origins -- they are not
inputs of this module.  Ports are never origins either: a port footprint on the support's level is no origin
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

Members keep their generating primitive in the lineage: ``<support>/ring<r>/edge<k>``, for straight members
and arcs alike; reflex arcs ``<support>/ring<r>/corner<k>``, end caps ``<support>/ring<r>/capstart<k>`` and ``capend<k>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from scgsim.geometry.sgb.cutfamilies.inset_regions import (
    PortClip,
    RequiredEdge,
    margin_cells,
)
from scgsim.geometry.sgb.noding.snap_rounding import Segment
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.records import circle_math
from scgsim.geometry.sgb.records.band_regions import (
    BandRegions,
    BandSeam,
    MarginRegion,
)
from scgsim.geometry.sgb.records.carriers import (
    ArcCarrier,
    Lineage,
    LineCarrier,
    LineEdge,
    RingPrimitive,
)
from scgsim.geometry.sgb.records.families import CurveFamilyRecord
from scgsim.geometry.sgb.records.support import SupportRegion

Vec = tuple[Decimal, Decimal]
PARALLEL_EPSILON = Decimal(
    "1e-9"
)  # tangents closer to parallel than this are a straight or convex junction
PIECE_SLACK = Decimal("1e-9")


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
    # the offsets, normals and turns are decimal steps written with operators
    with circle_math.working_precision():
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
                        candidates.add_arc(
                            name, prim.centre, radius, prim.ccw, start, end
                        )
                following = (pi + 1) % count
                if region.is_origin(ri, following):
                    _corner_arcs(
                        f"ring{ri}/corner{pi}",
                        prim,
                        ring[following],
                        margin,
                        candidates,
                    )
                else:
                    _cap_arcs(
                        f"ring{ri}/capend{pi}", prim, margin, candidates, at_end=True
                    )
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
    segments = [Segment(p, q, frozenset([cid])) for cid, p, q in candidates.lines]
    arc_pieces: dict[str, tuple[ArcCarrier, list[Point]]] = {}
    for arc in candidates.arcs.values():
        points = arc.witness_points()
        arc_pieces[arc.id] = (arc, points)
        segments.extend(
            Segment(p, q, frozenset([f"{arc.id}#{k}"]))
            for k, (p, q) in enumerate(zip(points, points[1:], strict=False))
        )
    cells = margin_cells(family_id, segments, region, margin, ports, grid_um)
    lines, line_seams = _lines_of(family_id, cells.required, band)
    arcs, arc_seams = _arcs_of(cells.required, arc_pieces, band)
    return _MarginSeams(
        lines, arcs, line_seams + arc_seams, MarginRegion(margin, cells.region_rings)
    )


def _lines_of(
    family_id: str, required: list[RequiredEdge], band: int
) -> tuple[list[LineCarrier], list[BandSeam]]:
    """Straight members: the seams that carry no arc witness chord, each with the side of ``R_m``."""
    lines: list[LineCarrier] = []
    seams: list[BandSeam] = []
    pieces_of_line: dict[str, int] = {}
    for seam in required:
        e = seam.edge
        if any("#" in c for c in e.carriers):
            continue
        # Coincident offsets of one margin from several primitives: the carrier is named after one of them
        # for its lineage; which side is R_m was read from the cells, never from that name.
        source = sorted(e.carriers)[0]
        n = pieces_of_line.get(source, 0)
        pieces_of_line[source] = n + 1
        member = source.split("@", 1)[0]
        carrier = LineCarrier(
            f"{source}.{n}",
            e.a,
            e.b,
            Lineage("family_member", family_id=family_id, member=member),
        )
        lines.append(carrier)
        seams.append(BandSeam(carrier.id, band, seam.inside_left))
    return lines, seams


def _arcs_of(
    required: list[RequiredEdge],
    arc_pieces: dict[str, tuple[ArcCarrier, list[Point]]],
    band: int,
) -> tuple[list[ArcCarrier], list[BandSeam]]:
    """Arc members: maximal runs of seam chords of each candidate arc with ``R_m`` on one side of the arc."""
    kept_side: dict[str, dict[int, bool]] = {}
    kept_points: dict[
        tuple[str, int], set[Point]
    ] = {}  # end points of the seam (possibly trimmed) part of each chord
    for seam in required:
        e = seam.edge
        for c in e.carriers:
            base, _, k = c.partition("#")
            if not k:
                continue
            chord = int(k)
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
