"""The distance band of a raw face, and the check that Imprint kept every band seam (stage S5 labelling).

The band regions are established before Imprint (``cutfamilies.inset``): per margin ``m_k`` the region
``R_{m_k}`` of the support, and the required seams between band ``k - 1`` and band ``k`` with the side each
band lies on.  Imprint is the level arrangement: the seams join every other curve of the level and are noded
with them.  This module owns the two questions asked of that arrangement afterwards:

* **Label.**  A face of the level lies in a support region when its interior point does (the support boundary
  is an arrangement edge, so no face straddles it).  Its band is the number of regions ``R_{m_k}`` that point
  lies in -- one point location per margin, no sampling of the face.  Only the supports whose bounding box holds
  the point are tested.  A lumped-port face is the exception: band seams stop at the port boundary, so the one
  face of a port may lie across several bands.  It is no band's surface (its area is left out of every band
  group), yet it lies in the support region and the coverage check counts it, so it carries band 0 of every
  family whose support holds it (``nominal``), and it is never compared with a seam.
* **Imprint check.**  Every required seam must be covered, without a gap, by level edges carrying its carrier,
  and the faces on either side of those edges must carry the bands the seam recorded.  A missing piece, a lost
  loop or a face labelled across a seam is a :class:`ValidationError` naming the place in um.  Clipping at the
  free boundary or at a port is no loss: the seams were already clipped when the regions were built, so the
  check asks for exactly the seams that exist.

The check proves that the boundaries the regions established survive Imprint.  It does not prove that the
candidate offsets held the whole iso-distance curve: a part the candidates missed was never a seam.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from scgsim.geometry.sgb.arrangement.dcel import Arrangement
from scgsim.geometry.sgb.errors import ValidationError
from scgsim.geometry.sgb.plane import Point, RepPoint
from scgsim.geometry.sgb.records.band_regions import BandRegions, BandSeam
from scgsim.geometry.sgb.records.families import CurveFamilyRecord
from scgsim.geometry.sgb.records.grid import Grid
from scgsim.geometry.sgb.records.support import SupportRegion

MAX_REPORTED = 5
"""How many problems one refusal lists; the rest are counted."""


@dataclass(frozen=True)
class LevelSupport:
    """A band family targeting a level: its support region, its band regions and the support's bounding box."""

    family_id: str
    support: SupportRegion
    regions: BandRegions
    box: tuple[int, int, int, int]

    def may_hold(self, point: RepPoint) -> bool:
        """Whether the point lies in the bounding box of the support (a cheap test before the exact one)."""
        x, y, den = point
        x0, y0, x1, y1 = self.box
        return x0 * den <= x <= x1 * den and y0 * den <= y <= y1 * den


def level_supports(families: Iterable[CurveFamilyRecord]) -> list[LevelSupport]:
    """The band families among ``families``, ready for labelling."""
    out = []
    for family in families:
        if family.support is None or family.regions is None:
            continue
        points = [p for ring in family.support.witness_rings() for p in ring]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        out.append(
            LevelSupport(
                family.family_id,
                family.support,
                family.regions,
                (min(xs), min(ys), max(xs), max(ys)),
            )
        )
    return out


def face_bands(
    point: RepPoint, supports: Sequence[LevelSupport], *, nominal: bool = False
) -> tuple[tuple[str, int], ...]:
    """``(family id, band)`` of every support holding a face's interior point (0 = closest to the origins).

    ``nominal`` is for a port face, which may straddle bands: band 0 wherever its support holds it.
    """
    return tuple(
        (s.family_id, 0 if nominal else s.regions.band_at(point))
        for s in supports
        if s.may_hold(point) and s.support.contains(point)
    )


# --------------------------------------------------------------------------- the Imprint check


@dataclass(frozen=True)
class _Piece:
    """One noding segment of a seam carrier: a line whole, or one witness chord of an arc."""

    carrier_id: str
    p: Point
    q: Point


@dataclass(frozen=True)
class _LevelView:
    """What the check reads of a level: its arrangement, the edges per carrier and the labels of its faces."""

    arr: Arrangement
    edges_of: dict[str, list[int]]
    bands: Sequence[dict[str, int]]
    port_faces: set[int]
    grid: Grid


def check_seams_kept(  # noqa: PLR0913 - a level, its labels and the families it must keep
    arr: Arrangement,
    bands: Sequence[dict[str, int]],
    port_faces: set[int],
    families: Iterable[CurveFamilyRecord],
    grid: Grid,
    plane: str,
) -> None:
    """Refuse a level arrangement that lost a required seam or labels a face across one.

    ``bands[f]`` is the band of face ``f`` per family; ``port_faces`` carry nominal labels and are not compared.
    """
    families = [f for f in families if f.regions is not None]
    pieces = {
        seam_key: found
        for family in families
        for seam_key, found in _pieces_of(family).items()
    }
    wanted = {piece.carrier_id for found in pieces.values() for piece in found}
    edges_of: dict[str, list[int]] = defaultdict(list)
    for ei, edge in enumerate(arr.edges):
        for carrier in edge.carriers:
            if carrier.id in wanted:
                edges_of[carrier.id].append(ei)
    level = _LevelView(arr, edges_of, bands, port_faces, grid)
    lost: list[str] = []
    mislabelled: list[str] = []
    for family in families:
        for seam_key, seam in _seams_of(family):
            gap, sides = _seam_problems(level, pieces[seam_key], seam, family.family_id)
            lost.extend(gap)
            mislabelled.extend(sides)
    # A lost seam first: it is why the faces beside it look mislabelled.
    problems = lost + mislabelled
    if problems:
        shown = "; ".join(problems[:MAX_REPORTED])
        more = len(problems) - MAX_REPORTED
        raise ValidationError(
            f"Imprint lost band boundaries on {plane}: {shown}"
            + (f" (and {more} more)" if more > 0 else "")
        )


def _seams_of(family: CurveFamilyRecord) -> list[tuple[tuple[str, str], BandSeam]]:
    assert family.regions is not None
    return [((family.family_id, s.carrier_id), s) for s in family.regions.seams]


def _pieces_of(family: CurveFamilyRecord) -> dict[tuple[str, str], list[_Piece]]:
    """Per seam, the segments its carrier entered the level as, in its direction of travel."""
    lines = {line.id: line for line in family.carriers}
    arcs = {arc.id: arc for arc in family.arcs}
    out: dict[tuple[str, str], list[_Piece]] = {}
    for seam_key, seam in _seams_of(family):
        if seam.carrier_id in lines:
            line = lines[seam.carrier_id]
            out[seam_key] = [_Piece(line.id, line.p, line.q)]
        elif seam.carrier_id in arcs:
            arc = arcs[seam.carrier_id]
            points = arc.witness_points()
            out[seam_key] = [
                _Piece(piece.id, p, q)
                for piece, p, q in zip(arc.pieces(), points, points[1:], strict=False)
            ]
        else:
            raise ValidationError(
                f"band seam {seam.carrier_id!r} of {family.family_id!r} is not a carrier of its family"
            )
    return out


def _seam_problems(
    level: _LevelView, pieces: list[_Piece], seam: BandSeam, family_id: str
) -> tuple[list[str], list[str]]:
    """What is wrong with one seam in the level: pieces not covered end to end, or faces in the wrong band."""
    where = f"seam {seam.carrier_id!r} of {family_id!r}"
    missing = [
        piece
        for piece in pieces
        if not _connects(
            level.arr, level.edges_of.get(piece.carrier_id, []), piece.p, piece.q
        )
    ]
    if missing:
        first = missing[0]
        return [
            f"{where}: {len(missing)} of its {len(pieces)} piece(s), the first from "
            f"{_um(first.p, level.grid)} to {_um(first.q, level.grid)}, not covered by "
            "level edges (a cut is missing)"
        ], []
    problems = []
    for piece in pieces:
        problems.extend(_side_problems(level, piece, seam, family_id, where))
    return [], problems


def _side_problems(
    level: _LevelView, piece: _Piece, seam: BandSeam, family_id: str, where: str
) -> list[str]:
    """Faces beside the level edges of a covered seam piece that do not carry the band recorded for their side."""
    problems = []
    dx, dy = piece.q[0] - piece.p[0], piece.q[1] - piece.p[1]
    for ei in level.edges_of.get(piece.carrier_id, []):
        edge = level.arr.edges[ei]
        along = (edge.b[0] - edge.a[0]) * dx + (edge.b[1] - edge.a[1]) * dy
        if along == 0:
            continue  # a snapped piece across the seam's direction: its sides say nothing about the seam
        left, right = level.arr.he_face[2 * ei], level.arr.he_face[2 * ei + 1]
        inside, outside = (
            (left, right) if (along > 0) == seam.inside_left else (right, left)
        )
        for face, expected in ((inside, seam.band), (outside, seam.band - 1)):
            if face in level.port_faces:
                continue
            got = level.bands[face].get(family_id)
            if got != expected:
                problems.append(
                    f"face beside {where} at {_um(edge.a, level.grid)} is in band {got}, "
                    f"expected {expected}"
                )
    return problems


def _connects(arr: Arrangement, edges: list[int], start: Point, end: Point) -> bool:
    """Whether the given level edges hold a path from ``start`` to ``end``."""
    neighbours: dict[Point, list[Point]] = defaultdict(list)
    for ei in edges:
        a, b = arr.edges[ei].a, arr.edges[ei].b
        neighbours[a].append(b)
        neighbours[b].append(a)
    seen, stack = {start}, [start]
    while stack:
        vertex = stack.pop()
        if vertex == end:
            return True
        for other in neighbours[vertex]:
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return False


def _um(point: Point, grid: Grid) -> str:
    return f"({grid.to_um(point[0]):.4f}, {grid.to_um(point[1]):.4f}) um"
