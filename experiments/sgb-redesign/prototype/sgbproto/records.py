"""Immutable source records (brief sections 2, 3, 5).

Owns the vocabulary every later stage shares: the grid, line carriers with
lineage, prisms, curve families and the label types that cells receive.
A prism is a planar footprint extruded over a Z interval; ``z0 == z1`` is a
sheet. All XY coordinates are integers in grid units; Z stays exact
(``Fraction``) because the XY grid does not snap Z.

Does not own any derivation; it only validates and normalises its inputs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

from sgbproto.geometry import Point, Ring, ring_twice_area

MaterialKind = Literal["conductor", "dielectric", "vacuum"]
EdgePolicy = Literal["required", "refinement_only"]  # tile seams are not a family: no seam geometry exists


def z_value(value: float | int | str | Fraction) -> Fraction:
    """Exact Z from a user number (decimal text, not binary float)."""
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


@dataclass(frozen=True)
class Grid:
    """The model grid ``g`` in micrometres (default 1 nm); declared once per model."""

    g: float = 0.001

    def snap(self, um: float) -> int:
        """Nearest grid coordinate of a length in micrometres (half rounds up)."""
        return math.floor(um / self.g + 0.5)

    def to_um(self, units: int | Fraction) -> float:
        return float(units) * self.g


@dataclass(frozen=True)
class Lineage:
    """Where a carrier came from: a source polygon edge or a curve-family member."""

    kind: str  # "polygon_edge" | "family_member" | "derived"
    entity_id: str = ""
    polygon: int = 0
    ring: int = 0
    edge: int = 0
    family_id: str = ""
    member: str = ""

    def describe(self) -> str:
        if self.kind == "polygon_edge":
            return f"{self.entity_id}/polygon{self.polygon}/ring{self.ring}/edge{self.edge}"
        return f"{self.family_id}/{self.member}"


@dataclass(frozen=True)
class LineCarrier:
    """A straight curve with exact integer endpoints and a lineage record."""

    id: str
    p: Point
    q: Point
    lineage: Lineage


MEMBER_ONLY = "~m:"  # carrier ids with this prefix exist only to cut cells for area-weighted membership; never required
PIECE_SEPARATOR = "#"  # carrier id of the k-th witness chord of an arc carrier: ``<arc id>#<k>``


def base_carrier_id(carrier_id: str) -> str:
    """Id of the carrier a (possibly witness-piece) carrier id belongs to."""
    return carrier_id.split(PIECE_SEPARATOR, 1)[0]


@dataclass(frozen=True)
class ArcCarrier:
    """A circular arc (< 360 degrees) with exact centre and radius in grid units.

    ``start`` and ``end`` are grid-unit points on the circle (exact for source arcs; snapped grid points for
    derived arcs). ``ccw`` is the direction of travel from ``start`` to ``end``. A full circle is four quarter
    arcs (see :func:`circle_arcs`) so no arc edge is ever a loop or a pair of parallel edges.

    ``root`` is the id of the source carrier this arc derives from (drift is measured against its exact circle).
    """

    id: str
    centre: tuple[Fraction, Fraction]
    radius: Fraction
    ccw: bool
    start: tuple[Fraction, Fraction]
    end: tuple[Fraction, Fraction]
    lineage: Lineage
    root: str = ""

    @property
    def root_id(self) -> str:
        return self.root or self.id

    def sweep(self) -> tuple[float, float]:
        """(start angle, signed sweep) in radians; the sweep is positive when ``ccw``."""
        cx, cy = float(self.centre[0]), float(self.centre[1])
        t0 = math.atan2(float(self.start[1]) - cy, float(self.start[0]) - cx)
        t1 = math.atan2(float(self.end[1]) - cy, float(self.end[0]) - cx)
        if self.ccw:
            return t0, (t1 - t0) % (2 * math.pi)
        return t0, -((t0 - t1) % (2 * math.pi))

    def witness_points(self) -> list[Point]:
        """Grid points ``start .. end`` of the polyline witness: chords with sagitta <= g/4 (g = 1 unit)."""
        r = float(self.radius)
        t0, sweep = self.sweep()
        max_step = 2 * math.acos(1 - 0.25 / r) if r > 0.25 else math.pi
        n = max(1, math.ceil(abs(sweep) / max_step))
        cx, cy = float(self.centre[0]), float(self.centre[1])
        pts: list[Point] = [_round_point(self.start)]
        for k in range(1, n):
            t = t0 + sweep * k / n
            p = (math.floor(cx + r * math.cos(t) + 0.5), math.floor(cy + r * math.sin(t) + 0.5))
            if p != pts[-1]:
                pts.append(p)
        end = _round_point(self.end)
        if end != pts[-1]:
            pts.append(end)
        return pts

    def piece_ids(self) -> list[str]:
        return [f"{self.id}{PIECE_SEPARATOR}{k}" for k in range(len(self.witness_points()) - 1)]


def _round_point(p: tuple[Fraction, Fraction]) -> Point:
    return (math.floor(p[0] + Fraction(1, 2)), math.floor(p[1] + Fraction(1, 2)))


def circle_arcs(
    cid: str, centre: tuple[int, int], radius: int, ccw: bool, lineage: Lineage, root: str = ""
) -> list[ArcCarrier]:
    """A full circle as four quarter arcs starting at angle 0 (exact integer end points)."""
    cx, cy, r = Fraction(centre[0]), Fraction(centre[1]), Fraction(radius)
    quarter = [(cx + r, cy), (cx, cy + r), (cx - r, cy), (cx, cy - r)]
    order = list(range(4)) if ccw else [0, 3, 2, 1]
    arcs = []
    for i in range(4):
        a, b = quarter[order[i]], quarter[order[(i + 1) % 4]]
        arcs.append(ArcCarrier(f"{cid}/q{i}", (cx, cy), r, ccw, a, b, lineage, root or cid))
    return arcs


@dataclass(frozen=True)
class LineEdge:
    """A straight source edge of a ring that also contains arcs."""

    p: Point
    q: Point


@dataclass(frozen=True)
class Footprint:
    """Outer ring (CCW) plus hole rings (CW), integer grid coordinates."""

    outer: Ring
    holes: tuple[Ring, ...] = ()
    # (ring index, first ring-edge index, arc id): the following ``n`` ring edges are the witness chords of that arc
    arc_runs: tuple[tuple[int, int, str], ...] = ()

    @staticmethod
    def from_primitives(
        outer: Sequence[LineEdge | ArcCarrier], holes: Iterable[Sequence[LineEdge | ArcCarrier]] = ()
    ) -> tuple[Footprint, tuple[ArcCarrier, ...]]:
        """A footprint whose rings mix straight edges and arcs (outer CCW, holes CW, already oriented).

        Rings are stored as their polyline witness (arcs as chords); ``arc_runs`` remembers which ring edges
        are witness pieces of which arc so the carriers keep the exact arc definition.
        """
        rings, runs, arcs = [], [], []
        for ring_index, prims in enumerate([outer, *holes]):
            pts: list[Point] = []
            for prim in prims:
                if isinstance(prim, LineEdge):
                    pts.append(prim.p)
                else:
                    witness = prim.witness_points()
                    runs.append((ring_index, len(pts), prim.id))
                    arcs.append(prim)
                    pts.extend(witness[:-1])
            if (ring_twice_area(pts) > 0) != (ring_index == 0):
                raise ValueError("ring orientation: outer must be CCW and holes CW")
            rings.append(tuple(pts))
        return Footprint(rings[0], tuple(rings[1:]), tuple(runs)), tuple(arcs)

    @staticmethod
    def from_rings(outer: Sequence[Point], holes: Iterable[Sequence[Point]] = ()) -> Footprint:
        return Footprint(_oriented(outer, ccw=True), tuple(_oriented(h, ccw=False) for h in holes))

    def rings(self) -> tuple[Ring, ...]:
        return (self.outer, *self.holes)

    def twice_area(self) -> int:
        return ring_twice_area(self.outer) + sum(ring_twice_area(h) for h in self.holes)


def _oriented(ring: Sequence[Point], *, ccw: bool) -> Ring:
    pts = tuple((int(x), int(y)) for x, y in ring)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError("a ring needs at least three distinct points")
    if (ring_twice_area(pts) > 0) != ccw:
        pts = tuple(reversed(pts))
    return pts


@dataclass(frozen=True)
class PrismRecord:
    """A footprint extruded over [z0, z1]; ``z0 == z1`` makes it a sheet."""

    entity_id: str
    footprint: Footprint
    z0: Fraction
    z1: Fraction
    material: str
    material_kind: MaterialKind
    net_id: str | None = None
    role: str = "body"  # "body" | "sheet" | "port"
    priority: int = 0
    polygon: int = 0
    arcs: tuple[ArcCarrier, ...] = ()  # arc carriers referenced by ``footprint.arc_runs``

    def __post_init__(self) -> None:
        if self.z1 < self.z0:
            raise ValueError(f"{self.entity_id}: z1 < z0")
        if self.material_kind == "conductor" and not self.net_id:
            raise ValueError(f"{self.entity_id}: conductor prism needs a net_id")
        if self.is_sheet and self.role == "body":
            object.__setattr__(self, "role", "sheet")

    @property
    def is_sheet(self) -> bool:
        return self.z0 == self.z1

    def carriers(self) -> list[LineCarrier]:
        """Boundary edges of every ring as lineage-carrying line carriers."""
        piece_of: dict[tuple[int, int], str] = {}
        by_id = {a.id: a for a in self.arcs}
        for ring_index, first, arc_id in self.footprint.arc_runs:
            for k, piece in enumerate(by_id[arc_id].piece_ids()):
                piece_of[(ring_index, first + k)] = piece
        out: list[LineCarrier] = []
        for ring_index, ring in enumerate(self.footprint.rings()):
            n = len(ring)
            for edge_index in range(n):
                lineage = Lineage("polygon_edge", self.entity_id, self.polygon, ring_index, edge_index)
                cid = piece_of.get((ring_index, edge_index)) or lineage.describe()
                out.append(LineCarrier(cid, ring[edge_index], ring[(edge_index + 1) % n], lineage))
        return out


@dataclass(frozen=True, order=True)
class Domain:
    """Occupying domain of a slab cell (what a volume is made of)."""

    kind: str  # "conductor" | "dielectric" | "vacuum"
    material: str
    net: str = ""

    @property
    def name(self) -> str:
        return self.net if self.kind == "conductor" else self.material


@dataclass(frozen=True, order=True)
class SheetOwner:
    """Owner of a zero-thickness sheet face: a conductor Net or a port."""

    kind: str  # "conductor" | "port"
    name: str


@dataclass(frozen=True)
class LevelLabel:
    """Label of a level-arrangement cell (brief section 2)."""

    below: Domain | None
    above: Domain | None
    sheet: SheetOwner | None = None
    bands: tuple = ()

    def needs_face(self) -> bool:
        return self.below != self.above or self.sheet is not None


@dataclass(frozen=True)
class CurveFamilyRecord:
    """A named set of curves added to the arrangement with a semantic role (brief section 5)."""

    family_id: str
    role: str
    carriers: tuple[LineCarrier, ...]
    target_levels: tuple[Fraction, ...]
    edge_policy: EdgePolicy = "required"
    arcs: tuple[ArcCarrier, ...] = ()  # arc members (noded through their polyline witness)
    support: object | None = None  # a ``inset.SupportRegion`` when the family is a set of inset bands

    def __post_init__(self) -> None:
        if self.edge_policy not in ("required", "refinement_only"):
            raise ValueError(f"{self.family_id}: unknown edge policy {self.edge_policy!r}")


@dataclass(frozen=True)
class ModelSpec:
    """Everything the pipeline needs: grid, prisms, families, route name."""

    grid: Grid
    prisms: tuple[PrismRecord, ...]
    families: tuple[CurveFamilyRecord, ...] = ()
    name: str = "model"
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def arc_registry(self) -> dict[str, ArcCarrier]:
        """Every source arc carrier by id (prism arcs and curve-family arcs)."""
        out = {a.id: a for p in self.prisms for a in p.arcs}
        out.update({a.id: a for fam in self.families for a in fam.arcs})
        return out
