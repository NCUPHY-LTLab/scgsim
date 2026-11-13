"""Prisms: a planar footprint extruded over a Z interval (design brief section 2).

Owns the source vocabulary the arrangement stages consume: the footprint (outer
ring plus holes, as integer grid points with the arc runs they contain), the
prism record and the catalogue of all prisms of one model.  A prism with
``z0 == z1`` is a sheet (a Route-A face metal or a lumped-port sheet).  All XY
coordinates are integers in grid units; Z stays exact (``Fraction``) because
the XY grid does not snap Z.

Does not own any derivation; it only validates and normalises its inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from scgsim.geometry.sgb.exact.carriers import (
    ArcCarrier,
    KnownCarrier,
    Lineage,
    LineCarrier,
    LineEdge,
    RingPrimitive,
)
from scgsim.geometry.sgb.exact.families import CurveFamilyRecord
from scgsim.geometry.sgb.exact.grid import Grid
from scgsim.geometry.sgb.exact.plane import Point, Ring, ring_twice_area

MaterialKind = Literal["conductor", "dielectric", "vacuum", "port"]
FILM_FEATURE_KINDS = frozenset({"stack_film", "local_film"})
"""Physical feature kinds of a conductor that is a *film* (a layer thin enough to be a sheet)."""


@dataclass(frozen=True)
class ArcRun:
    """The ring edges ``first_edge ...`` of ring ``ring`` are the witness chords of arc ``arc_id``."""

    ring: int
    first_edge: int
    arc_id: str


@dataclass(frozen=True)
class Footprint:
    """Outer ring (CCW) plus hole rings (CW), integer grid coordinates.

    Rings are stored as their polyline witness (arcs as chords); ``arc_runs``
    remembers which ring edges are witness pieces of which arc, so the carriers
    keep the exact arc definition.
    """

    outer: Ring
    holes: tuple[Ring, ...] = ()
    arc_runs: tuple[ArcRun, ...] = ()
    # Source edge number of every ring edge (outer first); empty means "number by position".
    edge_numbers: tuple[tuple[int, ...], ...] = ()

    @staticmethod
    def from_primitives(
        outer: Sequence[RingPrimitive],
        holes: Iterable[Sequence[RingPrimitive]] = (),
    ) -> tuple[Footprint, tuple[ArcCarrier, ...]]:
        """A footprint whose rings mix straight edges and arcs (already oriented).

        The outer ring must run counter-clockwise and holes clockwise.
        """
        rings: list[Ring] = []
        numbers: list[tuple[int, ...]] = []
        runs: list[ArcRun] = []
        arcs: list[ArcCarrier] = []
        for ring_index, primitives in enumerate([outer, *holes]):
            points: list[Point] = []
            ring_numbers: list[int] = []
            for position, primitive in enumerate(primitives):
                if isinstance(primitive, LineEdge):
                    points.append(primitive.p)
                    ring_numbers.append(
                        primitive.source_edge
                        if primitive.source_edge >= 0
                        else position
                    )
                    continue
                witness = primitive.witness_points()
                runs.append(ArcRun(ring_index, len(points), primitive.id))
                arcs.append(primitive)
                points.extend(witness[:-1])
                ring_numbers.extend([primitive.lineage.edge] * (len(witness) - 1))
            if (ring_twice_area(points) > 0) != (ring_index == 0):
                raise ValueError("ring orientation: outer must be CCW and holes CW")
            rings.append(tuple(points))
            numbers.append(tuple(ring_numbers))
        return (
            Footprint(rings[0], tuple(rings[1:]), tuple(runs), tuple(numbers)),
            tuple(arcs),
        )

    @staticmethod
    def from_rings(
        outer: Sequence[Point], holes: Iterable[Sequence[Point]] = ()
    ) -> Footprint:
        """A polygonal footprint; ring orientation is normalised, source edge numbers are kept."""
        rings, numbers = [], []
        for ring_index, points in enumerate([outer, *holes]):
            ring, ring_numbers = oriented_ring(points, ccw=ring_index == 0)
            rings.append(ring)
            numbers.append(ring_numbers)
        return Footprint(rings[0], tuple(rings[1:]), (), tuple(numbers))

    def rings(self) -> tuple[Ring, ...]:
        """The outer ring followed by the holes."""
        return (self.outer, *self.holes)

    def twice_area(self) -> int:
        """Twice the chord-polygon area (outer minus holes)."""
        return ring_twice_area(self.outer) + sum(
            ring_twice_area(hole) for hole in self.holes
        )


def oriented_ring(ring: Sequence[Point], *, ccw: bool) -> tuple[Ring, tuple[int, ...]]:
    """A ring with the requested orientation, plus the source edge number of each of its edges.

    The closing point and consecutive duplicate points are dropped (the edge
    that follows a dropped duplicate keeps its source number); reversing the
    ring reverses the edge numbers with it.
    """
    points = [(int(x), int(y)) for x, y in ring]
    source_index = list(range(len(points)))
    if len(points) > 1 and points[0] == points[-1]:
        points, source_index = points[:-1], source_index[:-1]
    kept: list[tuple[Point, int]] = []
    for point, index in zip(points, source_index, strict=False):
        if kept and kept[-1][0] == point:
            kept[-1] = (
                point,
                index,
            )  # the edge leaving a repeated point is the later one
        else:
            kept.append((point, index))
    if len(kept) > 1 and kept[0][0] == kept[-1][0]:
        kept.pop(0)
    if len(kept) < 3:
        raise ValueError("a ring needs at least three distinct points")
    cleaned = tuple(point for point, _ in kept)
    numbers = tuple(index for _, index in kept)
    if (ring_twice_area(cleaned) > 0) != ccw:
        count = len(cleaned)
        cleaned = tuple(reversed(cleaned))
        numbers = tuple(numbers[(count - 2 - j) % count] for j in range(count))
    return cleaned, numbers


@dataclass(frozen=True)
class PortIdentity:
    """Identity of a lumped-port sheet prism (the adapter-authored port facts)."""

    port_sheet_id: str
    source_name: str
    port_index: int
    direction: tuple[float, float, float]
    host_entity_ids: tuple[str, ...]
    target_layer: str
    source_layer: str
    # Whether the sheet's cells over a host conductor are that host (the port's landing) instead of
    # the port.  False for a region bound to a rectangular junction partition, which is already the
    # junction's central rectangle and keeps its full footprint.
    lands_on_hosts: bool = True


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
    priority: int = 0
    polygon: int = 0
    arcs: tuple[ArcCarrier, ...] = ()
    port: PortIdentity | None = None
    # The source physical declaration's feature kind (``stack_film``, ``local_film`` or
    # ``finite_volume``); ``None`` when the conductor has no declaration.
    feature_kind: str | None = None
    # Where the entity comes from, for errors that must say which source supplies a missing priority.
    occurrence: str | None = None  # the source occurrence (component instance) path
    layer: str | None = (
        None  # the source layer (logical level id, else the GDS layer/datatype)
    )

    def __post_init__(self) -> None:
        if self.z1 < self.z0:
            raise ValueError(f"{self.entity_id}: z1 < z0")
        if self.material_kind == "conductor" and not self.net_id:
            raise ValueError(f"{self.entity_id}: conductor prism needs a net_id")
        if (self.material_kind == "port") != (self.port is not None):
            raise ValueError(
                f"{self.entity_id}: exactly the port prisms carry a port identity"
            )
        if self.material_kind == "port" and self.z0 != self.z1:
            raise ValueError(f"{self.entity_id}: a port prism is a sheet")

    @property
    def is_sheet(self) -> bool:
        """Whether this prism has zero thickness."""
        return self.z0 == self.z1

    def carriers(self) -> list[LineCarrier]:
        """Boundary edges of every ring as lineage-carrying line carriers.

        A straight edge is numbered by its source edge (its position in the ring
        when the footprint records no numbers).  Arc runs contribute their witness
        chords, named ``<arc id>#<k>`` and carrying the arc's lineage.
        """
        runs = {(run.ring, run.first_edge): run for run in self.footprint.arc_runs}
        arcs = {arc.id: arc for arc in self.arcs}
        carriers: list[LineCarrier] = []
        for ring_index, ring in enumerate(self.footprint.rings()):
            numbers = (
                self.footprint.edge_numbers[ring_index]
                if self.footprint.edge_numbers
                else tuple(range(len(ring)))
            )
            carriers.extend(self._ring_carriers(ring_index, ring, numbers, runs, arcs))
        return carriers

    def _ring_carriers(  # noqa: PLR0913 - one ring's tables travel together
        self,
        ring_index: int,
        ring: Ring,
        numbers: tuple[int, ...],
        runs: dict[tuple[int, int], ArcRun],
        arcs: dict[str, ArcCarrier],
    ) -> list[LineCarrier]:
        carriers: list[LineCarrier] = []
        count = len(ring)
        edge = 0
        while edge < count:
            run = runs.get((ring_index, edge))
            if run is None:
                lineage = Lineage(
                    "polygon_edge",
                    self.entity_id,
                    self.polygon,
                    ring_index,
                    numbers[edge],
                )
                carriers.append(
                    LineCarrier(
                        lineage.describe(),
                        ring[edge],
                        ring[(edge + 1) % count],
                        lineage,
                        "source_edge",
                    )
                )
                edge += 1
                continue
            arc = arcs[run.arc_id]
            for piece in arc.pieces():
                carriers.append(
                    LineCarrier(
                        piece.id,
                        ring[edge],
                        ring[(edge + 1) % count],
                        arc.lineage,
                        "arc_chord",
                        piece.curve,
                        piece.piece,
                    )
                )
                edge += 1
        return carriers


@dataclass(frozen=True)
class PrismCatalogue:
    """Everything the arrangement stages need: the grid, the prisms and the cut families."""

    grid: Grid
    prisms: tuple[PrismRecord, ...]
    families: tuple[CurveFamilyRecord, ...] = ()
    name: str = "model"

    def known_carriers(self) -> dict[str, KnownCarrier]:
        """Every carrier id the catalogue makes, with its lineage and the curve it lies on.

        Prism boundary edges (straight edges and the witness chords of arcs), prism arcs, family lines, family
        arcs and the witness chords of family arcs.  Assembly, the 2D section and the sidecar read where a carrier
        id on an edge comes from here, never from the id.
        """
        known: dict[str, KnownCarrier] = {}
        for prism in self.prisms:
            for line in prism.carriers():
                known[line.id] = KnownCarrier(line.lineage, line.curve or line.id)
            for arc in prism.arcs:
                known[arc.id] = KnownCarrier(arc.lineage, arc.id)
        for family in self.families:
            for line in family.carriers:
                known[line.id] = KnownCarrier(line.lineage, line.id)
            for arc in family.arcs:
                known[arc.id] = KnownCarrier(arc.lineage, arc.id)
                for piece in arc.pieces():
                    known[piece.id] = KnownCarrier(arc.lineage, arc.id)
        return known

    def arc_registry(self) -> dict[str, ArcCarrier]:
        """Every source arc carrier by id (prism arcs and curve-family arcs)."""
        registry = {arc.id: arc for prism in self.prisms for arc in prism.arcs}
        registry.update({arc.id: arc for fam in self.families for arc in fam.arcs})
        return registry
