"""Cell labelling by representative points (design brief sections 2 and 4, stage S5 part 1).

Owns deciding the slab label (the occupying :class:`Domain`) and the sheet owner
of every arrangement cell.  A cell is labelled by an exact interior point and
exact point-in-footprint tests against the prisms active there; no floating
point is involved.

Slab label rule:

* a conductor wins over any solution region; conductors of different Nets that
  overlap raise :class:`OverlapError`;
* among solution regions a dielectric wins over vacuum, then the higher
  ``priority`` wins; an unresolved tie between different materials raises.

Does not own the arrangement or which edges survive (``dissolve``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from scgsim.geometry.exact.grid import Grid
from scgsim.geometry.exact.plane import RepPoint, point_in_ring, ring_bbox
from scgsim.geometry.sgb.arrangement.dcel import BoxIndex
from scgsim.geometry.sgb.errors import OverlapError
from scgsim.geometry.sgb.records.labels import Domain, SheetOwner
from scgsim.geometry.sgb.records.prisms import PrismRecord


class PrismSet:
    """A set of prisms with a spatial index for exact containment queries."""

    def __init__(
        self, prisms: Sequence[PrismRecord], indices: Sequence[int], grid: Grid
    ) -> None:
        self.prisms = prisms
        self.indices = tuple(indices)
        self.grid = grid
        self._rings = []  # (prism index, ring)
        boxes = []
        for k in self.indices:
            for ring in prisms[k].footprint.rings():
                self._rings.append((k, ring))
                boxes.append(ring_bbox(ring))
        self._index = BoxIndex(boxes) if boxes else None
        self._boxes = boxes

    def hit(self, p: RepPoint) -> list[int]:
        """Indices of prisms whose footprint contains the exact point (even-odd over rings)."""
        if self._index is None:
            return []
        den = p[2]
        parity: Counter[int] = Counter()
        for r in self._index.stab(p[0] // den, p[1] // den):
            x0, y0, x1, y1 = self._boxes[r]
            if not (x0 * den <= p[0] <= x1 * den and y0 * den <= p[1] <= y1 * den):
                continue
            k, ring = self._rings[r]
            if point_in_ring(ring, p):
                parity[k] += 1
        return sorted(k for k, count in parity.items() if count % 2 == 1)

    def domain_at(self, p: RepPoint) -> Domain | None:
        """Occupying domain at an exact point."""
        return resolve_domain([self.prisms[k] for k in self.hit(p)], p, self.grid)

    def owner_at(self, p: RepPoint) -> SheetOwner | None:
        """Owner of the sheet(s) at an exact point."""
        return resolve_owner([self.prisms[k] for k in self.hit(p)], p, self.grid)


def resolve_domain(
    hit: list[PrismRecord], where: RepPoint, grid: Grid
) -> Domain | None:
    """Occupying domain from the body prisms containing a point."""
    if not hit:
        return None
    conductors = [p for p in hit if p.material_kind == "conductor"]
    if conductors:
        return _resolve_conductor(conductors, where, grid)
    return _resolve_solution(hit, where, grid)


def _resolve_conductor(
    conductors: list[PrismRecord], where: RepPoint, grid: Grid
) -> Domain:
    """One owning conductor entity: different Nets overlapping is an error, same Net goes by priority."""
    winner = _owning_conductor(conductors, where, grid)
    return Domain("conductor", winner.material, winner.net_id or "", winner.entity_id)


def _owning_conductor(
    conductors: list[PrismRecord], where: RepPoint, grid: Grid
) -> PrismRecord:
    """The prism whose entity owns a point covered by several conductor prisms.

    Different Nets overlapping is an error.  Entities of one Net overlapping: the one
    with the higher declared priority owns the overlap (decision D5, reversible); equal
    priority fails closed naming both entities.
    """
    if len({p.net_id for p in conductors}) > 1:
        raise OverlapError(
            f"different-Net conductors overlap near {format_location(where, grid)}: "
            f"{named_entities(conductors)}. {FIX_NETS}"
        )
    top = max(p.priority for p in conductors)
    leaders = sorted({p.entity_id for p in conductors if p.priority == top})
    if len(leaders) > 1:
        tied = [p for p in conductors if p.priority == top]
        raise OverlapError(
            f"same-Net entities {' and '.join(leaders)} overlap near {format_location(where, grid)} with "
            f"equal priority {top}: declare distinct entity priorities (SemanticEntitySpec.priority) so "
            f"that exactly one of them owns the overlap; {priority_source(tied)}"
        )
    return next(p for p in conductors if p.priority == top)


def _resolve_solution(hit: list[PrismRecord], where: RepPoint, grid: Grid) -> Domain:
    def rank(prism: PrismRecord) -> tuple[bool, int]:
        return (prism.material_kind == "dielectric", prism.priority)

    top_rank = max(rank(p) for p in hit)
    winners = [p for p in hit if rank(p) == top_rank]
    materials = {(p.material_kind, p.material) for p in winners}
    if len(materials) > 1:
        names = ", ".join(sorted(p.entity_id for p in winners))
        raise OverlapError(
            "ambiguous equal-priority solution regions of different materials near "
            f"{format_location(where, grid)}: {names}; declare distinct entity priorities "
            f"(SemanticEntitySpec.priority); {priority_source(winners)}"
        )
    owner = max(winners, key=lambda p: p.entity_id)
    return Domain(owner.material_kind, owner.material, owner=owner.entity_id)


def priority_source(tied: list[PrismRecord]) -> str:
    """Which source must supply the missing priority for entities that tie (lower ``mesh_order`` wins).

    Entities of different layers are ordered by the LayerStack ``mesh_order``; entities of one layer in one
    component by the component annotation ``priority`` (the PDK fills it); entities of one layer that come
    from different occurrences by an assembly-level priority on the GeometryPlan.  Entities whose source is
    unknown (hand-made records) get the generic advice.
    """
    by_entity = {p.entity_id: p for p in tied}
    sources: list[str] = []
    entities = sorted(by_entity.values(), key=lambda p: p.entity_id)
    for i, first in enumerate(entities):
        for second in entities[i + 1 :]:
            source = _pair_source(first, second)
            if source not in sources:
                sources.append(source)
    return (
        "; ".join(sources)
        if sources
        else "the source that declares these entities must supply it"
    )


def _pair_source(first: PrismRecord, second: PrismRecord) -> str:
    pair = f"{first.entity_id} and {second.entity_id}"
    if (
        first.layer is not None
        and second.layer is not None
        and first.layer != second.layer
    ):
        return (
            f"{pair} are on different layers ({first.layer}, {second.layer}): the LayerStack "
            "`mesh_order` must differ (lower mesh_order wins)"
        )
    if (
        first.occurrence is not None
        and second.occurrence is not None
        and first.occurrence != second.occurrence
    ):
        return (
            f"{pair} come from different occurrences ({first.occurrence}, {second.occurrence}): "
            "the GeometryPlan assembly must give the overlap an assembly-level priority"
        )
    if first.layer is not None and second.layer is not None:
        return (
            f"{pair} share layer {first.layer} in one component: the component annotation `priority` "
            "must differ (the PDK fills it)"
        )
    return f"{pair}: the source that declares them must supply distinct priorities"


def resolve_owner(
    hit: list[PrismRecord], where: RepPoint, grid: Grid
) -> SheetOwner | None:
    """Owner of the sheet(s) at a point on a level.

    A port sheet never overlaps another sheet, except a conductor sheet that is one of its
    hosts: there the port lands on its host and the host owns the point.  Conductor sheets
    follow the same Net and priority rule as conductor bodies.
    """
    if not hit:
        return None
    ports = [p for p in hit if p.material_kind == "port"]
    if ports:
        landing = _landing_hosts(ports, hit)
        if len(ports) > 1 or len(hit) - len(ports) > len(landing):
            raise OverlapError(
                f"a port sheet overlaps another sheet near {format_location(where, grid)}: "
                f"{_named_sheets(hit)}. Fix the layout: draw each port sheet clear of every "
                "other sheet but the conductors it lands on."
            )
        if not landing:
            return SheetOwner("port", ports[0].entity_id)
        hit = landing
    winner = _owning_conductor(hit, where, grid)
    return SheetOwner("conductor", winner.entity_id, winner.net_id or "")


def _named_sheets(sheets: Sequence[PrismRecord]) -> str:
    """The sheets at a point, sorted: each port with its name and drawn layer, each other sheet with its layer."""
    named = [
        f"{p.entity_id} (port {p.port.source_name!r}) on layer {p.port.source_layer}"
        if p.port is not None
        else named_entities([p])
        for p in sheets
    ]
    return ", ".join(sorted(named))


def _landing_hosts(
    ports: list[PrismRecord], covering: list[PrismRecord]
) -> list[PrismRecord]:
    """The conductors covering a point that are hosts of a port there, when that port lands on them."""
    return [
        host
        for host in covering
        if host.material_kind == "conductor"
        and any(
            port.port is not None
            and port.port.lands_on_hosts
            and host.entity_id in port.port.host_entity_ids
            for port in ports
        )
    ]


def without_landings(
    owner: SheetOwner | None, covering: list[PrismRecord]
) -> tuple[SheetOwner | None, list[PrismRecord]]:
    """The sheet owner and the covering prisms of a level point once a port's landing is taken out.

    The port footprint is the authored region whole.  Where it lies over a host conductor of the
    port (a body above or below the port plane, or a sheet on it), the point is the host's surface:
    it carries no port, so the port keeps only its active remainder and shares the host's own
    edges.  ``covering`` lists every prism whose footprint contains the point (the slabs on both
    sides and the sheets on the level).
    """
    ports = [p for p in covering if p.material_kind == "port"]
    landed = [p for p in ports if _landing_hosts([p], covering)]
    if not landed:
        return owner, covering
    kept = [p for p in covering if p not in landed]
    if owner is not None and owner.kind == "port":
        owner = None
    return owner, kept


FIX_NETS = (
    "Fix the layout or the Net declaration: draw the conductors of different Nets apart, "
    "or declare them one Net."
)


def named_entities(prisms: Sequence[PrismRecord]) -> str:
    """The prisms' entities, each with its source layer, sorted, for error messages."""
    named = {
        p.entity_id if p.layer is None else f"{p.entity_id} on layer {p.layer}"
        for p in prisms
    }
    return ", ".join(sorted(named))


def format_location(where: RepPoint, grid: Grid) -> str:
    """A point of the plane in micrometres, for error messages."""
    x = grid.to_um(where[0] / where[2])
    y = grid.to_um(where[1] / where[2])
    return f"({x:.4f}, {y:.4f}) um"
