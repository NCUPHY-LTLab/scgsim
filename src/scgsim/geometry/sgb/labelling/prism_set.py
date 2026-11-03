"""Cell labelling by representative points (design brief sections 2 and 4, stage S5 part 1).

Owns deciding the slab label (the occupying :class:`Domain`) and the sheet owner
of every arrangement cell.  A cell is labelled by an exact interior point and
exact point-in-footprint tests against the prisms active there; no floating
point is involved.

Slab label rule (the same precedence the curved arrangement of the compiler used):

* a conductor wins over any solution region; conductors of different Nets that
  overlap raise :class:`OverlapError`;
* among solution regions a dielectric wins over vacuum, then the higher
  ``priority`` wins; an unresolved tie between different materials raises.

Does not own the arrangement or which edges survive (``dissolve``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from scgsim.geometry.sgb.arrangement.dcel import BoxIndex
from scgsim.geometry.sgb.errors import OverlapError
from scgsim.geometry.sgb.plane import RepPoint, point_in_ring, ring_bbox
from scgsim.geometry.sgb.records.grid import Grid
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
        names = ", ".join(sorted({p.entity_id for p in conductors}))
        raise OverlapError(
            f"different-Net conductors overlap near {format_location(where, grid)}: {names}"
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

    A port sheet never overlaps another sheet; conductor sheets follow the same
    Net and priority rule as conductor bodies.
    """
    if not hit:
        return None
    ports = [p for p in hit if p.material_kind == "port"]
    if ports:
        if len(hit) > 1:
            names = ", ".join(sorted(p.entity_id for p in hit))
            raise OverlapError(
                f"a port sheet overlaps another sheet near {format_location(where, grid)}: {names}"
            )
        return SheetOwner("port", ports[0].entity_id)
    winner = _owning_conductor(hit, where, grid)
    return SheetOwner("conductor", winner.entity_id, winner.net_id or "")


def format_location(where: RepPoint, grid: Grid) -> str:
    """A point of the plane in micrometres, for error messages."""
    x = grid.to_um(where[0] / where[2])
    y = grid.to_um(where[1] / where[2])
    return f"({x:.4f}, {y:.4f}) um"
