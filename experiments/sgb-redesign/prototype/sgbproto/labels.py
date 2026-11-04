"""Cell labelling by representative points (brief section 2 and 4, stage S5 part 1).

Owns deciding the slab label (occupying :class:`Domain`) and the level label
(:class:`LevelLabel`) of every arrangement cell. A cell is labelled by an exact
interior point and exact point-in-footprint tests against the prisms active
there; no floating point is involved.

Slab label rule: conductor wins; different-Net conductors overlapping raise;
dielectric/vacuum are resolved by ``priority`` (equal priority with different
materials raises).

Does not own the arrangement or which edges survive (``dissolve``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sgbproto.arrangement import Arrangement, BoxIndex, interior_point
from sgbproto.errors import OverlapError
from sgbproto.geometry import RepPoint, point_in_ring, ring_bbox
from sgbproto.records import Domain, LevelLabel, PrismRecord, SheetOwner


class PrismSet:
    """A set of prisms with a spatial index for exact containment queries."""

    def __init__(self, prisms: Sequence[PrismRecord], indices: Sequence[int]) -> None:
        self.prisms = prisms
        self.indices = tuple(indices)
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
        # Candidate rings by box (box stab is on the floor of the point).
        parity: Counter[int] = Counter()
        for r in self._index.stab(p[0] // den, p[1] // den):
            x0, y0, x1, y1 = self._boxes[r]
            if not (x0 * den <= p[0] <= x1 * den and y0 * den <= p[1] <= y1 * den):
                continue
            k, ring = self._rings[r]
            if point_in_ring(ring, p):
                parity[k] += 1
        return sorted(k for k, c in parity.items() if c % 2 == 1)

    def domain_at(self, p: RepPoint) -> Domain | None:
        return resolve_domain([self.prisms[k] for k in self.hit(p)], p)

    def owner_at(self, p: RepPoint) -> SheetOwner | None:
        return resolve_owner([self.prisms[k] for k in self.hit(p)], p)


def resolve_domain(hit: list[PrismRecord], where: RepPoint) -> Domain | None:
    """Occupying domain from the prisms containing a point."""
    if not hit:
        return None
    conductors = [p for p in hit if p.material_kind == "conductor"]
    if conductors:
        nets = {p.net_id for p in conductors}
        if len(nets) > 1:
            names = ", ".join(sorted(p.entity_id for p in conductors))
            raise OverlapError(f"different-Net conductors overlap near {_fmt(where)}: {names}")
        best = max(conductors, key=lambda p: (p.priority, p.material, p.entity_id))
        return Domain("conductor", best.material, best.net_id or "")
    top = max(p.priority for p in hit)
    winners = [p for p in hit if p.priority == top]
    kinds = {(p.material_kind, p.material) for p in winners}
    if len(kinds) > 1:
        names = ", ".join(sorted(p.entity_id for p in winners))
        raise OverlapError(f"ambiguous equal-priority materials near {_fmt(where)}: {names}")
    kind, material = kinds.pop()
    return Domain(kind, material)


def resolve_owner(hit: list[PrismRecord], where: RepPoint) -> SheetOwner | None:
    """Owner of the sheet(s) at a point on a level."""
    if not hit:
        return None
    owners = {SheetOwner("port" if p.role == "port" else "conductor", p.entity_id if p.role == "port" else p.net_id or "") for p in hit}
    if len(owners) > 1:
        raise OverlapError(f"sheets of different owners overlap near {_fmt(where)}: {sorted(o.name for o in owners)}")
    return owners.pop()


def _fmt(p: RepPoint) -> str:
    return f"({p[0] / p[2]:.3f}, {p[1] / p[2]:.3f}) grid units"


def label_slab(arr: Arrangement, cells: PrismSet) -> tuple[list[Domain | None], list[list[int]]]:
    """Slab label of every face (unbounded face: ``None``) and the prisms containing each face."""
    labels: list[Domain | None] = [None]
    hits: list[list[int]] = [[]]
    for fi in range(1, len(arr.faces)):
        p = interior_point(arr, fi)
        inside = cells.hit(p)
        hits.append(inside)
        labels.append(resolve_domain([cells.prisms[k] for k in inside], p))
    return labels, hits


def label_level(
    arr: Arrangement,
    below: list[Domain | None],
    above: list[Domain | None],
    sheets: PrismSet | None,
) -> list[LevelLabel]:
    """Level label ``(below, above, sheet owner, bands)`` of every face.

    ``below``/``above`` are the slab labels each face lies in (derived through
    ``linking`` so a level label can never disagree with the slab labels); only
    the sheet owner needs a footprint test at the face's representative point.
    """
    labels = [LevelLabel(None, None, None, ())]
    for fi in range(1, len(arr.faces)):
        owner = sheets.owner_at(interior_point(arr, fi)) if sheets is not None else None
        labels.append(LevelLabel(below[fi], above[fi], owner, ()))
    return labels
