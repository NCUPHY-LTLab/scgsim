"""Labels that arrangement cells receive (design brief sections 2 and 4).

A slab cell carries the occupying :class:`Domain`.  A level cell carries a
:class:`LevelLabel`: the domains below and above it plus the owner of a sheet
that lies on it.  Equal labels on both sides of an edge let the edge dissolve.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Domain:
    """Occupying domain of a slab cell (what a volume is made of).

    Every domain has one owning entity (design brief, decision D5: surfaces are
    attributed per Entity).  A conductor domain is its owning conductor entity --
    where two entities of one Net overlap, the overlap belongs to the one of higher
    declared priority -- and a dielectric or vacuum domain is its owning solution
    region.  Two entities of equal material therefore stay separate volumes with a
    required face between them; ``net`` records the electrical identity of a
    conductor for the backends that need it.
    """

    kind: str  # "conductor" | "dielectric" | "vacuum"
    material: str
    net: str = ""
    owner: str = ""

    @property
    def name(self) -> str:
        """The owning entity id (the material name when a hand-made domain has no owner)."""
        return self.owner or self.material


@dataclass(frozen=True, order=True)
class SheetOwner:
    """Owner of a zero-thickness sheet face: a conductor entity (with its Net) or a port."""

    kind: str  # "conductor" | "port"
    name: str  # the entity id of a conductor sheet, the port sheet id of a port
    net: str = ""


@dataclass(frozen=True)
class LevelLabel:
    """Label of a level-arrangement cell (design brief section 2)."""

    below: Domain | None
    above: Domain | None
    sheet: SheetOwner | None = None
    bands: tuple[tuple[str, int], ...] = ()

    def needs_face(self) -> bool:
        """Whether the cell is a face: a domain change across it, or a sheet on it."""
        return self.below != self.above or self.sheet is not None
