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

    Conductors are identified by Net (conductor entities of one Net merge into one
    domain); a dielectric or vacuum domain is identified by the solution
    region that owns it, so two solution regions of equal material stay
    separate volumes with a face between them.
    """

    kind: str  # "conductor" | "dielectric" | "vacuum"
    material: str
    net: str = ""
    owner: str = ""

    @property
    def name(self) -> str:
        """Net of a conductor domain, solution-region id of any other domain."""
        return self.net if self.kind == "conductor" else self.owner or self.material


@dataclass(frozen=True, order=True)
class SheetOwner:
    """Owner of a zero-thickness sheet face: a conductor Net or a port."""

    kind: str  # "conductor" | "port"
    name: str


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
