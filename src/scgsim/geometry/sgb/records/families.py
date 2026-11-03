"""Cut families: named sets of curves added to the arrangement with a semantic role.

A cut family (design brief section 5) is how every way of partitioning a
surface enters the arrangement.  Sub-stage 1a only needs the record and its
edge policy (``required`` keeps the curve as a cell boundary, ``refinement_only``
keeps it as an embedded curve without splitting the label); the generators that
produce families (inset bands, port footprints, refinement curves) come later.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from scgsim.geometry.sgb.records.carriers import ArcCarrier, LineCarrier
from scgsim.geometry.sgb.records.support import SupportRegion

EdgePolicy = Literal["required", "refinement_only"]


@dataclass(frozen=True)
class CurveFamilyRecord:
    """A named set of curves added to the arrangement with a semantic role."""

    family_id: str
    role: str
    carriers: tuple[LineCarrier, ...]
    target_levels: tuple[Fraction, ...]
    edge_policy: EdgePolicy = "required"
    arcs: tuple[ArcCarrier, ...] = ()
    support: SupportRegion | None = (
        None  # inset bands: the region whose bands the family partitions
    )

    def __post_init__(self) -> None:
        if self.edge_policy not in ("required", "refinement_only"):
            raise ValueError(
                f"{self.family_id}: unknown edge policy {self.edge_policy!r}"
            )
