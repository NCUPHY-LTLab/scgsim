"""Cut families: named sets of curves added to the arrangement with a semantic role.

A cut family (design brief section 5) is how every way of partitioning a
surface enters the arrangement.  Every curve of a family is required: it stays a
cell boundary even where the labels on its two sides agree.  The inset bands
(``bands.inset``) are the families SGB generates.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from scgsim.geometry.exact.carriers import ArcCarrier, LineCarrier
from scgsim.geometry.sgb.records.band_regions import BandRegions
from scgsim.geometry.sgb.records.support import SupportRegion


@dataclass(frozen=True)
class CurveFamilyRecord:
    """A named set of curves added to the arrangement with a semantic role."""

    family_id: str
    role: str
    carriers: tuple[LineCarrier, ...]
    target_levels: tuple[Fraction, ...]
    arcs: tuple[ArcCarrier, ...] = ()
    support: SupportRegion | None = (
        None  # inset bands: the region whose bands the family partitions
    )
    regions: BandRegions | None = (
        None  # inset bands: the regions R_m and the required seams, established before Imprint
    )
