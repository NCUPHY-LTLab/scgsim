"""The band regions of an inset family, established before Imprint (design brief sections 5 and 6).

Owns the record that travels from the region construction (``bands.inset``) to labelling and the
Imprint check (``arrangement.labelling.bands``):

* per margin ``m_k``, the region ``R_{m_k}`` -- the points of the support at least ``m_k`` from its distance
  origins -- as the boundary cycles of its cells (exact integer rings, even-odd containment);
* the *required seams*: the family's carriers that separate band ``k - 1`` from band ``k``, each with the side
  band ``k`` lies on.

A band index is the number of regions a point lies in, so band numbers stay tied to the margin intervals: an
empty interval (a strip narrower than twice a margin) leaves its band without area and renumbers nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scgsim.geometry.exact.plane import Point, RepPoint, point_in_ring, ring_bbox


@dataclass(frozen=True)
class BandSeam:
    """A carrier of the family that separates band ``band - 1`` (outside ``R_m``) from ``band`` (inside).

    ``inside_left`` says whether band ``band`` lies on the left of the carrier's direction of travel
    (``p -> q`` of a line, ``start -> end`` of an arc).
    """

    carrier_id: str
    band: int
    inside_left: bool


@dataclass(frozen=True)
class MarginRegion:
    """``R_m`` within the support: the boundary cycles of its cells, which no other cell overlaps."""

    margin: int
    rings: tuple[tuple[Point, ...], ...]
    boxes: tuple[tuple[int, int, int, int], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "boxes", tuple(ring_bbox(r) for r in self.rings))

    def contains(self, point: RepPoint) -> bool:
        """Even-odd containment of a point inside the support but off every cycle of the region.

        The cells of ``R_m`` are disjoint, so a point inside one cell lies inside its outer cycle and in no
        hole of it; a cell that surrounds it adds its outer cycle and the hole holding the point: an even count.
        """
        x, y, den = point
        inside = False
        for ring, (x0, y0, x1, y1) in zip(self.rings, self.boxes, strict=True):
            if x0 * den <= x <= x1 * den and y0 * den <= y <= y1 * den:
                inside ^= point_in_ring(ring, point)
        return inside


@dataclass(frozen=True)
class BandRegions:
    """The regions ``R_{m_1} \\supseteq ... \\supseteq R_{m_K}`` of one support and the seams between its bands."""

    regions: tuple[MarginRegion, ...]
    seams: tuple[BandSeam, ...]

    def band_at(self, point: RepPoint) -> int:
        """Band of a point of the support off every region cycle: the number of regions holding it."""
        return sum(1 for region in self.regions if region.contains(point))
