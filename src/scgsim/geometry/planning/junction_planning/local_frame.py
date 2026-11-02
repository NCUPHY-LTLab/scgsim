"""Transform authored junction geometry through one explicit local coordinate frame."""

from __future__ import annotations

from dataclasses import dataclass

from scgsim.geometry.models.input import LayoutPolygonSpec
from scgsim.geometry.planning.junction_planning.core import PRECISION_UM
from scgsim.geometry.planning.junction_planning.exact_geometry import (
    inside_ring,
    region_edges,
)


@dataclass(frozen=True)
class JunctionLocalFrame:
    """Carry one junction's orthogonal axes, origin and accumulated deviations."""

    dx: float
    dy: float
    px: float
    py: float
    origin: tuple[float, float]
    deviations: list[float]

    def convert(self, polygon: LayoutPolygonSpec):
        """Convert complete polygon rings and retain their existing precision checks."""
        rings = []
        for ring in (polygon.exterior, *polygon.holes):
            converted = []
            for x, y in ring:
                u, v = (
                    x * self.dx + y * self.dy - self.origin[0],
                    x * self.px + y * self.py - self.origin[1],
                )
                point = (round(u / PRECISION_UM), round(v / PRECISION_UM))
                self.deviations.append(
                    max(
                        abs(point[0] * PRECISION_UM - u),
                        abs(point[1] * PRECISION_UM - v),
                    )
                )
                converted.append(point)
            rings.append(tuple(converted))
        for index, ring in enumerate(rings):
            area = sum((a[0] * b[1] - b[0] * a[1] for a, b in region_edges((ring,))))
            if len(set(ring)) < 3 or area == 0:
                raise ValueError(
                    "numerical junction geometry failure: source polygon/hole collapses at Boolean precision"
                )
            if index and any((inside_ring(point, rings[0]) != 1 for point in ring)):
                raise ValueError(
                    "numerical junction geometry failure: source hole loses interior topology"
                )
        return tuple(rings)
