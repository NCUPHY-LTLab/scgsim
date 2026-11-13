"""Transform authored junction geometry through one explicit local coordinate frame, and back to the layout."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from scgsim.geometry.grid.grid_regions import grid_change_message
from scgsim.geometry.junctions.core import PRECISION_UM
from scgsim.geometry.junctions.exact_geometry import inside_ring, region_edges
from scgsim.geometry.models.grid import format_box, format_length
from scgsim.geometry.models.input import LayoutPolygonSpec


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

    def layout_point(self, u, v) -> tuple[float, float]:
        """The layout point, in micrometres, of the local point ``(u, v)`` in Boolean units."""
        along = self.origin[0] + float(u) * PRECISION_UM
        across = self.origin[1] + float(v) * PRECISION_UM
        return (along * self.dx + across * self.px, along * self.dy + across * self.py)


@dataclass(frozen=True)
class JunctionPlaces:
    """Where a junction's local points lie in the layout, and how its refusals name it.

    ``width`` is the sheet's width in Boolean units, half of which is its centre line; ``port``
    is the lumped port the sheet belongs to, when it is not the junction itself.
    """

    frame: JunctionLocalFrame
    grid_um: float | None
    width: int
    junction_id: str
    port: str | None
    layer: str

    def place(self, points: Iterable[Sequence]) -> str:
        """Where local points lie in the layout: the point, or the box around them."""
        world = [self.frame.layout_point(u, v) for u, v in points]
        xs, ys = [x for x, _ in world], [y for _, y in world]
        return format_box((min(xs), min(ys), max(xs), max(ys)), self.grid_um)

    def centre(self, u) -> str:
        """Where the sheet's centre line lies ``u`` Boolean units along the port."""
        return self.place([(u, Fraction(self.width, 2))])

    def gap(self, lower, upper) -> str:
        """The clear gap between the terminals along the centre line, and its length."""
        length = format_length((upper - lower) * PRECISION_UM, self.grid_um)
        return (
            f"the clear gap between its conductors runs from {self.centre(lower)} to "
            f"{self.centre(upper)}, {length} long"
        )

    def refusal(self, lead: str, findings: Sequence[str], action: str) -> str:
        """A layout refusal of this junction: ``lead``, the junction and its sheet's layer, where and who acts."""
        named = f"junction {self.junction_id!r}"
        if self.port is not None and self.port != self.junction_id:
            named += f" (lumped port {self.port!r})"
        return grid_change_message(
            f"{lead}: {named}", [self.layer], self.grid_um, findings, action
        )
