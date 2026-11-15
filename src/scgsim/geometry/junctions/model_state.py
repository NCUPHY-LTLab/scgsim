"""Carry exact junction source incidences and the shared emission registry."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from typing import Any, cast

from scgsim.geometry.grid.planar_loops import (
    cancel_reversed_planar_edges,
    simple_planar_loops_from_edges,
)
from scgsim.geometry.grid.scope import GridScope
from scgsim.geometry.junctions.core import PRECISION_UM, JunctionPartitionFormat
from scgsim.geometry.junctions.exact_geometry import cross, inside_ring, region_edges
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.refusal import internal

type IntegerPoint = tuple[int, int]
type RationalPoint = tuple[int | Fraction, int | Fraction]
type FloatPoint = tuple[float, float]
type EdgeKey = tuple[str, int, int]
type TracedEdge = tuple[RationalPoint, RationalPoint, tuple[EdgeKey, ...]]


def _direct_holes(ring, loops) -> list:
    """The loops inside ``ring``; a hole inside another hole is not supported."""
    holes = [
        hole for hole in loops if hole is not ring and inside_ring(hole[0], ring) == 1
    ]
    if any(
        any(inside_ring(hole[0], other) == 1 for other in holes if other is not hole)
        for hole in holes
    ):
        raise ValueError(
            internal("junction residual has unsupported nested components")
        )
    return holes


@dataclass
class JunctionModel:
    """Carry exact source incidences, cut pieces and shared emitted coordinates."""

    build_input: GeometryBuildInput
    originals: dict[str, LayoutPolygonSpec]
    records: dict[str, dict[str, Any]]
    output_format: JunctionPartitionFormat
    # The input's grid, which the model's Booleans run on.
    scope: GridScope
    authored: set[str] = field(init=False)
    eligible_polygons: dict[str, set[str]] = field(init=False)
    sheet_metals: defaultdict[str, set[str]] = field(init=False)
    metal_ids: set[str] = field(init=False)
    needed: set[str] = field(init=False)
    rings: dict[str, tuple[tuple[IntegerPoint, ...], ...]] = field(init=False)
    edges: dict[EdgeKey, tuple[IntegerPoint, IntegerPoint]] = field(init=False)
    events: defaultdict[EdgeKey, dict[RationalPoint, Fraction]] = field(init=False)
    members: defaultdict[RationalPoint, set[EdgeKey]] = field(init=False)
    intersection_deviation: float = field(init=False)
    affected_owners: set[str] = field(init=False)
    raw_free: dict[str, list[TracedEdge]] = field(init=False)
    sliced: dict[str, list[list[TracedEdge]]] = field(init=False)
    sheet_sliced: dict[str, list[list[TracedEdge]]] = field(init=False)
    free_edges: dict[str, list[TracedEdge]] = field(init=False)
    registry: dict[RationalPoint, FloatPoint] = field(init=False)
    reverse: dict[FloatPoint, RationalPoint] = field(init=False)
    affected_polygons: set[str] = field(init=False)
    model: dict[str, LayoutPolygonSpec] = field(init=False)
    additions: list[LayoutPolygonSpec] = field(init=False)
    appended: defaultdict[str, list[str]] = field(init=False)
    physical_ends: defaultdict[str, list[TracedEdge]] = field(init=False)
    entities: list[SemanticEntitySpec] = field(init=False)
    actual_rings: dict[str, tuple[tuple[tuple[Fraction, Fraction], ...], ...]] = field(
        init=False
    )

    def event(self, edge, point, parameter):
        """Register one exact source-edge parameter and its shared incident point."""
        prior = self.events[edge].get(point)
        if prior is not None and prior != parameter:
            raise ValueError(
                internal(
                    "numerical junction incidence collapses distinct source parameters"
                )
            )
        self.events[edge][point] = parameter
        self.members[point].add(edge)

    def refine(self, traced):
        """Reinsert registered source nodes in their exact edge-parameter order."""
        result = []
        for a, b, sources in traced:
            split = {Fraction(0): a, Fraction(1): b}
            for source in sources:
                ta, tb = (self.events[source][a], self.events[source][b])
                if ta == tb:
                    raise ValueError(
                        internal(
                            "numerical junction source edge has collapsed parameter span"
                        )
                    )
                for point, t in self.events[source].items():
                    u = (t - ta) / (tb - ta)
                    if 0 < u < 1:
                        if u in split and split[u] != point:
                            raise ValueError(
                                internal(
                                    "numerical junction source incidences disagree"
                                )
                            )
                        split[u] = point
            ordered = [point for _, point in sorted(split.items())]
            result.extend(((start, end, sources) for start, end in pairwise(ordered)))
        return result

    def emit(self, point):
        """Emit one shared point without collapsing distinct exact coordinates."""
        if point not in self.registry:
            result = cast(
                FloatPoint,
                tuple((float(v * Fraction(str(PRECISION_UM))) for v in point)),
            )
            if result in self.reverse and self.reverse[result] != point:
                raise ValueError(
                    internal(
                        "numerical junction shared vertices collapse on serialization"
                    )
                )
            self.reverse[result] = point
            self.registry[point] = result
        return self.registry[point]

    def _emitted_ring(self, index: int, loop, simplify: bool):
        """One ring of a lowered region, turned outward for the exterior and inward for a hole, as emitted points."""
        if simplify:
            loop = tuple(
                (
                    point
                    for n, point in enumerate(loop)
                    if cross(loop[n - 1], point, loop[(n + 1) % len(loop)]) != 0
                )
            )
        area = sum((a[0] * b[1] - b[0] * a[1] for a, b in region_edges((loop,))))
        if (area > 0) != (index == 0):
            loop = tuple(reversed(loop))
        return tuple((self.emit(point) for point in loop))

    def lower(self, traced, *, simplify=False):
        """Lower directed residual edges into emitted outer and hole rings."""
        raw = [(a, b) for a, b, _ in traced]
        raw = cancel_reversed_planar_edges(raw)
        if not raw:
            return []
        loops = simple_planar_loops_from_edges(raw)
        outer = [
            ring
            for ring in loops
            if not any(
                (
                    inside_ring(ring[0], other) == 1
                    for other in loops
                    if other is not ring
                )
            )
        ]
        regions = []
        for ring in outer:
            holes = _direct_holes(ring, loops)
            regions.append(
                tuple(
                    self._emitted_ring(index, loop, simplify)
                    for index, loop in enumerate((ring, *holes))
                )
            )
        if sum((len(region) for region in regions)) != len(loops):
            raise ValueError(
                internal("junction residual has point-connected components")
            )
        return regions
