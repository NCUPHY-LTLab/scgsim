"""Re-check an emitted junction partition in exact arithmetic.

The local proof (`junction_partition.py`) and the model lowering (`junction_model.py`)
both work from the authored geometry. These checks read only the emitted model
polygons and Entities, so they catch a lowering that drifted from the proof:
no metal inside the central RLC rectangle, each end wholly covered by its own net
and touching no other, open sides clear, and the three parts covering the free sheet.

Owns: those emitted-coordinate checks. Does not own: producing the geometry.
Coordinates are the emitted floats, converted to exact `Fraction` before any predicate.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

from scgsim.aedt.preparation.junction_exact import (
    _closed_contact,
    _covered,
    _cross,
    _edges,
    _inside_ring,
    _on,
    _section,
)
from scgsim.geometry._primitives.loops import _cancel_reversed_planar_edges


@dataclass
class FinalModel:
    """The emitted model, re-checked in exact arithmetic."""

    records: dict
    model: dict
    entities: list

    def check(self):
        """Check exactly emitted coordinates, independently of local selection."""
        actual_rings = {
            pid: tuple(
                tuple(tuple(Fraction(str(v)) for v in point) for point in ring)
                for ring in (polygon.exterior, *polygon.holes)
            )
            for pid, polygon in self.model.items()
        }
        conductors = [
            (e.net_id, pid, actual_rings[pid])
            for e in self.entities
            if e.material_kind == "conductor"
            for pid in e.polygon_ids
        ]
        for record in self.records.values():
            c = actual_rings[record["central_polygon_id"]][0]
            _check_central_interior_clear(c, conductors)
            for end in record["ends"]:
                self._check_end(end, c, actual_rings, conductors)
            _check_open_sides_clear(c, conductors)
            record["proof_conditions"]["all_derived_junctions_checked"] = True

    def _check_end(self, end, c, actual_rings, conductors):
        own = [region for net, _, region in conductors if net == end["net_id"]]
        other = [region for net, _, region in conductors if net != end["net_id"]]
        if end["polygon_id"]:
            piece = actual_rings[end["polygon_id"]]
            generated_end_ids = {
                part["polygon_id"] for r in self.records.values() for part in r["ends"]
            }
            arms = [
                actual_rings[pid]
                for e in self.entities
                if e.semantic_id == end["source_entity_id"]
                for pid in e.polygon_ids
                if pid not in generated_end_ids
            ]
            if not _fraction_boundary_contact([piece], arms):
                raise ValueError(
                    "final model end lacks positive-length source-arm contact"
                )
            if _closed_contact([piece], other):
                raise ValueError("final cross-junction end contacts another net")
        n = 0 if end["side"] == "A" else 1
        start, finish = (c[3], c[0]) if n == 0 else (c[1], c[2])
        if not _model_edge_covered(own, start, finish):
            raise ValueError("final model designated end edge is not wholly covered")
        if _model_edge_contacts(other, start, finish):
            raise ValueError("final cross-junction wrong-net end/corner contact")


def _strictly_inside(point, region):
    return _inside_ring(point, region[0]) == 1 and all(
        _inside_ring(point, hole) == -1 for hole in region[1:]
    )


def _check_central_interior_clear(c, conductors):
    """No emitted metal may enter the interior of the central RLC rectangle `c`."""
    midpoint = tuple(sum(point[i] for point in c) / 4 for i in (0, 1))
    for _, _, region in conductors:
        if (
            any(_strictly_inside(point, (c,)) for point in region[0])
            or any(_strictly_inside(point, region) for point in (*c, midpoint))
            or any(
                _cross(a, b, u) * _cross(a, b, v) < 0
                and _cross(u, v, a) * _cross(u, v, b) < 0
                for a, b in _edges(region)
                for u, v in _edges((c,))
            )
        ):
            raise ValueError("final cross-junction PEC intersects RLC interior")


def _check_open_sides_clear(c, conductors):
    """The two sides of `c` that do not touch a terminal must touch no metal."""
    for start, finish in ((c[0], c[1]), (c[2], c[3])):
        for _, _, region in conductors:
            intervals, points = _model_edge_section(region, start, finish)
            if intervals or any(0 < point < 1 for point in points):
                raise ValueError("final cross-junction open-side contact")


def _fraction_boundary_contact(left, right):
    """Whether two groups of exact-coordinate polygons share boundary of positive length."""

    def lines(regions):
        result = defaultdict(list)
        for region in regions:
            for a, b in _edges(region):
                if a[0] != b[0]:
                    slope = (b[1] - a[1]) / (b[0] - a[0])
                    key, axis = ("x", slope, a[1] - slope * a[0]), 0
                else:
                    key, axis = ("y", a[0]), 1
                result[key].append((min(a[axis], b[axis]), max(a[axis], b[axis])))
        return result

    a, b = lines(left), lines(right)
    return any(
        max(x0, y0) < min(x1, y1)
        for key in a.keys() & b.keys()
        for x0, x1 in a[key]
        for y0, y1 in b[key]
    )


def emitted_partition_coverage(free, parts):
    """Compare actual emitted boundaries, including central simplification."""

    def rational(regions):
        return [
            tuple(
                tuple(tuple(Fraction(str(v)) for v in point) for point in ring)
                for ring in region
            )
            for region in regions
        ]

    free, parts = rational(free), rational(parts)
    points = sorted(
        {point for region in (*free, *parts) for ring in region for point in ring}
    )
    xs = [point[0] for point in points]

    def edges(regions):
        result = []
        for region in regions:
            for a, b in _edges(region):
                candidates = points[
                    bisect_left(xs, min(a[0], b[0])) : bisect_right(xs, max(a[0], b[0]))
                ]
                ordered = sorted(
                    (point for point in candidates if _on(point, a, b)),
                    key=lambda point: (
                        (point[0] - a[0]) * (b[0] - a[0])
                        + (point[1] - a[1]) * (b[1] - a[1])
                    ),
                )
                result.extend(pairwise(ordered))
        return _cancel_reversed_planar_edges(result)

    return sorted(edges(free)) == sorted(edges(parts))


def _model_edge_coordinates(region, start, finish):
    """Express `region` in a frame whose first axis runs `start` to `finish`.

    Along is scaled to 0..1 over the edge; across is the (unnormalized) offset.
    """
    dx, dy = finish[0] - start[0], finish[1] - start[1]
    norm = dx * dx + dy * dy
    return tuple(
        tuple(
            (
                ((p[0] - start[0]) * dx + (p[1] - start[1]) * dy) / norm,
                dx * (p[1] - start[1]) - dy * (p[0] - start[0]),
            )
            for p in ring
        )
        for ring in region
    )


def _model_edge_section(region, start, finish):
    return _section(_model_edge_coordinates(region, start, finish), 1, 0, 0, 1)


def _model_edge_covered(regions, start, finish):
    return _covered(
        [_model_edge_coordinates(region, start, finish) for region in regions],
        1,
        0,
        0,
        1,
    )


def _model_edge_contacts(regions, start, finish):
    return any(any(_model_edge_section(region, start, finish)) for region in regions)
