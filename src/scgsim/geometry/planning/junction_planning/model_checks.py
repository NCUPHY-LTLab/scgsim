"""Check emitted partition coverage and exact final contact edges."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from fractions import Fraction
from itertools import pairwise

import gdstk

from scgsim.geometry.planning.junction_planning.core import PRECISION_UM
from scgsim.geometry.planning.junction_planning.exact_geometry import (
    covered,
    on,
    region_edges,
    section,
)
from scgsim.geometry.primitives.grid import polygon_boolean
from scgsim.geometry.primitives.planar_loops import cancel_reversed_planar_edges


def fraction_boundary_contact(left, right):
    """Test exact rational positive-length contact between emitted region boundaries."""

    def lines(regions):
        result = defaultdict(list)
        for region in regions:
            for a, b in region_edges(region):
                if a[0] != b[0]:
                    slope = (b[1] - a[1]) / (b[0] - a[0])
                    key, axis = (("x", slope, a[1] - slope * a[0]), 0)
                else:
                    key, axis = (("y", a[0]), 1)
                result[key].append((min(a[axis], b[axis]), max(a[axis], b[axis])))
        return result

    a, b = (lines(left), lines(right))
    return any(
        (
            max(x0, y0) < min(x1, y1)
            for key in a.keys() & b.keys()
            for x0, x1 in a[key]
            for y0, y1 in b[key]
        )
    )


def fraction_overlap(left, right):
    """The overlap of two region collections (rings of model points), as Boolean polygons."""

    def native(regions):
        polygons = []
        for region in regions:
            outer, *holes = (
                gdstk.Polygon([(float(point[0]), float(point[1])) for point in ring])
                for ring in region
            )
            polygons.extend(
                polygon_boolean([outer], holes, "not", precision=PRECISION_UM) or ()
            )
        return polygons

    return tuple(
        polygon_boolean(native(left), native(right), "and", precision=PRECISION_UM)
        or ()
    )


def fraction_area_overlap(left, right):
    """Test whether emitted regions overlap over a positive area (an end piece on its terminal)."""
    return bool(fraction_overlap(left, right))


def emitted_partition_coverage(free, parts):
    """Compare actual emitted boundaries, including central simplification."""

    def rational(regions):
        return [
            tuple(
                (
                    tuple((tuple((Fraction(str(v)) for v in point)) for point in ring))
                    for ring in region
                )
            )
            for region in regions
        ]

    free, parts = (rational(free), rational(parts))
    points = sorted(
        {point for region in (*free, *parts) for ring in region for point in ring}
    )
    xs = [point[0] for point in points]

    def edges(regions):
        result = []
        for region in regions:
            for a, b in region_edges(region):
                candidates = points[
                    bisect_left(xs, min(a[0], b[0])) : bisect_right(xs, max(a[0], b[0]))
                ]
                ordered = sorted(
                    (point for point in candidates if on(point, a, b)),
                    key=lambda point: (
                        (point[0] - a[0]) * (b[0] - a[0])
                        + (point[1] - a[1]) * (b[1] - a[1])
                    ),
                )
                result.extend(pairwise(ordered))
        return cancel_reversed_planar_edges(result)

    return sorted(edges(free)) == sorted(edges(parts))


def _model_edge_coordinates(region, start, finish):
    dx, dy = (finish[0] - start[0], finish[1] - start[1])
    norm = dx * dx + dy * dy
    return tuple(
        (
            tuple(
                (
                    (
                        ((p[0] - start[0]) * dx + (p[1] - start[1]) * dy) / norm,
                        dx * (p[1] - start[1]) - dy * (p[0] - start[0]),
                    )
                    for p in ring
                )
            )
            for ring in region
        )
    )


def model_edge_section(region, start, finish):
    """Read exact conductor sections along one emitted model edge."""
    return section(_model_edge_coordinates(region, start, finish), 1, 0, 0, 1)


def model_edge_covered(regions, start, finish):
    """Test complete conductor coverage of an emitted model edge."""
    return covered(
        [_model_edge_coordinates(region, start, finish) for region in regions],
        1,
        0,
        0,
        1,
    )


def model_edge_contacts(regions, start, finish):
    """Read interval and endpoint contact on one emitted model edge."""
    return any((any(model_edge_section(region, start, finish)) for region in regions))
