"""Convert junction polygon regions to and from the grid's native Boolean."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from itertools import pairwise

import gdstk

from scgsim.geometry.grid.grid import PolygonRegion, polygon_boolean
from scgsim.geometry.grid.planar_loops import (
    cancel_reversed_planar_edges,
    simple_planar_loops_from_edges,
)
from scgsim.geometry.junctions.core import PRECISION_UM
from scgsim.geometry.junctions.exact_geometry import inside_ring, on, region_edges


def build_polygon_region(polygons):
    """Construct native polygon regions while subtracting their authored holes."""
    result = []
    for rings in polygons:
        exterior = gdstk.Polygon(
            [(x * PRECISION_UM, y * PRECISION_UM) for x, y in rings[0]]
        )
        holes = [
            gdstk.Polygon([(x * PRECISION_UM, y * PRECISION_UM) for x, y in ring])
            for ring in rings[1:]
        ]
        result.extend(boolean([exterior], holes, "not"))
    return result


def boolean(left, right, operation):
    """Apply the existing integer-coordinate native junction Boolean operation."""
    return list(polygon_boolean(left, right, operation, precision=PRECISION_UM) or ())


def read_boolean_rings(region):
    """Recover connected outer and interior rings from the native Boolean output."""
    result = []
    for polygon in region:
        if isinstance(polygon, PolygonRegion):
            result.append(
                tuple(
                    (
                        tuple(
                            (
                                (round(x / PRECISION_UM), round(y / PRECISION_UM))
                                for x, y in ring
                            )
                        )
                        for ring in (polygon.exterior, *polygon.holes)
                    )
                )
            )
            continue
        points = tuple(
            (
                (round(x / PRECISION_UM), round(y / PRECISION_UM))
                for x, y in polygon.points
            )
        )
        ordered = sorted(set(points))
        xs = [point[0] for point in ordered]
        edges = []
        for a, b in region_edges((points,)):
            candidates = ordered[
                bisect_left(xs, min(a[0], b[0])) : bisect_right(xs, max(a[0], b[0]))
            ]
            on_edge = sorted(
                (p for p in candidates if on(p, a, b)),
                key=lambda p: (
                    (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
                ),
            )
            edges.extend(pairwise(on_edge))
        loops = simple_planar_loops_from_edges(cancel_reversed_planar_edges(edges))

        def area(ring):
            return abs(
                sum((a[0] * b[1] - b[0] * a[1] for a, b in region_edges((ring,))))
            )

        outer = max(loops, key=area)
        holes = tuple((ring for ring in loops if ring is not outer))
        if any((inside_ring(hole[0], outer) != 1 for hole in holes)):
            raise ValueError(
                "junction residual has point-connected or non-interior components"
            )
        result.append((outer, *holes))
    return result


def rectangle(x0, y0, x1, y1):
    """Construct one axis-aligned rectangle as a polygon region."""
    return ((((x0, y0), (x1, y0), (x1, y1), (x0, y1)),),)
