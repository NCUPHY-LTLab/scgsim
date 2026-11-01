"""Convert labelled overlay coordinates to atomic polygon rings and vertex indices."""

from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType

from scgsim.diagnostics import _stage


def polygon_components(geometry):
    """Return only polygon components from a GEOS polygon or collection."""
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "Polygon":
        return (geometry,)
    if geometry.geom_type not in {"MultiPolygon", "GeometryCollection"}:
        return ()
    return tuple(p for g in geometry.geoms for p in polygon_components(g))


def _ring(coords):
    points = tuple(tuple(p) for p in coords)[:-1]
    start = min(range(len(points)), key=points.__getitem__)
    return points[start:] + points[:start]


@dataclass(frozen=True)
class RegionVertexIndex:
    """Coordinate indices and exact keys shared during atomic ring reconstruction."""

    grid_size: float | None
    by_x: tuple
    by_y: tuple
    xs: tuple
    ys: tuple
    coordinates: Mapping
    point_keys: Mapping


def region_key(point, grid_size):
    """Map a point to its exact lattice key or ungridded rational coordinates."""

    return (
        tuple(round(value / grid_size) for value in point)
        if grid_size is not None
        else tuple(Fraction(value) for value in point)
    )


def _canonical_grid_point(point, grid_size):
    if grid_size is None:
        return tuple(point)

    # Decimal spelling is the declared layout step; avoid introducing a second
    # binary multiplication spelling for an otherwise identical lattice node.
    step = Decimal(str(grid_size))
    return tuple(float(value * step) for value in region_key(point, grid_size))


def region_vertex_index(vertices, grid_size):
    """Index realized vertices once for exact segment-incidence queries."""
    # Only realized lattice nodes share one key-to-coordinate identity. Parent
    # planes without a layout grid retain their original coordinates/occurrences.

    coordinates = {}
    point_keys = {}
    if grid_size is not None:
        step = Decimal(str(grid_size))
        for point in vertices:
            key = region_key(point, grid_size)
            if key not in coordinates:
                canonical = tuple(float(value * step) for value in key)
                coordinates[key] = canonical
                point_keys[canonical] = key
        vertices = tuple(coordinates.values())
    else:
        point_keys = {tuple(point): region_key(point, None) for point in vertices}
    rows = tuple(
        (index, point, point_keys[tuple(point)]) for index, point in enumerate(vertices)
    )
    by_x = tuple(sorted(rows, key=lambda row: row[1][0]))
    by_y = tuple(sorted(rows, key=lambda row: row[1][1]))
    return RegionVertexIndex(
        grid_size,
        by_x,
        by_y,
        tuple(row[1][0] for row in by_x),
        tuple(row[1][1] for row in by_y),
        MappingProxyType(coordinates),
        MappingProxyType(point_keys),
    )


def regions(geometry, vertices=(), grid_size=None):
    """Reconstruct atomic polygon rings using a shared realized-vertex index."""
    with _stage("geometry.grid.regions"):
        return regions_indexed(geometry, region_vertex_index(vertices, grid_size))


def regions_indexed(geometry, vertex_index):
    """Insert realized collinear vertices into each polygon boundary ring."""
    by_x, by_y = vertex_index.by_x, vertex_index.by_y
    xs, ys = vertex_index.xs, vertex_index.ys
    extra_keys = {}

    def point_key(point):
        point = tuple(point)
        key = vertex_index.point_keys.get(point)
        if key is None:
            key = extra_keys.get(point)
            if key is None:
                key = region_key(point, vertex_index.grid_size)
                extra_keys[point] = key
        return key

    def atomic(coords):
        if vertex_index.grid_size is not None:
            points = []
            for point in coords:
                key = point_key(point)
                canonical = vertex_index.coordinates.get(key)
                points.append(
                    _canonical_grid_point(point, vertex_index.grid_size)
                    if canonical is None
                    else canonical
                )
            points = tuple(points)
            coords = tuple(
                point
                for index, point in enumerate(points)
                if index == 0 or point != points[index - 1]
            )
        ring = _ring(coords)
        if not by_x:
            return ring
        ring_keys = tuple(point_key(point) for point in ring)
        output = []
        for index, (a, b) in enumerate(zip(ring, (*ring[1:], ring[0]), strict=False)):
            output.append(a)
            ak, bk = ring_keys[index], ring_keys[(index + 1) % len(ring)]
            xmin, xmax = min(a[0], b[0]), max(a[0], b[0])
            ymin, ymax = min(a[1], b[1]), max(a[1], b[1])
            xlo, xhi = bisect_left(xs, xmin), bisect_right(xs, xmax)
            ylo, yhi = bisect_left(ys, ymin), bisect_right(ys, ymax)
            candidates = by_x[xlo:xhi] if xhi - xlo <= yhi - ylo else by_y[ylo:yhi]
            along = [
                point
                for _, point, point_key in sorted(candidates, key=lambda row: row[0])
                if point != a
                and point != b
                and (point_key[0] - ak[0]) * (bk[1] - ak[1])
                == (point_key[1] - ak[1]) * (bk[0] - ak[0])
                and xmin <= point[0] <= xmax
                and ymin <= point[1] <= ymax
            ]
            axis = 0 if a[0] != b[0] else 1
            output.extend(
                sorted(along, key=lambda p: p[axis], reverse=b[axis] < a[axis])
            )
        return tuple(output)

    return tuple(
        sorted(
            (
                atomic(p.exterior.coords),
                tuple(sorted(atomic(h.coords) for h in p.interiors)),
            )
            for p in polygon_components(geometry)
        )
    )
