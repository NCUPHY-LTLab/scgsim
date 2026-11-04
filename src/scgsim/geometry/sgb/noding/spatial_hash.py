"""Uniform spatial hash used by the noding kernels.

Owns bucketing integer points by a uniform cell grid and walking the cells a
segment covers.  The hash is a performance device only: every candidate it
returns is confirmed by an exact integer predicate.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator

from scgsim.geometry.sgb.plane import Point

CellBounds = tuple[
    int, int, int, int
]  # (cx_lo, cy_lo, cx_hi, cy_hi), inclusive cell indices
MIN_CELL_SIZE = 64


def cell_size_for(extent: int, count: int) -> int:
    """Uniform-hash cell size giving roughly one object per cell, never below 64 units."""
    return max(MIN_CELL_SIZE, extent // max(4, math.isqrt(max(count, 1))))


def points_extent(points: Iterable[Point]) -> int:
    """Largest side of the bounding box of the points (at least 1)."""
    xs, ys = [], []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    if not xs:
        return 1
    return max(max(xs) - min(xs), max(ys) - min(ys), 1)


def cells_along(
    a: Point,
    b: Point,
    size: int,
    bounds: CellBounds | None = None,
) -> Iterator[tuple[int, int]]:
    """Hash cells covered by a segment, with a one-unit safety margin on the minor axis.

    ``bounds`` clips the walk, so a tile that only owns a part of a long edge
    does not traverse the rest of it.
    """
    x_bounds = None if bounds is None else (bounds[0], bounds[2])
    y_bounds = None if bounds is None else (bounds[1], bounds[3])
    if abs(b[0] - a[0]) >= abs(b[1] - a[1]):
        yield from _cells_along_major(
            (a[0], b[0]), (a[1], b[1]), size, x_bounds, y_bounds
        )
        return
    for cy, cx in _cells_along_major(
        (a[1], b[1]), (a[0], b[0]), size, y_bounds, x_bounds
    ):
        yield cx, cy


def _cells_along_major(
    major: tuple[int, int],
    minor: tuple[int, int],
    size: int,
    major_bounds: tuple[int, int] | None,
    minor_bounds: tuple[int, int] | None,
) -> Iterator[tuple[int, int]]:
    """Cells of a segment walked along its longer axis; yields (major cell, minor cell).

    ``major`` and ``minor`` are the (start, end) coordinates along each axis.
    """
    a_major, b_major = major
    a_minor, b_minor = minor
    lo, hi = (a_major, b_major) if a_major <= b_major else (b_major, a_major)
    minor_lo, minor_hi = (
        (a_minor, b_minor) if a_minor <= b_minor else (b_minor, a_minor)
    )
    d_major, d_minor = b_major - a_major, b_minor - a_minor
    first, last = lo // size, hi // size
    if major_bounds is not None:
        first, last = max(first, major_bounds[0]), min(last, major_bounds[1])
    for cell in range(first, last + 1):
        low = max(lo, cell * size)
        high = min(hi, (cell + 1) * size)
        if d_major == 0:
            span_lo, span_hi = minor_lo, minor_hi
        else:
            at_low = a_minor + (low - a_major) * d_minor / d_major
            at_high = a_minor + (high - a_major) * d_minor / d_major
            span_lo, span_hi = (
                (at_low, at_high) if at_low <= at_high else (at_high, at_low)
            )
        first_minor = math.floor((span_lo - 1) / size)
        last_minor = math.floor((span_hi + 1) / size)
        if minor_bounds is not None:
            first_minor = max(first_minor, minor_bounds[0])
            last_minor = min(last_minor, minor_bounds[1])
        for minor_cell in range(first_minor, last_minor + 1):
            yield cell, minor_cell


class HotGrid:
    """Hot pixels bucketed by a uniform hash for segment queries."""

    def __init__(self, points: Iterable[Point]) -> None:
        pixels = list(points)
        self.size = cell_size_for(points_extent(pixels), len(pixels))
        self.cells: dict[tuple[int, int], list[Point]] = defaultdict(list)
        size = self.size
        for p in pixels:
            self.cells[(p[0] // size, p[1] // size)].append(p)
        keys = list(self.cells)
        self.bounds: CellBounds = (
            (
                min(k[0] for k in keys),
                min(k[1] for k in keys),
                max(k[0] for k in keys),
                max(k[1] for k in keys),
            )
            if keys
            else (0, 0, -1, -1)
        )

    def touching(self, a: Point, b: Point) -> list[Point]:
        """Hot pixel centres whose half-open pixel the segment ab intersects."""
        ax, ay = a
        bx, by = b
        min_x, max_x = (ax, bx) if ax <= bx else (bx, ax)
        min_y, max_y = (ay, by) if ay <= by else (by, ay)
        dx, dy = bx - ax, by - ay
        rhs = abs(dx) + abs(dy)
        found: list[Point] = []
        cells = self.cells
        for cell in cells_along(a, b, self.size, self.bounds):
            bucket = cells.get(cell)
            if not bucket:
                continue
            for centre in bucket:
                cx, cy = centre
                if cx < min_x or cx > max_x or cy < min_y or cy > max_y:
                    continue
                cross_product = dx * (cy - ay) - dy * (cx - ax)
                lhs = 2 * abs(cross_product)
                if lhs < rhs or (
                    lhs == rhs and touches_only_lower_left_corner(cross_product, dx, dy)
                ):
                    found.append(centre)
        return found


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def touches_only_lower_left_corner(cross_product: int, dx: int, dy: int) -> bool:
    """A line touching a pixel only at a corner hits it iff that corner is the included one.

    The half-open pixel ``[c-1/2, c+1/2)^2`` contains its lower-left corner and
    no other corner.
    """
    if cross_product > 0:
        return _sign(dy) < 0 and -_sign(dx) < 0
    return -_sign(dy) < 0 and _sign(dx) < 0
