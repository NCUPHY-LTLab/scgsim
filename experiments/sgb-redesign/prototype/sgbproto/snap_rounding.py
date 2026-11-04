"""Iterated snap rounding on the integer grid (brief section 3, stage S4 noding).

Owns turning a set of exact integer line segments into a *noded* set of edges
whose vertices are all grid points, with no two edges crossing and no edge
passing through a hot pixel it does not end in.

Algorithm (Hobby 1999; Guibas-Marimont 1998; Halperin-Packer 2002):

* A *hot pixel* is the half-open unit square ``[c-1/2, c+1/2)^2`` centred on a
  grid point ``c`` that holds a segment endpoint or a segment crossing.
* Crossings are computed exactly with integer arithmetic and rounded to the
  pixel that contains them (half rounds up).
* Every segment is re-routed through the centres of all hot pixels it touches,
  ordered along the segment. Re-routed chains can touch hot pixels the original
  segment missed, so the process is iterated until nothing changes. Each round
  only re-examines edges that changed in the previous round (``changed``) or
  that touch a newly hot pixel (``delta``).
* Every output edge keeps the set of carrier ids of the input segments that
  produced it; coincident output edges merge their carrier sets.

The two heavy kernels (crossing detection, hot-pixel hit search) are pure
functions over lists so ``tiling`` can run them per tile and union the results
without changing the answer.

Does not own face extraction (``arrangement``) or any labelling.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from sgbproto.geometry import Point

EdgeKey = tuple[Point, Point]  # canonical: a < b lexicographically


class Segment(NamedTuple):
    a: Point
    b: Point
    carriers: frozenset[str]


@dataclass(frozen=True)
class NodedEdge:
    a: Point
    b: Point
    carriers: frozenset[str]


@dataclass(frozen=True)
class SnapResult:
    edges: list[NodedEdge]
    hot_pixels: int
    rounds: int


class RoundEngine(Protocol):
    """Where the two kernels run: in-process or tile-parallel (see ``tiling``)."""

    def crossing_pixels(self, keys: list[EdgeKey], changed: list[bool]) -> set[Point]: ...

    def hits(
        self, keys: list[EdgeKey], changed: list[bool], hot: set[Point], delta: set[Point]
    ) -> list[list[Point]]: ...


# --------------------------------------------------------------------------- spatial hash


def cell_size_for(extent: int, count: int) -> int:
    """Uniform-hash cell size giving roughly one object per cell, never below 64 units."""
    return max(64, extent // max(4, math.isqrt(max(count, 1))))


def cells_along(
    ax: int, ay: int, bx: int, by: int, size: int, bounds: tuple[int, int, int, int] | None = None
) -> Iterator[tuple[int, int]]:
    """Hash cells covered by a segment, with a one-unit safety margin on the minor axis.

    ``bounds = (cx_lo, cy_lo, cx_hi, cy_hi)`` (inclusive cell indices) clips the walk, so a tile that only
    owns a part of a long edge does not traverse the rest of it.
    """
    x0, x1 = (ax, bx) if ax <= bx else (bx, ax)
    y0, y1 = (ay, by) if ay <= by else (by, ay)
    dx, dy = bx - ax, by - ay
    if x1 - x0 >= y1 - y0:
        c_lo, c_hi = x0 // size, x1 // size
        if bounds is not None:
            c_lo, c_hi = max(c_lo, bounds[0]), min(c_hi, bounds[2])
        for cx in range(c_lo, c_hi + 1):
            xl = max(x0, cx * size)
            xh = min(x1, (cx + 1) * size)
            if dx == 0:
                yl, yh = y0, y1
            else:
                ya = ay + (xl - ax) * dy / dx
                yb = ay + (xh - ax) * dy / dx
                yl, yh = (ya, yb) if ya <= yb else (yb, ya)
            r_lo, r_hi = math.floor((yl - 1) / size), math.floor((yh + 1) / size)
            if bounds is not None:
                r_lo, r_hi = max(r_lo, bounds[1]), min(r_hi, bounds[3])
            for cy in range(r_lo, r_hi + 1):
                yield cx, cy
    else:
        c_lo, c_hi = y0 // size, y1 // size
        if bounds is not None:
            c_lo, c_hi = max(c_lo, bounds[1]), min(c_hi, bounds[3])
        for cy in range(c_lo, c_hi + 1):
            yl = max(y0, cy * size)
            yh = min(y1, (cy + 1) * size)
            xa = ax + (yl - ay) * dx / dy
            xb = ax + (yh - ay) * dx / dy
            xl, xh = (xa, xb) if xa <= xb else (xb, xa)
            r_lo, r_hi = math.floor((xl - 1) / size), math.floor((xh + 1) / size)
            if bounds is not None:
                r_lo, r_hi = max(r_lo, bounds[0]), min(r_hi, bounds[2])
            for cx in range(r_lo, r_hi + 1):
                yield cx, cy


def _extent(points: Iterable[Point]) -> int:
    xs, ys = [], []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    if not xs:
        return 1
    return max(max(xs) - min(xs), max(ys) - min(ys), 1)


class HotGrid:
    """Hot pixels bucketed by a uniform hash for segment queries."""

    def __init__(self, points: Iterable[Point]) -> None:
        pts = list(points)
        self.size = cell_size_for(_extent(pts), len(pts))
        self.cells: dict[tuple[int, int], list[Point]] = defaultdict(list)
        size = self.size
        for p in pts:
            self.cells[(p[0] // size, p[1] // size)].append(p)
        keys = list(self.cells)
        self.bounds = (
            (min(k[0] for k in keys), min(k[1] for k in keys), max(k[0] for k in keys), max(k[1] for k in keys))
            if keys else (0, 0, -1, -1)
        )

    def touching(self, a: Point, b: Point) -> list[Point]:
        """Hot pixel centres whose half-open pixel the segment ab intersects."""
        ax, ay = a
        bx, by = b
        minx, maxx = (ax, bx) if ax <= bx else (bx, ax)
        miny, maxy = (ay, by) if ay <= by else (by, ay)
        dx, dy = bx - ax, by - ay
        rhs = abs(dx) + abs(dy)
        out: list[Point] = []
        cells = self.cells
        for cell in cells_along(ax, ay, bx, by, self.size, self.bounds):
            bucket = cells.get(cell)
            if not bucket:
                continue
            for c in bucket:
                cx, cy = c
                if cx < minx or cx > maxx or cy < miny or cy > maxy:
                    continue
                cr = dx * (cy - ay) - dy * (cx - ax)
                lhs = 2 * abs(cr)
                if lhs < rhs or (lhs == rhs and _corner_is_lower_left(cr, dx, dy)):
                    out.append(c)
        return out


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


def _corner_is_lower_left(cr: int, dx: int, dy: int) -> bool:
    """A line touching a pixel only at a corner hits it iff that corner is the included one.

    The half-open pixel contains its lower-left corner and no other corner.
    """
    if cr > 0:
        return _sign(dy) < 0 and -_sign(dx) < 0
    return -_sign(dy) < 0 and _sign(dx) < 0


# --------------------------------------------------------------------------- kernels


def _crossing(a1: Point, b1: Point, a2: Point, b2: Point) -> tuple[int, int, int] | None:
    """Exact proper-crossing point ``(X, Y, den)`` (point = X/den, Y/den) or ``None``."""
    d1x, d1y = b1[0] - a1[0], b1[1] - a1[1]
    d2x, d2y = b2[0] - a2[0], b2[1] - a2[1]
    den = d1x * d2y - d1y * d2x
    if den == 0:
        return None
    ex, ey = a2[0] - a1[0], a2[1] - a1[1]
    tn = ex * d2y - ey * d2x
    un = ex * d1y - ey * d1x
    if den < 0:
        den, tn, un = -den, -tn, -un
    if not (0 < tn < den and 0 < un < den):
        return None
    return a1[0] * den + d1x * tn, a1[1] * den + d1y * tn, den


def _cell(v: int, size: int) -> int:
    return max(-(1 << 40), min(1 << 40, v)) // size


def crossing_pixels_kernel(
    keys: Sequence[EdgeKey],
    changed: Sequence[bool],
    box: tuple[int, int, int, int] | None = None,
) -> set[Point]:
    """Pixels containing proper crossings of pairs with at least one changed edge.

    ``box = (x_lo, y_lo, x_hi, y_hi)`` keeps only pixels whose centre satisfies
    ``x_lo <= x < x_hi`` and ``y_lo <= y < y_hi`` (tile ownership); ``None`` keeps all.
    """
    n = len(keys)
    if n < 2:
        return set()
    size = cell_size_for(_extent(p for k in keys for p in k), n)
    bounds = None
    if box is not None:  # one cell of slack: a crossing may lie up to half a pixel outside its pixel's tile
        bounds = (_cell(box[0], size) - 1, _cell(box[1], size) - 1, _cell(box[2], size) + 1, _cell(box[3], size) + 1)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (a, b) in enumerate(keys):
        for cell in cells_along(a[0], a[1], b[0], b[1], size, bounds):
            grid[cell].append(i)
    out: set[Point] = set()
    for cell, ids in grid.items():
        if len(ids) < 2:
            continue
        for ii in range(len(ids)):
            i = ids[ii]
            a1, b1 = keys[i]
            for jj in range(ii + 1, len(ids)):
                j = ids[jj]
                if not (changed[i] or changed[j]):
                    continue
                a2, b2 = keys[j]
                hit = _crossing(a1, b1, a2, b2)
                if hit is None:
                    continue
                x_num, y_num, den = hit
                # Each pair is processed only in the hash cell that contains its crossing point.
                if (x_num // (den * size), y_num // (den * size)) != cell:
                    continue
                px = (2 * x_num + den) // (2 * den)
                py = (2 * y_num + den) // (2 * den)
                if box is not None and not (box[0] <= px < box[2] and box[1] <= py < box[3]):
                    continue
                out.add((px, py))
    return out


def hits_kernel(
    keys: Sequence[EdgeKey],
    changed: Sequence[bool],
    hot: Iterable[Point],
    delta: Iterable[Point],
) -> list[list[Point]]:
    """Hot pixel centres each edge touches: changed edges against all hot pixels,
    unchanged edges only against the pixels that became hot since they were last routed."""
    hot_grid = HotGrid(hot) if any(changed) else None
    delta_pts = list(delta)
    delta_grid = HotGrid(delta_pts) if delta_pts else None
    out: list[list[Point]] = []
    for (a, b), flag in zip(keys, changed):
        grid = hot_grid if flag else delta_grid
        out.append(grid.touching(a, b) if grid is not None else [])
    return out


def assemble_chain(a: Point, b: Point, hits: Iterable[Point]) -> tuple[Point, ...]:
    """Order hot pixel centres along segment ab (unique by construction, see module docs)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    pts = set(hits)
    pts.add(a)
    pts.add(b)
    return tuple(sorted(pts, key=lambda c: ((c[0] - a[0]) * dx + (c[1] - a[1]) * dy, c)))


class SerialEngine:
    """Runs the kernels in the calling process (optionally booking their time on a stopwatch)."""

    def __init__(self, watch=None) -> None:
        self.watch = watch

    def crossing_pixels(self, keys: list[EdgeKey], changed: list[bool]) -> set[Point]:
        t0 = time.perf_counter()
        out = crossing_pixels_kernel(keys, changed)
        self._book(t0)
        return out

    def hits(
        self, keys: list[EdgeKey], changed: list[bool], hot: set[Point], delta: set[Point]
    ) -> list[list[Point]]:
        t0 = time.perf_counter()
        out = hits_kernel(keys, changed, hot, delta)
        self._book(t0)
        return out

    def _book(self, t0: float) -> None:
        if self.watch is not None:
            dt = time.perf_counter() - t0
            self.watch.add("noding_kernels", dt, True)
            self.watch.work["noding_kernels"] += dt


# --------------------------------------------------------------------------- driver


def canonical(a: Point, b: Point) -> EdgeKey:
    return (a, b) if a < b else (b, a)


def snap_round(
    segments: Iterable[Segment], engine: RoundEngine | None = None, max_rounds: int = 500
) -> SnapResult:
    """Iterated snap rounding of integer segments; see module docstring."""
    engine = engine or SerialEngine()
    edges: dict[EdgeKey, set[str]] = {}
    for seg in segments:
        if seg.a == seg.b:
            continue
        edges.setdefault(canonical(seg.a, seg.b), set()).update(seg.carriers)
    hot: set[Point] = {p for key in edges for p in key}
    changed: set[EdgeKey] = set(edges)
    delta: set[Point] = set(hot)
    rounds = 0
    while changed or delta:
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError("snap rounding did not converge")
        keys = sorted(edges)
        flags = [k in changed for k in keys]
        new_hot = engine.crossing_pixels(keys, flags) - hot
        hot |= new_hot
        delta |= new_hot
        chain_hits = engine.hits(keys, flags, hot, delta)
        new_edges: dict[EdgeKey, set[str]] = {}
        verified: set[EdgeKey] = set()  # edges whose route was re-checked and left unchanged
        for key, hits in zip(keys, chain_hits):
            chain = assemble_chain(key[0], key[1], hits)
            if len(chain) == 2:
                verified.add(key)
            for u, v in zip(chain, chain[1:]):
                new_edges.setdefault(canonical(u, v), set()).update(edges[key])
        # An edge that is only produced as a piece of another chain has not been checked
        # against the hot pixels yet, even if an edge with the same endpoints existed before.
        changed = set(new_edges) - verified
        delta = set()
        edges = new_edges
    out = [NodedEdge(a, b, frozenset(c)) for (a, b), c in sorted(edges.items())]
    return SnapResult(out, len(hot), rounds)
