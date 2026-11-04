"""Tile-parallel noding (brief section 9, parallel unit of stage S4).

Owns running the two snap-rounding kernels over a ``T x T`` tile grid and
merging the pieces so the result is *identical* to the single-tile run.

How the brief's halo / seam / dissolve maps onto this implementation
-------------------------------------------------------------------
* **Tiles own pixels.** Tile cut lines lie between grid columns/rows (at
  ``c - 1/2``), so every hot pixel centre belongs to exactly one tile. This is
  the deterministic seam function: it depends only on the model extent and ``T``.
* **Halo.** A tile receives every edge whose bounding box overlaps its pixel
  range (edges are *not* clipped), so it sees both sides of any crossing inside
  its pixels and every hot pixel an edge may pass through. The "halo" is thus
  the full edge, not a band of a few cells.
* **Seam splitting.** A chain that runs through several tiles is computed as
  one piece per tile (the hot pixels of that tile along the edge), i.e. the edge
  is split at the seams.
* **Dissolution.** Merging joins the pieces of an edge in order along the edge
  and unions crossing pixels; no seam vertex, seam edge or seam face ever
  survives into the arrangement, so there is nothing left to dissolve and the
  merged edge set is exactly the single-tile edge set.

Because hot-pixel sets are unioned after every round and a round's output only
depends on those sets, the result does not depend on ``T`` or on the worker count.

Does not own the snap-rounding algorithm itself (``snap_rounding``).
"""

from __future__ import annotations

import time
from bisect import bisect_right
from collections.abc import Iterable
from concurrent.futures import Executor

from sgbproto.geometry import Point
from sgbproto.snap_rounding import EdgeKey, crossing_pixels_kernel, hits_kernel

_BIG = 1 << 60
Box = tuple[int, int, int, int]  # x_lo, y_lo, x_hi, y_hi; pixel centres with lo <= c < hi


class TiledEngine:
    """Round engine that splits work over ``tiles x tiles`` pixel ranges."""

    def __init__(self, tiles: int, executor: Executor | None = None, watch=None) -> None:
        if tiles < 1:
            raise ValueError("tiles must be >= 1")
        self.tiles = tiles
        self.executor = executor
        self.watch = watch  # optional Stopwatch: kernel time is booked as a parallel stage

    # -- partition ---------------------------------------------------------------------

    def _cuts(self, keys: list[EdgeKey]) -> tuple[list[int], list[int]]:
        xs = [p[0] for k in keys for p in k]
        ys = [p[1] for k in keys for p in k]
        t = self.tiles

        def cuts(lo: int, hi: int) -> list[int]:
            span = hi - lo + 1
            inner = [lo + span * i // t for i in range(1, t)]
            return [-_BIG, *inner, _BIG]

        return cuts(min(xs), max(xs)), cuts(min(ys), max(ys))

    def _boxes(self, xc: list[int], yc: list[int]) -> list[Box]:
        return [(xc[i], yc[j], xc[i + 1], yc[j + 1]) for i in range(self.tiles) for j in range(self.tiles)]

    def _assign_edges(self, keys: list[EdgeKey], xc: list[int], yc: list[int]) -> list[list[int]]:
        """Edge indices per tile: every tile whose pixel range overlaps the edge bounding box."""
        buckets: list[list[int]] = [[] for _ in range(self.tiles * self.tiles)]
        for i, (a, b) in enumerate(keys):
            x0, x1 = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
            y0, y1 = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
            for tx in range(bisect_right(xc, x0) - 1, bisect_right(xc, x1)):
                for ty in range(bisect_right(yc, y0) - 1, bisect_right(yc, y1)):
                    buckets[tx * self.tiles + ty].append(i)
        return buckets

    def _map(self, fn, jobs: list) -> list:
        """Run tile jobs (in the pool when there is one) and book the workers' compute time."""
        if self.executor is None or len(jobs) <= 1:
            pairs = [_timed(fn, j) for j in jobs]
        else:
            pairs = list(self.executor.map(_timed, [fn] * len(jobs), jobs, chunksize=1))
        if self.watch is not None:
            self.watch.work["noding_kernels"] += sum(dt for _, dt in pairs)
        return [out for out, _ in pairs]

    # -- kernels -----------------------------------------------------------------------

    def crossing_pixels(self, keys: list[EdgeKey], changed: list[bool]) -> set[Point]:
        t0 = time.perf_counter()
        out = self._crossing_pixels(keys, changed)
        self._book("noding_kernels", t0)
        return out

    def hits(
        self, keys: list[EdgeKey], changed: list[bool], hot: set[Point], delta: set[Point]
    ) -> list[list[Point]]:
        t0 = time.perf_counter()
        out = self._hits(keys, changed, hot, delta)
        self._book("noding_kernels", t0)
        return out

    def _book(self, name: str, t0: float) -> None:
        if self.watch is not None:
            self.watch.add(name, time.perf_counter() - t0, True)

    def _crossing_pixels(self, keys: list[EdgeKey], changed: list[bool]) -> set[Point]:
        if len(keys) < 2:
            return set()
        xc, yc = self._cuts(keys)
        boxes = self._boxes(xc, yc)
        buckets = self._assign_edges(keys, xc, yc)
        jobs = []
        for box, idx in zip(boxes, buckets):
            if len(idx) >= 2:
                jobs.append(([keys[i] for i in idx], [changed[i] for i in idx], box))
        out: set[Point] = set()
        for part in self._map(_crossing_job, jobs):
            out |= part
        return out

    def _hits(
        self, keys: list[EdgeKey], changed: list[bool], hot: set[Point], delta: set[Point]
    ) -> list[list[Point]]:
        xc, yc = self._cuts(keys)
        boxes = self._boxes(xc, yc)
        buckets = self._assign_edges(keys, xc, yc)
        hot_by_tile = self._points_by_tile(hot, xc, yc)
        delta_by_tile = self._points_by_tile(delta, xc, yc)
        jobs, owners = [], []
        for t, idx in enumerate(buckets):
            if not idx or not hot_by_tile[t]:
                continue
            # Unchanged edges only matter where a new hot pixel appeared; do not ship them otherwise.
            if not delta_by_tile[t]:
                idx = [i for i in idx if changed[i]]
                if not idx:
                    continue
            jobs.append(([keys[i] for i in idx], [changed[i] for i in idx], hot_by_tile[t], delta_by_tile[t]))
            owners.append(idx)
        merged: list[list[Point]] = [[] for _ in keys]
        for idx, part in zip(owners, self._map(_hits_job, jobs)):
            for i, found in part:
                merged[idx[i]].extend(found)
        return merged

    def _points_by_tile(self, points: Iterable[Point], xc: list[int], yc: list[int]) -> list[list[Point]]:
        by_tile: list[list[Point]] = [[] for _ in range(self.tiles * self.tiles)]
        for p in points:
            tx = bisect_right(xc, p[0]) - 1
            ty = bisect_right(yc, p[1]) - 1
            by_tile[tx * self.tiles + ty].append(p)
        return by_tile


def _timed(fn, job):
    t0 = time.perf_counter()
    out = fn(job)
    return out, time.perf_counter() - t0


def _crossing_job(args) -> set[Point]:
    keys, changed, box = args
    return crossing_pixels_kernel(keys, changed, box)


def _hits_job(args) -> list[tuple[int, list[Point]]]:
    """Sparse result: (local edge index, pixels) only for edges that touch a hot pixel."""
    keys, changed, hot, delta = args
    return [(i, found) for i, found in enumerate(hits_kernel(keys, changed, hot, delta)) if found]
