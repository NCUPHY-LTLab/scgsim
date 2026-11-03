"""Tile-parallel noding (design brief section 9, the parallel unit of stage S4).

Owns running the two snap-rounding kernels over a ``T x T`` tile grid and
merging the pieces so the result is *identical* to the single-tile run.

How the tiles keep identity
---------------------------
* **Tiles own pixels.**  Tile cut lines lie between grid columns and rows (at
  ``c - 1/2``), so every hot pixel centre belongs to exactly one tile.  The cut
  function depends only on the model extent and ``T``.
* **No clipping.**  A tile receives every edge whose bounding box overlaps its
  pixel range, unclipped, so it sees both sides of any crossing inside its
  pixels and every hot pixel an edge may pass through.
* **Merge.**  The pieces of an edge are joined in order along the edge and the
  crossing pixels are unioned.  No seam vertex, seam edge or seam face ever
  enters the arrangement, so there is nothing to dissolve and the merged edge
  set is exactly the single-tile edge set.

Hot-pixel sets are unioned after every round and a round's output only depends
on those sets, so the result depends neither on ``T`` nor on the worker count.

Does not own the snap-rounding algorithm itself (``noding.snap_rounding``).
"""

from __future__ import annotations

import time
from bisect import bisect_right
from collections.abc import Callable, Iterable
from concurrent.futures import Executor
from typing import Any

from scgsim.geometry.sgb.forking import quiet_fork_warning
from scgsim.geometry.sgb.noding.snap_rounding import (
    EdgeKey,
    TileBox,
    crossing_pixels_kernel,
    hits_kernel,
)
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.stopwatch import Stopwatch, elapsed_seconds

UNBOUNDED = 1 << 60


def tile_cut_lines(low: int, high: int, tiles: int) -> list[int]:
    """Tile boundaries along one axis of a model extent (unbounded at both ends)."""
    span = high - low + 1
    inner = [low + span * i // tiles for i in range(1, tiles)]
    return [-UNBOUNDED, *inner, UNBOUNDED]


def tile_index(x_cuts: list[int], y_cuts: list[int], tiles: int, p: Point) -> int:
    """Index of the tile that owns grid point ``p``."""
    return (bisect_right(x_cuts, p[0]) - 1) * tiles + (bisect_right(y_cuts, p[1]) - 1)


def tile_boxes(x_cuts: list[int], y_cuts: list[int]) -> list[TileBox]:
    """The pixel-centre range of every tile, in tile-index order."""
    return [
        (x_cuts[i], y_cuts[j], x_cuts[i + 1], y_cuts[j + 1])
        for i in range(len(x_cuts) - 1)
        for j in range(len(y_cuts) - 1)
    ]


class TiledEngine:
    """Round engine that splits work over ``tiles x tiles`` pixel ranges."""

    def __init__(
        self,
        tiles: int,
        executor: Executor | None = None,
        watch: Stopwatch | None = None,
    ) -> None:
        if tiles < 1:
            raise ValueError("tiles must be >= 1")
        self.tiles = tiles
        self.executor = executor
        self.watch = watch  # optional: kernel time is booked as a parallel stage

    # -- kernels -----------------------------------------------------------------------

    def crossing_pixels(self, keys: list[EdgeKey], changed: list[bool]) -> set[Point]:
        """Crossing pixels, computed per tile and unioned."""
        started = time.perf_counter()
        pixels = self._crossing_pixels(keys, changed)
        self._book(started)
        return pixels

    def hits(
        self,
        keys: list[EdgeKey],
        changed: list[bool],
        hot: set[Point],
        delta: set[Point],
    ) -> list[list[Point]]:
        """Hot pixel centres of every edge, computed per tile and merged."""
        started = time.perf_counter()
        hits = self._hits(keys, changed, hot, delta)
        self._book(started)
        return hits

    def _book(self, started: float) -> None:
        if self.watch is not None:
            self.watch.add("noding_kernels", elapsed_seconds(started), parallel=True)

    def _crossing_pixels(self, keys: list[EdgeKey], changed: list[bool]) -> set[Point]:
        if len(keys) < 2:
            return set()
        x_cuts, y_cuts = self._cuts(keys)
        boxes = tile_boxes(x_cuts, y_cuts)
        buckets = self._assign_edges(keys, x_cuts, y_cuts)
        jobs = [
            ([keys[i] for i in members], [changed[i] for i in members], box)
            for box, members in zip(boxes, buckets, strict=False)
            if len(members) >= 2
        ]
        pixels: set[Point] = set()
        for part in self._map(_crossing_job, jobs):
            pixels |= part
        return pixels

    def _hits(
        self,
        keys: list[EdgeKey],
        changed: list[bool],
        hot: set[Point],
        delta: set[Point],
    ) -> list[list[Point]]:
        x_cuts, y_cuts = self._cuts(keys)
        buckets = self._assign_edges(keys, x_cuts, y_cuts)
        hot_by_tile = self._points_by_tile(hot, x_cuts, y_cuts)
        delta_by_tile = self._points_by_tile(delta, x_cuts, y_cuts)
        jobs, owners = [], []
        for tile, members in enumerate(buckets):
            if not members or not hot_by_tile[tile]:
                continue
            # Unchanged edges only matter where a new hot pixel appeared; do not ship them otherwise.
            if not delta_by_tile[tile]:
                members = [i for i in members if changed[i]]
                if not members:
                    continue
            jobs.append(
                (
                    [keys[i] for i in members],
                    [changed[i] for i in members],
                    hot_by_tile[tile],
                    delta_by_tile[tile],
                )
            )
            owners.append(members)
        merged: list[list[Point]] = [[] for _ in keys]
        for members, part in zip(owners, self._map(_hits_job, jobs), strict=False):
            for local, found in part:
                merged[members[local]].extend(found)
        return merged

    # -- partition ---------------------------------------------------------------------

    def _cuts(self, keys: list[EdgeKey]) -> tuple[list[int], list[int]]:
        xs = [p[0] for key in keys for p in key]
        ys = [p[1] for key in keys for p in key]
        return (
            tile_cut_lines(min(xs), max(xs), self.tiles),
            tile_cut_lines(min(ys), max(ys), self.tiles),
        )

    def _assign_edges(
        self, keys: list[EdgeKey], x_cuts: list[int], y_cuts: list[int]
    ) -> list[list[int]]:
        """Edge indices per tile: every tile whose pixel range overlaps the edge bounding box."""
        buckets: list[list[int]] = [[] for _ in range(self.tiles * self.tiles)]
        for i, (a, b) in enumerate(keys):
            x_lo, x_hi = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
            y_lo, y_hi = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
            for tx in range(bisect_right(x_cuts, x_lo) - 1, bisect_right(x_cuts, x_hi)):
                for ty in range(
                    bisect_right(y_cuts, y_lo) - 1, bisect_right(y_cuts, y_hi)
                ):
                    buckets[tx * self.tiles + ty].append(i)
        return buckets

    def _points_by_tile(
        self, points: Iterable[Point], x_cuts: list[int], y_cuts: list[int]
    ) -> list[list[Point]]:
        by_tile: list[list[Point]] = [[] for _ in range(self.tiles * self.tiles)]
        for p in points:
            by_tile[tile_index(x_cuts, y_cuts, self.tiles, p)].append(p)
        return by_tile

    def _map(self, job: Callable[[Any], Any], jobs: list[Any]) -> list[Any]:
        """Run tile jobs (in the pool when there is one) and book the workers' compute time."""
        if self.executor is None or len(jobs) <= 1:
            timed = [_timed(job, item) for item in jobs]
        else:
            with quiet_fork_warning():
                timed = list(
                    self.executor.map(_timed, [job] * len(jobs), jobs, chunksize=1)
                )
        if self.watch is not None:
            self.watch.add_work("noding_kernels", sum(seconds for _, seconds in timed))
        return [result for result, _ in timed]


def _timed(job: Callable[[Any], Any], item: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = job(item)
    return result, elapsed_seconds(started)


def _crossing_job(args: tuple[list[EdgeKey], list[bool], TileBox]) -> set[Point]:
    keys, changed, box = args
    return crossing_pixels_kernel(keys, changed, box)


def _hits_job(
    args: tuple[list[EdgeKey], list[bool], list[Point], list[Point]],
) -> list[tuple[int, list[Point]]]:
    """Sparse result: (local edge index, pixels) only for edges that touch a hot pixel."""
    keys, changed, hot, delta = args
    return [
        (i, found)
        for i, found in enumerate(hits_kernel(keys, changed, hot, delta))
        if found
    ]
