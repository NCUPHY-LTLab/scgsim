"""Per-tile parallel runtime (design brief section 9).

Owns the one way every parallel stage runs: items (faces, edges, vertices) are
*owned* by the tile that contains their anchor point; a stage function
``fn(shared, item_indices) -> list`` is applied once per non-empty tile and the
results are put back in item order.

* ``shared`` is the stage's read-only input.  With ``workers > 1`` it is
  inherited by forked workers (copy-on-write), so inputs are never pickled; only
  the compact results travel back.
* With ``workers == 1`` the same function runs in-process, tile by tile, so the
  serial and parallel paths are the same code and give identical results.
* Because every item is processed by a pure function and the merge is by item
  index, the output depends neither on the tile count nor on the worker count.

The runtime also keeps a persistent process pool for the noding kernels
(``noding.tiling.TiledEngine``) and a stopwatch that records every stage as
parallel or serial, which is how the Amdahl split is measured.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any

from scgsim.geometry.sgb.forking import fork_context, quiet_fork_warning
from scgsim.geometry.sgb.noding.snap_rounding import RoundEngine, SerialEngine
from scgsim.geometry.sgb.noding.tiling import TiledEngine, tile_cut_lines, tile_index
from scgsim.geometry.sgb.plane import Point
from scgsim.geometry.sgb.stopwatch import Stopwatch, elapsed_seconds

BoundingBox = tuple[int, int, int, int]
StageFunction = Callable[[Any, list[int]], list[Any]]

_SHARED: Any = None  # the stage input a forked worker inherits (set by ``Runtime.run`` just before forking)


def _run_tile(args: tuple[StageFunction, list[int]]) -> tuple[list[Any], float]:
    fn, items = args
    started = time.perf_counter()
    result = fn(_SHARED, items)
    return result, elapsed_seconds(started)


class Runtime:
    """Tile grid, worker processes and stage timing for one build."""

    def __init__(
        self, tiles: int = 1, workers: int = 1, bbox: BoundingBox | None = None
    ) -> None:
        self.tiles = max(1, tiles)
        self.workers = max(1, workers) if fork_context() is not None else 1
        self.watch = Stopwatch()
        self.keep_raw = False  # keep each layer's pre-dissolution arrangement (for figures and teaching examples)
        self._executor: ProcessPoolExecutor | None = None
        self._x_cuts = [-(1 << 60), 1 << 60]
        self._y_cuts = [-(1 << 60), 1 << 60]
        if bbox is not None:
            self.set_extent(bbox)

    # -- lifecycle ---------------------------------------------------------------------

    def __enter__(self) -> Runtime:
        context = fork_context()
        if self.workers > 1 and self.tiles > 1 and context is not None:
            with quiet_fork_warning():
                self._executor = ProcessPoolExecutor(self.workers, mp_context=context)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def engine(self) -> RoundEngine:
        """Round engine for snap rounding (plain serial kernels for one tile)."""
        if self.tiles == 1:
            return SerialEngine(self.watch)
        return TiledEngine(self.tiles, self._executor, self.watch)

    # -- tiles -------------------------------------------------------------------------

    def set_extent(self, bbox: BoundingBox) -> None:
        """Cut the model extent into the tile grid (cut lines between pixel columns and rows)."""
        self._x_cuts = tile_cut_lines(bbox[0], bbox[2], self.tiles)
        self._y_cuts = tile_cut_lines(bbox[1], bbox[3], self.tiles)

    def owner(self, p: Point) -> int:
        """Tile index owning grid point ``p``."""
        return tile_index(self._x_cuts, self._y_cuts, self.tiles, p)

    # -- stages ------------------------------------------------------------------------

    @contextmanager
    def serial(self, name: str) -> Iterator[None]:
        """Time a block as a serial stage."""
        with self.watch.serial_stage(name):
            yield

    @contextmanager
    def noding(self, name: str) -> Iterator[None]:
        """Book a snap-rounding call: kernel time was booked in parallel by the engine; the rest is serial driver."""
        before = self.watch.parallel.get("noding_kernels", 0.0)
        started = time.perf_counter()
        yield
        kernel = self.watch.parallel.get("noding_kernels", 0.0) - before
        self.watch.add(
            name, max(0.0, elapsed_seconds(started) - kernel), parallel=False
        )

    def run(
        self, name: str, fn: StageFunction, shared: Any, owners: Sequence[int]
    ) -> list[Any]:
        """Apply ``fn(shared, indices)`` once per owning tile; results come back in item order."""
        started = time.perf_counter()
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, tile in enumerate(owners):
            buckets[tile].append(i)
        jobs = [(fn, buckets[t]) for t in sorted(buckets)]
        timed = self._execute(fn, shared, jobs)
        self.watch.add_work(name, sum(seconds for _, seconds in timed))
        results: list[Any] = [None] * len(owners)
        for (_, items), (output, _) in zip(jobs, timed, strict=False):
            for i, value in zip(items, output, strict=False):
                results[i] = value
        self.watch.add(name, elapsed_seconds(started), parallel=True)
        return results

    def _execute(
        self,
        fn: StageFunction,
        shared: Any,
        jobs: list[tuple[StageFunction, list[int]]],
    ) -> list[tuple[list[Any], float]]:
        context = fork_context()
        if self.workers > 1 and len(jobs) > 1 and context is not None:
            global _SHARED  # noqa: PLW0603 - forked workers inherit the stage input; nothing is pickled
            _SHARED = shared
            try:
                with (
                    quiet_fork_warning(),
                    ProcessPoolExecutor(
                        min(self.workers, len(jobs)), mp_context=context
                    ) as pool,
                ):
                    return list(pool.map(_run_tile, jobs, chunksize=1))
            finally:
                _SHARED = None
        timed = []
        for _, items in jobs:
            started = time.perf_counter()
            timed.append((fn(shared, items), elapsed_seconds(started)))
        return timed
