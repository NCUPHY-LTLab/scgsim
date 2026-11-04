"""Per-tile parallel runtime (brief section 9).

Owns the one way every parallel stage runs: items (faces, edges, vertices,
volumes, wall keys) are *owned* by the tile that contains their anchor point;
a stage function ``fn(shared, item_indices) -> list`` is applied once per
non-empty tile and the results are put back in item order.

* ``shared`` is the stage's read-only input. With ``workers > 1`` it is
  inherited by forked workers (copy-on-write), so inputs are never pickled;
  only the compact results travel back.
* With ``workers == 1`` the same function runs in-process, tile by tile, so the
  serial and parallel paths are the same code and give identical results.
* Because every item is processed by a pure function and the merge is by item
  index, the output depends neither on the tile count nor on the worker count.

The runtime also keeps a persistent process pool for the noding kernels
(``tiling.TiledEngine``) and a stopwatch that records every stage as parallel
or serial, which is how the Amdahl split is measured.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import warnings
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field

from sgbproto.geometry import Point
from sgbproto.snap_rounding import RoundEngine, SerialEngine
from sgbproto.tiling import TiledEngine

# Workers are forked from a process that already runs pool-management threads; the stage functions
# are pure and never touch those threads' locks, so the interpreter's generic fork warning is noise here.
warnings.filterwarnings("ignore", message=".*multi-threaded.*fork.*", category=DeprecationWarning)
_BIG = 1 << 60
_SHARED: object = None


def _run_tile(args):
    fn, items = args
    t0 = time.perf_counter()
    out = fn(_SHARED, items)
    return out, time.perf_counter() - t0


@dataclass
class Stopwatch:
    """Seconds per named stage, split into serial and parallel stages."""

    serial: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    parallel: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    work: dict[str, float] = field(default_factory=lambda: defaultdict(float))  # summed worker compute per parallel stage

    def add(self, name: str, seconds: float, parallel: bool) -> None:
        (self.parallel if parallel else self.serial)[name] += seconds

    def totals(self) -> dict[str, float]:
        """Wall split plus the Amdahl estimate: ``work`` is compute inside tile jobs, the rest of a
        parallel stage's wall time is marshalling (partitioning, pickling, merging) that stays serial."""
        s, p, w = sum(self.serial.values()), sum(self.parallel.values()), sum(self.work.values())
        overhead = max(0.0, p - w) if w else 0.0
        return {
            "serial_stages_s": s,
            "parallel_stages_wall_s": p,
            "parallel_stages_work_s": w,
            "marshalling_s": overhead,
            "amdahl_parallel_fraction": w / (s + max(p, w)) if s + p else 0.0,
            "amdahl_speedup_limit": (s + max(p, w)) / (s + overhead) if s + overhead else 0.0,
        }


class Runtime:
    """Tile grid, worker processes and stage timing for one build."""

    def __init__(self, tiles: int = 1, workers: int = 1, bbox: tuple[int, int, int, int] | None = None) -> None:
        self.tiles = max(1, tiles)
        self.workers = max(1, workers)
        self.watch = Stopwatch()
        self.keep_raw = False  # keep each layer's pre-dissolution arrangement (for figures and teaching examples)
        self._executor: ProcessPoolExecutor | None = None
        self._xc: list[int] = [-_BIG, _BIG]
        self._yc: list[int] = [-_BIG, _BIG]
        if bbox is not None:
            self.set_extent(bbox)

    # -- lifecycle ---------------------------------------------------------------------

    def __enter__(self) -> Runtime:
        if self.workers > 1 and self.tiles > 1:
            self._executor = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("fork"))
        return self

    def __exit__(self, *exc) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def engine(self) -> RoundEngine | None:
        """Round engine for snap rounding (``None`` = plain serial kernels)."""
        if self.tiles == 1:
            return SerialEngine(self.watch)
        return TiledEngine(self.tiles, self._executor, self.watch)

    # -- tiles -------------------------------------------------------------------------

    def set_extent(self, bbox: tuple[int, int, int, int]) -> None:
        t = self.tiles

        def cuts(lo: int, hi: int) -> list[int]:
            span = hi - lo + 1
            return [-_BIG, *[lo + span * i // t for i in range(1, t)], _BIG]

        self._xc, self._yc = cuts(bbox[0], bbox[2]), cuts(bbox[1], bbox[3])

    def owner(self, p: Point) -> int:
        """Tile index owning grid point ``p`` (cut lines lie between pixel columns/rows)."""
        return (bisect_right(self._xc, p[0]) - 1) * self.tiles + (bisect_right(self._yc, p[1]) - 1)

    # -- stages ------------------------------------------------------------------------

    @contextmanager
    def serial(self, name: str):
        t0 = time.perf_counter()
        yield
        self.watch.add(name, time.perf_counter() - t0, False)

    @contextmanager
    def noding(self, name: str):
        """Book a snap-rounding call: kernel time was booked in parallel by the engine; the rest is serial driver."""
        before = self.watch.parallel.get("noding_kernels", 0.0)
        t0 = time.perf_counter()
        yield
        kernel = self.watch.parallel.get("noding_kernels", 0.0) - before
        self.watch.add(name, max(0.0, time.perf_counter() - t0 - kernel), False)

    def run(self, name: str, fn, shared, owners: list[int]) -> list:
        """Apply ``fn(shared, indices)`` once per owning tile; results come back in item order."""
        t0 = time.perf_counter()
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, tile in enumerate(owners):
            buckets[tile].append(i)
        jobs = [(fn, buckets[t]) for t in sorted(buckets)]
        if self.workers > 1 and len(jobs) > 1:
            global _SHARED
            _SHARED = shared
            try:
                with ProcessPoolExecutor(min(self.workers, len(jobs)), mp_context=mp.get_context("fork")) as pool:
                    pairs = list(pool.map(_run_tile, jobs, chunksize=1))
            finally:
                _SHARED = None
        else:
            pairs = []
            for _, items in jobs:
                t1 = time.perf_counter()
                pairs.append((fn(shared, items), time.perf_counter() - t1))
        outs = [out for out, _ in pairs]
        self.watch.work[name] += sum(dt for _, dt in pairs)
        results: list = [None] * len(owners)
        for (_, items), out in zip(jobs, outs):
            for i, value in zip(items, out):
                results[i] = value
        self.watch.add(name, time.perf_counter() - t0, True)
        return results
