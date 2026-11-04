"""Stage timing split into serial and parallel stages (the Amdahl accounting).

Owns the record of how long each named stage of a build took and whether it ran
serially or as a per-tile fork/join.  The split is how the parallel fraction and
the speed-up limit of a build are measured.  It never influences a result.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


def elapsed_seconds(started: float) -> float:
    """Seconds since a ``time.perf_counter()`` reading."""
    return time.perf_counter() - started


@dataclass
class Stopwatch:
    """Seconds per named stage, split into serial and parallel stages."""

    serial: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    parallel: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    work: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )  # summed worker compute per parallel stage

    def add(self, name: str, seconds: float, *, parallel: bool) -> None:
        """Book wall seconds on a stage."""
        (self.parallel if parallel else self.serial)[name] += seconds

    def add_work(self, name: str, seconds: float) -> None:
        """Book compute seconds spent inside tile jobs of a parallel stage."""
        self.work[name] += seconds

    @contextmanager
    def serial_stage(self, name: str) -> Iterator[None]:
        """Time a block as a serial stage."""
        started = time.perf_counter()
        yield
        self.add(name, elapsed_seconds(started), parallel=False)

    def totals(self) -> dict[str, float]:
        """Wall split plus the Amdahl estimate.

        ``work`` is compute inside tile jobs; the rest of a parallel stage's wall
        time is marshalling (partitioning, pickling, merging) that stays serial.
        """
        serial = sum(self.serial.values())
        parallel = sum(self.parallel.values())
        work = sum(self.work.values())
        overhead = max(0.0, parallel - work) if work else 0.0
        return {
            "serial_stages_s": serial,
            "parallel_stages_wall_s": parallel,
            "parallel_stages_work_s": work,
            "marshalling_s": overhead,
            "amdahl_parallel_fraction": work / (serial + max(parallel, work))
            if serial + parallel
            else 0.0,
            "amdahl_speedup_limit": (serial + max(parallel, work)) / (serial + overhead)
            if serial + overhead
            else 0.0,
        }
