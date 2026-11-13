"""Wall time of the named stages of one build.

Owns the record of how long each named stage of a build took.  A stage's seconds exclude the stages
timed inside it (the noding driver excludes its kernels), and a stage timed twice adds up.  A timing
never influences a result.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


def elapsed_seconds(started: float) -> float:
    """Seconds since a ``time.perf_counter()`` reading."""
    return time.perf_counter() - started


@dataclass
class Stopwatch:
    """Seconds per named stage, each without the stages timed inside it."""

    seconds: dict[str, float] = field(default_factory=dict)
    _inner: list[float] = field(
        default_factory=list, repr=False
    )  # seconds of the stages timed inside each open stage

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block under ``name``."""
        started = time.perf_counter()
        self._inner.append(0.0)
        try:
            yield
        finally:
            inner = self._inner.pop()
            total = elapsed_seconds(started)
            self.seconds[name] = self.seconds.get(name, 0.0) + total - inner
            if self._inner:
                self._inner[-1] += total
