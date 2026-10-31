"""Wall-clock timing helper shared by the AEDT runtime modules.

Owns only the rounding convention for recorded durations. It does not own
which phases are timed or where the numbers are stored.
"""

from __future__ import annotations

import time


def elapsed_seconds(started: float) -> float:
    """Seconds since ``started`` (a ``time.perf_counter()`` value), to microseconds."""
    return round(time.perf_counter() - started, 6)
