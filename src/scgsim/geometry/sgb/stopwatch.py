"""Wall-time readings of the build steps.

Owns the one way a step reads how long it took; a timing never influences a result.
"""

from __future__ import annotations

import time


def elapsed_seconds(started: float) -> float:
    """Seconds since a ``time.perf_counter()`` reading."""
    return time.perf_counter() - started
