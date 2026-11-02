"""Measure native lowering stages with the established timing and failure fields."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from scgsim.geometry.construction.native_types import NativeTimingRecord


@contextmanager
def debug_stage(
    enabled: bool,
    label: str,
    timings: list[NativeTimingRecord],
) -> Iterator[None]:
    """Measure the existing native stage and retain its completion or failure status."""
    started = time.perf_counter()
    status = "done"
    if enabled:
        print(f"[sgb:gmsh] start {label}", flush=True)
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        elapsed = time.perf_counter() - started
        timings.append(
            {
                "stage": label,
                "seconds": round(elapsed, 6),
                "status": status,
            }
        )
        if enabled:
            print(f"[sgb:gmsh] {status} {label} in {elapsed:.3f}s", flush=True)
