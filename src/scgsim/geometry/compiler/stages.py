"""Own compiler stage timing records and their existing diagnostic boundaries."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from scgsim.diagnostics import _stage
from scgsim.geometry.models.common import RouteLiteral


class TimingRecord(TypedDict):
    """One completed or failed compiler/native stage in timing JSON."""

    stage: str
    seconds: float
    status: Literal["failed", "done"]


class BuildTimingPayload(TypedDict):
    """The existing 00_timing sidecar and its nested stage inventories."""

    route: RouteLiteral
    total_seconds: float
    counts: dict[str, int]
    stages: list[TimingRecord]
    sidecar_bytes: dict[str, int]
    planning_stages: list[TimingRecord]
    backend_stages: list[TimingRecord]


@dataclass
class StageTimings:
    """Accumulate one pipeline's stages with its selected diagnostic prefix."""

    diagnostic_prefix: str | None = None
    records: list[TimingRecord] = field(default_factory=list)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Record the original boundary, including call argument evaluation."""
        started = time.perf_counter()
        try:
            if self.diagnostic_prefix is None:
                yield
            else:
                with _stage(self.diagnostic_prefix + stage):
                    yield
        except Exception:
            self.records.append(_timing_record(stage, started, "failed"))
            raise
        self.records.append(_timing_record(stage, started, "done"))


def _timing_record(
    stage: str, started: float, status: Literal["failed", "done"]
) -> TimingRecord:
    return {
        "stage": stage,
        "seconds": round(time.perf_counter() - started, 6),
        "status": status,
    }
