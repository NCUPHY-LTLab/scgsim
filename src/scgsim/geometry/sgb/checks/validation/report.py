"""The result of the structural validation (design brief section 11, stage S9)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.models.refusals import internal
from scgsim.geometry.sgb.errors import ValidationError
from scgsim.geometry.sgb.stopwatch import elapsed_seconds


@dataclass
class Check:
    """One named invariant, its outcome and a measured detail; a ``warning`` never stops a build."""

    name: str
    passed: bool
    detail: str = ""
    warning: bool = False


@dataclass
class ValidationReport:
    """Every check of one validation run with its wall time."""

    checks: list[Check] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # check function -> seconds

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        """Book the wall time of a block under ``name``."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + elapsed_seconds(started)

    def add(
        self, name: str, passed: bool, detail: str = "", *, warning: bool = False
    ) -> None:
        """Record one check."""
        self.checks.append(Check(name, passed, detail, warning))

    @property
    def failures(self) -> list[Check]:
        """The failed checks that are not warnings."""
        return [c for c in self.checks if not c.passed and not c.warning]

    def raise_if_failed(self) -> None:
        """Fail closed: any failed invariant stops the build."""
        if self.failures:
            lines = "; ".join(f"{c.name}: {c.detail}" for c in self.failures)
            raise ValidationError(
                internal(f"{len(self.failures)} invariant(s) failed: {lines}")
            )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready form for the validation report file."""
        return {
            "passed": not self.failures,
            "timings_s": {k: round(v, 5) for k, v in self.timings.items()},
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "warning": c.warning,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }
