"""Shared HFSS controls and pure request decoders."""

from __future__ import annotations

from dataclasses import (
    dataclass,
)
from numbers import (
    Number,
)
from typing import (
    Any,
)

from scgsim.aedt.specs.common import (
    POINT_COUNT,
    adaptive_controls,
    finite_number,
    text_value,
)


@dataclass(frozen=True)
class FrequencySweepSpec:
    """One HFSS Fast sweep with a caller-authored output count."""

    start_ghz: float
    stop_ghz: float
    points: int = POINT_COUNT

    def __post_init__(self) -> None:
        start = finite_number(self.start_ghz, "start_ghz")
        stop = finite_number(self.stop_ghz, "stop_ghz")
        if start <= 0 or stop <= start:
            raise ValueError("frequency sweep requires 0 < start_ghz < stop_ghz")
        if isinstance(self.points, bool) or not isinstance(self.points, Number):
            raise ValueError("points must be an exact whole-number numeric value")
        try:
            native_points = int(self.points)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "points must be an exact whole-number numeric value"
            ) from exc
        if self.points != native_points:
            raise ValueError("points must be an exact whole-number numeric value")
        object.__setattr__(self, "start_ghz", start)
        object.__setattr__(self, "stop_ghz", stop)

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical FrequencySweepSpec payload."""
        return {
            "start_ghz": self.start_ghz,
            "stop_ghz": self.stop_ghz,
            # Keep ``points`` as authored so the payload hash does not change.
            "points": self.points,
        }


@dataclass(frozen=True)
class HfssRunControl:
    """One Driven setup, adaptive convergence policy, and Fast sweep."""

    setup_name: str
    sweep_name: str
    sweep: FrequencySweepSpec
    maximum_passes: int = 99
    minimum_passes: int = 1
    minimum_converged_passes: int = 1
    percent_refinement: float = 30.0
    maximum_delta_s: float = 0.02

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "setup_name", text_value(self.setup_name, "setup_name")
        )
        object.__setattr__(
            self, "sweep_name", text_value(self.sweep_name, "sweep_name")
        )
        refinement = adaptive_controls(
            self.maximum_passes,
            self.minimum_passes,
            self.minimum_converged_passes,
            self.percent_refinement,
        )
        delta = finite_number(self.maximum_delta_s, "maximum_delta_s")
        if delta <= 0:
            raise ValueError("maximum_delta_s must be > 0")
        object.__setattr__(self, "percent_refinement", refinement)
        object.__setattr__(self, "maximum_delta_s", delta)

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical HfssRunControl payload."""
        return {
            "setup_name": self.setup_name,
            "sweep_name": self.sweep_name,
            "sweep": self.sweep.to_payload(),
            "maximum_passes": self.maximum_passes,
            "minimum_passes": self.minimum_passes,
            "minimum_converged_passes": self.minimum_converged_passes,
            "percent_refinement": self.percent_refinement,
            "maximum_delta_s": self.maximum_delta_s,
        }


@dataclass(frozen=True)
class EigenmodeRunControl:
    """One explicit HFSS Eigenmode adaptive setup."""

    setup_name: str
    minimum_frequency_ghz: float
    num_modes: int
    maximum_passes: int
    maximum_delta_frequency_percent: float
    minimum_passes: int = 1
    minimum_converged_passes: int = 1
    percent_refinement: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "setup_name", text_value(self.setup_name, "setup_name")
        )
        minimum = finite_number(self.minimum_frequency_ghz, "minimum_frequency_ghz")
        delta = finite_number(
            self.maximum_delta_frequency_percent,
            "maximum_delta_frequency_percent",
        )
        if minimum <= 0 or delta <= 0:
            raise ValueError("Eigenmode frequency and convergence percent must be > 0")
        if not isinstance(self.num_modes, int) or self.num_modes <= 0:
            raise ValueError("num_modes must be a positive integer")
        refinement = adaptive_controls(
            self.maximum_passes,
            self.minimum_passes,
            self.minimum_converged_passes,
            self.percent_refinement,
        )
        object.__setattr__(self, "minimum_frequency_ghz", minimum)
        object.__setattr__(self, "maximum_delta_frequency_percent", delta)
        object.__setattr__(self, "percent_refinement", refinement)

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical EigenmodeRunControl payload."""
        return {
            "setup_name": self.setup_name,
            "minimum_frequency_ghz": self.minimum_frequency_ghz,
            "num_modes": self.num_modes,
            "maximum_passes": self.maximum_passes,
            "maximum_delta_frequency_percent": self.maximum_delta_frequency_percent,
            "minimum_passes": self.minimum_passes,
            "minimum_converged_passes": self.minimum_converged_passes,
            "percent_refinement": self.percent_refinement,
        }
