"""The frequency sweep of a Driven solve: its ranges, the frequencies Palace solves at, and the sweep record.

``Linear``, ``Log`` and ``Points`` are the ranges ``DrivenSim.set_driven`` takes, in Hz; each writes one entry of
Palace's ``Solver.Driven.Samples``, in GHz (``samples_entry``).  Palace 0.18.1 merges every entry into one list of
frequencies (``palace_frequencies_ghz``).  That one rule serves both ends of a run: ``DrivenSweep`` checks the
merged list when the sweep is set, and the read-back (``scgsim.palace.results.driven_tables``) checks that Palace's
returned rows are that list.  Palace prints a frequency with 9 significant digits (``printed_unit_ghz``), so
``DrivenSweep`` refuses two frequencies Palace solves apart but prints too close to be read back apart.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from scgsim.refusal import Declared, internal

_WHERE_DRIVEN_SWEEP = "the call to set_driven"
# Palace 0.18.1 takes samples closer than this, in the config's GHz, as one frequency: 1 Hz (``DrivenSolverData``,
# ``delta_eps``, in ``palace/utils/configfile.cpp``).
_SAME_FREQUENCY_GHZ = 1e-9

# Palace 0.18.1 runs its adaptive sweep only on more frequencies than the two its reduced model starts from.
_ADAPTIVE_FREQUENCIES = 3

PRINTED_DIGITS = 9
"""The significant digits Palace 0.18.1 prints a frequency with, in GHz (``2.10000000e+00``)."""


@dataclass(frozen=True)
class Linear:
    """``points`` frequencies evenly spaced from ``start_hz`` to ``stop_hz``, both ends included.

    Raises on build:
        TypeError: a frequency is not a number, or ``points`` is not an integer.
        ValueError: ``start_hz`` is not finite and above 0 Hz, ``stop_hz`` is not finite and above ``start_hz``,
            or ``points`` is below 2.
    """

    start_hz: float
    stop_hz: float
    points: int

    def __post_init__(self) -> None:
        """Refuse a range Palace cannot sample, then hold the frequencies as floats."""
        start, stop = _checked_band(
            Declared("the Linear range", _WHERE_DRIVEN_SWEEP), self
        )
        # A frozen dataclass normalizes its own fields through object.__setattr__.
        object.__setattr__(self, "start_hz", start)
        object.__setattr__(self, "stop_hz", stop)

    def samples_entry(self) -> dict[str, Any]:
        """The range as Palace's ``Samples`` entry, in GHz."""
        return {
            "Type": "Linear",
            "MinFreq": self.start_hz / 1e9,
            "MaxFreq": self.stop_hz / 1e9,
            "NSample": self.points,
        }

    def to_payload(self) -> dict[str, Any]:
        """The range as ``input_summary()`` shows it, in Hz."""
        return {
            "kind": "linear",
            "start_hz": self.start_hz,
            "stop_hz": self.stop_hz,
            "points": self.points,
        }


@dataclass(frozen=True)
class Log:
    """``points`` frequencies from ``start_hz`` to ``stop_hz``, both ends included, evenly spaced on a log scale.

    Raises on build:
        TypeError: a frequency is not a number, or ``points`` is not an integer.
        ValueError: ``start_hz`` is not finite and above 0 Hz, ``stop_hz`` is not finite and above ``start_hz``,
            or ``points`` is below 2.
    """

    start_hz: float
    stop_hz: float
    points: int

    def __post_init__(self) -> None:
        """Refuse a range Palace cannot sample, then hold the frequencies as floats."""
        start, stop = _checked_band(
            Declared("the Log range", _WHERE_DRIVEN_SWEEP), self
        )
        object.__setattr__(self, "start_hz", start)
        object.__setattr__(self, "stop_hz", stop)

    def samples_entry(self) -> dict[str, Any]:
        """The range as Palace's ``Samples`` entry, in GHz."""
        return {
            "Type": "Log",
            "MinFreq": self.start_hz / 1e9,
            "MaxFreq": self.stop_hz / 1e9,
            "NSample": self.points,
        }

    def to_payload(self) -> dict[str, Any]:
        """The range as ``input_summary()`` shows it, in Hz."""
        return {
            "kind": "log",
            "start_hz": self.start_hz,
            "stop_hz": self.stop_hz,
            "points": self.points,
        }


@dataclass(frozen=True)
class Points:
    """Chosen frequencies, in Hz, in any order; Palace solves each one once.

    Raises on build:
        TypeError: ``frequencies_hz`` is not a list of frequencies, or one of them is not a number.
        ValueError: ``frequencies_hz`` is empty, or a frequency is not finite and above 0 Hz.
    """

    frequencies_hz: Sequence[float]

    def __post_init__(self) -> None:
        """Refuse an empty list or a frequency Palace cannot sample, then hold them as a tuple of floats."""
        where = Declared("the Points range", _WHERE_DRIVEN_SWEEP)
        values = self.frequencies_hz
        if isinstance(values, (str, bytes, Mapping)) or not isinstance(
            values, Iterable
        ):
            raise TypeError(
                where.expected("frequencies_hz", "a list of frequencies in Hz", values)
            )
        listed = list(values)
        if not listed:
            raise ValueError(
                where.expected("frequencies_hz", "at least one frequency", values)
            )
        object.__setattr__(
            self,
            "frequencies_hz",
            tuple(
                _positive_frequency(where, f"frequencies_hz[{k}]", value)
                for k, value in enumerate(listed)
            ),
        )

    def samples_entry(self) -> dict[str, Any]:
        """The frequencies as Palace's ``Samples`` entry, in GHz."""
        return {"Type": "Point", "Freq": [f / 1e9 for f in self.frequencies_hz]}

    def to_payload(self) -> dict[str, Any]:
        """The frequencies as ``input_summary()`` shows them, in Hz."""
        return {"kind": "points", "frequencies_hz": list(self.frequencies_hz)}


FrequencyRange = Linear | Log | Points
"""One range of a Driven sweep, as ``set_driven`` takes it."""


@dataclass(frozen=True)
class DrivenSweep:
    """The frequency sweep of one Driven solve: its ranges, how many frequencies Palace solves at, the sampling.

    ``ranges`` are written as Palace's ``Samples`` in the order given (``samples``).  Palace solves at their
    frequencies merged (``palace_frequencies_ghz``), ``frequency_count`` of them.  ``adaptive_tolerance`` None
    solves the field at every frequency (a uniform sweep).  A tolerance turns on Palace's adaptive fast sweep:
    it solves a few frequencies, adds solves until a reduced model built from them agrees with them to that
    relative error, and evaluates the model at every frequency.  ``adaptive_max_samples`` caps those solves per
    excitation; None leaves Palace's own cap.

    Raises on build:
        TypeError: ``ranges`` is not a list of ``Linear``, ``Log`` and ``Points`` ranges, the tolerance is not a
            number, or ``adaptive_max_samples`` is not an integer.
        ValueError: ``ranges`` is empty; two merged frequencies are closer than the 9 significant digits Palace
            prints them with; ``adaptive_tolerance`` is not finite and above zero; an adaptive sweep has fewer
            than 3 frequencies; or ``adaptive_max_samples`` is below 2 or is given without ``adaptive_tolerance``.
    """

    ranges: Sequence[FrequencyRange]
    adaptive_tolerance: float | None = None
    adaptive_max_samples: int | None = None
    frequency_count: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """Refuse a sweep Palace cannot sample or whose rows could not be read back, then count its frequencies."""
        where = Declared("the frequency sweep", _WHERE_DRIVEN_SWEEP)
        object.__setattr__(self, "ranges", _checked_ranges(where, self.ranges))
        _check_adaptive_settings(where, self)
        frequencies = palace_frequencies_ghz(self.samples())
        _require_printed_apart(where, frequencies)
        # Palace's reduced model starts from full solves at the lowest and the highest frequency, and Palace 0.18.1
        # solves an adaptive sweep of no more frequencies than those in full, with only a warning
        # (``DrivenSolver::Solve``, ``palace/drivers/drivensolver.cpp``), so its log would hold no adaptive sampling.
        if self.adaptive and len(frequencies) < _ADAPTIVE_FREQUENCIES:
            raise ValueError(
                where.refusal(
                    f"an adaptive sweep needs at least {_ADAPTIVE_FREQUENCIES} distinct frequencies, and the "
                    f"ranges give {len(frequencies)}: Palace starts its reduced model from the lowest and the "
                    "highest, and solves a sweep of no more in full. Add frequencies, or leave "
                    "adaptive_tolerance out to solve them in full"
                )
            )
        object.__setattr__(self, "frequency_count", len(frequencies))

    @property
    def adaptive(self) -> bool:
        """Whether the sweep is Palace's adaptive fast sweep rather than a solve at every frequency."""
        return self.adaptive_tolerance is not None

    def samples(self) -> list[dict[str, Any]]:
        """``Solver.Driven.Samples``: one entry per range, in the order given, in GHz."""
        return [item.samples_entry() for item in self.ranges]

    def to_payload(self) -> dict[str, Any]:
        """The sweep as ``input_summary()`` and ``SParameters.sweep`` show it, frequencies in Hz."""
        return {
            "sampling": "adaptive" if self.adaptive else "uniform",
            "ranges": [item.to_payload() for item in self.ranges],
            "frequency_count": self.frequency_count,
            "adaptive_tolerance": self.adaptive_tolerance,
            "adaptive_max_samples": self.adaptive_max_samples,
        }


def palace_frequencies_ghz(entries: Iterable[Mapping[str, Any]]) -> tuple[float, ...]:
    """The frequencies Palace 0.18.1 solves at for ``Solver.Driven.Samples`` entries as SCGSim writes them, in GHz.

    As ``DrivenSolverData`` does (``palace/utils/configfile.cpp``): each entry's frequencies in Palace's own
    arithmetic, every entry's together, sorted, with each frequency within 1 Hz of the last one kept left out.
    Palace's ``std::unique`` compares a frequency with the last one kept, not with the one just before it.
    """
    merged: list[float] = []
    for frequency in sorted(f for entry in entries for f in _entry_frequencies(entry)):
        if not merged or frequency - merged[-1] >= _SAME_FREQUENCY_GHZ:
            merged.append(frequency)
    return tuple(merged)


def printed_unit_ghz(frequency_ghz: float) -> float:
    """One unit in the last of the 9 digits Palace prints ``frequency_ghz`` with: 1e-8 GHz (10 Hz) from 1 to 10 GHz."""
    return 10.0 ** (math.floor(math.log10(frequency_ghz)) - (PRINTED_DIGITS - 1))


def _entry_frequencies(entry: Mapping[str, Any]) -> list[float]:
    """The frequencies of one ``Samples`` entry, in GHz, computed as ``ConstructLinearRange`` and ``ConstructLogRange`` do."""
    kind = entry["Type"]
    if kind == "Point":
        return [float(value) for value in entry["Freq"]]
    low, high, count = (
        float(entry["MinFreq"]),
        float(entry["MaxFreq"]),
        entry["NSample"],
    )
    if kind == "Linear":
        return [low + (i / (count - 1)) * (high - low) for i in range(count)]
    if kind == "Log":
        log_low, log_high = math.log10(low), math.log10(high)
        return [
            10.0 ** (log_low + (i / (count - 1)) * (log_high - log_low))
            for i in range(count)
        ]
    raise ValueError(internal(f"no Samples entry of Type {kind!r} is written"))


def _checked_ranges(where: Declared, ranges: object) -> tuple[FrequencyRange, ...]:
    """``ranges`` as a tuple; anything but a nonempty list of ``Linear``, ``Log`` and ``Points`` is refused."""
    # One range alone is a record, not a list of them: it is refused rather than read as a list.
    if isinstance(ranges, (str, bytes, Mapping, Linear, Log, Points)) or not isinstance(
        ranges, Iterable
    ):
        raise TypeError(
            where.expected(
                "frequencies", "a list of Linear, Log and Points ranges", ranges
            )
        )
    listed = tuple(ranges)
    if not listed:
        raise ValueError(where.expected("frequencies", "at least one range", ranges))
    for k, item in enumerate(listed):
        if not isinstance(item, (Linear, Log, Points)):
            raise TypeError(
                where.expected(
                    f"frequencies[{k}]", "a Linear, Log or Points range", item
                )
            )
    return listed


def _check_adaptive_settings(where: Declared, sweep: DrivenSweep) -> None:
    """Refuse a tolerance that is not above 0 or a cap without a tolerance, then hold the tolerance as a float."""
    if sweep.adaptive_tolerance is not None:
        tolerance = _sweep_number(where, "adaptive_tolerance", sweep.adaptive_tolerance)
        if tolerance <= 0.0:
            raise ValueError(
                where.expected(
                    "adaptive_tolerance",
                    "above 0; leave it out for a uniform sweep",
                    sweep.adaptive_tolerance,
                )
            )
        object.__setattr__(sweep, "adaptive_tolerance", tolerance)
    if sweep.adaptive_max_samples is None:
        return
    if sweep.adaptive_tolerance is None:
        raise ValueError(
            where.refusal(
                "adaptive_max_samples caps the solves of the adaptive sweep, and no "
                "adaptive_tolerance turns that sweep on: give adaptive_tolerance too, or leave "
                "adaptive_max_samples out"
            )
        )
    # Palace's model starts from full solves at the lowest and highest frequency, and Palace 0.18.1 stops with
    # an error when the cap is below those two (``SweepAdaptive``).
    if _sweep_integer(where, "adaptive_max_samples", sweep.adaptive_max_samples) < 2:
        raise ValueError(
            where.expected(
                "adaptive_max_samples",
                "at least 2, the solves at the lowest and highest frequency",
                sweep.adaptive_max_samples,
            )
        )


def _require_printed_apart(where: Declared, frequencies_ghz: Sequence[float]) -> None:
    """Refuse two neighbouring frequencies no more than one printed unit apart: their rows could not be told apart.

    More than one unit in the last printed digit apart, the two print as different numbers however Palace
    rounds them, so each returned row is read back as its own frequency.
    """
    for low, high in pairwise(frequencies_ghz):
        unit = printed_unit_ghz(high)
        if high - low <= unit:
            raise ValueError(
                where.refusal(
                    f"{low:.12g} GHz and {high:.12g} GHz are {(high - low) * 1e9:.3g} Hz apart, and Palace "
                    f"prints a frequency with {PRINTED_DIGITS} significant digits, {unit * 1e9:.3g} Hz here, "
                    f"so their rows could not be read back apart: keep the frequencies more than "
                    f"{unit * 1e9:.3g} Hz apart there"
                )
            )


def _checked_band(where: Declared, band: Linear | Log) -> tuple[float, float]:
    """The start and stop of ``band`` as floats; a band Palace cannot sample is refused."""
    start = _positive_frequency(where, "start_hz", band.start_hz)
    stop = _sweep_number(where, "stop_hz", band.stop_hz)
    if stop <= start:
        raise ValueError(
            where.expected("stop_hz", f"above start_hz ({start!r} Hz)", band.stop_hz)
        )
    # Palace spaces NSample points by (stop - start) / (NSample - 1), both ends included.
    if _sweep_integer(where, "points", band.points) < 2:
        raise ValueError(where.expected("points", "at least 2", band.points))
    return start, stop


def _positive_frequency(where: Declared, name: str, value: object) -> float:
    """``value`` as a frequency in Hz; one that is not finite and above zero is refused."""
    frequency = _sweep_number(where, name, value)
    # The frequency-domain problem is singular at zero frequency, so the sweep stays above it.
    if frequency <= 0.0:
        raise ValueError(where.expected(name, "above 0 Hz", value))
    return frequency


def _sweep_number(where: Declared, name: str, value: object) -> float:
    """``value`` as a finite float; a value that is not a finite number is refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(where.expected(name, "a number", value))
    if not math.isfinite(float(value)):
        raise ValueError(where.expected(name, "finite", value))
    return float(value)


def _sweep_integer(where: Declared, name: str, value: object) -> int:
    """``value`` as an int; a value that is not an integer is refused."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(where.expected(name, "an integer", value))
    return value


__all__ = [
    "PRINTED_DIGITS",
    "DrivenSweep",
    "FrequencyRange",
    "Linear",
    "Log",
    "Points",
    "palace_frequencies_ghz",
    "printed_unit_ghz",
]
