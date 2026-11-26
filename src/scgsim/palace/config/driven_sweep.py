"""The frequency sweep of a Driven solve: its ranges, the frequencies Palace solves at, and the sweep record.

``Linear``, ``Log`` and ``Points`` are the ranges ``DrivenSim.set_driven`` takes, in Hz; each writes one entry of
Palace's ``Solver.Driven.Samples``, in GHz (``samples_entry``).  Palace 0.18.1 merges every entry into one list of
frequencies (``palace_frequencies_ghz``).  That one rule serves both ends of a run: ``DrivenSweep`` checks the
merged list when the sweep is set, and the read-back (``scgsim.palace.results.driven_tables``) checks that Palace's
returned rows are that list.  Palace prints a frequency with 9 significant digits (``printed_unit_ghz``), so
``DrivenSweep`` refuses two frequencies Palace solves apart but prints too close to be read back apart.  A range
given ``full_solve=True`` writes Palace's ``AddToPROM``: an adaptive sweep solves its frequencies in full first.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from scgsim.refusal import Declared, internal

_WHERE_DRIVEN_SWEEP = "the call to set_driven"
# Palace 0.18.1 takes samples closer than this, in the config's GHz, as one frequency: 1 Hz (``DrivenSolverData``,
# ``delta_eps``, in ``palace/utils/configfile.cpp``).
_SAME_FREQUENCY_GHZ = 1e-9

# Palace 0.18.1 starts the reduced model of each port from full solves at the lowest and the highest frequency.
_MODEL_ENDS = 2
# Palace 0.18.1's cap on the full solves per port without ``AdaptiveMaxSamples`` (``Solver.Driven`` in its schema).
_PALACE_MAX_SAMPLES = 20
# Palace builds round a range's frequencies differently, by a unit or two in the last place: a stock 0.18.1 build
# fuses the multiply and the add of ``ConstructLinearRange``, so its 4.44 GHz of 4-9 GHz in 501 points is 4.44, not
# 4.4399999999999995.  Within this many units SCGSim cannot tell which of two frequencies a build keeps.
_ROUNDING_ULPS = 8

PRINTED_DIGITS = 9
"""The significant digits Palace 0.18.1 prints a frequency with, in GHz (``2.10000000e+00``)."""


@dataclass(frozen=True)
class Linear:
    """``points`` frequencies evenly spaced from ``start_hz`` to ``stop_hz``, both ends included.

    ``full_solve=True`` has an adaptive sweep solve each of them in full (``DrivenSweep``).

    Raises on build:
        TypeError: a frequency is not a number, ``points`` is not an integer, or ``full_solve`` is not a bool.
        ValueError: ``start_hz`` is not finite and above 0 Hz, ``stop_hz`` is not finite and above ``start_hz``,
            or ``points`` is below 2.
    """

    start_hz: float
    stop_hz: float
    points: int
    full_solve: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        """Refuse a range Palace cannot sample, then hold the frequencies as floats."""
        where = Declared("the Linear range", _WHERE_DRIVEN_SWEEP)
        start, stop = _checked_band(where, self)
        _check_full_solve(where, self.full_solve)
        # A frozen dataclass normalizes its own fields through object.__setattr__.
        object.__setattr__(self, "start_hz", start)
        object.__setattr__(self, "stop_hz", stop)

    def samples_entry(self) -> dict[str, Any]:
        """The range as Palace's ``Samples`` entry, in GHz."""
        entry = {
            "Type": "Linear",
            "MinFreq": self.start_hz / 1e9,
            "MaxFreq": self.stop_hz / 1e9,
            "NSample": self.points,
        }
        return _with_full_solve(entry, self.full_solve)

    def to_payload(self) -> dict[str, Any]:
        """The range as ``input_summary()`` shows it, in Hz."""
        return {
            "kind": "linear",
            "start_hz": self.start_hz,
            "stop_hz": self.stop_hz,
            "points": self.points,
            "full_solve": self.full_solve,
        }


@dataclass(frozen=True)
class Log:
    """``points`` frequencies from ``start_hz`` to ``stop_hz``, both ends included, evenly spaced on a log scale.

    ``full_solve=True`` has an adaptive sweep solve each of them in full (``DrivenSweep``).

    Raises on build:
        TypeError: a frequency is not a number, ``points`` is not an integer, or ``full_solve`` is not a bool.
        ValueError: ``start_hz`` is not finite and above 0 Hz, ``stop_hz`` is not finite and above ``start_hz``,
            or ``points`` is below 2.
    """

    start_hz: float
    stop_hz: float
    points: int
    full_solve: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        """Refuse a range Palace cannot sample, then hold the frequencies as floats."""
        where = Declared("the Log range", _WHERE_DRIVEN_SWEEP)
        start, stop = _checked_band(where, self)
        _check_full_solve(where, self.full_solve)
        object.__setattr__(self, "start_hz", start)
        object.__setattr__(self, "stop_hz", stop)

    def samples_entry(self) -> dict[str, Any]:
        """The range as Palace's ``Samples`` entry, in GHz."""
        entry = {
            "Type": "Log",
            "MinFreq": self.start_hz / 1e9,
            "MaxFreq": self.stop_hz / 1e9,
            "NSample": self.points,
        }
        return _with_full_solve(entry, self.full_solve)

    def to_payload(self) -> dict[str, Any]:
        """The range as ``input_summary()`` shows it, in Hz."""
        return {
            "kind": "log",
            "start_hz": self.start_hz,
            "stop_hz": self.stop_hz,
            "points": self.points,
            "full_solve": self.full_solve,
        }


@dataclass(frozen=True)
class Points:
    """Chosen frequencies, in Hz, in any order; Palace solves each one once.

    ``full_solve=True`` has an adaptive sweep solve each of them in full (``DrivenSweep``).

    Raises on build:
        TypeError: ``frequencies_hz`` is not a list of frequencies, one of them is not a number, or ``full_solve``
            is not a bool.
        ValueError: ``frequencies_hz`` is empty, or a frequency is not finite and above 0 Hz.
    """

    frequencies_hz: Sequence[float]
    full_solve: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        """Refuse an empty list or a frequency Palace cannot sample, then hold them as a tuple of floats."""
        where = Declared("the Points range", _WHERE_DRIVEN_SWEEP)
        _check_full_solve(where, self.full_solve)
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
        entry = {"Type": "Point", "Freq": [f / 1e9 for f in self.frequencies_hz]}
        return _with_full_solve(entry, self.full_solve)

    def to_payload(self) -> dict[str, Any]:
        """The frequencies as ``input_summary()`` shows them, in Hz."""
        return {
            "kind": "points",
            "frequencies_hz": list(self.frequencies_hz),
            "full_solve": self.full_solve,
        }


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
    excitation; None leaves Palace's own cap, 20.  The model starts from full solves at the lowest and the highest
    frequency, and at every frequency of a range given ``full_solve=True`` (Palace's ``AddToPROM``); these count
    toward the cap.

    Raises on build:
        TypeError: ``ranges`` is not a list of ``Linear``, ``Log`` and ``Points`` ranges, the tolerance is not a
            number, or ``adaptive_max_samples`` is not an integer.
        ValueError: ``ranges`` is empty; two merged frequencies are closer than the 9 significant digits Palace
            prints them with; ``adaptive_tolerance`` is not finite and above zero; ``adaptive_max_samples`` or
            ``full_solve`` is given without ``adaptive_tolerance``; an adaptive sweep has no more frequencies than
            the full solves its model starts from, or a cap below them; or a ``full_solve`` frequency is less than
            1 Hz above another frequency, which Palace keeps in its place and then stops.
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
        if self.adaptive:
            _check_model_start(where, self, frequencies)
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
    """The frequencies of one ``Samples`` entry, in GHz, computed as ``ConstructLinearRange`` and ``ConstructLogRange`` do.

    To the last place only as unfused arithmetic rounds them; a Palace build may differ there (``_ROUNDING_ULPS``).
    """
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
    """Refuse a tolerance that is not above 0, or a cap or ``full_solve`` without a tolerance; hold it as a float."""
    # Palace 0.18.1 ignores AddToPROM on a sweep that is not adaptive, with only a warning (``DrivenSolverData``).
    if sweep.adaptive_tolerance is None and any(
        item.full_solve for item in sweep.ranges
    ):
        raise ValueError(
            where.refusal(
                "full_solve applies only to an adaptive sweep, and no adaptive_tolerance turns one on: without "
                "it Palace solves every frequency in full already. Give adaptive_tolerance too, or leave "
                "full_solve out"
            )
        )
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
    _sweep_integer(where, "adaptive_max_samples", sweep.adaptive_max_samples)


def _check_model_start(
    where: Declared, sweep: DrivenSweep, frequencies_ghz: Sequence[float]
) -> None:
    """Refuse an adaptive sweep Palace 0.18.1 would solve in full, or stop on, before it builds its model.

    Palace starts the model of each port from full solves at the lowest and the highest frequency and at each
    ``full_solve`` one (``prom_indices``, ``DrivenSolverData`` in ``palace/utils/configfile.cpp``).  It solves a
    sweep of no more frequencies than those in full, with only a warning (``DrivenSolver::Solve``), so its log
    holds no adaptive sampling; on a cap below them it stops with an error (``SweepAdaptive``, both in
    ``palace/drivers/drivensolver.cpp``).
    """
    full_solves = _full_solves_ghz(where, sweep, frequencies_ghz)
    starting = _MODEL_ENDS + len(full_solves)
    solves = "the lowest and the highest frequency"
    if full_solves:
        noun = "frequency" if len(full_solves) == 1 else "frequencies"
        solves += f" and {len(full_solves)} full_solve {noun}"
    if len(frequencies_ghz) <= starting:
        also = " and full_solve" if full_solves else ""
        raise ValueError(
            where.refusal(
                f"an adaptive sweep needs at least {starting + 1} distinct frequencies, and the ranges give "
                f"{len(frequencies_ghz)}: Palace starts its reduced model from full solves at {solves}, and "
                f"solves a sweep of no more in full. Add frequencies, or leave adaptive_tolerance{also} out to "
                "solve them in full"
            )
        )
    cap = sweep.adaptive_max_samples
    if cap is None and starting > _PALACE_MAX_SAMPLES:
        raise ValueError(
            where.refusal(
                f"the reduced model starts from {starting} full solves per port, at {solves}, and without "
                f"adaptive_max_samples Palace caps them at {_PALACE_MAX_SAMPLES} and stops on a cap below "
                f"them. Give adaptive_max_samples of at least {starting}, or fewer full_solve frequencies"
            )
        )
    if cap is not None and cap < starting:
        raise ValueError(
            where.expected(
                "adaptive_max_samples",
                f"at least {starting}, the solves at {solves}",
                cap,
            )
        )


def _full_solves_ghz(
    where: Declared, sweep: DrivenSweep, frequencies_ghz: Sequence[float]
) -> tuple[float, ...]:
    """The ``full_solve`` frequencies Palace adds to the two its model starts from, in GHz.

    As ``DrivenSolverData`` reads ``AddToPROM``: every marked range's frequencies, merged as all of them are
    (``palace_frequencies_ghz``), less the lowest and the highest of the sweep.  Palace then looks for each one
    among the frequencies it keeps, by its exact value, and stops when one is not there.  So a ``full_solve``
    frequency less than 1 Hz above another, which Palace keeps in its place, is refused.  One within rounding of
    another is left to Palace's own check: which of the two a build keeps depends on how it rounds.
    """
    marked = palace_frequencies_ghz(
        item.samples_entry() for item in sweep.ranges if item.full_solve
    )
    ends = (frequencies_ghz[0], frequencies_ghz[-1])
    kept = set(frequencies_ghz)
    full_solves = tuple(frequency for frequency in marked if frequency not in ends)
    for frequency in full_solves:
        if frequency in kept:
            continue
        below = frequencies_ghz[bisect_right(frequencies_ghz, frequency) - 1]
        if frequency - below > _ROUNDING_ULPS * math.ulp(frequency):
            unit = printed_unit_ghz(frequency) * 1e9
            raise ValueError(
                where.refusal(
                    f"the full_solve frequency {frequency:.15g} GHz is {(frequency - below) * 1e9:.2g} Hz "
                    f"above {below:.15g} GHz, another frequency of the sweep. Palace solves the two as "
                    f"{below:.15g} GHz, then looks for the full solve at {frequency:.15g} GHz and stops: move "
                    f"the full_solve frequency more than {unit:.3g} Hz away from {below:.15g} GHz"
                )
            )
    return full_solves


def _check_full_solve(where: Declared, value: object) -> None:
    """Refuse a ``full_solve`` that is not True or False."""
    if not isinstance(value, bool):
        raise TypeError(where.expected("full_solve", "True or False", value))


def _with_full_solve(entry: dict[str, Any], full_solve: bool) -> dict[str, Any]:
    """``entry`` with Palace's ``AddToPROM`` when its frequencies are full solves; Palace's default is false."""
    return {**entry, "AddToPROM": True} if full_solve else entry


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
