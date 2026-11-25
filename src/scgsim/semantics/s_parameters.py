"""The S-parameters of a driven solve: the one record every solver returns, and its Touchstone file.

``SParameters`` is detached from the run it came from.  It holds the ports in the order the plan declares them,
the swept frequencies, and |S| in dB and arg S in degrees as the solver reported them, indexed
``[frequency][output][excited]``; ``s(output, excited)`` gives the complex values, 1-based as S21 is written.
Palace builds it from ``port-S.csv`` (``scgsim.palace``), HFSS from its own Touchstone file; each solver reads
its own files, and only this record and its writer are shared.

``write_touchstone`` writes Touchstone 1.1 with one reference impedance, so the ports must share one.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from scgsim.refusal import Declared, found_text

_WHERE_RECORD = "the values passed to SParameters"
_WHERE_TOUCHSTONE = "the call to SParameters.write_touchstone"
# Touchstone 1.1 puts at most four parameter pairs on one line; a longer matrix row continues on the next.
_PAIRS_PER_LINE = 4


@dataclass(frozen=True)
class SParameterPort:
    """One port of the S-matrix: its name in the plan, the sheet it sits on in the model, its impedance.

    ``name`` is the port as the plan declares it, the authored sheet's qualified name (``chip/o_lumped_left``);
    ``sheet_id`` is that sheet in the solver's model (``PORT_SHEET__chip/o_lumped_left__0000``);
    ``impedance_ohm`` is the impedance the port is terminated in and its waves are referenced to.
    """

    name: str
    sheet_id: str
    impedance_ohm: float

    def __post_init__(self) -> None:
        where = Declared(f"SParameterPort({self.name!r})", _WHERE_RECORD)
        for field in ("name", "sheet_id"):
            value = getattr(self, field)
            # The Touchstone writer puts both on one "!" comment line: a line break would end the comment and
            # turn the rest into an option or data line.  ``splitlines`` also refuses the empty string.
            if not isinstance(value, str) or value.splitlines() != [value]:
                raise ValueError(
                    where.expected(field, "nonempty text on one line", value)
                )
        impedance = _finite(where, "impedance_ohm", self.impedance_ohm)
        if impedance <= 0.0:
            raise ValueError(
                where.expected("impedance_ohm", "above 0 ohm", self.impedance_ohm)
            )
        object.__setattr__(self, "impedance_ohm", impedance)


@dataclass(frozen=True)
class SParameters:
    """The S-matrix of one driven solve at every swept frequency, with its ports, its sweep and its source.

    ``magnitude_db[k][o][e]`` and ``phase_deg[k][o][e]`` are |S| in dB (``20 log10 |S|``) and arg S in degrees
    at ``frequencies_hz[k]``, at output port ``o + 1`` with port ``e + 1`` excited, as the solver reported
    them.  ``sweep`` is the sweep the solver was asked for, in its own terms with frequencies in Hz;
    ``provenance`` names the run and the files the values were read from.

    Raises on build:
        ValueError: there is no port or a port is repeated, a frequency is not finite and above zero or not
            above the one before, or a matrix is missing, of the wrong shape or holds a value that is not finite.
        TypeError: a value is not a number, a port is not an ``SParameterPort``, or ``sweep`` or
            ``provenance`` is not a mapping.
    """

    ports: tuple[SParameterPort, ...]
    frequencies_hz: tuple[float, ...]
    magnitude_db: tuple[tuple[tuple[float, ...], ...], ...]
    phase_deg: tuple[tuple[tuple[float, ...], ...], ...]
    sweep: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        where = Declared("SParameters", _WHERE_RECORD)
        ports = _checked_ports(where, self.ports)
        frequencies = _checked_frequencies(where, self.frequencies_hz)
        shape = (len(frequencies), len(ports))
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "frequencies_hz", frequencies)
        for field in ("magnitude_db", "phase_deg"):
            object.__setattr__(
                self,
                field,
                _checked_matrices(where, field, getattr(self, field), shape),
            )
        for field in ("sweep", "provenance"):
            value = getattr(self, field)
            if not isinstance(value, Mapping):
                raise TypeError(where.expected(field, "a mapping", value))
            object.__setattr__(self, field, MappingProxyType(dict(value)))

    @property
    def reference_ohm(self) -> float | None:
        """The one impedance every port's waves are referenced to; None when the ports' impedances differ."""
        impedances = {port.impedance_ohm for port in self.ports}
        return impedances.pop() if len(impedances) == 1 else None

    def s(self, output: int, excited: int) -> tuple[complex, ...]:
        """S at output port ``output`` with port ``excited`` driven, at every frequency: ``s(2, 1)`` is S21.

        Raises:
            ValueError: ``output`` or ``excited`` is not a port number, 1 to the number of ports.
        """
        o, e = (self._position(output, "output"), self._position(excited, "excited"))
        return tuple(
            cmath.rect(10.0 ** (magnitude[o][e] / 20.0), math.radians(phase[o][e]))
            for magnitude, phase in zip(self.magnitude_db, self.phase_deg, strict=True)
        )

    def write_touchstone(self, path: str | Path) -> Path:
        """Write the record as a Touchstone 1.1 file ``*.s<N>p`` in GHz, |S| in dB and arg S in degrees.

        The values are written as the record holds them, digit for digit, so reading the file back gives the
        record's numbers.  Returns the path written.

        Raises:
            ValueError: ``path`` does not end in ``.s<N>p`` for the record's N ports, or the ports' impedances
                differ: Touchstone 1.1 has one reference impedance for every port.
        """
        target = Path(path)
        suffix = f".s{len(self.ports)}p"
        if target.suffix.lower() != suffix:
            raise ValueError(
                Declared(f"Touchstone path {str(target)!r}", _WHERE_TOUCHSTONE).refusal(
                    f"must end in {suffix}, the suffix of a {len(self.ports)}-port file"
                )
            )
        reference = self.reference_ohm
        if reference is None:
            raise ValueError(
                Declared("the record's ports", _WHERE_TOUCHSTONE).refusal(
                    "have different impedances, and Touchstone 1.1 references every port to one: "
                    + found_text({port.name: port.impedance_ohm for port in self.ports})
                    + ". Declare the ports with one impedance_ohm, or write the record with your own writer"
                )
            )
        target.write_text(self._touchstone_text(reference), encoding="utf-8")
        return target

    def _position(self, number: int, role: str) -> int:
        """The 0-based position of port ``number``, which counts from 1."""
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or not 1 <= number <= len(self.ports)
        ):
            raise ValueError(
                Declared(f"the {role} port", "the call to SParameters.s").refusal(
                    f"must be a port number from 1 to {len(self.ports)}: "
                    + found_text(number)
                )
            )
        return number - 1

    def _touchstone_text(self, reference: float) -> str:
        """The file: the ports as comments, the option line, then one block of lines per frequency."""
        lines = [
            "! S-parameters written by SCGSim; ports in the order the plan declares them",
            *(
                f"! Port {number}: {port.name} ({port.sheet_id}), {_text(port.impedance_ohm)} ohm"
                for number, port in enumerate(self.ports, start=1)
            ),
            f"# GHz S DB R {_text(reference)}",
        ]
        for frequency, magnitude, phase in zip(
            self.frequencies_hz, self.magnitude_db, self.phase_deg, strict=True
        ):
            pairs = [
                [f"{_text(magnitude[o][e])} {_text(phase[o][e])}" for o, e in row]
                for row in _touchstone_rows(len(self.ports))
            ]
            block = [
                " ".join(row[start : start + _PAIRS_PER_LINE])
                for row in pairs
                for start in range(0, len(row), _PAIRS_PER_LINE)
            ]
            block[0] = f"{_text(frequency / 1e9)} {block[0]}"
            lines.extend(block)
        return "\n".join(lines) + "\n"


def _touchstone_rows(ports: int) -> list[list[tuple[int, int]]]:
    """The (output, excited) positions of each data row in Touchstone 1.1's order.

    A two-port file is the one exception to row order: its one row is S11, S21, S12, S22.
    """
    if ports == 2:
        return [[(0, 0), (1, 0), (0, 1), (1, 1)]]
    return [[(o, e) for e in range(ports)] for o in range(ports)]


def _text(value: float) -> str:
    """A number as the file holds it: the shortest text that reads back as the same float, ``2`` for 2.0."""
    text = repr(float(value))
    return text.removesuffix(".0")


def _finite(where: Declared, field: str, value: object) -> float:
    """``value`` as a finite float; anything else is refused."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(where.expected(field, "a number", value))
    if not math.isfinite(value):
        raise ValueError(where.expected(field, "finite", value))
    return float(value)


def _checked_ports(where: Declared, ports: object) -> tuple[SParameterPort, ...]:
    """The ports as a tuple: at least one, each an ``SParameterPort``, no name twice."""
    if not isinstance(ports, Sequence) or not ports:
        raise ValueError(where.expected("ports", "at least one SParameterPort", ports))
    checked = tuple(ports)
    if not all(isinstance(port, SParameterPort) for port in checked):
        raise TypeError(where.expected("ports", "SParameterPort records", ports))
    names = [port.name for port in checked]
    if len(set(names)) != len(names):
        raise ValueError(where.expected("ports", "each named once", names))
    return checked


def _checked_frequencies(where: Declared, values: object) -> tuple[float, ...]:
    """The frequencies as floats: at least one, each finite and above zero, each above the one before."""
    if not isinstance(values, Sequence) or not values:
        raise ValueError(
            where.expected("frequencies_hz", "at least one frequency", values)
        )
    frequencies = tuple(_finite(where, "frequencies_hz", value) for value in values)
    if frequencies[0] <= 0.0 or any(
        later <= earlier
        for earlier, later in zip(frequencies[:-1], frequencies[1:], strict=True)
    ):
        raise ValueError(
            where.expected("frequencies_hz", "above 0 Hz and increasing", values)
        )
    return frequencies


def _checked_matrices(
    where: Declared, field: str, values: object, shape: tuple[int, int]
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """One ``ports`` x ``ports`` matrix of finite floats per frequency, as nested tuples."""
    frequencies, ports = shape
    expected = f"{frequencies} matrices of {ports} x {ports} finite numbers"
    if (
        not isinstance(values, Sequence)
        or len(values) != frequencies
        or any(
            not isinstance(matrix, Sequence)
            or len(matrix) != ports
            or any(not isinstance(row, Sequence) or len(row) != ports for row in matrix)
            for matrix in values
        )
    ):
        raise ValueError(where.expected(field, expected))
    return tuple(
        tuple(tuple(_finite(where, field, value) for value in row) for row in matrix)
        for matrix in values
    )
