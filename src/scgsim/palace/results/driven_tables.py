"""Validate the native tables of a Palace Driven run: Palace 0.18.1's headers, one row per requested frequency.

``DrivenSim`` excites each lumped port of the plan on its own, with ``Excitation`` equal to the port's ``Index``
(``driven_excitations``).  So each table but ``error-indicators`` has one block of columns per excitation, in
excitation order, and one row per frequency of the sweep the sealed config requests (``requested_sweep``,
``requested_frequencies_ghz``).  Its ``f (GHz)`` column must be those frequencies, in order, to the 9
significant digits Palace prints.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from scgsim.contracts.palace_run_folder import CONFIG, DRIVEN
from scgsim.palace.config.config_records import DrivenSweep
from scgsim.palace.results.artifacts import (
    checked_number,
    expect_int,
    expect_list,
    expect_mapping,
)
from scgsim.palace.results.resolved_records import ParsedTable
from scgsim.palace.results.table_rows import (
    require_finite_data,
    require_surface_q_data,
    validate_error_indicators,
)
from scgsim.palace.run.archive import RETURNED_RUN_FOLDER
from scgsim.refusal import Declared, found_text, internal

FREQUENCY_COLUMN = DRIVEN.series_index
"""The column of every Driven table that names the row's frequency, in GHz."""
# Palace prints a frequency with 9 significant digits (``2.10000000e+00``).
_PRINTED_DIGITS = 9


class _Counts(NamedTuple):
    """How many lumped ports, energy domains and Surface-EPR rows the index map configures."""

    ports: int
    domains: int
    surfaces: int


def validate_driven_table(
    table: ParsedTable, index_counts: dict[str, int], config: dict[str, Any]
) -> None:
    """Check one Driven table: its headers, finite values, and one row per requested frequency, in order."""
    if table.name == "error-indicators":
        validate_error_indicators(table)
        return
    columns = _EXCITATION_COLUMNS.get(table.name)
    if columns is None:
        raise ValueError(internal(f"no Driven validator reads table {table.name!r}"))
    counts = _Counts(
        index_counts["lumped_port"], index_counts["domain"], index_counts["surface"]
    )
    expected = (
        FREQUENCY_COLUMN,
        *(
            column
            for excitation in driven_excitations(config)
            for column in columns(excitation, counts)
        ),
    )
    _require_headers(table, expected)
    # Palace writes an infinite Q_surf for a lossless film; every other value is finite.
    check_row = (
        require_surface_q_data if table.name == "surface-Q" else require_finite_data
    )
    for row in table.rows:
        check_row(row)
    _require_requested_frequencies(table, requested_frequencies_ghz(config))


def driven_excitations(config: Mapping[str, Any]) -> tuple[int, ...]:
    """The excitation indices of a Driven config, in order: each lumped port excited by its own ``Index``.

    The column block of excitation e is then the one with port e driven.  An unexcited lumped element
    (``Excitation: false``) is refused: Palace 0.18.1's ``port-S`` columns for a run with one are not
    verified, so SCGSim does not read its S-matrix yet.
    """
    boundaries = expect_mapping(config.get("Boundaries"), "config.Boundaries")
    rows = expect_list(boundaries.get("LumpedPort"), "config.Boundaries.LumpedPort")
    excitations: list[int] = []
    for row in rows:
        port = expect_mapping(row, "config.Boundaries.LumpedPort row")
        index = expect_int(port.get("Index"), "config LumpedPort Index")
        excitation = port.get("Excitation")
        if excitation is False:
            raise NotImplementedError(
                Declared(
                    f"the Driven run's lumped element at LumpedPort Index {index}",
                    "the plan: GeometryPlan.set_lumped_elements, with the Driven ports only",
                ).refusal(
                    "is an unexcited lumped port, and SCGSim does not read the S-parameters of a run with "
                    "one yet: stock Palace 0.18.1's port-S columns for it are not verified"
                )
            )
        if type(excitation) is not int or excitation != index:
            raise ValueError(
                Declared(
                    f"{CONFIG} LumpedPort Index {index} Excitation", RETURNED_RUN_FOLDER
                ).refusal(
                    "must be the port's own Index, as DrivenSim writes it: "
                    + found_text(excitation, index)
                )
            )
        excitations.append(index)
    return tuple(sorted(excitations))


def requested_frequencies_ghz(config: Mapping[str, Any]) -> tuple[float, ...]:
    """The frequencies the sealed config asks Palace to solve at, in GHz.

    Palace spaces ``NSample`` samples evenly from ``MinFreq`` to ``MaxFreq``, both ends included.
    """
    sweep = _linear_range(config)
    step = (sweep.high_ghz - sweep.low_ghz) / (sweep.samples - 1)
    return tuple(sweep.low_ghz + step * k for k in range(sweep.samples))


def requested_sweep(config: Mapping[str, Any]) -> DrivenSweep:
    """The sweep the sealed config requests, as ``DrivenSim.set_driven`` set it, frequencies in Hz."""
    sweep = _linear_range(config)
    return DrivenSweep(
        start_hz=sweep.low_ghz * 1e9,
        stop_hz=sweep.high_ghz * 1e9,
        points=sweep.samples,
        adaptive_tolerance=sweep.driven.get("AdaptiveTol"),
        adaptive_max_samples=sweep.driven.get("AdaptiveMaxSamples"),
    )


class _LinearRange(NamedTuple):
    """The one ``Linear`` sample range of a Driven config, in GHz, and the ``Solver.Driven`` block it is in."""

    low_ghz: float
    high_ghz: float
    samples: int
    driven: dict[str, Any]


def _linear_range(config: Mapping[str, Any]) -> _LinearRange:
    """The sample range ``DrivenSim`` writes: one ``Linear`` range of at least 2 samples, upward."""
    solver = expect_mapping(config.get("Solver"), "config.Solver")
    driven = expect_mapping(solver.get(DRIVEN.name), f"config.Solver.{DRIVEN.name}")
    samples = expect_list(driven.get("Samples"), "config.Solver.Driven.Samples")
    sample = samples[0] if len(samples) == 1 else None
    if not isinstance(sample, dict) or sample.get("Type") != "Linear":
        raise ValueError(
            Declared(f"{CONFIG} Solver.Driven.Samples", RETURNED_RUN_FOLDER).refusal(
                "must be the one Linear range DrivenSim writes: " + found_text(samples)
            )
        )
    low = checked_number(sample.get("MinFreq"), "Samples MinFreq", positive=True)
    high = checked_number(sample.get("MaxFreq"), "Samples MaxFreq", positive=True)
    count = expect_int(sample.get("NSample"), "Samples NSample")
    if count < 2 or high <= low:
        raise ValueError(
            Declared(f"{CONFIG} Solver.Driven.Samples", RETURNED_RUN_FOLDER).refusal(
                "must range upward over at least 2 samples: " + found_text(sample)
            )
        )
    return _LinearRange(low, high, count, driven)


def _port_s_columns(excitation: int, counts: _Counts) -> list[str]:
    """``|S[o][e]|`` in dB and ``arg(S[o][e])`` in degrees at every port o, with port e excited."""
    return [
        column
        for port in range(1, counts.ports + 1)
        for column in (
            f"|S[{port}][{excitation}]| (dB)",
            f"arg(S[{port}][{excitation}]) (deg.)",
        )
    ]


def _port_wave_columns(kind: str, unit: str) -> Callable[[int, _Counts], list[str]]:
    """The ``port-V`` or ``port-I`` block of one excitation: its incident value, then each port's complex one."""

    def columns(excitation: int, counts: _Counts) -> list[str]:
        return [
            f"{kind}_inc[{excitation}][{excitation}] ({unit})",
            *(
                column
                for port in range(1, counts.ports + 1)
                for column in (
                    f"Re{{{kind}[{port}][{excitation}]}} ({unit})",
                    f"Im{{{kind}[{port}][{excitation}]}} ({unit})",
                )
            ),
        ]

    return columns


def _domain_columns(excitation: int, counts: _Counts) -> list[str]:
    """The field energies of one excitation, then each domain's energies and participations."""
    return [
        *(
            f"{name}[{excitation}] (J)"
            for name in ("E_elec", "E_mag", "E_cap", "E_ind")
        ),
        *(
            column
            for domain in range(1, counts.domains + 1)
            for column in (
                f"E_elec[{domain}][{excitation}] (J)",
                f"p_elec[{domain}][{excitation}]",
                f"E_mag[{domain}][{excitation}] (J)",
                f"p_mag[{domain}][{excitation}]",
            )
        ),
    ]


def _surface_columns(excitation: int, counts: _Counts) -> list[str]:
    """Each Surface-EPR row's participation and Q with one port excited."""
    return [
        column
        for surface in range(1, counts.surfaces + 1)
        for column in (
            f"p_surf[{surface}][{excitation}]",
            f"Q_surf[{surface}][{excitation}]",
        )
    ]


_EXCITATION_COLUMNS: dict[str, Callable[[int, _Counts], list[str]]] = {
    "port-S": _port_s_columns,
    "port-V": _port_wave_columns("V", "V"),
    "port-I": _port_wave_columns("I", "A"),
    "domain-E": _domain_columns,
    "surface-Q": _surface_columns,
}
"""The columns each Driven table writes for one excitation, after its ``f (GHz)`` column."""


def _require_headers(table: ParsedTable, expected: tuple[str, ...]) -> None:
    """Refuse a table whose headers are not Palace 0.18.1's, naming the first that differs."""
    if table.headers == expected:
        return
    first = next(
        (
            position
            for position, (found, wanted) in enumerate(
                zip(table.headers, expected, strict=False)
            )
            if found != wanted
        ),
        min(len(table.headers), len(expected)),
    )
    found = table.headers[first] if first < len(table.headers) else None
    wanted = expected[first] if first < len(expected) else None
    raise ValueError(
        Declared(f"{table.path} headers", RETURNED_RUN_FOLDER).refusal(
            f"are not the {len(expected)} columns stock Palace 0.18.1 writes for this Driven run "
            f"({len(table.headers)} found); column {first + 1}: "
            + found_text(found, wanted)
        )
    )


def _require_requested_frequencies(
    table: ParsedTable, requested: tuple[float, ...]
) -> None:
    """Refuse a table whose rows are not the requested frequencies, in order, to the digits Palace prints."""
    returned = [row[FREQUENCY_COLUMN] for row in table.rows]
    if len(returned) != len(requested):
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                f"has {len(returned)} frequency rows, and {CONFIG} requests {len(requested)} frequencies"
            )
        )
    for row, (found, wanted) in enumerate(
        zip(returned, requested, strict=True), start=1
    ):
        # One unit in the last printed digit: Palace rounds its own value of the frequency to that digit.
        unit = 10.0 ** (math.floor(math.log10(wanted)) - (_PRINTED_DIGITS - 1))
        if abs(found - wanted) > unit:
            raise ValueError(
                Declared(f"{table.path} row {row}", RETURNED_RUN_FOLDER).refusal(
                    f"is not at the requested frequency to the {_PRINTED_DIGITS} significant digits Palace "
                    f"prints, in GHz: " + found_text(found, wanted)
                )
            )
