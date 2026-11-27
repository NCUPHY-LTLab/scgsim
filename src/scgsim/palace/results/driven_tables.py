"""Validate the native tables of a Palace Driven run: Palace 0.18.1's headers, one row per requested frequency.

``DrivenSim`` excites each port of the plan on its own, with ``Excitation`` equal to the port's ``Index``, and
writes the plan's other lumped elements as unexcited lumped ports (``Excitation: false``; ``driven_excitations``).
So each table but ``error-indicators`` has one block of columns per excitation, in excitation order, and one row
per frequency of the sweep the sealed config requests (``requested_sweep``, ``requested_frequencies_ghz``): its
``Samples`` entries merged as Palace merges them (``palace_frequencies_ghz``).  Its ``f (GHz)`` column must be
those frequencies, in order, to the 9 significant digits Palace prints.

Within a block, ``port-S``, ``port-V`` and ``port-I`` have columns for every lumped port, the unexcited elements
included: stock Palace 0.18.1 wrote them so for an element of R 50 ohm and one of L 10 nH (issue 187).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from scgsim.contracts.palace_run_folder import (
    CONFIG,
    DRIVEN,
    PORT_S_MAGNITUDE_COLUMN,
    PORT_S_PHASE_COLUMN,
)
from scgsim.palace.config.driven_sweep import (
    PRINTED_DIGITS,
    DrivenSweep,
    FrequencyRange,
    Linear,
    Log,
    Points,
    palace_frequencies_ghz,
    printed_unit_ghz,
)
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


class _Counts(NamedTuple):
    """How many lumped ports (the elements included), energy domains and Surface-EPR rows the index map configures."""

    ports: int
    domains: int
    surfaces: int


_Columns = Callable[[int, str, _Counts], list[str]]
"""The columns of one table for one excitation, given the excitation, its header suffix and the counts."""


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
    excitations = driven_excitations(config)
    expected = (
        FREQUENCY_COLUMN,
        *(
            column
            for excitation in excitations
            for column in columns(
                excitation, excitation_suffix(excitation, excitations), counts
            )
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
    """The excitation indices of a Driven config, in order: each port excited by its own ``Index``.

    The column block of excitation e is then the one with port e driven.  An unexcited lumped element
    (``Excitation: false``) is no excitation; its columns are in every block.
    """
    boundaries = expect_mapping(config.get("Boundaries"), "config.Boundaries")
    rows = expect_list(boundaries.get("LumpedPort"), "config.Boundaries.LumpedPort")
    excitations: list[int] = []
    for row in rows:
        port = expect_mapping(row, "config.Boundaries.LumpedPort row")
        index = expect_int(port.get("Index"), "config LumpedPort Index")
        excitation = port.get("Excitation")
        if excitation is False:
            continue
        if type(excitation) is not int or excitation != index:
            raise ValueError(
                Declared(
                    f"{CONFIG} LumpedPort Index {index} Excitation", RETURNED_RUN_FOLDER
                ).refusal(
                    "must be the port's own Index, or false for a lumped element, as DrivenSim "
                    "writes it: " + found_text(excitation, index)
                )
            )
        excitations.append(index)
    if not excitations:
        raise ValueError(
            Declared(f"{CONFIG} Boundaries.LumpedPort", RETURNED_RUN_FOLDER).refusal(
                "excites no port, and a Driven run drives at least one"
            )
        )
    return tuple(sorted(excitations))


def excitation_suffix(excitation: int, excitations: tuple[int, ...]) -> str:
    """What follows a port, domain or row index in a header of excitation ``excitation``: ``[e]``, or nothing.

    Palace 0.18.1 writes the excitation into ``port-V``, ``port-I``, ``domain-E`` and ``surface-Q`` headers only
    when the run has more than one (``V_inc[2][2]``, ``p_surf[7][2]``); with one it writes ``V_inc[1]`` and
    ``p_surf[7]``.  ``port-S`` always writes it.
    """
    return f"[{excitation}]" if len(excitations) > 1 else ""


def requested_frequencies_ghz(config: Mapping[str, Any]) -> tuple[float, ...]:
    """The frequencies the sealed config asks Palace to solve at, in GHz.

    They are merged from the config's own ``Samples`` entries, not from the ranges read back in Hz, so each is the
    value Palace computed from the same numbers.
    """
    return palace_frequencies_ghz(_requested_samples(config).entries)


def requested_sweep(config: Mapping[str, Any]) -> DrivenSweep:
    """The sweep the sealed config requests, as ``DrivenSim.set_driven`` set it, frequencies in Hz."""
    requested = _requested_samples(config)
    return DrivenSweep(
        ranges=requested.ranges,
        adaptive_tolerance=requested.driven.get("AdaptiveTol"),
        adaptive_max_samples=requested.driven.get("AdaptiveMaxSamples"),
    )


class _RequestedSamples(NamedTuple):
    """The ``Samples`` entries of a Driven config, the ranges they were written from, and their ``Solver.Driven``."""

    entries: tuple[Mapping[str, Any], ...]
    ranges: tuple[FrequencyRange, ...]
    driven: dict[str, Any]


def _requested_samples(config: Mapping[str, Any]) -> _RequestedSamples:
    """The ``Samples`` entries ``DrivenSim`` writes, each read back as the range ``set_driven`` took, in Hz.

    Building each range checks its values, so the entries are ones Palace can merge.
    """
    solver = expect_mapping(config.get("Solver"), "config.Solver")
    driven = expect_mapping(solver.get(DRIVEN.name), f"config.Solver.{DRIVEN.name}")
    entries = expect_list(driven.get("Samples"), "config.Solver.Driven.Samples")
    return _RequestedSamples(
        tuple(entries), tuple(_requested_range(entry) for entry in entries), driven
    )


def _requested_range(entry: object) -> FrequencyRange:
    """One ``Samples`` entry as the range it was written from, in Hz; an entry ``DrivenSim`` never writes is refused."""
    sample = entry if isinstance(entry, dict) else {}
    kind = sample.get("Type")
    # Palace reads a missing AddToPROM as false; the range refuses one that is not a bool.
    full_solve = sample.get("AddToPROM", False)
    if kind == "Point":
        values = expect_list(sample.get("Freq"), "Samples Freq")
        return Points(
            [
                checked_number(value, "Samples Freq", positive=True) * 1e9
                for value in values
            ],
            full_solve=full_solve,
        )
    if kind in {"Linear", "Log"}:
        low = checked_number(sample.get("MinFreq"), "Samples MinFreq", positive=True)
        high = checked_number(sample.get("MaxFreq"), "Samples MaxFreq", positive=True)
        count = expect_int(sample.get("NSample"), "Samples NSample")
        band = Linear if kind == "Linear" else Log
        return band(low * 1e9, high * 1e9, count, full_solve=full_solve)
    raise ValueError(
        Declared(f"{CONFIG} Solver.Driven.Samples", RETURNED_RUN_FOLDER).refusal(
            "must hold only the Linear, Log and Point entries DrivenSim writes: "
            + found_text(entry)
        )
    )


def _port_s_columns(excitation: int, _suffix: str, counts: _Counts) -> list[str]:
    """``|S[o][e]|`` in dB and ``arg(S[o][e])`` in degrees at every lumped port o, with port e excited."""
    return [
        column
        for port in range(1, counts.ports + 1)
        for column in (
            PORT_S_MAGNITUDE_COLUMN.format(o=port, e=excitation),
            PORT_S_PHASE_COLUMN.format(o=port, e=excitation),
        )
    ]


def _port_wave_columns(kind: str, unit: str) -> _Columns:
    """The ``port-V`` or ``port-I`` block of one excitation: its incident value, then each lumped port's complex one."""

    def columns(excitation: int, suffix: str, counts: _Counts) -> list[str]:
        return [
            f"{kind}_inc[{excitation}]{suffix} ({unit})",
            *(
                column
                for port in range(1, counts.ports + 1)
                for column in (
                    f"Re{{{kind}[{port}]{suffix}}} ({unit})",
                    f"Im{{{kind}[{port}]{suffix}}} ({unit})",
                )
            ),
        ]

    return columns


def _domain_columns(_excitation: int, suffix: str, counts: _Counts) -> list[str]:
    """The field energies of one excitation, then each domain's energies and participations."""
    return [
        *(f"{name}{suffix} (J)" for name in ("E_elec", "E_mag", "E_cap", "E_ind")),
        *(
            column
            for domain in range(1, counts.domains + 1)
            for column in (
                f"E_elec[{domain}]{suffix} (J)",
                f"p_elec[{domain}]{suffix}",
                f"E_mag[{domain}]{suffix} (J)",
                f"p_mag[{domain}]{suffix}",
            )
        ),
    ]


def _surface_columns(_excitation: int, suffix: str, counts: _Counts) -> list[str]:
    """Each Surface-EPR row's participation and Q with one port excited."""
    return [
        column
        for surface in range(1, counts.surfaces + 1)
        for column in (f"p_surf[{surface}]{suffix}", f"Q_surf[{surface}]{suffix}")
    ]


_EXCITATION_COLUMNS: dict[str, _Columns] = {
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
        if abs(found - wanted) > printed_unit_ghz(wanted):
            raise ValueError(
                Declared(f"{table.path} row {row}", RETURNED_RUN_FOLDER).refusal(
                    f"is not at the requested frequency to the {PRINTED_DIGITS} significant digits Palace "
                    f"prints, in GHz: " + found_text(found, wanted)
                )
            )
