"""Read native Palace CSV tables and validate their problem-specific rows and units."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_index_map import (
    DIELECTRIC_SECTION,
    ENERGY_SECTION,
    LUMPED_PORT_SECTION,
    TERMINAL_SECTION,
)
from scgsim.contracts.palace_run_folder import (
    DRIVEN,
    EIGENMODE,
    ELECTROSTATIC,
    PalaceProblem,
    output_table,
    palace_problem,
)
from scgsim.palace.results.artifacts import (
    expect_int,
    expect_list,
    expect_mapping,
    expect_scalar,
)
from scgsim.palace.results.driven_tables import validate_driven_table
from scgsim.palace.results.resolved_records import ParsedTable
from scgsim.palace.results.table_rows import (
    require_finite_data,
    require_ordered_index,
    require_positive_index,
    require_surface_q_data,
    validate_error_indicators,
)
from scgsim.palace.run.archive import RETURNED_RUN_FOLDER
from scgsim.refusal import Declared, found_text, internal

READ_PROBLEMS = (ELECTROSTATIC.name, EIGENMODE.name, DRIVEN.name)
"""The problems whose returned run folders ``resolve_palace_result`` reads: every problem SCGSim writes."""
REPORTED_PROBLEMS = (ELECTROSTATIC.name, EIGENMODE.name)
"""The problems the trust report and the notebook views show.  A Driven run is read, and not shown yet."""


def read_problem(name: object, subject: str) -> PalaceProblem:
    """The problem record of the returned run folder ``subject`` names, for a problem SCGSim reads.

    Every reader of a returned run folder starts here, so a problem SCGSim prepares before it reads it is
    refused by name before any table of it is read as another problem's.

    Raises:
        ValueError: ``name`` is not a problem SCGSim writes.
        NotImplementedError: ``name`` is a problem SCGSim writes but does not read.
    """
    problem = palace_problem(name)
    if problem.name not in READ_PROBLEMS:
        raise NotImplementedError(
            Declared(
                f"the {problem.name} run folder {subject}",
                f"your own reading of the run: open {output_table(problem.primary_table)} and the "
                "other tables Palace wrote beside it",
            ).refusal(
                f"SCGSim prepares {problem.name} runs but does not read them back yet; it reads "
                f"{', '.join(READ_PROBLEMS)} runs"
            )
        )
    return problem


def reported_problem(name: object, subject: str) -> PalaceProblem:
    """The problem record of ``subject``, for a problem the trust report and the notebook views show.

    The report reads a run's tables as an Electrostatic or Eigenmode run's, so a Driven run is refused by name
    before any of its tables is read that way.

    Raises:
        ValueError: ``name`` is not a problem SCGSim writes.
        NotImplementedError: ``name`` is a problem SCGSim reads but does not show yet, Driven.
    """
    problem = read_problem(name, subject)
    if problem.name not in REPORTED_PROBLEMS:
        raise NotImplementedError(
            Declared(
                f"the {problem.name} run {subject}",
                "your notebook: read the run with resolve_palace_result(...).driven_result(), whose "
                "S-parameters, Surface EPR and adaptive sampling are the run's results",
            ).refusal(
                f"SCGSim reads {problem.name} runs and does not show a report or view of them yet; it shows "
                f"{' and '.join(REPORTED_PROBLEMS)} runs"
            )
        )
    return problem


def output_families(problem: str, index_counts: dict[str, int]) -> tuple[str, ...]:
    """The native result families of ``problem`` for the lumped ports and surfaces the index map lists."""
    return palace_problem(problem).written_families(
        lumped_ports=bool(index_counts["lumped_port"]),
        surfaces=bool(index_counts["surface"]),
    )


def validate_table(
    problem: str,
    table: ParsedTable,
    index_counts: dict[str, int],
    config: dict[str, Any],
) -> None:
    """Dispatch row validation by solver problem while preserving native table-family rules."""
    if not table.rows:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal("has no result rows")
        )

    # What a row is decides which validator reads the table: an excitation, a mode or a frequency.
    series = palace_problem(problem).series
    if series == "excitation":
        _validate_electrostatic_table(table, index_counts, config)
    elif series == "mode":
        _validate_eigenmode_table(table, index_counts, config)
    elif series == "frequency":
        validate_driven_table(table, index_counts, config)
    else:
        raise ValueError(
            internal(f"no table validator reads rows of series {series!r}")
        )


def _validate_matrix_table(
    table: ParsedTable,
    prefix: str,
    expected_count: int,
    *,
    include_index: bool = True,
) -> None:
    """Require a square indexed capacitance matrix with the exact native header prefix."""
    if len(table.rows) != expected_count:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                f"must contain {expected_count} data rows; found {len(table.rows)}"
            )
        )
    expected_cols = expected_count + 1
    if len(table.headers) != expected_cols:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                f"has {len(table.headers)} columns; expected {expected_cols}"
            )
        )

    if include_index and table.headers[0] != "i":
        raise ValueError(
            Declared(f"{table.path} row index column", RETURNED_RUN_FOLDER).refusal(
                "must be named 'i'"
            )
        )

    suffixes = {"C[i][": " (F)", "C_m[i][": " (F)", "C⁻¹[i][": " (1/F)"}
    suffix = suffixes.get(prefix)
    if suffix is None:
        raise ValueError(internal(f"unsupported native matrix prefix {prefix!r}"))
    expected_headers = (
        "i",
        *(f"{prefix}{column}]{suffix}" for column in range(1, expected_count + 1)),
    )
    if table.headers != expected_headers:
        raise ValueError(
            Declared(
                f"{table.path} native matrix headers", RETURNED_RUN_FOLDER
            ).refusal("are malformed")
        )

    for row in table.rows:
        require_finite_data(row)
        if include_index:
            require_positive_index(row.get("i"))
    require_ordered_index(table, "i", expected_count)


def _validate_terminal_voltage_table(table: ParsedTable, terminal_count: int) -> None:
    """Require ordered terminal-voltage rows with the configured terminal count."""
    if len(table.rows) != terminal_count:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                f"must contain {terminal_count} terminal voltage rows; found {len(table.rows)}"
            )
        )
    if len(table.headers) != 2 or table.headers[0] != "i":
        raise ValueError(
            Declared(
                f"{table.path} terminal voltage headers", RETURNED_RUN_FOLDER
            ).refusal("are malformed")
        )
    if table.headers[1] != "V_inc[i] (V)":
        raise ValueError(
            Declared(
                f"{table.path} terminal voltage header", RETURNED_RUN_FOLDER
            ).refusal("must be V_inc[i]")
        )
    for row in table.rows:
        require_finite_data(row)
        require_positive_index(row.get("i"))
    require_ordered_index(table, "i", terminal_count)


def _validate_eig_table(table: ParsedTable, expected_modes: int) -> None:
    """Require ordered mode rows and native frequency, Q and error observations."""
    if tuple(table.headers) != (
        "m",
        "Re{f} (GHz)",
        "Im{f} (GHz)",
        "Q",
        "Error (Bkwd.)",
        "Error (Abs.)",
    ):
        raise ValueError(
            Declared(f"{table.path} eig header", RETURNED_RUN_FOLDER).refusal(
                "does not match current Palace native schema"
            )
        )
    for row in table.rows:
        require_finite_data(
            {
                key: value
                for key, value in row.items()
                if not (key == "Q" and value == math.inf)
            }
        )
    if len(table.rows) != expected_modes:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                f"must contain {expected_modes} eig rows; found {len(table.rows)}"
            )
        )
    require_ordered_index(table, "m", expected_modes)


def _validate_port_table(
    table: ParsedTable, port_count: int, expected_modes: int, *, kind: str
) -> None:
    """Validate indexed port participation or complex current/voltage columns by mode."""
    if kind == "EPR":
        if port_count == 0:
            raise ValueError(
                Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                    "requires at least one lumped port"
                )
            )
        expected = 1 + port_count
        if len(table.headers) != expected:
            raise ValueError(
                Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                    f"has {len(table.headers)} columns; expected {expected}"
                )
            )
        expected_headers = ("m", *(f"p[{idx}]" for idx in range(1, port_count + 1)))
        if table.headers != expected_headers:
            raise ValueError(
                Declared(f"{table.path} port-EPR headers", RETURNED_RUN_FOLDER).refusal(
                    "are malformed"
                )
            )
        _validate_port_rows(table, expected)
    elif kind in {"I", "V"}:
        expected = 1 + 2 * port_count
        if len(table.headers) != expected:
            raise ValueError(
                Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                    f"has {len(table.headers)} columns; expected {expected}"
                )
            )
        unit = "A" if kind == "I" else "V"
        expected_headers = ["m"]
        for port in range(1, port_count + 1):
            expected_headers.extend(
                (f"Re{{{kind}[{port}]}} ({unit})", f"Im{{{kind}[{port}]}} ({unit})")
            )
        if table.headers != tuple(expected_headers):
            raise ValueError(
                Declared(
                    f"{table.path} port-{kind} headers", RETURNED_RUN_FOLDER
                ).refusal("are malformed")
            )
        _validate_port_rows(table, expected)
    else:
        raise ValueError(internal(f"unsupported port table kind {kind!r}"))
    if len(table.rows) != expected_modes:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                f"must contain {expected_modes} mode rows; found {len(table.rows)}"
            )
        )
    require_ordered_index(table, "m", expected_modes)


def _validate_domain_table(
    table: ParsedTable, domain_count: int, expected_rows: int
) -> None:
    """Validate total and per-domain energy columns against configured energy indices."""
    if domain_count <= 0:
        raise ValueError(
            Declared(
                "the index map configures no energy domain, but the domain-E table is present",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(domain_count))
        )
    expected = 1 + 4 + 4 * domain_count
    if len(table.headers) != expected:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                f"must contain {expected} columns for {domain_count} domains; found {len(table.headers)}"
            )
        )
    index_name = table.headers[0]
    if index_name not in {"i", "m"}:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                "has an invalid domain index column"
            )
        )
    expected_headers = [index_name, "E_elec (J)", "E_mag (J)", "E_cap (J)", "E_ind (J)"]
    for index in range(1, domain_count + 1):
        expected_headers.extend(
            (
                f"E_elec[{index}] (J)",
                f"p_elec[{index}]",
                f"E_mag[{index}] (J)",
                f"p_mag[{index}]",
            )
        )
    if table.headers != tuple(expected_headers):
        raise ValueError(
            Declared(
                f"{table.path} domain energy headers", RETURNED_RUN_FOLDER
            ).refusal("are malformed")
        )
    if len(table.rows) != expected_rows:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                f"must contain {expected_rows} rows; found {len(table.rows)}"
            )
        )
    for row in table.rows:
        require_finite_data(row)
    require_ordered_index(table, index_name, expected_rows)


def validate_surface_table(
    table: ParsedTable, surface_count: int, expected_rows: int
) -> None:
    """Validate baseline surface participation, keeping native lossless Q distinct from invalid data."""
    if surface_count <= 0:
        raise ValueError(
            Declared(
                "the index map configures no Surface-EPR row, but the surface-Q table is present",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(surface_count))
        )
    expected = 1 + 2 * surface_count
    if len(table.headers) != expected:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                f"must contain {expected} columns for {surface_count} surfaces; found {len(table.headers)}"
            )
        )
    index_name = table.headers[0]
    if index_name not in {"i", "m"}:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                "has an invalid surface index column"
            )
        )
    expected_headers = [index_name]
    for index in range(1, surface_count + 1):
        expected_headers.extend((f"p_surf[{index}]", f"Q_surf[{index}]"))
    if table.headers != tuple(expected_headers):
        raise ValueError(
            Declared(f"{table.path} surface Q headers", RETURNED_RUN_FOLDER).refusal(
                "are malformed"
            )
        )
    if len(table.rows) != expected_rows:
        raise ValueError(
            Declared(f"{table.path}", RETURNED_RUN_FOLDER).refusal(
                f"must contain {expected_rows} rows; found {len(table.rows)}"
            )
        )
    for row in table.rows:
        require_surface_q_data(row)
    require_ordered_index(table, index_name, expected_rows)


def read_csv_table(path: Path) -> ParsedTable:
    """Parse a native CSV artifact into typed cells without changing header or row order."""
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        rows = list(reader)

    if not rows:
        raise ValueError(
            Declared(f"{path}", RETURNED_RUN_FOLDER).refusal("has no header row")
        )

    raw_headers = tuple(cell.strip() for cell in rows[0])
    if not raw_headers or any(cell == "" for cell in raw_headers):
        raise ValueError(
            Declared(f"{path}", RETURNED_RUN_FOLDER).refusal(
                "must declare non-empty headers"
            )
        )
    if len(set(raw_headers)) != len(raw_headers):
        raise ValueError(
            Declared(f"{path}", RETURNED_RUN_FOLDER).refusal("has duplicate headers")
        )

    parsed_rows: list[dict[str, Any]] = []
    for row in rows[1:]:
        values = tuple(cell.strip() for cell in row)
        if not any(values):
            continue
        if len(values) != len(raw_headers):
            raise ValueError(
                Declared(f"{path}", RETURNED_RUN_FOLDER).refusal(
                    f"contains malformed row with {len(values)} cells; expected {len(raw_headers)}"
                )
            )
        parsed: dict[str, Any] = {}
        for name, value in zip(raw_headers, values, strict=False):
            parsed[name] = _parse_cell(value)
        parsed_rows.append(parsed)

    if not parsed_rows:
        raise ValueError(
            Declared(f"{path}", RETURNED_RUN_FOLDER).refusal("has no data rows")
        )
    return ParsedTable(
        name=path.stem, path=path, headers=raw_headers, rows=tuple(parsed_rows)
    )


# Palace writes one row per converged eigenpair in these tables, in mode order.
_MODE_FAMILIES = frozenset(
    {"eig", "port-EPR", "port-I", "port-V", "domain-E", "surface-Q"}
)


def read_result_table(
    path: Path,
    problem: str,
    index_counts: dict[str, int],
    config: dict[str, Any],
) -> ParsedTable:
    """Read one native result table as the sealed config and index map describe it."""
    table = read_csv_table(path)
    if palace_problem(problem).solves_modes and table.name in _MODE_FAMILIES:
        table = requested_modes(table, eigenmode_count(config))
    validate_table(problem, table, index_counts, config)
    return table


def requested_modes(table: ParsedTable, mode_count: int) -> ParsedTable:
    """Keep the requested modes of an Eigenmode table that reports every converged one.

    Palace solves for ``Solver.Eigenmode.N`` modes and reports each eigenpair that
    converged, so a mode table may hold more rows than were requested.  Only modes
    1..N belong to the result, so the rest are dropped unread.  The index must still
    be the complete ordered sequence over every reported row: a table that repeats or
    skips a mode is returned whole for the validator to refuse.
    """
    if len(table.rows) <= mode_count or table.headers[:1] != ("m",):
        return table
    if [row.get("m") for row in table.rows] != list(range(1, len(table.rows) + 1)):
        return table
    return replace(table, rows=table.rows[:mode_count])


def read_requested_modes(path: Path, mode_count: int) -> ParsedTable:
    """Read an Eigenmode mode table as exactly the requested modes.

    For a reader that takes columns without a family validator: once the modes
    nobody requested are dropped, the index must be 1..N.  A table that repeats,
    skips, reorders or lacks a requested mode is refused, never read by position.
    """
    table = requested_modes(read_csv_table(path), mode_count)
    require_ordered_index(table, "m", mode_count)
    return table


def configured_mode_count(config: Mapping[str, Any] | None) -> int | None:
    """Read the requested mode count of a config that an inspection may lack."""
    solver = config.get("Solver") if isinstance(config, Mapping) else None
    eigenmode = solver.get(EIGENMODE.name) if isinstance(solver, Mapping) else None
    count = eigenmode.get("N") if isinstance(eigenmode, Mapping) else None
    return count if type(count) is int and count >= 1 else None


_COUNTED_SECTIONS = {
    TERMINAL_SECTION: "terminal",
    LUMPED_PORT_SECTION: "lumped_port",
    ENERGY_SECTION: "domain",
    DIELECTRIC_SECTION: "surface",
}
"""The key ``count_index_entries`` counts each index-map section under."""


def count_index_entries(index_map: dict[str, Any]) -> dict[str, int]:
    """Count configured terminals, ports, energy domains and dielectric surfaces by index section."""
    entries = expect_list(index_map.get("entries"), "palace_index_map.entries")
    counts = dict.fromkeys(_COUNTED_SECTIONS.values(), 0)
    for entry in entries:
        data = expect_mapping(entry, "palace_index_map.entry")
        section = expect_scalar(data, "section", str, fallback="")
        if section in _COUNTED_SECTIONS:
            counts[_COUNTED_SECTIONS[section]] += 1
    return counts


def _parse_cell(value: str) -> Any:
    """Retain integer-looking indices, floating measurements and nonnumeric native text."""
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def eigenmode_count(config: dict[str, Any]) -> int:
    """Read the positive requested mode count from the sealed Eigenmode solver config."""
    solver = expect_mapping(config.get("Solver"), "config.Solver")
    eigenmode = expect_mapping(
        solver.get(EIGENMODE.name), f"config.Solver.{EIGENMODE.name}"
    )
    count = expect_int(eigenmode.get("N"), "config Solver.Eigenmode.N")
    if count is None or count < 1:
        raise ValueError(
            Declared(
                "config Solver.Eigenmode.N must be positive", RETURNED_RUN_FOLDER
            ).refusal(found_text(count))
        )
    return count


def _validate_electrostatic_table(
    table: ParsedTable, index_counts: dict[str, int], config: dict[str, Any]
) -> None:
    """Validate the named native table family without changing row order."""
    expected_rows = index_counts["terminal"]
    if table.name == "terminal-C":
        _validate_matrix_table(table, "C[i][", index_counts["terminal"])
    elif table.name == "terminal-Cm":
        _validate_matrix_table(table, "C_m[i][", index_counts["terminal"])
    elif table.name == "terminal-Cinv":
        _validate_matrix_table(table, "C⁻¹[i][", index_counts["terminal"])
    elif table.name == "terminal-V":
        _validate_terminal_voltage_table(table, index_counts["terminal"])
    elif table.name == "domain-E":
        _validate_domain_table(table, index_counts["domain"], expected_rows)
    elif table.name == "surface-Q":
        validate_surface_table(table, index_counts["surface"], expected_rows)
    elif table.name == "error-indicators":
        validate_error_indicators(table)


def _validate_eigenmode_table(
    table: ParsedTable, index_counts: dict[str, int], config: dict[str, Any]
) -> None:
    """Validate the named native table family without changing row order."""
    expected_rows = eigenmode_count(config)
    if table.name == "eig":
        _validate_eig_table(table, expected_rows)
    elif table.name == "port-EPR":
        _validate_port_table(
            table, index_counts["lumped_port"], expected_rows, kind="EPR"
        )
    elif table.name == "port-I":
        _validate_port_table(
            table, index_counts["lumped_port"], expected_rows, kind="I"
        )
    elif table.name == "port-V":
        _validate_port_table(
            table, index_counts["lumped_port"], expected_rows, kind="V"
        )
    elif table.name == "domain-E":
        _validate_domain_table(table, index_counts["domain"], expected_rows)
    elif table.name == "surface-Q":
        validate_surface_table(table, index_counts["surface"], expected_rows)
    elif table.name == "error-indicators":
        validate_error_indicators(table)


def _validate_port_rows(table: ParsedTable, expected: int) -> None:
    """Validate ordered finite native port rows with their declared width."""
    for idx, row in enumerate(table.rows, start=1):
        require_finite_data(row)
        if len(row) != expected:
            raise ValueError(
                Declared(f"{table.path} row {idx}", RETURNED_RUN_FOLDER).refusal(
                    "has malformed data width"
                )
            )
