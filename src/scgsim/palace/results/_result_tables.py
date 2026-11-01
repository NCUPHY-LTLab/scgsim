"""Read and check the Palace result CSV tables (``results/palace/*.csv``).

Which tables must exist depends on the problem. Electrostatic runs report terminal
capacitance, Eigenmode runs report modes. Both report domain energies, surface Q and
error indicators. Each table is checked for the exact native header Palace writes,
the expected row count, and finite numbers. A row index column is ``i`` for
Electrostatic and ``m`` for Eigenmode.

Only a native ``Q`` may be ``+inf``. That is how Palace reports a mode without loss
(docs/specs/palace-eigenmode.qmd, "Native Eigenmode Q availability").
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._expect import expect_int, expect_mapping, expect_scalar
from ._index_map import IndexCounts, SurfaceMaskEntry
from ._result_types import ParsedTable

PROBLEM_FAMILIES = {
    "Electrostatic": (
        "terminal-C",
        "terminal-Cm",
        "terminal-Cinv",
        "terminal-V",
        "domain-E",
        "surface-Q",
        "error-indicators",
    ),
    "Eigenmode": (
        "eig",
        "port-EPR",
        "port-I",
        "port-V",
        "domain-E",
        "surface-Q",
        "error-indicators",
    ),
}


def output_families(problem: str, index_counts: IndexCounts) -> tuple[str, ...]:
    if problem != "Eigenmode":
        return PROBLEM_FAMILIES[problem]
    return (
        "eig",
        *(("port-EPR", "port-I", "port-V") if index_counts["lumped_port"] else ()),
        "domain-E",
        *(("surface-Q",) if index_counts["surface"] else ()),
        "error-indicators",
    )


def eigenmode_count(config: dict[str, Any]) -> int:
    solver = expect_mapping(config.get("Solver"), "config.Solver")
    eigenmode = expect_mapping(solver.get("Eigenmode"), "config.Solver.Eigenmode")
    count = expect_int(eigenmode.get("N"))
    if count is None or count < 1:
        raise ValueError("config Solver.Eigenmode.N must be positive.")
    return count


# ---------------------------------------------------------------- reading


def read_csv_table(path: Path) -> ParsedTable:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} has no header row.")

    raw_headers = tuple(cell.strip() for cell in rows[0])
    if not raw_headers or any(cell == "" for cell in raw_headers):
        raise ValueError(f"{path} must declare non-empty headers.")
    if len(set(raw_headers)) != len(raw_headers):
        raise ValueError(f"{path} has duplicate headers.")

    parsed_rows: list[dict[str, Any]] = []
    for row in rows[1:]:
        values = tuple(cell.strip() for cell in row)
        if not any(values):
            continue
        if len(values) != len(raw_headers):
            raise ValueError(
                f"{path} contains malformed row with {len(values)} cells; expected {len(raw_headers)}."
            )
        parsed: dict[str, Any] = {}
        for name, value in zip(raw_headers, values, strict=False):
            parsed[name] = _parse_cell(value)
        parsed_rows.append(parsed)

    if not parsed_rows:
        raise ValueError(f"{path} has no data rows.")
    return ParsedTable(
        name=path.stem, path=path, headers=raw_headers, rows=tuple(parsed_rows)
    )


def _parse_cell(value: str) -> Any:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


# ------------------------------------------------- per-problem table dispatch


# TODO(human): why a table name that matches no branch below is accepted unchecked.
# Today only the families in ``output_families`` reach here.
def validate_table(
    problem: str,
    table: ParsedTable,
    index_counts: IndexCounts,
    config: dict[str, Any],
) -> None:
    if not table.rows:
        raise ValueError(f"{table.path} has no result rows.")

    if problem == "Electrostatic":
        _validate_electrostatic_table(table, index_counts)
    elif problem == "Eigenmode":
        _validate_eigenmode_table(table, index_counts, config)


def _validate_electrostatic_table(
    table: ParsedTable, index_counts: IndexCounts
) -> None:
    # One row per terminal in every Electrostatic table.
    expected_rows = index_counts["terminal"]
    if table.name == "terminal-C":
        _validate_matrix_table(table, "C[i][", index_counts["terminal"])
    elif table.name == "terminal-Cm":
        _validate_matrix_table(table, "C_m[i][", index_counts["terminal"])
    elif table.name == "terminal-Cinv":
        _validate_matrix_table(table, "C⁻¹[i][", index_counts["terminal"])
    elif table.name == "terminal-V":
        _validate_terminal_voltage_table(table, index_counts["terminal"])
    else:
        _validate_shared_table(table, index_counts, expected_rows)


def _validate_eigenmode_table(
    table: ParsedTable, index_counts: IndexCounts, config: dict[str, Any]
) -> None:
    # One row per requested mode in every Eigenmode table.
    expected_rows = eigenmode_count(config)
    port_kinds = {"port-EPR": "EPR", "port-I": "I", "port-V": "V"}
    if table.name == "eig":
        _validate_eig_table(table, expected_rows)
    elif table.name in port_kinds:
        _validate_port_table(
            table,
            index_counts["lumped_port"],
            expected_rows,
            kind=port_kinds[table.name],
        )
    else:
        _validate_shared_table(table, index_counts, expected_rows)


def _validate_shared_table(
    table: ParsedTable, index_counts: IndexCounts, expected_rows: int
) -> None:
    """Tables both problems write; unknown names are accepted unchecked."""
    if table.name == "domain-E":
        _validate_domain_table(table, index_counts["domain"], expected_rows)
    elif table.name == "surface-Q":
        validate_surface_table(table, index_counts["surface"], expected_rows)
    elif table.name == "error-indicators":
        _validate_error_indicators(table)


# ----------------------------------------------------- Electrostatic tables


# TODO(human): confirm which capacitance matrix `terminal-C` (header `C[i][j]`) and
# `terminal-Cm` (header `C_m[i][j]`) are. The code only checks names and units.
def _validate_matrix_table(
    table: ParsedTable,
    prefix: str,
    expected_count: int,
    *,
    include_index: bool = True,
) -> None:
    if len(table.rows) != expected_count:
        raise ValueError(
            f"{table.path} must contain {expected_count} data rows; found {len(table.rows)}."
        )
    expected_cols = expected_count + 1
    if len(table.headers) != expected_cols:
        raise ValueError(
            f"{table.path} has {len(table.headers)} columns; expected {expected_cols}."
        )

    if include_index and table.headers[0] != "i":
        raise ValueError(f"{table.path} row index column must be named 'i'.")

    suffixes = {"C[i][": " (F)", "C_m[i][": " (F)", "C⁻¹[i][": " (1/F)"}
    suffix = suffixes.get(prefix)
    if suffix is None:
        raise ValueError(f"unsupported native matrix prefix {prefix!r}.")
    expected_headers = (
        "i",
        *(f"{prefix}{column}]{suffix}" for column in range(1, expected_count + 1)),
    )
    if table.headers != expected_headers:
        raise ValueError(f"{table.path} native matrix headers are malformed.")

    for row in table.rows:
        _require_finite_data(row)
        if include_index:
            _require_positive_index(row.get("i"))
    _require_ordered_index(table, "i", expected_count)


def _validate_terminal_voltage_table(table: ParsedTable, terminal_count: int) -> None:
    if len(table.rows) != terminal_count:
        raise ValueError(
            f"{table.path} must contain {terminal_count} terminal voltage rows; "
            f"found {len(table.rows)}."
        )
    if len(table.headers) != 2 or table.headers[0] != "i":
        raise ValueError(f"{table.path} terminal voltage headers are malformed.")
    if table.headers[1] != "V_inc[i] (V)":
        raise ValueError(f"{table.path} terminal voltage header must be V_inc[i].")
    for row in table.rows:
        _require_finite_data(row)
        _require_positive_index(row.get("i"))
    _require_ordered_index(table, "i", terminal_count)


# ------------------------------------------------------- Eigenmode tables


# A native Q of +inf is kept as reported, and every other column must be finite
# (docs/specs/palace-eigenmode.qmd, native Eigenmode Q availability).
def _validate_eig_table(table: ParsedTable, expected_modes: int) -> None:
    if tuple(table.headers) != (
        "m",
        "Re{f} (GHz)",
        "Im{f} (GHz)",
        "Q",
        "Error (Bkwd.)",
        "Error (Abs.)",
    ):
        raise ValueError(
            f"{table.path} eig header does not match current Palace native schema."
        )
    for row in table.rows:
        _require_finite_data(
            {
                key: value
                for key, value in row.items()
                if not (key == "Q" and value == math.inf)
            }
        )
    if len(table.rows) != expected_modes:
        raise ValueError(
            f"{table.path} must contain {expected_modes} eig rows; found {len(table.rows)}."
        )
    _require_ordered_index(table, "m", expected_modes)


def _require_port_rows(table: ParsedTable, expected_width: int) -> None:
    for idx, row in enumerate(table.rows, start=1):
        _require_finite_data(row)
        if len(row) != expected_width:
            raise ValueError(f"{table.path} row {idx} has malformed data width.")


def _validate_port_table(
    table: ParsedTable, port_count: int, expected_modes: int, *, kind: str
) -> None:
    if kind == "EPR":
        if port_count == 0:
            raise ValueError(f"{table.path} requires at least one lumped port.")
        expected = 1 + port_count
        if len(table.headers) != expected:
            raise ValueError(
                f"{table.path} has {len(table.headers)} columns; expected {expected}."
            )
        expected_headers = ("m", *(f"p[{idx}]" for idx in range(1, port_count + 1)))
        if table.headers != expected_headers:
            raise ValueError(f"{table.path} port-EPR headers are malformed.")
        _require_port_rows(table, expected)
    elif kind in {"I", "V"}:
        expected = 1 + 2 * port_count
        if len(table.headers) != expected:
            raise ValueError(
                f"{table.path} has {len(table.headers)} columns; expected {expected}."
            )
        unit = "A" if kind == "I" else "V"
        expected_headers = ["m"]
        for port in range(1, port_count + 1):
            expected_headers.extend(
                (f"Re{{{kind}[{port}]}} ({unit})", f"Im{{{kind}[{port}]}} ({unit})")
            )
        if table.headers != tuple(expected_headers):
            raise ValueError(f"{table.path} port-{kind} headers are malformed.")
        _require_port_rows(table, expected)
    else:
        raise ValueError(f"unsupported port table kind {kind!r}.")
    if len(table.rows) != expected_modes:
        raise ValueError(
            f"{table.path} must contain {expected_modes} mode rows; found {len(table.rows)}."
        )
    _require_ordered_index(table, "m", expected_modes)


# -------------------------------------------- tables both problems produce


def _validate_domain_table(
    table: ParsedTable, domain_count: int, expected_rows: int
) -> None:
    if domain_count <= 0:
        raise ValueError("domain count must be positive when domain table is present.")
    expected = 1 + 4 + 4 * domain_count
    if len(table.headers) != expected:
        raise ValueError(
            f"{table.path} must contain {expected} columns for {domain_count} domains;"
            f" found {len(table.headers)}."
        )
    index_name = table.headers[0]
    if index_name not in {"i", "m"}:
        raise ValueError(f"{table.path} has an invalid domain index column.")
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
        raise ValueError(f"{table.path} domain energy headers are malformed.")
    if len(table.rows) != expected_rows:
        raise ValueError(
            f"{table.path} must contain {expected_rows} rows; found {len(table.rows)}."
        )
    for row in table.rows:
        _require_finite_data(row)
    _require_ordered_index(table, index_name, expected_rows)


def validate_surface_table(
    table: ParsedTable, surface_count: int, expected_rows: int
) -> None:
    if surface_count <= 0:
        raise ValueError(
            "surface count must be positive when surface table is present."
        )
    expected = 1 + 2 * surface_count
    if len(table.headers) != expected:
        raise ValueError(
            f"{table.path} must contain {expected} columns for {surface_count} surfaces;"
            f" found {len(table.headers)}."
        )
    index_name = table.headers[0]
    if index_name not in {"i", "m"}:
        raise ValueError(f"{table.path} has an invalid surface index column.")
    expected_headers = [index_name]
    for index in range(1, surface_count + 1):
        expected_headers.extend((f"p_surf[{index}]", f"Q_surf[{index}]"))
    if table.headers != tuple(expected_headers):
        raise ValueError(f"{table.path} surface Q headers are malformed.")
    if len(table.rows) != expected_rows:
        raise ValueError(
            f"{table.path} must contain {expected_rows} rows; found {len(table.rows)}."
        )
    for row in table.rows:
        _require_surface_q_data(row)
    _require_ordered_index(table, index_name, expected_rows)


def _validate_error_indicators(table: ParsedTable) -> None:
    if tuple(table.headers) != ("Norm", "Minimum", "Maximum", "Mean"):
        raise ValueError(f"{table.path} error-indicators header is malformed.")
    for row in table.rows:
        _require_finite_data(row)
    if len(table.rows) != 1:
        raise ValueError(f"{table.path} must contain exactly one error indicator row.")


# ------------------------------------- surface-mask tables (masked Surface EPR)


def validate_surface_mask_table(
    table: ParsedTable,
    *,
    problem: str,
    expected_rows: int,
    entries: tuple[SurfaceMaskEntry, ...],
) -> None:
    index_name = "m" if problem == "Eigenmode" else "i"
    indexes = [expect_scalar(entry, "index", int) for entry in entries]
    if table.name == "surface-mask-Q":
        expected_headers = (
            index_name,
            *(
                header
                for index in indexes
                for header in (f"p_surf_mask[{index}]", f"Q_surf_mask[{index}]")
            ),
        )
    elif table.name == "surface-mask-energy":
        expected_headers = (
            index_name,
            *(f"E_surf_mask[{index}] (J)" for index in indexes),
        )
    else:
        raise ValueError(f"unsupported surface mask table {table.name!r}.")
    if table.headers != expected_headers:
        raise ValueError(f"{table.path} native mask headers are malformed.")
    if len(table.rows) != expected_rows:
        raise ValueError(f"{table.path} must contain {expected_rows} result rows.")
    _require_ordered_index(table, index_name, expected_rows)
    for row in table.rows:
        for header in table.headers[1:]:
            value = row.get(header)
            if (
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or math.isnan(value)
            ):
                raise ValueError(f"{table.path} contains an invalid mask value.")
            if header.startswith("Q_surf_mask["):
                if value <= 0 or (math.isinf(value) and value < 0):
                    raise ValueError(
                        f"{table.path} mask Q must be positive finite or +inf."
                    )
            elif not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{table.path} mask values must be finite and non-negative."
                )


# ----------------------------------------------------- shared row checks


def _require_finite_data(
    row: Mapping[str, Any],
    *,
    allow_infinite: bool = False,
    allow_non_numeric: bool = False,
) -> None:
    for key, value in row.items():
        if value is None:
            raise ValueError("result rows must not contain missing entries.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            if not allow_non_numeric:
                raise ValueError(f"result value in column {key!r} is not numeric.")
            continue
        if math.isnan(value) or (not allow_infinite and not math.isfinite(value)):
            raise ValueError(f"result value in column {key!r} is non-finite.")


# Surface-Q keeps its own contract (docs/specs/palace-eigenmode.qmd): the Q columns
# may be +inf, while participations may not be.
def _require_surface_q_data(row: Mapping[str, Any]) -> None:
    """Permit only native lossless-Q infinities; all participation stays finite."""

    for key, value in row.items():
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(f"result value in column {key!r} is not numeric.")
        if math.isnan(value):
            raise ValueError(f"result value in column {key!r} is non-finite.")
        if math.isinf(value) and (
            value < 0 or not (key == "Q" or key.startswith("Q_surf["))
        ):
            raise ValueError(
                f"only a positive solver-native Q may contain infinity; got {key!r}."
            )


def _require_ordered_index(
    table: ParsedTable, column: str, expected_count: int
) -> None:
    values = [row.get(column) for row in table.rows]
    expected = list(range(1, expected_count + 1))
    if values != expected:
        raise ValueError(
            f"{table.path} {column!r} values must be ordered 1..{expected_count}."
        )


def _require_positive_index(value: Any) -> None:
    if not isinstance(value, (int, float)) or value != int(value) or value <= 0:
        raise ValueError("matrix row index must be a positive integer.")
