"""Read the AMR passes of a returned run folder as detached snapshots, one per pass.

Which folder is which pass is decided once, by ``scgsim.contracts.palace_convergence.returned_passes``, from
the identity of the files (each folder's own ``Problem.Iteration``, links into ``iteration<k>/``, an unfinished
top level); the convergence check uses the same enumeration.  This module reads the tables of each pass it
returns and compares no physics to decide which folder is which.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

from scgsim.contracts.palace_convergence import ReturnedPasses, returned_passes
from scgsim.contracts.palace_run_folder import (
    NATIVE_METADATA_NAME,
    OUTPUT_DIR,
    palace_problem,
    table_name,
)
from scgsim.palace.results.artifacts import (
    as_int,
    mapping_max,
    read_optional_json,
)
from scgsim.palace.results.inspection_records import (
    AmrPassSnapshot,
    CollectedSnapshots,
    SurfaceEprSeriesSnapshot,
    SurfaceMarginEprSeriesSnapshot,
)
from scgsim.palace.results.surface_readers import (
    SurfaceBinding,
    SurfaceEprReader,
    SurfaceMarginBinding,
    SurfaceMarginEprReader,
)
from scgsim.palace.results.tables import read_csv_table, read_requested_modes

_FREQ_HEADER = "Re{f} (GHz)"


_CAP_HEADER_PREFIX = "C[i]["


_INDEX_COLUMNS = {"m", "i"}


_SKIP_EIG_COLUMNS = {"Error (Bkwd.)", "Error (Abs.)"}


_ERROR_INDICATOR_TRACES = ("Norm", "Maximum", "Mean")


def collect_amr_passes(
    root: Path,
    problem: str,
    surface_bindings: tuple[SurfaceBinding, ...] | None,
    surface_margin_bindings: tuple[SurfaceMarginBinding, ...] | None,
    *,
    mode_count: int | None = None,
) -> CollectedSnapshots:
    """Read every pass ``returned_passes`` finds in the run folder, in order of its number.

    A pass whose two folders hold different tables is left out (which of them is that solve cannot be told),
    and so is one whose snapshot cannot be read.  ``final_snapshot_status`` is ``"readable"`` when the top level
    of ``results/palace`` is a solve of its own and was read, ``"missing"`` when it holds no table or not the
    problem's own table, and ``"unreadable"`` otherwise: links into an iteration, a solve under way, a copy of
    an iteration, or a number not established; ``final_reason`` and ``set_aside`` say why.

    ``mode_count`` is the requested Eigenmode count when the run's config is known.
    Each pass then keeps those modes of the tables Palace wrote for every converged
    eigenpair, and a pass whose mode index is not those modes in order is unreadable.
    Without it a pass reports the rows it finds.
    """
    results = root / OUTPUT_DIR
    if not results.is_dir():
        return CollectedSnapshots((), "missing", "there is no results/palace folder")
    returned = returned_passes(results)
    snapshots: list[AmrPassSnapshot] = []
    final_read = False
    for folder in returned.passes:
        is_final = folder.path == results
        if folder.identity_problem is not None:
            continue
        if is_final and not _has_primary_table(results, problem):
            continue
        snapshot = SnapshotReader(
            pass_index=folder.number - 1,
            source="final" if is_final else folder.path.name,
            path=folder.path,
            problem=problem,
            surface_bindings=surface_bindings,
            surface_margin_bindings=surface_margin_bindings,
            mode_count=mode_count,
        ).load_optional()
        if snapshot is not None:
            snapshots.append(snapshot)
            final_read = final_read or is_final
    return CollectedSnapshots(
        tuple(snapshots),
        _final_status(returned, results, problem, final_read),
        returned.final_reason,
        returned.set_aside,
    )


def _has_primary_table(results: Path, problem: str) -> bool:
    """Whether a folder holds the problem's own table (``eig.csv`` or ``terminal-C.csv``)."""
    return (results / table_name(palace_problem(problem).primary_table)).is_file()


def _final_status(
    returned: ReturnedPasses, results: Path, problem: str, final_read: bool
) -> Literal["readable", "missing", "unreadable"]:
    """How the top level of ``results/palace`` reads: a solve of its own, nothing, or something else."""
    if returned.final == "missing" or (
        returned.final == "pass" and not _has_primary_table(results, problem)
    ):
        return "missing"
    return "readable" if returned.final == "pass" and final_read else "unreadable"


@dataclass(frozen=True)
class SnapshotReader:
    """One native pass location, problem and surface bindings read as a detached snapshot."""

    pass_index: int
    source: str
    path: Path
    problem: str
    surface_bindings: tuple[SurfaceBinding, ...] | None
    surface_margin_bindings: tuple[SurfaceMarginBinding, ...] | None
    mode_count: int | None = None

    def load_optional(self) -> AmrPassSnapshot | None:
        """Return no snapshot when native files are absent or unreadable."""
        try:
            return self.load()
        except (OSError, TypeError, ValueError):
            return None

    def load(self) -> AmrPassSnapshot:
        """Read native electrical, surface, mesh and cost observations for this AMR pass."""
        tables = self._electrical_tables()
        error_indicators = _read_error_indicators(
            self.path / table_name("error-indicators")
        )
        frequencies = (
            None if tables.eig_columns is None else tables.eig_columns.get(_FREQ_HEADER)
        )
        expected_rows = (
            len(frequencies)
            if frequencies is not None
            else len(tables.capacitance)
            if tables.capacitance is not None
            else 0
        )
        surface_epr = self._whole_surface_epr(frequencies, expected_rows)
        surface_margin_epr = self._margin_epr(expected_rows)
        error_norm = None if error_indicators is None else error_indicators.get("Norm")
        palace_payload = read_optional_json(self.path / NATIVE_METADATA_NAME)
        problem_block = palace_payload.get("Problem") if palace_payload else None
        return AmrPassSnapshot(
            pass_index=self.pass_index,
            source=self.source,
            path=self.path,
            frequencies_ghz=frequencies,
            eig_columns=tables.eig_columns,
            port_epr=tables.port_epr,
            capacitance_matrix_f=tables.capacitance,
            surface_epr=surface_epr,
            error_indicators=error_indicators,
            error_norm=error_norm,
            degrees_of_freedom=_problem_count(problem_block, "DegreesOfFreedom"),
            mesh_elements=_problem_count(problem_block, "MeshElements"),
            elapsed_total_s=_recorded_total_seconds(palace_payload),
            peak_node_memory_mb=mapping_max(
                palace_payload.get("PeakNodeMemoryMegabytes")
                if palace_payload
                else None
            ),
            surface_margin_epr=surface_margin_epr,
            native_iteration=_native_iteration(problem_block),
        )

    def _electrical_tables(self) -> _ElectricalTables:
        """The pass's mode tables (Eigenmode) or its capacitance matrix (Electrostatic)."""
        if palace_problem(self.problem).solves_modes:
            return _ElectricalTables(
                eig_columns=_read_numeric_table_columns(
                    self.path / table_name("eig"), self.mode_count
                ),
                port_epr=_read_numeric_table_columns(
                    self.path / table_name("port-EPR"), self.mode_count
                ),
                capacitance=None,
            )
        return _ElectricalTables(
            eig_columns=None,
            port_epr=None,
            capacitance=_read_capacitance(self.path / table_name("terminal-C")),
        )

    def _surface_count(self) -> int:
        """Margin rows are Dielectric rows of the same table, so both readers expect all of its columns."""
        return len(self.surface_bindings or ()) + len(
            self.surface_margin_bindings or ()
        )

    def _whole_surface_epr(
        self, frequencies: tuple[float, ...] | None, expected_rows: int
    ) -> tuple[SurfaceEprSeriesSnapshot, ...] | None:
        """Read the whole-surface Surface-EPR rows of this pass's ``surface-Q`` table."""
        return SurfaceEprReader(
            self.path / table_name("surface-Q"),
            problem=self.problem,
            pass_index=self.pass_index,
            source=self.source,
            frequencies_ghz=frequencies,
            expected_rows=expected_rows,
            bindings=self.surface_bindings,
            surface_count=self._surface_count(),
            mode_count=self.mode_count,
        ).read()

    def _margin_epr(
        self, expected_rows: int
    ) -> tuple[SurfaceMarginEprSeriesSnapshot, ...] | None:
        """Read the margin rows of this pass's ``surface-Q`` table."""
        return SurfaceMarginEprReader(
            self.path / table_name("surface-Q"),
            problem=self.problem,
            pass_index=self.pass_index,
            source=self.source,
            expected_rows=expected_rows,
            bindings=self.surface_margin_bindings,
            surface_count=self._surface_count(),
            mode_count=self.mode_count,
        ).read()


class _ElectricalTables(NamedTuple):
    """One pass's electrical tables: the mode columns and port EPR of an Eigenmode run, or the capacitances."""

    eig_columns: dict[str, tuple[float, ...]] | None
    port_epr: dict[str, tuple[float, ...]] | None
    capacitance: tuple[tuple[float, ...], ...] | None


def _problem_count(problem_block: Any, key: str) -> int | None:
    """An integer count of Palace's ``Problem`` record, None when the record or the count is absent."""
    return as_int(problem_block.get(key) if isinstance(problem_block, dict) else None)


def _recorded_total_seconds(palace_payload: dict[str, Any] | None) -> float | None:
    """The ``ElapsedTime.Durations.Total`` Palace recorded for a pass, None when it is absent."""
    if palace_payload is None:
        return None
    durations = palace_payload.get("ElapsedTime", {})
    if not isinstance(durations, dict):
        return None
    duration_map = durations.get("Durations")
    if isinstance(duration_map, dict) and isinstance(
        duration_map.get("Total"), (int, float)
    ):
        return float(duration_map["Total"])
    return None


def _native_iteration(problem: Any) -> int | None:
    """Retain a positive native identity without deriving it from folder numbering."""
    value = problem.get("Iteration") if isinstance(problem, dict) else None
    return value if type(value) is int and value > 0 else None


def _read_numeric_table_columns(
    path: Path, mode_count: int | None = None
) -> dict[str, tuple[float, ...]] | None:
    """Read result columns as numeric traces while excluding index and solver-error columns.

    With ``mode_count`` the table must hold the requested modes in order, so a pass
    whose mode index is wrong is unreadable instead of reported by row position.
    """
    if not path.is_file():
        return None
    table = (
        read_csv_table(path)
        if mode_count is None
        else read_requested_modes(path, mode_count)
    )
    columns: dict[str, tuple[float, ...]] = {}
    for header in table.headers:
        if header in _INDEX_COLUMNS:
            continue
        values: list[float] = []
        usable = True
        for row in table.rows:
            value = row.get(header)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                usable = False
                break
            values.append(float(value))
        if usable and values:
            columns[header] = tuple(values)
    return columns or None


def _read_capacitance(path: Path) -> tuple[tuple[float, ...], ...] | None:
    """Read a complete Maxwell capacitance matrix in the solver-stored farads."""
    if not path.is_file():
        return None
    table = read_csv_table(path)
    matrix: list[tuple[float, ...]] = []
    value_headers = [
        header for header in table.headers if header.startswith(_CAP_HEADER_PREFIX)
    ]
    if not value_headers:
        return None
    for row in table.rows:
        values: list[float] = []
        for header in value_headers:
            value = row.get(header)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None
            values.append(float(value))
        matrix.append(tuple(values))
    return tuple(matrix) if matrix else None


def surface_snapshots(
    passes: Sequence[AmrPassSnapshot],
) -> tuple[SurfaceEprSeriesSnapshot, ...]:
    """Flatten available baseline surface series in their recorded pass order."""
    return tuple(snapshot for pass_ in passes for snapshot in (pass_.surface_epr or ()))


def surface_margin_snapshots(
    passes: Sequence[AmrPassSnapshot],
) -> tuple[SurfaceMarginEprSeriesSnapshot, ...]:
    """Flatten available margin-row series without merging them into whole-surface rows."""
    return tuple(
        snapshot for pass_ in passes for snapshot in (pass_.surface_margin_epr or ())
    )


def _read_error_indicators(path: Path) -> dict[str, float] | None:
    """Read available AMR indicator values without requiring a completed solve."""
    if not path.is_file():
        return None
    table = read_csv_table(path)
    if not table.rows:
        return None
    payload: dict[str, float] = {}
    for header in table.headers:
        value = table.rows[0].get(header)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            payload[header] = float(value)
    return payload or None
