"""Read ordered AMR passes while retaining final/iteration snapshot selection rules.

A pass is one Palace solve, and its number is the one its own ``palace.json`` records as
``Problem.Iteration``.  Palace 0.18.1 records 1 for the solve on the input mesh and k for the solve after
k - 1 refinements, and with ``SaveAdaptIterations`` copies solve k into ``iteration<k>`` (zero-padded to the
width of ``MaxIts``) before it refines; the last solve stays in ``results/palace`` only.  So ``iteration<k>``
records k and the final result records one more than the last refinement.  A folder whose record is missing,
or names another pass than its folder name, keeps a number from its place in the listing but carries an
``identity_problem``, so no comparison and no run of passes in a row is built on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

from scgsim.contracts.palace_run_folder import (
    NATIVE_METADATA_NAME,
    OUTPUT_DIR,
    iteration_number,
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


def _parent_has_physics(results: Path, problem: str) -> bool:
    """Check for the problem-specific final physics table before reading a parent snapshot."""
    return (results / table_name(palace_problem(problem).primary_table)).is_file()


def _same_primary_physics(left: AmrPassSnapshot, right: AmrPassSnapshot) -> bool:
    """Compare matching mode frequencies or capacitance matrices across two snapshots."""
    if left.frequencies_ghz is not None and right.frequencies_ghz is not None:
        return left.frequencies_ghz == right.frequencies_ghz
    if left.capacitance_matrix_f is not None and right.capacitance_matrix_f is not None:
        return left.capacitance_matrix_f == right.capacitance_matrix_f
    return False


def _same_margin_physics(
    left: tuple[SurfaceMarginEprSeriesSnapshot, ...] | None,
    right: tuple[SurfaceMarginEprSeriesSnapshot, ...] | None,
) -> bool:
    """Compare margin-row contributions while preserving unavailable margin data."""
    if left is None or right is None:
        return left is right
    return tuple(
        (snapshot.series_index, snapshot.series_kind, snapshot.records)
        for snapshot in left
    ) == tuple(
        (snapshot.series_index, snapshot.series_kind, snapshot.records)
        for snapshot in right
    )


def _problem_size_conflicts(left: AmrPassSnapshot, right: AmrPassSnapshot) -> bool:
    """Detect conflicting recorded mesh or solver sizes between otherwise matching snapshots."""
    return any(
        left_value is not None and right_value is not None and left_value != right_value
        for left_value, right_value in (
            (left.degrees_of_freedom, right.degrees_of_freedom),
            (left.mesh_elements, right.mesh_elements),
        )
    )


def collect_amr_passes(  # noqa: PLR0913 -- the run, its problem, both surface bindings and two read conditions
    root: Path,
    problem: str,
    surface_bindings: tuple[SurfaceBinding, ...] | None,
    surface_margin_bindings: tuple[SurfaceMarginBinding, ...] | None,
    *,
    failed_attempt: bool,
    mode_count: int | None = None,
) -> CollectedSnapshots:
    """Read the native AMR iterations in order, then the final result of the run folder.

    Each pass is numbered by what its folder records (``pass_identity``): ``iteration<N>`` must record ``N``,
    and the final result is the pass it records, not one after the largest folder number, so a folder that
    returned ``iteration1`` and the final result of pass 4 holds passes 1 and 4.

    The final result is the last iteration read again, and replaces it, only when all of these hold: both have
    the same primary physics, both are the same native iteration, and their margin physics agree.  In every
    other case it is appended after the iterations.  One exception: on a failed attempt, a final result with
    the last iteration's primary physics but a different recorded mesh or solver size makes the collection
    ``"unreadable"``, and the iterations are returned without it.

    A run folder with no ``results/palace`` folder, or whose final result has no physics table, is ``"missing"``; a final
    result that cannot be read is ``"unreadable"``.

    ``mode_count`` is the requested Eigenmode count when the run's config is known.
    Each pass then keeps those modes of the tables Palace wrote for every converged
    eigenpair, and a pass whose mode index is not those modes in order is unreadable.
    Without it a pass reports the rows it finds.
    """
    results = root / OUTPUT_DIR
    if not results.is_dir():
        return CollectedSnapshots((), "missing")
    iteration_dirs = []
    for child in results.iterdir():
        number = iteration_number(child.name)
        if number is not None and child.is_dir():
            iteration_dirs.append((number, child))
    iteration_dirs.sort()
    snapshots: list[AmrPassSnapshot] = []
    for index, path in iteration_dirs:
        identity = pass_identity(path, folder_number=index, fallback_number=index)
        snapshot = SnapshotReader(
            pass_index=identity.number - 1,
            source=path.name,
            path=path,
            problem=problem,
            surface_bindings=surface_bindings,
            surface_margin_bindings=surface_margin_bindings,
            mode_count=mode_count,
            identity_problem=identity.problem,
        ).load_optional()
        if snapshot is not None:
            snapshots.append(snapshot)
    if not _parent_has_physics(results, problem):
        return CollectedSnapshots(tuple(snapshots), "missing")
    identity = pass_identity(
        results,
        folder_number=None,
        fallback_number=max((index for index, _path in iteration_dirs), default=0) + 1,
    )
    parent = SnapshotReader(
        pass_index=identity.number - 1,
        source="final",
        path=results,
        problem=problem,
        surface_bindings=surface_bindings,
        surface_margin_bindings=surface_margin_bindings,
        mode_count=mode_count,
        identity_problem=identity.problem,
    ).load_optional()
    if parent is None:
        return CollectedSnapshots(tuple(snapshots), "unreadable")
    if snapshots and _same_primary_physics(snapshots[-1], parent):
        if failed_attempt and _problem_size_conflicts(snapshots[-1], parent):
            return CollectedSnapshots(tuple(snapshots), "unreadable")
        # Equal electrical values do not identify a mesh-refinement round.
        if (
            parent.native_iteration is None
            or snapshots[-1].native_iteration != parent.native_iteration
        ):
            return CollectedSnapshots((*snapshots, parent), "readable")
        parent = replace(parent, pass_index=snapshots[-1].pass_index)
        if _same_margin_physics(
            snapshots[-1].surface_margin_epr, parent.surface_margin_epr
        ):
            snapshots[-1] = replace(parent, pass_index=snapshots[-1].pass_index)
            return CollectedSnapshots(tuple(snapshots), "readable")
    return CollectedSnapshots((*snapshots, parent), "readable")


class PassIdentity(NamedTuple):
    """The pass a result folder is: its number, and why that number is not one the folder records, if so."""

    number: int
    problem: str | None


def pass_identity(
    folder: Path, *, folder_number: int | None, fallback_number: int
) -> PassIdentity:
    """The pass ``folder`` records about itself (``Problem.Iteration`` of its ``palace.json``).

    ``folder_number`` is the ``N`` of an ``iteration<N>`` folder (None for the final result), which must
    agree with the record.  Without a readable record, or when they disagree, the pass keeps
    ``fallback_number`` (its place in the listing) and the reason is returned with it.
    """
    try:
        payload = read_optional_json(folder / NATIVE_METADATA_NAME)
    except (OSError, TypeError, ValueError):
        payload = None
        reason = f"{_named(folder, folder_number)}: its {NATIVE_METADATA_NAME} cannot be read"
    else:
        reason = f"{_named(folder, folder_number)} has no {NATIVE_METADATA_NAME}"
    recorded = None
    if payload is not None:
        recorded = _native_iteration(payload.get("Problem"))
        reason = f"{_named(folder, folder_number)}: its {NATIVE_METADATA_NAME} records no Problem.Iteration"
    if recorded is None:
        return PassIdentity(fallback_number, reason)
    if folder_number is not None and recorded != folder_number:
        return PassIdentity(
            fallback_number,
            f"{folder.name} records Problem.Iteration {recorded}, not {folder_number}",
        )
    return PassIdentity(recorded, None)


def _named(folder: Path, folder_number: int | None) -> str:
    """How an identity reason names a pass folder: its ``iteration<N>`` name, or the final result."""
    return "the final result" if folder_number is None else folder.name


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
    identity_problem: str | None = None

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
            identity_problem=self.identity_problem,
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
