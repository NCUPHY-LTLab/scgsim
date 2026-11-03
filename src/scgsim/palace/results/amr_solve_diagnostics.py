"""Bind native terminal Krylov observations and capacitance attribution by iteration.

Main convergence qualifies capacitance comparisons. Recovery convergence belongs
to ZZ-estimator interpretation and is retained separately, never added as a C gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from scgsim.palace.results.amr_diagnostic_io import (
    NativeDiagnosticReader,
    native_float,
    native_indices,
    native_int,
    native_text,
)
from scgsim.palace.results.amr_models import (
    AmrLinearSolveStatistics,
    AmrSolveObservation,
    AmrTerminalSolveObservation,
    NativeResidual,
)


def read_solve_observation(
    reader: NativeDiagnosticReader, relative: str
) -> AmrSolveObservation | None:
    """Read one inventoried attempt, including attempts with no attributable C yet."""
    record = reader.read_record(relative, "solve")
    if record is None:
        return None
    iteration = native_int(record, "iteration", minimum=1)
    mesh_id = native_text(record, "mesh_id")
    state = native_text(record, "state")
    if state not in {"started", "completed", "failed"}:
        raise ValueError("invalid native AMR solve state")
    reader.referenced_paths(record, "mesh_paths")
    order = (
        None
        if record["capacitance_terminal_order"] is None
        else native_indices(record, "capacitance_terminal_order")
    )
    written = _native_bool(record, "capacitance_written")
    cap_iteration = (
        None
        if record["capacitance_iteration"] is None
        else native_int(record, "capacitance_iteration", minimum=1)
    )
    if written and (cap_iteration != iteration or order is None):
        raise ValueError("native AMR capacitance attribution disagrees with its solve")
    terminals = []
    for path in reader.referenced_paths(record, "terminal_records"):
        terminal = _read_terminal(reader, path, iteration, mesh_id)
        if terminal is not None:
            terminals.append(terminal)
    terminals.sort(key=lambda item: item.capacitance_column)
    _validate_terminal_columns(tuple(terminals), order)
    native_order = (
        order if order is not None else tuple(item.terminal_index for item in terminals)
    )
    by_column = {terminal.capacitance_column: terminal for terminal in terminals}
    ids = tuple(reader.source.terminal_net_ids.get(index) for index in native_order)
    known_ids = all(isinstance(name, str) and bool(name) for name in ids)
    convergence = tuple(
        by_column[column].main.converged if column in by_column else None
        for column in range(len(native_order))
    )
    recovery = tuple(
        by_column[column].recovery.converged if column in by_column else None
        for column in range(len(native_order))
    )
    return AmrSolveObservation(
        native_iteration=iteration,
        terminal_ids=cast(tuple[str, ...], ids) if known_ids else (),
        converged=convergence,
        source=reader.path(relative)
        .relative_to(reader.source.run_dir.resolve())
        .as_posix(),
        recovery_converged=recovery,
        terminal_statistics=tuple(terminals),
        terminal_identity_reason=(
            "capacitance_terminal_order_unavailable"
            if order is None and terminals
            else None
            if known_ids
            else "terminal_identity_unavailable"
        ),
        mesh_id=mesh_id,
        state=state,
        owned_cells=native_int(record, "owned_global_cells"),
        true_dofs=None
        if record["true_dofs"] is None
        else native_int(record, "true_dofs"),
        capacitance_written=written,
        capacitance_iteration=cap_iteration,
        capacitance_terminal_order=order,
        indicator_norm=None
        if record.get("indicator_norm") is None
        else native_float(record, "indicator_norm"),
        indicator_semantics=None
        if record.get("indicator_semantics") is None
        else native_text(record, "indicator_semantics"),
    )


def _read_terminal(
    reader: NativeDiagnosticReader, relative: str, iteration: int, mesh_id: str
) -> AmrTerminalSolveObservation | None:
    """Bind native terminal IDs and zero-based columns without subtracting one."""
    record = reader.read_record(relative, "terminal")
    if record is None:
        return None
    if (
        native_int(record, "iteration", minimum=1) != iteration
        or record.get("mesh_id") != mesh_id
    ):
        raise ValueError("native AMR terminal belongs to another solve or mesh")
    terminal = native_int(record, "terminal")
    statistics = record["statistics"]
    if (
        not isinstance(statistics, Mapping)
        or native_int(statistics, "terminal_index") != terminal
    ):
        raise ValueError("native AMR terminal statistics identity mismatch")
    return AmrTerminalSolveObservation(
        terminal_index=terminal,
        capacitance_column=native_int(statistics, "capacitance_column"),
        terminal_id=reader.source.terminal_net_ids.get(terminal),
        main=_linear_statistics(statistics["main"]),
        recovery=_linear_statistics(statistics["recovery"]),
        source=reader.path(relative)
        .relative_to(reader.source.run_dir.resolve())
        .as_posix(),
    )


def _validate_terminal_columns(
    terminals: tuple[AmrTerminalSolveObservation, ...], order: tuple[int, ...] | None
) -> None:
    """Require unique columns and exact correspondence to the supplied native order."""
    columns = tuple(item.capacitance_column for item in terminals)
    indices = tuple(item.terminal_index for item in terminals)
    if len(set(columns)) != len(columns) or len(set(indices)) != len(indices):
        raise ValueError("native AMR terminal records repeat a column or terminal")
    if order is not None and any(
        item.capacitance_column >= len(order)
        or order[item.capacitance_column] != item.terminal_index
        for item in terminals
    ):
        raise ValueError("native AMR terminal column disagrees with capacitance order")


def _linear_statistics(record: Any) -> AmrLinearSolveStatistics:
    """Preserve native residual definitions and nonfinite strings without reinterpretation."""
    if not isinstance(record, Mapping):
        raise TypeError("native AMR linear-solve statistics must be a mapping")
    available = _native_bool(record, "available")
    if not available:
        return AmrLinearSolveStatistics(False)
    algorithm = native_text(record, "algorithm")
    reference_source = native_text(record, "reference_residual_source")
    if algorithm not in {"cg", "gmres", "fgmres"} or reference_source not in {
        "rhs",
        "initial_residual",
    }:
        raise ValueError("unsupported native AMR linear-solve statistics semantics")
    return AmrLinearSolveStatistics(
        available=True,
        algorithm=algorithm,
        residual_norm=native_text(record, "residual_norm"),
        final_residual_is_estimate=_native_bool(record, "final_residual_is_estimate"),
        iterations=native_int(record, "iterations"),
        initial_residual=_residual(record, "initial_residual"),
        final_residual=_residual(record, "final_residual"),
        reference_residual=_residual(record, "reference_residual"),
        reference_residual_source=reference_source,
        relative_tolerance=native_float(record, "relative_tolerance"),
        absolute_tolerance=native_float(record, "absolute_tolerance"),
        maximum_iterations=native_int(record, "maximum_iterations", minimum=None),
        converged=_native_bool(record, "converged"),
    )


def _native_bool(record: Mapping[str, Any], field: str) -> bool:
    """Read an explicitly emitted native boolean rather than coercing another value."""
    value = record.get(field)
    if not isinstance(value, bool):
        raise TypeError(f"native AMR {field} must be a boolean")
    return value


def _residual(record: Mapping[str, Any], field: str) -> NativeResidual:
    """Keep producer-encoded NaN and infinity as strings in detached residual data."""
    value = record.get(field)
    if isinstance(value, str) and value in {"NaN", "Infinity", "-Infinity"}:
        return cast(NativeResidual, value)
    return native_float(record, field)
