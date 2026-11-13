"""Project detached Palace numerical and cost observations into report trace data."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

from scgsim.contracts.palace_run_folder import NATIVE_METADATA_NAME, palace_problem
from scgsim.palace.reports.plotting import Trace, TraceGroup
from scgsim.palace.results.artifacts import (
    as_int,
    mapping_max,
    read_optional_json,
)
from scgsim.palace.results.inspection_records import AmrPassSnapshot
from scgsim.palace.results.resolved_records import ResolvedRunData

FREQ_HEADER = "Re{f} (GHz)"


SKIP_EIG_COLUMNS = {"Error (Bkwd.)", "Error (Abs.)"}


ERROR_INDICATOR_TRACES = ("Norm", "Maximum", "Mean")


def scalar_physics(pass_: AmrPassSnapshot | None, problem: str) -> float | None:
    """Select the first mode frequency or largest absolute Maxwell diagonal for convergence."""
    if pass_ is None:
        return None
    if palace_problem(problem).solves_modes:
        if not pass_.frequencies_ghz:
            return None
        return pass_.frequencies_ghz[0]
    if pass_.capacitance_matrix_f is None:
        return None
    return _max_abs_diagonal(pass_.capacitance_matrix_f)


def _max_abs_diagonal(matrix: Sequence[Sequence[float]]) -> float:
    """Measure the largest absolute diagonal entry of a recorded capacitance matrix."""
    values = []
    for index, row in enumerate(matrix):
        if index < len(row):
            values.append(abs(row[index]))
    return max(values) if values else 0.0


def configured_refinement(config: dict[str, Any] | None) -> dict[str, Any]:
    """Read explicit refinement controls without treating them as adequacy thresholds."""
    if not config:
        return {}
    model = config.get("Model")
    if not isinstance(model, dict):
        return {}
    refinement = model.get("Refinement")
    return refinement if isinstance(refinement, dict) else {}


def latest_palace_json(
    passes: Sequence[AmrPassSnapshot], resolved: ResolvedRunData | None
) -> dict[str, Any] | None:
    """Select the latest readable native metadata or the already resolved final payload."""
    if resolved is not None:
        return resolved.provenance.palace_json
    for pass_ in reversed(passes):
        payload = read_optional_json(pass_.path / NATIVE_METADATA_NAME)
        if payload is not None:
            return payload
    return None


def recorded_durations(palace_payload: dict[str, Any] | None) -> dict[str, float]:
    """Project recorded elapsed durations, preserving unavailable observations."""
    if not palace_payload:
        return {}
    elapsed = palace_payload.get("ElapsedTime")
    if not isinstance(elapsed, dict):
        return {}
    raw = elapsed.get("Durations")
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


class CostCards(TypedDict, total=False):
    """The solver sizes, peak memory and parallel resources a report shows; empty when Palace recorded none."""

    problem_degrees_of_freedom: int | None
    mesh_elements: int | None
    mpi_size: int | None
    openmp_threads: int | None
    peak_memory_megabytes: float | None
    peak_node_memory_megabytes: float | None
    git_tag: str | None


def cost_cards(
    palace_payload: dict[str, Any] | None, resolved: ResolvedRunData | None
) -> CostCards:
    """Project native solver sizes, memory and parallel resources for readable cost cards."""
    if resolved is not None:
        return {
            "problem_degrees_of_freedom": resolved.cost.problem_degrees_of_freedom,
            "mesh_elements": resolved.cost.mesh_elements,
            "mpi_size": resolved.cost.mpi_size,
            "openmp_threads": resolved.cost.openmp_threads,
            "peak_memory_megabytes": mapping_max(resolved.cost.peak_memory_megabytes),
            "peak_node_memory_megabytes": mapping_max(
                resolved.cost.peak_node_memory_megabytes
            ),
            "git_tag": resolved.cost.git_tag,
        }
    if not palace_payload:
        return {}
    problem_block = palace_payload.get("Problem")
    problem_block = problem_block if isinstance(problem_block, dict) else {}
    return {
        "problem_degrees_of_freedom": as_int(problem_block.get("DegreesOfFreedom")),
        "mesh_elements": as_int(problem_block.get("MeshElements")),
        "mpi_size": as_int(problem_block.get("MPISize")),
        "openmp_threads": as_int(problem_block.get("OpenMPThreads")),
        "peak_memory_megabytes": mapping_max(palace_payload.get("PeakMemoryMegabytes")),
        "peak_node_memory_megabytes": mapping_max(
            palace_payload.get("PeakNodeMemoryMegabytes")
        ),
        "git_tag": palace_payload.get("GitTag")
        if isinstance(palace_payload.get("GitTag"), str)
        else None,
    }


def union_mapping_keys(
    passes: Sequence[AmrPassSnapshot], attribute: str
) -> tuple[str, ...]:
    """Preserve first-seen column order across available pass-local observation mappings."""
    keys: list[str] = []
    seen: set[str] = set()
    for pass_ in passes:
        mapping = getattr(pass_, attribute)
        if not isinstance(mapping, dict):
            continue
        for key in mapping:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return tuple(keys)


def mode_traces(
    passes: Sequence[AmrPassSnapshot],
    getter: Callable[[AmrPassSnapshot], Sequence[float] | None],
) -> list[Trace]:
    """Align each mode column across passes while retaining unavailable pass values."""
    mode_count = max((len(getter(pass_) or ()) for pass_ in passes), default=0)
    traces: list[Trace] = []
    for mode in range(mode_count):
        values = []
        for pass_ in passes:
            column = getter(pass_)
            if column is None or mode >= len(column):
                values.append(None)
            else:
                values.append(column[mode])
        traces.append(Trace(f"Mode {mode + 1}", values))
    return traces


def traces_are_finite(traces: Sequence[Trace]) -> bool:
    """Check whether every plotted trace value is available and finite."""
    numbers = [
        value
        for _name, ys in traces
        for value in ys
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return bool(numbers) and all(math.isfinite(value) for value in numbers)


def native_positive_infinity_q_notice(
    traces: Sequence[Trace],
) -> str | None:
    """Describe native lossless positive-infinity Q without attempting a finite convergence plot."""
    values = [value for _name, ys in traces for value in ys]
    contains_positive_infinity = any(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == math.inf
        for value in values
    )
    finite_or_positive_infinity_only = all(
        value is None
        or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (math.isfinite(value) or value == math.inf)
        )
        for value in values
    )
    if not contains_positive_infinity or not finite_or_positive_infinity_only:
        return None
    return (
        "<p style='opacity:0.75'>Native-reported Q includes +inf; "
        "a finite-Q convergence view is unavailable.</p>"
    )


def delta_traces(
    traces: Sequence[Trace],
) -> list[Trace]:
    """Compute successive absolute trace differences for the convergence view."""
    deltas: list[Trace] = []
    for name, ys in traces:
        series: list[float | None] = []
        previous: float | None = None
        for value in ys:
            if previous is None or value is None:
                series.append(None)
            else:
                series.append(abs(value - previous))
            if value is not None:
                previous = value
        deltas.append(Trace(name, series[1:]))
    return deltas


def capacitance_traces(
    passes: Sequence[AmrPassSnapshot],
) -> list[TraceGroup]:
    """Project individual Maxwell coefficients and their diagonal scale in femtofarads."""
    rows = max(
        (
            len(pass_.capacitance_matrix_f)
            for pass_ in passes
            if pass_.capacitance_matrix_f
        ),
        default=0,
    )
    cols = max(
        (
            len(row)
            for pass_ in passes
            if pass_.capacitance_matrix_f
            for row in pass_.capacitance_matrix_f
        ),
        default=0,
    )
    if rows == 0 or cols == 0:
        return []
    all_traces: list[Trace] = []
    per_element: list[TraceGroup] = []
    for row in range(rows):
        for col in range(cols):
            name = f"C[{row + 1}][{col + 1}]"
            values = []
            for pass_ in passes:
                matrix = pass_.capacitance_matrix_f
                if matrix is None or row >= len(matrix) or col >= len(matrix[row]):
                    values.append(None)
                else:
                    values.append(matrix[row][col] * 1e15)
            trace = Trace(name, values)
            all_traces.append(trace)
            per_element.append(TraceGroup(name, [trace]))
    if len(all_traces) > 9:
        return [TraceGroup("Maxwell C_ij", all_traces)]
    return per_element
