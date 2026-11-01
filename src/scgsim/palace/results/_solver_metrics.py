"""Read timing and resource cost out of Palace's own ``palace.json``.

Both builders refuse a payload with missing or non-finite numbers instead of filling
gaps with zeros, so a reported cost is always something the solver wrote.
"""

from __future__ import annotations

import math
from typing import Any

from ._expect import as_float_map, expect_int, to_float
from ._result_types import PalaceCost, PalacePerformance


def build_performance(palace_payload: dict[str, Any]) -> PalacePerformance:
    elapsed = palace_payload.get("ElapsedTime")
    if not isinstance(elapsed, dict):
        raise TypeError("palace.json missing ElapsedTime block.")

    counts_raw = elapsed.get("Counts")
    durations_raw = elapsed.get("Durations")
    if not isinstance(counts_raw, dict):
        raise TypeError("palace.json ElapsedTime.Counts must be a mapping.")
    if not isinstance(durations_raw, dict):
        raise TypeError("palace.json ElapsedTime.Durations must be a mapping.")

    counts: dict[str, int | float] = {
        str(key): float(value)
        for key, value in counts_raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    durations: dict[str, float] = {
        str(key): float(value)
        for key, value in durations_raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not counts or not durations:
        raise ValueError(
            "palace.json performance payload is missing numeric timing data."
        )
    if any(not math.isfinite(value) or value < 0 for value in counts.values()) or any(
        not math.isfinite(value) or value < 0 for value in durations.values()
    ):
        raise ValueError("palace.json timing payload contains invalid numeric data.")
    if (
        "Total" not in counts
        or "Total" not in durations
        or counts["Total"] < 1
        or durations["Total"] < 0
    ):
        raise ValueError("palace.json timing payload must include non-negative Total.")
    return PalacePerformance(counts=counts, durations=durations)


def build_cost(palace_payload: dict[str, Any]) -> PalaceCost:
    problem_block = palace_payload.get("Problem")
    if not isinstance(problem_block, dict):
        raise TypeError("palace.json missing Problem block.")

    linear_solver = palace_payload.get("LinearSolver")
    if not isinstance(linear_solver, dict):
        linear_solver = {}

    cost = PalaceCost(
        problem_degrees_of_freedom=expect_int(problem_block.get("DegreesOfFreedom")),
        mesh_elements=expect_int(problem_block.get("MeshElements")),
        mpi_size=expect_int(problem_block.get("MPISize")),
        openmp_threads=expect_int(problem_block.get("OpenMPThreads")),
        peak_memory_megabytes=as_float_map(
            palace_payload.get("PeakMemoryMegabytes"),
            ("Average", "Max", "Min", "Total"),
        ),
        peak_node_memory_megabytes=as_float_map(
            palace_payload.get("PeakNodeMemoryMegabytes"),
            ("Average", "Max", "Min", "Total"),
        ),
        linear_solver={
            key: to_float(value)
            for key, value in linear_solver.items()
            if isinstance(linear_solver, dict)
            and isinstance(key, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        },
        git_tag=(
            palace_payload["GitTag"]
            if isinstance(palace_payload.get("GitTag"), str)
            else None
        ),
    )
    if any(
        value is None
        for value in (
            cost.problem_degrees_of_freedom,
            cost.mesh_elements,
            cost.mpi_size,
            cost.openmp_threads,
            cost.git_tag,
        )
    ):
        raise ValueError("palace.json is missing required solver cost metadata.")
    if any(
        value is None or value <= 0
        for value in (
            cost.problem_degrees_of_freedom,
            cost.mesh_elements,
            cost.mpi_size,
            cost.openmp_threads,
        )
    ):
        raise ValueError("palace.json has invalid solver cost metadata.")
    if (
        not cost.linear_solver
        or any(
            value is None or not math.isfinite(value) or value < 0
            for value in cost.peak_memory_megabytes.values()
        )
        or any(
            value is None or not math.isfinite(value) or value < 0
            for value in cost.peak_node_memory_megabytes.values()
        )
        or any(
            not math.isfinite(value) or value < 0
            for value in cost.linear_solver.values()
        )
    ):
        raise ValueError("palace.json is missing required resource cost metadata.")
    return cost
