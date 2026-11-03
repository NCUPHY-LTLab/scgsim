"""Compare observed capacitance matrices using explicit identities and solve validity.

Adjacent change uses the current matrix norm. An explicitly supplied stable
reference uses the reference norm, after unit conversion and terminal alignment.
Raw capacitance remains available to callers even when its comparisons are not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from scgsim.palace.results.amr_models import (
    CAPACITANCE_UNIT_SCALES,
    AmrMatrixMetric,
    AmrMetric,
    AmrSolveObservation,
    CapacitanceComparison,
    CapacitanceMatrix,
    CapacitanceReference,
)
from scgsim.palace.results.report_models import AmrPassSnapshot


def compare_adjacent_capacitance(
    previous: AmrPassSnapshot,
    current: AmrPassSnapshot,
    solves: Sequence[AmrSolveObservation],
) -> CapacitanceComparison:
    """Compare two solved terminal matrices, normalizing by the current matrix."""
    reason = capacitance_validity(current, solves) or capacitance_validity(
        previous, solves
    )
    if reason is not None:
        return unavailable_comparison(reason, normalization="current")
    assert current.capacitance_matrix_f is not None
    assert previous.capacitance_matrix_f is not None
    assert current.terminal_ids is not None
    assert previous.terminal_ids is not None
    if set(current.terminal_ids) != set(previous.terminal_ids):
        return unavailable_comparison(
            "adjacent_terminal_identity_mismatch", normalization="current"
        )
    previous_values = _aligned_matrix(
        previous.capacitance_matrix_f, previous.terminal_ids, current.terminal_ids
    )
    return _compare_matrices(
        current.capacitance_matrix_f,
        previous_values,
        current.terminal_ids,
        normalization="current",
        source=previous.source,
    )


def compare_reference_capacitance(
    snapshot: AmrPassSnapshot,
    reference: CapacitanceReference,
    solves: Sequence[AmrSolveObservation],
) -> CapacitanceComparison:
    """Compare one pass to the named supplied reference, using its norm in farads."""
    reason = capacitance_validity(snapshot, solves)
    if reason is not None:
        return unavailable_comparison(
            reason, normalization="reference", source=reference.source
        )
    assert snapshot.capacitance_matrix_f is not None
    assert snapshot.terminal_ids is not None
    if set(snapshot.terminal_ids) != set(reference.terminal_ids):
        return unavailable_comparison(
            "reference_terminal_identity_mismatch",
            normalization="reference",
            source=reference.source,
        )
    scale = CAPACITANCE_UNIT_SCALES[reference.units]
    reference_f = tuple(
        tuple(value * scale for value in row) for row in reference.values
    )
    aligned = _aligned_matrix(
        reference_f, reference.terminal_ids, snapshot.terminal_ids
    )
    return _compare_matrices(
        snapshot.capacitance_matrix_f,
        aligned,
        snapshot.terminal_ids,
        normalization="reference",
        source=reference.source,
    )


def capacitance_validity(
    snapshot: AmrPassSnapshot, solves: Sequence[AmrSolveObservation]
) -> str | None:
    """Require a square identified matrix and native convergence for every terminal."""
    matrix = snapshot.capacitance_matrix_f
    if matrix is None:
        return "capacitance_unavailable"
    ids = snapshot.terminal_ids
    if ids is None:
        return "terminal_identity_unavailable"
    if len(set(ids)) != len(ids) or not ids or any(not name for name in ids):
        return "terminal_identity_invalid"
    if len(matrix) != len(ids) or any(len(row) != len(ids) for row in matrix):
        return "capacitance_shape_mismatch"
    if any(not math.isfinite(value) for row in matrix for value in row):
        return "nonfinite_capacitance"
    if snapshot.native_iteration is None:
        return "native_iteration_unavailable"
    observation, reason = solve_observation(snapshot, solves)
    if reason is not None:
        return reason
    assert observation is not None
    if set(observation.terminal_ids) != set(ids):
        return "linear_solve_terminal_identity_mismatch"
    return None


def solve_observation(
    snapshot: AmrPassSnapshot, solves: Sequence[AmrSolveObservation]
) -> tuple[AmrSolveObservation | None, str | None]:
    """Read exact per-terminal convergence without equating driver success with it."""
    matches = tuple(
        item for item in solves if item.native_iteration == snapshot.native_iteration
    )
    if not matches:
        return None, "linear_solve_validity_unavailable"
    if len(matches) != 1:
        return None, "duplicate_linear_solve_validity"
    observation = matches[0]
    if observation.state in {"started", "failed"} and not observation.converged:
        return observation, f"native_solve_{observation.state}"
    if observation.terminal_identity_reason is not None:
        return observation, observation.terminal_identity_reason
    if (
        not observation.terminal_ids
        or len(observation.terminal_ids) != len(observation.converged)
        or len(set(observation.terminal_ids)) != len(observation.terminal_ids)
    ):
        return observation, "linear_solve_validity_invalid"
    if snapshot.terminal_ids is None:
        return observation, "terminal_identity_unavailable"
    if set(snapshot.terminal_ids) != set(observation.terminal_ids):
        return observation, "linear_solve_terminal_identity_mismatch"
    if any(value is False for value in observation.converged):
        return observation, "linear_solve_unconverged"
    if any(value is not True for value in observation.converged):
        return observation, "linear_solve_validity_unavailable"
    return observation, None


def unavailable_comparison(
    reason: str,
    *,
    normalization: Literal["current", "reference"],
    source: str | None = None,
) -> CapacitanceComparison:
    """Keep undefined comparisons explicit rather than manufacturing zero change."""
    return CapacitanceComparison(
        relative_frobenius=AmrMetric(None, "unavailable", reason),
        absolute_delta_f=AmrMatrixMetric(None, None, "unavailable", reason),
        normalization=normalization,
        source=source,
    )


def _aligned_matrix(
    values: CapacitanceMatrix, source_ids: tuple[str, ...], target_ids: tuple[str, ...]
) -> CapacitanceMatrix:
    """Reorder both axes by terminal identity without changing physical units."""
    indices = tuple(source_ids.index(name) for name in target_ids)
    return tuple(tuple(values[row][column] for column in indices) for row in indices)


def _compare_matrices(
    current: CapacitanceMatrix,
    other: CapacitanceMatrix,
    terminal_ids: tuple[str, ...],
    *,
    normalization: Literal["current", "reference"],
    source: str,
) -> CapacitanceComparison:
    """Compute Frobenius change and absolute entry deltas using the named denominator."""
    if any(not math.isfinite(value) for row in other for value in row):
        return unavailable_comparison(
            "nonfinite_reference_capacitance",
            normalization=normalization,
            source=source,
        )
    delta = tuple(
        tuple(
            abs(value - reference)
            for value, reference in zip(row, reference_row, strict=True)
        )
        for row, reference_row in zip(current, other, strict=True)
    )
    matrix = AmrMatrixMetric(delta, terminal_ids, "available")
    numerator = math.hypot(*(value for row in delta for value in row))
    denominator_matrix = current if normalization == "current" else other
    denominator = math.hypot(*(value for row in denominator_matrix for value in row))
    if not math.isfinite(numerator):
        return unavailable_comparison(
            "nonfinite_capacitance_difference",
            normalization=normalization,
            source=source,
        )
    reason = f"zero_{normalization}_matrix_norm" if denominator == 0 else None
    if not math.isfinite(denominator):
        reason = f"nonfinite_{normalization}_matrix_norm"
    relative = (
        AmrMetric(None, "unavailable", reason)
        if reason
        else _relative_change(numerator, denominator)
    )
    return CapacitanceComparison(relative, matrix, normalization, source)


def _relative_change(numerator: float, denominator: float) -> AmrMetric:
    """Keep floating-point overflow distinct from a finite normalized change."""
    value = numerator / denominator
    return (
        AmrMetric(value, "available")
        if math.isfinite(value)
        else AmrMetric(None, "unavailable", "nonfinite_relative_capacitance")
    )
