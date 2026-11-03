"""Compute AMR transition metrics from exact owned-cell and solver observations.

M is the native marked-parent set and P is the actual refined-parent set.
No configured marking fraction or mesh growth substitutes for either set or
for the native squared-indicator sums.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scgsim.palace.results.amr_models import AmrMetric, AmrRefinementObservation


@dataclass(frozen=True)
class RefinementMetrics:
    """Refinement selection and closure quantities with independent availability."""

    marked_fraction: AmrMetric
    marked_squared_indicator_coverage: AmrMetric
    closure_amplification: AmrMetric
    closure_extra_parents: AmrMetric
    new_cell_amplification: AmrMetric


def refinement_metrics(
    observation: AmrRefinementObservation | None,
) -> RefinementMetrics:
    """Compute actual set-based refinement metrics without assuming M is within P."""
    if observation is None:
        return unavailable_refinement("native_refinement_diagnostics_unavailable")
    sets = (observation.marked_parents, observation.refined_parents)
    if any(
        cell.premesh_id != observation.premesh_id
        for cells in sets
        if cells is not None
        for cell in cells
    ):
        return unavailable_refinement("coarse_cell_mesh_identity_mismatch")
    marked = (
        None if observation.marked_parents is None else len(observation.marked_parents)
    )
    refined = (
        None
        if observation.refined_parents is None
        else len(observation.refined_parents)
    )
    marked_fraction = observed_ratio(
        marked,
        observation.cells_before,
        "marked_parent_set_unavailable",
        "zero_pre_refinement_cells",
    )
    coverage = observed_ratio(
        observation.marked_squared_indicator_sum,
        observation.total_squared_indicator_sum,
        "native_squared_indicator_sums_unavailable",
        "zero_squared_indicator_sum",
    )
    closure = observed_ratio(
        refined, marked, "native_parent_sets_unavailable", "zero_marked_parents"
    )
    if observation.marked_parents is None or observation.refined_parents is None:
        extra = unavailable_metric("native_parent_sets_unavailable", units="cells")
    else:
        extra = AmrMetric(
            len(observation.refined_parents - observation.marked_parents),
            "available",
            units="cells",
        )
    increase = observed_difference(
        observation.cells_after,
        observation.cells_before,
        "native_owned_cell_counts_unavailable",
    )
    amplification = observed_ratio(
        increase.value,
        marked,
        "native_owned_counts_or_marked_set_unavailable",
        "zero_marked_parents",
    )
    return RefinementMetrics(marked_fraction, coverage, closure, extra, amplification)


def unavailable_refinement(reason: str) -> RefinementMetrics:
    """Return unavailable refinement quantities without hiding missing diagnostics."""
    metric = unavailable_metric(reason)
    return RefinementMetrics(
        metric, metric, metric, unavailable_metric(reason, units="cells"), metric
    )


def observed_ratio(
    numerator: float | int | None,
    denominator: float | int | None,
    missing_reason: str,
    zero_reason: str,
) -> AmrMetric:
    """Divide actual observations; keep missing, zero and nonfinite cases explicit."""
    if numerator is None or denominator is None:
        return unavailable_metric(missing_reason)
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return unavailable_metric("nonfinite_native_observation")
    if denominator == 0:
        return unavailable_metric(zero_reason)
    value = numerator / denominator
    if not math.isfinite(value):
        return unavailable_metric("nonfinite_native_ratio")
    return AmrMetric(value, "available")


def observed_difference(
    current: int | None, previous: int | None, missing_reason: str, *, units: str = "1"
) -> AmrMetric:
    """Subtract recorded owned counts or true solver DOFs without using an estimate."""
    if current is None or previous is None:
        return unavailable_metric(missing_reason, units=units)
    return AmrMetric(current - previous, "available", units=units)


def unavailable_metric(reason: str, *, units: str = "1") -> AmrMetric:
    """Represent an undefined quantity independently from a valid numerical zero."""
    return AmrMetric(None, "unavailable", reason, units)
