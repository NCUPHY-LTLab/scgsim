"""Build detached AMR pass and transition analysis from verified source records.

This layer has no filesystem or solver access. Native iteration identity defines
adjacency; identical capacitance values do not collapse genuine solver rounds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scgsim.palace.results.amr_capacitance import (
    compare_adjacent_capacitance,
    compare_reference_capacitance,
    solve_observation,
    unavailable_comparison,
)
from scgsim.palace.results.amr_metrics import (
    observed_difference,
    observed_ratio,
    refinement_metrics,
    unavailable_metric,
    unavailable_refinement,
)
from scgsim.palace.results.amr_models import (
    AmrDiagnosticObservations,
    AmrMetric,
    AmrPassAnalysis,
    AmrRefinementObservation,
    AmrSolveObservation,
    AmrTransitionAnalysis,
    AmrTransitionEvidence,
    CapacitanceReference,
    PalaceAmrAnalysis,
)
from scgsim.palace.results.report_models import AmrPassSnapshot


@dataclass(frozen=True)
class AmrAnalysisContext:
    """Source location and run/identity state retained independently of metrics."""

    run_dir: Path
    problem: str
    run_status: str
    integrity: str


def analyze_snapshot_series(
    snapshots: Sequence[AmrPassSnapshot],
    context: AmrAnalysisContext,
    diagnostics: AmrDiagnosticObservations | None = None,
    reference: CapacitanceReference | None = None,
) -> PalaceAmrAnalysis:
    """Keep raw pass observations and compute only genuinely adjacent transitions."""
    if reference is not None and context.problem != "Electrostatic":
        raise ValueError("a capacitance reference requires an Electrostatic analysis")
    observed = diagnostics if diagnostics is not None else AmrDiagnosticObservations()
    passes = tuple(
        _analyze_pass(snapshot, observed, reference) for snapshot in snapshots
    )
    transitions = tuple(
        _analyze_transition(previous, current, observed)
        for previous, current in zip(snapshots, snapshots[1:], strict=False)
    )
    return PalaceAmrAnalysis(
        run_dir=context.run_dir,
        problem=context.problem,
        run_status=context.run_status,
        integrity=context.integrity,
        passes=passes,
        transitions=transitions,
        reference=reference,
        diagnostics_reason=observed.reason,
        diagnostic_missing_files=observed.missing_files,
    )


def _analyze_pass(
    snapshot: AmrPassSnapshot,
    diagnostics: AmrDiagnosticObservations,
    reference: CapacitanceReference | None,
) -> AmrPassAnalysis:
    """Retain capacitance and native ZZ norm even when comparison validity is absent."""
    observation, reason = solve_observation(snapshot, diagnostics.solves)
    state = (
        "converged"
        if reason is None
        else "unconverged"
        if reason == "linear_solve_unconverged"
        else "unavailable"
    )
    reference_comparison = (
        None
        if reference is None
        else compare_reference_capacitance(snapshot, reference, diagnostics.solves)
    )
    norm = (
        unavailable_metric("native_zz_norm_unavailable")
        if snapshot.error_norm is None
        else unavailable_metric("nonfinite_native_zz_norm")
        if not math.isfinite(snapshot.error_norm)
        else AmrMetric(snapshot.error_norm, "available")
    )
    recovery_state, recovery_reason = _recovery_state(snapshot, observation)
    capacitance_reason = None
    if observation is not None and observation.capacitance_written is False:
        capacitance_reason = "capacitance_not_written_for_native_attempt"
    elif snapshot.capacitance_matrix_f is None:
        capacitance_reason = "capacitance_artifact_unavailable"
    return AmrPassAnalysis(
        native_iteration=snapshot.native_iteration,
        source=snapshot.source,
        capacitance_matrix_f=snapshot.capacitance_matrix_f,
        terminal_ids=snapshot.terminal_ids,
        native_zz_norm=norm,
        reference_capacitance=reference_comparison,
        linear_solve_state=state,
        linear_solve_reason=reason,
        terminal_solve_observation=observation,
        recovery_solve_state=recovery_state,
        recovery_solve_reason=recovery_reason,
        capacitance_availability_reason=capacitance_reason,
        degrees_of_freedom=snapshot.degrees_of_freedom,
        mesh_elements=snapshot.mesh_elements,
        elapsed_cumulative_s=snapshot.elapsed_total_s,
        peak_node_memory_mb=snapshot.peak_node_memory_mb,
    )


def _analyze_transition(
    previous: AmrPassSnapshot,
    current: AmrPassSnapshot,
    diagnostics: AmrDiagnosticObservations,
) -> AmrTransitionAnalysis:
    """Distinguish native iteration gaps from unavailable optional diagnostics."""
    reason = _adjacency_reason(previous.native_iteration, current.native_iteration)
    observation, observation_reason = _refinement_observation(
        previous, current, diagnostics
    )
    refinement_reason = reason or observation_reason
    refinement = (
        unavailable_refinement(refinement_reason)
        if refinement_reason is not None
        else refinement_metrics(observation)
    )
    if reason is not None:
        adjacent = unavailable_comparison(reason, normalization="current")
        growth = increase = unavailable_metric(reason)
    else:
        adjacent = compare_adjacent_capacitance(previous, current, diagnostics.solves)
        growth = observed_ratio(
            current.degrees_of_freedom,
            previous.degrees_of_freedom,
            "native_true_dof_counts_unavailable",
            "zero_previous_true_dofs",
        )
        increase = observed_difference(
            current.degrees_of_freedom,
            previous.degrees_of_freedom,
            "native_true_dof_counts_unavailable",
            units="DOF",
        )
    return AmrTransitionAnalysis(
        before_iteration=previous.native_iteration,
        after_iteration=current.native_iteration,
        availability="unavailable" if reason else "available",
        reason=reason,
        marked_fraction=refinement.marked_fraction,
        marked_squared_indicator_coverage=refinement.marked_squared_indicator_coverage,
        closure_amplification=refinement.closure_amplification,
        closure_extra_parents=refinement.closure_extra_parents,
        new_cell_amplification=refinement.new_cell_amplification,
        dof_growth_ratio=growth,
        dof_increase=increase,
        adjacent_capacitance=adjacent,
        observations=_transition_evidence(observation, previous, current),
    )


def _adjacency_reason(before: int | None, after: int | None) -> str | None:
    """Require exact consecutive native identities without using directory ordinals."""
    if before is None or after is None:
        return "native_iteration_unavailable"
    if before == after:
        return "duplicate_native_iteration"
    if after != before + 1:
        return "native_iteration_gap"
    return None


def _refinement_observation(
    previous: AmrPassSnapshot,
    current: AmrPassSnapshot,
    diagnostics: AmrDiagnosticObservations,
) -> tuple[AmrRefinementObservation | None, str | None]:
    """Match diagnostics to the same native transition and owned-cell observations."""
    matches = tuple(
        observation
        for observation in diagnostics.refinements
        if observation.before_iteration == previous.native_iteration
        and observation.after_iteration == current.native_iteration
    )
    if not matches:
        return None, diagnostics.reason or "native_refinement_diagnostics_unavailable"
    if len(matches) != 1:
        return None, "duplicate_native_refinement_diagnostics"
    observation = matches[0]
    if any(
        reported is not None and native is not None and reported != native
        for reported, native in (
            (previous.mesh_elements, observation.cells_before),
            (current.mesh_elements, observation.cells_after),
        )
    ):
        return None, "native_transition_cell_count_mismatch"
    return observation, None


def _recovery_state(
    snapshot: AmrPassSnapshot, observation: AmrSolveObservation | None
) -> tuple[Literal["converged", "unconverged", "unavailable"], str | None]:
    """Keep recovery validity separate from the main capacitance solve and raw ZZ value."""
    if observation is None or not observation.recovery_converged:
        return "unavailable", "recovery_solve_validity_unavailable"
    if (
        snapshot.terminal_ids is None
        or observation.terminal_identity_reason is not None
    ):
        return (
            "unavailable",
            observation.terminal_identity_reason or "terminal_identity_unavailable",
        )
    if set(snapshot.terminal_ids) != set(observation.terminal_ids):
        return "unavailable", "recovery_solve_terminal_identity_mismatch"
    if len(observation.recovery_converged) != len(observation.terminal_ids):
        return "unavailable", "recovery_solve_validity_invalid"
    if any(value is False for value in observation.recovery_converged):
        return "unconverged", "recovery_solve_unconverged"
    if any(value is not True for value in observation.recovery_converged):
        return "unavailable", "recovery_solve_validity_unavailable"
    return "converged", None


def _transition_evidence(
    observation: AmrRefinementObservation | None,
    previous: AmrPassSnapshot,
    current: AmrPassSnapshot,
) -> AmrTransitionEvidence | None:
    """Detach compact native counts and source IDs while leaving full cell sets on disk."""
    if observation is None:
        return None
    marked, refined = observation.marked_parents, observation.refined_parents
    return AmrTransitionEvidence(
        cells_before=observation.cells_before,
        cells_after=observation.cells_after,
        marked_count=observation.declared_marked_count
        if marked is None
        else len(marked),
        refined_parent_count=observation.declared_refined_parent_count
        if refined is None
        else len(refined),
        extra_refined_parent_count=observation.declared_extra_refined_parent_count
        if marked is None or refined is None
        else len(refined - marked),
        marked_squared_indicator_sum=observation.marked_squared_indicator_sum,
        total_squared_indicator_sum=observation.total_squared_indicator_sum,
        true_dofs_before=previous.degrees_of_freedom,
        true_dofs_after=current.degrees_of_freedom,
        premesh_id=observation.premesh_id,
        source=observation.source,
        refined_mesh_id=observation.refined_mesh_id,
        next_mesh_id=observation.next_mesh_id,
        cells_after_refinement=observation.cells_after_refinement,
        cells_after_rebalance=observation.cells_after_rebalance,
        theta_requested=observation.theta_requested,
        threshold=observation.threshold,
        mesh_conformity=observation.mesh_conformity,
        max_nc_levels=observation.max_nc_levels,
        state=observation.state,
        counts_validation=observation.counts_validation,
    )
