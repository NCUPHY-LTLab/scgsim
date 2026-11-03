"""Detached adaptive-mesh observations and explicit metric availability.

Counts and solver values remain observations. These records define no numerical
acceptance threshold, synthetic reference, or inference of linear-solve validity.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

CapacitanceMatrix = tuple[tuple[float, ...], ...]
CAPACITANCE_UNIT_SCALES = {
    "F": 1.0,
    "mF": 1e-3,
    "uF": 1e-6,
    "µF": 1e-6,
    "μF": 1e-6,
    "nF": 1e-9,
    "pF": 1e-12,
    "fF": 1e-15,
    "aF": 1e-18,
}


@dataclass(frozen=True)
class CapacitanceReference:
    """An explicitly supplied square capacitance matrix with named terminals.

    ``terminal_ids`` are exact indexed conductor net IDs, not display names.
    ``values`` uses ``units`` on both axes ordered by ``terminal_ids``. ``source``
    identifies the supplied reference; no final AMR pass becomes a reference.
    """

    values: CapacitanceMatrix
    terminal_ids: tuple[str, ...]
    units: str
    source: str

    def __post_init__(self) -> None:
        values = tuple(tuple(value for value in row) for row in self.values)
        terminal_ids = tuple(self.terminal_ids)
        if not terminal_ids or any(
            not isinstance(name, str) or not name for name in terminal_ids
        ):
            raise ValueError("reference terminal_ids must be nonempty strings")
        if len(set(terminal_ids)) != len(terminal_ids):
            raise ValueError("reference terminal_ids must be unique")
        size = len(terminal_ids)
        if len(values) != size or any(len(row) != size for row in values):
            raise ValueError("reference values must be square and match terminal_ids")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for row in values
            for value in row
        ):
            raise TypeError("reference values must contain real capacitances")
        if self.units not in CAPACITANCE_UNIT_SCALES:
            raise ValueError("unsupported capacitance reference units")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("reference source must identify the supplied reference")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "terminal_ids", terminal_ids)


@dataclass(frozen=True)
class AmrMetric:
    """A scalar observation or an unavailable result with an explicit reason."""

    value: float | int | None
    availability: Literal["available", "unavailable"]
    reason: str | None = None
    units: str = "1"


@dataclass(frozen=True)
class AmrMatrixMetric:
    """An elementwise capacitance difference ordered by exact terminal identity."""

    value: CapacitanceMatrix | None
    terminal_ids: tuple[str, ...] | None
    availability: Literal["available", "unavailable"]
    reason: str | None = None
    units: str = "F"


@dataclass(frozen=True)
class CapacitanceComparison:
    """Keep absolute entry differences and normalized matrix change distinct."""

    relative_frobenius: AmrMetric
    absolute_delta_f: AmrMatrixMetric
    normalization: Literal["current", "reference"]
    source: str | None


@dataclass(frozen=True)
class AmrCellIdentity:
    """One owned coarse-cell identity in its pre-refinement mesh and rank scope."""

    premesh_id: str
    rank: int
    coarse_local_id: int


NativeResidual = float | Literal["NaN", "Infinity", "-Infinity"] | None


@dataclass(frozen=True)
class AmrLinearSolveStatistics:
    """Unmodified native Krylov residual semantics and reported convergence state."""

    available: bool
    algorithm: str | None = None
    residual_norm: str | None = None
    final_residual_is_estimate: bool | None = None
    iterations: int | None = None
    initial_residual: NativeResidual = None
    final_residual: NativeResidual = None
    reference_residual: NativeResidual = None
    reference_residual_source: str | None = None
    relative_tolerance: float | None = None
    absolute_tolerance: float | None = None
    maximum_iterations: int | None = None
    converged: bool | None = None


@dataclass(frozen=True)
class AmrTerminalSolveObservation:
    """One exact native terminal and matrix-column binding with separate solves."""

    terminal_index: int
    capacitance_column: int
    terminal_id: str | None
    main: AmrLinearSolveStatistics
    recovery: AmrLinearSolveStatistics
    source: str


@dataclass(frozen=True)
class AmrSolveObservation:
    """Native per-terminal linear-solve validity, independent of driver exit state."""

    native_iteration: int
    terminal_ids: tuple[str, ...]
    converged: tuple[bool | None, ...]
    source: str
    recovery_converged: tuple[bool | None, ...] = ()
    terminal_statistics: tuple[AmrTerminalSolveObservation, ...] = ()
    terminal_identity_reason: str | None = None
    mesh_id: str | None = None
    state: str | None = None
    owned_cells: int | None = None
    true_dofs: int | None = None
    capacitance_written: bool | None = None
    capacitance_iteration: int | None = None
    capacitance_terminal_order: tuple[int, ...] | None = None
    indicator_norm: float | None = None
    indicator_semantics: str | None = None


@dataclass(frozen=True)
class AmrRefinementObservation:
    """Exact marked/refined parent sets and owned-cell counts for one transition.

    ``marked_parents`` is M and ``refined_parents`` is P, both on the same
    ``premesh_id``. Squared-indicator sums are actual native sums over M and
    the complete pre-refinement owned-cell set, never count-based proxies.
    """

    before_iteration: int
    after_iteration: int
    premesh_id: str
    cells_before: int | None
    cells_after: int | None
    marked_parents: frozenset[AmrCellIdentity] | None
    refined_parents: frozenset[AmrCellIdentity] | None
    marked_squared_indicator_sum: float | None
    total_squared_indicator_sum: float | None
    source: str
    state: str | None = None
    refined_mesh_id: str | None = None
    next_mesh_id: str | None = None
    cells_after_refinement: int | None = None
    cells_after_rebalance: int | None = None
    theta_requested: float | None = None
    threshold: float | None = None
    mesh_conformity: str | None = None
    max_nc_levels: int | None = None
    declared_marked_count: int | None = None
    declared_refined_parent_count: int | None = None
    declared_extra_refined_parent_count: int | None = None
    counts_validation: str | None = None


@dataclass(frozen=True)
class AmrDiagnosticObservations:
    """Detached native facts after an adapter has verified its producer contract."""

    solves: tuple[AmrSolveObservation, ...] = ()
    refinements: tuple[AmrRefinementObservation, ...] = ()
    reason: str | None = "native_diagnostics_unavailable"
    enabled: bool = False
    source: str | None = None
    integrity: str = "unavailable"
    missing_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class AmrPassAnalysis:
    """One genuine native solve with retained raw capacitance and estimator norm."""

    native_iteration: int | None
    source: str
    capacitance_matrix_f: CapacitanceMatrix | None
    terminal_ids: tuple[str, ...] | None
    native_zz_norm: AmrMetric
    reference_capacitance: CapacitanceComparison | None
    linear_solve_state: Literal["converged", "unconverged", "unavailable"] = (
        "unavailable"
    )
    linear_solve_reason: str | None = "linear_solve_validity_unavailable"
    terminal_solve_observation: AmrSolveObservation | None = None
    recovery_solve_state: Literal["converged", "unconverged", "unavailable"] = (
        "unavailable"
    )
    recovery_solve_reason: str | None = "recovery_solve_validity_unavailable"
    capacitance_availability_reason: str | None = None
    degrees_of_freedom: int | None = None
    mesh_elements: int | None = None
    elapsed_cumulative_s: float | None = None
    peak_node_memory_mb: float | None = None


@dataclass(frozen=True)
class AmrTransitionEvidence:
    """Compact native counts and identities; exact large cell sets stay in source files."""

    cells_before: int | None
    cells_after: int | None
    marked_count: int | None
    refined_parent_count: int | None
    extra_refined_parent_count: int | None
    marked_squared_indicator_sum: float | None
    total_squared_indicator_sum: float | None
    true_dofs_before: int | None
    true_dofs_after: int | None
    premesh_id: str
    source: str
    refined_mesh_id: str | None = None
    next_mesh_id: str | None = None
    cells_after_refinement: int | None = None
    cells_after_rebalance: int | None = None
    theta_requested: float | None = None
    threshold: float | None = None
    mesh_conformity: str | None = None
    max_nc_levels: int | None = None
    state: str | None = None
    counts_validation: str | None = None


@dataclass(frozen=True)
class AmrTransitionAnalysis:
    """Metrics for an observed consecutive native refinement, or explicit gaps.

    New-cell amplification is (N_after - N_before)/|M|. Closure amplification is
    |P|/|M|, while extra closure parents are the exact |P\\M| set difference.
    """

    before_iteration: int | None
    after_iteration: int | None
    availability: Literal["available", "unavailable"]
    reason: str | None
    marked_fraction: AmrMetric
    marked_squared_indicator_coverage: AmrMetric
    closure_amplification: AmrMetric
    closure_extra_parents: AmrMetric
    new_cell_amplification: AmrMetric
    dof_growth_ratio: AmrMetric
    dof_increase: AmrMetric
    adjacent_capacitance: CapacitanceComparison
    observations: AmrTransitionEvidence | None = None


@dataclass(frozen=True)
class PalaceAmrAnalysis:
    """Detached adaptive-refinement analysis without solver or display state."""

    run_dir: Path
    problem: str
    run_status: str
    integrity: str
    passes: tuple[AmrPassAnalysis, ...]
    transitions: tuple[AmrTransitionAnalysis, ...]
    reference: CapacitanceReference | None = None
    diagnostics_reason: str | None = None
    diagnostic_missing_files: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        """Detach standard-library records into serializable report data."""
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        return _json_observations(payload)


def _json_observations(value: Any) -> Any:
    """Encode IEEE nonfinite observations as explicit native-style strings for JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_observations(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_observations(item) for item in value)
    if isinstance(value, list):
        return [_json_observations(item) for item in value]
    return value
