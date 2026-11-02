"""Detached AMR, surface, selection and failure records used by Palace reports.

AMR means adaptive mesh refinement; surface EPR means energy participation ratio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class SurfaceEprRecord:
    """One solver-native surface result bound to structured SGB provenance."""

    index: int
    interface_type: str
    surface_id: str
    face_kind: str
    owner_semantic_ids: tuple[str, ...]
    net_id: str | None
    equipotential_id: str | None
    source_provenance: dict[str, Any]
    participation: float
    quality_factor: float
    loss_tangent: float | None


@dataclass(frozen=True)
class SurfaceEprSeriesSnapshot:
    """Surface-EPR values for one mode or excitation at one solver snapshot."""

    pass_index: int
    source: str
    series_index: int
    series_kind: Literal["mode", "excitation"]
    records: tuple[SurfaceEprRecord, ...]
    quality_factor_total: float | None
    t1_seconds: float | None
    loss_status: Literal["available", "unavailable_missing", "unavailable_nonfinite"]


@dataclass(frozen=True)
class SurfaceMaskEprRecord:
    """One native Inset-mask result, explicitly bound to its unmasked surface."""

    index: int
    baseline_index: int
    margin_index: int
    margin_um: float
    native_margin: float
    model_l0_m: float
    interface_type: str
    surface_id: str
    face_kind: str
    owner_semantic_ids: tuple[str, ...]
    net_id: str | None
    equipotential_id: str | None
    source_provenance: dict[str, Any]
    participation: float
    quality_factor: float
    energy_j: float
    loss_tangent: float | None
    contribution_status: Literal["available", "zero_contribution"]
    retained_area_status: Literal["unavailable"] = "unavailable"


@dataclass(frozen=True)
class SurfaceMaskEprSeriesSnapshot:
    """Native Inset-mask values for one mode or excitation at one source pass."""

    pass_index: int
    source: str
    series_index: int
    series_kind: Literal["mode", "excitation"]
    records: tuple[SurfaceMaskEprRecord, ...]


@dataclass(frozen=True)
class AmrPassSnapshot:
    """One solver snapshot from an AMR pass or the parent results folder."""

    pass_index: int
    source: str
    path: Path
    frequencies_ghz: tuple[float, ...] | None
    eig_columns: dict[str, tuple[float, ...]] | None
    port_epr: dict[str, tuple[float, ...]] | None
    capacitance_matrix_f: tuple[tuple[float, ...], ...] | None
    surface_epr: tuple[SurfaceEprSeriesSnapshot, ...] | None
    error_indicators: dict[str, float] | None
    error_norm: float | None
    degrees_of_freedom: int | None
    mesh_elements: int | None
    elapsed_total_s: float | None
    peak_node_memory_mb: float | None
    surface_mask_epr: tuple[SurfaceMaskEprSeriesSnapshot, ...] | None = None


@dataclass(frozen=True)
class PassCostRecord:
    """One AMR pass worth of solver cost, for this run and later accumulation."""

    pass_index: int
    source: str
    degrees_of_freedom: int | None
    mesh_elements: int | None
    elapsed_cumulative_s: float | None
    elapsed_pass_s: float | None
    seconds_per_million_dof: float | None
    peak_node_memory_mb: float | None


@dataclass(frozen=True)
class PalaceResultSelection:
    """The readable snapshot selected independently from the run outcome.

    ``selected_path`` is always relative to the run directory.
    """

    final_snapshot_status: Literal["readable", "missing", "unreadable"]
    selected_source: str | None
    selected_path: Path | None
    selected_pass_index: int | None
    reason: Literal[
        "final_snapshot",
        "latest_complete_iteration_after_failed_attempt",
        "no_complete_snapshot",
    ]
    integrity: Literal["receipt_bound", "observed_unsealed", "unavailable"]


@dataclass(frozen=True)
class PalaceFailureDiagnosis:
    """Evidence-based diagnosis that never changes the returned run status."""

    category: Literal[
        "config_compatibility",
        "out_of_memory",
        "signal_killed",
        "solver_error",
        "output_capture_error",
        "unknown",
    ]
    exit_code: int | None
    execution_stage: Literal["config_compatibility", "solver"] | None
    solver_invoked: bool | None
    preflight_exit_code: int | None
    solver_exit_code: int | None
    tee_exit_code: int | None
    summary: str
    evidence: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CollectedSnapshots:
    """Readable AMR snapshots plus the independently observed final-table status."""

    passes: tuple[AmrPassSnapshot, ...]
    final_snapshot_status: Literal["readable", "missing", "unreadable"]
