"""Detached AMR, surface, selection and failure records used by Palace reports.

AMR means adaptive mesh refinement; surface EPR means energy participation ratio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scgsim.contracts.palace_convergence import SetAside


def pass_number(pass_index: int) -> int:
    """The pass a snapshot's ``pass_index`` is, as Palace counts its solves.

    Pass 1 is the solve on the input mesh, written to ``iteration1``; ``pass_index`` counts the refinements
    before the pass, so it is one less.
    """
    return pass_index + 1


@dataclass(frozen=True)
class SurfaceEprRecord:
    """One solver-native Surface-EPR row over a whole surface, bound to the sidecar rows it integrates.

    ``rows`` are the sidecar ``surface_epr_row`` ids; ``entities`` are the SGB Entities whose surfaces they
    are and ``nets`` their exact Nets.  ``lineage`` maps each face to the Entities that share it, as
    attribution only: it weights nothing.
    """

    index: int
    interface_type: str
    name: str
    rows: tuple[str, ...]
    face_kinds: tuple[str, ...]
    entities: tuple[str, ...]
    nets: tuple[str, ...]
    attributes: tuple[int, ...]
    lineage: dict[str, dict[str, float]]
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
class SurfaceMarginEprRecord:
    """One margin row: the union of the SGB bands from ``margin_um`` inwards, bound to its whole surface.

    The row is an ordinary Palace ``Dielectric`` row over the attributes of those bands, so it is one of
    the alternatives to its whole-surface row ``baseline_index`` and never adds to it.
    """

    index: int
    baseline_index: int
    margin_index: int
    margin_um: float
    bands: tuple[int, ...]
    interface_type: str
    name: str
    rows: tuple[str, ...]
    face_kinds: tuple[str, ...]
    entities: tuple[str, ...]
    nets: tuple[str, ...]
    attributes: tuple[int, ...]
    participation: float
    quality_factor: float
    loss_tangent: float | None


@dataclass(frozen=True)
class SurfaceMarginEprSeriesSnapshot:
    """Margin-row values for one mode or excitation at one source pass."""

    pass_index: int
    source: str
    series_index: int
    series_kind: Literal["mode", "excitation"]
    records: tuple[SurfaceMarginEprRecord, ...]


@dataclass(frozen=True)
class AmrPassSnapshot:
    """One solver snapshot from an AMR pass or the final output folder ``results/palace``.

    ``pass_index`` counts the refinements before the pass (0 for the solve on the input mesh);
    ``pass_number`` is the pass as Palace counts its solves, from 1, as the folder records it.
    """

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
    surface_margin_epr: tuple[SurfaceMarginEprSeriesSnapshot, ...] | None = None
    native_iteration: int | None = None

    @property
    def pass_number(self) -> int:
        """The pass as Palace counts its solves (``pass_number``)."""
        return pass_number(self.pass_index)


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
    native_iteration: int | None = None
    elapsed_pass_reason: str | None = None


@dataclass(frozen=True)
class PalaceResultSelection:
    """The readable snapshot selected independently from the run outcome.

    ``selected_path`` is always relative to the run folder.
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


FailureCategory = Literal[
    "config_compatibility",
    "out_of_memory",
    "signal_killed",
    "solver_error",
    "output_capture_error",
    "unknown",
]
"""What a failed run's diagnosis attributes the failure to."""


@dataclass(frozen=True)
class PalaceFailureDiagnosis:
    """Evidence-based diagnosis that never changes the returned run status."""

    category: FailureCategory
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
    """Readable AMR snapshots plus the independently observed final-table status.

    ``final_reason`` says why the top level of ``results/palace`` is no pass of its own, when it is not;
    ``set_aside`` are the folders that are not read as passes, each with why.
    """

    passes: tuple[AmrPassSnapshot, ...]
    final_snapshot_status: Literal["readable", "missing", "unreadable"]
    final_reason: str | None = None
    set_aside: tuple[SetAside, ...] = ()
