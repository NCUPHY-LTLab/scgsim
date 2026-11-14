"""Detached resolved-run records shared by validators, EPR extraction and reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_convergence import ConvergenceCheck
from scgsim.palace.mesh.summary import MeshSummary, recorded_mesh_summary


@dataclass(frozen=True)
class ParsedTable:
    """Single parsed CSV artifact with typed row payload."""

    name: str
    path: Path
    headers: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PalacePerformance:
    """Canonical elapsed count/duration timing payload."""

    counts: dict[str, int | float]
    durations: dict[str, float]


@dataclass(frozen=True)
class PalaceCost:
    """Conventional solver resource/cost metadata."""

    problem_degrees_of_freedom: int | None
    mesh_elements: int | None
    mpi_size: int | None
    openmp_threads: int | None
    peak_memory_megabytes: dict[str, float | None]
    peak_node_memory_megabytes: dict[str, float | None]
    linear_solver: dict[str, float | int]
    git_tag: str | None


@dataclass(frozen=True)
class PalaceReturnedReceipt:
    """Normalized payload from returned receipt writer."""

    schema: str
    schema_version: int
    handoff_id: str
    status: str
    exit_code: int
    solver_exit_code: int
    tee_exit_code: int
    route: str
    problem: str
    identity_verified: bool
    timestamp_utc: str
    job_identity: dict[str, Any]
    input_hashes: tuple[dict[str, Any], ...]
    output_files: tuple[dict[str, Any], ...]
    log: dict[str, Any] | None
    solver_identity: dict[str, Any]
    execution_stage: str | None = None
    solver_invoked: bool | None = None
    preflight_exit_code: int | None = None


@dataclass(frozen=True)
class PalaceProvenance:
    """Resolved provenance payloads used to claim run identity."""

    handoff_metadata: dict[str, Any]
    run_metadata: dict[str, Any]
    resource_record: dict[str, Any]
    handoff_archive_manifest: dict[str, Any]
    index_map: dict[str, Any]
    mesh_manifest: dict[str, Any]
    config: dict[str, Any]
    palace_json: dict[str, Any]
    returned_receipt: dict[str, Any]


@dataclass(frozen=True)
class ResolvedRunData:
    """The checked records of one completed Palace run folder: identity, outcome, cost, timing and tables.

    ``convergence`` is the declared convergence checked on every pass of the run: the plan, the table of
    passes and the verdict, whose state is ``not_declared`` when the run folder has no plan.
    """

    run_dir: Path
    problem: str
    route: str
    has_returned_outputs: bool
    status: str
    performance: PalacePerformance
    cost: PalaceCost
    tables: dict[str, ParsedTable]
    returned_receipt: PalaceReturnedReceipt
    provenance: PalaceProvenance
    convergence: ConvergenceCheck

    def mesh_summary(self) -> MeshSummary:
        """Detach sealed mesh observations; estimates are not native solver DOF."""

        return recorded_mesh_summary(
            self.provenance.mesh_manifest,
            handoff=self.provenance.handoff_metadata,
            config=self.provenance.config,
            problem=self.problem,
        )
