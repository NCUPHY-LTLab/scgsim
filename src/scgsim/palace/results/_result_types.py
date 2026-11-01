"""The value types a resolved Palace run is made of.

These are plain records. ``resolve.py`` re-exports them, and that is the import path
the rest of the package and users rely on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._run_documents import (
    ArchiveManifest,
    HandoffMetadata,
    IndexMap,
    JobIdentity,
    ResourceRecord,
    ReturnedReceiptPayload,
    RunMetadata,
)


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
    job_identity: JobIdentity
    input_hashes: tuple[dict[str, Any], ...]
    output_files: tuple[dict[str, Any], ...]
    log: dict[str, Any]
    solver_identity: dict[str, Any]
    execution_stage: str
    solver_invoked: bool
    preflight_exit_code: int


@dataclass(frozen=True)
class PalaceProvenance:
    """Resolved provenance payloads used to claim run identity."""

    handoff_metadata: HandoffMetadata
    run_metadata: RunMetadata
    resource_record: ResourceRecord
    handoff_archive_manifest: ArchiveManifest
    index_map: IndexMap
    mesh_manifest: dict[str, Any]
    config: dict[str, Any]  # Palace's own config.json
    palace_json: dict[str, Any]  # Palace's own palace.json
    returned_receipt: ReturnedReceiptPayload


@dataclass(frozen=True)
class ResolvedPalaceResult:
    """Complete resolved view of one Palace run directory."""

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

    def mesh_summary(self) -> dict[str, Any]:
        """Detach sealed mesh observations; estimates are not native solver DOF."""
        from ..mesh.summary import mesh_summary

        hashes = {
            entry["path"]: entry["sha256"]
            for entry in self.provenance.handoff_metadata["hashes"]
        }
        return mesh_summary(
            self.provenance.mesh_manifest,
            problem=self.problem,
            mesh_sha256=hashes["palace.msh"],
            manifest_sha256=hashes["metadata/mesh_manifest.json"],
            fem_order=self.provenance.config["Solver"]["Order"],
        )

    def epr_result(self):
        """Detach the verified final Eigenmode EPR quantities for offline work."""
        from .epr import epr_result

        return epr_result(self)

    def show_run_trustworthiness(
        self, *, theme: str = "light", show_details: bool = False
    ):
        from ..presentation.reports import render_run_trustworthiness

        return render_run_trustworthiness(
            self,
            theme=theme,
            show_details=show_details,
        )

    def show_all_results(
        self,
        *,
        theme: str = "light",
        ranking_limit: int | None = 20,
        show_details: bool = False,
    ) -> None:
        from ..presentation.reports import render_all_results

        render_all_results(
            self,
            theme=theme,
            ranking_limit=ranking_limit,
            show_details=show_details,
        )

    def show_physics_quantities(
        self, *, theme: str = "light", ranking_limit: int | None = 20
    ):
        from ..presentation.reports import render_physics_quantities

        return render_physics_quantities(self, theme=theme, ranking_limit=ranking_limit)

    def show_simulation_benchmark(self, *, show_details: bool = False):
        from ..presentation.reports import render_simulation_benchmark

        return render_simulation_benchmark(self, show_details=show_details)
