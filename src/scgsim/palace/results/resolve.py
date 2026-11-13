"""Resolve a sealed manual Palace run into validated data and convenience views.

Validation retains the recorded artifact order; this module never runs a solver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scgsim.palace.reports.plotting import ReportTheme
from scgsim.palace.reports.reports import (
    show_all_results,
    show_physics_quantities,
    show_run_trustworthiness,
    show_simulation_benchmark,
)
from scgsim.palace.results.artifacts import (
    as_float_map,
    expect_int,
    expect_scalar,
    read_json,
    to_float,
    validate_contract_schema,
)
from scgsim.palace.results.epr import epr_result
from scgsim.palace.results.index_contract import (
    RouteConsistency,
    require_index_map_schema,
    require_manifest_schema,
    validate_config_index_correspondence,
    validate_config_problem,
)
from scgsim.palace.results.models import (
    PalaceCost,
    PalacePerformance,
    PalaceProvenance,
    PalaceReturnedReceipt,
    ParsedTable,
    ResolvedRunData,
)
from scgsim.palace.results.provenance_contract import (
    validate_archive_manifest,
    validate_archive_record,
    validate_execution_identity,
    validate_handoff_id,
    validate_hashes,
    validate_resource_identity,
    validate_scalar_identity,
)
from scgsim.palace.results.receipt_contract import (
    RETURNED_RECEIPT_FILE,
    ReceiptValidation,
    build_returned_receipt,
    validate_returned_state_agreement,
)
from scgsim.palace.results.tables import (
    PROBLEM_FAMILIES,
    count_index_entries,
    output_families,
    read_result_table,
)

_REQUIRED_FILES = (
    "metadata/palace_handoff_metadata.json",
    "metadata/palace_run_metadata.json",
    "metadata/palace_resource_record.json",
    "metadata/palace_handoff_archive_manifest.json",
    "metadata/palace_index_map.json",
    "metadata/mesh_manifest.json",
    "metadata/palace_returned_run_receipt.json",
    "config.json",
    "palace.msh",
    "results/palace/palace.json",
)


_SCHEMA_FILES = (
    "metadata/palace_index_map.json",
    "metadata/mesh_manifest.json",
)


def _require_file(path: Path) -> None:
    """Raise when a required artifact of the run folder is missing."""
    if not path.exists():
        raise FileNotFoundError(f"required artifact missing: {path}")


@dataclass(frozen=True)
class ResolvedPalaceResult(ResolvedRunData):
    """Complete resolved view of one Palace run directory."""

    def epr_result(self):
        """Detach the verified final Eigenmode EPR quantities for offline work."""

        return epr_result(self)

    def show_run_trustworthiness(
        self, *, theme: str = "light", show_details: bool = False
    ):
        """Build the identity and AMR view using the original artifact-validation order."""
        return show_run_trustworthiness(
            self,
            theme=cast(ReportTheme, theme),
            show_details=show_details,
        )

    def show_all_results(
        self,
        *,
        theme: str = "light",
        ranking_limit: int | None = 20,
        show_details: bool = False,
    ) -> None:
        """Display identity, benchmark and physics views in the established notebook sequence."""
        show_all_results(
            self,
            theme=cast(ReportTheme, theme),
            ranking_limit=ranking_limit,
            show_details=show_details,
        )

    def show_physics_quantities(
        self, *, theme: str = "light", ranking_limit: int | None = 20
    ):
        """Build the selected physics view from this sealed run without recomputing solver data."""
        return show_physics_quantities(
            self, theme=cast(ReportTheme, theme), ranking_limit=ranking_limit
        )

    def show_simulation_benchmark(
        self,
        *,
        show_details: bool = False,
    ):
        """Project recorded run cost and timing while retaining complete benchmark metadata."""
        return show_simulation_benchmark(self, show_details=show_details)


def resolve_palace_result(
    run_dir: str | Path, *, expected_handoff_id: str
) -> ResolvedPalaceResult:
    """Validate and resolve a completed Palace manual-handoff run directory."""

    if not isinstance(expected_handoff_id, str) or not expected_handoff_id:
        raise ValueError("expected_handoff_id must be a non-empty string.")

    root = Path(run_dir).expanduser().resolve()
    # The recorded schemas come first: a run folder of an older schema is rejected with its recorded
    # version, whatever solver output it lacks.
    for relative in _SCHEMA_FILES:
        _require_file(root / relative)
    index_map = read_json(root / "metadata/palace_index_map.json")
    mesh_manifest = read_json(root / "metadata/mesh_manifest.json")
    require_manifest_schema(mesh_manifest)
    require_index_map_schema(index_map)
    for relative in _REQUIRED_FILES:
        _require_file(root / relative)

    handoff_metadata = read_json(root / "metadata" / "palace_handoff_metadata.json")
    run_metadata = read_json(root / "metadata" / "palace_run_metadata.json")
    resource_record = read_json(root / "metadata" / "palace_resource_record.json")
    archive_manifest = read_json(
        root / "metadata" / "palace_handoff_archive_manifest.json"
    )
    config = read_json(root / "config.json")
    palace_payload = read_json(root / "results/palace/palace.json")
    receipt_payload = read_json(root / RETURNED_RECEIPT_FILE)

    route = expect_scalar(handoff_metadata, "route", str)
    problem = expect_scalar(handoff_metadata, "problem", str)

    if route not in {"A", "B"}:
        raise ValueError(f"invalid route {route!r}; expected 'A' or 'B'.")
    if problem not in PROBLEM_FAMILIES:
        raise ValueError(
            f"unsupported problem {problem!r}; expected one of {tuple(PROBLEM_FAMILIES)}."
        )

    _validate_metadata_headers(handoff_metadata, resource_record, run_metadata)

    validate_scalar_identity(handoff_metadata, run_metadata)
    validate_execution_identity(root, handoff_metadata)
    validate_handoff_id(handoff_metadata, expected_handoff_id)
    validate_resource_identity(resource_record, handoff_metadata)

    ReceiptValidation(
        receipt_payload=receipt_payload,
        expected_handoff_id=expected_handoff_id,
        handoff_metadata=handoff_metadata,
        run_metadata=run_metadata,
        problem=problem,
        route=route,
        root=root,
    ).validate()
    validate_returned_state_agreement(
        receipt_payload=receipt_payload,
        run_metadata=run_metadata,
        resource_record=resource_record,
    )
    validate_hashes(root, handoff_metadata, "handoff")
    validate_hashes(root, run_metadata, "run")
    validate_archive_record(root, resource_record)
    validate_archive_manifest(root, archive_manifest, resource_record, handoff_metadata)
    index_counts = count_index_entries(index_map)
    RouteConsistency(
        route,
        run_metadata,
        handoff_metadata,
        mesh_manifest,
        index_map,
        index_counts,
        problem,
    ).validate()
    validate_config_problem(config, problem)
    validate_config_index_correspondence(config, index_map, index_counts, problem)

    tables = _discover_results(root, problem, index_counts, config)

    returned_receipt = build_returned_receipt(receipt_payload)
    cost = _build_cost(palace_payload)
    performance = _build_performance(palace_payload)

    return ResolvedPalaceResult(
        run_dir=root,
        problem=problem,
        route=route,
        has_returned_outputs=True,
        status=expect_scalar(run_metadata, "status", str),
        performance=performance,
        cost=cost,
        tables=tables,
        returned_receipt=returned_receipt,
        provenance=PalaceProvenance(
            handoff_metadata=handoff_metadata,
            run_metadata=run_metadata,
            resource_record=resource_record,
            handoff_archive_manifest=archive_manifest,
            index_map=index_map,
            mesh_manifest=mesh_manifest,
            config=config,
            palace_json=palace_payload,
            returned_receipt=receipt_payload,
        ),
    )


def _discover_results(
    root: Path,
    problem: str,
    index_counts: dict[str, int],
    config: dict[str, Any],
) -> dict[str, ParsedTable]:
    """Read every required final native table and validate its sealed config/index correspondence."""
    results_root = root / "results" / "palace"
    discovered: dict[str, ParsedTable] = {}
    for family in output_families(problem, index_counts):
        path = results_root / f"{family}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"required result family is missing: {path}")
        discovered[family] = read_result_table(path, problem, index_counts, config)
    return discovered


def _build_performance(palace_payload: dict[str, Any]) -> PalacePerformance:
    """Detach native elapsed counts and durations without filling absent measurements."""
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


def _build_cost(palace_payload: dict[str, Any]) -> PalaceCost:
    """Detach solver, mesh, memory and parallel-resource observations from Palace JSON."""
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


__all__ = [
    "PalaceCost",
    "PalacePerformance",
    "PalaceProvenance",
    "PalaceReturnedReceipt",
    "ParsedTable",
    "ResolvedPalaceResult",
    "resolve_palace_result",
]


def _validate_metadata_headers(
    handoff_metadata: Any, resource_record: Any, run_metadata: Any
) -> None:
    """Validate manual-handoff kind, schema and nonempty state before identity checks."""
    if run_metadata.get("kind") != "manual_handoff":
        raise ValueError("run metadata does not identify manual_handoff kind.")
    if handoff_metadata.get("kind") != "manual_handoff":
        raise ValueError("handoff metadata does not identify manual_handoff kind.")
    validate_contract_schema(handoff_metadata, "handoff metadata")
    validate_contract_schema(run_metadata, "run metadata")
    validate_contract_schema(resource_record, "resource record")

    if expect_scalar(run_metadata, "status", str, fallback="") == "":
        raise ValueError("run metadata status must be a non-empty string.")
    if expect_scalar(resource_record, "status", str, fallback="") == "":
        raise ValueError("resource record status must be a non-empty string.")
