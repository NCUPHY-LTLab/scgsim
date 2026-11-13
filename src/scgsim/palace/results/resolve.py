"""Read a sealed manual Palace run folder into the checked run record ``ResolvedRunData``.

Validation keeps the recorded artifact order; this module never runs a solver."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_run_folder import (
    ARCHIVE_MANIFEST,
    CONFIG,
    HANDOFF_METADATA,
    INDEX_MAP,
    MESH,
    MESH_MANIFEST,
    NATIVE_METADATA,
    NATIVE_METADATA_NAME,
    RESOURCE_RECORD,
    RETURNED_RECEIPT,
    RUN_METADATA,
    output_table,
    palace_problem,
    read_json,
)
from scgsim.geometry.models.refusals import Declared
from scgsim.palace.results.artifacts import (
    RETURNED_RUN_FOLDER,
    as_float_map,
    expect_int,
    expect_scalar,
    to_float,
    validate_contract_schema,
)
from scgsim.palace.results.index_contract import (
    RouteConsistency,
    require_index_map_schema,
    require_manifest_schema,
    validate_config_index_correspondence,
    validate_config_problem,
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
    ReceiptValidation,
    build_returned_receipt,
    validate_returned_state_agreement,
)
from scgsim.palace.results.resolved_records import (
    PalaceCost,
    PalacePerformance,
    PalaceProvenance,
    ParsedTable,
    ResolvedRunData,
)
from scgsim.palace.results.tables import (
    count_index_entries,
    output_families,
    read_result_table,
)
from scgsim.refusal import found_text

_WHERE_RESOLVE_CALL = "the call to resolve_palace_result"

_REQUIRED_FILES = (
    HANDOFF_METADATA,
    RUN_METADATA,
    RESOURCE_RECORD,
    ARCHIVE_MANIFEST,
    INDEX_MAP,
    MESH_MANIFEST,
    RETURNED_RECEIPT,
    CONFIG,
    MESH,
    NATIVE_METADATA,
)


_SCHEMA_FILES = (INDEX_MAP, MESH_MANIFEST)


def _require_file(path: Path) -> None:
    """Raise when a required artifact of the run folder is missing."""
    if not path.exists():
        raise FileNotFoundError(
            Declared("required artifact", RETURNED_RUN_FOLDER).refusal(
                f"missing: {path}"
            )
        )


def read_resolved_run(
    run_dir: str | Path, *, expected_handoff_id: str
) -> ResolvedRunData:
    """Read a completed Palace run folder and check every record in it against the handoff it must match."""

    if not isinstance(expected_handoff_id, str) or not expected_handoff_id:
        raise ValueError(
            Declared("expected_handoff_id", _WHERE_RESOLVE_CALL).refusal(
                f"must be a non-empty string; got {expected_handoff_id!r}. Pass the handoff_id that prepare_handoff() returned"
            )
        )
    root = Path(run_dir).expanduser().resolve()
    try:
        return _checked_run(root, expected_handoff_id)
    except (FileNotFoundError, TypeError, ValueError) as error:
        # Each check names what disagrees; every one of them is acted on the same way.
        error.add_note(
            f"Run folder {root}, expected handoff {expected_handoff_id!r}: resolve_palace_result reads only "
            "the folder Palace returned for that handoff, complete and unchanged. Fetch the returned folder "
            "again, or prepare and run a new handoff."
        )
        raise


def _checked_run(root: Path, expected_handoff_id: str) -> ResolvedRunData:
    """Read the run folder's records, check them in the recorded order and read the result tables."""
    records = _read_records(root)
    handoff = records.handoff_metadata
    route = expect_scalar(handoff, "route", str)
    problem = expect_scalar(handoff, "problem", str)
    if route not in {"A", "B"}:
        raise ValueError(
            Declared(
                f"route {route!r} of {HANDOFF_METADATA}", RETURNED_RUN_FOLDER
            ).refusal("must be 'A' or 'B'")
        )
    palace_problem(
        problem
    )  # refuses a problem SCGSim does not write, naming those it does
    _check_identity(
        root,
        records,
        expected_handoff_id=expected_handoff_id,
        route=route,
        problem=problem,
    )
    index_counts = count_index_entries(records.index_map)
    _check_index_and_config(records, index_counts, route=route, problem=problem)
    tables = _discover_results(root, problem, index_counts, records.config)
    returned_receipt = build_returned_receipt(records.returned_receipt)
    cost = _build_cost(records.palace_json)
    performance = _build_performance(records.palace_json)
    return ResolvedRunData(
        run_dir=root,
        problem=problem,
        route=route,
        has_returned_outputs=True,
        status=expect_scalar(records.run_metadata, "status", str),
        performance=performance,
        cost=cost,
        tables=tables,
        returned_receipt=returned_receipt,
        provenance=records,
    )


def _read_records(root: Path) -> PalaceProvenance:
    """Read every record of the run folder, the two schema records first.

    A run folder of another schema is refused with its recorded version, whatever solver output it lacks.
    """
    for relative in _SCHEMA_FILES:
        _require_file(root / relative)
    index_map = read_json(root / INDEX_MAP)
    mesh_manifest = read_json(root / MESH_MANIFEST)
    require_manifest_schema(mesh_manifest)
    require_index_map_schema(index_map)
    for relative in _REQUIRED_FILES:
        _require_file(root / relative)
    return PalaceProvenance(
        handoff_metadata=read_json(root / HANDOFF_METADATA),
        run_metadata=read_json(root / RUN_METADATA),
        resource_record=read_json(root / RESOURCE_RECORD),
        handoff_archive_manifest=read_json(root / ARCHIVE_MANIFEST),
        index_map=index_map,
        mesh_manifest=mesh_manifest,
        config=read_json(root / CONFIG),
        palace_json=read_json(root / NATIVE_METADATA),
        returned_receipt=read_json(root / RETURNED_RECEIPT),
    )


def _check_identity(
    root: Path,
    records: PalaceProvenance,
    *,
    expected_handoff_id: str,
    route: str,
    problem: str,
) -> None:
    """Check that the metadata, the receipt, the hashes and the archive all name the expected handoff."""
    handoff, run, resource = (
        records.handoff_metadata,
        records.run_metadata,
        records.resource_record,
    )
    _validate_metadata_headers(handoff, resource, run)
    validate_scalar_identity(handoff, run)
    validate_execution_identity(root, handoff)
    validate_handoff_id(handoff, expected_handoff_id)
    validate_resource_identity(resource, handoff)
    ReceiptValidation(
        receipt_payload=records.returned_receipt,
        expected_handoff_id=expected_handoff_id,
        handoff_metadata=handoff,
        run_metadata=run,
        problem=problem,
        route=route,
        root=root,
    ).validate()
    validate_returned_state_agreement(
        receipt_payload=records.returned_receipt,
        run_metadata=run,
        resource_record=resource,
    )
    validate_hashes(root, handoff, "handoff")
    validate_hashes(root, run, "run")
    validate_archive_record(root, resource)
    validate_archive_manifest(root, records.handoff_archive_manifest, resource, handoff)


def _check_index_and_config(
    records: PalaceProvenance,
    index_counts: dict[str, int],
    *,
    route: str,
    problem: str,
) -> None:
    """Check that the route, the mesh manifest, the index map and the config describe one model."""
    RouteConsistency(
        route,
        records.run_metadata,
        records.handoff_metadata,
        records.mesh_manifest,
        records.index_map,
        index_counts,
        problem,
    ).validate()
    validate_config_problem(records.config, problem)
    validate_config_index_correspondence(
        records.config, records.index_map, index_counts, problem
    )


def _discover_results(
    root: Path,
    problem: str,
    index_counts: dict[str, int],
    config: dict[str, Any],
) -> dict[str, ParsedTable]:
    """Read every required final native table and validate its sealed config/index correspondence."""
    discovered: dict[str, ParsedTable] = {}
    for family in output_families(problem, index_counts):
        path = root / output_table(family)
        if not path.is_file():
            raise FileNotFoundError(
                Declared("required result family", RETURNED_RUN_FOLDER).refusal(
                    f"is missing: {path}"
                )
            )
        discovered[family] = read_result_table(path, problem, index_counts, config)
    return discovered


def _build_performance(palace_payload: dict[str, Any]) -> PalacePerformance:
    """Detach native elapsed counts and durations without filling absent measurements."""
    elapsed = palace_payload.get("ElapsedTime")
    if not isinstance(elapsed, dict):
        raise TypeError(
            Declared(f"{NATIVE_METADATA_NAME}", RETURNED_RUN_FOLDER).refusal(
                "missing ElapsedTime block"
            )
        )

    counts_raw = elapsed.get("Counts")
    durations_raw = elapsed.get("Durations")
    if not isinstance(counts_raw, dict):
        raise TypeError(
            Declared(
                f"{NATIVE_METADATA_NAME} ElapsedTime.Counts", RETURNED_RUN_FOLDER
            ).refusal("must be a mapping")
        )
    if not isinstance(durations_raw, dict):
        raise TypeError(
            Declared(
                f"{NATIVE_METADATA_NAME} ElapsedTime.Durations", RETURNED_RUN_FOLDER
            ).refusal("must be a mapping")
        )

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
            Declared(
                f"{NATIVE_METADATA_NAME} performance payload", RETURNED_RUN_FOLDER
            ).refusal("is missing numeric timing data")
        )
    if any(not math.isfinite(value) or value < 0 for value in counts.values()) or any(
        not math.isfinite(value) or value < 0 for value in durations.values()
    ):
        raise ValueError(
            Declared(
                f"{NATIVE_METADATA_NAME} timing payload", RETURNED_RUN_FOLDER
            ).refusal("contains invalid numeric data")
        )
    if (
        "Total" not in counts
        or "Total" not in durations
        or counts["Total"] < 1
        or durations["Total"] < 0
    ):
        raise ValueError(
            Declared(
                f"{NATIVE_METADATA_NAME} timing payload", RETURNED_RUN_FOLDER
            ).refusal("must include non-negative Total")
        )
    return PalacePerformance(counts=counts, durations=durations)


def _build_cost(palace_payload: dict[str, Any]) -> PalaceCost:
    """Detach solver, mesh, memory and parallel-resource observations from Palace JSON."""
    problem_block = palace_payload.get("Problem")
    if not isinstance(problem_block, dict):
        raise TypeError(
            Declared(f"{NATIVE_METADATA_NAME}", RETURNED_RUN_FOLDER).refusal(
                "missing Problem block"
            )
        )

    linear_solver = palace_payload.get("LinearSolver")
    if not isinstance(linear_solver, dict):
        linear_solver = {}

    cost = PalaceCost(
        problem_degrees_of_freedom=expect_int(
            problem_block.get("DegreesOfFreedom"),
            f"{NATIVE_METADATA_NAME} Problem.DegreesOfFreedom",
        ),
        mesh_elements=expect_int(
            problem_block.get("MeshElements"),
            f"{NATIVE_METADATA_NAME} Problem.MeshElements",
        ),
        mpi_size=expect_int(
            problem_block.get("MPISize"), f"{NATIVE_METADATA_NAME} Problem.MPISize"
        ),
        openmp_threads=expect_int(
            problem_block.get("OpenMPThreads"),
            f"{NATIVE_METADATA_NAME} Problem.OpenMPThreads",
        ),
        peak_memory_megabytes=as_float_map(
            palace_payload.get("PeakMemoryMegabytes"),
            ("Average", "Max", "Min", "Total"),
        ),
        peak_node_memory_megabytes=as_float_map(
            palace_payload.get("PeakNodeMemoryMegabytes"),
            ("Average", "Max", "Min", "Total"),
        ),
        linear_solver={
            key: to_float(value, f"{NATIVE_METADATA_NAME} LinearSolver.{key}")
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
    _check_cost(cost)
    return cost


def _check_cost(cost: PalaceCost) -> None:
    """Refuse a cost record that lacks a size, a positive size, the tag, or readable memory and solver figures."""
    sizes = {
        "Problem.DegreesOfFreedom": cost.problem_degrees_of_freedom,
        "Problem.MeshElements": cost.mesh_elements,
        "Problem.MPISize": cost.mpi_size,
        "Problem.OpenMPThreads": cost.openmp_threads,
    }
    missing = [
        name
        for name, value in {**sizes, "GitTag": cost.git_tag}.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            Declared(f"{NATIVE_METADATA_NAME}", RETURNED_RUN_FOLDER).refusal(
                f"is missing required solver cost metadata: {missing}; the run Palace returned is incomplete"
            )
        )
    not_positive = {
        name: value for name, value in sizes.items() if value is None or value <= 0
    }
    if not_positive:
        raise ValueError(
            Declared(f"{NATIVE_METADATA_NAME}", RETURNED_RUN_FOLDER).refusal(
                f"has invalid solver cost metadata: {not_positive} must be positive"
            )
        )
    unreadable = [
        *([] if cost.linear_solver else ["LinearSolver"]),
        *(
            f"PeakMemoryMegabytes.{key}"
            for key, value in cost.peak_memory_megabytes.items()
            if value is None or not math.isfinite(value) or value < 0
        ),
        *(
            f"PeakNodeMemoryMegabytes.{key}"
            for key, value in cost.peak_node_memory_megabytes.items()
            if value is None or not math.isfinite(value) or value < 0
        ),
        *(
            f"LinearSolver.{key}"
            for key, value in cost.linear_solver.items()
            if not math.isfinite(value) or value < 0
        ),
    ]
    if unreadable:
        raise ValueError(
            Declared(f"{NATIVE_METADATA_NAME}", RETURNED_RUN_FOLDER).refusal(
                f"is missing required resource cost metadata: {unreadable} are absent, negative or not finite"
            )
        )


__all__ = ["read_resolved_run"]


def _validate_metadata_headers(
    handoff_metadata: Any, resource_record: Any, run_metadata: Any
) -> None:
    """Validate manual-handoff kind, schema and nonempty state before identity checks."""
    if run_metadata.get("kind") != "manual_handoff":
        raise ValueError(
            Declared(
                "run metadata does not identify manual_handoff kind",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(run_metadata.get("kind"), "manual_handoff"))
        )
    if handoff_metadata.get("kind") != "manual_handoff":
        raise ValueError(
            Declared(
                "handoff metadata does not identify manual_handoff kind",
                RETURNED_RUN_FOLDER,
            ).refusal(found_text(handoff_metadata.get("kind"), "manual_handoff"))
        )
    validate_contract_schema(handoff_metadata, "handoff metadata")
    validate_contract_schema(run_metadata, "run metadata")
    validate_contract_schema(resource_record, "resource record")

    if expect_scalar(run_metadata, "status", str, fallback="") == "":
        raise ValueError(
            Declared(
                "run metadata status must be a non-empty string", RETURNED_RUN_FOLDER
            ).refusal(found_text(run_metadata.get("status")))
        )
    if expect_scalar(resource_record, "status", str, fallback="") == "":
        raise ValueError(
            Declared(
                "resource record status must be a non-empty string", RETURNED_RUN_FOLDER
            ).refusal(found_text(resource_record.get("status")))
        )
