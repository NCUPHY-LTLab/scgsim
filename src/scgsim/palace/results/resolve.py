"""Resolve a completed manual Palace run directory into one verified result.

``resolve_palace_result`` is the only way into a returned Palace run. It reads the
metadata, proves that it all describes the run the caller expects, parses the result
tables, and returns a ``ResolvedPalaceResult``. It fails closed: any inconsistency
raises, and nothing is repaired or guessed.

The steps, in the order they run (the order decides which error a user sees first):

1. Read every metadata artifact (``_read_provenance``).
2. Check the metadata headers and that handoff, run and resource records agree
   (``_run_identity.py``).
3. Check the returned receipt (``_returned_receipt.py``).
4. Check the files the metadata hashes and the handoff archive
   (``_run_identity.py``, ``_archive_manifest.py``).
5. Check ``config.json`` against the index map (``_config_checks.py``,
   ``_index_map.py``).
6. Read and validate the result tables (``_result_tables.py``).
7. Read solver timing and cost (``_solver_metrics.py``).

Not here: running Palace and writing handoffs (``palace/execution``) and plotting
(``palace/presentation``). The value types live in ``_result_types.py`` and are
re-exported below.

Route A and Route B are the two Palace geometry routes. Route A uses zero-thickness
face-metal PEC sheets; Route B treats the closed shell of a finite metal volume as PEC
(docs/geometry-sgb.qmd).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ._archive_manifest import validate_archive_manifest
from ._artifact_files import read_json
from ._config_checks import (
    validate_config_index_correspondence,
    validate_config_problem,
    validate_route_consistency,
)
from ._expect import expect_scalar
from ._index_map import IndexCounts, count_index_entries, surface_mask_entries
from ._result_tables import (
    PROBLEM_FAMILIES,
    eigenmode_count,
    output_families,
    read_csv_table,
    validate_surface_mask_table,
    validate_table,
)
from ._result_types import (
    PalaceCost,
    PalacePerformance,
    PalaceProvenance,
    PalaceReturnedReceipt,
    ParsedTable,
    ResolvedPalaceResult,
)
from ._returned_receipt import (
    RETURNED_RECEIPT_FILE,
    build_returned_receipt,
    validate_receipt_payload,
    validate_returned_state_agreement,
)
from ._run_documents import (
    ArchiveManifest,
    HandoffMetadata,
    IndexMap,
    ResourceRecord,
    ReturnedReceiptPayload,
    RunMetadata,
)
from ._run_identity import (
    validate_archive_record,
    validate_contract_schema,
    validate_execution_identity,
    validate_handoff_id,
    validate_hashes,
    validate_resource_identity,
    validate_scalar_identity,
)
from ._solver_metrics import build_cost, build_performance

# Files that must exist before any of them is read.
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


def resolve_palace_result(
    run_dir: str | Path, *, expected_handoff_id: str
) -> ResolvedPalaceResult:
    """Validate and resolve a completed Palace manual-handoff run directory."""

    if not isinstance(expected_handoff_id, str) or not expected_handoff_id:
        raise ValueError("expected_handoff_id must be a non-empty string.")

    root = Path(run_dir).expanduser().resolve()
    _require_run_files(root)
    provenance = _read_provenance(root)
    route, problem = _read_route_and_problem(provenance.handoff_metadata)

    _check_metadata_headers(provenance)
    _check_handoff_identity(root, provenance, expected_handoff_id)
    _check_returned_receipt(root, provenance, expected_handoff_id, route, problem)
    _check_sealed_files(root, provenance)
    index_counts = _check_config_against_index_map(provenance, route, problem)

    tables = _discover_results(root, problem, index_counts, provenance.config)

    returned_receipt = build_returned_receipt(provenance.returned_receipt)
    cost = build_cost(provenance.palace_json)
    performance = build_performance(provenance.palace_json)

    return ResolvedPalaceResult(
        run_dir=root,
        problem=problem,
        route=route,
        has_returned_outputs=True,
        status=expect_scalar(provenance.run_metadata, "status", str),
        performance=performance,
        cost=cost,
        tables=tables,
        returned_receipt=returned_receipt,
        provenance=provenance,
    )


# ------------------------------------------------------ the steps, in call order


def _require_run_files(root: Path) -> None:
    for relative in _REQUIRED_FILES:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"required artifact missing: {path}")


def _read_provenance(root: Path) -> PalaceProvenance:
    """Parse every metadata artifact. Read order decides which bad file errors first.

    The typed dicts name the fields the documents should have. Their shape is only
    proven by the checks that follow, not by this read.
    """
    return PalaceProvenance(
        handoff_metadata=cast(
            HandoffMetadata,
            read_json(root / "metadata" / "palace_handoff_metadata.json"),
        ),
        run_metadata=cast(
            RunMetadata, read_json(root / "metadata" / "palace_run_metadata.json")
        ),
        resource_record=cast(
            ResourceRecord,
            read_json(root / "metadata" / "palace_resource_record.json"),
        ),
        handoff_archive_manifest=cast(
            ArchiveManifest,
            read_json(root / "metadata" / "palace_handoff_archive_manifest.json"),
        ),
        index_map=cast(IndexMap, read_json(root / "metadata/palace_index_map.json")),
        mesh_manifest=read_json(root / "metadata/mesh_manifest.json"),
        config=read_json(root / "config.json"),
        palace_json=read_json(root / "results/palace/palace.json"),
        returned_receipt=cast(
            ReturnedReceiptPayload, read_json(root / RETURNED_RECEIPT_FILE)
        ),
    )


def _read_route_and_problem(handoff_metadata: HandoffMetadata) -> tuple[str, str]:
    route = expect_scalar(handoff_metadata, "route", str)
    problem = expect_scalar(handoff_metadata, "problem", str)

    if route not in {"A", "B"}:
        raise ValueError(f"invalid route {route!r}; expected 'A' or 'B'.")
    if problem not in PROBLEM_FAMILIES:
        raise ValueError(
            f"unsupported problem {problem!r}; expected one of {tuple(PROBLEM_FAMILIES)}."
        )
    return route, problem


def _check_metadata_headers(provenance: PalaceProvenance) -> None:
    run_metadata = provenance.run_metadata
    handoff_metadata = provenance.handoff_metadata
    resource_record = provenance.resource_record

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


def _check_handoff_identity(
    root: Path, provenance: PalaceProvenance, expected_handoff_id: str
) -> None:
    """Handoff, run and resource records are one run, and it is the expected one."""
    validate_scalar_identity(provenance.handoff_metadata, provenance.run_metadata)
    validate_execution_identity(root, provenance.handoff_metadata)
    validate_handoff_id(provenance.handoff_metadata, expected_handoff_id)
    validate_resource_identity(provenance.resource_record, provenance.handoff_metadata)


def _check_returned_receipt(
    root: Path,
    provenance: PalaceProvenance,
    expected_handoff_id: str,
    route: str,
    problem: str,
) -> None:
    validate_receipt_payload(
        receipt_payload=provenance.returned_receipt,
        expected_handoff_id=expected_handoff_id,
        handoff_metadata=provenance.handoff_metadata,
        run_metadata=provenance.run_metadata,
        problem=problem,
        route=route,
        root=root,
    )
    validate_returned_state_agreement(
        receipt_payload=provenance.returned_receipt,
        run_metadata=provenance.run_metadata,
        resource_record=provenance.resource_record,
    )


def _check_sealed_files(root: Path, provenance: PalaceProvenance) -> None:
    """Files hashed by the metadata, and the handoff archive, are what was sealed."""
    validate_hashes(root, provenance.handoff_metadata, "handoff")
    validate_hashes(root, provenance.run_metadata, "run")
    validate_archive_record(root, provenance.resource_record)
    validate_archive_manifest(
        root,
        provenance.handoff_archive_manifest,
        provenance.resource_record,
        provenance.handoff_metadata,
    )


def _check_config_against_index_map(
    provenance: PalaceProvenance, route: str, problem: str
) -> IndexCounts:
    """Return the index counts, which later steps use to size the result tables."""
    index_counts = count_index_entries(provenance.index_map)
    validate_route_consistency(
        route,
        provenance.run_metadata,
        provenance.handoff_metadata,
        provenance.mesh_manifest,
        provenance.index_map,
        index_counts,
        problem,
    )
    validate_config_problem(provenance.config, problem)
    validate_config_index_correspondence(
        provenance.config, provenance.index_map, index_counts, problem
    )
    return index_counts


def _load_result_family(path: Path) -> ParsedTable:
    if not path.is_file():
        raise FileNotFoundError(f"required result family is missing: {path}")
    return read_csv_table(path)


def _discover_results(
    root: Path,
    problem: str,
    index_counts: IndexCounts,
    config: dict[str, Any],
) -> dict[str, ParsedTable]:
    results_root = root / "results" / "palace"
    discovered: dict[str, ParsedTable] = {}
    for family in output_families(problem, index_counts):
        table = _load_result_family(results_root / f"{family}.csv")
        validate_table(problem, table, index_counts, config)
        discovered[family] = table
    # Masked Surface EPR adds two more tables, only when the index map has masks.
    mask_entries = surface_mask_entries(
        read_json(root / "metadata" / "palace_index_map.json")
    )
    if mask_entries:
        expected_rows = (
            eigenmode_count(config)
            if problem == "Eigenmode"
            else index_counts["terminal"]
        )
        for name in ("surface-mask-Q", "surface-mask-energy"):
            table = _load_result_family(results_root / f"{name}.csv")
            validate_surface_mask_table(
                table,
                problem=problem,
                expected_rows=expected_rows,
                entries=mask_entries,
            )
            discovered[name] = table
    return discovered


__all__ = [
    "PalaceCost",
    "PalacePerformance",
    "PalaceProvenance",
    "PalaceReturnedReceipt",
    "ParsedTable",
    "ResolvedPalaceResult",
    "resolve_palace_result",
]
