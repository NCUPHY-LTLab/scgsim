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
from typing import Any

from ._archive_manifest import _validate_archive_manifest
from ._artifact_files import _read_json
from ._config_checks import (
    _validate_config_index_correspondence,
    _validate_config_problem,
    _validate_route_consistency,
)
from ._expect import _expect_scalar
from ._index_map import IndexCounts, _index_counts, _surface_mask_entries
from ._result_tables import (
    _PROBLEM_FAMILIES,
    _eigenmode_count,
    _output_families,
    _read_csv_table,
    _validate_surface_mask_table,
    _validate_table,
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
    _RETURNED_RECEIPT_FILE,
    _build_returned_receipt,
    _validate_receipt_payload,
    _validate_returned_state_agreement,
)
from ._run_identity import (
    _validate_archive_record,
    _validate_contract_schema,
    _validate_execution_identity,
    _validate_handoff_id,
    _validate_hashes,
    _validate_resource_identity,
    _validate_scalar_identity,
)
from ._solver_metrics import _build_cost, _build_performance

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

    returned_receipt = _build_returned_receipt(provenance.returned_receipt)
    cost = _build_cost(provenance.palace_json)
    performance = _build_performance(provenance.palace_json)

    return ResolvedPalaceResult(
        run_dir=root,
        problem=problem,
        route=route,
        has_returned_outputs=True,
        status=_expect_scalar(provenance.run_metadata, "status", str),
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
    """Parse every metadata artifact. Read order decides which bad file errors first."""
    return PalaceProvenance(
        handoff_metadata=_read_json(root / "metadata" / "palace_handoff_metadata.json"),
        run_metadata=_read_json(root / "metadata" / "palace_run_metadata.json"),
        resource_record=_read_json(root / "metadata" / "palace_resource_record.json"),
        handoff_archive_manifest=_read_json(
            root / "metadata" / "palace_handoff_archive_manifest.json"
        ),
        index_map=_read_json(root / "metadata/palace_index_map.json"),
        mesh_manifest=_read_json(root / "metadata/mesh_manifest.json"),
        config=_read_json(root / "config.json"),
        palace_json=_read_json(root / "results/palace/palace.json"),
        returned_receipt=_read_json(root / _RETURNED_RECEIPT_FILE),
    )


def _read_route_and_problem(handoff_metadata: dict[str, Any]) -> tuple[str, str]:
    route = _expect_scalar(handoff_metadata, "route", str)
    problem = _expect_scalar(handoff_metadata, "problem", str)

    if route not in {"A", "B"}:
        raise ValueError(f"invalid route {route!r}; expected 'A' or 'B'.")
    if problem not in _PROBLEM_FAMILIES:
        raise ValueError(
            f"unsupported problem {problem!r}; expected one of {tuple(_PROBLEM_FAMILIES)}."
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
    _validate_contract_schema(handoff_metadata, "handoff metadata")
    _validate_contract_schema(run_metadata, "run metadata")
    _validate_contract_schema(resource_record, "resource record")

    if _expect_scalar(run_metadata, "status", str, fallback="") == "":
        raise ValueError("run metadata status must be a non-empty string.")
    if _expect_scalar(resource_record, "status", str, fallback="") == "":
        raise ValueError("resource record status must be a non-empty string.")


def _check_handoff_identity(
    root: Path, provenance: PalaceProvenance, expected_handoff_id: str
) -> None:
    """Handoff, run and resource records are one run, and it is the expected one."""
    _validate_scalar_identity(provenance.handoff_metadata, provenance.run_metadata)
    _validate_execution_identity(root, provenance.handoff_metadata)
    _validate_handoff_id(provenance.handoff_metadata, expected_handoff_id)
    _validate_resource_identity(provenance.resource_record, provenance.handoff_metadata)


def _check_returned_receipt(
    root: Path,
    provenance: PalaceProvenance,
    expected_handoff_id: str,
    route: str,
    problem: str,
) -> None:
    _validate_receipt_payload(
        receipt_payload=provenance.returned_receipt,
        expected_handoff_id=expected_handoff_id,
        handoff_metadata=provenance.handoff_metadata,
        run_metadata=provenance.run_metadata,
        problem=problem,
        route=route,
        root=root,
    )
    _validate_returned_state_agreement(
        receipt_payload=provenance.returned_receipt,
        run_metadata=provenance.run_metadata,
        resource_record=provenance.resource_record,
    )


def _check_sealed_files(root: Path, provenance: PalaceProvenance) -> None:
    """Files hashed by the metadata, and the handoff archive, are what was sealed."""
    _validate_hashes(root, provenance.handoff_metadata, "handoff")
    _validate_hashes(root, provenance.run_metadata, "run")
    _validate_archive_record(root, provenance.resource_record)
    _validate_archive_manifest(
        root,
        provenance.handoff_archive_manifest,
        provenance.resource_record,
        provenance.handoff_metadata,
    )


def _check_config_against_index_map(
    provenance: PalaceProvenance, route: str, problem: str
) -> IndexCounts:
    """Return the index counts, which later steps use to size the result tables."""
    index_counts = _index_counts(provenance.index_map)
    _validate_route_consistency(
        route,
        provenance.run_metadata,
        provenance.handoff_metadata,
        provenance.mesh_manifest,
        provenance.index_map,
        index_counts,
        problem,
    )
    _validate_config_problem(provenance.config, problem)
    _validate_config_index_correspondence(
        provenance.config, provenance.index_map, index_counts, problem
    )
    return index_counts


def _load_result_family(path: Path) -> ParsedTable:
    if not path.is_file():
        raise FileNotFoundError(f"required result family is missing: {path}")
    return _read_csv_table(path)


def _discover_results(
    root: Path,
    problem: str,
    index_counts: IndexCounts,
    config: dict[str, Any],
) -> dict[str, ParsedTable]:
    results_root = root / "results" / "palace"
    discovered: dict[str, ParsedTable] = {}
    for family in _output_families(problem, index_counts):
        table = _load_result_family(results_root / f"{family}.csv")
        _validate_table(problem, table, index_counts, config)
        discovered[family] = table
    # Masked Surface EPR adds two more tables, only when the index map has masks.
    mask_entries = _surface_mask_entries(
        _read_json(root / "metadata" / "palace_index_map.json")
    )
    if mask_entries:
        expected_rows = (
            _eigenmode_count(config)
            if problem == "Eigenmode"
            else index_counts["terminal"]
        )
        for name in ("surface-mask-Q", "surface-mask-energy"):
            table = _load_result_family(results_root / f"{name}.csv")
            _validate_surface_mask_table(
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
