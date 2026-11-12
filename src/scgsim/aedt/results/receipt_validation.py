"""Strict offline validation for receipt paths, cohort history, hashes, and file bindings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.aedt.io import file_sha256, read_json
from scgsim.aedt.preparation.cohort import (
    canonical_handoff_paths,
    canonical_member_paths,
    validate_hfss_import_source,
)
from scgsim.aedt.results.provenance import (
    RECEIPT_V3,
    initial_receipt_payload,
    initial_receipt_sha256,
    validate_runtime_source,
)
from scgsim.aedt.specs.hfss import (
    HfssDrivenGeometrySpec,
    HfssEprSpec,
)


def contained(root: Path, relative: str) -> Path:
    """Resolve a handoff- or receipt-relative path without allowing root escape."""
    requested = Path(relative)
    if requested.is_absolute():
        raise RuntimeError(f"path escapes its root directory: {relative!r}")
    resolved = (root / requested).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise RuntimeError(f"path escapes its root directory: {relative!r}")
    return resolved


@dataclass(frozen=True)
class CompletionCohort:
    """Preparation metadata and manifest read from a completed receipt."""

    version: str
    mode: str
    metadata: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    epr: bool
    workflow: Any


def validate_completion_cohort(root: Path, receipt: dict[str, Any], spec: Any) -> None:
    """Verify the immutable preparation cohort and complete execution provenance."""
    cohort = _read_completion_cohort(root, receipt, spec)
    _validate_cohort_workflow(cohort, spec)
    _validate_cohort_file_map(cohort)
    prepared_receipt_sha = _validate_cohort_provenance(receipt, cohort)
    _validate_cohort_members(root, spec, cohort, prepared_receipt_sha)


def _read_completion_cohort(
    root: Path, receipt: dict[str, Any], spec: Any
) -> CompletionCohort:
    if receipt.get("schema_version") != RECEIPT_V3:
        raise RuntimeError("completed receipt cohort schema is invalid")
    version = RECEIPT_V3.rsplit(".", 1)[-1]
    if receipt.get("expected_receipt_schema") != RECEIPT_V3:
        raise RuntimeError(f"completed {version} receipt expectation marker is invalid")
    mode = receipt.get("mode")
    if mode not in {"terminal", "modal", "eigenmode", "q3d", "q2d"}:
        raise RuntimeError("AEDT receipt mode is invalid")
    metadata_path = contained(root, "metadata/aedt_handoff_metadata.json")
    manifest_path = contained(root, "metadata/aedt_handoff_manifest.json")
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"completed {version} receipt lacks preparation cohort files"
        )
    metadata = read_json(metadata_path)
    manifest = read_json(manifest_path)
    epr = isinstance(spec, (HfssEprSpec, HfssDrivenGeometrySpec))
    expected_manifest_schema = (
        "scgsim.aedt.handoff-manifest.v2" if epr else "scgsim.aedt.handoff-manifest.v1"
    )
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version")
        != ("scgsim.aedt.handoff.v2" if epr else "scgsim.aedt.handoff.v1")
        or metadata.get("status") != "prepared"
        or metadata.get("mode") != mode
        or not isinstance(manifest, dict)
        or manifest.get("schema_version") != expected_manifest_schema
    ):
        raise RuntimeError(f"completed {version} preparation cohort is invalid")
    return CompletionCohort(
        version,
        mode,
        metadata,
        manifest,
        manifest_path,
        epr,
        metadata.get("workflow"),
    )


def _validate_cohort_workflow(cohort: CompletionCohort, spec: Any) -> None:
    if (
        isinstance(spec, HfssDrivenGeometrySpec)
        and cohort.workflow != "body_first_driven_terminal"
    ):
        raise RuntimeError("body-first Driven Terminal preparation workflow is invalid")
    if isinstance(spec, HfssEprSpec) and cohort.workflow not in {
        "body_first_eigenmode",
        "epr",
    }:
        raise RuntimeError("body-first Eigenmode preparation workflow is invalid")


def _validate_cohort_file_map(cohort: CompletionCohort) -> None:
    expected_files = {
        "spec": "aedt_spec.json",
        "receipt": "metadata/aedt_run_receipt.json",
    }
    if cohort.mode not in {"q2d", "q3d"} and not cohort.epr:
        expected_files["gds"] = "geometry/design.gds"
    if cohort.metadata.get("files") != expected_files:
        raise RuntimeError(
            f"completed {cohort.version} preparation file map is not canonical"
        )


def _validate_cohort_provenance(
    receipt: dict[str, Any], cohort: CompletionCohort
) -> str:
    metadata = cohort.metadata
    manifest = cohort.manifest
    if receipt.get("preparation_cohort") != "prepared_v3":
        raise RuntimeError(f"completed {cohort.version} preparation cohort is invalid")
    if (
        metadata.get("expected_receipt_schema") != RECEIPT_V3
        or manifest.get("expected_receipt_schema") != RECEIPT_V3
    ):
        raise RuntimeError("completed v3 preparation markers are inconsistent")
    validate_runtime_source(receipt.get("prepared_runtime_source"), stage="prepared")
    validate_runtime_source(receipt.get("runtime_source"), stage="actual")
    prepared_receipt_sha = receipt.get("prepared_receipt_sha256")
    verified_manifest_sha = receipt.get("verified_prepared_manifest_sha256")
    if (
        not isinstance(prepared_receipt_sha, str)
        or not is_sha256_text(prepared_receipt_sha)
        or not isinstance(verified_manifest_sha, str)
        or not is_sha256_text(verified_manifest_sha)
        or file_sha256(cohort.manifest_path) != verified_manifest_sha
        or _initial_receipt_sha256(receipt) != prepared_receipt_sha
    ):
        raise RuntimeError(
            f"completed {cohort.version} preparation hash bindings are invalid"
        )
    return prepared_receipt_sha


def _validate_cohort_members(
    root: Path, spec: Any, cohort: CompletionCohort, prepared_receipt_sha: str
) -> None:
    expected_paths = canonical_handoff_paths(spec)
    if cohort.manifest.get("allowed_paths") != expected_paths:
        raise RuntimeError(
            f"completed {cohort.version} manifest allowed paths are not canonical"
        )
    members = cohort.manifest.get("members")
    if (
        not isinstance(members, list)
        or len(members) != len(expected_paths) - 1
        or [item.get("path") if isinstance(item, dict) else None for item in members]
        != canonical_member_paths(spec)
    ):
        raise RuntimeError(
            f"completed {cohort.version} manifest members are not canonical"
        )
    _verify_cohort_member_hashes(root, cohort, members, prepared_receipt_sha)
    validate_hfss_import_source(root, read_json(root / "aedt_spec.json"), spec)


def _verify_cohort_member_hashes(
    root: Path,
    cohort: CompletionCohort,
    members: list[Any],
    prepared_receipt_sha: str,
) -> None:
    for member in members:
        if (
            not isinstance(member, dict)
            or set(member) != {"path", "bytes", "sha256"}
            or not isinstance(member.get("bytes"), int)
            or isinstance(member.get("bytes"), bool)
            or member["bytes"] < 0
            or not isinstance(member.get("sha256"), str)
            or not is_sha256_text(member["sha256"])
        ):
            raise RuntimeError(f"completed {cohort.version} manifest member is invalid")
        relative = member["path"]
        if relative == "metadata/aedt_run_receipt.json":
            if member["sha256"] != prepared_receipt_sha:
                raise RuntimeError(
                    f"completed {cohort.version} initial receipt binding is invalid"
                )
            continue
        path = contained(root, relative)
        if (
            not path.is_file()
            or path.stat().st_size != member["bytes"]
            or file_sha256(path) != member["sha256"]
        ):
            raise RuntimeError(
                f"completed {cohort.version} manifest member mismatch: {relative}"
            )


def is_sha256_text(value: str) -> bool:
    """Return whether a value is a lowercase SHA-256 hexadecimal string."""
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _initial_receipt_sha256(receipt: dict[str, Any]) -> str:
    initial = initial_receipt_payload(
        schema_version=RECEIPT_V3,
        mode=receipt.get("mode"),
        requested=receipt.get("requested"),
        pdk_materials=receipt.get("pdk_materials"),
        vacuum_material_id=receipt.get("vacuum_material_id"),
        source=receipt.get("source"),
        prepared_runtime_source_value=receipt.get("prepared_runtime_source"),
        outputs={},
        prepared_at_utc=receipt.get("prepared_at_utc"),
    )
    return initial_receipt_sha256(initial)


def verified(
    root: Path, relative: Any, hashes: dict[str, Any], hash_key: str | None = None
) -> Path:
    """Verify that a receipt-bound file exists and matches its recorded hash."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError("receipt has an invalid relative output path")
    path = contained(root, relative)
    expected = hashes.get(hash_key or relative)
    if (
        not isinstance(expected, str)
        or not path.is_file()
        or file_sha256(path) != expected
    ):
        raise RuntimeError(f"receipt output hash mismatch: {relative}")
    return path
