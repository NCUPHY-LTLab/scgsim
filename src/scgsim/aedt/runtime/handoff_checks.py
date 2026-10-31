"""Decide whether a prepared AEDT handoff may run, before AEDT or the receipt is touched.

A handoff is one-shot: a directory written by preparation, run once, and kept as the
record of that run. Everything here reads only. A failure raises and leaves the
receipt at ``not_run``, so a rejected handoff is not mistaken for a failed solve.

``load_verified_handoff`` is the entry point. It does not cover the prepared hashes,
which ``verify_prepared_hashes`` checks later: a mismatch there must be recorded in
the receipt as a failed run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.aedt._io import file_sha256, read_json
from scgsim.aedt.preparation.cohort import (
    canonical_handoff_paths,
    canonical_member_paths,
    validate_geometry_source,
    validate_hfss_import_source,
)
from scgsim.aedt.results.provenance import RECEIPT_V3, validate_runtime_source
from scgsim.aedt.runtime._text import unstripped_nonempty_text
from scgsim.aedt.specs.common import AedtResources
from scgsim.aedt.specs.hfss import HfssDrivenSpec, HfssEigenmodeSpec
from scgsim.aedt.specs.hfss_prepared import (
    HfssDrivenGeometrySpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
)
from scgsim.aedt.specs.parse import AedtSpec, parse_aedt_spec
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec

METADATA_RELATIVE_PATH = "metadata/aedt_handoff_metadata.json"
RECEIPT_RELATIVE_PATH = "metadata/aedt_run_receipt.json"
SPEC_RELATIVE_PATH = "aedt_spec.json"
GDS_RELATIVE_PATH = "geometry/design.gds"

# The only specs that import a copied GDS; the others carry their geometry in the spec.
_IMPORTED_GDS_SPECS = (HfssDrivenSpec, HfssEigenmodeSpec)
# Geometry-first HFSS specs: their source is bound by planar hashes, not a GDS hash.
_PLANAR_SOURCE_SPECS = (HfssEprAnalysisSpec, HfssEprSpec, HfssDrivenGeometrySpec)

_HANDOFF_V1 = "scgsim.aedt.handoff.v1"
_HANDOFF_V2 = "scgsim.aedt.handoff.v2"
_V2_WORKFLOWS = {
    "body_first_driven_terminal",
    "body_first_eigenmode",
    "epr",
    "epr_analysis",
}
_INITIAL_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "mode",
    "requested",
    "pdk_materials",
    "vacuum_material_id",
    "source",
    "outputs",
    "prepared_at_utc",
    "prepared_runtime_source",
}


@dataclass(frozen=True)
class VerifiedHandoff:
    """One handoff that passed every check possible before the run starts."""

    run_dir: Path
    metadata: dict[str, Any]
    receipt: dict[str, Any]
    receipt_path: Path
    spec: AedtSpec
    spec_path: Path
    resources: AedtResources | None
    # Identity of the prepared files, copied into the running receipt.
    cohort: dict[str, Any]


def load_verified_handoff(
    metadata_path: Path,
    *,
    prepare_only: bool,
    analyze_epr: bool,
    cores: int | None,
    ram_limit_percent: int | None,
) -> VerifiedHandoff:
    """Read a handoff and reject it unless it can run with the requested flags."""
    run_dir = metadata_path.parent.parent
    metadata = json_object(read_json(metadata_path), "handoff metadata")
    files = _canonical_metadata_files(metadata, metadata_path, run_dir)
    receipt_path = run_dir / files["receipt"]
    receipt = json_object(read_json(receipt_path), "receipt")
    if receipt.get("schema_version") != RECEIPT_V3:
        raise RuntimeError("handoff receipt schema is invalid")
    if receipt.get("status") != "not_run":
        raise RuntimeError("one-shot handoff is not in not_run state")
    spec_path = run_dir / files["spec"]
    spec = parse_aedt_spec(json_object(read_json(spec_path), "spec"), base_dir=run_dir)
    if not prepare_only and not analyze_epr and spec.run_control is None:
        raise ValueError("solving requires an explicit complete run_control")
    resources = _execution_resources(metadata, spec, cores, ram_limit_percent)
    if (
        isinstance(spec, _IMPORTED_GDS_SPECS)
        and spec.gds_path.resolve() != (run_dir / GDS_RELATIVE_PATH).resolve()
    ):
        raise RuntimeError("prepared spec must use geometry/design.gds")
    _require_matching_flags(spec, prepare_only=prepare_only, analyze_epr=analyze_epr)
    _require_pristine_run(run_dir, spec)
    cohort = _verify_prepared_cohort(run_dir, metadata, receipt, spec)
    return VerifiedHandoff(
        run_dir=run_dir,
        metadata=metadata,
        receipt=receipt,
        receipt_path=receipt_path,
        spec=spec,
        spec_path=spec_path,
        resources=resources,
        cohort=cohort,
    )


def _require_matching_flags(
    spec: AedtSpec, *, prepare_only: bool, analyze_epr: bool
) -> None:
    """Saved-field analysis needs --analyze-epr and nothing else uses it."""
    if isinstance(spec, HfssEprAnalysisSpec):
        valid = analyze_epr and not prepare_only
    else:
        valid = not analyze_epr
    if not valid:
        raise RuntimeError(
            "EPR execution flags do not match the prepared handoff workflow"
        )


def _require_pristine_run(run_dir: Path, spec: AedtSpec) -> None:
    if isinstance(spec, HfssEprAnalysisSpec):
        if (run_dir / "analysis-work").exists():
            raise RuntimeError("saved-field analysis workcopy already exists")
        return
    if (run_dir / f"{spec.project_name}.aedt").exists() or (
        run_dir / "results"
    ).exists():
        raise RuntimeError("one-shot handoff already owns project or results artifacts")


def _execution_resources(
    metadata: dict[str, Any],
    spec: AedtSpec,
    cores: int | None,
    ram_limit_percent: int | None,
) -> AedtResources | None:
    """Combine prepared resources with CLI overrides; an override may set one field."""
    execution = json_object(metadata.get("execution", {}), "handoff execution")
    prepared = execution.get("resources")
    if prepared is not None:
        prepared = json_object(prepared, "prepared resources")
        if set(prepared) != {"cores", "ram_limit_percent"}:
            raise ValueError("prepared resource fields are not canonical")
    if cores is not None or ram_limit_percent is not None:
        if isinstance(spec, HfssEprAnalysisSpec):
            raise ValueError("saved-field analysis does not run a solve")
        if prepared is None and (cores is None or ram_limit_percent is None):
            raise ValueError(
                "both resource overrides are required without prepared resources"
            )
        prepared = {
            "cores": cores if cores is not None else prepared["cores"],
            "ram_limit_percent": (
                ram_limit_percent
                if ram_limit_percent is not None
                else prepared["ram_limit_percent"]
            ),
        }
    if isinstance(spec, HfssEprAnalysisSpec) and prepared is not None:
        raise ValueError("saved-field analysis cannot request solve resources")
    return AedtResources(**prepared) if prepared is not None else None


def _canonical_metadata_files(
    metadata: dict[str, Any], metadata_path: Path, run_dir: Path
) -> dict[str, str]:
    """Return the file map, which must hold exactly what this workflow prepares."""
    if metadata_path != run_dir / METADATA_RELATIVE_PATH:
        raise RuntimeError("handoff metadata path is not canonical")
    schema = metadata.get("schema_version")
    if schema not in {_HANDOFF_V1, _HANDOFF_V2} or metadata.get("status") != "prepared":
        raise RuntimeError("handoff metadata schema or status is invalid")
    files = json_object(metadata.get("files"), "files")
    expected = {"spec": SPEC_RELATIVE_PATH, "receipt": RECEIPT_RELATIVE_PATH}
    workflow = metadata.get("workflow")
    if schema == _HANDOFF_V1 and metadata.get("mode") not in {"q2d", "q3d"}:
        expected["gds"] = GDS_RELATIVE_PATH
    elif schema == _HANDOFF_V2 and workflow == "epr_analysis":
        for key in ("saved_project", "saved_results"):
            value = files.get(key)
            if (
                not isinstance(value, str)
                or not value.startswith("saved/")
                or contained_path(run_dir, value) != run_dir / value
            ):
                raise RuntimeError("analysis handoff saved paths are not canonical")
            expected[key] = value
    if (schema == _HANDOFF_V2) != (workflow in _V2_WORKFLOWS):
        raise RuntimeError("handoff metadata workflow is inconsistent")
    if files != expected:
        raise RuntimeError("handoff metadata file map is not canonical")
    return expected


def _verify_prepared_cohort(
    run_dir: Path, metadata: dict[str, Any], receipt: dict[str, Any], spec: AedtSpec
) -> dict[str, Any]:
    """Prove the files on disk are the ones preparation wrote, and nothing else."""
    manifest_path = run_dir / "metadata/aedt_handoff_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"handoff manifest is missing: {manifest_path}")
    manifest = json_object(read_json(manifest_path), "handoff manifest")
    _require_current_prepared_receipt(metadata, manifest, receipt, spec)
    hashes = _verified_member_hashes(run_dir, manifest, spec)
    if isinstance(spec, Q3dSpec):
        validate_geometry_source(run_dir, receipt["source"], spec)
    validate_hfss_import_source(run_dir, read_json(run_dir / SPEC_RELATIVE_PATH), spec)
    return {
        "preparation_cohort": "prepared_v3",
        "prepared_runtime_source": receipt["prepared_runtime_source"],
        "prepared_receipt_sha256": hashes[RECEIPT_RELATIVE_PATH],
        "verified_prepared_manifest_sha256": file_sha256(manifest_path),
    }


def _require_current_prepared_receipt(
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    receipt: dict[str, Any],
    spec: AedtSpec,
) -> None:
    expected_manifest_schema = (
        "scgsim.aedt.handoff-manifest.v2"
        if isinstance(spec, _PLANAR_SOURCE_SPECS)
        else "scgsim.aedt.handoff-manifest.v1"
    )
    if manifest.get("schema_version") != expected_manifest_schema:
        raise RuntimeError("handoff manifest schema is invalid")
    if receipt.get("mode") != spec.mode:
        raise RuntimeError("prepared receipt mode does not match the spec")
    if (
        metadata.get("expected_receipt_schema") != RECEIPT_V3
        or manifest.get("expected_receipt_schema") != RECEIPT_V3
    ):
        raise RuntimeError("prepared receipt cohort markers are inconsistent")
    validate_runtime_source(receipt.get("prepared_runtime_source"), stage="prepared")
    if set(receipt) != _INITIAL_RECEIPT_KEYS or receipt.get("outputs") != {}:
        raise RuntimeError("prepared receipt members are not canonical")


def _verified_member_hashes(
    run_dir: Path, manifest: dict[str, Any], spec: AedtSpec
) -> dict[str, str]:
    """Check each manifest member's path, size and hash; return hash by path."""
    expected_paths = canonical_handoff_paths(spec)
    if manifest.get("allowed_paths") != expected_paths:
        raise RuntimeError("handoff manifest allowed paths are not canonical")
    members = manifest.get("members")
    # The manifest cannot list its own hash, so it has one member fewer than paths.
    if not isinstance(members, list) or len(members) != len(expected_paths) - 1:
        raise RuntimeError("handoff manifest members are not canonical")
    if [
        item.get("path") if isinstance(item, dict) else None for item in members
    ] != canonical_member_paths(spec):
        raise RuntimeError("handoff manifest members are not canonical")
    hashes: dict[str, str] = {}
    for item in members:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise RuntimeError("handoff manifest member is invalid")
        relative = item["path"]
        path = contained_path(run_dir, relative)
        if (
            not isinstance(item["bytes"], int)
            or isinstance(item["bytes"], bool)
            or item["bytes"] < 0
            or not isinstance(item["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            or not path.is_file()
            or path.stat().st_size != item["bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise RuntimeError(f"handoff manifest member mismatch: {relative}")
        hashes[relative] = item["sha256"]
    return hashes


def verify_prepared_hashes(
    metadata: dict[str, Any], spec_path: Path, spec: AedtSpec
) -> None:
    """Compare prepared source hashes; which ones exist depends on the spec family."""
    receipt = json_object(
        read_json(spec_path.parent / RECEIPT_RELATIVE_PATH), "receipt"
    )
    source = json_object(receipt.get("source"), "receipt.source")
    if source.get("spec") != SPEC_RELATIVE_PATH:
        raise RuntimeError("receipt source paths are not canonical")
    if file_sha256(spec_path) != unstripped_nonempty_text(
        source.get("spec_sha256"), "source.spec_sha256"
    ):
        raise RuntimeError("prepared spec hash mismatch")
    if isinstance(spec, _PLANAR_SOURCE_SPECS):
        _verify_planar_source_hashes(metadata, source, spec)
    elif isinstance(spec, (Q2dSpec, Q3dSpec)):
        if set(source) != {"spec", "spec_sha256"} or "gds_sha256" in metadata:
            raise RuntimeError(
                "Q2D handoff must not contain a GDS source"
                if isinstance(spec, Q2dSpec)
                else "Q3D body handoff must not contain an imported GDS source"
            )
    else:
        _verify_copied_gds_hashes(metadata, source, spec.gds_path)


def _verify_planar_source_hashes(
    metadata: dict[str, Any], source: dict[str, Any], spec: AedtSpec
) -> None:
    expected_source_keys = {
        "spec",
        "spec_sha256",
        "planar_source_sha256",
        "planar_model_sha256",
    }
    if isinstance(spec, HfssEprAnalysisSpec):
        expected_source_keys.add("saved_solution_sha256")
    if set(source) != expected_source_keys:
        raise RuntimeError("EPR handoff source members are not canonical")
    if source.get("planar_source_sha256") != spec.geometry.source_sha256:
        raise RuntimeError("prepared planar source hash mismatch")
    if source.get("planar_model_sha256") != spec.geometry.model_sha256:
        raise RuntimeError("prepared planar model hash mismatch")
    if "gds_sha256" in metadata:
        raise RuntimeError("EPR handoff must not contain a GDS source")
    if (
        isinstance(spec, HfssEprAnalysisSpec)
        and source.get("saved_solution_sha256") != spec.saved_solution.content_sha256
    ):
        raise RuntimeError("prepared saved solution hash mismatch")


def _verify_copied_gds_hashes(
    metadata: dict[str, Any], source: dict[str, Any], gds_path: Path
) -> None:
    if source.get("gds") != GDS_RELATIVE_PATH:
        raise RuntimeError("receipt GDS path is not canonical")
    if file_sha256(gds_path) != unstripped_nonempty_text(
        metadata.get("gds_sha256"), "gds_sha256"
    ):
        raise RuntimeError("copied GDS hash mismatch")
    if file_sha256(gds_path) != unstripped_nonempty_text(
        source.get("gds_sha256"), "source.gds_sha256"
    ):
        raise RuntimeError("receipt GDS hash mismatch")


def contained_path(root: Path, relative: str) -> Path:
    """Resolve a handoff-relative path, refusing anything outside the run directory."""
    requested = Path(relative)
    if requested.is_absolute():
        raise RuntimeError(f"handoff path escapes run directory: {relative!r}")
    path = (root / requested).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError(f"handoff path escapes run directory: {relative!r}")
    return path


def json_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return value
