"""Validate the canonical prepared handoff before native execution begins."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scgsim.aedt.io import file_sha256, read_json
from scgsim.aedt.preparation.cohort import (
    canonical_handoff_paths,
    canonical_member_paths,
    validate_geometry_source,
    validate_hfss_import_source,
)
from scgsim.aedt.results.provenance import (
    RECEIPT_V1,
    RECEIPT_V2,
    RECEIPT_V3,
    validate_runtime_source,
)
from scgsim.aedt.results.receipt_validation import contained
from scgsim.aedt.specs.common import AedtResources
from scgsim.aedt.specs.hfss import (
    HfssDrivenGeometrySpec,
    HfssDrivenSpec,
    HfssEigenmodeSpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
)
from scgsim.aedt.specs.parse import AedtSpec
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec


@dataclass(frozen=True)
class _PreparedCohort:
    """The validated prepared receipt schema and its source identity."""

    name: str
    runtime_source: dict[str, Any]


def execution_resources(
    metadata: dict[str, Any],
    spec: AedtSpec,
    cores: int | None,
    ram_limit_percent: int | None,
) -> AedtResources | None:
    """Resolve prepared solve resources and explicit CLI overrides."""
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
            "cores": cores
            if cores is not None
            else cast(dict[str, Any], prepared)["cores"],
            "ram_limit_percent": (
                ram_limit_percent
                if ram_limit_percent is not None
                else cast(dict[str, Any], prepared)["ram_limit_percent"]
            ),
        }
    if isinstance(spec, HfssEprAnalysisSpec) and prepared is not None:
        raise ValueError("saved-field analysis cannot request solve resources")
    return AedtResources(**prepared) if prepared is not None else None


def canonical_metadata_files(
    metadata: dict[str, Any], metadata_path: Path, run_dir: Path
) -> dict[str, str]:
    """Require the canonical metadata location and workflow file map."""
    expected_path = run_dir / "metadata/aedt_handoff_metadata.json"
    if metadata_path != expected_path:
        raise RuntimeError("handoff metadata path is not canonical")
    schema = metadata.get("schema_version")
    if (
        schema not in {"scgsim.aedt.handoff.v1", "scgsim.aedt.handoff.v2"}
        or metadata.get("status") != "prepared"
    ):
        raise RuntimeError("handoff metadata schema or status is invalid")
    files = json_object(metadata.get("files"), "files")
    expected = {"spec": "aedt_spec.json", "receipt": "metadata/aedt_run_receipt.json"}
    workflow = metadata.get("workflow")
    if schema == "scgsim.aedt.handoff.v1" and metadata.get("mode") not in {
        "q2d",
        "q3d",
    }:
        expected["gds"] = "geometry/design.gds"
    elif schema == "scgsim.aedt.handoff.v2" and workflow == "epr_analysis":
        for key in ("saved_project", "saved_results"):
            value = files.get(key)
            if (
                not isinstance(value, str)
                or not value.startswith("saved/")
                or contained(run_dir, value) != run_dir / value
            ):
                raise RuntimeError("analysis handoff saved paths are not canonical")
            expected[key] = value
    if (schema == "scgsim.aedt.handoff.v2") != (
        workflow
        in {"body_first_driven_terminal", "body_first_eigenmode", "epr", "epr_analysis"}
    ):
        raise RuntimeError("handoff metadata workflow is inconsistent")
    if files != expected:
        raise RuntimeError("handoff metadata file map is not canonical")
    return expected


def verify_prepared_hashes(
    metadata: dict[str, Any], spec_path: Path, spec: AedtSpec
) -> None:
    """Verify spec and geometry bytes before opening a native application."""
    receipt = json_object(
        read_json(spec_path.parent / "metadata/aedt_run_receipt.json"), "receipt"
    )
    source = json_object(receipt.get("source"), "receipt.source")
    if source.get("spec") != "aedt_spec.json":
        raise RuntimeError("receipt source paths are not canonical")
    if file_sha256(spec_path) != _text(source.get("spec_sha256"), "source.spec_sha256"):
        raise RuntimeError("prepared spec hash mismatch")
    if isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec, HfssDrivenGeometrySpec)):
        _verify_planar_hashes(metadata, source, spec)
        return
    if isinstance(spec, (Q2dSpec, Q3dSpec)):
        if set(source) != {"spec", "spec_sha256"} or "gds_sha256" in metadata:
            raise RuntimeError(
                "Q2D handoff must not contain a GDS source"
                if isinstance(spec, Q2dSpec)
                else "Q3D body handoff must not contain an imported GDS source"
            )
        return
    gds_path = cast(Path, spec.gds_path)
    if source.get("gds") != "geometry/design.gds":
        raise RuntimeError("receipt GDS path is not canonical")
    if file_sha256(gds_path) != _text(metadata.get("gds_sha256"), "gds_sha256"):
        raise RuntimeError("copied GDS hash mismatch")
    if file_sha256(gds_path) != _text(source.get("gds_sha256"), "source.gds_sha256"):
        raise RuntimeError("receipt GDS hash mismatch")


def _verify_planar_hashes(
    metadata: dict[str, Any],
    source: dict[str, Any],
    spec: HfssEprAnalysisSpec | HfssEprSpec | HfssDrivenGeometrySpec,
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


def verify_prepared_cohort(
    run_dir: Path, metadata: dict[str, Any], receipt: dict[str, Any], spec: AedtSpec
) -> dict[str, Any]:
    """Verify the prepared receipt cohort and all manifest members."""
    manifest_path = run_dir / "metadata/aedt_handoff_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"handoff manifest is missing: {manifest_path}")
    manifest = json_object(read_json(manifest_path), "handoff manifest")
    expected_manifest_schema = (
        "scgsim.aedt.handoff-manifest.v2"
        if isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec, HfssDrivenGeometrySpec))
        else "scgsim.aedt.handoff-manifest.v1"
    )
    if manifest.get("schema_version") != expected_manifest_schema:
        raise RuntimeError("handoff manifest schema is invalid")
    cohort = _verify_receipt_cohort(metadata, manifest, receipt, spec)
    expected_paths = canonical_handoff_paths(spec)
    if manifest.get("allowed_paths") != expected_paths:
        raise RuntimeError("handoff manifest allowed paths are not canonical")
    members = manifest.get("members")
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
        path = contained(run_dir, relative)
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
    if isinstance(spec, Q3dSpec):
        validate_geometry_source(run_dir, receipt["source"], spec)
    validate_hfss_import_source(run_dir, read_json(run_dir / "aedt_spec.json"), spec)
    return {
        "preparation_cohort": cohort.name,
        "prepared_runtime_source": cohort.runtime_source,
        "prepared_receipt_sha256": hashes["metadata/aedt_run_receipt.json"],
        "verified_prepared_manifest_sha256": file_sha256(manifest_path),
    }


def _verify_receipt_cohort(
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    receipt: dict[str, Any],
    spec: AedtSpec,
) -> _PreparedCohort:
    metadata_has = "expected_receipt_schema" in metadata
    manifest_has = "expected_receipt_schema" in manifest
    markers = (
        metadata.get("expected_receipt_schema"),
        manifest.get("expected_receipt_schema"),
    )
    schema = receipt.get("schema_version")
    if receipt.get("mode") != spec.mode:
        raise RuntimeError("prepared receipt mode does not match the spec")
    initial_keys = {
        "schema_version",
        "status",
        "mode",
        "requested",
        "pdk_materials",
        "vacuum_material_id",
        "source",
        "outputs",
        "prepared_at_utc",
    }
    if schema in {RECEIPT_V2, RECEIPT_V3}:
        initial_keys.add("prepared_runtime_source")
        if markers != (schema, schema):
            raise RuntimeError("prepared receipt cohort markers are inconsistent")
        validate_runtime_source(
            receipt.get("prepared_runtime_source"), stage="prepared"
        )
        preparation_cohort = "prepared_v3" if schema == RECEIPT_V3 else "prepared_v2"
        prepared_source = receipt["prepared_runtime_source"]
    elif schema == RECEIPT_V1:
        if metadata_has or manifest_has:
            if RECEIPT_V2 in markers:
                raise RuntimeError(
                    "legacy receipt conflicts with v2 expectation markers"
                )
            raise RuntimeError(
                "legacy receipt conflicts with later expectation markers"
            )
        if "prepared_runtime_source" in receipt or "expected_receipt_schema" in receipt:
            raise RuntimeError("legacy receipt contains mixed preparation provenance")
        preparation_cohort = "verified_legacy_v1"
        prepared_source = {"status": "legacy_v1_not_recorded"}
    else:
        raise RuntimeError("handoff receipt schema is invalid")
    if set(receipt) != initial_keys or receipt.get("outputs") != {}:
        raise RuntimeError("prepared receipt members are not canonical")

    return _PreparedCohort(preparation_cohort, prepared_source)


def require_pristine_run(run_dir: Path, spec: AedtSpec) -> None:
    """Require a new one-shot project or saved-analysis workcopy."""
    if isinstance(spec, HfssEprAnalysisSpec):
        if (run_dir / "analysis-work").exists():
            raise RuntimeError("saved-field analysis workcopy already exists")
        return
    if (run_dir / f"{spec.project_name}.aedt").exists() or (
        run_dir / "results"
    ).exists():
        raise RuntimeError("one-shot handoff already owns project or results artifacts")


def json_object(value: Any, field: str) -> dict[str, Any]:
    """Require a JSON object while retaining its dictionary representation."""
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def validate_execution_flags(
    spec: AedtSpec, run_dir: Path, prepare_only: bool, analyze_epr: bool
) -> None:
    """Require the canonical source path and matching execution action."""
    if (
        not isinstance(
            spec,
            (
                HfssEprAnalysisSpec,
                HfssEprSpec,
                HfssDrivenGeometrySpec,
                Q2dSpec,
                Q3dSpec,
            ),
        )
        and cast(Path, spec.gds_path).resolve()
        != (run_dir / "geometry/design.gds").resolve()
    ):
        raise RuntimeError("prepared spec must use geometry/design.gds")
    valid_flags = (
        (
            isinstance(
                spec,
                (
                    HfssEprSpec,
                    HfssDrivenGeometrySpec,
                    HfssDrivenSpec,
                    HfssEigenmodeSpec,
                    Q3dSpec,
                    Q2dSpec,
                ),
            )
            and not analyze_epr
        )
        or (isinstance(spec, HfssEprAnalysisSpec) and analyze_epr and not prepare_only)
        or (
            not isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec))
            and not prepare_only
            and not analyze_epr
        )
    )
    if not valid_flags:
        raise RuntimeError(
            "EPR execution flags do not match the prepared handoff workflow"
        )


def warn_thin_film_surfaces(spec: AedtSpec) -> None:
    """Preserve the pinned-version thin-film surface interpretation warning."""
    if isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec)):
        geometry = spec.geometry
        request = spec.epr_request
        selected_surfaces = (
            None if request is None else request.surface_contribution_ids
        )
        if (
            geometry.modeling == "thin_film"
            and request is not None
            and any(
                contribution.interface_kind != "MM"
                and (
                    selected_surfaces is None
                    or contribution.contribution_id in selected_surfaces
                )
                for contribution in geometry.contributions
            )
        ):
            warnings.warn(
                "ThinFilm Surface-EPR uses FacePrimitive.normal as a geometric "
                "direction, not native calculator-side readback. The "
                "Surface/Adjacent mapping is empirical for pinned PyAEDT 1.3.0 / "
                "AEDT 2024.2; it is unverified for 2023.2 and may select the "
                "wrong material side; ThinFilm "
                "MA/MS participation and loss are not physically validated.",
                UserWarning,
                stacklevel=2,
            )
