"""Strict offline model-stage reading, independent of completed physics results.

The transaction owns construction, save and release. This reader verifies those
recorded facts and detached source/native assignments; it does not open AEDT or
claim native contact, meshing, fields or solved quantities from source intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.aedt.data import detached, freeze
from scgsim.aedt.io import file_sha256, read_json
from scgsim.aedt.preparation.cohort import validate_geometry_source
from scgsim.aedt.results.hfss_readback import (
    validate_hfss_pec_bindings,
    validate_hfss_setup,
    validate_lumped_readback,
    validate_modal_native_evidence,
)
from scgsim.aedt.results.provenance import RECEIPT_V3
from scgsim.aedt.results.q2d_readback import (
    validate_q2d_model_readback,
    validate_q2d_setup,
)
from scgsim.aedt.results.q3d_readback import (
    validate_q3d_model_readback,
    validate_q3d_setup,
)
from scgsim.aedt.results.readback_common import positive_unique_ids
from scgsim.aedt.results.receipt_validation import (
    contained,
    validate_completion_cohort,
    verified,
)
from scgsim.aedt.results.terminal_readback import (
    validate_body_terminal_readback,
    validate_terminal_native_evidence,
)
from scgsim.aedt.specs.hfss import (
    HfssDrivenGeometrySpec,
    HfssDrivenSpec,
    HfssEigenmodeSpec,
    HfssEprSpec,
)
from scgsim.aedt.specs.parse import parse_aedt_spec
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec


@dataclass(frozen=True)
class PreparedModel:
    """Verified saved model and immutable native readback, without physics data."""

    mode: str
    project_path: Path
    receipt_path: Path
    model_stage: str
    _readback: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "_readback", freeze(self._readback))

    @property
    def readback(self) -> dict[str, Any]:
        """Return a detached copy of verified native readback evidence."""
        return detached(self._readback)


@dataclass(frozen=True)
class PreparedModelContext:
    """Receipt and parsed source facts carried through prepared-model checks."""

    root: Path
    receipt_path: Path
    receipt: dict[str, Any]
    save: dict[str, Any]
    source_binding: dict[str, Any]
    spec: Any


def resolve_prepared_model(source: str | Path) -> PreparedModel:
    """Verify one model-stage run directory or its handoff metadata path."""
    path = Path(source).resolve()
    if path.is_file():
        if path.name != "aedt_handoff_metadata.json" or path.parent.name != "metadata":
            raise ValueError(
                "prepared model requires a run directory or handoff metadata"
            )
        root = path.parent.parent
    else:
        root = path
    receipt_path = contained(root, "metadata/aedt_run_receipt.json")
    receipt = read_json(receipt_path)
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != RECEIPT_V3
        or receipt.get("status") != "native_preparation_only"
        or receipt.get("solver_invoked") is not False
    ):
        raise RuntimeError("receipt is not a successful native prepared-model stage")
    if receipt.get("release") != {"ok": True}:
        raise RuntimeError("prepared model lacks successful owned Desktop release")
    save = receipt.get("save")
    if not isinstance(save, dict) or save.get("ok") is not True:
        raise RuntimeError("prepared model lacks an actual successful save")
    if any(
        key in receipt
        for key in ("convergence", "result_readback", "epr_result", "benchmark")
    ):
        raise RuntimeError("prepared model receipt contains solved-result evidence")
    source_binding = receipt.get("source")
    if (
        not isinstance(source_binding, dict)
        or source_binding.get("spec") != "aedt_spec.json"
    ):
        raise RuntimeError("prepared model source paths are not canonical")
    spec_path = verified(root, "aedt_spec.json", source_binding, "spec_sha256")
    spec = parse_aedt_spec(
        read_json(spec_path), base_dir=root, allow_historical_modeling=True
    )
    context = PreparedModelContext(
        root, receipt_path, receipt, save, source_binding, spec
    )
    expected_stage = _validate_prepared_identity(context)
    project = _verify_prepared_project(context, expected_stage)
    _validate_prepared_source(context)
    return PreparedModel(spec.mode, project, receipt_path, expected_stage, receipt)


def _validate_prepared_identity(context: PreparedModelContext) -> str:
    """Check family support, cohort identity, and the declared model stage."""
    spec = context.spec
    receipt = context.receipt
    if not isinstance(
        spec,
        (
            Q2dSpec,
            Q3dSpec,
            HfssEprSpec,
            HfssDrivenGeometrySpec,
            HfssDrivenSpec,
            HfssEigenmodeSpec,
        ),
    ):
        raise TypeError("this family has no supported prepared-model stage")
    validate_completion_cohort(context.root, receipt, spec)
    if receipt.get("mode") != spec.mode or receipt.get("connected") != {
        "aedt_version": spec.aedt_version,
        "pyaedt_version": spec.pyaedt_version,
    }:
        raise RuntimeError("prepared model connected family/version differs")
    expected_stage = "model_only" if spec.run_control is None else "model_with_setup"
    # Older prepared-with-setup receipts omit the marker but still bind controls.
    if "model_stage" in receipt:
        if receipt["model_stage"] != expected_stage:
            raise RuntimeError("prepared model stage differs from its declaration")
    elif spec.run_control is None:
        raise RuntimeError("model-only receipt lacks its explicit stage marker")
    return expected_stage


def _verify_prepared_project(
    context: PreparedModelContext, expected_stage: str
) -> Path:
    """Verify the sole saved project and its authored setup, when present."""
    receipt = context.receipt
    spec = context.spec
    project_relative = f"{spec.project_name}.aedt"
    outputs = receipt.get("outputs")
    if (
        receipt.get("project") != project_relative
        or not isinstance(outputs, dict)
        or set(outputs) != {project_relative}
    ):
        raise RuntimeError("prepared model must bind only its actual saved project")
    project = verified(context.root, project_relative, outputs)
    if context.save.get("project_sha256") != file_sha256(project):
        raise RuntimeError("prepared model saved-project identity differs")
    if spec.run_control is None:
        if "setup" not in receipt or receipt["setup"] is not None:
            raise RuntimeError("model-only receipt must declare no authored setup")
    elif isinstance(spec, Q3dSpec):
        validate_q3d_setup(receipt, spec)
    elif isinstance(spec, Q2dSpec):
        validate_q2d_setup(receipt, spec)
    else:
        validate_hfss_setup(receipt, spec)
    return project


def _validate_planar_conductors(
    objects: list[Any], source: dict[str, Any], junctions: Any
) -> int:
    """Bind each non-junction source conductor to its native object identity."""
    junction_polygons = {item.source_polygon_id for item in junctions}
    wanted = {
        (entity["semantic_id"], pid): entity
        for entity in source["conductors"]
        for pid in entity["polygon_ids"]
        if pid not in junction_polygons
    }
    actual = {
        (item.get("semantic_id"), item.get("source_polygon_id")): item
        for item in objects
        if item.get("kind") == "conductor"
    }
    if set(actual) != set(wanted):
        raise RuntimeError("prepared model conductor source inventory differs")
    for key, entity in wanted.items():
        item = actual[key]
        physical = detached(entity["physical_layer"])
        observed = item.get("observed", {})
        if item.get("physical_layer") != physical or item.get("native_object_type") != (
            "Sheet" if entity["representation"] == "sheet" else "Solid"
        ):
            raise RuntimeError(
                "prepared model conductor physical/native identity differs"
            )
        if entity["representation"] == "solid" and (
            str(observed.get("native_material_name", "")).casefold() != "pec"
            or observed.get("native_solve_inside") is not False
        ):
            raise RuntimeError("prepared model finite PEC assignment differs")
        if not isinstance(item.get("native_face_ids"), list) or not positive_unique_ids(
            item["native_face_ids"], len(item["native_face_ids"])
        ):
            raise RuntimeError("prepared model conductor native faces are absent")
    return len(actual)


def _validate_prepared_source(context: PreparedModelContext) -> None:
    """Bind source identity to the family-specific native model readback."""
    spec = context.spec
    if isinstance(spec, (HfssEprSpec, HfssDrivenGeometrySpec)):
        _validate_body_first_source(context)
    elif isinstance(spec, Q3dSpec):
        _validate_q3d_source(context)
    elif isinstance(spec, Q2dSpec):
        _validate_q2d_source(context)
    else:
        _validate_imported_gds_source(context)


def _validate_body_first_source(context: PreparedModelContext) -> None:
    """Verify the body-first planar source and its native object roster."""
    spec = context.spec
    receipt = context.receipt
    source_binding = context.source_binding
    if set(source_binding) != {
        "spec",
        "spec_sha256",
        "planar_source_sha256",
        "planar_model_sha256",
    } or (
        source_binding["planar_source_sha256"] != spec.geometry.source_sha256
        or source_binding["planar_model_sha256"] != spec.geometry.model_sha256
    ):
        raise RuntimeError("prepared model planar source identity differs")
    _validate_planar_model(receipt, spec.geometry)
    validate_lumped_readback(receipt, spec.geometry)
    if isinstance(spec, HfssDrivenGeometrySpec):
        ports = receipt.get("ports")
        if not isinstance(ports, list) or len(ports) != len(spec.ports):
            raise RuntimeError("prepared model native port inventory differs")
        validate_body_terminal_readback(receipt, spec)
    elif "ports" in receipt:
        raise RuntimeError(
            "body-first Eigenmode prepared receipt must not contain ports"
        )


def _validate_q3d_source(context: PreparedModelContext) -> None:
    """Verify Q3D source identity and native body assignments."""
    if set(context.source_binding) != {"spec", "spec_sha256"}:
        raise RuntimeError("Q3D prepared source must not contain an imported GDS")
    validate_geometry_source(context.root, context.source_binding, context.spec)
    validate_q3d_model_readback(
        context.root, context.receipt, context.spec, prepared=True
    )


def _validate_q2d_source(context: PreparedModelContext) -> None:
    """Verify Q2D source identity and native conductor assignments."""
    if set(context.source_binding) != {"spec", "spec_sha256"}:
        raise RuntimeError("Q2D prepared source must not contain an imported GDS")
    validate_q2d_model_readback(context.receipt, context.spec, prepared=True)


def _validate_imported_gds_source(context: PreparedModelContext) -> None:
    """Verify imported GDS identity, PEC assignments, and HFSS native ports."""
    receipt = context.receipt
    spec = context.spec
    if context.source_binding.get("gds") != "geometry/design.gds":
        raise RuntimeError("prepared model imported source path differs")
    verified(context.root, "geometry/design.gds", context.source_binding, "gds_sha256")
    validate_hfss_pec_bindings(receipt, spec)
    if isinstance(spec, HfssEigenmodeSpec):
        if receipt.get("ports") != []:
            raise RuntimeError("Eigenmode prepared receipt must have no ports")
    elif spec.mode == "terminal":
        validate_terminal_native_evidence(receipt.get("ports"), spec)
    else:
        validate_modal_native_evidence(receipt.get("ports"), spec)


def _validate_planar_model(receipt: dict[str, Any], prepared: Any) -> None:
    """Bind the source object roster and recorded native identities/materials."""
    geometry = receipt.get("geometry")
    if (
        not isinstance(geometry, dict)
        or geometry.get("source_sha256") != prepared.source_sha256
        or geometry.get("modeling") != prepared.modeling
    ):
        raise RuntimeError("prepared model native geometry source differs")
    objects = geometry.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RuntimeError("prepared model lacks actual native object inventory")
    names = [item.get("object_name") for item in objects if isinstance(item, dict)]
    ids = [item.get("native_object_id") for item in objects if isinstance(item, dict)]
    if (
        len(names) != len(objects)
        or len(set(names)) != len(objects)
        or not positive_unique_ids(ids, len(objects))
    ):
        raise RuntimeError("prepared model native object identity is invalid")
    conductor_count = _validate_planar_conductors(
        objects, prepared.source, prepared.junctions
    )
    domains = {
        item["semantic_id"]: item
        for item in prepared.source["solution_regions"]
        if not item["metadata"].get("is_auto_vacuum_region")
    }
    domains["Region"] = {"material_id": prepared.source["native_region"]["material_id"]}
    native_domains = {
        item.get("semantic_id"): item
        for item in objects
        if item.get("kind") == "solution_domain"
    }
    if set(native_domains) != set(domains) or len(objects) != conductor_count + len(
        native_domains
    ):
        raise RuntimeError("prepared model solution-domain inventory differs")
    for sid, domain in domains.items():
        item = native_domains[sid]
        if (
            item.get("material_id") != domain["material_id"]
            or item.get("native_object_type") != "Solid"
        ):
            raise RuntimeError("prepared model domain material/native identity differs")
    junctions = geometry.get("junctions")
    if not isinstance(junctions, list) or [
        item.get("junction_id") for item in junctions
    ] != [item.junction_id for item in prepared.junctions]:
        raise RuntimeError("prepared model declared junction inventory differs")
    if not isinstance(geometry.get("closed_enclosure"), dict):
        raise RuntimeError("prepared model lacks native enclosure readback")
