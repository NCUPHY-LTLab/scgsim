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
from scgsim.aedt.preparation.sgb_handoff import SOURCE_KEYS, check_sgb_model_source
from scgsim.aedt.results.hfss_readback import (
    validate_hfss_setup,
)
from scgsim.aedt.results.provenance import require_current_receipt
from scgsim.aedt.results.q2d_readback import (
    validate_q2d_model_readback,
    validate_q2d_setup,
)
from scgsim.aedt.results.q3d_readback import (
    validate_q3d_model_readback,
    validate_q3d_setup,
)
from scgsim.aedt.results.readback_common import (
    validate_imported_sgb_model,
)
from scgsim.aedt.results.receipt_validation import (
    contained,
    validate_completion_cohort,
    verified,
)
from scgsim.aedt.results.terminal_readback import (
    validate_body_terminal_readback,
)
from scgsim.aedt.specs.hfss import (
    HfssDrivenGeometrySpec,
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
    require_current_receipt(receipt)
    if (
        receipt.get("status") != "native_preparation_only"
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
    spec = parse_aedt_spec(read_json(spec_path), base_dir=root)
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
    # A prepared-with-setup receipt may omit the marker; only a model-only receipt must carry it.
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


def _validate_prepared_source(context: PreparedModelContext) -> None:
    """Bind source identity to the family-specific native model readback."""
    spec = context.spec
    if isinstance(spec, HfssEprSpec):
        _validate_sgb_eigenmode_source(context)
    elif isinstance(spec, HfssDrivenGeometrySpec):
        _validate_body_first_source(context)
    elif isinstance(spec, Q3dSpec):
        _validate_q3d_source(context)
    else:
        _validate_q2d_source(context)


def _validate_sgb_eigenmode_source(context: PreparedModelContext) -> None:
    """The receipt binds the request's SGB model, and its geometry is that model as AEDT imported it."""
    try:
        check_sgb_model_source(context.source_binding, context.spec.model)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if set(context.source_binding) != {"spec", "spec_sha256", *SOURCE_KEYS}:
        raise RuntimeError("HFSS Eigenmode source members are not canonical")
    validate_imported_sgb_model(context.receipt, context.spec.model)
    if "ports" in context.receipt:
        raise RuntimeError("HFSS Eigenmode prepared receipt must not contain ports")


def _validate_body_first_source(context: PreparedModelContext) -> None:
    """The receipt binds the request's SGB model, and its geometry and lumped ports are as imported."""
    try:
        check_sgb_model_source(context.source_binding, context.spec.model)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if set(context.source_binding) != {"spec", "spec_sha256", *SOURCE_KEYS}:
        raise RuntimeError("HFSS Driven source members are not canonical")
    validate_body_terminal_readback(context.receipt, context.spec)


def _validate_q3d_source(context: PreparedModelContext) -> None:
    """The receipt binds the request's SGB model, and its Nets and terminals are the request's."""
    try:
        check_sgb_model_source(context.source_binding, context.spec.model)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if set(context.source_binding) != {"spec", "spec_sha256", *SOURCE_KEYS}:
        raise RuntimeError("Q3D source members are not canonical")
    validate_q3d_model_readback(
        context.root, context.receipt, context.spec, prepared=True
    )


def _validate_q2d_source(context: PreparedModelContext) -> None:
    """Verify Q2D source identity and native conductor assignments."""
    if set(context.source_binding) != {"spec", "spec_sha256"}:
        raise RuntimeError("Q2D prepared source must not contain an imported GDS")
    validate_q2d_model_readback(context.receipt, context.spec, prepared=True)
