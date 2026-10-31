"""Public entry points that turn a GeometryPlan into an AEDT handoff.

Owns: the Q3D, HFSS eigenmode and HFSS driven preparation functions, and the
shared writer for body-first HFSS handoffs. Does not own: spec contents
(`scgsim.aedt.specs`), planar geometry preparation (`planar.py`) or the
sealing steps and GDS-based handoffs (`handoff.py`).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from scgsim.aedt._data import (
    detached,
)
from scgsim.aedt._io import file_sha256, write_json
from scgsim.aedt.preparation.handoff import (
    HandoffPlan,
    ReceiptSeed,
    _utc_now,
    _write_script,
    new_run_directory,
    prepare_handoff,
    seal_handoff,
    single_vacuum_material_id,
)
from scgsim.aedt.preparation.planar import validate_geometry_workers
from scgsim.aedt.results.provenance import (
    RECEIPT_V3,
    prepared_runtime_source,
)
from scgsim.aedt.specs.common import (
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    REQUIRED_AEDT_VERSION,
    AedtResources,
    LumpedRlc,
    LumpedTerminalPort,
    MatrixRunControl,
    PdkMaterial,
    TerminalPort,
)
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    ExpressionCacheConvergence,
)
from scgsim.aedt.specs.geometry import (
    PreparedPlanarGeometry,
)
from scgsim.aedt.specs.hfss import EigenmodeRunControl, HfssRunControl
from scgsim.aedt.specs.hfss_prepared import HfssDrivenGeometrySpec, HfssEprSpec
from scgsim.aedt.specs.modeling import Modeling
from scgsim.geometry import GeometryPlanSnapshot


def prepare_q3d_from_geometry(  # noqa: PLR0913 - public keyword-only API
    snapshot: GeometryPlanSnapshot,
    *,
    modeling: Modeling,
    output_dir: str | Path,
    project_name: str,
    design_name: str,
    materials: Mapping[str, PdkMaterial],
    net_types: Mapping[str, Literal["Signal", "Ground"]],
    physical_ground_nets: Sequence[str],
    run_control: MatrixRunControl | None = None,
    region_padding_um: Sequence[float],
    resources: AedtResources | None = None,
) -> HandoffPlan:
    """Prepare finite C/G geometry from detached source facts, without AEDT.

    The caller owns final Net membership and native net types independently
    of physical ground roles. Component/PDK source geometry remains unchanged;
    generated import layers, source trace and originals travel in the handoff.
    Omitted run_control declares model-only preparation without solve setup.
    """
    from scgsim.aedt.preparation.q3d_geometry import lower_q3d_geometry

    with TemporaryDirectory(prefix="scgsim-q3d-geometry-") as temporary:
        spec = lower_q3d_geometry(
            snapshot,
            modeling=modeling,
            directory=Path(temporary),
            project_name=project_name,
            design_name=design_name,
            materials=materials,
            net_types=net_types,
            physical_ground_nets=physical_ground_nets,
            run_control=run_control,
            region_padding_um=region_padding_um,
        )
        return prepare_handoff(spec=spec, output_dir=output_dir, resources=resources)


def prepare_hfss_eigenmode_from_geometry(  # noqa: PLR0913 - public keyword-only API
    *,
    modeling: Modeling,
    geometry: PreparedPlanarGeometry,
    project_name: str,
    design_name: str,
    run_control: EigenmodeRunControl | None = None,
    output_dir: str | Path,
    epr_request: EprAnalysisRequest | None = None,
    lumped_rlcs: Sequence[LumpedRlc] | None = None,
    expression_convergence: ExpressionCacheConvergence | None = None,
    geometry_workers: int | None = None,
    resources: AedtResources | None = None,
) -> HandoffPlan:
    """Prepare a body-first model; omitted run_control authors no solve setup."""
    if lumped_rlcs is not None:
        geometry = geometry.with_lumped_rlcs(lumped_rlcs)
    spec = HfssEprSpec(
        modeling=modeling,
        project_name=project_name,
        design_name=design_name,
        geometry=geometry,
        run_control=run_control,
        epr_request=epr_request,
        expression_convergence=expression_convergence,
    )
    return _prepare_body_hfss_handoff(
        spec=spec,
        geometry=geometry,
        output_dir=output_dir,
        workflow="epr" if epr_request is not None else "body_first_eigenmode",
        geometry_workers=geometry_workers,
        resources=resources,
    )


def prepare_hfss_driven_from_geometry(  # noqa: PLR0913 - public keyword-only API
    *,
    modeling: Modeling,
    geometry: PreparedPlanarGeometry,
    project_name: str,
    design_name: str,
    run_control: HfssRunControl | None = None,
    ports: Sequence[TerminalPort | LumpedTerminalPort],
    output_dir: str | Path,
    geometry_workers: int | None = None,
    resources: AedtResources | None = None,
) -> HandoffPlan:
    """Prepare a Terminal model and ports, with optional complete solve setup."""
    spec = HfssDrivenGeometrySpec(
        modeling=modeling,
        geometry=geometry,
        project_name=project_name,
        design_name=design_name,
        run_control=run_control,
        ports=tuple(ports),
    )
    return _prepare_body_hfss_handoff(
        spec=spec,
        geometry=geometry,
        output_dir=output_dir,
        workflow="body_first_driven_terminal",
        geometry_workers=geometry_workers,
        resources=resources,
    )


def _prepare_body_hfss_handoff(  # noqa: PLR0913 - one keyword per handoff input
    *, spec, geometry, output_dir, workflow, geometry_workers, resources
):
    """One canonical portable cohort writer for body-first HFSS requests."""
    validate_geometry_workers(geometry_workers)
    if resources is not None and not isinstance(resources, AedtResources):
        raise TypeError("resources must be AedtResources or None")
    plan = HandoffPlan.under(new_run_directory(output_dir))
    plan.metadata_path.parent.mkdir(parents=True)
    prepared_at = _utc_now()
    started = time.perf_counter()
    prepared_source = prepared_runtime_source()
    payload = spec.to_payload()
    materials = detached(geometry.source)["materials"]
    vacuum_id = single_vacuum_material_id(materials)

    _write_script(plan.script_path)
    write_json(plan.spec_path, payload)
    files = {
        "spec": plan.spec_path.name,
        "receipt": "metadata/aedt_run_receipt.json",
    }
    metadata = {
        "schema_version": "scgsim.aedt.handoff.v2",
        "expected_receipt_schema": RECEIPT_V3,
        "status": "prepared",
        "mode": spec.mode,
        "workflow": workflow,
        "project": payload["project"],
        "materials": materials,
        "vacuum_material_id": vacuum_id,
        "run_control": payload["run_control"],
        "execution": {
            "geometry_workers": geometry_workers,
            "resources": resources.to_payload() if resources else None,
        },
        "pyaedt": payload["pyaedt"],
        "aedt": payload["aedt"],
        "files": files,
        "prepared_at_utc": prepared_at,
        "preparation_seconds": round(time.perf_counter() - started, 6),
    }
    source = {
        "spec": plan.spec_path.name,
        "spec_sha256": file_sha256(plan.spec_path),
        "planar_source_sha256": geometry.source_sha256,
        "planar_model_sha256": geometry.model_sha256,
    }
    seal_handoff(
        plan,
        metadata=metadata,
        receipt=ReceiptSeed(
            mode=spec.mode,
            requested={
                "aedt_version": REQUIRED_AEDT_VERSION,
                "pyaedt_version": LOCKED_PYAEDT,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
                "workflow": workflow,
            },
            pdk_materials=materials,
            vacuum_material_id=vacuum_id,
            source=source,
            prepared_runtime_source_value=prepared_source,
            prepared_at_utc=prepared_at,
        ),
        manifest_schema="scgsim.aedt.handoff-manifest.v2",
        allowed=(
            plan.script_path,
            plan.spec_path,
            plan.metadata_path,
            plan.receipt_path,
            plan.manifest_path,
        ),
    )
    return plan
