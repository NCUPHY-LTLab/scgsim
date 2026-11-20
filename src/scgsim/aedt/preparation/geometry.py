"""GeometryPlan-to-AEDT request preparation adapters: each writes the handoff of an SGB model."""

from __future__ import annotations

import tarfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from scgsim.aedt.data import (
    detached,
)
from scgsim.aedt.io import file_sha256, write_json
from scgsim.aedt.preparation.cohort import canonical_handoff_paths
from scgsim.aedt.preparation.handoff import (
    HandoffPlan,
    prepare_handoff,
)
from scgsim.aedt.preparation.handoff_files import handoff_member, utc_now, write_script
from scgsim.aedt.preparation.lumped_elements import hfss_driven_inputs
from scgsim.aedt.preparation.sgb_handoff import attach_sgb_model, sgb_model_source
from scgsim.aedt.preparation.sgb_model import (
    PreparedSgbModel,
    aedt_net_names,
    prepare_sgb_model,
)
from scgsim.aedt.preparation.sgb_q3d import q3d_conductors
from scgsim.aedt.results.provenance import (
    RECEIPT_V3,
    encode_initial_receipt,
    initial_receipt_payload,
    record_runtime_source,
)
from scgsim.aedt.runtime.families.sgb_import import plan_import
from scgsim.aedt.specs.common import (
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    AedtResources,
    MatrixRunControl,
)
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    ExpressionCacheConvergence,
    ExpressionCacheConvergenceSet,
)
from scgsim.aedt.specs.hfss import (
    HfssDrivenGeometrySpec,
    HfssEprSpec,
)
from scgsim.aedt.specs.hfss_common import EigenmodeRunControl, HfssRunControl
from scgsim.aedt.specs.modeling import Modeling
from scgsim.aedt.specs.q3d import Q3dNetSpec, Q3dSpec
from scgsim.geometry import GeometryPlanSnapshot


def prepare_q3d_from_geometry(  # noqa: PLR0913 -- preserve the supported explicit preparation keyword API.
    snapshot: GeometryPlanSnapshot,
    *,
    aedt_version: str,
    modeling: Modeling,
    output_dir: str | Path,
    project_name: str,
    design_name: str,
    net_types: Mapping[str, Literal["Signal", "Ground"]],
    physical_ground_nets: Sequence[str],
    run_control: MatrixRunControl | None = None,
    resources: AedtResources | None = None,
) -> HandoffPlan:
    """Prepare the capacitance-only Q3D handoff of the plan's SGB model, without AEDT.

    The plan is built as the SGB model AEDT imports (:func:`prepare_sgb_model`, issue #67); its Region is the
    plan's ``VACUUM_REGION`` (decision 5).  ``net_types`` gives every Net of the model its Q3D type, keyed
    by Net id; ``physical_ground_nets`` records which Nets are physically ground, independently of their Q3D
    type.  Omitted ``run_control`` declares model-only preparation without solve setup.  The import's own
    first step (:func:`plan_import`) runs on the model before the handoff is written, so a model the Q3D
    import cannot realize is refused here, before AEDT is started (issue #266).
    """
    if isinstance(physical_ground_nets, str | bytes):
        raise TypeError("physical_ground_nets must be a sequence of Net ids")
    if not isinstance(snapshot, GeometryPlanSnapshot):
        raise TypeError("snapshot must be GeometryPlanSnapshot")
    with TemporaryDirectory(prefix="scgsim-q3d-model-") as temporary:
        model = prepare_sgb_model(
            snapshot.geometry_input,
            prepared_stack=snapshot.stack,
            modeling=modeling,
            output_dir=Path(temporary),
            ground_net_ids=tuple(physical_ground_nets),
        )
        plan_import(model, family="q3d")
        spec = Q3dSpec(
            aedt_version=aedt_version,
            modeling=model.modeling,
            project_name=project_name,
            design_name=design_name,
            model=model,
            nets=q3d_nets(model, net_types),
            run_control=run_control,
        )
        return prepare_handoff(spec=spec, output_dir=output_dir, resources=resources)


def q3d_nets(
    model: PreparedSgbModel,
    net_types: Mapping[str, Literal["Signal", "Ground"]],
) -> tuple[Q3dNetSpec, ...]:
    """One :class:`Q3dNetSpec` per Net of ``model``: its type and objects; ``net_types`` names exactly its Nets."""
    conductors = q3d_conductors(model)
    if set(net_types) != set(conductors):
        raise ValueError(
            f"net_types names {sorted(net_types)}, the model's Nets are {sorted(conductors)}"
        )
    names = aedt_net_names(conductors)
    return tuple(
        Q3dNetSpec(
            name=names[net],
            net_type=net_types[net],
            object_names=tuple(sorted(objects)),
        )
        for net, objects in conductors.items()
    )


def prepare_hfss_eigenmode_from_geometry(  # noqa: PLR0913 -- preserve the supported explicit preparation keyword API.
    *,
    aedt_version: str,
    model: PreparedSgbModel,
    project_name: str,
    design_name: str,
    run_control: EigenmodeRunControl | None = None,
    output_dir: str | Path,
    epr_request: EprAnalysisRequest | None = None,
    expression_convergence: ExpressionCacheConvergence
    | ExpressionCacheConvergenceSet
    | None = None,
    resources: AedtResources | None = None,
) -> HandoffPlan:
    """Prepare the HFSS Eigenmode handoff of an SGB model; omitted run_control authors no solve setup.

    The handoff carries the model's four files in its ``model/`` folder (issue #67).
    """
    spec = HfssEprSpec(
        aedt_version=aedt_version,
        modeling=model.modeling,
        project_name=project_name,
        design_name=design_name,
        model=model,
        run_control=run_control,
        epr_request=epr_request,
        expression_convergence=expression_convergence,
    )
    return _BodyHfssHandoff(
        spec=spec,
        materials=detached(model.materials),
        output_dir=output_dir,
        workflow="epr" if epr_request is not None else "body_first_eigenmode",
        resources=resources,
    ).build()


def prepare_hfss_driven_from_geometry(  # noqa: PLR0913 -- preserve the supported explicit preparation keyword API.
    snapshot: GeometryPlanSnapshot,
    *,
    aedt_version: str,
    modeling: Modeling,
    output_dir: str | Path,
    project_name: str,
    design_name: str,
    run_control: HfssRunControl | None = None,
    resources: AedtResources | None = None,
) -> HandoffPlan:
    """Prepare the HFSS Driven Terminal handoff of the plan's SGB model, without AEDT; omitted run_control authors no solve setup.

    The plan is built as the SGB model AEDT imports (:func:`prepare_sgb_model`, issue #67), with the plan's two
    excitation ports (``GeometryPlan.set_lumped_elements``) as its lumped Terminal ports, port *i* the plan's
    *i*-th ``LumpedPort`` (:func:`~scgsim.aedt.preparation.lumped_elements.hfss_driven_inputs`).  A lumped element
    that is not a port is refused by name, and the plan's Surface-EPR specs are not read.  The handoff carries
    the model's four files in its ``model/`` folder.  The import's own first step (:func:`plan_import`) runs on
    the model first, so a model the HFSS import cannot realize is refused here, before AEDT is started (issue
    #266).
    """
    if not isinstance(snapshot, GeometryPlanSnapshot):
        raise TypeError("snapshot must be GeometryPlanSnapshot")
    lumped = hfss_driven_inputs(snapshot)
    with TemporaryDirectory(prefix="scgsim-driven-model-") as temporary:
        model = prepare_sgb_model(
            lumped.build_input,
            prepared_stack=snapshot.stack,
            modeling=modeling,
            output_dir=Path(temporary),
            ports=lumped.lumped_ports,
        )
        plan_import(model, family="hfss")
        spec = HfssDrivenGeometrySpec(
            aedt_version=aedt_version,
            modeling=model.modeling,
            model=model,
            project_name=project_name,
            design_name=design_name,
            run_control=run_control,
        )
        return _BodyHfssHandoff(
            spec=spec,
            materials=detached(model.materials),
            output_dir=output_dir,
            workflow="body_first_driven_terminal",
            resources=resources,
        ).build()


@dataclass
class _BodyHfssHandoff:
    """Carry one request through canonical file writing and packaging."""

    spec: HfssDrivenGeometrySpec | HfssEprSpec
    materials: dict[str, Any]
    output_dir: str | Path
    workflow: str
    resources: AedtResources | None

    def build(self) -> HandoffPlan:
        """Write the portable cohort in its required order."""
        self._prepare_payload()
        self._write_metadata()
        self._write_receipt()
        return self._package()

    def _prepare_payload(self) -> None:
        if self.resources is not None and not isinstance(self.resources, AedtResources):
            raise TypeError("resources must be AedtResources or None")
        self.run_dir = Path(self.output_dir).expanduser().resolve()
        if self.run_dir.exists():
            raise FileExistsError(
                "output_dir must be new; prepared handoffs never reuse directories"
            )
        self.metadata_dir = self.run_dir / "metadata"
        self.metadata_dir.mkdir(parents=True)
        self.script_path = self.run_dir / "run_aedt.sh"
        self.spec_path = self.run_dir / "aedt_spec.json"
        self.metadata_path = self.metadata_dir / "aedt_handoff_metadata.json"
        self.receipt_path = self.metadata_dir / "aedt_run_receipt.json"
        self.manifest_path = self.metadata_dir / "aedt_handoff_manifest.json"
        self.archive_path = self.run_dir / "aedt_handoff.tar.gz"
        self.prepared_at = utc_now()
        self.started = time.perf_counter()
        self.prepared_source = record_runtime_source("prepared")
        self.payload = self.spec.to_payload()
        attach_sgb_model(self.spec.model, self.run_dir)
        self.vacuum_ids = sorted(
            material_id
            for material_id, record in self.materials.items()
            if isinstance(record, dict) and record.get("kind") == "vacuum"
        )
        if len(self.vacuum_ids) != 1:
            raise ValueError(
                "prepared EPR geometry requires exactly one vacuum material"
            )

    def _write_metadata(self) -> None:
        write_script(self.script_path)
        write_json(self.spec_path, self.payload)
        self.files = {
            "spec": self.spec_path.name,
            "receipt": "metadata/aedt_run_receipt.json",
        }
        self.metadata = {
            "schema_version": "scgsim.aedt.handoff.v2",
            "expected_receipt_schema": RECEIPT_V3,
            "status": "prepared",
            "mode": self.spec.mode,
            "workflow": self.workflow,
            "project": self.payload["project"],
            "materials": self.materials,
            "vacuum_material_id": self.vacuum_ids[0],
            "run_control": self.payload["run_control"],
            "execution": {
                "resources": self.resources.to_payload() if self.resources else None,
            },
            "pyaedt": self.payload["pyaedt"],
            "aedt": self.payload["aedt"],
            "files": self.files,
            "prepared_at_utc": self.prepared_at,
            "preparation_seconds": round(time.perf_counter() - self.started, 6),
        }
        self.source = {
            "spec": self.spec_path.name,
            "spec_sha256": file_sha256(self.spec_path),
            **sgb_model_source(self.spec.model),
        }
        write_json(self.metadata_path, self.metadata)

    def _write_receipt(self) -> None:
        self.receipt_path.write_bytes(
            encode_initial_receipt(
                initial_receipt_payload(
                    schema_version=RECEIPT_V3,
                    mode=self.spec.mode,
                    requested={
                        "aedt_version": self.spec.aedt_version,
                        "pyaedt_version": LOCKED_PYAEDT,
                        "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
                        "workflow": self.workflow,
                    },
                    pdk_materials=self.materials,
                    vacuum_material_id=self.vacuum_ids[0],
                    source=self.source,
                    prepared_runtime_source_value=self.prepared_source,
                    outputs={},
                    prepared_at_utc=self.prepared_at,
                )
            )
        )

    def _package(self) -> HandoffPlan:
        self.allowed = tuple(
            self.run_dir / path for path in canonical_handoff_paths(self.spec)
        )
        write_json(
            self.manifest_path,
            {
                "schema_version": "scgsim.aedt.handoff-manifest.v2",
                "expected_receipt_schema": RECEIPT_V3,
                "allowed_paths": [
                    path.relative_to(self.run_dir).as_posix() for path in self.allowed
                ],
                "members": [
                    handoff_member(path, self.run_dir)
                    for path in self.allowed
                    if path != self.manifest_path
                ],
            },
        )
        with tarfile.open(self.archive_path, "w:gz") as archive:
            for path in self.allowed:
                archive.add(
                    path,
                    arcname=path.relative_to(self.run_dir).as_posix(),
                    recursive=False,
                )
        return HandoffPlan(
            self.run_dir,
            self.script_path,
            self.spec_path,
            self.metadata_path,
            self.receipt_path,
            self.manifest_path,
            self.archive_path,
        )
