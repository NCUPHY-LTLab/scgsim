"""GeometryPlan-to-AEDT request preparation adapters."""

from __future__ import annotations

import tarfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from scgsim.aedt.data import (
    detached,
)
from scgsim.aedt.io import file_sha256, write_json
from scgsim.aedt.preparation.handoff import (
    HandoffPlan,
    prepare_handoff,
)
from scgsim.aedt.preparation.handoff_files import handoff_member, utc_now, write_script
from scgsim.aedt.preparation.planar import validate_geometry_workers
from scgsim.aedt.preparation.q3d_geometry import lower_q3d_geometry
from scgsim.aedt.results.provenance import (
    RECEIPT_V3,
    encode_initial_receipt,
    initial_receipt_payload,
    record_runtime_source,
)
from scgsim.aedt.specs.common import (
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
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
    ExpressionCacheConvergenceSet,
)
from scgsim.aedt.specs.geometry import (
    PreparedPlanarGeometry,
)
from scgsim.aedt.specs.hfss import (
    HfssDrivenGeometrySpec,
    HfssEprSpec,
)
from scgsim.aedt.specs.hfss_common import EigenmodeRunControl, HfssRunControl
from scgsim.aedt.specs.modeling import Modeling
from scgsim.aedt.specs.q3d import Q3dTerminalPair
from scgsim.geometry import GeometryPlanSnapshot


def prepare_q3d_from_geometry(  # noqa: PLR0913 -- preserve the supported explicit preparation keyword API.
    snapshot: GeometryPlanSnapshot,
    *,
    aedt_version: str,
    modeling: Modeling,
    output_dir: str | Path,
    project_name: str,
    design_name: str,
    materials: Mapping[str, PdkMaterial],
    net_types: Mapping[str, Literal["Signal", "Ground"]],
    physical_ground_nets: Sequence[str],
    terminal_pairs: Mapping[str, Q3dTerminalPair] | None = None,
    run_control: MatrixRunControl | None = None,
    region_padding_um: Sequence[float],
    resources: AedtResources | None = None,
) -> HandoffPlan:
    """Prepare finite C/G geometry from detached source facts, without AEDT.

    The caller owns final Net membership and native net types independently
    of physical ground roles. Component/PDK source geometry remains unchanged;
    generated import layers, source trace and originals travel in the handoff.
    Omitted run_control declares model-only preparation without solve setup.
    Optional terminal_pairs map final Net IDs to exact source Entity identities
    and caller-selected global sides; each selected Entity must emit one body.
    """

    with TemporaryDirectory(prefix="scgsim-q3d-geometry-") as temporary:
        spec = lower_q3d_geometry(
            snapshot,
            aedt_version=aedt_version,
            modeling=modeling,
            directory=Path(temporary),
            project_name=project_name,
            design_name=design_name,
            materials=materials,
            net_types=net_types,
            physical_ground_nets=physical_ground_nets,
            terminal_pairs=terminal_pairs,
            run_control=run_control,
            region_padding_um=region_padding_um,
        )
        return prepare_handoff(spec=spec, output_dir=output_dir, resources=resources)


def prepare_hfss_eigenmode_from_geometry(  # noqa: PLR0913 -- preserve the supported explicit preparation keyword API.
    *,
    aedt_version: str,
    modeling: Modeling,
    geometry: PreparedPlanarGeometry,
    project_name: str,
    design_name: str,
    run_control: EigenmodeRunControl | None = None,
    output_dir: str | Path,
    epr_request: EprAnalysisRequest | None = None,
    lumped_rlcs: Sequence[LumpedRlc] | None = None,
    expression_convergence: ExpressionCacheConvergence
    | ExpressionCacheConvergenceSet
    | None = None,
    geometry_workers: int | None = None,
    resources: AedtResources | None = None,
) -> HandoffPlan:
    """Prepare a body-first model; omitted run_control authors no solve setup."""

    if lumped_rlcs is not None:
        geometry = geometry.with_lumped_rlcs(lumped_rlcs)
    spec = HfssEprSpec(
        aedt_version=aedt_version,
        modeling=modeling,
        project_name=project_name,
        design_name=design_name,
        geometry=geometry,
        run_control=run_control,
        epr_request=epr_request,
        expression_convergence=expression_convergence,
    )
    return _BodyHfssHandoff(
        spec=spec,
        geometry=geometry,
        output_dir=output_dir,
        workflow="epr" if epr_request is not None else "body_first_eigenmode",
        geometry_workers=geometry_workers,
        resources=resources,
    ).build()


def prepare_hfss_driven_from_geometry(  # noqa: PLR0913 -- preserve the supported explicit preparation keyword API.
    *,
    aedt_version: str,
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
        aedt_version=aedt_version,
        modeling=modeling,
        geometry=geometry,
        project_name=project_name,
        design_name=design_name,
        run_control=run_control,
        ports=tuple(ports),
    )
    return _BodyHfssHandoff(
        spec=spec,
        geometry=geometry,
        output_dir=output_dir,
        workflow="body_first_driven_terminal",
        geometry_workers=geometry_workers,
        resources=resources,
    ).build()


@dataclass
class _BodyHfssHandoff:
    """Carry one request through canonical file writing and packaging."""

    spec: HfssDrivenGeometrySpec | HfssEprSpec
    geometry: PreparedPlanarGeometry
    output_dir: str | Path
    workflow: str
    geometry_workers: int | None
    resources: AedtResources | None

    def build(self) -> HandoffPlan:
        """Write the portable cohort in its required order."""
        self._prepare_payload()
        self._write_metadata()
        self._write_receipt()
        return self._package()

    def _prepare_payload(self) -> None:
        validate_geometry_workers(self.geometry_workers)
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
        self.materials = detached(self.geometry.source)["materials"]
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
                "geometry_workers": self.geometry_workers,
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
            "planar_source_sha256": self.geometry.source_sha256,
            "planar_model_sha256": self.geometry.model_sha256,
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
        self.allowed = (
            self.script_path,
            self.spec_path,
            self.metadata_path,
            self.receipt_path,
            self.manifest_path,
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
