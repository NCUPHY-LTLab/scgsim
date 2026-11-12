"""Portable AEDT handoff writing and canonical input cohorts."""

from __future__ import annotations

import shutil
import tarfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from scgsim.aedt.data import (
    detached,
)
from scgsim.aedt.io import file_sha256, write_json
from scgsim.aedt.preparation.cohort import (
    GEOMETRY_SOURCE_PATHS,
    canonical_handoff_paths,
)
from scgsim.aedt.preparation.gds import GdsPreflight, gds_preflight, write_import_gds
from scgsim.aedt.preparation.handoff_files import (
    handoff_member,
    utc_now,
    write_script,
)
from scgsim.aedt.preparation.planar import validate_geometry_workers
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
)
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    SavedSolution,
)
from scgsim.aedt.specs.geometry import (
    PreparedPlanarGeometry,
)
from scgsim.aedt.specs.hfss import (
    HfssDrivenSpec,
    HfssEigenmodeSpec,
    HfssEprAnalysisSpec,
)
from scgsim.aedt.specs.hfss_common import EigenmodeRunControl
from scgsim.aedt.specs.parse import AedtSpec
from scgsim.aedt.specs.q2d import Q2dSpec
from scgsim.aedt.specs.q3d import Q3dSpec
from scgsim.aedt.versions import normalize_aedt_version


@dataclass(frozen=True)
class HandoffPlan:
    """The fixed, portable file set. Preparing it never opens AEDT."""

    run_dir: Path
    script_path: Path
    spec_path: Path
    metadata_path: Path
    receipt_path: Path
    manifest_path: Path
    archive_path: Path


def prepare_handoff(
    *, spec: AedtSpec, output_dir: str | Path, resources: AedtResources | None = None
) -> HandoffPlan:
    """Prepare one portable source cohort without opening AEDT."""
    return _PortableHandoff(spec, output_dir, resources).build()


@dataclass
class _PortableHandoff:
    """Carry one request through canonical file writing and packaging."""

    spec: AedtSpec
    output_dir: str | Path
    resources: AedtResources | None

    def build(self) -> HandoffPlan:
        """Write the portable cohort in its required order."""
        self._validate_source()
        self._copy_source()
        self._write_spec()
        self._write_metadata()
        self._write_receipt()
        return self._package()

    def _validate_source(self) -> None:
        if self.resources is not None and not isinstance(self.resources, AedtResources):
            raise TypeError("resources must be AedtResources or None")
        if (
            not isinstance(self.spec, Q2dSpec)
            and getattr(self.spec, "modeling", None) is None
        ):
            raise ValueError("new AEDT 3D execution requires explicit modeling")
        self.source_gds: Path | None = None
        self.preflight: GdsPreflight | None = None
        if not isinstance(self.spec, (Q2dSpec, Q3dSpec)):
            self.source_gds = cast(
                Path, cast(HfssDrivenSpec | HfssEigenmodeSpec, self.spec).gds_path
            ).resolve()
            if not self.source_gds.is_file():
                raise FileNotFoundError(
                    f"gds_path must be an existing file: {self.source_gds}"
                )
            self.preflight = gds_preflight(self.source_gds)
            actual_pairs = self.preflight["polygon_layer_datatypes"]
            declared_pairs = {
                (item.layer, item.datatype)
                for item in cast(
                    HfssDrivenSpec | HfssEigenmodeSpec, self.spec
                ).layer_imports
            }
            if actual_pairs != declared_pairs:
                raise ValueError(
                    f"GDS layer/datatype pairs {sorted(actual_pairs)!r} do not match spec {sorted(declared_pairs)!r}"
                )

    def _copy_source(self) -> None:
        self.run_dir = Path(self.output_dir).expanduser().resolve()
        if self.run_dir.exists():
            raise FileExistsError(
                "output_dir must be new; prepared handoffs never reuse directories"
            )
        self.metadata_dir = self.run_dir / "metadata"
        self.metadata_dir.mkdir(parents=True)
        self.copied_gds: Path | None = None
        if self.source_gds is not None:
            geometry_dir = self.run_dir / "geometry"
            geometry_dir.mkdir()
            self.copied_gds = geometry_dir / "design.gds"
            self.original_gds = geometry_dir / "source.gds"
            shutil.copy2(self.source_gds, self.original_gds)
            write_import_gds(
                self.source_gds,
                self.copied_gds,
                cast(
                    HfssDrivenSpec | HfssEigenmodeSpec, self.spec
                ).effective_layer_imports,
            )

        self.script_path = self.run_dir / "run_aedt.sh"
        self.spec_path = self.run_dir / "aedt_spec.json"
        self.metadata_path = self.metadata_dir / "aedt_handoff_metadata.json"
        self.receipt_path = self.metadata_dir / "aedt_run_receipt.json"
        self.manifest_path = self.metadata_dir / "aedt_handoff_manifest.json"
        self.archive_path = self.run_dir / "aedt_handoff.tar.gz"
        self.prepared_at = utc_now()
        self.started = time.perf_counter()
        self.prepared_source = record_runtime_source("prepared")

    def _write_spec(self) -> None:
        write_script(self.script_path)
        self.payload = self.spec.to_payload()
        if not isinstance(self.spec, (Q2dSpec, Q3dSpec)):
            self.payload["gds"]["path"] = "geometry/design.gds"
            self.payload["gds"]["source_path"] = "geometry/source.gds"
            self.payload["gds"]["source_sha256"] = file_sha256(self.original_gds)
            self.payload["gds"]["import_layer_map"] = [
                {
                    key: item[key]
                    for key in (
                        "layer",
                        "datatype",
                        "layer_name",
                        "physical_layer_id",
                        "native_import_layer",
                    )
                }
                for item in cast(
                    HfssDrivenSpec | HfssEigenmodeSpec, self.spec
                ).effective_layer_imports
            ]
        if isinstance(self.spec, Q3dSpec) and self.spec.geometry_source is not None:
            geometry_source = detached(self.spec.geometry_source)
            for key, destination in GEOMETRY_SOURCE_PATHS.items():
                reference = geometry_source["files"][key]
                source_path = Path(reference["path"])
                if file_sha256(source_path) != reference["sha256"]:
                    raise ValueError(f"Q3D source attachment identity differs: {key}")
                target = self.run_dir / destination
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)
                reference["path"] = destination
            self.payload["geometry_source"] = geometry_source
        write_json(self.spec_path, self.payload)

    def _write_metadata(self) -> None:
        self.files = {
            "spec": self.spec_path.name,
            "receipt": "metadata/aedt_run_receipt.json",
        }
        self.metadata = {
            "schema_version": "scgsim.aedt.handoff.v1",
            "expected_receipt_schema": RECEIPT_V3,
            "status": "prepared",
            "mode": self.spec.mode,
            "project": self.payload["project"],
            "materials": self.payload["materials"],
            "vacuum_material_id": self.payload["vacuum_material_id"],
            "run_control": self.payload["run_control"],
            "execution": {
                "resources": self.resources.to_payload() if self.resources else None
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
        }
        if self.copied_gds is not None and self.preflight is not None:
            self.files["gds"] = "geometry/design.gds"
            self.metadata["gds_sha256"] = file_sha256(self.copied_gds)
            self.metadata["gds_preflight"] = {
                "polygon_layer_datatypes": [
                    list(pair)
                    for pair in sorted(self.preflight["polygon_layer_datatypes"])
                ],
                "path_count": self.preflight["path_count"],
                "label_count": self.preflight["label_count"],
                "labels": "ignored as non-geometry",
            }
            self.source["gds"] = "geometry/design.gds"
            self.source["gds_sha256"] = file_sha256(self.copied_gds)
        write_json(self.metadata_path, self.metadata)

    def _write_receipt(self) -> None:
        self.receipt_path.write_bytes(
            encode_initial_receipt(
                initial_receipt_payload(
                    schema_version=RECEIPT_V3,
                    mode=self.spec.mode,
                    requested={
                        "aedt_version": self.spec.aedt_version,
                        "pyaedt_version": self.spec.pyaedt_version,
                        "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
                    },
                    pdk_materials=self.payload["materials"],
                    vacuum_material_id=self.payload["vacuum_material_id"],
                    source=self.source,
                    prepared_runtime_source_value=self.prepared_source,
                    outputs={},
                    prepared_at_utc=self.prepared_at,
                )
            )
        )

    def _package(self) -> HandoffPlan:
        self.allowed = tuple(
            self.run_dir / relative for relative in canonical_handoff_paths(self.spec)
        )
        write_json(
            self.manifest_path,
            {
                "schema_version": "scgsim.aedt.handoff-manifest.v1",
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


def prepare_epr_analysis_handoff(  # noqa: PLR0913 -- preserve the saved-field preparation keyword API.
    *,
    aedt_version: str,
    saved_solution: SavedSolution,
    geometry: PreparedPlanarGeometry,
    project_name: str,
    design_name: str,
    run_control: EigenmodeRunControl,
    epr_request: EprAnalysisRequest,
    output_dir: str | Path,
    geometry_workers: int | None,
    surface_field_selectors: Mapping[str, Literal["surface", "adjacent"]] | None = None,
) -> HandoffPlan:
    """Prepare one portable source cohort without opening AEDT."""
    return _SavedAnalysisHandoff(
        aedt_version,
        saved_solution,
        geometry,
        project_name,
        design_name,
        run_control,
        epr_request,
        output_dir,
        geometry_workers,
        surface_field_selectors,
    ).build()


@dataclass
class _SavedAnalysisHandoff:
    """Carry one request through canonical file writing and packaging."""

    aedt_version: str
    saved_solution: SavedSolution
    geometry: PreparedPlanarGeometry
    project_name: str
    design_name: str
    run_control: EigenmodeRunControl
    epr_request: EprAnalysisRequest
    output_dir: str | Path
    geometry_workers: int | None
    surface_field_selectors: Mapping[str, Literal["surface", "adjacent"]] | None

    def build(self) -> HandoffPlan:
        """Write the portable cohort in its required order."""
        self._copy_saved_solution()
        self._bind_spec()
        self._prepare_payload()
        self._write_metadata()
        self._write_receipt()
        return self._package()

    def _copy_saved_solution(self) -> None:
        if not isinstance(self.saved_solution, SavedSolution):
            raise TypeError("saved_solution must be SavedSolution")
        validate_geometry_workers(self.geometry_workers)
        self.aedt_version = normalize_aedt_version(self.aedt_version)
        self.run_dir = Path(self.output_dir).expanduser().resolve()
        if self.run_dir.exists():
            raise FileExistsError(
                "output_dir must be new; prepared handoffs never reuse directories"
            )
        self.saved_root = self.run_dir / "saved"
        self.metadata_dir = self.run_dir / "metadata"
        self.metadata_dir.mkdir(parents=True)
        self.copied_paths: list[Path] = []
        for member in self.saved_solution.members:
            relative = Path(member["path"])
            source = (self.saved_solution.root / relative).resolve()
            if (
                not source.is_relative_to(self.saved_solution.root)
                or source.is_symlink()
            ):
                raise RuntimeError("saved solution member escapes its sealed root")
            if (
                not source.is_file()
                or source.stat().st_size != member["bytes"]
                or file_sha256(source) != member["sha256"]
            ):
                raise RuntimeError(
                    f"saved solution member changed: {relative.as_posix()}"
                )
            target = self.saved_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            self.copied_paths.append(target)

    def _bind_spec(self) -> None:
        self.copied_solution = SavedSolution(
            root=self.saved_root,
            project_path=self.saved_root
            / self.saved_solution.project_path.relative_to(self.saved_solution.root),
            result_path=self.saved_root
            / self.saved_solution.result_path.relative_to(self.saved_solution.root),
            members=self.saved_solution.members,
            content_sha256=self.saved_solution.content_sha256,
            identity=self.saved_solution.identity,
            aedt_version=self.saved_solution.aedt_version,
        )
        self.spec = HfssEprAnalysisSpec(
            surface_field_selectors=self.surface_field_selectors,
            aedt_version=self.aedt_version,
            project_name=self.project_name,
            design_name=self.design_name,
            geometry=self.geometry,
            run_control=self.run_control,
            epr_request=self.epr_request,
            saved_solution=self.copied_solution,
        )

    def _prepare_payload(self) -> None:
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
        write_script(self.script_path, analyze_epr=True)
        write_json(self.spec_path, self.payload)
        self.files = {
            "spec": self.spec_path.name,
            "receipt": "metadata/aedt_run_receipt.json",
            "saved_project": (
                "saved/"
                + self.copied_solution.project_path.relative_to(
                    self.saved_root
                ).as_posix()
            ),
            "saved_results": (
                "saved/"
                + self.copied_solution.result_path.relative_to(
                    self.saved_root
                ).as_posix()
            ),
        }
        self.metadata = {
            "schema_version": "scgsim.aedt.handoff.v2",
            "expected_receipt_schema": RECEIPT_V3,
            "status": "prepared",
            "mode": self.spec.mode,
            "workflow": "epr_analysis",
            "project": self.payload["project"],
            "materials": self.materials,
            "vacuum_material_id": self.vacuum_ids[0],
            "run_control": self.payload["run_control"],
            "execution": {"geometry_workers": self.geometry_workers},
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
            "saved_solution_sha256": self.copied_solution.content_sha256,
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
                        "workflow": "epr_analysis",
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
            *sorted(self.copied_paths),
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
