"""Orchestrate one Palace mesh: SGB geometry, the meshing module, Palace's lowering, paired artifacts.

Portions adapted from gsim 0.2.0+scq.1, revision
8f5dc6c05255d003a9c6d8959537bcf8068379d3.
Copyright 2025 GDSFactory. Licensed under the Apache License, Version 2.0.
Modified for SCGSim in-tree geometry and mesh handoff integration.
See NOTICE and LICENSES/gsim-LICENSE.txt for retained upstream notices.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.contracts.palace_run_folder import (
    GEOMETRY_DIR,
    LOGS_DIR,
    MESH,
    MESH_DIR,
    MESH_MANIFEST,
    METADATA_DIR,
    OUTPUT_DIR,
    SOURCE_GDS,
    SOURCE_STACK,
    json_text,
    route_provenance,
)
from scgsim.contracts.sgb_sidecar import read_sidecar
from scgsim.diagnostics import diagnostic_stage
from scgsim.geometry.models.grid import geometry_grid_context
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.sgb import SgbArtifacts
from scgsim.geometry.source.port_sheets import with_port_sheet_metadata
from scgsim.geometry.surface_epr.build import build_banded_geometry
from scgsim.meshing import (
    FineStructureLimits,
    MeshResult,
    mesh_sgb,
)
from scgsim.meshing.fine_structure import below_steps
from scgsim.preparation import native_preparation

from .controls import MeshControls
from .manifest import SGB_RUNTIME_AUTHORITY, build_mesh_manifest
from .physical_groups import Groups, groups_from_sidecar
from .process import lower_mesh_in_worker
from .source import (
    route_a_thin_film_provenance,
    scgsim_version,
    write_json,
)


@dataclass(frozen=True)
class MeshBuildResult:
    """Result of SGB-backed mesh generation."""

    output_dir: Path
    mesh_path: Path
    groups: Groups
    mesh_manifest_path: Path
    provenance_path: Path
    stack_path: Path
    xao_path: Path
    sidecar_path: Path
    quality_report_path: Path


@diagnostic_stage("palace.mesh.prepare")
def mesh_route_geometry(
    built: RouteGeometry, controls: MeshControls
) -> MeshBuildResult:
    """Mesh a built SGB geometry to the quality threshold, lower it to Palace, and write the paired records.

    ``built`` is what :func:`build_route_geometry` wrote into the run folder; the mesh, the manifest and the
    provenance go into the same folder.  A mesh the meshing module cannot bring to the threshold of
    ``controls`` raises :class:`scgsim.meshing.MeshRepairError` carrying the quality report.
    """
    paths, artifacts = built.paths, built.artifacts
    sidecar_bytes = artifacts.sidecar.read_bytes()
    sidecar = read_sidecar(
        artifacts.sidecar
    )  # the bytes above are what the manifest pins
    groups = groups_from_sidecar(sidecar)

    # The mesh and the lowering run in processes of their own; the preparation budget is reserved
    # for the whole native stage.
    with native_preparation():
        meshed = _mesh_geometry(artifacts, paths, controls)
        observed = lower_mesh_in_worker(
            source_mesh=meshed.mesh,
            mesh_path=paths.mesh_path,
            groups=groups,
            run_dir=paths.run_dir,
        )
    # The module's linear-or-elevated MSH 4.1 is an intermediate; palace.msh carries the same elements.
    meshed.mesh.unlink()

    manifest = build_mesh_manifest(
        groups=groups,
        sidecar=sidecar,
        sidecar_file={
            "path": paths.relative(artifacts.sidecar),
            "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        },
        mesh_name=paths.mesh_path.name,
        route=built.route,
        route_a_thin_film=built.route_a_thin_film,
        meshing=_meshing_record(meshed, observed, paths),
    )
    write_json(paths.manifest_path, manifest)
    write_json(
        paths.provenance_path,
        {
            "scgsim_version": scgsim_version(),
            "route": built.route,
            **(
                {"route_a_thin_film": built.route_a_thin_film}
                if built.route_a_thin_film is not None
                else {}
            ),
            "geometry_authority": SGB_RUNTIME_AUTHORITY,
            "component_top_cell": built.top_cell,
            "run_dir": ".",
            "stack_path": paths.relative(paths.stack_path),
            "xao_path": paths.relative(artifacts.xao),
            "sidecar_path": paths.relative(artifacts.sidecar),
        },
    )
    return MeshBuildResult(
        output_dir=paths.run_dir,
        mesh_path=paths.mesh_path,
        groups=groups,
        mesh_manifest_path=paths.manifest_path,
        provenance_path=paths.provenance_path,
        stack_path=paths.stack_path,
        xao_path=artifacts.xao,
        sidecar_path=artifacts.sidecar,
        quality_report_path=meshed.report_path,
    )


@dataclass(frozen=True)
class BuildPaths:
    """Paired source and result locations shared by one mesh preparation."""

    run_dir: Path
    geometry_dir: Path
    mesh_dir: Path
    gds_path: Path
    stack_path: Path
    mesh_path: Path
    manifest_path: Path
    provenance_path: Path

    def relative(self, path: Path) -> str:
        """``path`` as a POSIX path relative to the run folder."""
        return path.relative_to(self.run_dir).as_posix()


@dataclass(frozen=True)
class RouteGeometry:
    """The SGB geometry of one run folder, before any meshing."""

    paths: BuildPaths
    artifacts: SgbArtifacts
    route: str
    top_cell: str
    route_a_thin_film: dict[str, Any] | None


@diagnostic_stage("palace.mesh.geometry")
def build_route_geometry(  # noqa: PLR0913 -- the plan's sources, its prepared input, and the route and bands
    *,
    stack: Mapping[str, Any],
    route: str,
    output_dir: str | Path,
    source_gds_bytes: bytes,
    surface_epr_specs: Mapping[str, Mapping[str, Any]] | None = None,
    port_sheet_source_layers: Sequence[Mapping[str, Any]] = (),
    source_geometry_input: GeometryBuildInput | None = None,
    representation_context: RepresentationContext | None = None,
) -> RouteGeometry:
    """Write the sources and build the Route-A/Route-B SGB geometry, the stages of a mesh before meshing.

    This is the modeling path ``mesh()`` and the layout modeling check share: a layout SGB
    cannot model is refused here, with the error SGB raises.  A Surface-EPR band narrower than the
    meshing stage's face-width limit is refused first, before anything is written or built.
    """
    if representation_context is None or source_geometry_input is None:
        raise ValueError(
            "fresh Palace mesh requires neutral effective geometry input and representation_context from preparation"
        )
    if not isinstance(stack, Mapping):
        raise TypeError("stack must be a mapping.")
    if route not in {"A", "B"}:
        raise ValueError("route must be either 'A' or 'B'.")
    _refuse_bands_narrower_than_a_face(surface_epr_specs, source_geometry_input)
    thin_film = route_a_thin_film_provenance(stack, route)
    stack = with_port_sheet_metadata(stack, port_sheet_source_layers)
    paths = _prepare_paths(output_dir, route)
    top_cell = _write_sources(stack, source_gds_bytes, paths)
    artifacts = build_banded_geometry(
        source_geometry_input,
        representation_context,
        route=route,
        surface_epr_specs=surface_epr_specs,
        output_dir=paths.geometry_dir,
    )
    return RouteGeometry(paths, artifacts, route, top_cell, thin_film)


def _refuse_bands_narrower_than_a_face(
    specs: Mapping[str, Mapping[str, Any]] | None, build_input: GeometryBuildInput
) -> None:
    """Refuse every requested band narrower than the face width the meshing stage accepts.

    A band's width is set by its margins alone: band 0 is ``m1`` wide and band ``k`` is ``m(k+1) - m(k)``.
    A band below the class B limit becomes a face the meshing stage refuses after the build, and its lineage
    (one source edge, one band offset) would read as an SGB defect; the request made it narrow.  This is the
    first point where both the margins and the model grid are known: the specs come with the simulation,
    the grid with the prepared input, and nothing is built yet.
    """
    margin_lists = _margin_lists(specs or {})
    if not margin_lists:
        return
    grid_um = geometry_grid_context(build_input).grid_size_um
    limit = FineStructureLimits().face_width_steps
    narrow = [
        f"{label} band={band} ({lower:g} to {upper:g} um) is {upper - lower:.6g} um "
        f"({(upper - lower) / grid_um:.3g} grid steps) wide"
        for label, margins in margin_lists
        for band, (lower, upper) in enumerate(
            zip((0.0, *margins), margins, strict=False)
        )
        if below_steps(upper - lower, limit, grid_um)
    ]
    if narrow:
        raise ValueError(
            "Surface EPR bands narrower than the meshing stage's face-width limit of "
            f"{limit:g} grid steps ({limit * grid_um:g} um): {'; '.join(narrow)}. A band this narrow is a "
            "face the mesher cannot use. Widen the margins in surface_epr_specs so that every band is at "
            f"least {limit:g} grid steps wide."
        )


def _margin_lists(
    specs: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, tuple[float, ...]]]:
    """Each margin list of the specs, named as the user wrote it: ``MA inset_margins_um`` or per conductor."""
    lists = []
    for interface_type, spec in specs.items():
        if spec.get("inset_margins_um"):
            lists.append(
                (f"{interface_type} inset_margins_um", tuple(spec["inset_margins_um"]))
            )
        for entity, margins in (spec.get("inset_margins_by_entity_um") or {}).items():
            lists.append(
                (
                    f"{interface_type} inset_margins_by_entity_um[{entity!r}]",
                    tuple(margins),
                )
            )
    return lists


def _prepare_paths(output_dir: str | Path, route: str) -> BuildPaths:
    run_dir = Path(output_dir).expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"output_dir must be empty before mesh: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for folder in (GEOMETRY_DIR, MESH_DIR, METADATA_DIR, LOGS_DIR, OUTPUT_DIR):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)

    return BuildPaths(
        run_dir=run_dir,
        geometry_dir=run_dir / GEOMETRY_DIR,
        mesh_dir=run_dir / MESH_DIR,
        gds_path=run_dir / SOURCE_GDS,
        stack_path=run_dir / SOURCE_STACK,
        mesh_path=run_dir / MESH,
        manifest_path=run_dir / MESH_MANIFEST,
        provenance_path=run_dir / route_provenance(route),
    )


def _write_sources(
    stack: Mapping[str, Any], source_gds_bytes: bytes, paths: BuildPaths
) -> str:
    """Write the plan's GDS bytes and the stack paired with them; return the plan's top cell."""
    if not isinstance(source_gds_bytes, bytes):
        raise TypeError("plan GDS source must be bytes")
    wanted = stack.get("metadata", {}).get("gds_sha256")
    observed = hashlib.sha256(source_gds_bytes).hexdigest()
    if wanted != observed:
        raise ValueError("plan GDS bytes differ from paired stack source")
    top_cell = stack.get("metadata", {}).get("top_cell_name")
    if not isinstance(top_cell, str) or not top_cell:
        raise ValueError("plan stack requires top_cell_name")
    paths.gds_path.write_bytes(source_gds_bytes)
    paths.stack_path.write_text(json_text(dict(stack)), encoding="utf-8")

    return top_cell


@diagnostic_stage("palace.mesh.module")
def _mesh_geometry(
    artifacts: SgbArtifacts, paths: BuildPaths, controls: MeshControls
) -> MeshResult:
    """Mesh the XAO to the threshold with the meshing module; the sidecar is the pair of this XAO."""
    return mesh_sgb(
        artifacts.xao,
        artifacts.sidecar,
        paths.mesh_dir,
        threshold=controls.quality_threshold,
        recipe=controls.recipe,
        budgets=controls.budgets,
        mesher=controls.mesher,
    )


def _meshing_record(
    meshed: MeshResult, observed: Mapping[str, Any], paths: BuildPaths
) -> dict[str, Any]:
    """The meshing section of the manifest: the settings and report of the module, and the written mesh."""
    report = meshed.report
    return {
        "recipe": report["recipe"],
        "repair": report["repair"],
        "mesher": report["mesher"],
        "budgets": report["budgets"],
        "threshold": report["threshold"],
        "quality_measure": report["quality_measure"],
        "stop": report["stop"],
        "rounds": report["rounds"],
        "linear_mesh": report["mesh"],
        "quality_report": {
            "path": paths.relative(meshed.report_path),
            "sha256": hashlib.sha256(meshed.report_path.read_bytes()).hexdigest(),
        },
        "gmsh_version": observed["gmsh_version"],
        "statistics": observed["statistics"],
    }
