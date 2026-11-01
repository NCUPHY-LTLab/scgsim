"""Orchestrate geometry preparation, the fresh mesh worker, and paired artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from scgsim.diagnostics import _stage
from scgsim.geometry.compiler.builder import SemanticGeometryBuilder

from .controls import mesh_controls, normalize_mesh_sizing
from .process import mesh_from_route_xao
from .provenance import (
    GSIM_HIGH_ORDER_DERIVATION_SHA,
    GSIM_SHA,
    SGB_DERIVATION_BASE_SHA,
    SGB_DERIVATION_IMPORTED_SHA,
    SGB_RUNTIME_AUTHORITY,
)
from .source import (
    build_mesh_manifest,
    component_gds_top_cell_name,
    route_a_thin_film_provenance,
    scgsim_version,
    with_port_sheet_metadata,
    write_component_gds,
    write_indium_ground_bump_receipt,
    write_json,
)


@dataclass(frozen=True)
class MeshBuildResult:
    """Result of SGB-backed mesh generation."""

    output_dir: Path
    mesh_path: Path
    groups: dict[str, dict[str, Any]]
    mesh_manifest_path: Path
    provenance_path: Path
    stack_path: Path
    xao_path: Path
    sidecar_path: Path


@_stage("palace.mesh.prepare")
def build_route_mesh(  # noqa: PLR0913 -- preserve the supported mesh preparation API.
    *,
    component: Any,
    stack: Mapping[str, Any],
    route: str,
    output_dir: str | Path,
    size_policy: str = "scgsim_default",
    refined_mesh_size: float | None = None,
    max_mesh_size: float | None = None,
    algorithm_3d: str = "Delaunay",
    threads: int = 1,
    surface_threads: int = 1,
    geometry_order: int = 1,
    high_order_optimize: bool = True,
    port_sheet_source_layers: Sequence[Mapping[str, Any]] = (),
    indium_ground_bump_fill: Mapping[str, Any] | None = None,
    source_gds_bytes: bytes | None = None,
    source_geometry_input: Any | None = None,
    representation_context: Any | None = None,
) -> MeshBuildResult:
    """Lower Route-A/Route-B SGB geometry, mesh it, and emit artifacts."""

    if representation_context is None or source_geometry_input is None:
        raise ValueError(
            "fresh Palace mesh requires neutral effective geometry input and representation_context from preparation"
        )
    if not isinstance(stack, Mapping):
        raise TypeError("stack must be a mapping after set_stack processing.")
    if route not in {"A", "B"}:
        raise ValueError("route must be either 'A' or 'B'.")

    sizing = normalize_mesh_sizing(
        size_policy=size_policy,
        refined_mesh_size=refined_mesh_size,
        max_mesh_size=max_mesh_size,
    )
    refined_mesh_size = sizing.refined_mesh_size
    max_mesh_size = sizing.max_mesh_size
    controls = mesh_controls(
        size_policy=sizing.size_policy,
        algorithm_3d=algorithm_3d,
        threads=threads,
        surface_threads=surface_threads,
        geometry_order=geometry_order,
        high_order_optimize=high_order_optimize,
    )
    thin_film = route_a_thin_film_provenance(stack, route)
    stack = with_port_sheet_metadata(stack, port_sheet_source_layers)
    paths = _prepare_paths(output_dir, route)
    top_cell = _write_sources(component, stack, source_gds_bytes, paths)
    # Source placement, curves, holes and endpoint projection are already bound.
    build_input = source_geometry_input
    source_curve_provenance = {
        "boundary_curves": [asdict(r) for r in build_input.boundary_curves],
        "boundary_reconstruction": [
            asdict(r) for r in build_input.boundary_reconstruction
        ],
    }
    _write_optional_bump_receipt(indium_ground_bump_fill, paths)

    # SGB/OCC construction remains in the calling process. Only Gmsh lowering
    # receives a fresh native-process boundary after an XAO has been written.
    SemanticGeometryBuilder().build(
        build_input,
        route=cast(Literal["A", "B"], route),
        run_folder=paths.run_dir,
        representation_context=representation_context,
    )
    if not paths.xao_path.is_file():
        raise FileNotFoundError(
            f"SGB Route-{route} XAO not generated: {paths.xao_path}"
        )
    if not paths.sidecar_path.is_file():
        raise FileNotFoundError(
            f"SGB Route-{route} sidecar not generated: {paths.sidecar_path}"
        )

    records = json.loads(paths.sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("04_export_physical_groups.json must contain a list")

    mesh_path, groups, meshing = mesh_from_route_xao(
        xao_path=paths.xao_path,
        route=route,
        records=records,
        stack=stack,
        mesh_path=paths.mesh_path,
        refined_mesh_size=refined_mesh_size,
        max_mesh_size=max_mesh_size,
        mesh_controls=controls,
    )

    manifest = build_mesh_manifest(
        groups=groups,
        mesh_name=mesh_path.name,
        route=route,
        route_a_thin_film=thin_film,
    )
    manifest["meshing"] = meshing
    manifest["curve_source"] = source_curve_provenance
    construction_plan = json.loads(
        (paths.semantic_dir / "02_build_route_construction_plan.json").read_text(
            encoding="utf-8"
        )
    )
    native_curves = construction_plan.get("metadata", {}).get("curved_arrangement")
    if native_curves is not None:
        manifest["curved_arrangement"] = copy.deepcopy(native_curves)
    elif build_input.boundary_curves or build_input.boundary_reconstruction:
        raise ValueError(
            "curved SGB construction lacks native curved-arrangement provenance"
        )
    manifest["source"]["gsim_high_order_meshing"] = GSIM_HIGH_ORDER_DERIVATION_SHA
    write_json(paths.manifest_path, manifest)
    write_json(
        paths.provenance_path,
        {
            "scgsim_version": scgsim_version(),
            "route": route,
            **({"route_a_thin_film": thin_film} if thin_film is not None else {}),
            "source_sha": {
                "gsim": GSIM_SHA,
                "scgsim_sgb": SGB_RUNTIME_AUTHORITY,
                "sgb_derivation": {
                    "base": SGB_DERIVATION_BASE_SHA,
                    "imported_development": SGB_DERIVATION_IMPORTED_SHA,
                },
            },
            "component_top_cell": top_cell,
            "run_dir": ".",
            "stack_path": str(paths.stack_path.relative_to(paths.run_dir)),
            "xao_path": str(paths.xao_path.relative_to(paths.run_dir)),
            "sidecar_path": str(paths.sidecar_path.relative_to(paths.run_dir)),
        },
    )

    return MeshBuildResult(
        output_dir=paths.run_dir,
        mesh_path=mesh_path,
        groups=groups,
        mesh_manifest_path=paths.manifest_path,
        provenance_path=paths.provenance_path,
        stack_path=paths.stack_path,
        xao_path=paths.xao_path,
        sidecar_path=paths.sidecar_path,
    )


@dataclass(frozen=True)
class _BuildPaths:
    """Paired source and result locations shared by one mesh preparation."""

    run_dir: Path
    geometry_dir: Path
    semantic_dir: Path
    gds_path: Path
    stack_path: Path
    xao_path: Path
    sidecar_path: Path
    mesh_path: Path
    manifest_path: Path
    provenance_path: Path


def _prepare_paths(output_dir: str | Path, route: str) -> _BuildPaths:
    run_dir = Path(output_dir).expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"output_dir must be empty before mesh: {run_dir}")
    geometry_dir = run_dir / "geometry"
    semantic_dir = run_dir / "metadata" / "semantic_geometry"

    for path in (
        run_dir,
        geometry_dir,
        semantic_dir,
        run_dir / "logs",
        run_dir / "results" / "palace",
    ):
        path.mkdir(parents=True, exist_ok=True)

    gds_path = geometry_dir / "design.gds"
    stack_path = geometry_dir / "design.stack.json"
    xao_path = geometry_dir / f"semantic_geometry_route_{route.lower()}.xao"
    sidecar_path = semantic_dir / "04_export_physical_groups.json"
    mesh_path = run_dir / "palace.msh"
    manifest_path = run_dir / "metadata" / "mesh_manifest.json"
    provenance_path = run_dir / "metadata" / f"route_{route.lower()}_provenance.json"

    return _BuildPaths(
        run_dir,
        geometry_dir,
        semantic_dir,
        gds_path,
        stack_path,
        xao_path,
        sidecar_path,
        mesh_path,
        manifest_path,
        provenance_path,
    )


def _write_sources(
    component: Any,
    stack: Mapping[str, Any],
    source_gds_bytes: bytes | None,
    paths: _BuildPaths,
) -> str:
    if source_gds_bytes is None:
        if component is None:
            raise ValueError("component or immutable plan GDS source is required")
        write_component_gds(component=component, path=paths.gds_path)
        top_cell = component_gds_top_cell_name(
            component=component, gds_path=paths.gds_path
        )
    else:
        if component is not None or not isinstance(source_gds_bytes, bytes):
            raise TypeError("plan GDS bytes cannot be mixed with a live component")
        wanted = stack.get("metadata", {}).get("gds_sha256")
        observed = hashlib.sha256(source_gds_bytes).hexdigest()
        if wanted != observed:
            raise ValueError("plan GDS bytes differ from paired stack source")
        top_cell = stack.get("metadata", {}).get("top_cell_name")
        if not isinstance(top_cell, str) or not top_cell:
            raise ValueError("plan stack requires top_cell_name")
        paths.gds_path.write_bytes(source_gds_bytes)
    paths.stack_path.write_text(
        json.dumps(dict(stack), indent=2) + "\n", encoding="utf-8"
    )

    return top_cell


def _write_optional_bump_receipt(
    indium_ground_bump_fill: Mapping[str, Any] | None, paths: _BuildPaths
) -> None:
    if indium_ground_bump_fill is not None:
        receipt = indium_ground_bump_fill.get("receipt")
        if not isinstance(receipt, Mapping):
            raise TypeError("indium ground bump fill requires a structured receipt.")
        write_indium_ground_bump_receipt(
            run_dir=paths.run_dir,
            gds_path=paths.gds_path,
            receipt=receipt,
        )
