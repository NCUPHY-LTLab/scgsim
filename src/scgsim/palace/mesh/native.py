"""Own one Gmsh session, mesh generation, validation, and saved-mesh readback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scgsim.diagnostics import diagnostic_stage

from .controls import MeshSizing
from .physical_groups import (
    augment_groups_from_live_records,
    build_groups_from_sgb_records,
    live_physical_groups,
    solution_material_kinds,
)
from .record_validation import sgb_records_route_context
from .refinement import apply_mesh_sizing, embed_route_b_port_surfaces
from .summary import observe_native_mesh
from .validation import (
    validate_port_pec_node_intersections,
    validate_port_tetrahedron_face_conformality,
    validate_tetrahedral_solution_groups,
)


@diagnostic_stage("palace.mesh.worker")
def mesh_xao_in_current_process(  # noqa: PLR0913 -- worker request preserves explicit source identities and controls
    *,
    xao_path: Path,
    route: str,
    records: Sequence[Any],
    stack: Mapping[str, Any],
    mesh_path: Path,
    refined_mesh_size: float | None,
    max_mesh_size: float | None,
    mesh_controls: Mapping[str, Any],
) -> dict[str, Any]:
    """Current-process implementation used only by the private fresh worker."""

    sizing = MeshSizing(
        mesh_controls.get("size_policy", "scgsim_default"),
        refined_mesh_size,
        max_mesh_size,
    )
    route = route.upper()
    sgb_records_route_context(records, route)
    groups = build_groups_from_sgb_records()
    material_kinds = solution_material_kinds(records, stack)

    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    gmsh.initialize(readConfigFiles=False)
    try:
        with diagnostic_stage("palace.mesh.open_xao"):
            gmsh.open(str(xao_path))
        gmsh.option.setNumber("General.Terminal", 0)

        route = sgb_records_route_context(records, route)
        live = live_physical_groups()
        augment_groups_from_live_records(
            records=records,
            groups=groups,
            live=live,
            route=route,
            solution_material_kinds=material_kinds,
        )
        if not groups["volumes"]:
            raise ValueError(
                "SGB XAO did not expose any solver-active solution-volume groups. "
                "Check that 04_export_physical_groups.json matches the XAO."
            )

        embed_route_b_port_surfaces(groups)
        live = live_physical_groups()
        apply_mesh_sizing(groups, sizing)
        optimization = _generate_mesh(mesh_controls)
        with diagnostic_stage("palace.mesh.validate"):
            validate_tetrahedral_solution_groups(groups=groups, live=live)
            validate_port_tetrahedron_face_conformality(groups)
            validate_port_pec_node_intersections(groups)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        with diagnostic_stage("palace.mesh.write_mesh"):
            gmsh.write(str(mesh_path))

        requested = dict(mesh_controls)
        requested.update(sizing.to_payload())
        effective = dict(requested)
        effective.update(
            {
                name: gmsh.option.getNumber(option)
                for name, option in {
                    "algorithm_3d_code": "Mesh.Algorithm3D",
                    "threads": "General.NumThreads",
                    "volume_threads": "Mesh.MaxNumThreads3D",
                    "surface_threads_1d": "Mesh.MaxNumThreads1D",
                    "surface_threads_2d": "Mesh.MaxNumThreads2D",
                    "automatic_high_order_optimize": "Mesh.HighOrderOptimize",
                    "refined_mesh_size": "Mesh.MeshSizeMin",
                    "max_mesh_size": "Mesh.MeshSizeMax",
                    "mesh_size_factor": "Mesh.MeshSizeFactor",
                    "mesh_size_from_points": "Mesh.MeshSizeFromPoints",
                    "mesh_size_from_curvature": "Mesh.MeshSizeFromCurvature",
                    "mesh_size_extend_from_boundary": "Mesh.MeshSizeExtendFromBoundary",
                }.items()
            }
        )
        # Read the written artifact, including only entities retained by SaveAll=0.
        gmsh.clear()
        gmsh.open(str(mesh_path))
        with diagnostic_stage("palace.mesh.saved_mesh_readback"):
            statistics = observe_native_mesh(groups)
        orders = sorted(
            {r["order"] for r in statistics["element_types"] if r["dimension"] == 3}
        )
        effective["geometry_order"] = orders[0] if len(orders) == 1 else None
        effective["geometry_orders_3d"] = orders
        return {
            "groups": groups,
            "meshing": {
                "gmsh_version": gmsh.__version__,
                "requested": requested,
                "effective": effective,
                "high_order_optimization": optimization,
                "statistics": statistics,
            },
        }
    finally:
        gmsh.clear()
        gmsh.finalize()


def _generate_mesh(mesh_controls: Mapping[str, Any]) -> dict[str, Any]:
    """Apply generation controls before elevating and optionally optimizing order."""
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    native_algorithm = {"Delaunay": 1, "HXT": 10}[mesh_controls["algorithm_3d"]]
    gmsh.option.setNumber("Mesh.Algorithm3D", native_algorithm)
    gmsh.option.setNumber("General.NumThreads", mesh_controls["threads"])
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", mesh_controls["threads"])
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", mesh_controls["surface_threads"])
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", mesh_controls["surface_threads"])
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
    with diagnostic_stage("palace.mesh.generate"):
        gmsh.model.mesh.generate(3)
    geometry_order = mesh_controls["geometry_order"]
    optimization = {"status": "not_applicable", "invocation_count": 0}
    if geometry_order > 1:
        with diagnostic_stage("palace.mesh.set_order"):
            gmsh.model.mesh.setOrder(geometry_order)
        optimization["status"] = "skipped"
        if mesh_controls["high_order_optimize"]:
            with diagnostic_stage("palace.mesh.optimize_high_order"):
                gmsh.model.mesh.optimize("HighOrder")
            optimization = {"status": "completed", "invocation_count": 1}
    return optimization
