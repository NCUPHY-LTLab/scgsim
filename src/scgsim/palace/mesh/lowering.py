"""Lower an SGB XAO to a Palace ``.msh`` inside one Gmsh session.

This is the Gmsh-facing kernel. It runs in the fresh child process started by
``worker_process.py`` (never in the parent), opens the XAO that SGB wrote, maps its
physical groups to the structured records, meshes, validates and writes the MSH.

It owns the order of Gmsh calls. Gmsh numbers entities and physical groups in call
order, and the written ``palace.msh`` must stay byte-identical across refactors,
so steps here run in exactly this sequence and are not reordered, merged or
skipped. Record mapping is in ``groups.py``, model edits in ``refinement.py`` and
mesh checks in ``checks_gmsh.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checks_gmsh import (
    validate_port_pec_node_intersections,
    validate_port_tetrahedron_face_conformality,
    validate_tetrahedral_solution_groups,
)
from .groups import (
    LiveGroups,
    LivePhysicalGroup,
    MeshGroups,
    augment_groups_from_live_records,
    empty_groups,
)
from .record_checks import sgb_records_route_context, solution_material_kinds
from .refinement import embed_route_b_port_surfaces, setup_xao_refinement
from .summary import observe_native_mesh

# Gmsh's numeric codes for Mesh.Algorithm3D.
_ALGORITHM_3D_CODES = {"Delaunay": 1, "HXT": 10}


@dataclass(frozen=True)
class XaoMeshRequest:
    """Everything the Gmsh child needs to mesh one XAO."""

    xao_path: Path
    route: str
    records: Sequence[Any]
    stack: Mapping[str, Any]
    mesh_path: Path
    refined_mesh_size: float
    max_mesh_size: float
    mesh_controls: Mapping[str, Any]

    def to_payload(self, *, run_dir: Path) -> dict[str, Any]:
        """Return the JSON request; paths are relative to ``run_dir``, the child's cwd."""
        return {
            "xao_path": self.xao_path.relative_to(run_dir).as_posix(),
            "route": self.route,
            "records": list(self.records),
            "stack": dict(self.stack),
            "mesh_path": self.mesh_path.relative_to(run_dir).as_posix(),
            "refined_mesh_size": self.refined_mesh_size,
            "max_mesh_size": self.max_mesh_size,
            "mesh_controls": dict(self.mesh_controls),
        }


def mesh_xao_in_process(request: XaoMeshRequest) -> dict[str, Any]:
    """Mesh the XAO in this process and return the groups and meshing observations."""
    route = request.route.upper()
    sgb_records_route_context(request.records, route)
    groups = empty_groups()
    material_kinds = solution_material_kinds(request.records, request.stack)

    request.mesh_path.parent.mkdir(parents=True, exist_ok=True)
    import gmsh

    gmsh.initialize(readConfigFiles=False)
    try:
        live = _open_xao_and_map_groups(request, route, groups, material_kinds)
        optimization = _generate_mesh(request, groups)
        _validate_and_write_mesh(request.mesh_path, groups, live)
        requested, effective = _meshing_settings(request)
        # Read the written artifact, including only entities retained by SaveAll=0.
        gmsh.clear()
        gmsh.open(str(request.mesh_path))
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


def _open_xao_and_map_groups(
    request: XaoMeshRequest,
    route: str,
    groups: MeshGroups,
    material_kinds: Mapping[str, str],
) -> LiveGroups:
    """Open the XAO, join its physical groups to the records and embed Route-B ports.

    Returns the live groups as they stand after embedding, which removes the
    XAO-only transport groups; later checks must see that final state.
    """
    import gmsh

    gmsh.open(str(request.xao_path))
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", request.refined_mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", request.max_mesh_size)

    route = sgb_records_route_context(request.records, route)
    live = live_physical_groups()
    augment_groups_from_live_records(
        records=request.records,
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
    return live_physical_groups()


def _generate_mesh(request: XaoMeshRequest, groups: MeshGroups) -> dict[str, Any]:
    """Set the size field and native options, generate the 3D mesh, raise its order.

    Returns the high-order optimization status for the manifest.
    """
    import gmsh

    controls = request.mesh_controls
    setup_xao_refinement(groups, request.refined_mesh_size, request.max_mesh_size)
    gmsh.option.setNumber(
        "Mesh.Algorithm3D", _ALGORITHM_3D_CODES[controls["algorithm_3d"]]
    )
    gmsh.option.setNumber("General.NumThreads", controls["threads"])
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", controls["threads"])
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", controls["surface_threads"])
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", controls["surface_threads"])
    # Generate a linear mesh first; the geometry order is raised and optimized
    # explicitly below, so Gmsh's own automatic optimization stays off.
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
    gmsh.model.mesh.generate(3)
    geometry_order = controls["geometry_order"]
    optimization = {"status": "not_applicable", "invocation_count": 0}
    if geometry_order > 1:
        gmsh.model.mesh.setOrder(geometry_order)
        optimization["status"] = "skipped"
        if controls["high_order_optimize"]:
            gmsh.model.mesh.optimize("HighOrder")
            optimization = {"status": "completed", "invocation_count": 1}
    return optimization


def _validate_and_write_mesh(
    mesh_path: Path, groups: MeshGroups, live: LiveGroups
) -> None:
    """Check the finished mesh, then write it as ASCII MSH 2.2 with only physical entities."""
    import gmsh

    validate_tetrahedral_solution_groups(groups=groups, live=live)
    validate_port_tetrahedron_face_conformality(groups)
    validate_port_pec_node_intersections(groups)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(mesh_path))


def _meshing_settings(request: XaoMeshRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (requested, effective) settings; effective is read back from Gmsh."""
    import gmsh

    requested = dict(request.mesh_controls)
    requested.update(
        refined_mesh_size=request.refined_mesh_size,
        max_mesh_size=request.max_mesh_size,
    )
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
            }.items()
        }
    )
    return requested, effective


def live_physical_groups() -> dict[tuple[int, str], LivePhysicalGroup]:
    """Map live gmsh (dimension, physical name) -> (physical-group, tags)."""
    import gmsh

    live: dict[tuple[int, str], LivePhysicalGroup] = {}
    for dim, phys_group in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, phys_group)
        if not name:
            continue
        live[(int(dim), str(name))] = LivePhysicalGroup(
            int(phys_group),
            tuple(
                int(tag)
                for tag in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_group)
            ),
        )
    return live
