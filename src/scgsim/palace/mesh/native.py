"""Own one Gmsh session that turns the meshing module's mesh into Palace's solver mesh.

The meshing module (``scgsim.meshing``) delivers an MSH 4.1 mesh whose physical groups are SGB's primary
groups.  Palace's config lists solver attributes instead, so this session opens that mesh, replaces the
primary groups by the attributes of the group dict, runs the structural checks, writes ``palace.msh``
(MSH 2.2 ASCII, the format Palace reads) and reads the written file back.
No CAD is needed: the module has already elevated the order, because that needs the CAD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scgsim.diagnostics import diagnostic_stage

from .physical_groups import Groups
from .summary import observe_native_mesh
from .validation import (
    live_physical_groups,
    validate_mesh_matches_groups,
    validate_port_pec_node_intersections,
    validate_port_tetrahedron_face_conformality,
    validate_tetrahedral_solution_groups,
)

# Palace reads MSH 2.2; Gmsh 4 writes it as ASCII with these options.
_PALACE_MSH_VERSION = 2.2


@diagnostic_stage("palace.mesh.worker")
def lower_mesh_in_current_process(
    *, source_mesh: Path, mesh_path: Path, groups: Groups
) -> dict[str, Any]:
    """Current-process implementation used only by the private fresh worker."""
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        with diagnostic_stage("palace.mesh.open_mesh"):
            gmsh.open(str(source_mesh))
        validate_mesh_matches_groups(groups, live_physical_groups())
        apply_attributes(groups)
        with diagnostic_stage("palace.mesh.validate"):
            validate_tetrahedral_solution_groups(
                groups=groups, live=live_physical_groups()
            )
            validate_port_tetrahedron_face_conformality(groups)
            validate_port_pec_node_intersections(groups)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", _PALACE_MSH_VERSION)
        with diagnostic_stage("palace.mesh.write_mesh"):
            gmsh.write(str(mesh_path))
        # Read the written artifact, including only entities retained by SaveAll=0.
        gmsh.clear()
        gmsh.open(str(mesh_path))
        with diagnostic_stage("palace.mesh.saved_mesh_readback"):
            statistics = observe_native_mesh(groups)
        return {"gmsh_version": gmsh.__version__, "statistics": statistics}
    finally:
        gmsh.clear()
        gmsh.finalize()


def apply_attributes(groups: Groups) -> None:
    """Replace the primary physical groups of the open model by the solver attributes of ``groups``.

    A volume group is one attribute under its own name.  A surface group is one attribute per signature of
    Surface-EPR rows (see :mod:`.attributes`), so the attributes partition the faces and no group overlaps
    another.  Entity tags are unchanged: the attribute only chooses which elements share a physical tag.
    """
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    # Gmsh keeps the name table when the groups are removed, and a kept name wins over the name a new
    # group is added with: once a split group shifts the numbering, every later tag would carry the name
    # of the primary group that had its number.  So the names go first.
    for dim, tag in gmsh.model.getPhysicalGroups():
        gmsh.model.removePhysicalName(gmsh.model.getPhysicalName(dim, tag))
    gmsh.model.removePhysicalGroups()
    for name, record in sorted(groups["volumes"].items()):
        (attribute,) = record["attributes"]
        gmsh.model.addPhysicalGroup(3, record["tags"], tag=attribute, name=name)
    for part in _surface_parts(groups):
        gmsh.model.addPhysicalGroup(
            2, part["tags"], tag=part["attribute"], name=part["name"]
        )


def _surface_parts(groups: Groups) -> list[dict[str, Any]]:
    """Every surface attribute of boundaries and ports, in attribute order."""
    parts = [
        part
        for section in ("boundary_surfaces", "port_surfaces")
        for record in groups[section].values()
        for part in record["parts"]
    ]
    return sorted(parts, key=lambda part: part["attribute"])
