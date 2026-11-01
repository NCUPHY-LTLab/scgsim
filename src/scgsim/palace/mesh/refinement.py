"""Gmsh size fields and Route-B port embedding applied before meshing.

Owns the two Gmsh model edits that happen between mapping the groups and calling
``generate``: the background size field that refines the mesh along surface edges,
and embedding Route-B lumped-port faces into their host volumes. It does not map
records to groups (``groups.py``) or check the finished mesh (``checks_gmsh.py``).

Both functions mutate the open Gmsh model, so they must run in the order
``lowering.py`` calls them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .record_checks import is_nonempty_string_sequence, required_string

# Number of sample points Gmsh's Distance field places on each refined curve.
_DISTANCE_SAMPLING = 200


def setup_xao_refinement(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    refined_cellsize: float,
    max_cellsize: float,
) -> None:
    """Refine the mesh toward the edges of every boundary, PEC and port surface."""
    import gmsh

    lines: set[int] = set()
    for surface_group in ("boundary_surfaces", "pec_surfaces", "port_surfaces"):
        for info in groups.get(surface_group, {}).values():
            for tag in info.get("tags", ()):  # pragma: no branch
                try:
                    boundary = gmsh.model.getBoundary(
                        [(2, int(tag))],
                        combined=False,
                        oriented=False,
                        recursive=False,
                    )
                except RuntimeError:
                    continue
                lines.update(int(btag) for bdim, btag in boundary if bdim == 1)
    if not lines:
        return

    field_id = _setup_mesh_refinement(
        sorted(lines),
        refined_cellsize,
        max_cellsize,
    )
    _finalize_mesh_fields([field_id])


def embed_route_b_port_surfaces(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """Embed each Route-B port face in its host volumes, then drop the transport groups.

    Embedding makes the volume mesh conform to the port face, so the face becomes
    triangles shared by two tetrahedra instead of a free sheet.
    """
    import gmsh

    hosts, faces, transport_groups = _collect_transport_entities()
    for key, port in groups.get("port_surfaces", {}).items():
        if port.get("route") != "B":
            continue
        _embed_port_faces(key, port, hosts, faces)

    # Transport groups are XAO-only identities, never solver material attributes.
    if transport_groups:
        gmsh.model.removePhysicalGroups(transport_groups)


def _collect_transport_entities() -> tuple[
    dict[str, int], dict[str, int], list[tuple[int, int]]
]:
    """Read the compiler's ``SGB_VOLUME::`` / ``SGB_PORT_SURFACE::`` identity groups.

    Returns (host volume tag by plan id, port face tag by surface id, the physical
    groups that carried them). Each identity must bind exactly one entity.
    """
    import gmsh

    hosts: dict[str, int] = {}
    faces: dict[str, int] = {}
    transport_groups: list[tuple[int, int]] = []
    families = (("SGB_VOLUME::", 3, hosts), ("SGB_PORT_SURFACE::", 2, faces))
    for dim, physical_tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, physical_tag)
        for prefix, expected_dim, identities in families:
            if not name.startswith(prefix):
                continue
            plan_id = required_string(name[len(prefix) :], "SGB transport identity")
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, physical_tag)
            if dim != expected_dim or len(entity_tags) != 1 or plan_id in identities:
                raise ValueError(
                    f"SGB transport {name!r} must uniquely bind one dimension-{expected_dim} entity."
                )
            identities[plan_id] = int(entity_tags[0])
            transport_groups.append((int(dim), int(physical_tag)))
            break
    return hosts, faces, transport_groups


def _embed_port_faces(
    key: str,
    port: Mapping[str, Any],
    hosts: Mapping[str, int],
    faces: Mapping[str, int],
) -> None:
    """Embed every face of one port in the host volumes the compiler named for it."""
    import gmsh

    attribute = port.get("physical_attribute", {})
    bindings = attribute.get("embedded_volume_plan_ids_by_surface")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError(
            f"{key} requires compiler embedded_volume_plan_ids_by_surface."
        )
    surface_tags = {int(tag) for tag in port.get("tags", ())}
    bound_faces: set[int] = set()
    for surface_id, plan_ids in bindings.items():
        surface_id = required_string(surface_id, f"{key} compiler surface identity")
        if surface_id not in faces:
            raise ValueError(
                f"SGB Route-B port {key!r} is missing surface transport {surface_id!r}."
            )
        face_tag = faces[surface_id]
        if face_tag not in surface_tags:
            raise ValueError(f"SGB port surface {surface_id!r} is outside {key!r}.")
        if not is_nonempty_string_sequence(plan_ids):
            raise ValueError(
                f"{key} surface {surface_id!r} requires compiler host identities."
            )
        bound_faces.add(face_tag)
        for plan_id in plan_ids:
            if plan_id not in hosts:
                raise ValueError(
                    f"SGB Route-B port {key!r} is missing volume transport {plan_id!r}."
                )
            gmsh.model.mesh.embed(2, [face_tag], 3, hosts[plan_id])
    if bound_faces != surface_tags:
        raise ValueError(f"SGB Route-B port {key!r} has unbound live surface pieces.")


def _setup_mesh_refinement(
    boundary_line_tags: list[int],
    refined_cellsize: float,
    max_cellsize: float,
) -> int:
    """Add a Distance + Threshold size field around the boundary lines.

    Cells are ``refined_cellsize`` on the lines and grow to ``max_cellsize`` at
    ``max_cellsize`` distance. Returns the id of the Threshold field.
    """
    import gmsh

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", boundary_line_tags)
    gmsh.model.mesh.field.setNumber(1, "Sampling", _DISTANCE_SAMPLING)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", refined_cellsize)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", max_cellsize)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", max_cellsize)
    return 2


def _finalize_mesh_fields(field_ids: list[int]) -> None:
    """Take the minimum of the size fields as the only size source.

    Sizes from points, curvature and boundary extension are switched off so the
    background field alone sets the cell size; the 2D algorithm 5 is Delaunay.
    """
    import gmsh

    min_field_id = max(field_ids) + 1
    gmsh.model.mesh.field.add("Min", min_field_id)
    gmsh.model.mesh.field.setNumbers(min_field_id, "FieldsList", field_ids)
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field_id)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
