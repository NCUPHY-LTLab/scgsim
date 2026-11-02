"""Apply declared sizing and port embedding to the current native Gmsh session."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from .controls import MeshSizing
from .record_validation import is_nonempty_string_sequence, required_string


def apply_mesh_sizing(groups: Mapping[str, Any], sizing: MeshSizing) -> None:
    """Apply SCG sizing only when selected; native sizing is left untouched."""
    if sizing.size_policy == "gmsh_default":
        return
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    gmsh.option.setNumber("Mesh.MeshSizeMin", sizing.refined_mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", sizing.max_mesh_size)
    _setup_xao_refinement(
        groups, cast(float, sizing.refined_mesh_size), cast(float, sizing.max_mesh_size)
    )


def _setup_xao_refinement(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    refined_cellsize: float,
    max_cellsize: float,
) -> None:
    """Apply line-based mesh refinement from Route-B structured boundary lines."""
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

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
    """Consume compiler geometric-host transport independently of material groups."""
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

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

    for key, port in groups.get("port_surfaces", {}).items():
        if port.get("route") != "B":
            continue
        _embed_port(key, port, hosts, faces)

    # Transport groups are XAO-only identities, never solver material attributes.
    if transport_groups:
        gmsh.model.removePhysicalGroups(transport_groups)


def _setup_mesh_refinement(  # noqa: PLR0913 -- native field controls mirror the established independent Gmsh parameters
    boundary_line_tags: list[int],
    refined_cellsize: float,
    max_cellsize: float,
    *,
    sampling: int = 200,
    dist_min: float = 0.0,
    dist_max: float | None = None,
) -> int:
    """Set up gmsh size-transition field around boundary lines."""
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", boundary_line_tags)
    gmsh.model.mesh.field.setNumber(1, "Sampling", int(sampling))
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", refined_cellsize)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", max_cellsize)
    gmsh.model.mesh.field.setNumber(2, "DistMin", dist_min)
    gmsh.model.mesh.field.setNumber(
        2,
        "DistMax",
        max_cellsize if dist_max is None else float(dist_max),
    )
    return 2


def _finalize_mesh_fields(field_ids: list[int]) -> None:
    """Finalize gmsh background field setup from one or more fields."""
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    min_field_id = max(field_ids) + 1
    gmsh.model.mesh.field.add("Min", min_field_id)
    gmsh.model.mesh.field.setNumbers(min_field_id, "FieldsList", field_ids)
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field_id)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)


def _embed_port(
    key: str,
    port: Mapping[str, Any],
    hosts: Mapping[str, int],
    faces: Mapping[str, int],
) -> None:
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

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
