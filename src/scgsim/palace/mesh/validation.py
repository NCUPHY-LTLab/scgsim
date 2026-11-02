"""Validate saved native connectivity and port-to-volume conformity."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def validate_tetrahedral_solution_groups(
    *,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    live: Mapping[tuple[int, str], tuple[int, tuple[int, ...]]],
) -> None:
    """Require exact live solver volume ownership and at least one 3D element each."""
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    expected = set(groups.get("volumes", {}))
    live_volume_names = {name for (dim, name) in live if dim == 3}
    if live_volume_names != expected:
        raise ValueError(
            "XAO live 3D physical groups differ from structured solution volumes: "
            f"live={sorted(live_volume_names)}, solution={sorted(expected)}."
        )
    for name, info in groups.get("volumes", {}).items():
        has_elements = False
        for tag in info["tags"]:
            element_types, element_tags, _ = gmsh.model.mesh.getElements(3, int(tag))
            for element_type in element_types:
                element_name, dimension, *_ = gmsh.model.mesh.getElementProperties(
                    element_type
                )
                if dimension != 3 or "tetrahedron" not in element_name.lower():
                    raise ValueError(
                        f"Solver solution volume {name!r} has non-tetrahedral 3D element {element_name!r}."
                    )
            has_elements = has_elements or any(len(items) for items in element_tags)
        if not has_elements:
            raise ValueError(
                f"Solver solution volume {name!r} has no 3D mesh elements."
            )


def validate_port_pec_node_intersections(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """Require each SGB port terminal edge to share actual nodes with its PEC owner."""
    for key, port in groups.get("port_surfaces", {}).items():
        port_edges = _surface_mesh_edges(port.get("tags", ()))
        if not port_edges:
            raise ValueError(f"SGB lumped-port sheet {key!r} has no mesh nodes.")
        for owner_id in port.get("owner_semantic_ids", ()):
            owner_tags = {
                int(tag)
                for info in groups.get("pec_surfaces", {}).values()
                if owner_id in info.get("owner_semantic_ids", ())
                for tag in info.get("tags", ())
            }
            if not owner_tags:
                raise ValueError(
                    f"SGB lumped-port sheet {key!r} owner {owner_id!r} has no PEC surface."
                )
            if not port_edges.intersection(_surface_mesh_edges(owner_tags)):
                raise ValueError(
                    f"SGB lumped-port sheet {key!r} owner {owner_id!r} does not share a mesh edge."
                )


def validate_port_tetrahedron_face_conformality(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """Require every full-node port triangle to be a two-sided tetrahedron face."""
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    port_triangles: dict[str, list[tuple[int, ...]]] = {}
    for key, port in groups.get("port_surfaces", {}).items():
        triangles: list[tuple[int, ...]] = []
        for surface_tag in port.get("tags", ()):
            element_types, _, element_nodes = gmsh.model.mesh.getElements(
                2, int(surface_tag)
            )
            for element_type, nodes in zip(element_types, element_nodes, strict=True):
                name, dimension, _, node_count, _, primary_count = (
                    gmsh.model.mesh.getElementProperties(element_type)
                )
                if (
                    dimension != 2
                    or "triangle" not in name.lower()
                    or primary_count != 3
                ):
                    raise ValueError(
                        f"SGB lumped-port sheet {key!r} has unsupported element {name!r}."
                    )
                triangles.extend(
                    tuple(
                        sorted(int(node) for node in nodes[index : index + node_count])
                    )
                    for index in range(0, len(nodes), node_count)
                )
        port_triangles[key] = triangles

    tetrahedron_faces: Counter[tuple[int, ...]] = Counter(
        {triangle: 0 for triangles in port_triangles.values() for triangle in triangles}
    )
    volume_tags = {
        int(tag)
        for volume in groups.get("volumes", {}).values()
        for tag in volume.get("tags", ())
    }
    for volume_tag in volume_tags:
        element_types, _, _ = gmsh.model.mesh.getElements(3, volume_tag)
        for element_type in element_types:
            name, dimension, order, *_ = gmsh.model.mesh.getElementProperties(
                element_type
            )
            if dimension != 3 or "tetrahedron" not in name.lower():
                continue
            face_nodes = gmsh.model.mesh.getElementFaceNodes(
                int(element_type), 3, volume_tag, primary=False
            )
            face_size = (order + 1) * (order + 2) // 2
            if len(face_nodes) % face_size:
                raise ValueError("Gmsh returned an incomplete tetrahedron face list.")
            for index in range(0, len(face_nodes), face_size):
                face = tuple(
                    sorted(int(node) for node in face_nodes[index : index + face_size])
                )
                if face in tetrahedron_faces:
                    tetrahedron_faces[face] += 1

    for key, triangles in port_triangles.items():
        multiplicities = Counter(tetrahedron_faces[triangle] for triangle in triangles)
        invalid = {
            count: amount for count, amount in multiplicities.items() if count != 2
        }
        if not triangles or invalid:
            raise ValueError(
                f"SGB lumped-port sheet {key!r} is non-conformal to the tetrahedron mesh: "
                f"{len(triangles)} triangle(s), invalid tetra-face multiplicities {invalid}; "
                "every internal port triangle must be shared by exactly two tetrahedra."
            )


def _surface_mesh_edges(tags: Iterable[Any]) -> set[tuple[int, ...]]:
    """Use complete native edge nodes, including elevated interpolation nodes."""
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    edges: set[tuple[int, ...]] = set()
    for tag in tags:
        element_types, _, _ = gmsh.model.mesh.getElements(2, int(tag))
        for code in element_types:
            _, _, order, *_ = gmsh.model.mesh.getElementProperties(code)
            nodes = gmsh.model.mesh.getElementEdgeNodes(
                int(code), int(tag), primary=False
            )
            size = order + 1
            edges.update(
                tuple(sorted(int(n) for n in nodes[i : i + size]))
                for i in range(0, len(nodes), size)
            )
    return edges
