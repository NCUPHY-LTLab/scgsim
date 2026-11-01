"""Native mesh observations and detached summaries; no solver-DOF inference.

Owns what is read back from the final Gmsh mesh (node and element statistics,
physical-region chord lengths, tetrahedral topology) and the detached
``mesh_summary`` built from the recorded manifest. DOF means degrees of freedom;
this module only estimates edge-element counts from topology and never reports a
solver's actual count. It does not generate or validate meshes (``lowering.py``).
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _corner_chord_statistics(
    edges: set[tuple[int, int]], points: Mapping[int, tuple[float, ...]]
) -> dict[str, Any]:
    """Measure unique primary-edge chords in Palace's micrometre coordinates."""
    lengths = [math.dist(points[a], points[b]) for a, b in edges]
    return dict(
        unit="um",
        interpretation="primary-corner chords, not curved edge lengths",
        count=len(lengths),
        minimum=min(lengths) if lengths else None,
        maximum=max(lengths) if lengths else None,
        mean=sum(lengths) / len(lengths) if lengths else None,
    )


def observe_native_mesh(groups: Mapping[str, Any]) -> dict[str, Any]:
    """Read final Gmsh connectivity before its native session is finalized."""
    import gmsh

    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    points = {
        int(tag): tuple(float(x) for x in coordinates[3 * i : 3 * i + 3])
        for i, tag in enumerate(node_tags)
    }
    element_types, element_tags, _ = gmsh.model.mesh.getElements()
    types = _element_type_rows(element_types, element_tags)
    regions = _physical_region_rows(points)
    topology = _tetrahedral_topology(groups)
    return dict(
        node_count=len(node_tags),
        element_count=sum(len(t) for t in element_tags),
        element_types=types,
        physical_regions=regions,
        tetrahedral_topology=dict(
            vertices=len(topology.vertices),
            edges=len(topology.edges),
            faces=len(topology.faces),
            tetrahedra=len(topology.tetrahedra),
        ),
        corner_chord_edge_lengths=_corner_chord_statistics(topology.edges, points),
    )


def _element_type_rows(element_types: Any, element_tags: Any) -> list[dict[str, Any]]:
    """Describe each Gmsh element type present in the mesh and how many there are."""
    import gmsh

    types = []
    for code, tags in zip(element_types, element_tags, strict=True):
        name, dimension, order, nodes, reference, primary = (
            gmsh.model.mesh.getElementProperties(code)
        )
        types.append(
            dict(
                code=int(code),
                name=name,
                dimension=int(dimension),
                order=int(order),
                nodes_per_element=int(nodes),
                primary_nodes=int(primary),
                count=len(tags),
                reference_coordinates=[float(x) for x in reference],
            )
        )
    return types


def _physical_region_rows(
    points: Mapping[int, tuple[float, ...]],
) -> list[dict[str, Any]]:
    """Describe every physical group: its elements and its corner-chord lengths."""
    import gmsh

    regions = []
    for dim, attribute in gmsh.model.getPhysicalGroups():
        codes: dict[int, set[int]] = {}
        region_edges: set[tuple[int, int]] = set()
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, attribute):
            ts, ids, connectivity = gmsh.model.mesh.getElements(dim, int(entity))
            for code, tags, nodes in zip(ts, ids, connectivity, strict=True):
                codes.setdefault(int(code), set()).update(int(t) for t in tags)
                endpoints = _element_edge_endpoints(dim, code, int(entity), tags, nodes)
                region_edges.update(
                    tuple(sorted((int(endpoints[i]), int(endpoints[i + 1]))))
                    for i in range(0, len(endpoints), 2)
                )
        regions.append(
            dict(
                dimension=int(dim),
                attribute=int(attribute),
                name=gmsh.model.getPhysicalName(dim, attribute),
                element_count=sum(len(v) for v in codes.values()),
                element_types={str(k): len(v) for k, v in codes.items()},
                corner_chord_edge_lengths=_corner_chord_statistics(
                    region_edges, points
                ),
            )
        )
    return regions


def _element_edge_endpoints(
    dim: int, code: Any, entity: int, tags: Any, nodes: Any
) -> Any:
    """Return the flat endpoint node list of one element type's primary edges."""
    import gmsh

    if dim == 1:
        _, _, _, count, _, _ = gmsh.model.mesh.getElementProperties(code)
        return [
            int(n) for i in range(len(tags)) for n in nodes[i * count : i * count + 2]
        ]
    if dim > 1:
        # Native edge topology avoids treating quad/hex diagonals as edges.
        return gmsh.model.mesh.getElementEdgeNodes(int(code), entity, primary=True)
    return ()


@dataclass
class _TetrahedralTopology:
    """Unique corner vertices, edges, faces and tetrahedra of the solution volumes."""

    edges: set[tuple[int, int]] = field(default_factory=set)
    faces: set[tuple[int, int, int]] = field(default_factory=set)
    tetrahedra: set[int] = field(default_factory=set)
    vertices: set[int] = field(default_factory=set)


def _tetrahedral_topology(groups: Mapping[str, Any]) -> _TetrahedralTopology:
    """Count unique primary-corner topology of all solution-volume tetrahedra.

    Also requires that two tetrahedra sharing a face agree on its high-order nodes.
    """
    import gmsh

    topology = _TetrahedralTopology()
    tetrahedra = topology.tetrahedra
    vertices = topology.vertices
    edges = topology.edges
    faces = topology.faces
    shared: dict[tuple[int, ...], tuple[int, ...]] = {}
    volumes = {int(t) for v in groups["volumes"].values() for t in v["tags"]}
    for volume in volumes:
        ts, ids, connectivity = gmsh.model.mesh.getElements(3, volume)
        for code, tags, nodes in zip(ts, ids, connectivity, strict=True):
            name, dim, order, count, _, primary = gmsh.model.mesh.getElementProperties(
                code
            )
            if "tetrahedron" not in name.lower():
                continue
            for i, tag in enumerate(tags):
                if int(tag) in tetrahedra:
                    continue
                tetrahedra.add(int(tag))
                corners = tuple(int(n) for n in nodes[i * count : i * count + primary])
                vertices.update(corners)
                edges.update(
                    tuple(sorted(e)) for e in itertools.combinations(corners, 2)
                )
                faces.update(
                    tuple(sorted(f)) for f in itertools.combinations(corners, 3)
                )
            full = gmsh.model.mesh.getElementFaceNodes(
                int(code), 3, volume, primary=False
            )
            corners = gmsh.model.mesh.getElementFaceNodes(
                int(code), 3, volume, primary=True
            )
            face_size = (order + 1) * (order + 2) // 2
            for i in range(len(corners) // 3):
                key = tuple(sorted(int(n) for n in corners[3 * i : 3 * i + 3]))
                value = tuple(
                    sorted(int(n) for n in full[face_size * i : face_size * (i + 1)])
                )
                if key in shared and shared[key] != value:
                    raise ValueError(
                        "shared tetrahedral face has different high-order nodes"
                    )
                shared[key] = value
    return topology


def mesh_summary(
    manifest: Mapping[str, Any],
    *,
    problem: str,
    fem_order: int | None,
    mesh_sha256: str | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Detach recorded observations and a separately labeled FEM topology estimate."""
    observed = manifest.get("meshing")
    result: dict[str, Any] = dict(
        status="recorded" if observed is not None else "not_recorded",
        mesh_sha256=mesh_sha256,
        manifest_sha256=manifest_sha256,
        meshing=copy.deepcopy(observed),
        curve_source=copy.deepcopy(manifest.get("curve_source")),
        curved_arrangement=copy.deepcopy(manifest.get("curved_arrangement")),
        finite_element_order=fem_order,
        dof_estimate=dict(
            status="unavailable",
            actual_solver_dof=False,
            reason="H1 estimate not implemented"
            if problem == "Electrostatic"
            else "tetrahedral topology not recorded",
        ),
    )
    if problem == "Eigenmode" and observed is not None and fem_order is not None:
        topology = observed["statistics"]["tetrahedral_topology"]
        p = fem_order
        estimate = (
            p * topology["edges"]
            + p * (p - 1) * topology["faces"]
            + p * (p - 1) * (p - 2) * topology["tetrahedra"] // 2
        )
        result["dof_estimate"] = dict(
            status="available",
            space="ND",
            value=estimate,
            basis="tetrahedral primary-corner topology; before boundary constraints",
            actual_solver_dof=False,
        )
    return result


def summary_from_files(
    mesh_path: Path, manifest_path: Path, *, problem: str, fem_order: int
) -> dict[str, Any]:
    """Detach current facade artifacts without changing solver or mesh state."""
    payload = manifest_path.read_bytes()
    return mesh_summary(
        json.loads(payload),
        problem=problem,
        fem_order=fem_order,
        mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )
