"""Native mesh observations and detached summaries; no solver-DOF inference."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, TypedDict

from scgsim.contracts.palace_run_folder import (
    MESH,
    MESH_MANIFEST,
    palace_problem,
    sha256,
)
from scgsim.refusal import refusal


def _corner_chord_statistics(
    edges: set[tuple[int, ...]], points: Mapping[int, tuple[float, ...]]
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
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    points = {
        int(tag): tuple(float(x) for x in coordinates[3 * i : 3 * i + 3])
        for i, tag in enumerate(node_tags)
    }
    element_types, element_tags, _ = gmsh.model.mesh.getElements()
    types = _element_type_records(element_types, element_tags)
    regions = _observe_physical_regions(points)
    topology = _TetrahedralTopology()
    volumes = {int(t) for v in groups["volumes"].values() for t in v["tags"]}
    for volume in volumes:
        topology.add_volume(volume)
    return dict(
        node_count=len(node_tags),
        element_count=sum(len(t) for t in element_tags),
        element_types=types,
        physical_regions=regions,
        tetrahedral_topology=topology.counts(),
        corner_chord_edge_lengths=_corner_chord_statistics(topology.edges, points),
    )


def _element_type_records(
    element_types: Any, element_tags: Any
) -> list[dict[str, Any]]:
    """One record per Gmsh element type in the mesh: its properties and how many elements it has."""
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

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


@dataclass
class _TetrahedralTopology:
    """The corner vertices, edges, faces and tetrahedra of the solution volumes, gathered volume by volume.

    ``shared`` keeps the high-order nodes of each corner face, so a face two tetrahedra share must carry the
    same nodes on both sides.
    """

    vertices: set[int] = field(default_factory=set)
    edges: set[tuple[int, ...]] = field(default_factory=set)
    faces: set[tuple[int, ...]] = field(default_factory=set)
    tetrahedra: set[int] = field(default_factory=set)
    shared: dict[tuple[int, ...], tuple[int, ...]] = field(default_factory=dict)

    def add_volume(self, volume: int) -> None:
        """Add the tetrahedra of one volume and check its faces against those already seen."""
        import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

        ts, ids, connectivity = gmsh.model.mesh.getElements(3, volume)
        for code, tags, nodes in zip(ts, ids, connectivity, strict=True):
            name, _dim, order, count, _, primary = gmsh.model.mesh.getElementProperties(
                code
            )
            if "tetrahedron" not in name.lower():
                continue
            self._add_corners(tags, nodes, count, primary)
            self._check_shared_faces(int(code), volume, order)

    def _add_corners(self, tags: Any, nodes: Any, count: int, primary: int) -> None:
        for i, tag in enumerate(tags):
            if int(tag) in self.tetrahedra:
                continue
            self.tetrahedra.add(int(tag))
            corners = tuple(int(n) for n in nodes[i * count : i * count + primary])
            self.vertices.update(corners)
            self.edges.update(
                tuple(sorted(e)) for e in itertools.combinations(corners, 2)
            )
            self.faces.update(
                tuple(sorted(f)) for f in itertools.combinations(corners, 3)
            )

    def _check_shared_faces(self, code: int, volume: int, order: int) -> None:
        import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

        full = gmsh.model.mesh.getElementFaceNodes(code, 3, volume, primary=False)
        corners = gmsh.model.mesh.getElementFaceNodes(code, 3, volume, primary=True)
        face_size = (order + 1) * (order + 2) // 2
        for i in range(len(corners) // 3):
            key = tuple(sorted(int(n) for n in corners[3 * i : 3 * i + 3]))
            value = tuple(
                sorted(int(n) for n in full[face_size * i : face_size * (i + 1)])
            )
            if key in self.shared and self.shared[key] != value:
                raise ValueError(
                    refusal(
                        "shared tetrahedral face has different high-order nodes",
                        found={"face": key, "nodes": value},
                        expected=self.shared[key],
                    )
                )
            self.shared[key] = value

    def counts(self) -> dict[str, int]:
        """The number of each kind of corner element, as the mesh manifest records them."""
        return dict(
            vertices=len(self.vertices),
            edges=len(self.edges),
            faces=len(self.faces),
            tetrahedra=len(self.tetrahedra),
        )


class DofEstimate(TypedDict, total=False):
    """The topological estimate of the solver's degrees of freedom, or why there is none."""

    status: Literal["available", "unavailable"]
    actual_solver_dof: bool
    reason: str
    space: str
    value: int
    basis: str


class MeshSummary(TypedDict):
    """What a person reads about a mesh: the manifest's meshing record, the hashes and a DOF estimate."""

    status: Literal["recorded", "not_recorded"]
    mesh_sha256: str | None
    manifest_sha256: str | None
    meshing: dict[str, Any] | None
    finite_element_order: int | None
    dof_estimate: DofEstimate


def mesh_summary(
    manifest: Mapping[str, Any],
    *,
    problem: str,
    fem_order: int | None,
    mesh_sha256: str | None = None,
    manifest_sha256: str | None = None,
) -> MeshSummary:
    """Detach recorded observations and a separately labeled FEM topology estimate."""
    observed = manifest.get("meshing")
    fem_space = palace_problem(problem).fem_space
    result = MeshSummary(
        status="recorded" if observed is not None else "not_recorded",
        mesh_sha256=mesh_sha256,
        manifest_sha256=manifest_sha256,
        meshing=copy.deepcopy(observed),
        finite_element_order=fem_order,
        dof_estimate=DofEstimate(
            status="unavailable",
            actual_solver_dof=False,
            reason="H1 estimate not implemented"
            if fem_space == "H1"
            else "tetrahedral topology not recorded",
        ),
    )
    if fem_space == "ND" and observed is not None and fem_order is not None:
        topology = observed["statistics"]["tetrahedral_topology"]
        p = fem_order
        estimate = (
            p * topology["edges"]
            + p * (p - 1) * topology["faces"]
            + p * (p - 1) * (p - 2) * topology["tetrahedra"] // 2
        )
        result["dof_estimate"] = DofEstimate(
            status="available",
            space="ND",
            value=estimate,
            basis="tetrahedral primary-corner topology; before boundary constraints",
            actual_solver_dof=False,
        )
    return result


def recorded_mesh_summary(
    mesh_manifest: Mapping[str, Any] | None,
    *,
    handoff: Mapping[str, Any],
    config: Mapping[str, Any] | None,
    problem: str,
) -> MeshSummary:
    """The mesh summary of a returned run folder, from its manifest, the hashes the handoff sealed and the config.

    A run that stopped early may lack the manifest or the config; the summary then says what was not recorded.
    """
    hashes = {entry["path"]: entry["sha256"] for entry in (handoff.get("hashes") or ())}
    solver = config.get("Solver", {}) if config is not None else {}
    fem_order = solver.get("Order")
    summary = mesh_summary(
        mesh_manifest if mesh_manifest is not None else {},
        problem=problem,
        fem_order=fem_order,
        mesh_sha256=hashes.get(MESH),
        manifest_sha256=hashes.get(MESH_MANIFEST),
    )
    if fem_order is None and palace_problem(problem).fem_space == "ND":
        summary["dof_estimate"]["reason"] = "FEM order not recorded"
    return summary


def summary_from_files(
    mesh_path: Path, manifest_path: Path, *, problem: str, fem_order: int
) -> MeshSummary:
    """Detach current facade artifacts without changing solver or mesh state."""
    payload = manifest_path.read_bytes()
    return mesh_summary(
        json.loads(payload),
        problem=problem,
        fem_order=fem_order,
        mesh_sha256=sha256(mesh_path),
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _observe_physical_regions(
    points: Mapping[int, tuple[float, ...]],
) -> list[dict[str, Any]]:
    import gmsh  # noqa: PLC0415 -- Gmsh is optional and loaded only on native paths.

    regions = []
    for dim, attribute in gmsh.model.getPhysicalGroups():
        codes: dict[int, set[int]] = {}
        region_edges: set[tuple[int, ...]] = set()
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, attribute):
            ts, ids, connectivity = gmsh.model.mesh.getElements(dim, int(entity))
            for code, tags, nodes in zip(ts, ids, connectivity, strict=True):
                codes.setdefault(int(code), set()).update(int(t) for t in tags)
                if dim == 1:
                    _, _, _, count, _, _ = gmsh.model.mesh.getElementProperties(code)
                    endpoints = [
                        int(n)
                        for i in range(len(tags))
                        for n in nodes[i * count : i * count + 2]
                    ]
                elif dim > 1:
                    # Native edge topology avoids treating quad/hex diagonals as edges.
                    endpoints = gmsh.model.mesh.getElementEdgeNodes(
                        int(code), int(entity), primary=True
                    )
                else:
                    endpoints = ()
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
