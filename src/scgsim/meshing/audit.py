"""Quality audit of the mesh in the current Gmsh session, with attribution of bad elements to SGB entities.

Owns the measurement and the mapping from a bad tetrahedron back to the model: every mesh node sits on
exactly one geometric entity (its classification), the entity's native tag is the sidecar binding's
``1 + position``, and the tag gives the planned id, the physical groups of the faces around it and the
kinds of the sidecar features that cover its curves.  Nothing is guessed from coordinates.

This module imports Gmsh at load time, so only the worker process (``worker``) imports it.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence

import gmsh
import numpy as np

from scgsim.refusal import found_text, internal

from .features import SidecarFeatures
from .findings import QUALITY_MEASURE, BadElement, CurveCandidate, MeshAudit

_TETRAHEDRON = 4
_LINE = 1
_DIMENSIONS = (0, 1, 2, 3)


def audit_current_mesh(
    features: SidecarFeatures,
    threshold: float,
    *,
    attribute: bool,
    mesh_seconds: float,
) -> MeshAudit:
    """Measure every tetrahedron of the generated mesh and describe those below ``threshold``.

    ``attribute=False`` only counts them (references that are not repaired need no attribution).
    """
    started = time.perf_counter()
    element_tags, element_nodes = _tetrahedra()
    quality = _qualities(element_tags, QUALITY_MEASURE)
    gamma = _qualities(element_tags, "gamma")
    bad_rows = np.flatnonzero(quality < threshold)
    bad: tuple[BadElement, ...] = ()
    if attribute and len(bad_rows):
        bad = Attribution(features).describe(
            element_tags[bad_rows], element_nodes[bad_rows], quality[bad_rows]
        )
    return MeshAudit(
        tetrahedra=len(element_tags),
        nodes=len(gmsh.model.mesh.getNodes()[0]),
        min_quality=float(quality.min()),
        min_gamma=float(gamma.min()),
        bad_count=int(len(bad_rows)),
        bad=bad,
        mesh_seconds=mesh_seconds,
        audit_seconds=time.perf_counter() - started,
    )


def _tetrahedra() -> tuple[np.ndarray, np.ndarray]:
    """Element tags and the four node tags of each, over every volume."""
    types, tags, nodes = gmsh.model.mesh.getElements(3)
    found = [
        (np.asarray(t, dtype=np.int64), np.asarray(n, dtype=np.int64).reshape(-1, 4))
        for kind, t, n in zip(types, tags, nodes, strict=True)
        if kind == _TETRAHEDRON
    ]
    if not found:
        raise RuntimeError(
            internal(
                "the generated mesh has no tetrahedra" + ": " + found_text(len(found))
            )
        )
    return np.concatenate([t for t, _ in found]), np.concatenate([n for _, n in found])


def _qualities(element_tags: np.ndarray, measure: str) -> np.ndarray:
    return np.asarray(
        gmsh.model.mesh.getElementQualities(element_tags.tolist(), measure)
    )


class Attribution:
    """Node classification and entity adjacency of the mesh, used to describe bad elements.

    The tables are built once per audit because every bad element reads them.
    """

    def __init__(self, features: SidecarFeatures) -> None:
        self._binding = features.binding
        self._groups_of_face = _groups_of_faces(features)
        self._kinds_of_curve = {
            tag: sorted({f.kind for f in covering})
            for tag, covering in features.features_by_curve().items()
        }
        self._curve_faces, self._point_curves = _adjacency()
        self._node_dimension, self._node_entity = _classify_nodes()
        self._node_xyz = _node_coordinates()
        self._curve_edges: dict[int, np.ndarray] = {}
        self._curve_length: dict[int, float] = {}

    def describe(
        self, element_tags: np.ndarray, element_nodes: np.ndarray, quality: np.ndarray
    ) -> tuple[BadElement, ...]:
        """One record per element, in element-tag order."""
        order = np.argsort(element_tags)
        return tuple(
            self._describe_element(
                int(element_tags[row]), element_nodes[row], float(quality[row])
            )
            for row in order
        )

    def _describe_element(
        self, element: int, nodes: np.ndarray, quality: float
    ) -> BadElement:
        direct = self._entities_of_nodes(nodes)
        corners = self._node_xyz[nodes]
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        lengths = [float(np.linalg.norm(corners[i] - corners[j])) for i, j in edges]
        shortest_i, shortest_j = edges[int(np.argmin(lengths))]
        curves_via_points = sorted(
            {c for p in direct[0] for c in self._point_curves.get(p, ())}
        )
        candidates = self._candidates(direct[1] | set(curves_via_points), nodes)
        return BadElement(
            element=element,
            quality=quality,
            edge_min=min(lengths),
            edge_max=max(lengths),
            centroid=_triple(corners.mean(axis=0)),
            shortest_edge_midpoint=_triple(
                (corners[shortest_i] + corners[shortest_j]) / 2
            ),
            location=_location(direct),
            points=tuple(sorted(direct[0])),
            curves=tuple(sorted(direct[1])),
            surfaces=tuple(sorted(direct[2])),
            volumes=tuple(sorted(direct[3])),
            sgb_ids=self._planned_ids(direct),
            groups=self._groups_around(direct, curves_via_points),
            feature_kinds=self._feature_kinds(c.tag for c in candidates),
            candidates=candidates,
        )

    def _entities_of_nodes(self, nodes: np.ndarray) -> dict[int, set[int]]:
        """Entity tags the element's nodes are classified on, by dimension."""
        direct: dict[int, set[int]] = {dim: set() for dim in _DIMENSIONS}
        for node in nodes:
            direct[int(self._node_dimension[node])].add(int(self._node_entity[node]))
        return direct

    def _planned_ids(self, direct: dict[int, set[int]]) -> tuple[str, ...]:
        return tuple(
            planned
            for dim in _DIMENSIONS
            for tag in sorted(direct[dim])
            if (planned := self._binding.id_of(dim, tag)) is not None
        )

    def _groups_around(
        self, direct: dict[int, set[int]], curves_via_points: Sequence[int]
    ) -> tuple[str, ...]:
        """Group names of the faces the element touches: directly, through its curves, through its points.

        A curve or a point has no group of its own; it belongs to those of the faces that meet there.
        """
        faces = set(direct[2])
        for curve in sorted(direct[1]) + list(curves_via_points):
            faces.update(self._curve_faces.get(curve, ()))
        names: set[str] = set()
        for face in faces:
            planned = self._binding.id_of(2, face)
            if planned is not None:
                names.update(self._groups_of_face.get(planned, ()))
        return tuple(sorted(names))

    def _feature_kinds(self, curves: Iterable[int]) -> tuple[str, ...]:
        kinds: set[str] = set()
        for curve in curves:
            kinds.update(self._kinds_of_curve.get(curve, ()))
        return tuple(sorted(kinds))

    def _candidates(
        self, curves: set[int], nodes: np.ndarray
    ) -> tuple[CurveCandidate, ...]:
        """The curves through which the element can be repaired, with the edge length they have next to it."""
        found = []
        for curve in sorted(curves):
            edges = self._edges_of(curve)
            touching = np.isin(edges, nodes).any(axis=1)
            if touching.any():
                span = (
                    self._node_xyz[edges[touching, 0]]
                    - self._node_xyz[edges[touching, 1]]
                )
                found.append(
                    CurveCandidate(
                        tag=curve,
                        local_edge=float(np.linalg.norm(span, axis=1).max()),
                        length=self._length_of(curve),
                    )
                )
        return tuple(found)

    def _edges_of(self, curve: int) -> np.ndarray:
        """Node-tag pairs of the 1D mesh of ``curve`` (cached)."""
        if curve not in self._curve_edges:
            types, _tags, nodes = gmsh.model.mesh.getElements(1, curve)
            pairs = [
                np.asarray(n, dtype=np.int64).reshape(-1, 2)
                for kind, n in zip(types, nodes, strict=True)
                if kind == _LINE
            ]
            self._curve_edges[curve] = (
                np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)
            )
        return self._curve_edges[curve]

    def _length_of(self, curve: int) -> float:
        """Length of ``curve`` as the sum of its 1D mesh edges (cached)."""
        if curve not in self._curve_length:
            edges = self._edges_of(curve)
            span = self._node_xyz[edges[:, 0]] - self._node_xyz[edges[:, 1]]
            self._curve_length[curve] = float(np.linalg.norm(span, axis=1).sum())
        return self._curve_length[curve]


def _location(direct: dict[int, set[int]]) -> str:
    for name, dim in (("point", 0), ("curve", 1), ("surface", 2)):
        if direct[dim]:
            return name
    return "interior"


def _triple(values: np.ndarray) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _groups_of_faces(features: SidecarFeatures) -> dict[str, list[str]]:
    """Physical-group names by planned face id."""
    groups: dict[str, list[str]] = {}
    for group in features.groups:
        if group.dimension == 2:
            for face_id in group.ids:
                groups.setdefault(face_id, []).append(group.name)
    return groups


def _adjacency() -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    """Faces bounded by each curve, and curves ending at each point, from the model's boundary relations."""
    curve_faces: dict[int, tuple[int, ...]] = {}
    point_curves: dict[int, list[int]] = {}
    for _, curve in gmsh.model.getEntities(1):
        up, down = gmsh.model.getAdjacencies(1, curve)
        curve_faces[int(curve)] = tuple(int(f) for f in up)
        for point in down:
            point_curves.setdefault(int(point), []).append(int(curve))
    return curve_faces, {p: tuple(c) for p, c in point_curves.items()}


def _classify_nodes() -> tuple[np.ndarray, np.ndarray]:
    """Dimension and tag of the entity each node is classified on (Gmsh puts a node on exactly one)."""
    node_tags = np.asarray(gmsh.model.mesh.getNodes()[0], dtype=np.int64)
    dimension = np.full(int(node_tags.max()) + 1, -1, dtype=np.int8)
    entity = np.zeros(int(node_tags.max()) + 1, dtype=np.int64)
    for dim in _DIMENSIONS:
        for _, tag in gmsh.model.getEntities(dim):
            owned = np.asarray(
                gmsh.model.mesh.getNodes(dim, tag, includeBoundary=False)[0],
                dtype=np.int64,
            )
            dimension[owned] = dim
            entity[owned] = tag
    return dimension, entity


def _node_coordinates() -> np.ndarray:
    """Coordinates indexed by node tag."""
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags, dtype=np.int64)
    table = np.zeros((int(tags.max()) + 1, 3))
    table[tags] = np.asarray(xyz).reshape(-1, 3)
    return table
