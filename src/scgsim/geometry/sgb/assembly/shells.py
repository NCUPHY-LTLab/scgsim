"""Shells of the volumes and their closure checks (design brief sections 2 and 11).

Owns grouping the faces around each volume into connected closed surfaces and
classifying each as an outer shell or an inner void by its exact signed volume
(integer arithmetic with Z scaled by the common level denominator).  The chord
polyhedron is exact; the cylinder patches of curved walls add or remove
``segment area x height`` through each wall's ``bulge_volume``.

Shells are computed per volume, one parallel stage.  The same module offers
``shell_checks``, which recomputes closure, orientation and the Euler characteristic of a
shell from its faces, independently of how the shell was built (used by validation).
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import Any

from scgsim.geometry.sgb.assembly.records import Shell, TopoEdge, TopoFace, TopoVertex

Vec = tuple[int, int, int]


def volume_shells(
    faces: dict[str, TopoFace],
    edges: dict[str, TopoEdge],
    vertices: dict[str, TopoVertex],
    zden: int,
    entries: list[tuple[str, int]],
) -> list[Shell]:
    """The shells of one volume from its incident ``(face id, +1 if the normal points out)`` entries."""
    by_edge: dict[str, list[int]] = defaultdict(list)
    for i, (fid, _) in enumerate(entries):
        for loop in faces[fid].loops:
            for eid, _sign in loop:
                by_edge[eid].append(i)
    parent = list(range(len(entries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for members in by_edge.values():
        for member in members[1:]:
            root_a, root_b = find(members[0]), find(member)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
    components: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for i, entry in enumerate(entries):
        components[find(i)].append(entry)
    shells = []
    for group in components.values():
        six_volume = sum(
            _six_volume(faces, edges, vertices, fid, sign) for fid, sign in group
        )
        bulge = sum(
            sign * faces[fid].bulge_volume for fid, sign in group if faces[fid].curved
        )
        volume = Fraction(six_volume, 6 * zden) + (Fraction(bulge) if bulge else 0)
        shells.append(
            Shell(tuple(sorted(group)), volume, "outer" if six_volume > 0 else "inner")
        )
    shells.sort(key=lambda shell: (shell.role != "outer", shell.faces))
    return shells


def shell_job(shared: Any, items: list[int]) -> list[list[Shell]]:
    """Parallel stage function: the shells of the volumes at ``items``."""
    faces, edges, vertices, zden, entries_per_volume = shared
    return [
        volume_shells(faces, edges, vertices, zden, entries_per_volume[i])
        for i in items
    ]


def _six_volume(
    faces: dict[str, TopoFace],
    edges: dict[str, TopoEdge],
    vertices: dict[str, TopoVertex],
    face_id: str,
    sign: int,
) -> int:
    """Six times the divergence-theorem volume contribution of an outward-oriented face (z scaled to ints)."""
    face = faces[face_id]
    if face.kind == "horizontal":
        z = vertices[edges[face.loops[0][0][0]].v0].zs
        return sign * z * face.twice_xy_area
    points: list[Vec] = []
    for eid, edge_sign in face.loops[0]:
        edge = edges[eid]
        vertex = vertices[edge.v0 if edge_sign > 0 else edge.v1]
        points.append((int(vertex.x), int(vertex.y), vertex.zs))
    if sign < 0:
        points.reverse()
    return sum(
        _det(points[0], points[i], points[i + 1]) for i in range(1, len(points) - 1)
    )


def shell_checks(
    faces: dict[str, TopoFace],
    edges: dict[str, TopoEdge],
    entries: list[tuple[str, int]],
) -> tuple[list[str], int]:
    """Every edge used twice with opposite orientation; Euler characteristic ``2 - 2 * genus`` (even for voids)."""
    uses: dict[str, list[int]] = defaultdict(list)
    vertices: set[str] = set()
    chi = 0
    missing = []
    for fid, sign in entries:
        if fid not in faces:
            missing.append(f"shell lists unknown face {fid}")
            continue
        face = faces[fid]
        for loop in face.loops:
            for eid, edge_sign in loop:
                uses[eid].append(edge_sign * sign)
                vertices.update((edges[eid].v0, edges[eid].v1))
        chi += 1 - (len(face.loops) - 1)
    issues = list(missing)
    for eid, signs in uses.items():
        if len(signs) != 2:
            issues.append(f"edge {eid} used {len(signs)} times")
        elif signs[0] + signs[1] != 0:
            issues.append(f"edge {eid} not used with opposite orientation")
    chi += len(vertices) - len(uses)
    return issues, chi


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec, b: Vec) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _det(a: Vec, b: Vec, c: Vec) -> int:
    return _dot(a, _cross(b, c))


def is_planar(points: list[Vec]) -> bool:
    """Whether integer 3D points lie in one plane (exact)."""
    if len(points) <= 3:
        return True
    p0 = points[0]
    normal: Vec | None = None
    for i in range(1, len(points) - 1):
        a = _sub(points[i], p0)
        for j in range(i + 1, len(points)):
            c = _cross(a, _sub(points[j], p0))
            if any(c):
                normal = c
                break
        if normal:
            break
    return normal is None or all(_dot(normal, _sub(p, p0)) == 0 for p in points)
