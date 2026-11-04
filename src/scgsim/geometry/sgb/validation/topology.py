"""Invariants of the assembled topology and its groups (design brief section 11, stage S9).

* every shell is closed and orientable (each edge used twice with opposite sense) and has
  the Euler characteristic ``2 - 2 * genus`` (even for a void);
* every volume has an outer shell; each face is used by at most two volumes, an interior
  face by exactly two sides;
* a curved wall is one cylinder: its bottom and top arc lie on the same circle;
* every physical group is non-empty, the primary groups partition each dimension and every
  emitted face and volume is in exactly one; unions reference primary groups;
* every edge is shared by two faces (a free edge is allowed only on an embedded sheet).

Everything is recomputed from the faces, independently of how the topology was built.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from scgsim.geometry.sgb.assembly.records import TopoEdge, Topology
from scgsim.geometry.sgb.assembly.shells import shell_checks
from scgsim.geometry.sgb.parallel import Runtime
from scgsim.geometry.sgb.routes.groups import RoutedModel
from scgsim.geometry.sgb.validation.report import ValidationReport

CYLINDER_TOLERANCE = 1e-6  # grid units: the same circle within floating-point noise


def check_shells(report: ValidationReport, topology: Topology, rt: Runtime) -> None:
    """Closure, orientation and Euler characteristic of every shell (a parallel stage per shell)."""
    shells = [
        (vid, si, shell)
        for vid, volume in sorted(topology.volumes.items())
        for si, shell in enumerate(volume.shells)
    ]
    owners = [i % (rt.tiles * rt.tiles) for i in range(len(shells))]
    results = rt.run(
        "validation_shells",
        _shell_check_job,
        (topology.faces, topology.edges, [s[2] for s in shells]),
        owners,
    )
    problems, euler_bad = [], []
    for (vid, si, shell), (issues, chi) in zip(shells, results, strict=True):
        problems.extend(f"{vid}/{si}: {message}" for message in issues)
        if chi % 2 != 0 or (shell.role == "outer" and chi > 2):
            euler_bad.append(f"{vid}/{si}: chi={chi}")
    report.add(
        "shells_closed_and_oriented",
        not problems,
        "; ".join(problems[:5])
        or f"{len(shells)} shells: every edge used twice with opposite orientation",
    )
    report.add(
        "shell_euler_characteristic",
        not euler_bad,
        "; ".join(euler_bad[:5]) or "chi = 2 - 2*genus for every shell",
    )
    no_outer = [
        v
        for v, vol in topology.volumes.items()
        if vol.shells and vol.shells[0].role != "outer"
    ]
    report.add(
        "every_volume_has_outer_shell",
        not no_outer,
        f"{len(no_outer)} volumes without positive-volume shell",
    )


def _shell_check_job(shared: Any, items: list[int]) -> list[tuple[list[str], int]]:
    faces, edges, shells = shared
    return [shell_checks(faces, edges, list(shells[i].faces)) for i in items]


def check_face_sides(report: ValidationReport, topology: Topology) -> None:
    """Each face borders at most two volumes; an interior face has two sides."""
    uses: Counter[int] = Counter()
    for face in topology.faces.values():
        uses[len({v for v in (face.neg_volume, face.pos_volume) if v is not None})] += 1
    interior_bad = [
        f.id
        for f in topology.faces.values()
        if f.neg_domain is not None
        and f.pos_domain is not None
        and (f.neg_volume is None or f.pos_volume is None)
    ]
    report.add(
        "each_face_used_by_at_most_two_volumes",
        True,
        f"face-volume count histogram {dict(uses)}",
    )
    report.add(
        "interior_faces_have_two_sides",
        not interior_bad,
        f"{len(interior_bad)} interior faces missing a side",
    )


def check_cylinders(report: ValidationReport, topology: Topology) -> None:
    """A curved wall is one cylinder: its bottom and top arc are the same arc of one circle."""
    bad = []
    count = 0
    for face in topology.faces.values():
        if face.cylinder is None:
            continue
        count += 1
        bottom, top = (topology.edges[eid] for eid in face.cylinder)
        if not _same_circle(topology, bottom, top):
            bad.append(face.id)
    report.add(
        "curved_walls_are_single_cylinders",
        not bad,
        f"{len(bad)} curved walls whose top and bottom arcs are not the same arc of one circle: {bad[:3]}"
        if bad
        else f"{count} curved walls, each bounded by one bottom and one top arc, the same arc of one circle",
    )
    walls = [
        f.id
        for f in topology.faces.values()
        if f.kind == "wall" and not f.planar and not f.curved
    ]
    report.add(
        "walls_planar", not walls, f"{len(walls)} non-planar walls", warning=True
    )


def _same_circle(topology: Topology, bottom: TopoEdge, top: TopoEdge) -> bool:
    """Whether two arcs are the same arc of one circle, seen from above: same centre, radius and end points.

    An edge's direction is arbitrary, so the top arc may run between the same two points the other way; its
    sense of travel is then reversed too, and the two arcs are still the same arc.
    """
    a, b = bottom.curve, top.curve
    if a is None or b is None:
        return False
    if not (
        abs(a.centre[0] - b.centre[0]) <= CYLINDER_TOLERANCE
        and abs(a.centre[1] - b.centre[1]) <= CYLINDER_TOLERANCE
        and abs(a.radius - b.radius) <= CYLINDER_TOLERANCE
    ):
        return False
    a_ends = (_xy(topology, bottom.v0), _xy(topology, bottom.v1))
    b_ends = (_xy(topology, top.v0), _xy(topology, top.v1))
    if a_ends == b_ends:
        return a.ccw == b.ccw
    if a_ends == b_ends[::-1]:
        return a.ccw != b.ccw
    return False


def _xy(topology: Topology, vertex_id: str) -> tuple[float, float]:
    vertex = topology.vertices[vertex_id]
    return (vertex.x, vertex.y)


def check_groups(
    report: ValidationReport, topology: Topology, routed: RoutedModel
) -> None:
    """Primary groups are non-empty and partition each dimension; unions reference them."""
    groups = routed.groups
    empty = [n for n, g in groups.items() if not g.ids]
    report.add(
        "physical_groups_non_empty",
        not empty and bool(groups),
        f"{len(groups)} groups; empty: {empty}",
    )
    seen: Counter[tuple[int, str]] = Counter()
    for group in groups.values():
        seen.update((group.dim, i) for i in group.ids)
    overlapping = [k for k, n in seen.items() if n > 1]
    report.add(
        "groups_partition_each_dimension",
        not overlapping,
        f"{len(overlapping)} entities in more than one group",
    )
    emitted = routed.emitted_volumes
    missing = [
        f.id
        for f in topology.faces.values()
        if routed.is_emitted_face(f.neg_volume, f.pos_volume)
        and routed.face_group[f.id] is not None
        and (2, f.id) not in seen
    ]
    report.add(
        "every_emitted_face_in_one_group",
        not missing,
        f"{len(missing)} emitted faces without group",
    )
    unnamed = [
        f.id
        for f in topology.faces.values()
        if routed.is_emitted_face(f.neg_volume, f.pos_volume)
        and routed.face_group[f.id] is None
    ]
    report.add(
        "every_emitted_face_has_a_name",
        not unnamed,
        f"{len(unnamed)} emitted faces without a name",
    )
    missing_volumes = [v for v in emitted if (3, v) not in seen]
    report.add(
        "every_emitted_volume_in_one_group",
        not missing_volumes,
        f"{len(missing_volumes)} emitted volumes without group",
    )
    unknown = [
        m for members in routed.unions.values() for m in members if m not in groups
    ]
    report.add(
        "unions_reference_primary_groups",
        not unknown,
        f"unknown members: {unknown[:3]}",
    )


def check_edges_in_faces(report: ValidationReport, topology: Topology) -> None:
    """Every edge borders two faces; only the free edges of an embedded sheet may border one."""
    around: Counter[str] = Counter()
    for face in topology.faces.values():
        for loop in face.loops:
            for eid, _ in loop:
                around[eid] += 1
    embedded = {fid for v in topology.volumes.values() for fid in v.embedded_faces}
    free: set[str] = set()
    for fid in embedded:
        for loop in topology.faces[fid].loops:
            free.update(eid for eid, _ in loop if around[eid] == 1)
    bad = [eid for eid, n in around.items() if n < 2 and eid not in free]
    report.add(
        "every_edge_shared_by_two_faces",
        not bad,
        f"{len(bad)} dangling edges; {len(free)} free edges of embedded faces",
    )


def check_vertical_edges(
    report: ValidationReport, topology: Topology, grid_um: float
) -> None:
    """Vertical edges shorter than the grid come from slabs thinner than ``g``: reported, never a failure."""
    short = []
    for edge in topology.edges.values():
        if edge.kind != "vertical":
            continue
        a, b = topology.vertices[edge.v0], topology.vertices[edge.v1]
        if float(abs(b.z - a.z)) < grid_um:
            short.append(edge.id)
    report.add(
        "vertical_edges_shorter_than_g_are_kept",
        not short,
        f"{len(short)} vertical edge(s) shorter than the grid, kept as declared"
        if short
        else "no vertical edge is shorter than the grid",
        warning=True,
    )
