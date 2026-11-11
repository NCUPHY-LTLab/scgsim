"""The flat, numbered description of the shapes to print (design brief decision D8).

Owns :class:`BrepPlan`: arrays and index lists over the shapes of one model in
``TopExp::MapShapes`` order, which is also the Gmsh tag order (tag = index + 1, verified by tests).
:func:`plan_brep` is the serial, small stage in front of the chunked, parallelisable text rendering
(``records``, ``writer``): it walks the emitted volumes (``walk``), builds the surfaces
(``surfaces``), the edge and arc arrays, and checks that every edge lies on the surfaces that will
carry its pcurve, within a tenth of the tolerance.

Units are micrometres.  Every shape carries ``TOLERANCE`` = 1e-7 um (OCCT's own
``Precision::Confusion``), 1e4 times smaller than the 1 nm model grid.

Does not own the text of the records (``records``) or the file layout (``writer``).
"""

from __future__ import annotations

import gc
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from scgsim.geometry.models.grid import format_um
from scgsim.geometry.sgb.assembly.records import Topology
from scgsim.geometry.sgb.emission.numbers import PI, angle_of
from scgsim.geometry.sgb.emission.surfaces import (
    CYLINDER,
    SurfaceBuilder,
    check_edges_lie_on_surfaces,
)
from scgsim.geometry.sgb.emission.walk import Numbering, walk_emitted_volumes
from scgsim.geometry.sgb.errors import SgbError
from scgsim.geometry.sgb.routes.groups import RoutedModel

TOLERANCE = 1e-7  # um; every vertex, edge and face carries this tolerance
LINE, ARC = 0, 1  # edge kinds
MIN_EDGE_TOLERANCES = 10  # an edge shorter than this many tolerances is refused


@dataclass(frozen=True)
class GroupPlan:
    """A primary physical group as the XAO carries it: elements are 0-based indices in tag order."""

    name: str
    dim: int  # 2 faces, 3 solids
    tag: int  # physical tag, counted from 1 per dimension in sorted-name order
    elements: tuple[int, ...]


@dataclass
class BrepPlan:
    """Numbered, flat description of the shapes to print (indices are 0-based in tag order)."""

    tolerance: float
    found: Numbering
    verts: np.ndarray  # (nV, 3) um
    edge_v: np.ndarray  # (nE, 2) vertex indices, start/end
    edge_kind: np.ndarray  # (nE,) LINE | ARC
    edge_dir: (
        np.ndarray
    )  # (nE, 3) unit direction (line) / unit centre -> start vector (arc)
    edge_span: (
        np.ndarray
    )  # (nE,) parameter range end: length (line) / angle (arc); the range starts at 0
    arc_centre: np.ndarray  # (nE, 3) (zeros for lines)
    arc_radius: np.ndarray  # (nE,)
    arc_sense: (
        np.ndarray
    )  # (nE,) +1 / -1: the circle's normal is +z / -z (zero for lines)
    surf_kind: np.ndarray  # (nS,) PLANE | CYLINDER
    surf_geom: np.ndarray  # (nS, 4, 3): origin, normal (axis), x axis, y axis
    surf_radius: np.ndarray  # (nS,) cylinders only
    pc_edge: np.ndarray  # (nPC,) edge of each (edge, surface) pcurve
    pc_surf: np.ndarray  # (nPC,)
    edge_pcs: list[list[int]]  # per edge: the pcurves it carries
    groups: list[GroupPlan]
    write_pcurves: bool = True
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def vertex_ids(self) -> list[str]:
        """Planned vertex ids in tag order."""
        return self.found.vertex_ids

    @property
    def edge_ids(self) -> list[str]:
        """Planned edge ids in tag order."""
        return self.found.edge_ids

    @property
    def face_ids(self) -> list[str]:
        """Planned face ids in tag order."""
        return self.found.face_ids

    @property
    def solid_ids(self) -> list[str]:
        """Planned volume ids in tag order."""
        return self.found.solid_ids

    @property
    def counts(self) -> dict[str, int]:
        """Entity counts by kind, as the file will hold them."""
        f = self.found
        return {
            "vertices": len(f.vertex_ids),
            "edges": len(f.edge_ids),
            "faces": len(f.face_ids),
            "solids": len(f.solid_ids),
            "wires": len(f.wires),
            "shells": len(f.shells),
            "surfaces": len(self.surf_kind),
            "cylinders": int((self.surf_kind == CYLINDER).sum()),
            "arcs": int((self.edge_kind == ARC).sum()),
            "pcurves": len(self.pc_edge) if self.write_pcurves else 0,
            "free_faces": len(f.free_faces),
            "free_edges": len(f.free_edges),
        }

    def tags(self) -> dict[str, tuple[int, int]]:
        """Planned id -> (dim, Gmsh tag) after loading the file (tag = index + 1)."""
        out: dict[str, tuple[int, int]] = {}
        for dim, ids in (
            (0, self.vertex_ids),
            (1, self.edge_ids),
            (2, self.face_ids),
            (3, self.solid_ids),
        ):
            out.update({pid: (dim, i + 1) for i, pid in enumerate(ids)})
        return out

    def section_bases(self) -> dict[str, int]:
        """Reference of the first shape of each kind: shapes are referenced from the end of the TShapes section."""
        c = self.counts
        total = (
            1
            + c["vertices"]
            + c["edges"]
            + c["wires"]
            + c["faces"]
            + c["shells"]
            + c["solids"]
        )
        base_e = total - c["vertices"]
        base_w = base_e - c["edges"]
        base_f = base_w - c["wires"]
        base_s = base_f - c["faces"]
        base_so = base_s - c["shells"]
        return {
            "total": total,
            "V": total,
            "E": base_e,
            "W": base_w,
            "F": base_f,
            "S": base_s,
            "SO": base_so,
        }


def plan_brep(
    topology: Topology,
    routed: RoutedModel,
    tolerance: float = TOLERANCE,
    *,
    pcurves: bool = True,
) -> BrepPlan:
    """Number every shape of the emitted volumes in ``TopExp::MapShapes`` order and collect surfaces and pcurves.

    The cyclic collector is paused: the walk allocates millions of acyclic objects next to a heap of
    the same size, and re-scanning it cost a third of the planning time at 20000 features.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        surfaces = SurfaceBuilder(topology)
        found = walk_emitted_volumes(topology, routed, surfaces)
        return _finish(topology, routed, surfaces, found, tolerance, pcurves)
    finally:
        if was_enabled:
            gc.enable()


def _finish(  # noqa: PLR0913 - the walk's products travel together
    topology: Topology,
    routed: RoutedModel,
    surfaces: SurfaceBuilder,
    found: Numbering,
    tolerance: float,
    pcurves: bool,
) -> BrepPlan:
    verts = np.array(
        [surfaces.point(v) for v in found.vertex_ids], dtype=float
    ).reshape(-1, 3)
    edge_v = np.array(found.edge_ends, dtype=np.int64).reshape(-1, 2)
    delta = verts[edge_v[:, 1]] - verts[edge_v[:, 0]]
    span = np.sqrt(
        delta[:, 0] * delta[:, 0]
        + delta[:, 1] * delta[:, 1]
        + delta[:, 2] * delta[:, 2]
    )
    if span.size and span.min() <= MIN_EDGE_TOLERANCES * tolerance:
        raise SgbError(_short_edge(topology, found, verts[edge_v], span, tolerance))
    direction = delta / span[:, None]
    n_edges = len(found.edge_ids)
    kind = np.zeros(n_edges, dtype=np.int8)
    centre, radius, sense = np.zeros((n_edges, 3)), np.zeros(n_edges), np.zeros(n_edges)
    _arc_records(
        topology,
        found,
        tolerance,
        verts,
        edge_v,
        (kind, direction, span, centre, radius, sense),
    )
    records = surfaces.records
    surf_kind = np.array([s.kind for s in records], dtype=np.int8)
    surf_geom = np.array([s.frame for s in records], dtype=float).reshape(-1, 4, 3)
    surf_radius = np.array([s.radius for s in records], dtype=float)
    pc_keys = np.fromiter(found.pcurves, dtype=np.int64, count=len(found.pcurves))
    pc_edge, pc_surf = pc_keys >> 32, pc_keys & 0xFFFFFFFF
    check_edges_lie_on_surfaces(
        verts,
        edge_v,
        kind,
        centre,
        surf_kind,
        surf_geom,
        surf_radius,
        pc_edge,
        pc_surf,
        tolerance,
        ARC,
    )
    edge_pcs: list[list[int]] = [[] for _ in found.edge_ids]
    for pc, e in enumerate(pc_edge.tolist()):
        edge_pcs[e].append(pc)
    check_one_tag_per_id(found)
    return BrepPlan(
        tolerance, found, verts, edge_v, kind, direction, span, centre, radius, sense, surf_kind, surf_geom,
        surf_radius, pc_edge, pc_surf, edge_pcs, _group_plans(routed, found), write_pcurves=pcurves,
    )  # fmt: skip


def check_one_tag_per_id(found: Numbering) -> None:
    """Every planned id maps to exactly one native tag: no id is numbered twice, within or across dimensions.

    The sidecar binds an id to ``tag = 1 + position`` and a consumer inverts that map; an id listed twice
    would send two native shapes to one planned entity.
    """
    listed = [
        *found.vertex_ids,
        *found.edge_ids,
        *found.face_ids,
        *found.solid_ids,
    ]
    repeated = sorted(i for i, n in Counter(listed).items() if n > 1)
    if repeated:
        raise SgbError(
            f"{len(repeated)} planned ids map to more than one native tag: {repeated[:3]}"
        )


def _arc_records(  # noqa: PLR0913 - the edge arrays travel together
    topology: Topology,
    found: Numbering,
    tolerance: float,
    verts: np.ndarray,
    edge_v: np.ndarray,
    arrays: tuple[np.ndarray, ...],
) -> None:
    """Circle records of the arc edges: normal +z (counter-clockwise travel) or -z, x axis towards the start point."""
    kind, direction, span, centre, radius, sense = arrays
    g = topology.grid_um
    for number, edge_id in enumerate(found.edge_ids):
        arc = topology.edges[edge_id].curve
        if arc is None:
            continue
        start, end = verts[edge_v[number, 0]], verts[edge_v[number, 1]]
        c = np.array([arc.centre[0] * g, arc.centre[1] * g, start[2]])
        r = arc.radius * g
        rel0, rel1 = start - c, end - c
        length0 = np.sqrt(rel0[0] * rel0[0] + rel0[1] * rel0[1])
        length1 = np.sqrt(rel1[0] * rel1[0] + rel1[1] * rel1[1])
        if (
            max(abs(length0 - r), abs(length1 - r), abs(end[2] - start[2]))
            > tolerance / 10
        ):
            raise SgbError(
                f"arc edge {edge_id}: end points are not on its circle within {tolerance / 10:g} um"
            )
        s = 1.0 if arc.ccw else -1.0
        cross = s * (rel0[0] * rel1[1] - rel0[1] * rel1[0])
        dot = rel0[0] * rel1[0] + rel0[1] * rel1[1]
        angle = float(angle_of(np.array([cross]), np.array([dot]))[0])
        if angle <= 0.0:
            angle += 2 * PI
        # the circle is parametrised by its angle (radians) from the start point
        kind[number], direction[number], span[number] = ARC, rel0 / length0, angle
        centre[number], radius[number], sense[number] = c, r, s


def _group_plans(routed: RoutedModel, found: Numbering) -> list[GroupPlan]:
    """Physical groups as element index lists, tagged per dimension from 1 in sorted-name order."""
    face_no = {fid: i for i, fid in enumerate(found.face_ids)}
    solid_no = {vid: i for i, vid in enumerate(found.solid_ids)}
    next_tag = {2: 1, 3: 1}
    out = []
    for name, group in routed.groups.items():
        lookup = face_no if group.dim == 2 else solid_no
        elements = tuple(lookup[i] for i in group.ids if i in lookup)
        if not elements:
            continue
        out.append(GroupPlan(name, group.dim, next_tag[group.dim], elements))
        next_tag[group.dim] += 1
    return out


def _short_edge(
    topology: Topology,
    found: Numbering,
    ends: np.ndarray,
    span: np.ndarray,
    tolerance: float,
) -> str:
    """The refusal of the shortest edge: where it runs, its sources, and which declaration sets its length.

    In the plane every edge is at least one grid step long, so an edge this short is a film
    thickness or a gap between levels unless the model grid itself is that fine.
    """
    index = int(span.argmin())
    edge_id = found.edge_ids[index]
    (x0, y0, z0), (x1, y1, z1) = ends[index]
    grid = topology.grid_um
    points = " to ".join(
        f"({format_um(x, grid)}, {format_um(y, grid)}, {z!r}) um"
        for x, y, z in ((x0, y0, float(z0)), (x1, y1, float(z1)))
    )
    sources = ", ".join(topology.lineage.get(edge_id, ())) or "no source carrier"
    limit = MIN_EDGE_TOLERANCES * tolerance
    fix = (
        f"Fix the layer stack: this edge is a film thickness or a gap between levels; "
        f"make each at least {limit:g} um."
        if (x0, y0) == (x1, y1)
        else f"Fix the declaration: use a model grid of at least {limit:g} um."
    )
    return (
        f"edge shorter than {MIN_EDGE_TOLERANCES} x tolerance ({span.min():.3g} um): "
        f"edge {edge_id} runs from {points}, from {sources}. {fix}"
    )
