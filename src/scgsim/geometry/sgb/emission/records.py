"""The text of every BREP and XAO section (design brief decision D8).

Owns turning index ranges of a :class:`~scgsim.geometry.sgb.emission.plan.BrepPlan` into the ASCII
records of OpenCASCADE's ``CASCADE Topology V1`` format and the XAO topology lists.  Every function
is pure over the read-only plan, so ranges render independently (``writer`` runs them in chunks, in
parallel).

Records that depend on the kind of curve or surface are looked up in tables
(``CURVE_RECORDS``, ``SURFACE_RECORDS``, ``PCURVE_RECORDS``); a B-spline wall would add a surface
record and a pcurve record per edge kind and change nothing else.

Number policy (``numbers``): coordinates print with 12 significant digits (they are lattice integers
times the grid, exact to far below the tolerance), unit vectors, angles and radii with 15.  Both go
through Python ``%`` formatting, which is exact on every platform.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from scgsim.geometry.sgb.emission.numbers import PI, angle_of, rows
from scgsim.geometry.sgb.emission.plan import ARC, LINE, BrepPlan
from scgsim.geometry.sgb.emission.surfaces import CYLINDER, PLANE

# Shape flag words copied from files written by Gmsh 4.15.2 (free, modified, checked, orientable, closed, ...).
FLAGS = {
    "Ve": "0101101",
    "Ed": "0101000",
    "Wi": "0101100",
    "Fa": "0101000",
    "Sh": "0101100",
    "So": "0100000",
    "Co": "1100000",
}
COORD = "%.12g"
UNIT = "%.15g"
SECTION_TEXT: dict[str, Callable[[BrepPlan, int, int], str]] = {}


def section(
    name: str,
) -> Callable[
    [Callable[[BrepPlan, int, int], str]], Callable[[BrepPlan, int, int], str]
]:
    """Register a function as the renderer of the section ``name``."""

    def register(
        fn: Callable[[BrepPlan, int, int], str],
    ) -> Callable[[BrepPlan, int, int], str]:
        SECTION_TEXT[name] = fn
        return fn

    return register


def _tolerance(plan: BrepPlan) -> str:
    return "%.9g" % plan.tolerance


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise dot product of 3-vectors, summed in a fixed order (``einsum`` may fuse or reorder)."""
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def _interleave(
    codes: np.ndarray,
    writers: dict[int, Callable[..., list[str]]],
    plan: BrepPlan,
    *index_arrays: np.ndarray,
) -> str:
    """Records of mixed kinds in item order: ``codes[i]`` selects the writer of item ``i``."""
    out = np.empty(len(codes), dtype=object)
    for code in np.unique(codes).tolist():
        picked = np.nonzero(codes == code)[0]
        out[picked] = writers[code](plan, *(a[picked] for a in index_arrays))
    return "".join(out.tolist())


# -- pcurves: one record per (edge kind, surface kind) combination ---------------------


def _pcurve_line_on_plane(plan: BrepPlan, e: np.ndarray, p: np.ndarray) -> list[str]:
    d0 = plan.verts[plan.edge_v[e, 0]] - plan.surf_geom[p, 0]
    direction, x_axis, y_axis = (
        plan.edge_dir[e],
        plan.surf_geom[p, 2],
        plan.surf_geom[p, 3],
    )
    table = np.stack(
        [
            _dot(d0, x_axis),
            _dot(d0, y_axis),
            _dot(direction, x_axis),
            _dot(direction, y_axis),
        ],
        axis=1,
    )
    return rows(f"1 {COORD} {COORD} {UNIT} {UNIT}\n", table)


def _pcurve_arc_on_plane(plan: BrepPlan, e: np.ndarray, p: np.ndarray) -> list[str]:
    """Circle2d: centre, x axis, y axis (indirect when the circle runs clockwise), radius."""
    c0 = plan.arc_centre[e] - plan.surf_geom[p, 0]
    x3 = plan.edge_dir[e]
    sense = plan.arc_sense[e]
    y3 = np.stack([-sense * x3[:, 1], sense * x3[:, 0], np.zeros(len(e))], axis=1)
    x_axis, y_axis = plan.surf_geom[p, 2], plan.surf_geom[p, 3]
    table = np.stack(
        [
            _dot(c0, x_axis), _dot(c0, y_axis), _dot(x3, x_axis), _dot(x3, y_axis),
            _dot(y3, x_axis), _dot(y3, y_axis), plan.arc_radius[e],
        ],
        axis=1,
    )  # fmt: skip
    return rows("2 " + " ".join([UNIT] * 7) + "\n", table)


def _angle_on_cylinder(
    plan: BrepPlan, e: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) of an edge's start point on its cylinder: the angle from the x axis in [0, 2 pi) and the height."""
    d0 = plan.verts[plan.edge_v[e, 0]] - plan.surf_geom[p, 0]
    u = angle_of(_dot(d0, plan.surf_geom[p, 3]), _dot(d0, plan.surf_geom[p, 2]))
    return np.where(u < -1e-12, u + 2 * PI, u), _dot(d0, plan.surf_geom[p, 1])


def _pcurve_line_on_cylinder(plan: BrepPlan, e: np.ndarray, p: np.ndarray) -> list[str]:
    """A vertical edge: a straight line along v (the height)."""
    u, v = _angle_on_cylinder(plan, e, p)
    table = np.stack(
        [u, v, np.zeros(len(e)), _dot(plan.edge_dir[e], plan.surf_geom[p, 1])], axis=1
    )
    return rows(f"1 {UNIT} {UNIT} {UNIT} {UNIT}\n", table)


def _pcurve_arc_on_cylinder(plan: BrepPlan, e: np.ndarray, p: np.ndarray) -> list[str]:
    """An arc about the axis: a straight line along u (the angle), which increases for counter-clockwise travel."""
    u, v = _angle_on_cylinder(plan, e, p)
    table = np.stack([u, v, plan.arc_sense[e], np.zeros(len(e))], axis=1)
    return rows(f"1 {UNIT} {UNIT} {UNIT} {UNIT}\n", table)


PCURVE_RECORDS = {  # (edge kind, surface kind) -> record writer; the extension point for new curve/surface pairs
    (LINE, PLANE): _pcurve_line_on_plane,
    (ARC, PLANE): _pcurve_arc_on_plane,
    (LINE, CYLINDER): _pcurve_line_on_cylinder,
    (ARC, CYLINDER): _pcurve_arc_on_cylinder,
}


@section("curve2ds")
def text_pcurves(plan: BrepPlan, lo: int, hi: int) -> str:
    """Pcurve records ``lo`` to ``hi``."""
    e, p = plan.pc_edge[lo:hi], plan.pc_surf[lo:hi]
    codes = plan.edge_kind[e].astype(int) * 2 + plan.surf_kind[p]
    writers = {ek * 2 + sk: fn for (ek, sk), fn in PCURVE_RECORDS.items()}
    return _interleave(codes, writers, plan, e, p)


# -- 3D curves and surfaces --------------------------------------------------------------


def _curve_line(plan: BrepPlan, e: np.ndarray) -> list[str]:
    table = np.concatenate([plan.verts[plan.edge_v[e, 0]], plan.edge_dir[e]], axis=1)
    return rows(f"1 {COORD} {COORD} {COORD} {UNIT} {UNIT} {UNIT}\n", table)


def _curve_circle(plan: BrepPlan, e: np.ndarray) -> list[str]:
    x3 = plan.edge_dir[e]
    s = plan.arc_sense[e]
    n3 = np.stack([np.zeros(len(e)), np.zeros(len(e)), s], axis=1)
    y3 = np.stack([-s * x3[:, 1], s * x3[:, 0], np.zeros(len(e))], axis=1)  # N x X
    table = np.concatenate(
        [plan.arc_centre[e], n3, x3, y3, plan.arc_radius[e][:, None]], axis=1
    )
    return rows("2 " + " ".join([UNIT] * 13) + "\n", table)


CURVE_RECORDS = {
    LINE: _curve_line,
    ARC: _curve_circle,
}  # extension point: a new curve kind adds an entry


@section("curves")
def text_curves(plan: BrepPlan, lo: int, hi: int) -> str:
    """Curve records ``lo`` to ``hi`` (one per edge)."""
    return _interleave(plan.edge_kind[lo:hi], CURVE_RECORDS, plan, np.arange(lo, hi))


def _surface_plane(plan: BrepPlan, s: np.ndarray) -> list[str]:
    return rows(
        "1 " + f"{COORD} " * 3 + f"{UNIT} " * 8 + f"{UNIT}\n",
        plan.surf_geom[s].reshape(-1, 12),
    )


def _surface_cylinder(plan: BrepPlan, s: np.ndarray) -> list[str]:
    table = np.concatenate(
        [plan.surf_geom[s].reshape(-1, 12), plan.surf_radius[s][:, None]], axis=1
    )
    return rows("2 " + f"{UNIT} " * 12 + f"{UNIT}\n", table)


SURFACE_RECORDS = {PLANE: _surface_plane, CYLINDER: _surface_cylinder}


@section("surfaces")
def text_surfaces(plan: BrepPlan, lo: int, hi: int) -> str:
    """Surface records ``lo`` to ``hi``."""
    return _interleave(plan.surf_kind[lo:hi], SURFACE_RECORDS, plan, np.arange(lo, hi))


# -- shapes ------------------------------------------------------------------------------


@section("vertices")
def text_vertices(plan: BrepPlan, lo: int, hi: int) -> str:
    """Vertex shapes ``lo`` to ``hi``."""
    template = (
        "Ve\n"
        + _tolerance(plan)
        + f"\n{COORD} {COORD} {COORD}\n0 0\n\n"
        + FLAGS["Ve"]
        + "\n*\n"
    )
    return "".join(rows(template, plan.verts[lo:hi]))


@section("edges")
def text_edges(plan: BrepPlan, lo: int, hi: int) -> str:
    """Edge shapes ``lo`` to ``hi`` with their curve and pcurve representations."""
    ref_v, tolerance = plan.section_bases()["V"], _tolerance(plan)
    spans = [UNIT % x for x in plan.edge_span[lo:hi].tolist()]
    starts, ends = plan.edge_v[lo:hi, 0].tolist(), plan.edge_v[lo:hi, 1].tolist()
    pc_surf = plan.pc_surf
    head = "Ed\n " + tolerance + " 1 1 0\n1  %d 0 0 %s\n"
    tail = "0\n\n" + FLAGS["Ed"] + "\n-%d 0 +%d 0 *\n"
    out = []
    for k, i in enumerate(range(lo, hi)):
        representations = "".join(
            "2  %d %d 0 0 %s\n" % (pc + 1, pc_surf[pc] + 1, spans[k])
            for pc in plan.edge_pcs[i]
        )
        out.append(
            head % (i + 1, spans[k])
            + representations
            + tail % (ref_v - ends[k], ref_v - starts[k])
        )
    return "".join(out)


def _children(base: int, items: list[int]) -> str:
    """Child references of a wire or shell: ``+``/``-`` orientation and the index counted back from ``base``."""
    return "".join(
        "%s%d 0 " % ("+" if s > 0 else "-", base - abs(s) + 1) for s in items
    )


@section("wires")
def text_wires(plan: BrepPlan, lo: int, hi: int) -> str:
    """Wire shapes ``lo`` to ``hi``."""
    base_e = plan.section_bases()["E"]
    head = "Wi\n\n" + FLAGS["Wi"] + "\n"
    return "".join(
        head + _children(base_e, plan.found.wires[i]) + "*\n" for i in range(lo, hi)
    )


@section("faces")
def text_faces(plan: BrepPlan, lo: int, hi: int) -> str:
    """Face shapes ``lo`` to ``hi``."""
    base_w = plan.section_bases()["W"]
    tolerance, flags = _tolerance(plan), FLAGS["Fa"]
    found = plan.found
    out = []
    for i in range(lo, hi):
        first = found.face_first_wire[i]
        wires = "".join(
            "+%d 0 " % (base_w - w)
            for w in range(first, first + found.face_wire_count[i])
        )
        out.append(
            "Fa\n0  %s %d 0\n\n%s\n%s*\n"
            % (tolerance, found.face_surface[i] + 1, flags, wires)
        )
    return "".join(out)


@section("shells")
def text_shells(plan: BrepPlan, lo: int, hi: int) -> str:
    """Shell shapes ``lo`` to ``hi``."""
    base_f = plan.section_bases()["F"]
    head = "Sh\n\n" + FLAGS["Sh"] + "\n"
    return "".join(
        head + _children(base_f, plan.found.shells[i]) + "*\n" for i in range(lo, hi)
    )


@section("solids")
def text_solids(plan: BrepPlan, lo: int, hi: int) -> str:
    """Solid shapes ``lo`` to ``hi``."""
    base_s = plan.section_bases()["S"]
    head = "So\n\n" + FLAGS["So"] + "\n"
    found = plan.found
    out = []
    for i in range(lo, hi):
        first = found.solid_first_shell[i]
        shells = "".join(
            "+%d 0 " % (base_s - s)
            for s in range(first, first + found.solid_shell_count[i])
        )
        out.append(head + shells + "*\n")
    return "".join(out)


def compound_text(plan: BrepPlan) -> str:
    """The root compound: every solid, then the free faces, then the free edges."""
    b = plan.section_bases()
    found = plan.found
    children = ["+%d 0 " % (b["SO"] - i) for i in range(len(found.solid_ids))]
    children += ["+%d 0 " % (b["F"] - i) for i in found.free_faces]
    children += ["+%d 0 " % (b["E"] - i) for i in found.free_edges]
    return "Co\n\n" + FLAGS["Co"] + "\n" + "".join(children) + "*\n\n+1 0 \n"


# -- the XAO topology lists ---------------------------------------------------------------


def _xao_entries(kind: str, references: list[int], lo: int, hi: int) -> str:
    return "".join(
        f'        <{kind} index="{i}" name="" reference="{references[i]}"/>\n'
        for i in range(lo, hi)
    )


@section("xao_vertices")
def text_xao_vertices(plan: BrepPlan, lo: int, hi: int) -> str:
    """XAO vertex entries ``lo`` to ``hi``."""
    return _xao_entries("vertex", plan.found.vertex_ref, lo, hi)


@section("xao_edges")
def text_xao_edges(plan: BrepPlan, lo: int, hi: int) -> str:
    """XAO edge entries ``lo`` to ``hi``."""
    return _xao_entries("edge", plan.found.edge_ref, lo, hi)


@section("xao_faces")
def text_xao_faces(plan: BrepPlan, lo: int, hi: int) -> str:
    """XAO face entries ``lo`` to ``hi``."""
    return _xao_entries("face", plan.found.face_ref, lo, hi)
