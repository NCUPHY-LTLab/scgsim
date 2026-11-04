"""Catalogue of minimal extreme-case fixtures (phase 4).

Every fixture is a handful of prisms (a few dozen faces) that targets exactly one risk, so that all the official
checkers stay cheap. The catalogue is data: :data:`FIXTURES` maps a name to a :class:`Fixture` (risk, builder,
routes, expected outcome per route). :func:`run_fixture` runs every check on one fixture/route pair and returns a
row with a time per check:

``build`` (pipeline S1-S9 including the structural validation and the contact / manifold checks),
``write_brep`` / ``write_xao`` (direct writer), ``gmsh_load`` + ``gmsh_verify`` (counts, boundaries, groups, volumes,
areas against the plan), ``gmsh_mesh`` (conformal 3D mesh), ``occt_*`` (``BRepCheck_Analyzer``,
``BOPAlgo_ArgumentAnalyzer`` with every mode including self-interference, ``ShapeAnalysis``, plan cross-check).

Fixtures that must be refused (a different-Net contact, a same-Net non-manifold pinch) carry the expected error
class; the row then records the early error and its message instead of the later checks.
"""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from sgbproto.errors import ContactError, NonManifoldError
from sgbproto.inset import SupportRegion, inset_family
from sgbproto.records import (
    CurveFamilyRecord, Footprint, Grid, LineCarrier, LineEdge, Lineage, ModelSpec, PrismRecord, circle_arcs, z_value,
)
from sgbproto.scenes import rect_um, to_grid

GRID = Grid()
METAL_THICKNESS = "0.2"
Point = tuple[float, float]


# --------------------------------------------------------------------------- building blocks


def chip(half: float = 20.0, depth: float = 10.0) -> list[PrismRecord]:
    """Substrate below z = 0 and vacuum above, both ``2 half`` wide: a small box instead of the 300 um example chip."""
    box = Footprint.from_rings(to_grid(GRID, rect_um(-half, -half, half, half)))
    return [
        PrismRecord("substrate", box, z_value(-depth), z_value(0), "substrate", "dielectric", role="body"),
        PrismRecord("vacuum", box, z_value(0), z_value(depth), "vacuum", "vacuum", role="body"),
    ]


def metal_z(route: str, base: float | str = 0, thickness: str = METAL_THICKNESS) -> tuple[Fraction, Fraction]:
    """(z0, z1) of a conductor: a sheet at ``base`` in Route A, a ``thickness`` thick prism in Route B."""
    z0 = z_value(base)
    return (z0, z0) if route == "A" else (z0, z0 + z_value(thickness))


def metal(name: str, net: str, outer: list[Point], route: str, holes: tuple[list[Point], ...] = (), base: float | str = 0,
          z: tuple[Fraction, Fraction] | None = None) -> PrismRecord:
    z0, z1 = z or metal_z(route, base)
    footprint = Footprint.from_rings(to_grid(GRID, outer), [to_grid(GRID, h) for h in holes])
    return PrismRecord(name, footprint, z0, z1, "Al", "conductor", net_id=net)


def grid_metal(name: str, net: str, outer_units: list[tuple[int, int]], route: str) -> PrismRecord:
    """A conductor whose outline is given in grid units (to place vertices 1, 2, 3 g apart)."""
    z0, z1 = metal_z(route)
    return PrismRecord(name, Footprint.from_rings(outer_units), z0, z1, "Al", "conductor", net_id=net)


def disk(name: str, net: str, centre_units: tuple[int, int], radius_units: int, route: str, base: float | str = 0) -> PrismRecord:
    """A conductor disk of exact radius: four quarter arcs, no polygon approximation."""
    arcs = circle_arcs(f"{name}/c", centre_units, radius_units, True, Lineage("polygon_edge", name, 0, 0, 0))
    footprint, carriers = Footprint.from_primitives(arcs)
    z0, z1 = metal_z(route, base)
    return PrismRecord(name, footprint, z0, z1, "Al", "conductor", net_id=net, arcs=carriers)


def spec_of(name: str, route: str, prisms: list[PrismRecord], families: tuple[CurveFamilyRecord, ...] = ()) -> ModelSpec:
    return ModelSpec(GRID, tuple(prisms), families, name=f"{name}_{route}", meta={"route": route})


def ring_of(points: list[Point]) -> tuple[LineEdge, ...]:
    """Counter-clockwise line primitives of a polygon given in um (the interior is on the left)."""
    pts = to_grid(GRID, points)
    if sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])) < 0:
        pts.reverse()
    return tuple(LineEdge(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))


def band_families(route: str, outline: list[Point], margins_um: tuple[float, ...]) -> tuple[CurveFamilyRecord, ...]:
    """Surface-EPR inset bands on the top of the conductor (Route A: its sheet) and, in Route B, also its bottom."""
    margins = tuple(GRID.snap(m) for m in margins_um)
    region = SupportRegion((ring_of(outline),), margins)
    top = z_value(0) if route == "A" else z_value(METAL_THICKNESS)
    families = [inset_family("band_top", region, (top,))]
    if route == "B":
        families.append(inset_family("band_bottom", region, (z_value(0),)))
    return tuple(families)


# --------------------------------------------------------------------------- fixtures


def baseline(route: str) -> ModelSpec:
    return spec_of("baseline", route, chip() + [metal("pad", "N", rect_um(-8, -5, 8, 5), route)])


def _tangent_pair(route: str, bar_bottom: float, name: str) -> ModelSpec:
    """A disk (r = 5 um) and a bar of the same Net whose lower edge cuts the circle ``5 - bar_bottom`` um deep."""
    bar = metal("bar", "N", rect_um(-12, bar_bottom, 12, 8), route)
    return spec_of(name, route, chip() + [disk("disk", "N", (0, 0), GRID.snap(5), route), bar])


def near_tangent_arc_line(route: str) -> ModelSpec:
    return _tangent_pair(route, 4.998, "near_tangent_arc_line")  # 2 nm secant depth: crossing angle 0.03 rad, chord 0.28 um


def exactly_tangent_arc_line(route: str) -> ModelSpec:
    return _tangent_pair(route, 5.0, "exactly_tangent_arc_line")  # touches at one point (0, 5)


def arcs_crossing_shallow(route: str) -> ModelSpec:
    """Two same-Net disks (r = 5 um) whose centres are 9.999 um apart: the circles cross at about 1.6 degrees."""
    left = disk("left", "N", (-5000, 0), 5000, route)
    right = disk("right", "N", (4999, 0), 5000, route)
    return spec_of("arcs_crossing_shallow", route, chip() + [left, right])


def edges_1_to_3_g(route: str) -> ModelSpec:
    """A 10 um square whose right side carries jogs of 1, 2, 3 g and 1.4 g (diagonal) edges."""
    outline = [(0, 0), (10000, 0), (10000, 4000), (10001, 4000), (10001, 4002), (10004, 4002), (10004, 6000), (10003, 6001),
               (10003, 6003), (10004, 6004), (10004, 10000), (0, 10000)]
    shifted = [(x - 5000, y - 5000) for x, y in outline]
    return spec_of("edges_1_to_3_g", route, chip() + [grid_metal("jogged", "N", shifted, route)])


def kissing_corners_same_net(route: str) -> ModelSpec:
    return spec_of("kissing_corners_same_net", route, chip() + [
        metal("padL", "N", rect_um(-10, -10, 0, 0), route), metal("padR", "N", rect_um(0, 0, 10, 10), route)])


def kissing_corners_different_nets(route: str) -> ModelSpec:
    return spec_of("kissing_corners_different_nets", route, chip() + [
        metal("padL", "N1", rect_um(-10, -10, 0, 0), route), metal("padR", "N2", rect_um(0, 0, 10, 10), route)])


def collinear_overlap(route: str) -> ModelSpec:
    """Three same-Net rectangles whose edges overlap collinearly (x = -4 and y = 4 each shared by a partial range)."""
    return spec_of("collinear_overlap", route, chip() + [
        metal("padL", "N", rect_um(-14, -6, -4, 4), route), metal("padR", "N", rect_um(-4, 0, 6, 10), route),
        metal("padM", "N", rect_um(-10, 4, 0, 12), route)])


def hole_touches_outer(route: str) -> ModelSpec:
    """A conductor whose diamond hole touches the outer boundary at the single point (10, 0)."""
    diamond = [(10.0, 0.0), (5.0, -5.0), (0.0, 0.0), (5.0, 5.0)]
    return spec_of("hole_touches_outer", route, chip() + [metal("ring", "N", rect_um(-10, -10, 10, 10), route, (diamond,))])


def neck_split_by_inset(route: str) -> ModelSpec:
    """A dumbbell (two 8 um squares joined by a 0.5 um wide neck) whose 0.4 um inset splits at the neck."""
    outline = [(-12, -4), (-4, -4), (-4, -0.25), (4, -0.25), (4, -4), (12, -4), (12, 4), (4, 4), (4, 0.25), (-4, 0.25),
               (-4, 4), (-12, 4)]
    return spec_of("neck_split_by_inset", route, chip() + [metal("dumbbell", "N", outline, route)],
                   band_families(route, outline, (0.1, 0.4)))


def band_vanishes(route: str) -> ModelSpec:
    """A 0.3 um wide strip: the 0.2 um inset is empty (the innermost band does not exist)."""
    outline = rect_um(-10, -0.15, 10, 0.15)
    return spec_of("band_vanishes", route, chip() + [metal("strip", "N", outline, route)],
                   band_families(route, outline, (0.1, 0.2)))


def reflex_small_angle(route: str) -> ModelSpec:
    """A slit with a 0.6 degree opening cut into a conductor: the reflex corner at its apex has a tiny angle."""
    # the slit apex is at (-10, 0): vacuum wedge between (0.2, +-0.05) closing to a point; written as a non-convex ring
    outline = [(-10, -6), (10, -6), (10, 6), (-10, 6), (-10, 0.05), (0.0, 0.0), (-10, -0.05)]
    return spec_of("reflex_small_angle", route, chip() + [metal("slit", "N", outline, route)],
                   band_families(route, outline, (0.05, 0.2)))


def _foreign_vertex_family(route: str) -> tuple[CurveFamilyRecord, ...]:
    """A refinement curve whose end point (4330, 2500) g lies 0.1 g from the circle r = 5000 g."""
    z = z_value(0) if route == "A" else z_value(METAL_THICKNESS)
    carrier = LineCarrier("ref0", (0, 0), (4330, 2500), Lineage("family_member", family_id="ref", member="ref0"))
    return (CurveFamilyRecord("ref", "refinement", (carrier,), (z,), "refinement_only"),)


def arc_with_foreign_vertex(route: str) -> ModelSpec:
    """The arc of a disk carries a vertex that belongs to another curve: the cylinder wall has 5 edges (Route B)."""
    return spec_of("arc_with_foreign_vertex", route, chip() + [disk("disk", "N", (0, 0), 5000, route)], _foreign_vertex_family(route))


def arc_with_foreign_vertex_from_post(route: str) -> ModelSpec:
    """Same, with the foreign vertex coming from the corner of a same-Net post standing on the disk (Route B)."""
    post = metal("post", "N", [(4.330, 2.5), (6.5, 2.5), (6.5, 6.0), (4.330, 6.0)], route, base=METAL_THICKNESS,
                 z=(z_value(METAL_THICKNESS), z_value(3)))
    return spec_of("arc_with_foreign_vertex_from_post", route, chip() + [disk("disk", "N", (0, 0), 5000, route), post])


def floating_conductor(route: str) -> ModelSpec:
    """A conductor enclosed by vacuum: Route B gives the vacuum an inner (void) shell, Route A an embedded sheet."""
    return spec_of("floating_conductor", route, chip() + [metal("float", "floating", rect_um(-6, -4, 6, 4), route, base=3)])


def nested_holes(route: str) -> ModelSpec:
    """Ring (hole +-8) around a ring island (+-6, hole +-3) around an island (+-1): three nested loops, three Nets."""
    return spec_of("nested_holes", route, chip() + [
        metal("outer", "N1", rect_um(-14, -14, 14, 14), route, (rect_um(-8, -8, 8, 8),)),
        metal("middle", "N2", rect_um(-6, -6, 6, 6), route, (rect_um(-3, -3, 3, 3),)),
        metal("core", "N3", rect_um(-1, -1, 1, 1), route)])


def _stack(route: str, name: str, upper: list[Point]) -> ModelSpec:
    lower = metal("lower", "N", rect_um(-12, -6, -2, 4), route, z=(z_value(0), z_value(1)))
    top = metal("upper", "N", upper, route, z=(z_value(1), z_value(2)))
    return spec_of(name, route, chip() + [lower, top])


def stacked_corner_touch(route: str) -> ModelSpec:
    """Two same-Net slabs stacked so that only their corners (-2, 4) meet at z = 1 (a level touching at a vertex)."""
    return _stack(route, "stacked_corner_touch", rect_um(-2, 4, 8, 14))


def stacked_vertex_on_wall_edge(route: str) -> ModelSpec:
    """The upper slab is a triangle whose apex (-2, 0) touches the lower slab's top edge x = -2 at one point."""
    return _stack(route, "stacked_vertex_on_wall_edge", [(-2.0, 0.0), (8.0, -5.0), (8.0, 5.0)])


def port_edge_on_metal_sidewall(route: str) -> ModelSpec:
    """A lumped-port sheet at mid-height of a 0.2 um metal whose edge x = 8 runs along the metal sidewall partly."""
    ground = metal("ground", "G", rect_um(-16, -6, 8, 6), route)
    port_z = z_value(0) if route == "A" else z_value("0.1")
    port = PrismRecord("p1", Footprint.from_rings(to_grid(GRID, rect_um(8, -2, 14, 9))), port_z, port_z, "lumped_port",
                       "dielectric", role="port")
    return spec_of("port_edge_on_metal_sidewall", route, chip() + [ground, port])


def refinement_crossing_band_edge(route: str) -> ModelSpec:
    """A refinement-only line crossing the bands' boundaries of a banded pad (it ends inside the outermost band)."""
    outline = rect_um(-10, -5, 10, 5)
    z = z_value(0) if route == "A" else z_value(METAL_THICKNESS)
    carrier = LineCarrier("ref0", (-9_900, 500), (9_900, 500), Lineage("family_member", family_id="ref", member="ref0"))
    ref = CurveFamilyRecord("ref", "refinement", (carrier,), (z,), "refinement_only")
    return spec_of("refinement_crossing_band_edge", route, chip() + [metal("pad", "N", outline, route)],
                   band_families(route, outline, (0.5, 2.0)) + (ref,))


def _layers_offset(route: str, name: str, offset_um: float) -> ModelSpec:
    lower = metal("lower", "N", rect_um(-10, -4, 0, 4), route, z=(z_value(0), z_value(1)))
    upper = metal("upper", "N", rect_um(-10, -4, offset_um, 4), route, z=(z_value(1), z_value(2)))
    return spec_of(name, route, chip() + [lower, upper])


def near_coincident_layers_sub_grid(route: str) -> ModelSpec:
    return _layers_offset(route, "near_coincident_layers_sub_grid", 0.0004)  # 0.4 g: snaps to the same pixel, no sliver


def near_coincident_layers_2g(route: str) -> ModelSpec:
    return _layers_offset(route, "near_coincident_layers_2g", 0.002)  # 2 g: a real 2 nm exposed sliver of the level z = 1


# --------------------------------------------------------------------------- catalogue


@dataclass(frozen=True)
class Fixture:
    name: str
    risk: str
    build: Callable[[str], ModelSpec]
    routes: tuple[str, ...] = ("A", "B")
    expect_error: dict[str, type[Exception]] = field(default_factory=dict)  # route -> the error the build must raise
    expect_write_refusal: dict[str, str] = field(default_factory=dict)  # route -> text of the error the BREP writer must raise
    mesh_size: float = 4.0
    mesh_size_min: float = 0.0
    note: str = ""


FIXTURES: dict[str, Fixture] = {f.name: f for f in (
    Fixture("baseline", "control: one rectangular pad on the chip", baseline),
    Fixture("near_tangent_arc_line", "near-tangent arc and line (secant 2 nm deep): shallow crossing, thin vacuum cusp",
            near_tangent_arc_line),
    Fixture("exactly_tangent_arc_line", "arc and line touching at one point (same Net)", exactly_tangent_arc_line,
            expect_error={"B": NonManifoldError}, note="Route A: two sheets meeting at a point are legal"),
    Fixture("arcs_crossing_shallow", "two arcs crossing at 1.6 degrees (vacuum cusp between two disks)", arcs_crossing_shallow),
    Fixture("edges_1_to_3_g", "edges 1, 2, 3 g long and a 1.4 g diagonal", edges_1_to_3_g),
    Fixture("kissing_corners_same_net", "vertex-to-vertex contact of two same-Net pads", kissing_corners_same_net,
            expect_error={"B": NonManifoldError}, note="Route A: two sheets sharing a corner are legal"),
    Fixture("kissing_corners_different_nets", "vertex-to-vertex contact of two different Nets", kissing_corners_different_nets,
            expect_error={"A": ContactError, "B": ContactError}),
    Fixture("collinear_overlap", "collinear overlapping edges from different prisms", collinear_overlap),
    Fixture("hole_touches_outer", "a hole touching the outer boundary at one point", hole_touches_outer,
            expect_error={"B": NonManifoldError}, note="Route A: a sheet with a pinched hole is legal"),
    Fixture("neck_split_by_inset", "a neck that the 0.4 um inset splits (topology change of the bands)", neck_split_by_inset),
    Fixture("band_vanishes", "a band that vanishes (strip narrower than twice the margin)", band_vanishes),
    Fixture("reflex_small_angle", "reflex vertex of a 0.6 degree slit with inset bands", reflex_small_angle),
    Fixture("arc_with_foreign_vertex", "arc whose edge carries a foreign hot pixel: 5-edge cylinder wall", arc_with_foreign_vertex,
            expect_write_refusal={"B": "lie on different circles"},
            note="Route B: the direct writer refuses the 5-edge wall (no B-spline record); the gmsh-API oracle writes it"),
    Fixture("arc_with_foreign_vertex_from_post", "same, vertex from a stacked same-Net post", arc_with_foreign_vertex_from_post,
            routes=("B",), expect_write_refusal={"B": "lie on different circles"},
            note="the direct writer refuses the 5-edge wall; the gmsh-API oracle writes it"),
    Fixture("floating_conductor", "inner void shell (B) / embedded sheet (A)", floating_conductor),
    Fixture("nested_holes", "nested holes: an island in a hole in an island in a hole", nested_holes),
    Fixture("stacked_corner_touch", "stacked slabs touching at one vertex of a level", stacked_corner_touch, routes=("B",)),
    Fixture("stacked_vertex_on_wall_edge", "stacked slabs: a level vertex touching a wall's top edge", stacked_vertex_on_wall_edge,
            routes=("B",)),
    Fixture("port_edge_on_metal_sidewall", "port sheet whose edge lies partly on a metal sidewall", port_edge_on_metal_sidewall),
    Fixture("refinement_crossing_band_edge", "refinement curve crossing band boundaries", refinement_crossing_band_edge),
    Fixture("near_coincident_layers_sub_grid", "near-coincident edges of two layers, 0.4 g apart: must merge, no sliver",
            near_coincident_layers_sub_grid, routes=("B",)),
    Fixture("near_coincident_layers_2g", "near-coincident edges of two layers, 2 g apart: a real 2 nm sliver face",
            near_coincident_layers_2g, routes=("B",)),
)}


def cases() -> list[tuple[Fixture, str]]:
    """Every (fixture, route) pair of the catalogue."""
    return [(fixture, route) for fixture in FIXTURES.values() for route in fixture.routes]


# --------------------------------------------------------------------------- runner


class _Timer:
    def __init__(self, row: dict, name: str):
        self.row, self.name = row, name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        self.row["times_s"][self.name] = round(time.perf_counter() - self.t0, 5)


def run_fixture(fixture: Fixture, route: str, workdir: Path | None = None, occt: bool = True, mesh: bool = True) -> dict:
    """Run every check on one fixture and route; the returned row carries a time per check.

    ``row["ok"]`` is true when the outcome is the expected one: all checks clean, or the expected early error.
    """
    from sgbproto.pipeline import build

    spec = fixture.build(route)
    row: dict = {"fixture": fixture.name, "route": route, "risk": fixture.risk, "times_s": {}, "ok": False}
    expected = fixture.expect_error.get(route)
    t0 = time.perf_counter()
    try:
        result = build(spec)
    except Exception as error:  # noqa: BLE001 - the expected refusals are SgbError subclasses; anything else is reported too
        row["times_s"]["build"] = round(time.perf_counter() - t0, 5)
        row["error"] = {"type": type(error).__name__, "message": str(error)}
        row["ok"] = expected is not None and isinstance(error, expected)
        row["expected_error"] = expected.__name__ if expected else None
        return row
    row["times_s"]["build"] = round(time.perf_counter() - t0, 5)
    if expected is not None:
        row["error"] = None
        row["expected_error"] = expected.__name__
        return row
    topo = result.topology
    row["counts"] = {k: result.counts[k] for k in ("prisms", "slabs", "levels", "vertices", "edges", "faces", "volumes", "hot_pixels")
                     if k in result.counts}
    row["structural_validation"] = {"failures": len(result.report.failures), "checks": len(result.report.checks)}
    row["stage_times_s"] = {k: round(v, 5) for k, v in result.timings.items()
                            if k.startswith("validation") or k in ("contact_check", "S2_levels")}
    own = workdir or Path(tempfile.mkdtemp(prefix="sgb_extreme_"))
    own.mkdir(parents=True, exist_ok=True)
    stem = f"{fixture.name}_{route}"
    from sgbproto.brep_writer import write_brep, write_xao
    from sgbproto.errors import SgbError

    refusal = fixture.expect_write_refusal.get(route)
    try:
        with _Timer(row, "write_brep"):
            brep = write_brep(topo, own / f"{stem}.brep", sidecar=False)
        with _Timer(row, "write_xao"):
            xao = write_xao(topo, own / f"{stem}.xao", name="model")
    except SgbError as error:
        row["writer_refused"] = str(error)
        row["ok"] = refusal is not None and refusal in str(error)
        row["oracle"] = _oracle_check(topo, own / f"{stem}_api", fixture, row) if mesh else None
        row["ok"] = row["ok"] and (row["oracle"] is None or row["oracle"]["ok"])
        return row
    if refusal is not None:
        row["ok"] = False
        row["writer_refused"] = None
        return row
    row["plan_counts"] = brep.plan.counts
    row["files_bytes"] = {"brep": brep.bytes_written, "xao": xao.bytes_written}
    supports = {f.family_id: f.support for f in spec.families if f.support is not None}
    row["gmsh"] = {kind: _gmsh_check(topo, written, fixture, supports if supports else None, mesh, kind == "xao", row, kind)
                   for kind, written in (("brep", brep), ("xao", xao))}
    if occt:
        row["occt"] = {kind: _occt_check(written, topo, row, kind) for kind, written in (("brep", brep), ("xao", xao))}
    row["ok"] = all(g["ok"] for g in row["gmsh"].values()) and all(o["ok"] for o in row.get("occt", {}).values())
    return row


def _oracle_check(topo, folder: Path, fixture: Fixture, row: dict) -> dict:
    """The gmsh-API emitter (the test oracle) on a model the direct writer refuses: it must still mesh conformally."""
    from sgbproto.emit import emit

    t0 = time.perf_counter()
    result = emit(topo, folder, "model", mesh_size=fixture.mesh_size, mesh_size_min=fixture.mesh_size_min)
    row["times_s"]["oracle_emit_and_mesh"] = round(time.perf_counter() - t0, 5)
    mesh = getattr(result, "mesh", None) or {}
    return {"ok": bool(mesh.get("conformal", False)), "mesh_conformal": mesh.get("conformal"),
            "wall_deviation": getattr(result, "wall_deviation", None)}


def _gmsh_check(topo, written, fixture: Fixture, supports, mesh: bool, groups: bool, row: dict, kind: str) -> dict:
    import gmsh

    from sgbproto.brep_writer import verify_loaded

    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
        t0 = time.perf_counter()
        (gmsh.open if written.path.suffix == ".xao" else gmsh.merge)(str(written.path))
        gmsh.model.occ.synchronize()
        row["times_s"][f"gmsh_load_{kind}"] = round(time.perf_counter() - t0, 5)
        t0 = time.perf_counter()
        report = verify_loaded(topo, written.plan, mesh_size=fixture.mesh_size if mesh else None,
                               mesh_size_min=fixture.mesh_size_min, groups=groups, supports=supports)
        total = time.perf_counter() - t0
    finally:
        gmsh.finalize()
    mesh_s = report.get("mesh", {}).get("mesh_seconds", 0.0)
    row["times_s"][f"gmsh_verify_{kind}"] = round(total - mesh_s, 5)
    if mesh:
        row["times_s"][f"gmsh_mesh_{kind}"] = round(mesh_s, 5)
    out = {"ok": report["ok"], "problems": report["problems"][:3]}
    if mesh:
        out["conformal"] = report["mesh"]["conformal"]
        out["mesh_problems"] = report["mesh"].get("problems", [])[:3]
        out["tets"] = report["mesh"].get("tetrahedra")
    return out


def _occt_check(written, topo, row: dict, kind: str) -> dict:
    from sgbproto.occt_check import check_file

    report = check_file(written.path, topo, written.plan, self_interference=True)
    for name, seconds in report.timings.items():
        row["times_s"][f"occt_{name}_{kind}"] = round(seconds, 5)
    return {"ok": report.ok, "problems": report.problems[:4], "counts": report.counts,
            "brepcheck_faulty": report.brepcheck["faulty_count"], "bop_results": report.arguments["result_count"],
            "bop_faulty": report.arguments["faulty"], "shape_analysis": {k: report.shape_analysis[k] for k in (
                "shells_with_free_edges", "shells_with_bad_edges", "edges_below_10x_tolerance")},
            "plan_mismatches": len(report.properties.get("mismatches", []))}


def band_klayout_rows(fixture: Fixture, route: str) -> list[dict]:
    """KLayout erosion cross-check of every inset family of a banded fixture (exact band areas vs the oracle)."""
    from sgbproto.klayout_check import comparison_rows
    from sgbproto.pipeline import build

    spec = fixture.build(route)
    result = build(spec)
    rows = []
    for family in spec.families:
        if family.support is not None:
            level = result.levels.levels.index(family.target_levels[0])
            rows += comparison_rows(result, family.support, family.family_id, level)
    return rows
