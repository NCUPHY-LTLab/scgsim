"""Figures and tables for ``docs/design/sgb/inset-imprint.qmd``.

Run, with the Python of the checkout that contains the production SGB core (``scgsim.geometry.sgb``)::

    SGB_CORE=<sgb-core> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/inset.py

Every geometry figure is computed by the production package: the pad-in-ground running example
(``tests/sgb_scenes.py::pad_in_ground``, its lumped port on the support plane z = 0) is built with
Surface-EPR inset bands through ``build_model_from_catalogue`` and read from the labelled level arrangement,
the assembled topology and the route plan.  The KLayout cross-check uses the test-side oracle
``tests/sgb_klayout.py``; the mesh check meshes the written XAO with Gmsh (no Palace and no AEDT is run).
Outputs go to ``docs/design/sgb/figures/`` with the prefix ``inset-``:

* ``inset-erosion-construction``  corner rule on large margins (computed with margins 1.5, 3, 6 um)
* ``inset-bands-pad``             the z = 0 level arrangement with bands (margins 0.1, 0.5, 2 um); html + svg
* ``inset-arcs-vs-polygon``       exact offset arcs against a 512-gon erosion, radial deviation; html + svg
* ``inset-topology-change``       margins 4, 9, 30 um: a neck splits, a band vanishes; html + svg
* ``inset-mask-vs-imprint``       an element cut by a mask against a mesh aligned to the band edge, and the
                                  area error of both as the element size shrinks (2-D gmsh illustration)
* ``inset-routes-section``        conceptual cross-section of Route A (one sheet) and Route B (top and bottom faces)
* ``inset-tables.md``             markdown tables quoted by the page (band areas, KLayout cross-check, mesh check)
"""

from __future__ import annotations

import json
import math
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
import numpy as np  # noqa: E402
from inset_kit import G_UM as G
from inset_kit import MARGINS, Line, Panel, Poly, Pts, primitive_polyline  # noqa: E402
from scgsim.geometry.sgb.arrangement.arcs import face_exact_area  # noqa: E402
from scgsim.geometry.sgb.build import emit_model  # noqa: E402
from scgsim.geometry.sgb.emission.plan import plan_brep  # noqa: E402
from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402
from scgsim.geometry.sgb.sidecar.files import read_sidecar  # noqa: E402
from scgsim.geometry.sgb.verification.loaded import verify_loaded  # noqa: E402
from sgb_extreme_cases import FIXTURES, build_model  # noqa: E402
from sgb_gmsh_oracle import band_mesh_report, gmsh_session, mesh_coarsely, refine_near  # noqa: E402
from sgb_klayout import comparison_rows, eroded_klayout, region_of_rings  # noqa: E402
from sgb_scenes import exact_band_areas  # noqa: E402

LEVEL_Z0 = kit.LEVEL_Z0
MEMBER_MARGIN = re.compile(r"@m(\d+(?:\.\d+)?)$")

VIEWS = {
    "whole chip": ((-155, 155), (-155, 155)),
    "cutout": ((-85, 85), (-65, 65)),
    "pad": ((-55, 55), (-35, 35)),
    "reflex notch": ((-16, -4), (-21, -9)),
    "rounded corner": ((-52, -40), (-32, -20)),
    "hole": ((-35, -15), (-5, 15)),
}


# --------------------------------------------------------------------------- geometry helpers


def region_boundary_lines(region, width: float = 2.2) -> list[Line]:
    """The support region's own boundary, which is the only distance origin."""
    out = []
    for ring in region.rings:
        pts = [p for prim in ring for p in primitive_polyline(prim)]
        out.append(
            Line(
                [p[0] for p in pts],
                [p[1] for p in pts],
                kit.INK,
                width,
                name="boundary of R (distance origin)",
            )
        )
    return out


def member_margin(carrier: str) -> float | None:
    """The margin (grid units) a family-member carrier id ends with (``band_pad_top/ring0/edge9@m500.0``), else None."""
    match = MEMBER_MARGIN.search(carrier)
    return float(match[1]) if match else None


def clip_runs(points, xr, yr, pad: float = 1.0):
    """Contiguous runs of a polyline that lie inside a view window (plus ``pad`` um); keeps figure files small."""
    runs, run = [], []
    for x, y in points:
        if xr[0] - pad <= x <= xr[1] + pad and yr[0] - pad <= y <= yr[1] + pad:
            run.append((x, y))
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return [r for r in runs if len(r) > 1]


def face_style(label) -> str:
    """Fill colour of a level cell of the running example."""
    if label.sheet is not None and label.sheet.kind == "port":
        return kit.PORT
    if label.sheet is not None and label.bands:
        return kit.PAD_BANDS[label.bands[0][1]]
    if label.sheet is not None:
        return kit.GROUND
    if label.bands:
        return kit.GAP_BANDS[label.bands[0][1]]
    return kit.NEUTRAL


def level_polys(model, level_index: int = LEVEL_Z0) -> list[Poly]:
    """One polygon per cell of a level arrangement, hover text = the cell's physical group and face id."""
    layer = model.labelled.level_layers[level_index]
    polys = []
    for fi in range(1, len(layer.arr.faces)):
        fid = layer.face_ids[fi]
        group = model.routed.face_group.get(fid) or "(no face)"
        polys.append(
            Poly(
                kit.face_rings_um(layer, fi),
                fill=face_style(layer.labels[fi]),
                edge="rgba(0,0,0,0.6)",
                edge_width=0.4,
                hover=f"{group}<br>{fid}",
            )
        )
    return polys


def support_of(model, family_id: str):
    """The support region of one inset family."""
    return next(
        f.support for f in model.labelled.catalogue.families if f.family_id == family_id
    )


# --------------------------------------------------------------------------- 1. erosion construction


def figure_construction() -> None:
    """Corner rule on margins large enough to see (1.5, 3 and 6 um on the 5 um rounded corner)."""
    margins = (1.5, 3.0, 6.0)
    model = kit.pad_model("A", margins, with_port=False, regions=("pad",))
    layer = model.labelled.level_layers[LEVEL_Z0]
    region = support_of(model, "band_pad_top")
    colours = {1500: "#1b7837", 3000: "#e08214", 6000: "#762a83"}
    names = {1500: "m = 1.5 um", 3000: "m = 3 um", 6000: "m = 6 um"}

    def curves(m: int) -> list[Line]:
        return [
            Line(
                [p[0] for p in poly],
                [p[1] for p in poly],
                colours[m],
                2.0,
                name=f"offset curve, {names[m]}",
            )
            for poly in kit.layer_edge_polylines(
                layer,
                lambda src, m=m: any(
                    s.startswith("band_pad_top/") and member_margin(s) == m for s in src
                ),
            )
        ]

    def circle(cx, cy, r, colour, name="") -> Line:
        t = np.linspace(0, 2 * math.pi, 181)
        return Line(
            list(cx + r * np.cos(t)),
            list(cy + r * np.sin(t)),
            colour,
            0.9,
            "dot",
            name=name,
        )

    panels = []
    boundary = region_boundary_lines(region)
    # reflex corner (the notch vertex): arcs of radius m about the vertex
    items = list(boundary) + [c for m in colours for c in curves(m)]
    items += [
        circle(-10, -15, m / 1000, colours[m], "disc D(m) about the vertex")
        for m in colours
    ]
    items += [Pts([-10], [-15], kit.INK, 7, name="reflex vertex", symbol="x")]
    panels.append(
        Panel(
            "reflex corner: offset lines leave a gap, bridged by an arc of radius m",
            (-19, -1),
            (-24, -6),
            items=items,
        )
    )
    # convex corner (chamfer): the offset lines cross and the corner stays sharp
    items = list(boundary) + [c for m in colours for c in curves(m)]
    items += [Pts([50, 38], [18, 30], kit.INK, 7, name="convex vertices", symbol="x")]
    panels.append(
        Panel(
            "convex corner (chamfer): offset lines cross, corner stays sharp",
            (30, 53),
            (6, 33),
            items=items,
        )
    )
    # convex arc: rho -> rho - m, vanishes for m >= rho
    items = list(boundary) + [c for m in colours for c in curves(m)]
    items += [Pts([-45], [-25], kit.INK, 6, name="arc centre", symbol="x")]
    panels.append(
        Panel(
            "convex arc: radius rho becomes rho - m, and vanishes when m >= rho",
            (-53, -33),
            (-33, -13),
            items=items,
        )
    )
    # hole
    items = list(boundary) + [c for m in colours for c in curves(m)]
    items += [Pts([-25], [5], kit.INK, 6, name="hole centre", symbol="x")]
    panels.append(
        Panel(
            "hole of radius r = 8 um: concentric circle of radius r + m",
            (-42, -8),
            (-12, 22),
            items=items,
        )
    )
    kit.write_both(
        panels,
        "inset-erosion-construction",
        ncols=2,
        svg_size=(10, 8.4),
        html_size=(1100, 900),
        title="Boundary of the eroded region R minus D(m) for m = 1.5, 3, 6 um (computed, production inset generator)",
        shared_legend=True,
    )


# --------------------------------------------------------------------------- 2. bands on the running example


def figure_bands(model_a) -> None:
    polys = level_polys(model_a)
    legend = [
        Pts(
            [-1e4],
            [-1e4],
            kit.PAD_BANDS[0],
            8,
            name="pad::band=0  (< 0.1 um from the edge)",
            symbol="square",
        ),
        Pts(
            [-1e4],
            [-1e4],
            kit.PAD_BANDS[1],
            8,
            name="band 1  (0.1-0.5 um)",
            symbol="square",
        ),
        Pts(
            [-1e4],
            [-1e4],
            kit.PAD_BANDS[2],
            8,
            name="band 2  (0.5-2 um)",
            symbol="square",
        ),
        Pts(
            [-1e4],
            [-1e4],
            kit.PAD_BANDS[3],
            8,
            name="band 3  (>= 2 um)",
            symbol="square",
        ),
        Pts(
            [-1e4],
            [-1e4],
            kit.GAP_BANDS[0],
            8,
            name="SA::substrate|vacuum::interface::band=0..3 (gap, orange ramp)",
            symbol="square",
        ),
        Pts(
            [-1e4],
            [-1e4],
            kit.PORT,
            8,
            name="PORT::p1 (z = 0, on the substrate surface)",
            symbol="square",
        ),
        Pts(
            [-1e4],
            [-1e4],
            kit.GROUND,
            8,
            name="ground (Route A sheet)",
            symbol="square",
        ),
    ]
    one = Panel(
        "z = 0 level arrangement of the Route A running example, coloured by label and band",
        VIEWS["pad"][0],
        VIEWS["pad"][1],
        xlabel="x (um)",
        ylabel="y (um)",
        items=polys + legend,
    )
    kit.render_html(
        [one],
        kit.FIGURES / "inset-bands-pad.html",
        size=(1000, 720),
        buttons=VIEWS,
        title="Pad-in-ground, Route A: cells of the z = 0 level coloured by physical group (margins 0.1, 0.5, 2 um)",
    )
    static = []
    for name in ("pad", "reflex notch", "hole", "rounded corner"):
        xr, yr = VIEWS[name]
        static.append(
            Panel(
                f"{name}",
                xr,
                yr,
                xlabel="x (um)",
                ylabel="y (um)",
                items=polys + legend[:6],
            )
        )
    kit.render_svg(
        static,
        kit.FIGURES / "inset-bands-pad.svg",
        ncols=2,
        size=(10, 8),
        suptitle="z = 0 level of the Route A running example: bands of the pad sheet (blue) and of the gap (orange); port red",
        shared_legend=True,
    )
    print("wrote inset-bands-pad")


# --------------------------------------------------------------------------- 3. exact arcs vs polygon


def klayout_rings(region, margin_um: float) -> list[list[tuple[float, float]]]:
    eroded = eroded_klayout(region_of_rings(region), round(margin_um / G))
    rings = []
    for poly in eroded.each():
        rings.append([(p.x * G, p.y * G) for p in poly.each_point_hull()])
        for h in range(poly.holes()):
            rings.append([(p.x * G, p.y * G) for p in poly.each_point_hole(h)])
    return rings


def figure_arcs(model_a) -> None:
    layer = model_a.labelled.level_layers[LEVEL_Z0]
    region = support_of(model_a, "band_pad_top")
    exact_curves = kit.layer_edge_polylines(
        layer, lambda src: any(s.startswith("band_pad_top/") for s in src)
    )
    rings = {m: klayout_rings(region, m) for m in MARGINS}
    features = [
        (
            "reflex notch corner",
            ((-12.5, -7.5), (-17.5, -12.5)),
            (-10.0, -15.0),
            lambda m: m,
        ),
        (
            "rounded corner (rho = 5 um)",
            ((-51, -39), (-31, -19)),
            (-45.0, -25.0),
            lambda m: 5.0 - m,
        ),
        ("hole (r = 8 um)", ((-35, -15), (-5, 15)), (-25.0, 5.0), lambda m: 8.0 + m),
    ]
    top, bottom = [], []
    for title, (xr, yr), centre, radius_of in features:
        items = [
            Line(
                [p[0] for p in run],
                [p[1] for p in run],
                kit.EXACT,
                2.2,
                name="exact carriers (arcs realised)",
            )
            for poly in exact_curves
            for run in clip_runs(poly, xr, yr)
        ]
        for m in MARGINS:
            for ring in rings[m]:
                for run in clip_runs(ring + [ring[0]], xr, yr):
                    items.append(
                        Line(
                            [p[0] for p in run],
                            [p[1] for p in run],
                            kit.OTHER,
                            1.0,
                            "dot",
                            name="KLayout erosion by a 512-gon",
                        )
                    )
        top.append(Panel(title, xr, yr, items=items, xlabel="x (um)", ylabel="y (um)"))
        dev_items = []
        for m in MARGINS:
            r = radius_of(m)
            if r <= 0:
                continue

            def deviation(points):
                data = [
                    (
                        math.degrees(math.atan2(y - centre[1], x - centre[0])),
                        (math.hypot(x - centre[0], y - centre[1]) - r) * 1000,
                    )
                    for x, y in points
                    if abs(math.hypot(x - centre[0], y - centre[1]) - r) < 0.02
                ]
                return data[:: max(1, len(data) // 300)]

            ex = deviation([p for poly in exact_curves for p in poly])
            kl = deviation([p for ring in rings[m] for p in ring])
            dev_items.append(
                Pts(
                    [d[0] for d in ex],
                    [d[1] for d in ex],
                    kit.EXACT,
                    3,
                    name="exact carriers",
                    hover=f"m = {m} um",
                )
            )
            dev_items.append(
                Pts(
                    [d[0] for d in kl],
                    [d[1] for d in kl],
                    kit.OTHER,
                    3,
                    name="512-gon (integer-rounded vertices)",
                    hover=f"m = {m} um",
                )
            )
        bottom.append(
            Panel(
                "radial deviation from the exact radius (nm)",
                None,
                None,
                equal=False,
                xlabel="angle about the centre (deg)",
                ylabel="deviation (nm)",
                items=dev_items,
            )
        )
    kit.write_both(
        top + bottom,
        "inset-arcs-vs-polygon",
        ncols=3,
        svg_size=(14, 8),
        html_size=(1300, 800),
        title="Exact offset arcs against a 512-gon erosion, margins 0.1, 0.5, 2 um (computed)",
        shared_legend=True,
    )


# --------------------------------------------------------------------------- 4. topology change


def figure_topology_change() -> dict:
    margins = (4.0, 9.0, 30.0)
    model = kit.pad_model("A", margins, with_port=False, regions=("pad",))
    layer = model.labelled.level_layers[LEVEL_Z0]
    palette = {0: "#08306b", 1: "#4292c6", 2: "#fdae61", 3: "#d9d9d9"}
    polys, areas = [], {}
    for fi in range(1, len(layer.arr.faces)):
        lab = layer.labels[fi]
        if lab.sheet is None or not lab.bands:
            continue
        k = lab.bands[0][1]
        area = face_exact_area(layer.arr, layer.edge_curve, fi) * 1e-6
        areas.setdefault(k, []).append(area)
        polys.append(
            Poly(
                kit.face_rings_um(layer, fi),
                palette[k],
                "rgba(0,0,0,0.6)",
                0.5,
                hover=f"band {k}: {area:.2f} um^2",
            )
        )
    legend = [
        Pts(
            [-1e4],
            [-1e4],
            palette[0],
            8,
            name="B0: closer than 4 um to the edge",
            symbol="square",
        ),
        Pts([-1e4], [-1e4], palette[1], 8, name="B1: 4-9 um", symbol="square"),
        Pts(
            [-1e4],
            [-1e4],
            palette[2],
            8,
            name="B2: 9-30 um (three separate faces)",
            symbol="square",
        ),
        Pts(
            [-1e4],
            [-1e4],
            palette[3],
            8,
            name="B3: at least 30 um (empty)",
            symbol="square",
        ),
    ]
    panel = Panel(
        "pad top, margins 4, 9 and 30 um",
        (-55, 55),
        (-35, 35),
        xlabel="x (um)",
        ylabel="y (um)",
        items=polys + legend,
    )
    kit.write_both(
        [panel],
        "inset-topology-change",
        svg_size=(9, 6.2),
        html_size=(1000, 650),
        shared_legend=True,
        title="Erosion splits the pad into three islands at 9 um; nothing is 30 um from the edge",
    )
    # connected components of R_m by KLayout (the oracle), at the margins of the page's table
    support = region_of_rings(support_of(model, "band_pad_top"))
    components = {
        m: eroded_klayout(support, round(m * 1000)).count()
        for m in (0, 4, 9, 21, 23, 30)
    }
    return {
        "margins_um": list(margins),
        "band_faces": {str(k): len(v) for k, v in sorted(areas.items())},
        "band_face_areas_um2": {str(k): sorted(v) for k, v in sorted(areas.items())},
        "bands_present": sorted(areas),
        "klayout_components_of_R_m": {str(m): c for m, c in components.items()},
    }


# --------------------------------------------------------------------------- 5. mask vs imprint (2-D gmsh illustration)

RADIUS = 0.5  # mask boundary: circle of this radius; masked region = the square minus the disc


def disc_mesh(h: float, aligned: bool):
    """Triangles of the square [-1, 1]^2 with element size ``h``; ``aligned`` makes the circle a mesh boundary."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("mask")
        geo = gmsh.model.geo
        corners = [
            geo.addPoint(x, y, 0) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        outer = geo.addCurveLoop(
            [geo.addLine(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        )
        inside_tag = None
        if aligned:
            c = geo.addPoint(0, 0, 0)
            q = [
                geo.addPoint(RADIUS * math.cos(a), RADIUS * math.sin(a), 0)
                for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)
            ]
            arcs = [geo.addCircleArc(q[i], c, q[(i + 1) % 4]) for i in range(4)]
            ring = geo.addCurveLoop(arcs)
            outside_tag = geo.addPlaneSurface([outer, ring])
            inside_tag = geo.addPlaneSurface([ring])
        else:
            outside_tag = geo.addPlaneSurface([outer])
        geo.synchronize()
        for key, value in (
            ("Mesh.MeshSizeMax", h),
            ("Mesh.MeshSizeMin", 0.0),
            ("Mesh.MeshSizeFromPoints", 0),
            ("Mesh.MeshSizeFromCurvature", 0),
            ("Mesh.MeshSizeExtendFromBoundary", 0),
            ("Mesh.Algorithm", 6),
        ):
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.generate(2)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        index = {int(t): i for i, t in enumerate(node_tags)}
        xy = np.asarray(coords).reshape(-1, 3)[:, :2]

        def triangles(tag):
            _, _, nodes = gmsh.model.mesh.getElements(2, tag)
            return (
                np.array(
                    [
                        [index[int(n)] for n in tri]
                        for tri in np.asarray(nodes[0]).reshape(-1, 3)
                    ]
                )
                if len(nodes)
                else np.zeros((0, 3), int)
            )

        return (
            xy,
            triangles(outside_tag),
            (triangles(inside_tag) if inside_tag else None),
        )
    finally:
        gmsh.finalize()


def tri_areas(xy, tri):
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    return 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )


SHIFTS = [
    (0.0, 0.0),
    (0.011, 0.007),
    (-0.017, 0.013),
    (0.023, -0.019),
    (-0.007, -0.029),
    (0.031, 0.002),
    (-0.013, 0.021),
    (0.005, -0.011),
]


def sampled_mask_area(xy, tri, area, centre) -> tuple[float, float]:
    """Masked area (square minus disc about ``centre``) when the mask is sampled at 1 and at 3 points per element."""
    cx, cy = centre
    mid = (
        xy[tri[:, [0, 1]]].mean(axis=1),
        xy[tri[:, [1, 2]]].mean(axis=1),
        xy[tri[:, [2, 0]]].mean(axis=1),
    )
    centroid = xy[tri].mean(axis=1)
    one = (np.hypot(centroid[:, 0] - cx, centroid[:, 1] - cy) >= RADIUS).astype(float)
    three = (
        sum((np.hypot(m[:, 0] - cx, m[:, 1] - cy) >= RADIUS).astype(float) for m in mid)
        / 3.0
    )
    return float((area * one).sum()), float((area * three).sum())


def mask_metrics(h: float) -> dict:
    """Masked area (square minus disc) from a mesh the mask cuts (several circle positions), and from a mesh aligned to the contour."""
    exact = 4.0 - math.pi * RADIUS**2
    xy, tri, _ = disc_mesh(h, aligned=False)
    area = tri_areas(xy, tri)
    radial = np.hypot(xy[:, 0], xy[:, 1])
    side = radial[tri] - RADIUS
    cut = int(np.sum((side.min(axis=1) < -1e-9) & (side.max(axis=1) > 1e-9)))
    one, three = zip(*(sampled_mask_area(xy, tri, area, shift) for shift in SHIFTS))
    err1 = [abs(v - exact) / exact for v in one]
    err3 = [abs(v - exact) / exact for v in three]
    xy2, tri_out, tri_in = disc_mesh(h, aligned=True)
    radial2 = np.hypot(xy2[:, 0], xy2[:, 1])
    side2 = radial2[tri_out] - RADIUS
    side_in = radial2[tri_in] - RADIUS
    crossing = int(
        np.sum((side2.min(axis=1) < -1e-6) & (side2.max(axis=1) > 1e-6))
    ) + int(np.sum((side_in.max(axis=1) > 1e-6)))
    aligned_area = float(tri_areas(xy2, tri_out).sum())
    return {
        "h": h,
        "exact": exact,
        "cut_triangles": cut,
        "triangles_unaligned": int(len(tri)),
        "err_one_point_median": float(np.median(err1)),
        "err_one_point_max": max(err1),
        "err_one_point_min": min(err1),
        "err_three_point_median": float(np.median(err3)),
        "err_three_point_max": max(err3),
        "err_three_point_min": min(err3),
        "crossing_aligned": crossing,
        "triangles_aligned": int(len(tri_out) + len(tri_in)),
        "err_aligned": abs(aligned_area - exact) / exact,
    }


def figure_mask_vs_imprint() -> list[dict]:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    h = 0.16
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.9), gridspec_kw={"width_ratios": [1, 1, 1.15]}
    )
    fig.patch.set_facecolor("white")
    xy, tri, _ = disc_mesh(h, aligned=False)
    radial = np.hypot(xy[:, 0], xy[:, 1]) - RADIUS
    side = radial[tri]
    is_cut = (side.min(axis=1) < 0) & (side.max(axis=1) > 0)
    centroid = xy[tri].mean(axis=1)
    kept = np.hypot(centroid[:, 0], centroid[:, 1]) >= RADIUS
    colours = np.where(is_cut, kit.ACCENT, np.where(kept, kit.PAD_BANDS[3], "#ffffff"))
    ax = axes[0]
    ax.add_collection(
        PolyCollection(xy[tri], facecolors=colours, edgecolors="#555", linewidths=0.4)
    )
    t = np.linspace(0, 2 * math.pi, 400)
    ax.plot(RADIUS * np.cos(t), RADIUS * np.sin(t), color=kit.INK, lw=1.6)
    ax.set_title(
        f"mask evaluated on a mesh that does not know it:\n{int(is_cut.sum())} of {len(tri)} triangles straddle the contour (orange)",
        fontsize=9,
    )
    xy2, tri_out, tri_in = disc_mesh(h, aligned=True)
    ax = axes[1]
    ax.add_collection(
        PolyCollection(
            xy2[tri_out], facecolors=kit.PAD_BANDS[3], edgecolors="#555", linewidths=0.4
        )
    )
    ax.add_collection(
        PolyCollection(
            xy2[tri_in], facecolors="#ffffff", edgecolors="#555", linewidths=0.4
        )
    )
    ax.plot(RADIUS * np.cos(t), RADIUS * np.sin(t), color=kit.INK, lw=1.6)
    ax.set_title(
        f"contour is a mesh boundary (an imprinted band edge):\n0 of {len(tri_out) + len(tri_in)} triangles straddle it",
        fontsize=9,
    )
    for ax in axes[:2]:
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    rows = [mask_metrics(x) for x in (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125)]
    ax = axes[2]
    hs = np.array([r["h"] for r in rows])
    for key, colour, marker, label in (
        ("one_point", kit.ACCENT, "o", "mask sampled at 1 point per element"),
        ("three_point", kit.OTHER, "s", "mask sampled at 3 points per element"),
    ):
        ax.loglog(
            hs,
            [r[f"err_{key}_median"] for r in rows],
            marker + "-",
            color=colour,
            label=label + " (median of 8 contour positions)",
        )
        ax.fill_between(
            hs,
            [r[f"err_{key}_min"] for r in rows],
            [r[f"err_{key}_max"] for r in rows],
            color=colour,
            alpha=0.15,
            linewidth=0,
        )
    ax.loglog(
        hs,
        [r["err_aligned"] for r in rows],
        "^-",
        color=kit.EXACT,
        lw=2,
        label="contour is a mesh boundary (sum over band-group faces)",
    )
    ax.loglog(hs, 0.03 * (hs / 0.4) ** 2, color="#777", ls=":", lw=1, label="slope 2")
    ax.loglog(hs, 0.03 * (hs / 0.4), color="#777", ls="--", lw=1, label="slope 1")
    ax.set_xticks(hs)
    ax.set_xticklabels([f"{x:g}" for x in hs], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("element size h", fontsize=8)
    ax.set_ylabel("relative error of the masked area", fontsize=8)
    ax.set_title(
        "masked area of a surface against a circular contour\n(first-order elements; exact value 4 - pi r^2, r = 0.5)",
        fontsize=9,
    )
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.3, which="major")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    kit.save_svg(fig, "inset-mask-vs-imprint")
    (kit.FIGURES / "inset-mask-vs-imprint.json").write_text(json.dumps(rows, indent=1))
    print("wrote inset-mask-vs-imprint")
    return rows


# --------------------------------------------------------------------------- 6. conceptual cross-section of the routes


def figure_routes_section() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(-2.2, 3.4)
        ax.axis("off")
        ax.add_patch(Rectangle((0, -2), 10, 2, facecolor="#e5e5e5", edgecolor="none"))
        ax.text(0.15, -1.2, "substrate", fontsize=8)
        ax.text(0.15, 2.7, "vacuum", fontsize=8)
    # Route A: a sheet; bands lie on the sheet; one group serves the MA and the MS row
    ax = axes[0]
    ax.set_title("Route A: zero-thickness sheet", fontsize=9)
    edges = [2, 2.5, 3.5, 6.5, 7.5, 8]
    for i in range(len(edges) - 1):
        k = min(i, 4 - i)
        ax.plot(
            [edges[i], edges[i + 1]],
            [0, 0],
            color=kit.PAD_BANDS[k],
            lw=7,
            solid_capstyle="butt",
        )
    ax.text(5, 0.65, "one face per band: pad::band=k", fontsize=8, ha="center")
    ax.text(
        5,
        0.2,
        "(the sheet is metal-vacuum and metal-substrate at once)",
        fontsize=7,
        ha="center",
        color="#555",
    )
    ax.annotate(
        "MA row and MS row both\nread these band groups",
        (2.2, 0),
        (0.3, 1.5),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    # Route B: a finite prism; the top face and bottom face each carry bands
    ax = axes[1]
    ax.set_title(
        "Route B: finite metal (excluded volume), top and bottom faces banded separately",
        fontsize=9,
    )
    ax.add_patch(
        Rectangle(
            (2, 0), 6, 0.6, facecolor="white", edgecolor=kit.INK, lw=0.8, hatch="//"
        )
    )
    for i in range(len(edges) - 1):
        k = min(i, 4 - i)
        ax.plot(
            [edges[i], edges[i + 1]],
            [0.6, 0.6],
            color=kit.PAD_BANDS[k],
            lw=6,
            solid_capstyle="butt",
        )
        ax.plot(
            [edges[i], edges[i + 1]],
            [0.0, 0.0],
            color=kit.PAD_BANDS[k],
            lw=6,
            solid_capstyle="butt",
        )
    ax.text(5, 1.2, "top face: MA::pad::top::band=k", fontsize=8, ha="center")
    ax.text(5, -0.55, "bottom face: MS::pad::bottom::band=k", fontsize=8, ha="center")
    ax.text(
        8.25, 0.3, "side walls:\nnot banded in v1", fontsize=7, ha="left", color="#555"
    )
    fig.tight_layout()
    kit.save_svg(fig, "inset-routes-section")
    print("wrote inset-routes-section")


# --------------------------------------------------------------------------- the mesh check (Gmsh only; no Palace is run)


def mesh_check(model, size: float) -> dict:
    """Write a model, mesh it with a size field on the band curves, and check every band group's triangles.

    The check of the production test ``test_band_faces_through_the_direct_writer``: each triangle of a band group is a
    face of one tetrahedron on each side and its centroid lies in the band; the whole mesh is conformal.  Mesh size
    ``size``, 1 um on the band curves growing to ``size`` over 1 to 20 um.
    """
    import gmsh
    from sgb_gmsh_oracle import check_mesh_conformity

    with tempfile.TemporaryDirectory(prefix="inset-mesh-") as scratch:
        artifacts = emit_model(model, Path(scratch))
        sidecar = read_sidecar(artifacts.sidecar)
        plan = plan_brep(model.topology, model.routed)
        tags = plan.tags()
        supports = {
            f.family_id: f.support
            for f in model.labelled.catalogue.families
            if f.support
        }
        with gmsh_session(artifacts.xao):
            loaded = verify_loaded(sidecar)
            assert loaded.ok, loaded
            for name, group in sidecar[
                "physical_groups"
            ].items():  # the loaded area of every band group equals the plan
                if "::band=" in name:
                    area = sum(
                        gmsh.model.occ.getMass(2, t) for t in group["native_tags"]
                    )
                    assert abs(area - group["area_um2"]) <= 1e-9 * group["area_um2"], (
                        name
                    )
            refine_near(
                model,
                tags,
                ("band_pad_top", "band_pad_bottom", "band_gap"),
                size,
                1.0,
                (1.0, 20.0),
            )
            mesh_coarsely(size)
            bands = band_mesh_report(model, tags, supports)
            conformity = check_mesh_conformity(model, tags)
    return {
        "size_um": size,
        "nodes": conformity["nodes"],
        "tets": conformity["tets"],
        "duplicate_nodes": conformity["duplicate_nodes"],
        "conformal": conformity["conformal"],
        "bands": bands,
    }


# --------------------------------------------------------------------------- tables for the page


def group_face_areas(model, level_index: int = LEVEL_Z0) -> dict[str, list[float]]:
    """Area (um^2) of every face of the level arrangement, by primary group (the faces of a group, largest first)."""
    layer = model.labelled.level_layers[level_index]
    out: dict[str, list[float]] = {}
    for fi in range(1, len(layer.arr.faces)):
        group = model.routed.face_group.get(layer.face_ids[fi])
        if group:
            out.setdefault(group, []).append(
                face_exact_area(layer.arr, layer.edge_curve, fi) * 1e-6
            )
    return {g: sorted(v, reverse=True) for g, v in out.items()}


def build_tables(
    model_a, topology_change: dict, mask_rows: list[dict], meshes: dict[str, dict]
) -> None:
    areas = group_face_areas(model_a)
    out = [
        "## Band areas (computed by inset.py from the production Route A build, margins 0.1, 0.5, 2 um; areas of the faces of each band group)",
        "",
        "| support surface | band | group | faces | area (um^2) | R_m = bands >= k (um^2) |",
        "|---|---|---|---|---|---|",
    ]
    for prefix, label in (
        ("pad::band=", "pad top"),
        ("SA::substrate|vacuum::interface::band=", "gap"),
    ):
        per_band = {k: areas[f"{prefix}{k}"] for k in range(4)}
        total = {k: sum(v) for k, v in per_band.items()}
        for k in range(4):
            out.append(
                f"| {label} | B{k} | `{prefix}{k}` | {len(per_band[k])} | {total[k]:.4f} | {sum(a for b, a in total.items() if b >= k):.4f} |"
            )
    port = areas["PORT::p1"]
    out += [
        "",
        f"PORT::p1: {len(port)} faces, {sum(port):.4f} um^2 (the port footprint takes this area out of the gap bands)",
    ]
    # the exact band areas by family (cells the band groups carry; the port face is no band's surface)
    out += [
        "",
        "## Band areas by family (cells the band groups carry, the port face left out)",
        "",
        "| family | band | area (um^2) |",
        "|---|---|---|",
    ]
    for family in ("band_pad_top", "band_gap"):
        for k, a in sorted(
            exact_band_areas(model_a, family, include_ports=False).items()
        ):
            out.append(f"| {family} | B{k} | {a * 1e-6:.4f} |")
    out += [
        "",
        "## KLayout cross-check (tests/sgb_klayout.py::comparison_rows on the Route A build)",
        "",
        "| region | margin (um) | components (KLayout) | exact area (um^2) | KLayout area (um^2) | difference (um^2) | relative | oracle rounding bound (um^2) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for family, label in (("band_pad_top", "pad top"), ("band_gap", "gap")):
        for r in comparison_rows(model_a, family):
            rows.append({**r, "label": label})
            out.append(
                f"| {label} | {r['margin_units'] / 1000:g} | {r['components_klayout']} | {r['exact_um2']:.4f} | {r['klayout_um2']:.4f} | "
                f"{r['difference_um2']:+.5f} | {r['relative']:+.1e} | {r['oracle_rounding_bound_um2']:.3f} |"
            )
    (kit.FIGURES / "inset-klayout-comparison.json").write_text(
        json.dumps(rows, indent=1)
    )
    mesh = meshes["A"]
    out += [
        "",
        "## Mesh check of the Route A band groups (production writer, Gmsh; no Palace is run)",
        "",
        f"mesh size {mesh['size_um']:g} um, 1 um on the band curves; nodes: {mesh['nodes']}, tetrahedra: {mesh['tets']}, duplicate nodes: {mesh['duplicate_nodes']}, "
        f"conformal: {mesh['conformal']}",
        "",
        "| band group | faces | triangles | tets on each side | all triangles shared by tets | centroids inside band |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in sorted(mesh["bands"].items()):
        sides = sorted(row["tets_per_side"].values())
        out.append(
            f"| `{name}` | {row['faces']} | {row['triangles']} | {sides[0]} + {sides[1]} | {row['all_shared_by_tets']} | {row['centroids_in_band']} |"
        )
    out += [
        "",
        "## The same mesh check under every route",
        "",
        "| route | mesh size (um) | band groups | nodes | tetrahedra | duplicate nodes | conformal | every band group passes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for route, m in meshes.items():
        ok = all(
            r["triangles"] > 0
            and r["all_shared_by_tets"]
            and r["centroids_in_band"]
            and len(set(r["tets_per_side"].values())) == 1
            for r in m["bands"].values()
        )
        out.append(
            f"| {route} | {m['size_um']:g} | {len(m['bands'])} | {m['nodes']} | {m['tets']} | {m['duplicate_nodes']} | {m['conformal']} | {ok} |"
        )
    out += [
        "",
        "## Mask against imprint (inset-mask-vs-imprint.json)",
        "",
        "| h | triangles | straddling the contour | error, 1-point mask (median / max) | error, 3-point mask (median / max) | triangles (aligned) | straddling (aligned) | error (aligned) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in mask_rows:
        out.append(
            f"| {r['h']} | {r['triangles_unaligned']} | {r['cut_triangles']} | {r['err_one_point_median']:.1e} / {r['err_one_point_max']:.1e} | {r['err_three_point_median']:.1e} / {r['err_three_point_max']:.1e} | "
            f"{r['triangles_aligned']} | {r['crossing_aligned']} | {r['err_aligned']:.2e} |"
        )
    # Route B: the example bands the top face (MA) and the bottom face (MS)
    model_b = kit.pad_model("B", MARGINS)
    groups = sorted(
        n
        for n in model_b.routed.groups
        if "::band=" in n and n.startswith(("MA::pad", "MS::pad"))
    )
    counts = model_b.topology.counts()
    union = model_b.routed.unions["NET::pad"]
    out += [
        "",
        "## Route B (top and bottom faces banded)",
        "",
        "groups: " + ", ".join(groups),
        f"validation passed: {not model_b.report.failures} ({len(model_b.report.checks)} checks)",
        f"counts: {counts['faces']} faces, {counts['edges']} edges, {counts['vertices']} vertices, {len(model_b.routed.groups)} groups",
        "unions NET::pad: " + json.dumps(union),
        "surface_epr_rows (Route A): " + json.dumps(model_a.routed.surface_epr_rows),
    ]
    # the port moves no band: same supports, and it only takes its footprint out of the gap bands
    without = kit.pad_model("A", MARGINS, with_port=False)
    same = [f.support for f in model_a.labelled.catalogue.families] == [
        f.support for f in without.labelled.catalogue.families
    ]
    increase, removed, pad_top_equal = 0.0, 0.0, True
    for family in ("band_pad_top", "band_gap"):
        a, b = (
            exact_band_areas(model_a, family, include_ports=False),
            exact_band_areas(without, family),
        )
        increase = max(increase, max(a[k] - b[k] for k in a))
        if family == "band_gap":
            removed = (sum(b.values()) - sum(a.values())) * 1e-6
        else:
            pad_top_equal = a == exact_band_areas(without, family)
    out += [
        "",
        "## Port invariance (Route A, with and without the lumped port)",
        "",
        f"supports of the families equal: {same}",
        f"largest increase of a band area: {increase:.3g} grid unit^2",
        f"gap area taken out by the port: {removed:.4f} um^2",
        f"pad-top band areas equal: {pad_top_equal}",
    ]
    out += [
        "",
        "## Topology change (margins 4, 9, 30 um on the pad top)",
        "",
        json.dumps(topology_change),
    ]
    out += [
        "",
        "## Sagitta of a 128-gon disc, m (1 - cos(pi / 128)), in nm",
        "",
        "| margin (um) | sagitta (nm) |",
        "|---|---|",
    ] + [f"| {m} | {m * 1000 * (1 - math.cos(math.pi / 128)):.3f} |" for m in MARGINS]
    (kit.FIGURES / "inset-tables.md").write_text("\n".join(out) + "\n")
    print("wrote inset-tables.md")


def arc_deviation(model_a) -> dict:
    """Deviation of the arc edges of the assembled topology from their exact circles in the Route A model with bands (grid units)."""
    deviations = [
        e.curve.deviation
        for e in model_a.topology.edges.values()
        if e.curve is not None
    ]
    return {
        "arc_edges": len(deviations),
        "mean": float(np.mean(deviations)),
        "max": float(np.max(deviations)),
        "snap_bound": math.sqrt(2) / 2,
    }


def route_b_comparison() -> dict:
    """Route B's top and bottom faces against the oracle, and against the Route A pad sheet (same support, same numbers)."""
    model_b = kit.pad_model("B", MARGINS)
    rows_a = comparison_rows(kit.pad_model("A", MARGINS), "band_pad_top")
    out = {}
    for family in ("band_pad_top", "band_pad_bottom"):
        rows = comparison_rows(model_b, family)
        out[family] = {
            "max_relative": max(abs(r["relative"]) for r in rows),
            "same_exact_areas_as_route_a_pad": all(
                abs(r["exact_um2"] - a["exact_um2"]) < 1e-9
                for r, a in zip(rows, rows_a)
            ),
            "within_oracle_noise": all(
                abs(r["difference_um2"]) <= r["oracle_rounding_bound_um2"] for r in rows
            ),
        }
    return out


def extreme_band_comparisons() -> dict:
    """The band fixtures of the extreme-case catalogue against KLayout: every family of every route, every margin."""
    rows = []
    for name in (
        "neck_split_by_inset",
        "band_vanishes",
        "reflex_small_angle",
        "refinement_crossing_band_edge",
    ):
        for route in "ABC":
            model = build_model(FIXTURES[name], route)
            for family in (
                f.family_id for f in model.labelled.catalogue.families if f.support
            ):
                for r in comparison_rows(model, family):
                    rows.append(
                        {
                            "fixture": name,
                            "route": route,
                            "family": family,
                            "margin_units": r["margin_units"],
                            "components_klayout": r["components_klayout"],
                            "difference_um2": r["difference_um2"],
                            "relative": r["relative"],
                            "within_oracle_noise": abs(r["difference_um2"])
                            <= r["oracle_rounding_bound_um2"],
                        }
                    )
    return {
        "rows": len(rows),
        "families": len({(r["fixture"], r["route"], r["family"]) for r in rows}),
        "all_within_oracle_noise": all(r["within_oracle_noise"] for r in rows),
        "max_relative": max(abs(r["relative"]) for r in rows),
    }


def noded_edges() -> dict:
    """The z = 0 level of the Route A example before and after dissolution (the witness chords make the first number large)."""
    catalogue = kit.pad_catalogue("A")
    model = build_model_from_catalogue(
        catalogue, "A", insets=kit.pad_requests("A"), keep_raw=True
    )
    layer = model.labelled.level_layers[LEVEL_Z0]
    raw = layer.raw_arrangement
    return {
        "noded_edges_before_dissolution": len(raw.edges),
        "hot_pixels": len(raw.vertices),
        "edges_after_dissolution": len(layer.arr.edges),
        "faces": len(layer.arr.faces) - 1,
    }


def main() -> None:
    kit.ensure_plotly_js()
    model_a = kit.pad_model("A", MARGINS)
    figure_construction()
    figure_bands(model_a)
    figure_arcs(model_a)
    topology_change = figure_topology_change()
    mask_rows = figure_mask_vs_imprint()
    figure_routes_section()
    meshes = {
        "A": mesh_check(model_a, 40.0),
        "B": mesh_check(kit.pad_model("B", MARGINS), 60.0),
        "C": mesh_check(
            build_model_from_catalogue(
                kit.pad_catalogue("B"), "C", insets=kit.pad_requests("B")
            ),
            60.0,
        ),
    }
    build_tables(model_a, topology_change, mask_rows, meshes)
    extra = {
        "arc_deviation": arc_deviation(model_a),
        "level_z0": noded_edges(),
        "extreme_band_comparisons": extreme_band_comparisons(),
        "route_b_comparison": route_b_comparison(),
    }
    (kit.FIGURES / "inset-numbers.json").write_text(json.dumps(extra, indent=1))
    print(json.dumps(extra))


if __name__ == "__main__":
    main()
