"""Figures of the inset-imprint page: the corner rule, the bands of the running example, exact arcs against a KLayout polygon and the topology change, with the helpers they share."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
import numpy as np  # noqa: E402
from inset_kit import G_UM as G
from inset_kit import MARGINS, Line, Panel, Poly, Pts, primitive_polyline  # noqa: E402
from sgb_klayout import eroded_klayout, region_of_rings  # noqa: E402

from scgsim.geometry.sgb.arrangement.arcs import face_exact_area  # noqa: E402

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
    """The bands of the running example as a plan view and by margin (``inset-bands-pad``)."""
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


def _klayout_rings(region, margin_um: float) -> list[list[tuple[float, float]]]:
    eroded = eroded_klayout(region_of_rings(region), round(margin_um / G))
    rings = []
    for poly in eroded.each():
        rings.append([(p.x * G, p.y * G) for p in poly.each_point_hull()])
        for h in range(poly.holes()):
            rings.append([(p.x * G, p.y * G) for p in poly.each_point_hole(h)])
    return rings


def _radial_deviation(points, centre, radius):
    """(angle in degrees, radial deviation in nm) of the points within 20 nm of a circle, thinned to about 300."""
    data = [
        (
            math.degrees(math.atan2(y - centre[1], x - centre[0])),
            (math.hypot(x - centre[0], y - centre[1]) - radius) * 1000,
        )
        for x, y in points
        if abs(math.hypot(x - centre[0], y - centre[1]) - radius) < 0.02
    ]
    return data[:: max(1, len(data) // 300)]


def figure_arcs(model_a) -> None:
    """Exact arcs against a KLayout 512-gon, and the radial deviation of both (``inset-arcs-vs-polygon``)."""
    layer = model_a.labelled.level_layers[LEVEL_Z0]
    region = support_of(model_a, "band_pad_top")
    exact_curves = kit.layer_edge_polylines(
        layer, lambda src: any(s.startswith("band_pad_top/") for s in src)
    )
    rings = {m: _klayout_rings(region, m) for m in MARGINS}
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

            ex = _radial_deviation(
                [p for poly in exact_curves for p in poly], centre, r
            )
            kl = _radial_deviation([p for ring in rings[m] for p in ring], centre, r)
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
    """The neck split and the empty band when the margin grows (``inset-topology-change``); returns the band areas it draws."""
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
