"""Figures and numbers of ``slivers.qmd``: one computed example per sliver source of brief section 10 (production package).

Each example is small and drawn at the scale of the 1 nm grid (pixel grid lines visible).  Writes into
``docs/design/sgb/figures/``:

* ``slivers-gallery.svg/.html``  four rows, "as the source geometry would give it" next to "what SGB builds":
  1. quantization sliver (three nearly concurrent curves)        -> snap rounding,
  2. phantom sliver (edges of two unrelated layers 4 nm apart)    -> local arrangements,
  3. seam sliver (two polygons of one entity overlapping by 2 nm) -> dissolution,
  4. source near-coincidence (5 nm gap ring; a derived band curve 4 nm from a source curve) -> reported, not removed;
* ``slivers-numbers.json``      the measurements, including the report entries and the summary table the page quotes.

Which scene is which.  Rows 1 to 3 and the 10 nm-grid stand-in are hand-made stage-level scenes (prism catalogues built with the
helpers of ``tests/sgb_scenes.py``, the way the stage tests do): they need layers or sub-grid features no real input has.  Row 4 and
the summary table use the production real input ``gds_pad_in_ground`` (``tests/sgb_inputs.py``): the 5 nm gap is the same GDS with a
pad polygon 5 nm short of the cutout, the derived-versus-source example is the real pad sheet with its Surface-EPR bands.

"As the source would give it" is the exact arrangement (1), the old global projection emulated by a ``required`` family that
carries every layer's boundary on every level (2), and the noded cells before dissolution (3).  Example 4c's "after" is an
*emulation* of derived-to-source snapping (production has no such policy): the source curve is moved onto the derived curve by
4 nm, which gives the same topology the policy would.

Run:  SGB_CORE=<sgb-core checkout> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/slivers.py
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import gdstk
from labels_common import (
    PALETTE,
    build_model,
    cycle_polyline,
    go,
    mpl_path,
    np,
    plt,
    real_catalogue,
    save_plotly,
    save_svg,
    with_extra_families,
    write_numbers,
)
from matplotlib.patches import Patch, PathPatch
from plotly.subplots import make_subplots  # noqa: F401 -- kept for parity with the other figure scripts
from sgb_scenes import GRID, chip_prisms, conductor, footprint, rect_um
from shapely.geometry import Polygon, box

from scgsim.geometry.sgb.arrangement.dcel import build_arrangement
from scgsim.geometry.sgb.cutfamilies.requests import (
    ConductorSurface,
    InsetRequest,
    SolutionInterface,
)
from scgsim.geometry.sgb.errors import ContactError
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.noding.snap_rounding import Segment, snap_round
from scgsim.geometry.sgb.prisms.levels import build_levels
from scgsim.geometry.sgb.records.carriers import Lineage, LineCarrier
from scgsim.geometry.sgb.records.families import CurveFamilyRecord
from scgsim.geometry.sgb.records.grid import Grid, exact_z
from scgsim.geometry.sgb.records.prisms import PrismCatalogue, PrismRecord
from scgsim.geometry.sgb.slivers.report import sliver_report

RED = PALETTE["bad"]
MARGINS = (0.1, 0.5, 2.0)


def rings_grid(arr, curves, fi: int, window=None):
    """Boundary rings of a face in grid units; with ``window = (xlim, ylim)`` the face is clipped to it (keeps the SVG small)."""
    rings = [cycle_polyline(arr, curves, c, 4.0) for c in arr.face_cycles(fi)]
    if window is None:
        return rings

    (x0, x1), (y0, y1) = window
    clipped = (
        Polygon(rings[0], rings[1:])
        .buffer(0)
        .intersection(box(x0 - 1, y0 - 1, x1 + 1, y1 + 1))
    )
    out = []
    for part in getattr(clipped, "geoms", [clipped]):
        if part.is_empty or part.geom_type != "Polygon":
            continue
        out.append(list(part.exterior.coords)[:-1])
        out.extend(list(r.coords)[:-1] for r in part.interiors)
    return out


def face_width(arr, fi: int) -> float:
    """Width 2A/P of a face in grid units (holes count in the perimeter)."""
    area = arr.face_twice_area(fi) / 2
    per = sum(math.dist(p, q) for p, q in arr.face_segments(fi))
    return 2 * area / per if per else 0.0


def _pixel_grid(ax, xlim, ylim) -> None:
    for x in range(math.ceil(xlim[0]), math.floor(xlim[1]) + 1):
        ax.axvline(x - 0.5, color="#d0d0d0", lw=0.5, zorder=0)
    for y in range(math.ceil(ylim[0]), math.floor(ylim[1]) + 1):
        ax.axhline(y - 0.5, color="#d0d0d0", lw=0.5, zorder=0)


def _draw_cells(  # noqa: PLR0913 -- the axes, the colour rule, the ranges and the drawing options
    ax, layer, colour_of, xlim, ylim, narrow=10.0, raw=False, lw=1.0
) -> list[float]:
    arr = layer.raw_arrangement if raw else layer.arr
    labels = layer.raw_labels if raw else layer.labels
    curves = [None] * len(arr.edges) if raw else layer.edge_curve
    widths = []
    for fi in range(1, len(arr.faces)):
        w = face_width(arr, fi)
        widths.append(w)
        rings = rings_grid(arr, curves, fi, (xlim, ylim))
        if not rings:  # the face lies outside the window
            continue
        ax.add_patch(
            PathPatch(
                mpl_path(rings),
                facecolor=colour_of(labels[fi], fi),
                edgecolor=PALETTE["ink"],
                lw=lw,
                hatch="////" if w < narrow else None,
                zorder=2,
            )
        )
    _pixel_grid(ax, xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 5).round())
    ax.ticklabel_format(useOffset=False, style="plain")
    return widths


def _pair_colour(label, fi=None) -> str:
    below = getattr(label, "below", None)
    above = getattr(label, "above", None)
    names = (getattr(below, "name", "-"), getattr(above, "name", "-"))
    if label.sheet is not None:
        return "#b9b9b9" if label.sheet.kind == "conductor" else "#f3b4ae"
    if names[0] == "substrate" and names[1] == "vacuum":
        return "#a9cbe8"
    if names[0] == "substrate":
        return "#7b7f85"
    return "#fbf6df"


def _domain_fill(domain, fi=None) -> str:
    if domain is None:
        return "#f2f2f2"
    return {"conductor": "#7aa3cf", "vacuum": "#fbf6df", "dielectric": "#cfe0f1"}[
        domain.kind
    ]


def _level_layer(model, z: Fraction):
    return model.labelled.level_layers[model.labelled.levels.levels.index(z)]


# --------------------------------------------------------------------------- 1. quantization sliver


def _cross(s, t):
    (x1, y1), (x2, y2) = s
    (x3, y3), (x4, y4) = t
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    a, b = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
    return (
        Fraction(a * (x3 - x4) - (x1 - x2) * b, d),
        Fraction(a * (y3 - y4) - (y1 - y2) * b, d),
    )


def _quantization(numbers: dict):
    l1, l2, l3 = ((0, 0), (1000, 1000)), ((0, 1000), (1000, 0)), ((0, 495), (995, 505))
    exact = [_cross(l1, l2), _cross(l1, l3), _cross(l2, l3)]
    sides = [
        math.dist(map(float, p), map(float, q))
        for p, q in itertools.combinations(exact, 2)
    ]
    area = (
        abs(
            float(
                (exact[1][0] - exact[0][0]) * (exact[2][1] - exact[0][1])
                - (exact[2][0] - exact[0][0]) * (exact[1][1] - exact[0][1])
            )
        )
        / 2
    )
    snapped = snap_round(
        [
            Segment(s[0], s[1], frozenset([n]))
            for s, n in zip((l1, l2, l3), ("L1", "L2", "L3"), strict=True)
        ]
    )
    arr = build_arrangement(snapped.edges)
    lengths = sorted(math.dist(e.a, e.b) for e in arr.edges)
    numbers["quantization"] = {
        "curves": "L1 (0,0)-(1000,1000), L2 (0,1000)-(1000,0), L3 (0,495)-(995,505), grid units",
        "exact_triangle_vertices": [[float(x), float(y)] for x, y in exact],
        "exact_triangle_sides_grid": sides,
        "exact_triangle_area_grid2": area,
        "exact_edges_shorter_than_g": sum(s < 1 for s in sides),
        "snapped_bounded_faces": len(arr.faces) - 1,
        "snapped_edges": len(arr.edges),
        "snapped_shortest_edge_grid": lengths[0],
        "snapped_hot_pixels": snapped.hot_pixels,
        "snapped_rounds": snapped.rounds,
    }
    return l1, l2, l3, exact, arr


def _draw_quantization(ax_left, ax_right, data) -> None:
    l1, l2, l3, exact, arr = data
    for line, colour in ((l1, "#1f77b4"), (l2, "#2ca02c"), (l3, "#d62728")):
        (x0, y0), (x1, y1) = line
        xs = np.array([499.9, 500.1])
        ys = y0 + (y1 - y0) * (xs - x0) / (x1 - x0)
        ax_left.plot(xs, ys, color=colour, lw=1.6)
        xr = np.array([496.0, 504.0])
        ax_right.plot(
            xr,
            y0 + (y1 - y0) * (xr - x0) / (x1 - x0),
            color=colour,
            lw=0.7,
            ls=":",
            zorder=1,
        )
    tri = [(float(x), float(y)) for x, y in exact]
    ax_left.fill(
        [p[0] for p in tri],
        [p[1] for p in tri],
        color=RED,
        alpha=0.35,
        hatch="////",
        zorder=3,
    )
    ax_left.plot([p[0] for p in tri], [p[1] for p in tri], "o", color=RED, ms=3)
    ax_left.ticklabel_format(useOffset=False, style="plain")
    ax_left.set_xlim(499.93, 500.07)
    ax_left.set_ylim(499.93, 500.07)
    ax_left.set_aspect("equal")
    ax_left.set_title(
        "1a. Exact arrangement of three curves, zoom 0.14 g wide:\na face with three edges shorter than g",
        fontsize=8.5,
    )
    for e in arr.edges:
        ax_right.plot(
            [e.a[0], e.b[0]], [e.a[1], e.b[1]], color=PALETTE["ink"], lw=1.6, zorder=3
        )
    verts = {v for e in arr.edges for v in (e.a, e.b)}
    ax_right.plot(
        [v[0] for v in verts],
        [v[1] for v in verts],
        "o",
        color=PALETTE["ink"],
        ms=4,
        zorder=4,
    )
    _pixel_grid(ax_right, (496, 504), (496, 504))
    ax_right.set_xlim(496, 504)
    ax_right.set_ylim(496, 504)
    ax_right.set_aspect("equal")
    ax_right.set_title(
        "1b. Snap rounding, window 8 g wide (dotted: the exact curves):\none vertex, no face, no edge shorter than g",
        fontsize=8.5,
    )


# --------------------------------------------------------------------------- 2. phantom sliver


def _phantom_catalogue(global_projection: bool) -> PrismCatalogue:
    prisms = chip_prisms(GRID, 150.0)
    m1 = conductor("m1", rect_um(-100, -50, 100, 50), 0, "0.2", "m1")
    m2 = conductor("m2", rect_um(-99.996, -50, 100.004, 50), 3, "3.3", "m2")
    prisms += [m1, m2]
    families = ()
    if global_projection:
        levels = build_levels(tuple(prisms)).levels
        carriers = tuple(c for p in (m1, m2) for c in p.carriers())
        families = (
            CurveFamilyRecord(
                "global_projection",
                "projection",
                tuple(
                    LineCarrier(
                        f"gp/{c.id}",
                        c.p,
                        c.q,
                        Lineage("family_member", family_id="gp", member=c.id),
                    )
                    for c in carriers
                ),
                tuple(levels),
                "required",
            ),
        )
    return PrismCatalogue(GRID, tuple(prisms), families, name="phantom")


def _phantom(numbers: dict):
    local, glob = (
        build_model_from_catalogue(_phantom_catalogue(False), "B"),
        build_model_from_catalogue(_phantom_catalogue(True), "B"),
    )
    out = {}
    for name, m in (("local", local), ("global_emulated", glob)):
        rep = sliver_report(m, top=200, drift=False)
        thin = [f for f in rep["narrowest_faces"] if f["value_um"] < 0.01]
        out[name] = {
            "faces_narrower_than_10nm": len(thin),
            "planes_of_those_faces": sorted({f["plane"] for f in thin}),
            "model_faces": m.counts["faces"],
            "model_edges": m.counts["edges"],
            "level_cells": m.counts["level_cells"],
            "narrowest_face": rep["narrowest_faces"][0],
            "shortest_edge_um": rep["summary"]["shortest_edge_um"],
            "narrowest_face_um": rep["summary"]["narrowest_face_um"],
            "validation_passed": not m.report.failures,
        }
    numbers["phantom"] = out
    return local, glob


def _draw_phantom(ax_left, ax_right, local, glob) -> None:
    xlim, ylim = (99996, 100008), (-6, 6)
    for ax, model, title in (
        (
            ax_left,
            glob,
            "2a. Old global projection (emulated): level z = 0 also carries the upper layer's\nedge, 4 nm from the lower layer's: a 4 nm substrate-vacuum face",
        ),
        (
            ax_right,
            local,
            "2b. Local arrangement: level z = 0 holds only curves that meet there",
        ),
    ):
        _draw_cells(ax, _level_layer(model, Fraction(0)), _pair_colour, xlim, ylim)
        ax.set_title(title, fontsize=8.5)


# --------------------------------------------------------------------------- 3. seam sliver


def seam_catalogue(
    *, one_entity: bool = True, priorities: tuple[int, int] = (0, 0)
) -> PrismCatalogue:
    """Two overlapping rectangles of one Net: two polygons of one entity (one_entity) or two entities."""
    prisms = chip_prisms(GRID, 150.0)
    a = footprint(rect_um(-100, -20, 0.002, 20))
    b = footprint(rect_um(0.0, -20, 100, 20))
    z0, z1 = exact_z(0), exact_z("0.2")
    if one_entity:
        prisms += [
            PrismRecord("n", a, z0, z1, "Al", "conductor", net_id="N", polygon=0),
            PrismRecord("n", b, z0, z1, "Al", "conductor", net_id="N", polygon=1),
        ]
    else:
        prisms += [
            PrismRecord(
                "a", a, z0, z1, "Al", "conductor", net_id="N", priority=priorities[0]
            ),
            PrismRecord(
                "b", b, z0, z1, "Al", "conductor", net_id="N", priority=priorities[1]
            ),
        ]
    return PrismCatalogue(GRID, tuple(prisms), name="seam")


def _seam(numbers: dict):
    model = build_model_from_catalogue(seam_catalogue(), "B", keep_raw=True)
    layer = next(s for s in model.labelled.slabs if s.plane == "slab[0,1/5]")
    raw = layer.raw_arrangement
    raw_widths = sorted(face_width(raw, fi) for fi in range(1, len(raw.faces)))
    dis_widths = sorted(
        face_width(layer.arr, fi) for fi in range(1, len(layer.arr.faces))
    )
    numbers["seam"] = {
        "scene": "stage-level: two polygons of one entity, same Net",
        "slab": layer.plane,
        "raw_cells": len(raw.faces) - 1,
        "cells_after": len(layer.arr.faces) - 1,
        "raw_edges": len(raw.edges),
        "edges_after": len(layer.arr.edges),
        "raw_narrowest_cell_width_grid": raw_widths[0],
        "after_narrowest_cell_width_grid": dis_widths[0],
        "overlap_um": 0.002,
    }
    # production keeps the seam between two entities: the 2 nm overlap is a cell of the higher-priority entity, and it is reported
    two = build_model_from_catalogue(
        seam_catalogue(one_entity=False, priorities=(1, 0)), "B"
    )
    two_layer = next(s for s in two.labelled.slabs if s.plane == "slab[0,1/5]")
    rep = sliver_report(two, top=3, drift=False)
    numbers["seam_as_two_entities"] = {
        "cells_after": len(two_layer.arr.faces) - 1,
        "edges_after": len(two_layer.arr.edges),
        "labels": sorted(d.name for d in two_layer.labels[1:]),
        "narrowest_face": rep["narrowest_faces"][0],
        "narrowest_face_um": rep["summary"]["narrowest_face_um"],
    }
    return model, layer


def _draw_seam(ax_left, ax_right, layer) -> None:
    raw, raw_labels = layer.raw_arrangement, layer.raw_labels
    xlim, ylim = (-4, 6), (-5, 5)
    no_curves = [None] * len(raw.edges)
    for fi in range(1, len(raw.faces)):
        rings = rings_grid(raw, no_curves, fi, (xlim, ylim))
        if rings:
            ax_left.add_patch(
                PathPatch(
                    mpl_path(rings),
                    facecolor=_domain_fill(raw_labels[fi]),
                    edgecolor="none",
                    hatch="////" if face_width(raw, fi) < 10 else None,
                    zorder=2,
                )
            )
    for ei, e in enumerate(raw.edges):
        left, right = (
            raw_labels[raw.he_face[2 * ei]],
            raw_labels[raw.he_face[2 * ei + 1]],
        )
        ax_left.plot(
            [e.a[0], e.b[0]],
            [e.a[1], e.b[1]],
            color=RED if left == right else PALETTE["ink"],
            lw=2.0 if left == right else 1.2,
            zorder=3,
        )
    _pixel_grid(ax_left, xlim, ylim)
    ax_left.set_xlim(*xlim)
    ax_left.set_ylim(*ylim)
    ax_left.set_aspect("equal")
    ax_left.set_title(
        "3a. Noded cells before dissolution: the 2 nm overlap of two polygons\nof one entity is a cell; its two edges (red) have equal labels on both sides",
        fontsize=8.5,
    )
    _draw_cells(ax_right, layer, _domain_fill, xlim, ylim)
    ax_right.set_title(
        "3b. After dissolution: one conductor cell, no seam", fontsize=8.5
    )


# --------------------------------------------------------------------------- 4. source near-coincidence


def ring_catalogue(gap_um: float = 0.005):
    """The real input with a pad polygon ``gap_um`` short of the ground cutout on every side (no port): a ring-shaped gap."""
    pad = gdstk.rectangle((30 + gap_um, 30 + gap_um), (70 - gap_um, 70 - gap_um))
    return real_catalogue(film=True, with_port=False, pad_polygon=pad)


def _source_gap(numbers: dict):
    model = build_model(ring_catalogue(), "A")
    rep = sliver_report(model, top=3, drift=False)
    numbers["source_gap"] = {
        "scene": "real input (thin_film, no port) with the pad 5 nm short of the cutout on every side",
        "gap_um": 0.005,
        "narrowest_faces": rep["narrowest_faces"][:2],
        "shortest_edges": rep["shortest_edges"][:1],
        "summary": rep["summary"],
        "model_faces": model.counts["faces"],
        "validation_passed": not model.report.failures,
    }
    return model


def coarse_grid_stand_in(numbers: dict) -> None:
    """Stand-in for a coarser snap grid: the 4 nm gap ring rebuilt on a whole-model 10 nm grid (stage-level scene, one grid per model).

    The gap closes, the two Nets touch, and the build must stop with a contact error."""
    out = {}
    for label, g in (("grid_1nm", 0.001), ("grid_10nm", 0.01)):
        grid = Grid(g)
        prisms = chip_prisms(grid, 150.0)
        gap = 0.004
        ground = footprint(
            rect_um(-150, -150, 150, 150),
            [rect_um(-50 - gap, -30 - gap, 50 + gap, 30 + gap)],
            grid=grid,
        )
        pad = footprint(rect_um(-50, -30, 50, 30), grid=grid)
        prisms += [
            PrismRecord(
                "ground",
                ground,
                exact_z(0),
                exact_z(0),
                "Al",
                "conductor",
                net_id="Ground",
            ),
            PrismRecord(
                "pad", pad, exact_z(0), exact_z(0), "Al", "conductor", net_id="pad"
            ),
        ]
        try:
            m = build_model_from_catalogue(
                PrismCatalogue(grid, tuple(prisms), name="coarse"), "A"
            )
        except ContactError as e:  # the gap closes on the coarse grid: different-Net contact stops the build
            out[label] = {"gap_nm": 4, "stopped": "ContactError", "message": str(e)}
            continue
        out[label] = {
            "gap_nm": 4,
            "model_faces": m.counts["faces"],
            "gap_faces": sum(
                m.routed.face_group[f.id] == "SA::substrate|vacuum__interface"
                for f in m.topology.faces.values()
            ),
            "validation_passed": not m.report.failures,
        }
    numbers["coarser_grid_stand_in"] = out


def derived_catalogue(edge_um: float):
    """The real pad sheet (no port) with bands, plus a ``window`` family whose right edge is ``edge_um``: the 2 um band curve is at x = 42."""
    pts = [(41.0, 45.0), (edge_um, 45.0), (edge_um, 55.0), (41.0, 55.0)]
    ip = [(round(x * 1000), round(y * 1000)) for x, y in pts]
    carriers = tuple(
        LineCarrier(
            f"window/{i}",
            ip[i],
            ip[(i + 1) % 4],
            Lineage("family_member", family_id="window", member=f"window/{i}"),
        )
        for i in range(4)
    )
    window = CurveFamilyRecord(
        "window", "partition", carriers, (Fraction(0),), "required"
    )
    return with_extra_families(real_catalogue(film=True, with_port=False), window)


def _derived(numbers: dict):
    bands = (InsetRequest("band_pad", ConductorSurface("pad", "sheet"), MARGINS),)
    near = build_model(derived_catalogue(42.004), "A", insets=bands)
    snapped = build_model(derived_catalogue(42.0), "A", insets=bands)
    out = {}
    for name, m in (("before", near), ("after_emulated_snap", snapped)):
        rep = sliver_report(m, top=3, drift=False)
        thin = [
            f
            for f in sliver_report(m, top=200, drift=False)["narrowest_faces"]
            if f["value_um"] < 0.01
        ]
        out[name] = {
            "level_cells": len(_level_layer(m, Fraction(0)).arr.faces) - 1,
            "model_faces": m.counts["faces"],
            "narrowest_face": rep["narrowest_faces"][0],
            "faces_narrower_than_10nm": thin,
            "narrowest_face_um": rep["summary"]["narrowest_face_um"],
            "validation_passed": not m.report.failures,
        }
    numbers["derived_vs_source"] = out
    return near, snapped


def _band_fill(label, fi=None) -> str:
    if label.sheet is not None and label.sheet.kind == "port":
        return "#f3b4ae"
    if label.sheet is not None and label.bands:
        return ["#08306b", "#2171b5", "#6baed6", "#c6dbef"][label.bands[0][1]]
    return "#dddddd"


def _draw_source(ax_gap, ax_a, ax_b, ring, near, snapped) -> None:  # noqa: PLR0913 -- the three axes and the three rings the example shows
    xlim, ylim = (29996, 30010), (49994, 50006)
    layer = _level_layer(ring, Fraction(0))

    def ring_fill(label, fi=None):
        if label.sheet is None:
            return "#a9cbe8"
        return "#7aa3cf" if label.sheet.name == "pad" else "#9a9a9a"

    _draw_cells(ax_gap, layer, ring_fill, xlim, ylim)
    ax_gap.set_title(
        "4a. A 5 nm gap between pad and ground (source fact): the gap cell\nis 5 nm wide along its whole length; kept and reported",
        fontsize=8.5,
    )
    xlim2, ylim2 = (41988, 42012), (49992, 50008)
    for ax, model, title in (
        (
            ax_a,
            near,
            "4b. A band curve (derived, 2 µm from the pad edge) and a source curve\n4 nm apart: a 4 nm cell, kept and reported",
        ),
        (
            ax_b,
            snapped,
            "4c. Emulated derived-to-source snapping: the two curves coincide,\nthe cell is gone",
        ),
    ):
        _draw_cells(
            ax, _level_layer(model, Fraction(0)), _band_fill, xlim2, ylim2, lw=0.9
        )
        ax.set_title(title, fontsize=8.5)


# --------------------------------------------------------------------------- the summary table of the running example


def summary_table(numbers: dict) -> None:
    """Sliver-report summary of the running example (real input) and of the airbridge (stage-level scene)."""
    from sgb_emit_cases import CASES  # noqa: PLC0415 -- the named stage-level models of the emission tests

    real = real_catalogue()
    real_a = real_catalogue(film=True)
    margins = MARGINS
    bands_a = (
        InsetRequest("band_pad", ConductorSurface("pad", "sheet"), margins),
        InsetRequest(
            "band_gap",
            SolutionInterface("substrate", "VACUUM_REGION", z_um=0.0),
            margins,
        ),
    )
    bands_b = (
        InsetRequest("band_pad_top", ConductorSurface("pad", "top"), margins),
        InsetRequest("band_pad_bottom", ConductorSurface("pad", "bottom"), margins),
        InsetRequest(
            "band_gap",
            SolutionInterface("substrate", "VACUUM_REGION", z_um=0.0),
            margins,
        ),
    )
    models = {
        "real input, Route A": build_model(real_a, "A"),
        "real input, Route B": build_model(real, "B"),
        "real input, Route A with bands": build_model(real_a, "A", insets=bands_a),
        "real input, Route B with bands": build_model(real, "B", insets=bands_b),
        "airbridge_B (stage-level scene)": CASES["bridge_b"](),
        # the stage-level scene with true arcs (rounded corner, circular hole) and the same bands: where an arc deviation shows
        "pad_a_bands (stage-level scene with arcs)": CASES["pad_a_bands"](),
        "pad_b_bands (stage-level scene with arcs)": CASES["pad_b_bands"](),
    }
    out = {}
    for name, m in models.items():
        rep = sliver_report(m)
        out[name] = {
            **rep["summary"],
            "shortest_edge_entry": rep["shortest_edges"][0],
            "narrowest_face_entry": rep["narrowest_faces"][0],
            "thin_slabs": rep["thin_slabs"],
            "arc_edges": {k: v for k, v in rep["arc_edges"].items() if k != "worst"},
            "validation_passed": not m.report.failures,
        }
    numbers["example_reports"] = out


# --------------------------------------------------------------------------- orchestration


def main() -> None:
    """Write the sliver gallery and its numbers."""
    numbers: dict = {}
    q = _quantization(numbers)
    local, glob = _phantom(numbers)
    _, seam_layer = _seam(numbers)
    ring = _source_gap(numbers)
    near, snapped = _derived(numbers)
    coarse_grid_stand_in(numbers)
    summary_table(numbers)

    fig, axes = plt.subplots(
        4, 3, figsize=(14.5, 15.5), gridspec_kw={"width_ratios": [1, 1, 1]}
    )
    for ax in axes.flat:
        ax.tick_params(labelsize=7)
    _draw_quantization(axes[0][0], axes[0][1], q)
    axes[0][2].axis("off")
    axes[0][2].text(
        0,
        0.5,
        "Eliminated by construction:\nsnap rounding\n\nexact crossing triangle with\nthree edges of 0.04 to 0.05 g\nbecomes one vertex",
        fontsize=9,
        va="center",
    )
    _draw_phantom(axes[1][0], axes[1][1], local, glob)
    axes[1][2].axis("off")
    axes[1][2].text(
        0,
        0.5,
        "Eliminated by construction:\nlocal arrangements\n\nlevel z = 0 never meets the\nz = 3 layer, so its curve is\nnot on this level",
        fontsize=9,
        va="center",
    )
    _draw_seam(axes[2][0], axes[2][1], seam_layer)
    axes[2][2].axis("off")
    axes[2][2].text(
        0,
        0.5,
        "Eliminated by construction:\ndissolution\n\nthe overlap cell and its two\nseam edges disappear",
        fontsize=9,
        va="center",
    )
    _draw_source(axes[3][0], axes[3][1], axes[3][2], ring, near, snapped)
    for ax in axes.flat:
        if ax.axison:
            ax.set_xlabel("x (grid units, 1 nm)", fontsize=7)
            ax.set_ylabel("y (grid units)", fontsize=7)
    fig.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor="#222",
                hatch="////",
                label="cell narrower than 10 g (a sliver)",
            ),
            plt.Line2D(
                [0],
                [0],
                color=RED,
                lw=2,
                label="seam edge (equal labels on both sides)",
            ),
        ],
        loc="lower center",
        ncol=2,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save_svg(fig, "slivers-gallery")

    # interactive: the same four comparisons as a small bar chart of narrowest widths
    rows = [
        (
            "1 quantization: shortest exact edge (g)",
            min(numbers["quantization"]["exact_triangle_sides_grid"]),
            numbers["quantization"]["snapped_shortest_edge_grid"],
        ),
        (
            "2 phantom: narrowest face (nm)",
            numbers["phantom"]["global_emulated"]["narrowest_face_um"] * 1000,
            numbers["phantom"]["local"]["narrowest_face_um"] * 1000,
        ),
        (
            "3 seam: narrowest cell (g)",
            numbers["seam"]["raw_narrowest_cell_width_grid"],
            numbers["seam"]["after_narrowest_cell_width_grid"],
        ),
        (
            "4b derived vs source: narrowest face (nm)",
            numbers["derived_vs_source"]["before"]["narrowest_face_um"] * 1000,
            numbers["derived_vs_source"]["after_emulated_snap"]["narrowest_face_um"]
            * 1000,
        ),
    ]
    pcs = go.Figure()
    pcs.add_trace(
        go.Bar(
            y=[r[0] for r in rows],
            x=[r[1] for r in rows],
            orientation="h",
            name="without the rule (or before)",
            marker_color=RED,
        )
    )
    pcs.add_trace(
        go.Bar(
            y=[r[0] for r in rows],
            x=[r[2] for r in rows],
            orientation="h",
            name="with SGB (or after)",
            marker_color=PALETTE["good"],
        )
    )
    pcs.update_xaxes(
        type="log", title="measured width or length (log scale; units as labelled)"
    )
    pcs.update_layout(
        barmode="group",
        title="Smallest measured feature per sliver source, without and with the rule (computed by slivers.py)",
        width=1100,
        height=520,
        margin=dict(l=320),
    )
    save_plotly(pcs, "slivers-gallery")
    numbers["gallery_bars"] = [
        {"label": r[0], "without": r[1], "with": r[2]} for r in rows
    ]
    write_numbers("slivers", numbers)
    print("slivers: done")


if __name__ == "__main__":
    main()
