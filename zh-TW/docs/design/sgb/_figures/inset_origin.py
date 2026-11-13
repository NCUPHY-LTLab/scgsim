"""Figures and numbers for the section "Inset band 怎麼量" of ``docs/design/sgb/inset-imprint.qmd``.

Run, with the Python of the checkout that contains the production SGB core (``scgsim.geometry.sgb``)::

    SGB_CORE=<sgb-core> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/inset_origin.py

Two real inputs of the production tests (``tests/sgb_inputs.py``) are built with ``build_sgb_geometry`` and bands on every
surface, margins (0.5, 1.0) um:

* ``gds_pad_in_ground`` (a GDS and a stack file, a 100 um die, a ground plane with a 40 um cutout, a 20 um pad, a lumped
  port), Route B: bands on the top and bottom face of both metals and on the substrate-vacuum interface;
* ``thin_film_pair`` (two sheets on a larger substrate joined by a lumped port), Route A: bands on both sheets and on the
  substrate-vacuum interface.

The drawings read the z = 0 level arrangement of the validated model (every cell with its band label) and the distance
origins of each support region (``SupportRegion.origins`` and ``free_boundary``); the numbers come from the sidecar the
build wrote.  Outputs go to ``docs/design/sgb/figures/`` with the prefix ``inset-origin-``:

* ``inset-origin-ma``       the pad-in-ground die coloured by band, with zooms on the cutout corner, the port and the die corner
* ``inset-origin-sa``       the thin-film pair coloured by band
* ``inset-origin-section``  an x-z section at y = 50 um through the support plane z = 0 (metal, port, substrate)
* ``inset-origin-numbers.json``  band areas, origin lengths and the error of a support without a gap edge
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
from inset_kit import Label, Line, Panel, Poly, Pts, primitive_polyline  # noqa: E402
from sgb_inputs import gds_pad_in_ground, thin_film_pair  # noqa: E402
from sgb_scenes import slab_model  # noqa: E402

from scgsim.contracts.sgb_sidecar import read_sidecar  # noqa: E402
from scgsim.geometry.sgb.bands.requests import (  # noqa: E402
    ConductorSurface,
    InsetRequest,
    SolutionInterface,
)
from scgsim.geometry.sgb.build import build_sgb_geometry  # noqa: E402
from scgsim.geometry.sgb.errors import InputError  # noqa: E402
from scgsim.geometry.sgb.exact.carriers import LineEdge  # noqa: E402
from scgsim.geometry.sgb.model import build_conformal_model, build_model_from_catalogue  # noqa: E402

G = kit.G_UM
MARGINS = (0.5, 1.0)
METAL = ["#08306b", "#4292c6", "#c6dbef"]  # band 0, 1, last band on a conductor face
GAP = [
    "#7f2704",
    "#e6550d",
    "#fdd0a2",
]  # the same on the exposed substrate surface (SA)
ORIGIN = "#000000"
FREE = "#6a3d9a"


# --------------------------------------------------------------------------- the two builds


def pad_in_ground_requests() -> tuple[InsetRequest, ...]:
    """Bands on the top and bottom face of both metals and on the substrate-vacuum interface (Route B)."""
    out = [
        InsetRequest(f"{entity}_{side}", ConductorSurface(entity, side), MARGINS)
        for entity in ("ground_0000", "pad")
        for side in ("top", "bottom")
    ]
    out.append(
        InsetRequest(
            "sa", SolutionInterface("substrate", "VACUUM_REGION", z_um=0.0), MARGINS
        )
    )
    return tuple(out)


def thin_film_requests() -> tuple[InsetRequest, ...]:
    """Bands on both sheets and on the substrate-vacuum interface (Route A)."""
    out = [
        InsetRequest(f"{entity}_sheet", ConductorSurface(entity, "sheet"), MARGINS)
        for entity in ("chip/left", "chip/right")
    ]
    out.append(
        InsetRequest(
            "sa", SolutionInterface("substrate", "upper_air", z_um=0.0), MARGINS
        )
    )
    return tuple(out)


def build_case(effective, route: str, requests, scratch: Path):
    """The model and the sidecar of one build through the public entry point ``build_sgb_geometry``."""
    artifacts = build_sgb_geometry(
        effective.build_input,
        effective.representation,
        route=route,
        insets=requests,
        output_dir=scratch / route,
    )
    model = build_conformal_model(
        effective.build_input, effective.representation, route, insets=requests
    )
    return model, read_sidecar(artifacts.sidecar), artifacts.counts


# --------------------------------------------------------------------------- drawing


def length_um(prims) -> float:
    """Length of ring primitives in um."""
    total = 0.0
    for prim in prims:
        if isinstance(prim, LineEdge):
            total += math.hypot(prim.q[0] - prim.p[0], prim.q[1] - prim.p[1])
        else:
            total += float(prim.radius) * abs(float(prim.sweep()[1]))
    return total * G


def on_conductor(label) -> bool:
    """Whether a z = 0 level cell lies under a conductor footprint (a sheet, or a finite body above the plane)."""
    return (label.sheet is not None and label.sheet.kind == "conductor") or any(
        d is not None and d.kind == "conductor" for d in (label.below, label.above)
    )


def cell_style(label) -> tuple[str, str]:
    """Fill colour and key of a z = 0 level cell: port, conductor band, exposed-substrate band or nothing."""
    if label.sheet is not None and label.sheet.kind == "port":
        return kit.PORT, "port"
    if not label.bands:
        return "#ffffff", "outside"
    on_metal = on_conductor(label)
    return (METAL if on_metal else GAP)[
        label.bands[0][1]
    ], "metal" if on_metal else "gap"


def level_polys(model, edge_width: float) -> list[Poly]:
    """One polygon per cell of the z = 0 level arrangement, coloured by band."""
    layer = model.labelled.level_layers[model.labelled.levels.levels.index(Fraction(0))]
    polys = []
    for fi in range(1, len(layer.arr.faces)):
        fill, _ = cell_style(layer.labels[fi])
        group = model.routed.face_group.get(layer.face_ids[fi]) or "(no face)"
        polys.append(
            Poly(
                kit.face_rings_um(layer, fi),
                fill=fill,
                edge="rgba(0,0,0,0.5)",
                edge_width=edge_width,
                hover=group,
            )
        )
    return polys


def boundary_lines(model, width: float = 3.0) -> list[Line]:
    """Gap edges (the distance origins of every support) and the support boundary that is not an origin."""
    seen: set[tuple] = set()
    out: list[Line] = []
    for family in model.labelled.catalogue.families:
        for prims, colour, dash, name, w in (
            (
                family.support.origins,
                ORIGIN,
                "solid",
                "gap edge (distance origin)",
                width,
            ),
            (
                family.support.free_boundary,
                FREE,
                "dash",
                "die edge (not an origin)",
                width - 0.8,
            ),
        ):
            for prim in prims:
                points = primitive_polyline(prim)
                key = (
                    name,
                    tuple(
                        sorted(
                            (round(x, 4), round(y, 4))
                            for x, y in (points[0], points[-1])
                        )
                    ),
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    Line(
                        [p[0] for p in points],
                        [p[1] for p in points],
                        colour,
                        w,
                        dash,
                        name=name,
                    )
                )
    return out


def legend(*, metal: bool = True) -> list[Pts]:
    """Legend entries (off-screen markers) for the band colours."""
    rows = []
    if metal:
        rows += [
            Pts(
                [-1e4],
                [-1e4],
                METAL[0],
                8,
                name="metal, band 0  (d < 0.5 um)",
                symbol="square",
            ),
            Pts(
                [-1e4],
                [-1e4],
                METAL[1],
                8,
                name="metal, band 1  (0.5 <= d < 1.0 um)",
                symbol="square",
            ),
            Pts(
                [-1e4],
                [-1e4],
                METAL[2],
                8,
                name="metal, last band  (d >= 1.0 um)",
                symbol="square",
            ),
        ]
    rows += [
        Pts(
            [-1e4],
            [-1e4],
            GAP[0],
            8,
            name="substrate surface (SA), band 0",
            symbol="square",
        ),
        Pts([-1e4], [-1e4], GAP[1], 8, name="SA, band 1", symbol="square"),
        Pts([-1e4], [-1e4], GAP[2], 8, name="SA, last band", symbol="square"),
        Pts(
            [-1e4],
            [-1e4],
            kit.PORT,
            8,
            name="lumped port (takes its footprint out of SA)",
            symbol="square",
        ),
    ]
    return rows


def figure_ma(model) -> None:
    """The pad-in-ground die: bands only along the cutout; the die edge is no origin."""
    boundaries = boundary_lines(model)
    views = [
        (
            "whole die (100 um): the bands are too narrow to see at this scale",
            (-3, 103),
            (-3, 103),
            0.0,
        ),
        (
            "corner of the cutout: bands ripple out of the metal on both sides",
            (27, 37),
            (27, 37),
            0.4,
        ),
        (
            "the port: only its footprint leaves SA, no band moves",
            (26, 44),
            (44, 56),
            0.4,
        ),
        (
            "corner of the die: ground metal reaches the edge, no band",
            (-1, 9),
            (-1, 9),
            0.4,
        ),
    ]
    panels = []
    for title, xr, yr, edge_width in views:
        polys = level_polys(model, edge_width)
        panels.append(
            Panel(
                title,
                xr,
                yr,
                xlabel="x (um)",
                ylabel="y (um)",
                items=polys + boundaries + legend(),
            )
        )
    kit.write_both(
        panels,
        "inset-origin-ma",
        ncols=2,
        svg_size=(10, 9.4),
        html_size=(1100, 950),
        shared_legend=True,
        title="Route B, pad-in-ground (GDS input): z = 0 level coloured by band, margins 0.5 and 1.0 um",
    )


def figure_sa(model) -> None:
    """The thin-film pair: bands ripple out of the two sheets only."""
    items = level_polys(model, 0.4) + boundary_lines(model) + legend()
    panel = Panel(
        "two 4 x 4 um sheets and a lumped port on a 16 x 6 um substrate",
        (-3.6, 13.6),
        (-1.6, 5.6),
        xlabel="x (um)",
        ylabel="y (um)",
        items=items,
    )
    kit.write_both(
        [panel],
        "inset-origin-sa",
        svg_size=(9.5, 5.4),
        html_size=(1100, 600),
        shared_legend=True,
        title="Route A, thin-film pair: z = 0 level coloured by band, margins 0.5 and 1.0 um",
    )


# --------------------------------------------------------------------------- the side section


def scan(model, y_um: float) -> list[tuple[float, float, str, int | None, bool]]:
    """The cells a horizontal line at ``y_um`` crosses in the z = 0 level: ``(x0, x1, kind, band, under_port)`` along x, in um."""
    layer = model.labelled.level_layers[model.labelled.levels.levels.index(Fraction(0))]
    runs = []
    for fi in range(1, len(layer.arr.faces)):
        crossings = []
        for ring in kit.face_rings_um(layer, fi):
            for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1], strict=True):
                if (y0 <= y_um < y1) or (y1 <= y_um < y0):
                    crossings.append(x0 + (y_um - y0) * (x1 - x0) / (y1 - y0))
        crossings.sort()
        label = layer.labels[fi]
        under_port = label.sheet is not None and label.sheet.kind == "port"
        kind = (
            "outside"
            if not label.bands
            else ("metal" if on_conductor(label) else "gap")
        )
        for a, b in zip(crossings[::2], crossings[1::2], strict=False):
            runs.append(
                (a, b, kind, label.bands[0][1] if label.bands else None, under_port)
            )
    return sorted(runs)


def merged(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Touching intervals joined."""
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and abs(out[-1][1] - a) < 1e-9:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def figure_section(model, ys_um: tuple[float, float] = (50.0, 44.0)) -> dict:
    """x-z sections: the support plane z = 0 carries the metal footprint, the SA surface and the port."""
    thickness = 0.2
    xr, zr = (26.0, 44.0), (-0.75, 0.75)
    info: dict = {}

    def panel(y_um: float) -> Panel:
        runs = scan(model, y_um)
        die = (
            min(a for a, _, kind, _, _ in runs if kind != "outside"),
            max(b for _, b, kind, _, _ in runs if kind != "outside"),
        )
        metal = merged([(a, b) for a, b, kind, _, _ in runs if kind == "metal"])
        gap_edges = sorted(
            x for x in {x for iv in metal for x in iv} - set(die) if xr[0] <= x <= xr[1]
        )
        port = merged([(a, b) for a, b, _, _, under_port in runs if under_port])
        info[f"y={y_um:g}"] = {
            "die_x_um": list(die),
            "metal_x_um": [list(iv) for iv in metal],
            "gap_edges_x_um": gap_edges,
            "port_x_um": [list(iv) for iv in port],
        }
        items = [
            Poly(
                [[(xr[0], zr[0]), (xr[1], zr[0]), (xr[1], 0.0), (xr[0], 0.0)]],
                "#f2efe6",
                "rgba(0,0,0,0)",
                0.0,
            )
        ]
        for a, b in metal:  # the metal: a 0.2 um film standing on the substrate surface
            items.append(
                Poly(
                    [[(a, 0.0), (b, 0.0), (b, thickness), (a, thickness)]],
                    "#9aa0a6",
                    "rgba(0,0,0,0.6)",
                    0.5,
                )
            )
        for a, b, kind, band, under_port in runs:
            if band is None or under_port:
                continue
            if (
                kind == "metal"
            ):  # MS face (bottom, z = 0) and MA face (top, z = 0.2): the same footprint, the same bands
                items.append(Line([a, b], [-0.035, -0.035], METAL[band], 5.0))
                items.append(Line([a, b], [thickness + 0.035] * 2, METAL[band], 5.0))
            elif kind == "gap":
                items.append(Line([a, b], [-0.035, -0.035], GAP[band], 5.0))
        for a, b in port:
            items.append(
                Line([a, b], [0.0, 0.0], kit.PORT, 6.0, name="lumped port on z = 0")
            )
        for x in gap_edges:
            items.append(
                Line(
                    [x, x],
                    [-0.12, thickness + 0.12],
                    ORIGIN,
                    3.0,
                    name="gap edge (distance origin)",
                )
            )
            items.append(
                Label(
                    x + (0.9 if x == min(gap_edges) else -0.9),
                    0.55,
                    "gap edge",
                    ORIGIN,
                    8,
                )
            )
        items += [
            Label(35.0, -0.5, "substrate (Si)", "#555", 9),
            Label(35.0, 0.62, "vacuum", "#555", 9),
            Label(28.0, 0.42, "MA: top face", "#08306b", 8),
            Label(28.0, -0.2, "MS: bottom face", "#08306b", 8),
        ]
        items.append(
            Line(list(xr), [0.0, 0.0], "#222", 0.8, "dot", name="support plane z = 0")
        )
        title = f"section at y = {y_um:g} um" + (
            ", through the port"
            if port
            else ", beside the port: SA bands (orange) on the substrate surface"
        )
        return Panel(
            title,
            xr,
            zr,
            equal=False,
            xlabel="x (um)",
            ylabel="z (um, stretched)",
            items=items,
        )

    panels = [panel(y) for y in ys_um]
    panels[0].items += legend()
    kit.render_svg(
        panels,
        kit.FIGURES / "inset-origin-section.svg",
        ncols=1,
        size=(9.5, 7.4),
        shared_legend=True,
    )
    print("wrote inset-origin-section")
    return info


# --------------------------------------------------------------------------- numbers


def band_areas(sidecar: dict, family_id: str) -> list[dict]:
    """Per band of one band family (the sidecar's ``band_family``): the group names and their total area (um^2).

    Each name of ``bands`` is a group the reader checked; a metal-metal contact no group carries is in
    ``hidden_contacts`` and has no area to add.
    """
    family = sidecar["band_family"][family_id]
    written = sidecar["group"]
    out = []
    for band in sorted(family["bands"], key=int):
        groups = family["bands"][band]
        out.append(
            {
                "band": int(band),
                "groups": groups,
                "area_um2": round(sum(written[g]["area_um2"] for g in groups), 6),
            }
        )
    return out


def describe(model, sidecar: dict, counts: dict) -> dict:
    """What the page quotes of one case."""
    supports = {}
    for family in model.labelled.catalogue.families:
        supports[family.family_id] = {
            "origin_length_um": round(length_um(family.support.origins), 6),
            "free_boundary_length_um": round(
                length_um(family.support.free_boundary), 6
            ),
            "support_area_um2": sidecar["band_family"][family.family_id][
                "support_area_um2"
            ],
            "bands": band_areas(sidecar, family.family_id),
        }
    port_groups = {
        port["group"]: sidecar["group"][port["group"]]["area_um2"]
        for port in sidecar["port"].values()
    }
    return {
        "supports": supports,
        "port_area_um2": port_groups,
        "faces": counts["faces"],
        "groups": counts["groups"],
    }


def no_gap_edge_error() -> str:
    """The ``InputError`` of a ground plane that covers the whole die (a conductor with no gap edge)."""
    request = InsetRequest("ground", ConductorSurface("pad", "sheet"), MARGINS)
    try:
        build_model_from_catalogue(
            slab_model("A", [("pad", "N1", (-40.0, -40.0, 40.0, 40.0))]),
            "A",
            insets=(request,),
        )
    except InputError as error:
        return str(error)
    raise SystemExit("a support without a gap edge was accepted")


def main() -> None:
    """Write the inset-origin figures and numbers."""
    kit.ensure_plotly_js()
    numbers: dict = {"margins_um": list(MARGINS)}
    with tempfile.TemporaryDirectory(prefix="inset-origin-") as scratch_name:
        scratch = Path(scratch_name)
        (scratch / "ma").mkdir()
        effective = gds_pad_in_ground(scratch / "ma")
        model_b, sidecar_b, counts_b = build_case(
            effective, "B", pad_in_ground_requests(), scratch / "ma"
        )
        figure_ma(model_b)
        numbers["pad_in_ground_route_b"] = describe(model_b, sidecar_b, counts_b)
        numbers["section"] = figure_section(model_b)
        model_a, sidecar_a, counts_a = build_case(
            thin_film_pair(), "A", thin_film_requests(), scratch
        )
        figure_sa(model_a)
        numbers["thin_film_pair_route_a"] = describe(model_a, sidecar_a, counts_a)
    numbers["no_gap_edge_error"] = no_gap_edge_error()
    (kit.FIGURES / "inset-origin-numbers.json").write_text(
        json.dumps(numbers, indent=1)
    )
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
