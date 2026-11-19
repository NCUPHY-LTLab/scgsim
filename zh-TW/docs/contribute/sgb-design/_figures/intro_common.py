"""Shared helpers for the figure scripts of the overview, why-plan-first, layered-model and arrangement pages.

Owns: the production import path, output directories, the colour palette, the Plotly and SVG writers, the
running example (pad in ground, port on the support plane) as a prism catalogue, and the drawing of footprints
and arrangement cells.  It derives nothing itself; every geometry comes from ``scgsim.geometry.sgb``.

Run every script with the Python of the sgb-core checkout (branch ``claude/sgb-core``), which has ``scgsim`` with
``geometry.sgb`` installed::

    SGB_CORE=<sgb-core worktree>   # optional; defaults to this checkout
    $SGB_CORE/.artifacts/env/bin/python docs/design/sgb/_figures/<script>.py

The test-side scene builders (``sgb_scenes``, ``sgb_emit_cases``, ``sgb_catalogue_report`` ...) are imported from
``$SGB_CORE/tests``, which is put on ``sys.path`` here at run time; nothing of them is copied.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import LineString, Polygon

HERE = Path(__file__).resolve().parent
SGB_DOCS = HERE.parent
FIG = SGB_DOCS / "figures"
REPO = SGB_DOCS.parents[3]
SGB_CORE = Path(os.environ.get("SGB_CORE", str(REPO)))
sys.path.insert(0, str(SGB_CORE / "tests"))
sys.path.insert(0, str(HERE))

from sgb_scenes import lumped_port, pad_in_ground, rect_um  # noqa: E402

from scgsim.geometry.sgb.arrangement.arcs import sample_arc  # noqa: E402
from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402

FIG.mkdir(parents=True, exist_ok=True)
G_UM = 0.001  # the model grid of the running example in micrometres (1 nm)

COLOURS = {
    "substrate": "#cfe0f1",
    "vacuum": "#fbf6df",
    "ground": "#9aa0a6",
    "pad": "#3a6ea5",
    "port": "#d6453d",
    "wire": "#222222",
    "accent": "#e08a1e",
    "good": "#2a8f5a",
    "bad": "#c0392b",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
    }
)


def save_svg(fig, name: str) -> Path:
    """Save a Matplotlib figure as a static SVG under the figures folder, and as a PNG as well when ``SGB_QA_DIR`` is set."""
    path = FIG / f"{name}.svg"
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None},
    )
    qa = os.environ.get(
        "SGB_QA_DIR"
    )  # optional PNG copy for visual checks while developing
    if qa:
        fig.savefig(
            Path(qa) / f"{name}.png",
            format="png",
            dpi=110,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    return path


def save_plotly(fig: go.Figure, name: str) -> Path:
    """Plotly HTML that loads the one shared ``plotly.min.js`` next to it (plotly writes it if missing)."""
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#222")
    )
    path = FIG / f"{name}.html"
    fig.write_html(path, include_plotlyjs="directory", config={"displaylogo": False})
    return path


def write_numbers(name: str, data: dict) -> Path:
    """Small JSON of the numbers a page quotes (page prose cites this file)."""
    path = FIG / f"{name}-numbers.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return path


# --------------------------------------------------------------------------- the running example


PORT_OUTLINE_UM = (
    50,
    -5,
    80,
    5,
)  # the gap between the pad edge (x = 50) and the ground cutout (x = 80)


def running_example(route: str = "B", *, port_plane: str = "support"):
    """The pad-in-ground prism catalogue under ``route`` with the lumped port on the support plane (z = 0).

    The footprints are the test scene ``sgb_scenes.pad_in_ground`` (chamfer, reflex notch, rounded corner and a
    circular hole, with exact arcs).  Its own Route-B port lies at the metal mid-plane z = 0.1 for the stage
    tests; this function replaces it with the port as the compiler places it (``prisms.ports``): a sheet on the
    plane the hosts' physical declaration names, here the substrate surface z = 0, whatever the route.
    ``port_plane="mid"`` returns the scene's own mid-plane port (an embedded sheet), used only where a figure
    illustrates what such a sheet does.
    """
    catalogue = pad_in_ground("A" if route == "A" else "B", with_port=False)
    plane_route = "A" if port_plane == "support" else "B"
    port = lumped_port("p1", rect_um(*PORT_OUTLINE_UM), plane_route)
    name = f"pad_in_ground_{route}" + ("" if port_plane == "support" else "_midport")
    return replace(catalogue, prisms=(*catalogue.prisms, port), name=name)


def build_example(
    route: str = "B",
    *,
    port_plane: str = "support",
    insets=(),
    **kwargs,
):
    """The validated ``ConformalModel`` of the running example."""
    return build_model_from_catalogue(
        running_example(route, port_plane=port_plane),
        route,
        insets=insets,
        **kwargs,
    )


# --------------------------------------------------------------------------- drawing


def ring_um(ring) -> list[tuple[float, float]]:
    """A ring of grid integers in micrometres."""
    return [(x * G_UM, y * G_UM) for x, y in ring]


def mpl_path(rings: list[list[tuple[float, float]]]) -> MplPath:
    """One compound Matplotlib path from several rings, so holes cut out of their outer ring."""
    verts, codes = [], []
    for ring in rings:
        verts += list(ring) + [ring[0]]
        codes += (
            [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1) + [MplPath.CLOSEPOLY]
        )
    return MplPath(verts, codes)


def draw_footprint(ax, footprint, **kwargs) -> None:
    """A footprint (outer ring plus holes) as one even-odd patch, in micrometres."""
    rings = [ring_um(r) for r in footprint.rings()]
    ax.add_patch(PathPatch(mpl_path(rings), **kwargs))


def cycle_polyline(
    arr, curves, cycle_index: int, step: float = 20.0
) -> list[tuple[float, float]]:
    """The boundary cycle of an arrangement as a polyline in grid units, arc edges sampled every ``step`` grid units."""
    points: list[tuple[float, float]] = []
    for half_edge in arr.cycles[cycle_index].half_edges:
        edge_index = half_edge >> 1
        edge, curve = arr.edges[edge_index], curves[edge_index]
        if curve is None:
            points.append(arr.origin(half_edge))
            continue
        chord = math.hypot(edge.b[0] - edge.a[0], edge.b[1] - edge.a[1])
        samples = sample_arc(curve, edge.a, edge.b, max(3, int(chord / step) + 2))
        points.extend(samples[:-1] if half_edge % 2 == 0 else samples[::-1][:-1])
    return points


def cell_rings_um(
    layer, face_index: int, step: float = 20.0
) -> list[list[tuple[float, float]]]:
    """Boundary rings of an arrangement face with arc edges sampled, in micrometres."""
    arr = layer.arr
    return [
        [
            (x * G_UM, y * G_UM)
            for x, y in cycle_polyline(arr, layer.edge_curve, c, step)
        ]
        for c in arr.face_cycles(face_index)
    ]


def domain_colour(domain) -> str:
    """The fill colour of a labelled domain: conductor, substrate, vacuum or outside."""
    if domain is None:
        return "white"
    if domain.kind == "conductor":
        return COLOURS["pad"] if domain.owner == "pad" else COLOURS["ground"]
    return COLOURS["substrate"] if domain.material == "substrate" else COLOURS["vacuum"]


def domain_name(domain) -> str:
    """The owning entity id of a domain (``none`` outside the model)."""
    return "none" if domain is None else domain.name


def stretch_z(z: float, metal: float = 200.0) -> float:
    """Display-only z map of the running example: the 0.2 um metal layer is drawn ``metal`` times thicker."""
    top = 0.2 * metal
    if z <= 0:
        return z
    if z <= 0.2:
        return z * metal
    return z + top - 0.2


def prism_colour(prism) -> str:
    """The fill colour of a prism by its material kind and owner."""
    if prism.material_kind == "port":
        return COLOURS["port"]
    if prism.material_kind == "conductor":
        return COLOURS["pad"] if prism.entity_id == "pad" else COLOURS["ground"]
    return COLOURS["substrate"] if prism.material == "substrate" else COLOURS["vacuum"]


def footprint_polygon(prism):
    """Shapely polygon of a prism footprint in micrometres (arcs as their polyline witness)."""
    rings = [ring_um(r) for r in prism.footprint.rings()]
    return Polygon(rings[0], rings[1:])


def section_intervals(prism, y_um: float) -> list[tuple[float, float]]:
    """x intervals (micrometres) where the plane y = y_um cuts the prism footprint."""
    cut = footprint_polygon(prism).intersection(LineString([(-1e4, y_um), (1e4, y_um)]))
    if cut.is_empty:
        return []
    parts = list(cut.geoms) if hasattr(cut, "geoms") else [cut]
    return sorted((min(p.coords)[0], max(p.coords)[0]) for p in parts if p.length > 0)


def draw_section(  # noqa: PLR0913 -- the axes, the catalogue and the drawing options of one section
    ax,
    catalogue,
    levels,
    y_um: float,
    x_range=(-155, 155),
    label_slabs: bool = True,
    skip: tuple[str, ...] = (),
) -> None:
    """x-z section of a model at y = y_um with levels drawn equally spaced (the vertical axis is not to scale)."""
    index = {z: k for k, z in enumerate(levels)}
    for prism in catalogue.prisms:
        if prism.entity_id in skip:
            continue
        for x0, x1 in section_intervals(prism, y_um):
            k0, k1 = index[prism.z0], index[prism.z1]
            if prism.is_sheet:
                ax.plot(
                    [x0, x1],
                    [k0, k0],
                    color=prism_colour(prism),
                    lw=3.5,
                    solid_capstyle="butt",
                    zorder=5,
                )
            else:
                alpha = 0.55 if prism.material_kind != "conductor" else 0.95
                ax.add_patch(
                    plt.Rectangle(
                        (x0, k0),
                        x1 - x0,
                        k1 - k0,
                        facecolor=prism_colour(prism),
                        alpha=alpha,
                        edgecolor="#333",
                        lw=0.6,
                        zorder=3 if prism.material_kind != "conductor" else 4,
                    )
                )
    for z, k in index.items():
        ax.axhline(k, color="#555", lw=0.7, ls="--", zorder=6)
        ax.text(
            x_range[1] + 3,
            k,
            f"level {k}: z = {float(z):g} µm",
            va="center",
            ha="left",
            fontsize=8,
        )
    if label_slabs:
        for k in range(len(levels) - 1):
            ax.text(
                x_range[0] + 2,
                k + 0.5,
                f"slab {k}",
                va="center",
                ha="left",
                fontsize=8,
                color="#222",
                zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
            )
    ax.set_xlim(*x_range)
    ax.set_ylim(-0.1, len(levels) - 0.9)
    ax.set_xlabel("x (µm)")
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
