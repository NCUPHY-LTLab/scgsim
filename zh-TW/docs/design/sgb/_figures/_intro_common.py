"""Shared helpers for the figure scripts of the overview, why-plan-first, layered-model and arrangement pages.

Owns: the prototype import path, output directories, the colour palette, the Plotly and SVG writers, and the
drawing of footprints and arrangement cells. It derives nothing; every geometry comes from the prototype.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
SGB_DOCS = HERE.parent
FIG = SGB_DOCS / "figures"
REPO = SGB_DOCS.parents[3]
PROTOTYPE = REPO / "experiments" / "sgb-redesign" / "prototype"
OUTPUTS = PROTOTYPE / "outputs"
sys.path.insert(0, str(PROTOTYPE))

FIG.mkdir(parents=True, exist_ok=True)
G_UM = 0.001  # grid unit of the prototype in micrometres

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

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
})


def save_svg(fig, name: str) -> Path:
    path = FIG / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    qa = os.environ.get("SGB_QA_DIR")  # optional PNG copy for visual checks while developing
    if qa:
        fig.savefig(Path(qa) / f"{name}.png", format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_plotly(fig: go.Figure, name: str) -> Path:
    """Plotly HTML that loads the one shared ``plotly.min.js`` next to it (plotly writes it if missing)."""
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#222"))
    path = FIG / f"{name}.html"
    fig.write_html(path, include_plotlyjs="directory", config={"displaylogo": False})
    return path


def write_numbers(name: str, data: dict) -> Path:
    """Small JSON of the numbers a page quotes (page prose cites this file)."""
    path = FIG / f"{name}-numbers.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return path


def ring_um(ring) -> list[tuple[float, float]]:
    return [(x * G_UM, y * G_UM) for x, y in ring]


def mpl_path(rings: list[list[tuple[float, float]]]) -> MplPath:
    verts, codes = [], []
    for ring in rings:
        verts += list(ring) + [ring[0]]
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def draw_footprint(ax, footprint, **kwargs) -> None:
    """A footprint (outer ring plus holes) as one even-odd patch, in micrometres."""
    rings = [ring_um(r) for r in footprint.rings()]
    ax.add_patch(PathPatch(mpl_path(rings), **kwargs))


def cell_rings_um(layer, face_index: int, step: float = 20.0) -> list[list[tuple[float, float]]]:
    """Boundary rings of an arrangement face with arc edges sampled, in micrometres."""
    from sgbproto.arcs import cycle_polyline

    arr = layer.arr
    return [[(x * G_UM, y * G_UM) for x, y in cycle_polyline(arr, layer.edge_curve, c, step)] for c in arr.face_cycles(face_index)]


def domain_colour(domain) -> str:
    if domain is None:
        return "white"
    if domain.kind == "conductor":
        return COLOURS["pad"] if domain.net == "pad" else COLOURS["ground"]
    return COLOURS["substrate"] if domain.material == "substrate" else COLOURS["vacuum"]


def domain_name(domain) -> str:
    return "none" if domain is None else (domain.net if domain.kind == "conductor" else domain.material)


def stretch_z(z: float, metal: float = 200.0) -> float:
    """Display-only z map of the running example: the 0.2 um metal layer is drawn ``metal`` times thicker."""
    top = 0.2 * metal
    if z <= 0:
        return z
    if z <= 0.2:
        return z * metal
    return z + top - 0.2


def prism_colour(prism) -> str:
    if prism.role == "port":
        return COLOURS["port"]
    if prism.material_kind == "conductor":
        return COLOURS["pad"] if prism.net_id == "pad" else COLOURS["ground"]
    return COLOURS["substrate"] if prism.material == "substrate" else COLOURS["vacuum"]


def footprint_polygon(prism):
    """Shapely polygon of a prism footprint in micrometres (arcs as their polyline witness)."""
    from shapely.geometry import Polygon

    rings = [ring_um(r) for r in prism.footprint.rings()]
    return Polygon(rings[0], rings[1:])


def section_intervals(prism, y_um: float) -> list[tuple[float, float]]:
    """x intervals (micrometres) where the plane y = y_um cuts the prism footprint."""
    from shapely.geometry import LineString

    cut = footprint_polygon(prism).intersection(LineString([(-1e4, y_um), (1e4, y_um)]))
    if cut.is_empty:
        return []
    parts = list(cut.geoms) if hasattr(cut, "geoms") else [cut]
    return sorted((min(p.coords)[0], max(p.coords)[0]) for p in parts if p.length > 0)


def draw_section(ax, spec, levels, y_um: float, x_range=(-155, 155), label_slabs: bool = True,
                 skip: tuple[str, ...] = ()) -> None:
    """x-z section of a model at y = y_um with levels drawn equally spaced (the vertical axis is not to scale)."""
    index = {z: k for k, z in enumerate(levels)}
    for prism in spec.prisms:
        if prism.entity_id in skip:
            continue
        for x0, x1 in section_intervals(prism, y_um):
            k0, k1 = index[prism.z0], index[prism.z1]
            if prism.is_sheet:
                ax.plot([x0, x1], [k0, k0], color=prism_colour(prism), lw=3.5, solid_capstyle="butt", zorder=5)
            else:
                alpha = 0.55 if prism.material_kind != "conductor" else 0.95
                ax.add_patch(plt.Rectangle((x0, k0), x1 - x0, k1 - k0, facecolor=prism_colour(prism), alpha=alpha,
                                           edgecolor="#333", lw=0.6, zorder=3 if prism.material_kind != "conductor" else 4))
    for z, k in index.items():
        ax.axhline(k, color="#555", lw=0.7, ls="--", zorder=6)
        ax.text(x_range[1] + 3, k, f"level {k}: z = {float(z):g} µm", va="center", ha="left", fontsize=8)
    if label_slabs:
        for k in range(len(levels) - 1):
            ax.text(x_range[0] + 2, k + 0.5, f"slab {k}", va="center", ha="left", fontsize=8, color="#222", zorder=10,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    ax.set_xlim(*x_range)
    ax.set_ylim(-0.1, len(levels) - 0.9)
    ax.set_xlabel("x (µm)")
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
