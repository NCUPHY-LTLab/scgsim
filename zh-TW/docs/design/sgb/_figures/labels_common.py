"""Shared helpers of the figure scripts of the labels, assembly, routes, families, slivers and validation pages.

Owns: the read-only prototype import path, output folders, one palette, the Plotly and SVG writers, the drawing of
arrangement cells and edges, and compact face triangulation for 3D views. It derives nothing about SGB; every
geometry and number comes from the prototype (``experiments/sgb-redesign/prototype``) or from the calling script.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
PROTOTYPE = HERE.parents[4] / "experiments" / "sgb-redesign" / "prototype"
OUTPUTS = PROTOTYPE / "outputs"
if str(PROTOTYPE) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

G_UM = 0.001  # um per grid unit (default grid, 1 nm)

PALETTE = {
    "substrate": "#cfe0f1",
    "vacuum": "#fbf6df",
    "ground": "#9aa0a6",
    "pad": "#3a6ea5",
    "port": "#d6453d",
    "ink": "#222222",
    "accent": "#e08a1e",
    "good": "#2a8f5a",
    "bad": "#c0392b",
    "seam": "#c0392b",
    "band": "#2171b5",
    "cap": "#2a8f5a",
    "refine": "#8e44ad",
    "none": "#f2f2f2",
}
# primary physical-group prefix -> colour (shared by the 3D views)
GROUP_COLOURS = {"SA": "#9ecae1", "MA": "#2171b5", "MS": "#6a51a3", "EXT": "#bdbdbd", "PORT": "#d62728", "PEC": "#7f7f7f"}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
})


def ensure_plotly_js() -> None:
    """The one ``plotly.min.js`` shared by all interactive figures of the design pages."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    target = FIGURES / "plotly.min.js"
    if not target.exists():
        import plotly

        shutil.copyfile(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", target)


def save_svg(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    qa = os.environ.get("SGB_QA_DIR")  # optional PNG copy for visual checks while developing
    if qa:
        fig.savefig(Path(qa) / f"{name}.png", format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_plotly(fig: go.Figure, name: str) -> Path:
    """Plotly HTML that loads the shared ``plotly.min.js`` next to it."""
    ensure_plotly_js()
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#222"))
    path = FIGURES / f"{name}.html"
    fig.write_html(path, include_plotlyjs="directory", config={"displaylogo": False})
    return path


def write_numbers(name: str, data: dict) -> Path:
    """Small JSON with the numbers a page quotes; the page names this file as the source."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}-numbers.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")
    return path


# --------------------------------------------------------------------------- 2D drawing


def mpl_path(rings) -> MplPath:
    verts, codes = [], []
    for ring in rings:
        verts += list(ring) + [ring[0]]
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def svg_path(rings) -> str:
    return " ".join("M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in ring) + " Z" for ring in rings)


def cell_rings_um(layer, face_index: int, step: float = 150.0, arr=None, curves=None):
    """Boundary rings of an arrangement face (arcs sampled) in micrometres.

    ``arr``/``curves`` default to the layer's dissolved arrangement; pass the raw arrangement for pre-dissolution cells
    (those have no realised arcs: pass ``curves=[None] * len(arr.edges)``).
    """
    from sgbproto.arcs import cycle_polyline

    arr = arr if arr is not None else layer.arr
    curves = curves if curves is not None else layer.edge_curve
    return [[(x * G_UM, y * G_UM) for x, y in cycle_polyline(arr, curves, c, step)] for c in arr.face_cycles(face_index)]


def edge_polyline_um(arr, curves, ei: int, step: float = 150.0) -> list[tuple[float, float]]:
    """One arrangement edge as a polyline in micrometres (arc edges sampled)."""
    import math

    from sgbproto.arcs import sample_arc

    e = arr.edges[ei]
    curve = curves[ei]
    if curve is None:
        pts = [e.a, e.b]
    else:
        chord = math.hypot(e.b[0] - e.a[0], e.b[1] - e.a[1])
        pts = sample_arc(curve, e.a, e.b, max(3, int(chord / step) + 2))
    return [(x * G_UM, y * G_UM) for x, y in pts]


def draw_cells(ax, layer, colour_of, alpha: float = 1.0, edgecolor="#222", lw: float = 0.5) -> None:
    for fi in range(1, len(layer.arr.faces)):
        ax.add_patch(PathPatch(mpl_path(cell_rings_um(layer, fi)), facecolor=colour_of(layer.labels[fi]), alpha=alpha,
                               edgecolor=edgecolor, lw=lw))


def finish_plan(ax, xlim, ylim, title: str | None = None) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    if title:
        ax.set_title(title)


def plotly_cells(fig, layer, colour_of, text_of, row: int, col: int, xlim, ylim, line="rgba(0,0,0,0.55)") -> None:
    """Add the cells of a layer as filled even-odd paths with one hover marker per cell (``colour_of(layer, face)``)."""
    hx, hy, ht = [], [], []
    for fi in range(1, len(layer.arr.faces)):
        rings = cell_rings_um(layer, fi)
        fig.add_shape(type="path", path=svg_path(rings), fillcolor=colour_of(layer, fi), fillrule="evenodd",
                      line=dict(color=line, width=0.5), layer="below", row=row, col=col)
        x, y = rings[0][0]
        hx.append(x)
        hy.append(y)
        ht.append(text_of(layer, fi))
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=4, color="black"), text=ht, hoverinfo="text",
                             showlegend=False), row=row, col=col)
    fig.update_xaxes(range=list(xlim), row=row, col=col)
    fig.update_yaxes(range=list(ylim), row=row, col=col)


# --------------------------------------------------------------------------- label colours and names


def domain_colour(domain) -> str:
    if domain is None:
        return PALETTE["none"]
    if domain.kind == "conductor":
        return PALETTE["pad"] if domain.net == "pad" else PALETTE["ground"] if domain.net == "Ground" else PALETTE["accent"]
    return PALETTE["substrate"] if domain.material == "substrate" else PALETTE["vacuum"]


def domain_name(domain) -> str:
    return "outside" if domain is None else (domain.net if domain.kind == "conductor" else domain.material)


def level_label_text(label) -> str:
    text = f"below {domain_name(label.below)} | above {domain_name(label.above)}"
    if label.sheet is not None:
        text += f" | sheet {label.sheet.kind}:{label.sheet.name}"
    if label.bands:
        text += " | " + ", ".join(f"{f}:band {k}" for f, k in label.bands)
    return text


# --------------------------------------------------------------------------- 3D


def stretch_z(z, metal: float = 200.0):
    """Display-only z map of the 0.2 um metal layer of the running example: drawn ``metal`` times thicker."""
    z = np.asarray(z, dtype=float)
    top = 0.2 * metal
    return np.where(z <= 0, z, np.where(z <= 0.2, z * metal, z + top - 0.2))


def compact_triangles(topo, mesh_size: float, mesh_size_min: float = 0.0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per face id ``(points, triangles)`` with only the points that face uses (small HTML)."""
    from sgbproto.emit import surface_triangles

    out = {}
    for fid, (xyz, tri) in surface_triangles(topo, mesh_size, mesh_size_min).items():
        used, inverse = np.unique(tri, return_inverse=True)
        out[fid] = (xyz[used], inverse.reshape(-1, 3))
    return out


def mesh_trace(points, tri, colour, name, hover, opacity=0.6, **kw) -> go.Mesh3d:
    return go.Mesh3d(x=np.round(points[:, 0], 3), y=np.round(points[:, 1], 3), z=np.round(points[:, 2], 3),
                     i=tri[:, 0], j=tri[:, 1], k=tri[:, 2], color=colour, opacity=opacity, flatshading=True,
                     name=name, hovertext=hover, hoverinfo="text", showlegend=False, **kw)
