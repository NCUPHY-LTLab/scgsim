"""Shared helpers of the figure scripts of the labels, assembly, routes, families, slivers and validation pages.

Owns: the import path of the production package, output folders, one palette, the Plotly and SVG writers, the drawing of
arrangement cells and edges, the production inputs of the running example, and the surface triangulation used by the 3D
views.  It derives nothing about SGB; every geometry and number comes from ``scgsim.geometry.sgb`` or from the calling
script.

The scripts compute with the production package (branch ``claude/sgb-core``).  Run them with that checkout's interpreter
and point ``SGB_CORE`` at the checkout; its ``tests/`` folder is put on ``sys.path`` at run time for the test-side
builders ``sgb_inputs`` (real inputs), ``sgb_scenes`` (stage-level scenes) and ``sgb_emit_cases`` (named models)::

    SGB_CORE=<sgb-core checkout> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/<script>.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
ROOT = HERE.parents[4]
SGB_CORE = Path(os.environ.get("SGB_CORE", str(ROOT)))
if str(SGB_CORE / "tests") not in sys.path:
    sys.path.insert(0, str(SGB_CORE / "tests"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from dataclasses import replace  # noqa: E402

import gmsh  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scgsim.geometry.sgb.arrangement.arcs import sample_arc  # noqa: E402
from scgsim.geometry.sgb.emission.plan import plan_brep  # noqa: E402
from scgsim.geometry.sgb.emission.writer import write_brep_file, write_xao_file  # noqa: E402
from scgsim.geometry.sgb.model import build_model_from_catalogue  # noqa: E402
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue  # noqa: E402

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
GROUP_COLOURS = {
    "SA": "#9ecae1",
    "MA": "#2171b5",
    "MS": "#6a51a3",
    "MM": "#e08a1e",
    "EXT": "#bdbdbd",
    "PORT": "#d62728",
    "SHEET": "#7f7f7f",
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


def ensure_plotly_js() -> None:
    """The one ``plotly.min.js`` shared by all interactive figures of the design pages."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    target = FIGURES / "plotly.min.js"
    if not target.exists():
        shutil.copyfile(
            Path(plotly.__file__).parent / "package_data" / "plotly.min.js", target
        )


def save_svg(fig, name: str) -> Path:
    """Save a Matplotlib figure as a static SVG under the figures folder."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    qa = os.environ.get(
        "SGB_QA_DIR"
    )  # optional PNG copy for visual checks while developing
    if qa:
        fig.savefig(
            Path(qa) / f"{name}.png",
            format="png",
            dpi=100,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    return path


def save_plotly(fig: go.Figure, name: str) -> Path:
    """Plotly HTML that loads the shared ``plotly.min.js`` next to it."""
    ensure_plotly_js()
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#222")
    )
    path = FIGURES / f"{name}.html"
    fig.write_html(path, include_plotlyjs="directory", config={"displaylogo": False})
    return path


def write_numbers(name: str, data: dict) -> Path:
    """Small JSON with the numbers a page quotes; the page names this file as the source."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}-numbers.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")
    return path


# --------------------------------------------------------------------------- production inputs


def real_input(*, film: bool = False, **options):
    """The production real input of the running example: ``gds_pad_in_ground`` of ``tests/sgb_inputs.py``.

    A 100 um die with a ground plane, a 20 um pad in a 40 um cutout and a lumped port that lands 1 um on each host.
    The metal is a 0.2 um film on the substrate top (attachment face ``lower``), so the port lies on the support
    plane z = 0.  ``options`` are the builder's own keyword arguments (``with_port``, ``pad_polygon``, ...).

    ``film=True`` is the same GDS and stack interpreted with the ``thin_film`` profile, which collapses both metals
    into zero-thickness sheets for Route A: the test builder always uses the ``solid`` profile, so the script swaps
    the profile (and gives both metals one shared physical layer, as one film has) while it builds, then restores both.
    """
    import sgb_inputs  # noqa: PLC0415 -- the test module, found through SGB_CORE

    saved = sgb_inputs.ApproximationProfile, sgb_inputs.physical_layer
    if film:
        profile, layer = saved

        def collapse(name, version, parameters):
            return profile("thin_film", version, parameters)

        def one_film(name, thickness, declaration):
            record = layer(name, thickness, declaration)
            if record:
                record["physical_layer"]["physical_layer_id"] = "metal_film"
            return record

        sgb_inputs.ApproximationProfile, sgb_inputs.physical_layer = collapse, one_film
    try:
        with tempfile.TemporaryDirectory() as tmp:
            return sgb_inputs.gds_pad_in_ground(Path(tmp), **options)
    finally:
        sgb_inputs.ApproximationProfile, sgb_inputs.physical_layer = saved


def real_catalogue(*, film: bool = False, **options):
    """The prism catalogue (stage S2) of :func:`real_input`."""

    effective = real_input(film=film, **options)
    return build_prism_catalogue(effective.build_input, effective.representation)


def build_model(
    catalogue, route: str, *, insets=(), refinements=(), keep_raw: bool = False
):
    """The validated conformal model of a catalogue under a route (stages S4 to S9)."""

    return build_model_from_catalogue(
        catalogue, route, insets=insets, refinements=refinements, keep_raw=keep_raw
    )


def with_extra_families(catalogue, *families):
    """The catalogue with hand-made ``CurveFamilyRecord`` objects appended (the typed requests cannot express them)."""

    return replace(catalogue, families=(*catalogue.families, *families))


# --------------------------------------------------------------------------- 2D drawing


def mpl_path(rings) -> MplPath:
    """One compound Matplotlib path from several rings, so holes cut out of their outer ring."""
    verts, codes = [], []
    for ring in rings:
        verts += list(ring) + [ring[0]]
        codes += (
            [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1) + [MplPath.CLOSEPOLY]
        )
    return MplPath(verts, codes)


def svg_path(rings) -> str:
    """An SVG path string that closes every ring of a polygon with holes."""
    return " ".join(
        "M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in ring) + " Z" for ring in rings
    )


def cycle_polyline(
    arr, curves, cycle_index: int, step: float = 150.0
) -> list[tuple[float, float]]:
    """A boundary cycle as a closed float polyline, grid units, every realised arc edge sampled."""

    pts: list[tuple[float, float]] = []
    for h in arr.cycles[cycle_index].half_edges:
        a = arr.origin(h)
        curve = curves[h >> 1]
        if curve is None:
            pts.append((float(a[0]), float(a[1])))
            continue
        ea, eb = arr.edges[h >> 1].a, arr.edges[h >> 1].b
        chord = math.hypot(eb[0] - ea[0], eb[1] - ea[1])
        sampled = sample_arc(curve, ea, eb, max(3, int(chord / step) + 2))
        pts.extend(sampled[:-1] if h % 2 == 0 else reversed(sampled[1:]))
    return pts


def cell_rings_um(layer, face_index: int, step: float = 150.0, arr=None, curves=None):
    """Boundary rings of an arrangement face (arcs sampled) in micrometres.

    ``arr``/``curves`` default to the layer's dissolved arrangement; pass the raw arrangement for pre-dissolution cells
    (those have no realised arcs: pass ``curves=[None] * len(arr.edges)``).
    """
    arr = arr if arr is not None else layer.arr
    curves = curves if curves is not None else layer.edge_curve
    return [
        [(x * G_UM, y * G_UM) for x, y in cycle_polyline(arr, curves, c, step)]
        for c in arr.face_cycles(face_index)
    ]


def edge_polyline_um(
    arr, curves, ei: int, step: float = 150.0
) -> list[tuple[float, float]]:
    """One arrangement edge as a polyline in micrometres (arc edges sampled)."""

    e = arr.edges[ei]
    curve = curves[ei]
    if curve is None:
        pts = [e.a, e.b]
    else:
        chord = math.hypot(e.b[0] - e.a[0], e.b[1] - e.a[1])
        pts = sample_arc(curve, e.a, e.b, max(3, int(chord / step) + 2))
    return [(x * G_UM, y * G_UM) for x, y in pts]


def finish_plan(ax, xlim, ylim, title: str | None = None) -> None:
    """Fix the limits and equal aspect of a plan view and label its axes in micrometres."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    if title:
        ax.set_title(title)


def plotly_cells(  # noqa: PLR0913 -- the figure, its cell, the colour and text rules and the axes ranges
    fig,
    layer,
    colour_of,
    text_of,
    row: int,
    col: int,
    xlim,
    ylim,
    line="rgba(0,0,0,0.55)",
) -> None:
    """Add the cells of a layer as filled even-odd paths with one hover marker per cell (``colour_of(layer, face)``)."""
    hx, hy, ht = [], [], []
    for fi in range(1, len(layer.arr.faces)):
        rings = cell_rings_um(layer, fi)
        fig.add_shape(
            type="path",
            path=svg_path(rings),
            fillcolor=colour_of(layer, fi),
            fillrule="evenodd",
            line=dict(color=line, width=0.5),
            layer="below",
            row=row,
            col=col,
        )
        x, y = rings[0][0]
        hx.append(x)
        hy.append(y)
        ht.append(text_of(layer, fi))
    fig.add_trace(
        go.Scatter(
            x=hx,
            y=hy,
            mode="markers",
            marker=dict(size=4, color="black"),
            text=ht,
            hoverinfo="text",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(range=list(xlim), row=row, col=col)
    fig.update_yaxes(range=list(ylim), row=row, col=col)


# --------------------------------------------------------------------------- label colours and names

# colour of a domain by its owning entity (production names: the real input's entities are what the groups show)
OWNER_COLOURS = {
    "ground_0000": PALETTE["ground"],
    "ground": PALETTE["ground"],
    "pad": PALETTE["pad"],
}


def domain_colour(domain) -> str:
    """The fill colour of a labelled domain: conductor, substrate, vacuum or outside."""
    if domain is None:
        return PALETTE["none"]
    if domain.kind == "conductor":
        return OWNER_COLOURS.get(domain.name, PALETTE["accent"])
    return PALETTE["substrate"] if domain.kind == "dielectric" else PALETTE["vacuum"]


def domain_name(domain) -> str:
    """The owning entity id of a domain, or ``outside`` for no domain."""
    return "outside" if domain is None else domain.name


def level_label_text(label) -> str:
    """Hover text of a level label: the domains below and above it and its sheet, if any."""
    text = f"below {domain_name(label.below)} | above {domain_name(label.above)}"
    if label.sheet is not None:
        text += f" | sheet {label.sheet.kind}:{label.sheet.name}"
    if label.bands:
        text += " | " + ", ".join(f"{f}:band {k}" for f, k in label.bands)
    return text


# --------------------------------------------------------------------------- 3D


def stretch_z(z, metal: float = 200.0, top: float = 0.2):
    """Display-only z map of the ``top`` um metal layer of the running example: drawn ``metal`` times thicker."""
    z = np.asarray(z, dtype=float)
    return np.where(z <= 0, z, np.where(z <= top, z * metal, z + top * metal - top))


@contextmanager
def loaded_model(model) -> Iterator[dict[str, tuple[int, int]]]:
    """Write the model's BREP/XAO with the production writer, open it in Gmsh and yield ``planned id -> (dim, tag)``."""

    plan = plan_brep(model.topology, model.routed)
    with tempfile.TemporaryDirectory() as tmp:
        brep, xao = Path(tmp) / "m.brep", Path(tmp) / "m.xao"
        write_brep_file(plan, brep)
        write_xao_file(plan, xao, name="m", brep=brep)
        gmsh.initialize(["-noenv"])
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
            gmsh.open(str(xao))
            gmsh.model.occ.synchronize()
            yield plan.tags()
        finally:
            gmsh.finalize()


def compact_triangles(
    model, mesh_size: float, mesh_size_min: float = 0.0
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per emitted face id ``(points, triangles)`` of a surface mesh of the written model; only the points that face uses.

    The surface mesh is Gmsh's, made from the XAO that the production writer produced (display only).  A face that
    bounds no emitted volume is not in the file, so use Route C when every face of every volume is wanted.
    """

    out = {}
    with loaded_model(model) as tags:
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.model.mesh.generate(2)
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        index = {int(t): i for i, t in enumerate(node_tags)}
        xyz = np.array(coords).reshape(-1, 3)
        for fid in model.topology.faces:
            dim, tag = tags.get(fid, (None, None))
            if dim != 2:
                continue
            _, _, nodes = gmsh.model.mesh.getElements(2, tag)
            if not len(nodes):
                continue
            tri = np.array([index[int(n)] for n in nodes[0]]).reshape(-1, 3)
            used, inverse = np.unique(tri, return_inverse=True)
            out[fid] = (xyz[used], inverse.reshape(-1, 3))
    return out


def mesh_trace(points, tri, colour, name, hover, opacity=0.6, **kw) -> go.Mesh3d:  # noqa: PLR0913 -- the mesh, its look and its hover text
    """A flat-shaded Plotly triangle mesh that keeps its hover text out of the legend."""
    return go.Mesh3d(
        x=np.round(points[:, 0], 3),
        y=np.round(points[:, 1], 3),
        z=np.round(points[:, 2], 3),
        i=tri[:, 0],
        j=tri[:, 1],
        k=tri[:, 2],
        color=colour,
        opacity=opacity,
        flatshading=True,
        name=name,
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
        **kw,
    )
