"""Shared helpers for the inset, identity and parallel figure scripts.

Owns three things only:

* locating the read-only prototype and the output folder (``docs/design/sgb/figures``);
* one small scene description (panels of lines, polygons, points and labels) that is drawn twice,
  as a self-contained-page Plotly HTML (sharing one ``plotly.min.js``) and as a matplotlib SVG;
* the colour palette used on those pages.

It computes nothing about SGB; every number comes from the prototype or from the page's own script.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
PROTOTYPE = HERE.parents[4] / "experiments" / "sgb-redesign" / "prototype"
PROTOTYPE_OUTPUTS = PROTOTYPE / "outputs"

if str(PROTOTYPE) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

G_UM = 0.001  # um per grid unit (default grid, 1 nm)

# One colour family per kind of thing; sequential ramps run from the edge band (dark) inwards (light).
PAD_BANDS = ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]
GAP_BANDS = ["#7f2704", "#d94801", "#fd8d3c", "#fdd0a2"]
PORT = "#b2182b"
GROUND = "#bdbdbd"
NEUTRAL = "#e8e8e8"
INK = "#1b1b1b"
EXACT = "#1f77b4"
OTHER = "#d62728"
ACCENT = "#e08214"


def ensure_plotly_js() -> None:
    """The one ``plotly.min.js`` shared by every interactive figure of the design pages."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    target = FIGURES / "plotly.min.js"
    if not target.exists():
        import plotly

        shutil.copyfile(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", target)


# --------------------------------------------------------------------------- scene description


@dataclass
class Line:
    xs: list[float]
    ys: list[float]
    color: str = INK
    width: float = 1.0
    dash: str = "solid"  # solid | dot | dash
    name: str = ""
    hover: str = ""


@dataclass
class Poly:
    rings: list[list[tuple[float, float]]]  # outer + holes (even-odd)
    fill: str = NEUTRAL
    edge: str = "rgba(0,0,0,0.55)"
    edge_width: float = 0.5
    alpha: float = 1.0
    name: str = ""
    hover: str = ""


@dataclass
class Pts:
    xs: list[float]
    ys: list[float]
    color: str = INK
    size: float = 4.0
    name: str = ""
    hover: list[str] | str = ""
    symbol: str = "circle"


@dataclass
class Label:
    x: float
    y: float
    text: str
    color: str = INK
    size: float = 10.0


@dataclass
class Panel:
    title: str = ""
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    equal: bool = True
    xlabel: str = ""
    ylabel: str = ""
    items: list = field(default_factory=list)


def _svg_path(rings) -> str:
    return " ".join("M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in ring) + " Z" for ring in rings)


def _mpl_path(rings) -> MplPath:
    verts, codes = [], []
    for ring in rings:
        verts += list(ring) + [ring[0]]
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def _mpl_colour(colour: str):
    """matplotlib colour from a hex string or a Plotly ``rgba(r,g,b,a)`` string."""
    if colour.startswith("rgba("):
        r, g, b, a = (float(v) for v in colour[5:-1].split(","))
        return (r / 255, g / 255, b / 255, a)
    return colour


_MPL_DASH = {"solid": "-", "dot": ":", "dash": "--"}


def save_svg(fig, stem: str) -> None:
    """Write ``figures/<stem>.svg`` from a matplotlib figure (and a PNG copy when SGB_QA_DIR is set)."""
    fig.savefig(FIGURES / f"{stem}.svg", format="svg", bbox_inches="tight")
    preview = os.environ.get("SGB_QA_DIR") or os.environ.get("SGB_PREVIEW_DIR")  # optional PNG copy for a quick look
    if preview:
        fig.savefig(Path(preview) / f"{stem}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_svg(panels: list[Panel], path: Path, ncols: int = 1, size: tuple[float, float] = (8, 5.5), suptitle: str = "",
               shared_legend: bool = False) -> None:
    """Static SVG of a scene (text kept as text, white background)."""
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False)
    fig.patch.set_facecolor("white")
    flat = [ax for row in axes for ax in row]
    all_legend: dict = {}
    for ax, panel in zip(flat, panels):
        ax.set_facecolor("white")
        legend = {}
        for item in panel.items:
            if isinstance(item, Poly):
                ax.add_patch(PathPatch(_mpl_path(item.rings), facecolor=item.fill, edgecolor=_mpl_colour(item.edge) if item.edge_width else "none",
                                       lw=item.edge_width, alpha=item.alpha, fill=True))
            elif isinstance(item, Line):
                (h,) = ax.plot(item.xs, item.ys, color=item.color, lw=item.width, ls=_MPL_DASH[item.dash])
                if item.name:
                    legend.setdefault(item.name, h)
            elif isinstance(item, Pts):
                h = ax.scatter(item.xs, item.ys, s=item.size**2 * 0.6, c=item.color, marker={"circle": "o", "square": "s", "x": "x"}.get(item.symbol, "o"),
                               linewidths=0.5, zorder=3)
                if item.name:
                    legend.setdefault(item.name, h)
            elif isinstance(item, Label):
                ax.text(item.x, item.y, item.text, color=item.color, fontsize=item.size - 1, ha="center", va="center", zorder=5)
        if panel.xlim:
            ax.set_xlim(*panel.xlim)
        if panel.ylim:
            ax.set_ylim(*panel.ylim)
        if panel.equal:
            ax.set_aspect("equal")
        ax.set_title(panel.title, fontsize=9)
        ax.set_xlabel(panel.xlabel, fontsize=8)
        ax.set_ylabel(panel.ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        all_legend.update({k: v for k, v in legend.items() if k not in all_legend})
        if legend and not shared_legend:
            ax.legend(legend.values(), legend.keys(), fontsize=6.5, loc="best", frameon=True)
    for ax in flat[len(panels):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    if shared_legend and all_legend:
        fig.legend(all_legend.values(), all_legend.keys(), fontsize=8, loc="lower center", ncol=min(3, len(all_legend)), frameon=False,
                   bbox_to_anchor=(0.5, -0.06))
    save_svg(fig, Path(path).stem)


def render_html(
    panels: list[Panel],
    path: Path,
    ncols: int = 1,
    size: tuple[int, int] = (1000, 700),
    title: str = "",
    buttons: dict[str, tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> None:
    """Interactive Plotly page; the library is the shared ``plotly.min.js`` next to the file."""
    ensure_plotly_js()
    nrows = -(-len(panels) // ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[p.title for p in panels], horizontal_spacing=0.07, vertical_spacing=0.12)
    shown: set[str] = set()
    for index, panel in enumerate(panels):
        row, col = index // ncols + 1, index % ncols + 1
        axis_suffix = "" if index == 0 else str(index + 1)
        for item in panel.items:
            if isinstance(item, Poly):
                fig.add_shape(type="path", path=_svg_path(item.rings), fillcolor=item.fill, fillrule="evenodd", opacity=item.alpha,
                              line=dict(color=item.edge, width=item.edge_width), layer="below", row=row, col=col)
                if item.hover:
                    x, y = item.rings[0][0]
                    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(size=5, color="rgba(0,0,0,0.35)"),
                                             hoverinfo="text", text=[item.hover], showlegend=False), row=row, col=col)
            elif isinstance(item, Line):
                first = bool(item.name) and item.name not in shown
                shown.add(item.name)
                fig.add_trace(go.Scatter(x=item.xs, y=item.ys, mode="lines", line=dict(color=item.color, width=item.width, dash=item.dash),
                                         name=item.name or None, legendgroup=item.name or None, showlegend=first,
                                         hoverinfo="text" if item.hover else "skip", text=item.hover or None), row=row, col=col)
            elif isinstance(item, Pts):
                first = bool(item.name) and item.name not in shown
                shown.add(item.name)
                fig.add_trace(go.Scatter(x=item.xs, y=item.ys, mode="markers", marker=dict(size=item.size, color=item.color, symbol=item.symbol),
                                         name=item.name or None, legendgroup=item.name or None, showlegend=first,
                                         hoverinfo="text" if item.hover else "skip", text=item.hover or None), row=row, col=col)
            elif isinstance(item, Label):
                fig.add_annotation(x=item.x, y=item.y, text=item.text, showarrow=False, font=dict(size=item.size, color=item.color),
                                   xref=f"x{axis_suffix}", yref=f"y{axis_suffix}")
        fig.update_xaxes(range=list(panel.xlim) if panel.xlim else None, title=panel.xlabel, row=row, col=col, constrain="domain")
        fig.update_yaxes(range=list(panel.ylim) if panel.ylim else None, title=panel.ylabel, row=row, col=col,
                         **({"scaleanchor": f"x{axis_suffix}", "scaleratio": 1} if panel.equal else {}))
    layout = dict(title=title, width=size[0], height=size[1], paper_bgcolor="white", plot_bgcolor="white", font=dict(color=INK, size=12),
                  margin=dict(l=60, r=30, t=90 if buttons else 70, b=50), legend=dict(orientation="h", y=-0.12))
    if buttons:
        layout["updatemenus"] = [dict(type="buttons", direction="right", x=0, y=1.14, buttons=[
            dict(label=name, method="relayout", args=[{"xaxis.range": list(xr), "yaxis.range": list(yr)}]) for name, (xr, yr) in buttons.items()])]
    fig.update_layout(**layout)
    fig.write_html(path, include_plotlyjs="directory", full_html=True)
    # The directory mode writes plotly.min.js next to the page; keep only the one shared copy.
    FIGURES.joinpath("plotly.min.js").exists() or ensure_plotly_js()


def write_both(panels: list[Panel], stem: str, ncols: int = 1, svg_size: tuple[float, float] = (8, 5.5), html_size: tuple[int, int] = (1000, 700),
               title: str = "", buttons=None, shared_legend: bool = False) -> None:
    render_svg(panels, FIGURES / f"{stem}.svg", ncols=ncols, size=svg_size, suptitle=title, shared_legend=shared_legend)
    render_html(panels, FIGURES / f"{stem}.html", ncols=ncols, size=html_size, title=title, buttons=buttons)
    print("wrote", stem, f"({(FIGURES / f'{stem}.html').stat().st_size // 1024} kB html, {(FIGURES / f'{stem}.svg').stat().st_size // 1024} kB svg)")
