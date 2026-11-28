"""Figures of arrangement.qmd: iterated snap rounding on a tiny input, step by step.

The input is five exact integer carriers on a 12 x 12 grid (a triangle c1-c3, a long crossing segment c4 and a short
segment c5). The script re-runs the production kernels of ``scgsim.geometry.sgb.arrangement.noding.snap_rounding``
(``crossing_pixels_kernel``, ``hits_kernel``, ``assemble_chain``) round by round, records
every state, checks that the result equals ``snap_round`` and then builds the DCEL faces with
``scgsim.geometry.sgb.arrangement.dcel.build_arrangement``.

Writes ``figures/arrangement-steps.svg`` (six panels), ``figures/arrangement-steps.html`` (one step at a time)
and ``figures/arrangement-steps-numbers.json``.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from intro_common import save_plotly, save_svg, write_numbers

from scgsim.geometry.sgb.arrangement.dcel import build_arrangement, interior_point
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import (
    NodedEdge,
    Segment,
    assemble_chain,
    crossing_pixels_kernel,
    hits_kernel,
    snap_round,
)
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import (
    canonical_key as canonical,
)
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import (
    proper_crossing as _crossing,
)

CARRIERS = [
    ((1, 1), (10, 4)),
    ((10, 4), (12, 9)),
    ((12, 9), (1, 1)),
    ((2, 0), (12, 8)),
    ((6, 2), (10, 5)),
]
NAMES = [f"c{i + 1}" for i in range(len(CARRIERS))]
CARRIER_COLOUR = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
FACE_COLOURS = ["#cfe0f1", "#fbe3b8", "#cfe8cf", "#e6d3ee", "#f6c6c6"]


# --------------------------------------------------------------------------- the round-by-round trace


def _trace(carriers):
    segments = [
        Segment(a, b, frozenset([n])) for (a, b), n in zip(carriers, NAMES, strict=True)
    ]
    edges: dict = {}
    for seg in segments:
        edges.setdefault(canonical(seg.a, seg.b), set()).update(seg.carriers)
    hot = {p for key in edges for p in key}
    endpoints = set(hot)
    changed, delta = set(edges), set(hot)
    rounds = []
    while changed or delta:
        keys = sorted(edges)
        flags = [k in changed for k in keys]
        new_hot = crossing_pixels_kernel(keys, flags) - hot
        hot |= new_hot
        delta |= new_hot
        hits = hits_kernel(keys, flags, hot, delta)
        new_edges: dict = {}
        verified = set()
        routes = {}
        for key, h in zip(keys, hits, strict=True):
            chain = assemble_chain(key[0], key[1], h)
            routes[key] = chain
            if len(chain) == 2:
                verified.add(key)
            for u, v in zip(chain, chain[1:], strict=False):
                new_edges.setdefault(canonical(u, v), set()).update(edges[key])
        rounds.append(
            {
                "before": dict(edges),
                "new_hot": set(new_hot),
                "hot": set(hot),
                "after": new_edges,
                "routes": routes,
                "changed_in": set(changed),
            }
        )
        changed = set(new_edges) - verified
        delta = set()
        edges = new_edges
    return {"endpoints": endpoints, "rounds": rounds, "final": edges, "hot": hot}


def exact_crossings(carriers):
    """Exact proper crossing points of the input carriers, as (x, y, carrier a, carrier b)."""
    out = []
    for i in range(len(carriers)):
        for j in range(i + 1, len(carriers)):
            hit = _crossing(*carriers[i], *carriers[j])
            if hit is not None:
                x, y, den = hit
                out.append((Fraction(x, den), Fraction(y, den), NAMES[i], NAMES[j]))
    return out


def deviation(final, carriers):
    """Max distance (grid units) from an edge's end points to the nearest of its carriers, over all edges."""
    by_name = dict(zip(NAMES, carriers, strict=True))
    worst = (0.0, None)
    per_edge = {}
    for (a, b), names in final.items():
        best = None
        for n in names:
            p, q = by_name[n]
            d = max(_point_segment(a, p, q), _point_segment(b, p, q))
            best = d if best is None else min(best, d)
        per_edge[(a, b)] = best
        if best > worst[0]:
            worst = (best, (a, b))
    return worst, per_edge


def _point_segment(pt, p, q) -> float:
    px, py = pt
    dx, dy = q[0] - p[0], q[1] - p[1]
    t = max(0.0, min(1.0, ((px - p[0]) * dx + (py - p[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (p[0] + t * dx), py - (p[1] + t * dy))


# --------------------------------------------------------------------------- scenes (shared by both renderers)


def _square(c):
    x, y = c
    return [
        (x - 0.5, y - 0.5),
        (x + 0.5, y - 0.5),
        (x + 0.5, y + 0.5),
        (x - 0.5, y + 0.5),
        (x - 0.5, y - 0.5),
    ]


def _carrier_lines(alpha_dash="solid", width=1.6):
    return [
        dict(
            kind="line",
            pts=[a, b],
            colour=CARRIER_COLOUR[i],
            width=width,
            dash=alpha_dash,
            name=NAMES[i],
        )
        for i, (a, b) in enumerate(CARRIERS)
    ]


def _edge_lines(edges, width=2.6, highlight=frozenset()):
    out = []
    for (a, b), names in sorted(edges.items()):
        first = sorted(names)[0]
        out.append(
            dict(
                kind="line",
                pts=[a, b],
                colour="#111111"
                if (a, b) in highlight
                else CARRIER_COLOUR[NAMES.index(first)],
                width=width + (1.5 if (a, b) in highlight else 0),
                dash="solid",
                name="+".join(sorted(names)),
            )
        )
    return out


def _vertex_points(edges, colour="#111"):
    return [
        dict(kind="pt", pt=p, colour=colour, size=6, text=f"({p[0]}, {p[1]})")
        for p in sorted({p for e in edges for p in e})
    ]


def _build_scenes(t):
    r1, r2 = t["rounds"][0], t["rounds"][1]
    cross = exact_crossings(CARRIERS)
    scenes = []
    scenes += _input_scenes()
    scenes += _hot_pixel_scenes(t, r1, cross)
    scenes += _round2_scenes(r1, r2)
    scenes += _round2_result_scenes(t)
    arr = build_arrangement(
        [NodedEdge(a, b, frozenset(n)) for (a, b), n in t["final"].items()]
    )
    faces = []
    for fi in range(1, len(arr.faces)):
        ring = [(float(x), float(y)) for x, y in arr.cycle_points(arr.faces[fi].outer)]
        area = arr.face_twice_area(fi) / 2
        rep = interior_point(arr, fi)
        faces.append(
            dict(
                kind="poly",
                pts=ring + [ring[0]],
                fill=FACE_COLOURS[(fi - 1) % len(FACE_COLOURS)],
                edge="#222",
                name=f"face F{fi}: area {area:g} g²",
            )
        )
        faces.append(
            dict(
                kind="text",
                pt=(rep[0] / rep[2], rep[1] / rep[2]),
                text=f"F{fi}\n{area:g} g²",
                size=7,
            )
        )
    scenes.append(
        dict(
            title=f"6  DCEL faces: {len(arr.faces) - 1} bounded faces (areas in g²)",
            elements=faces
            + _edge_lines(t["final"], width=1.6)
            + _vertex_points(t["final"]),
        )
    )
    return scenes, arr


def _input_scenes():
    """Scene 1: the five exact carriers and their end points."""
    scenes = []
    scenes.append(
        dict(
            title="1  Input: five exact carriers",
            elements=_carrier_lines()
            + [
                dict(
                    kind="pt",
                    pt=p,
                    colour=CARRIER_COLOUR[i],
                    size=5,
                    text=f"{NAMES[i]} end point {p}",
                )
                for i, (a, b) in enumerate(CARRIERS)
                for p in (a, b)
            ],
        )
    )
    return scenes


def _hot_pixel_scenes(t, r1, cross):
    """Scenes 2 and 3: the hot pixels of the end points, then of round 1."""
    scenes = []
    pixels = []
    for c in sorted(t["endpoints"]):
        pixels.append(
            dict(
                kind="poly",
                pts=_square(c),
                fill="#f6d9a8",
                edge="#e08a1e",
                name=f"hot pixel {c} (end point)",
            )
        )
    for c in sorted(r1["new_hot"]):
        pixels.append(
            dict(
                kind="poly",
                pts=_square(c),
                fill="#f2b4ae",
                edge="#c0392b",
                name=f"hot pixel {c} (crossing)",
            )
        )
    scenes.append(
        dict(
            title="2  Round 1: hot pixels (orange: end points, red: crossings)",
            elements=pixels
            + _carrier_lines(width=1.2)
            + [
                dict(
                    kind="pt",
                    pt=(float(x), float(y)),
                    colour="#111",
                    size=7,
                    marker="x",
                    text=f"exact crossing of {a} and {b}: ({float(x):.3f}, {float(y):.3f})",
                )
                for x, y, a, b in cross
            ],
        )
    )
    scenes.append(
        dict(
            title="3  Round 1: each carrier re-routed through the pixels it touches",
            elements=[
                dict(
                    kind="poly",
                    pts=_square(c),
                    fill="#f6d9a8" if c in t["endpoints"] else "#f2b4ae",
                    edge="#999",
                    name=f"hot pixel {c}",
                )
                for c in sorted(r1["hot"])
            ]
            + _carrier_lines("dot", 1.0)
            + _edge_lines(r1["after"])
            + _vertex_points(r1["after"]),
        )
    )
    return scenes


def _round2_scenes(r1, r2):
    """Scene 4: the chain that round 2 re-routes onto a pixel it missed."""
    scenes = []
    added = set(r2["after"]) - set(r1["after"])
    removed = set(r1["after"]) - set(r2["after"])
    new_pixel = sorted({p for e in added for p in e} - {p for e in removed for p in e})
    scenes.append(
        dict(
            title="4  Round 2: a re-routed chain now touches a pixel it missed",
            elements=[
                dict(
                    kind="poly",
                    pts=_square(c),
                    fill="#ffffff",
                    edge="#999",
                    name=f"hot pixel {c}",
                )
                for c in sorted(r2["hot"])
            ]
            + [
                dict(
                    kind="poly",
                    pts=_square(c),
                    fill="#f6d9a8",
                    edge="#e08a1e",
                    name=f"pixel {c} newly touched",
                )
                for c in new_pixel
            ]
            + [
                dict(
                    kind="line",
                    pts=list(e),
                    colour="#bbbbbb",
                    width=2.6,
                    dash="dash",
                    name=f"replaced edge {e}",
                )
                for e in sorted(removed)
            ]
            + _edge_lines(r2["after"], highlight=frozenset(added))
            + _vertex_points(r2["after"]),
        )
    )
    return scenes


def _round2_result_scenes(t):
    """Scene 5: the noded edges after round 2."""
    scenes = []
    scenes.append(
        dict(
            title="5  Round 3: nothing changes; the noded arrangement",
            elements=_carrier_lines("dot", 1.0)
            + _edge_lines(t["final"])
            + _vertex_points(t["final"])
            + [
                dict(
                    kind="text",
                    pt=((a[0] + b[0]) / 2 + 0.15, (a[1] + b[1]) / 2 + 0.15),
                    text="+".join(sorted(n)),
                    size=6.5,
                )
                for (a, b), n in t["final"].items()
                if len(n) > 1
            ],
        )
    )
    return scenes


# --------------------------------------------------------------------------- renderers


def _draw_mpl(ax, scene):
    ls = {"solid": "-", "dash": "--", "dot": ":"}
    for el in scene["elements"]:
        k = el["kind"]
        if k == "poly":
            xs, ys = zip(*el["pts"], strict=True)
            ax.fill(
                xs, ys, facecolor=el["fill"], edgecolor=el["edge"], lw=0.8, alpha=0.85
            )
        elif k == "line":
            xs, ys = zip(*el["pts"], strict=True)
            ax.plot(
                xs,
                ys,
                color=el["colour"],
                lw=el["width"],
                ls=ls[el["dash"]],
                solid_capstyle="round",
            )
        elif k == "pt":
            ax.plot(
                [el["pt"][0]],
                [el["pt"][1]],
                marker=el.get("marker", "o"),
                color=el["colour"],
                ms=el["size"] * 0.6,
                ls="none",
                zorder=5,
            )
        elif k == "text":
            ax.text(
                el["pt"][0],
                el["pt"][1],
                el["text"],
                fontsize=el["size"],
                ha="center",
                va="center",
                zorder=6,
            )
    ax.set_xlim(-0.8, 13.0)
    ax.set_ylim(-0.8, 10.0)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 13))
    ax.set_yticks(range(0, 10))
    ax.tick_params(labelsize=6)
    ax.grid(True, color="#e5e5e5", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(scene["title"], fontsize=8.5, loc="left")


def _static_figure(scenes) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    for ax, scene in zip(axes.flat, scenes, strict=True):
        _draw_mpl(ax, scene)
    fig.tight_layout()
    save_svg(fig, "arrangement-steps")


def _interactive_figure(scenes) -> None:
    dash = {"solid": "solid", "dash": "dash", "dot": "dot"}
    fig = go.Figure()
    owner = []
    for si, scene in enumerate(scenes):
        for el in scene["elements"]:
            k = el["kind"]
            if k == "poly":
                xs, ys = zip(*el["pts"], strict=True)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        fill="toself",
                        fillcolor=el["fill"],
                        line=dict(color=el["edge"], width=1),
                        name=el["name"],
                        hoverinfo="name",
                        opacity=0.9,
                        showlegend=False,
                    )
                )
            elif k == "line":
                xs, ys = zip(*el["pts"], strict=True)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(
                            color=el["colour"], width=el["width"], dash=dash[el["dash"]]
                        ),
                        name=el["name"],
                        hoverinfo="name",
                        showlegend=False,
                    )
                )
            elif k == "pt":
                fig.add_trace(
                    go.Scatter(
                        x=[el["pt"][0]],
                        y=[el["pt"][1]],
                        mode="markers",
                        marker=dict(
                            color=el["colour"],
                            size=el["size"] + 2,
                            symbol="x" if el.get("marker") == "x" else "circle",
                        ),
                        text=[el["text"]],
                        hoverinfo="text",
                        showlegend=False,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[el["pt"][0]],
                        y=[el["pt"][1]],
                        mode="text",
                        text=[el["text"].replace("\n", "<br>")],
                        textfont=dict(size=11),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            owner.append(si)
    for i, tr in enumerate(fig.data):
        tr.visible = owner[i] == 0
    buttons = [
        dict(
            label=s["title"].split("  ")[0],
            method="update",
            args=[{"visible": [o == si for o in owner]}, {"title": s["title"]}],
        )
        for si, s in enumerate(scenes)
    ]
    fig.update_layout(
        title=scenes[0]["title"],
        height=640,
        xaxis=dict(
            range=[-0.8, 13], dtick=1, title="x (grid units g)", gridcolor="#e5e5e5"
        ),
        yaxis=dict(
            range=[-0.8, 10],
            dtick=1,
            scaleanchor="x",
            title="y (grid units g)",
            gridcolor="#e5e5e5",
        ),
        updatemenus=[
            dict(type="buttons", direction="right", x=0, y=1.14, buttons=buttons)
        ],
    )
    save_plotly(fig, "arrangement-steps")


def main() -> None:
    """Write the iterated snap-rounding figure and its numbers."""
    t = _trace(CARRIERS)
    reference = snap_round(
        [
            Segment(a, b, frozenset([n]))
            for (a, b), n in zip(CARRIERS, NAMES, strict=True)
        ]
    )
    assert {(e.a, e.b): set(e.carriers) for e in reference.edges} == t["final"], (
        "trace differs from snap_round"
    )
    scenes, arr = _build_scenes(t)
    _static_figure(scenes)
    _interactive_figure(scenes)
    (worst, edge), per_edge = deviation(t["final"], CARRIERS)
    r1, r2 = t["rounds"][0], t["rounds"][1]
    numbers = {
        "carriers": {
            n: {"from": list(a), "to": list(b)}
            for n, (a, b) in zip(NAMES, CARRIERS, strict=True)
        },
        "hot_pixels": {
            "end_points": len(t["endpoints"]),
            "round_1_crossings": sorted(r1["new_hot"]),
            "total": len(t["hot"]),
        },
        "exact_crossings": [
            {"carriers": [a, b], "x": round(float(x), 4), "y": round(float(y), 4)}
            for x, y, a, b in exact_crossings(CARRIERS)
        ],
        "rounds_total": len(t["rounds"]),
        "rounds_with_change": sum(1 for r in t["rounds"] if r["after"] != r["before"]),
        "round_2_added_edges": sorted(map(list, set(r2["after"]) - set(r1["after"]))),
        "round_2_removed_edges": sorted(map(list, set(r1["after"]) - set(r2["after"]))),
        "noded_edges": len(t["final"]),
        "vertices": len({p for e in t["final"] for p in e}),
        "bounded_faces": len(arr.faces) - 1,
        "face_areas_g2": [
            arr.face_twice_area(fi) / 2 for fi in range(1, len(arr.faces))
        ],
        "max_edge_deviation_g": round(worst, 4),
        "max_deviation_edge": [list(edge[0]), list(edge[1])],
        "shared_edges": [
            {"edge": [list(a), list(b)], "carriers": sorted(n)}
            for (a, b), n in t["final"].items()
            if len(n) > 1
        ],
        "bound_sqrt2_over_2": round(math.sqrt(2) / 2, 4),
    }
    write_numbers("arrangement-steps", numbers)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
