"""Figure of why-plan-first.qmd: what a "build solids, then fragment" workflow hands to a Boolean kernel.

This script does not run a Boolean operation and does not time one. It counts structure that is a fact of the
input: for the Route B random layouts of the production test scenes (``sgb_scenes.random_layout``), treat every prism as a separately built solid and count
(a) the pairs whose bounding boxes overlap, split by whether a chip-sized body is involved, (b) the inner wires
on the host faces of the ground plane, and (c) the coplanar face pairs whose boxes overlap.

Writes ``figures/why-structure.html`` and ``.svg`` plus ``figures/why-structure-numbers.json``.
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from _intro_common import COLOURS, draw_footprint, save_plotly, save_svg, write_numbers
from sgb_scenes import random_layout

NS = (100, 500, 1000, 5000, 20000)
CHIP_BODIES = ("substrate", "vacuum", "ground")


def box(prism):
    pts = [p for ring in prism.footprint.rings() for p in ring]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys), float(prism.z0), float(prism.z1))


def overlap(a, b, tol=0.0) -> bool:
    """Axis-aligned box overlap in x, y and z (touching counts as overlap, like a tolerance-enlarged box)."""
    return a[0] <= b[2] + tol and b[0] <= a[2] + tol and a[1] <= b[3] + tol and b[1] <= a[3] + tol and a[4] <= b[5] and b[4] <= a[5]


def feature_pairs(boxes) -> int:
    """Overlapping box pairs among small bodies, through a uniform hash (one cell per feature pitch)."""
    cell = 260_000  # grid units, the layout pitch (260 um)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, b in enumerate(boxes):
        for cx in range(b[0] // cell, b[2] // cell + 1):
            for cy in range(b[1] // cell, b[3] // cell + 1):
                grid[(cx, cy)].append(i)
    seen = set()
    for ids in grid.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                if (i, j) not in seen and overlap(boxes[i], boxes[j]):
                    seen.add((i, j))
    return len(seen)


def structure(n: int) -> dict:
    spec = random_layout(n, seed=2, route="B")
    big = {p.entity_id: p for p in spec.prisms if p.entity_id in CHIP_BODIES}
    small = [p for p in spec.prisms if p.entity_id not in CHIP_BODIES]
    big_boxes = {k: box(p) for k, p in big.items()}
    small_boxes = [box(p) for p in small]
    with_big = sum(overlap(b, bb) for b in small_boxes for bb in big_boxes.values())
    chip_area = (big_boxes["substrate"][2] - big_boxes["substrate"][0]) * (big_boxes["substrate"][3] - big_boxes["substrate"][1])
    mean_small = sum((b[2] - b[0]) * (b[3] - b[1]) for b in small_boxes) / max(len(small_boxes), 1)
    # coplanar faces whose boxes overlap: a small body's bottom face (z = 0) against the bottom face of the ground,
    # the top face of the substrate and the bottom face of the vacuum; its top face (z = 0.2 um) against the ground top.
    coplanar = 4 * len(small)
    return {
        "n_features": n,
        "small_bodies": len(small),
        "chip_sized_bodies": len(big),
        "small_x_chip_pairs": with_big,
        "small_x_small_pairs": feature_pairs([tuple(int(v) for v in b[:4]) + b[4:] for b in small_boxes]),
        "ground_inner_wires_per_face": len(big["ground"].footprint.holes),
        "coplanar_face_pairs": coplanar,
        "mean_small_box_over_chip_area": mean_small / chip_area,
    }


def schematic(ax, n: int = 36) -> None:
    spec = random_layout(n, seed=2, route="B")
    for p in spec.prisms:
        if p.entity_id in CHIP_BODIES:
            continue
        draw_footprint(ax, p.footprint, facecolor=COLOURS["pad"], edgecolor="#222", lw=0.4, alpha=0.9)
        b = box(p)
        ax.add_patch(plt.Rectangle((b[0] * 0.001, b[1] * 0.001), (b[2] - b[0]) * 0.001, (b[3] - b[1]) * 0.001, fill=False, edgecolor=COLOURS["accent"], lw=0.6))
    half = max(abs(v) for v in spec.prisms[0].footprint.outer[0]) * 0.001
    ax.add_patch(plt.Rectangle((-half, -half), 2 * half, 2 * half, fill=False, edgecolor=COLOURS["bad"], lw=1.6, ls="--"))
    ax.text(-half + 8, half - 22, "box of substrate, vacuum and ground", color=COLOURS["bad"], fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xlim(-half - 15, half + 15); ax.set_ylim(-half - 15, half + 15); ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(f"N = {n} features: every small box (orange) lies inside the chip-sized boxes")


def main() -> None:
    rows = [structure(n) for n in NS]
    ns = [r["n_features"] for r in rows]
    series = (
        ("small_x_chip_pairs", "small body × chip-sized body", COLOURS["bad"]),
        ("small_x_small_pairs", "small body × small body", COLOURS["pad"]),
        ("ground_inner_wires_per_face", "inner wires on each ground face", COLOURS["accent"]),
        ("coplanar_face_pairs", "coplanar face pairs with overlapping boxes", COLOURS["good"]),
    )
    f, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    schematic(a1)
    for key, name, colour in series:
        a2.loglog(ns, [r[key] for r in rows], marker="o", color=colour, label=name)
    a2.set_xticks(ns); a2.set_xticklabels([str(n) for n in ns]); a2.minorticks_off()
    a2.set_xlabel("features N"); a2.set_ylabel("count"); a2.grid(True, color="#ddd", lw=0.6)
    a2.legend(fontsize=8, frameon=False); a2.set_title("Candidate pairs and host-face wires (counted, not timed)")
    f.tight_layout()
    save_svg(f, "why-structure")

    fig = go.Figure()
    for key, name, colour in series:
        fig.add_trace(go.Scatter(x=ns, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=colour)))
    fig.update_xaxes(type="log", title="features N", tickvals=ns)
    fig.update_yaxes(type="log", title="count")
    fig.update_layout(height=460, title="Counted structure of a solids-first model (Route B random layouts; no Boolean operation was run)",
                      legend=dict(orientation="h", y=-0.25))
    save_plotly(fig, "why-structure")
    numbers = {"rows": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
               "assumptions": "each prism is one solid; boxes touching count as overlapping; layout pitch 260 um, seed 2"}
    write_numbers("why-structure", numbers)
    print(json.dumps(numbers["rows"], indent=1))


if __name__ == "__main__":
    main()
