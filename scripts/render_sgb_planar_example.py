"""Render the method chapter's exact labelled planar subdivision.

Run from the repository root with PYTHONPATH=src:tests. The same public
notched-pad fixture supplies A; M is an explicit auxiliary operand. This uses
the real SGB planar compiler and does not start Gmsh or a solver.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import plotly.graph_objects as go
from shapely import Polygon

from scgsim.geometry._primitives.grid import GridOperand, build_grid_arrangement
from scgsim.geometry.models.grid import GeometryGridContext
from sgb_method_cases import junction_hole_case


def closed(ring):
    return (*ring, ring[0])


def outline(figure, ring, name, color, *, dash="solid", legend=True):
    points = closed(ring)
    figure.add_trace(go.Scatter(
        x=[p[0] for p in points], y=[p[1] for p in points],
        mode="lines", name=name, showlegend=legend,
        line={"color": color, "width": 2, "dash": dash},
        hovertemplate="x=%{x:g} µm<br>y=%{y:g} µm<extra>%{fullData.name}</extra>",
    ))


def plot(graph, operands, chinese):
    figure = go.Figure()
    labels = {
        ("A",): ("只屬於 A · 5 µm²", "A only · 5 µm²", "#a5cee8"),
        ("A", "M"): ("同時屬於 A、M · 3 µm²", "A and M · 3 µm²", "#b69ad4"),
        ("M",): ("只屬於 M · 9 µm²", "M only · 9 µm²", "#f3ca91"),
    }
    for face in graph.faces:
        zh, en, color = labels[face.operand_ids]
        ring = closed(face.exterior)
        figure.add_trace(go.Scatter(
            x=[p[0] for p in ring], y=[p[1] for p in ring],
            mode="lines", line={"width": 0}, fill="toself", fillcolor=color,
            name=zh if chinese else en,
            hoveron="fills", hoverinfo="name",
        ))
        for hole in face.holes:
            ring = closed(hole)
            figure.add_trace(go.Scatter(
                x=[p[0] for p in ring], y=[p[1] for p in ring],
                mode="lines", line={"width": 0}, fill="toself",
                fillcolor="white", showlegend=False, hoverinfo="skip",
            ))
    a, m = operands
    outline(figure, a.exterior, "來源 A 外環與孔環" if chinese else "Source A exterior and hole", "#263e57")
    for hole in a.holes:
        outline(figure, hole, "A 孔洞" if chinese else "A hole", "#263e57", legend=False)
    outline(figure, m.exterior, "輔助運算元 M" if chinese else "Auxiliary operand M", "#ad641b", dash="dash")
    for y in (0.25, 1.25):
        figure.add_trace(go.Scatter(
            x=[-2.5, 4.5], y=[y, y], mode="lines",
            line={"color": "#555", "width": 1, "dash": "dot"},
            name=f"y = {y} µm", visible="legendonly",
            hoverinfo="name",
        ))
    figure.update_layout(
        template="plotly_white", autosize=True,
        margin={"l": 52, "r": 16, "t": 18, "b": 180},
        font={"size": 13}, xaxis={"title": "x (µm)", "range": [-2.5, 4.5]},
        yaxis={"title": "y (µm)", "range": [-1.5, 3.5], "scaleanchor": "x", "scaleratio": 1},
        legend={"orientation": "h", "x": 0, "y": -0.2, "yanchor": "top"},
    )
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/images/sgb-concept"))
    args = parser.parse_args()
    source, _, _ = junction_hole_case()
    pad = next(p for p in source.polygons if p.polygon_id == "left")
    operands = (
        GridOperand("A", pad.exterior, pad.holes),
        GridOperand("M", ((1, -1), (4, -1), (4, 3), (1, 3)), physical=False, contact=False),
    )
    graph = build_grid_arrangement(operands, GeometryGridContext(0.001))
    areas = {"+".join(f.operand_ids): Polygon(f.exterior, f.holes).area for f in graph.faces}
    if areas != {"A": 5.0, "A+M": 3.0, "M": 9.0}:
        raise ValueError(f"the chapter's exact fixture and subdivision disagree: {areas}")
    args.output.mkdir(parents=True, exist_ok=True)
    figures = {}
    for chinese in (False, True):
        path = args.output / (("" if chinese else "en-") + "planar-arrangement.html")
        body = plot(graph, operands, chinese).to_html(
            full_html=False, include_plotlyjs=False, div_id="scientific-figure",
            config={"responsive": True, "displaylogo": False, "scrollZoom": True},
        )
        path.write_text(
            '<!doctype html><html lang="' + ('zh-TW' if chinese else 'en') + '"><head>'
            '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<script src="plotly.min.js"></script><style>html,body{margin:0;width:100%;height:100%;overflow:hidden}'
            '#scientific-figure{width:100%;height:100%}</style></head><body>' + body + '</body></html>\n',
            encoding="utf-8",
        )
        figures[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    fixture = Path(__file__).resolve().parents[1] / "tests/sgb_method_cases.py"
    record = {
        "schema": "scgsim.sgb.planar-method-example.v1", "units": "um", "grid_size_um": 0.001,
        "source_fixture": "tests/sgb_method_cases.py:junction_hole_case",
        "source_fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "operands": [asdict(o) for o in operands], "faces": [asdict(f) for f in graph.faces],
        "areas_um2": areas, "source_hole_area_um2": Polygon(pad.holes[0]).area,
        "membership_walk_y_um": [0.25, 1.25], "figure_sha256": figures,
    }
    (args.output / "planar-arrangement.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
