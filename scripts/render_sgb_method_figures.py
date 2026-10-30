"""Render public SGB method figures from actual XAO observations.

Run the public cases first. This renderer does not build geometry or launch a
solver. Each input contains native point/curve/face observations from the same
public fixture, rather than a drawing of the intended lowering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import plotly.graph_objects as go


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curve_trace(native, tags, name, color, width=4):
    coordinates = [[], [], []]
    for curve in native["curves"]:
        if curve["tag"] in tags:
            for xyz in curve["xyz"]:
                for axis in range(3):
                    coordinates[axis].append(xyz[axis])
            for axis in coordinates:
                axis.append(None)
    return go.Scatter3d(
        x=coordinates[0], y=coordinates[1], z=coordinates[2], mode="lines",
        line={"color": color, "width": width}, name=name,
        hovertemplate="x=%{x:.4g} µm<br>y=%{y:.4g} µm<br>z=%{z:.4g} µm<extra>%{fullData.name}</extra>",
    )


def native_plot(native, chinese, shared=False):
    figure = go.Figure()
    surfaces = {s["tag"]: s for s in native["surfaces"]}
    all_edges = {c["tag"] for c in native["curves"]}
    figure.add_trace(curve_trace(
        native, all_edges, "材料區域外框" if chinese else "Material-region boundaries",
        "#a5b4c5", 2,
    ))
    if shared:
        shared_faces = [s for s in native["surfaces"] if len(s["upward_volumes"]) == 2]
        if len(shared_faces) != 1:
            raise ValueError("the two-volume case must have exactly one shared native face")
        edges = set(shared_faces[0]["curves"])
        figure.add_trace(curve_trace(native, edges,
            "兩個體積共用的面，z = 0" if chinese else "One face shared by two volumes, z = 0",
            "#d97706", 7))
        vertices = sorted({tuple(p) for c in native["curves"] if c["tag"] in edges for p in c["xyz"]})
        if len(vertices) != 4:
            raise ValueError("the shared-face example must retain its four native corners")
        # Display facets of the observed planar rectangle, not FEM elements.
        order = [0, 1, 3, 2]
        points = [vertices[i] for i in order]
        figure.add_trace(go.Mesh3d(
            x=[p[0] for p in points], y=[p[1] for p in points], z=[p[2] for p in points],
            i=[0, 0], j=[1, 2], k=[2, 3], opacity=0.25, color="#d97706",
            name="共享面" if chinese else "Shared face", showlegend=False,
            hoverinfo="skip",
        ))
    else:
        choices = (
            ("MS_MA__chip/left", "Net A：MS／MA 共用幾何", "Net A: shared MS / MA geometry", "#2867a3"),
            ("MS_MA__chip/right", "Net B：MS／MA 共用幾何", "Net B: shared MS / MA geometry", "#ae5b37"),
            ("LUMPED_PORT__", "中央集總埠面", "Central lumped-port sheet", "#9d3da5"),
        )
        for prefix, zh, en, color in choices:
            faces = {tag for g in native["groups"] if g["dimension"] == 2 and g["name"].startswith(prefix) for tag in g["entities"]}
            edges = {tag for face in faces for tag in surfaces[face]["curves"]}
            if not edges:
                raise ValueError(f"missing observed group {prefix}")
            figure.add_trace(curve_trace(native, edges, zh if chinese else en, color, 6))
    points = native["points"]
    figure.add_trace(go.Scatter3d(
        x=[p["xyz"][0] for p in points], y=[p["xyz"][1] for p in points],
        z=[p["xyz"][2] for p in points], mode="markers", marker={"size": 3, "color": "#243444"},
        name="原生拓樸頂點" if chinese else "Native topology vertices",
        hovertemplate="x=%{x:.4g} µm<br>y=%{y:.4g} µm<br>z=%{z:.4g} µm<extra></extra>",
    ))
    figure.update_layout(
        template="plotly_white", autosize=True, margin={"l": 15, "r": 35, "t": 12, "b": 115},
        font={"size": 13}, scene={
            "xaxis_title": "x (µm)", "yaxis_title": "y (µm)", "zaxis_title": "z (µm)",
            "aspectmode": "data", "camera": {"eye": {"x": 1.4, "y": -1.8, "z": 1.2}},
        }, legend={"orientation": "h", "x": 0, "y": -0.03, "yanchor": "top"},
    )
    return figure


def write_plot(figure, path, chinese):
    # Preserve physical axis proportions while fitting a portrait iframe.
    # Refit only when the viewport changes; ordinary rotation keeps its camera.
    responsive_camera = """
const figure = document.getElementById('scientific-figure');
let previousSize = '';
function fitCamera() {
    const width = figure.clientWidth, height = figure.clientHeight;
    if (!width || !height) return;
    const size = width + 'x' + height;
    if (size === previousSize) return;
    previousSize = size;
    const scale = 1.1 * Math.max(1, height / width);
    Plotly.relayout(figure, {'scene.camera.eye': {
        x: 1.4 * scale, y: -1.8 * scale, z: 1.2 * scale
    }});
}
new ResizeObserver(fitCamera).observe(figure);
fitCamera();
"""
    body = figure.to_html(
        full_html=False, include_plotlyjs=False, div_id="scientific-figure",
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
        post_script=responsive_camera,
    )
    path.write_text(
        '<!doctype html><html lang="' + ('zh-TW' if chinese else 'en') + '"><head>'
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<script src="plotly.min.js"></script><style>html,body{margin:0;width:100%;height:100%;overflow:hidden}'
        '#scientific-figure{width:100%;height:100%}</style></head><body>' + body + '</body></html>\n',
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/images/sgb-concept"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = {"shared-volumes": "shared-volumes", "junction": "construction-3d", "circular-hole": "curved-construction-3d"}
    manifest = {"schema": "scgsim.sgb.method-public-cases.v1", "classification": "PUBLIC_GENERIC_FIXTURE", "geometry_bands": False, "cases": []}
    for case, output in names.items():
        path = args.observations / ("public-candidate-" + case) / "RESULT.json"
        observed = json.loads(path.read_text())
        if observed["status"] != "PASS" or observed["case"] not in {"shared-volumes", "junction-hole", "circular-hole"}:
            raise ValueError("only successful public method-case observations are supported")
        native = observed["native"]
        for chinese in (False, True):
            file = ("" if chinese else "en-") + output + ".html"
            write_plot(native_plot(native, chinese, shared=case == "shared-volumes"), args.output / file, chinese)
        entry = {"case": observed["case"], "fixture": "tests/sgb_method_cases.py", "fixture_sha256": observed["case_source_sha256"],
                 "xao_sha256": native["xao_sha256"], "points": len(native["points"]), "curves": len(native["curves"]),
                 "surfaces": len(native["surfaces"]), "volumes": native["volume_count"],
                 "files": {name: digest(args.output / name) for name in (output + ".html", "en-" + output + ".html")}}
        manifest["cases"].append(entry)
    (args.output / "method-public-case.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
