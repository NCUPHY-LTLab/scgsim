"""Figures and tables for ``docs/design/sgb/identity.qmd``.

Run (from anywhere)::

    .venv/bin/python zh-TW/docs/design/sgb/_figures/identity.py

Computes, from the read-only prototype, a real mapping-chain lookup on the pad-in-ground example and the
determinism evidence quoted by the page. Outputs go to ``docs/design/sgb/figures/`` (prefix ``identity-``):

* ``identity-trace.json``        every step of the lookup  Palace attribute -> source curve (also quoted in the page)
* ``identity-trace-plan.html``   the traced faces and edge in the plan view (+ ``.svg``)
* ``identity-determinism.json``  one build digest per (tile grid, workers, PYTHONHASHSEED); all must agree
* ``identity-stability.json``    which atomic ids survive an unrelated edit (removing the lumped port)
* ``identity-tables.md``         markdown tables quoted by the page
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the prototype on sys.path)
from inset_kit import Line, Panel, Poly, Pts  # noqa: E402

from sgbproto.arcs import cycle_polyline, sample_arc  # noqa: E402
from sgbproto.inset import distance_to_primitive  # noqa: E402
from sgbproto.pipeline import build, ids_digest  # noqa: E402
from sgbproto.records import ArcCarrier, base_carrier_id  # noqa: E402
from sgbproto.scenes import pad_in_ground  # noqa: E402

G = kit.G_UM
MARGINS = (0.1, 0.5, 2.0)
CASE = "pad_in_ground_B_bands"
TARGET_GROUP = "MA::pad::band=2"
ALT_GROUP = "MA::pad::band=0"
WALL_GROUP = "MA::pad"
OUT = kit.PROTOTYPE_OUTPUTS / CASE  # identity.json of the derivation
SIDECAR = kit.PROTOTYPE_OUTPUTS / "e3_brep" / CASE / "model.sidecar.json"  # sgbproto-sidecar-2 of the direct writer
CARRIER = re.compile(r"^(?P<family>[^/]+)/ring(?P<ring>\d+)/(?P<kind>edge|corner)(?P<index>\d+)@m(?P<margin>\d+)\.(?P<piece>\d+)$")


def load(name: str):
    return json.loads((OUT / name).read_text())


def describe_edge(spec, sidecar, edge_id: str) -> dict:
    """One edge with its carriers read from the sidecar lineage, resolved to the generating support curve."""
    carriers = list(sidecar["lineage"]["edges"][edge_id])
    entry = {"edge": edge_id, "carriers": carriers, "resolved": []}
    for carrier in carriers:
        match = CARRIER.match(carrier)
        if match:
            family = next(f for f in spec.families if f.family_id == match["family"])
            ring, index = int(match["ring"]), int(match["index"])
            record = {"kind": "family member", "family": match["family"], "margin_nm": int(match["margin"]), "support_ring": ring,
                      "member": f"{match['family']}/ring{ring}/{match['kind']}{index}"}
            if match["kind"] == "edge":
                primitive = family.support.rings[ring][index]
                if isinstance(primitive, ArcCarrier):
                    record.update({"support_curve": primitive.id, "source": primitive.lineage.describe(), "centre_um": [float(c) * G for c in primitive.centre],
                                   "radius_um": float(primitive.radius) * G})
                else:
                    record.update({"support_curve": "straight edge", "support_primitive": index})
            else:
                record["support_curve"] = f"reflex-corner arc {index} of ring {ring}"
            entry["resolved"].append(record)
        elif "/polygon" in carrier:
            entity, polygon, ring, edge_index = carrier.split("/")
            entry["resolved"].append({"kind": "source edge", "entity": entity, "polygon": polygon, "ring": ring, "edge": edge_index, "carrier": carrier})
        elif base_carrier_id(carrier) in spec.arc_registry():
            arc = spec.arc_registry()[base_carrier_id(carrier)]
            entry["resolved"].append({"kind": "source arc carrier", "carrier": carrier, "source": arc.lineage.describe(), "centre_um": [float(c) * G for c in arc.centre],
                                      "radius_um": float(arc.radius) * G})
        else:
            entry["resolved"].append({"kind": "source entity (vertical edge)", "entity": carrier})
    return entry


def build_trace(spec, result) -> dict:
    topo = result.topology
    sidecar = json.loads(SIDECAR.read_text())
    stored = load("identity.json")
    # the rebuilt model must be the stored one: same groups with the same member ids
    assert {k: v["ids"] for k, v in topo.groups.items()} == {k: v["ids"] for k, v in stored["groups"].items()}
    assert {k: v["ids"] for k, v in topo.groups.items()} == {k: v["ids"] for k, v in sidecar["physical_groups"].items()}
    native = {fid: 1 + i for i, fid in enumerate(sidecar["ids_by_tag"]["2"])}
    trace = {"case": CASE, "route": "B", "margins_um": list(MARGINS), "sidecar_format": sidecar["format"]}
    group = sidecar["physical_groups"][TARGET_GROUP]
    attribute = group["physical_tag"]
    by_tag = {g["physical_tag"]: name for name, g in sidecar["physical_groups"].items() if g["dim"] == 2}
    trace["step1_palace_attribute"] = {"attribute": attribute, "resolved_name": by_tag[attribute], "source": "model.sidecar.json: physical_groups[*].physical_tag"}
    assert [native[f] for f in group["ids"]] == group["native_tags"]
    trace["step2_physical_group"] = {"name": TARGET_GROUP, "dim": group["dim"], "faces": group["ids"], "native_surface_tags": group["native_tags"],
                                     "union_membership": [u for u, members in sidecar["unions"].items() if TARGET_GROUP in members],
                                     "surface_epr_row": sidecar["surface_epr_rows"][TARGET_GROUP]}
    faces = []
    for fid in group["ids"]:
        face = topo.faces[fid]
        loops = [[describe_edge(spec, sidecar, eid) | {"orientation": sgn} for eid, sgn in loop] for loop in face.loops]
        faces.append({"face": fid, "native_tag": native[fid], "kind": face.kind, "plane": face.plane, "z_um": str(face.z_range[0]),
                      "label_bands": [list(b) for b in face.bands], "neg_volume": face.neg_volume, "pos_volume": face.pos_volume,
                      "sources_lineage": sidecar["lineage"]["faces"][fid], "membership": sidecar["lineage"]["membership"][fid], "loops": loops})
    trace["step3_atomic_faces"] = faces
    alt_face = sidecar["physical_groups"][ALT_GROUP]["ids"][0]
    alt = topo.faces[alt_face]
    trace["alt_group"] = {"name": ALT_GROUP, "face": alt_face, "sources_lineage": sidecar["lineage"]["faces"][alt_face],
                          "membership": sidecar["lineage"]["membership"][alt_face],
                          "loops": [[describe_edge(spec, sidecar, eid) | {"orientation": sgn} for eid, sgn in loop] for loop in alt.loops]}
    wall = next(fid for fid in sidecar["physical_groups"][WALL_GROUP]["ids"] if topo.faces[fid].kind == "wall" and not topo.faces[fid].curved)
    wall_face = topo.faces[wall]
    trace["wall"] = {"name": WALL_GROUP, "face": wall, "z_range": [str(z) for z in wall_face.z_range], "sources_lineage": sidecar["lineage"]["faces"][wall],
                     "membership": sidecar["lineage"]["membership"][wall],
                     "loops": [[describe_edge(spec, sidecar, eid) | {"orientation": sgn} for eid, sgn in loop] for loop in wall_face.loops]}
    return trace


def figure_trace(spec, result, trace) -> None:
    topo = result.topology
    layer = next(a for a in result.level_layers if a.plane == "level[1/5]")
    face_ids = set(topo.groups[TARGET_GROUP]["ids"])
    chosen = next(e for f in trace["step3_atomic_faces"] for loop in f["loops"] for e in loop if any(c.startswith("band_pad_top/ring0/edge9@m500") for c in e["carriers"]))
    polys, highlight = [], []
    for fi in range(1, len(layer.arr.faces)):
        fid = layer.face_ids[fi]
        group = topo.faces[fid].group if fid in topo.faces else "(not a face)"
        rings = [[(x * G, y * G) for x, y in cycle_polyline(layer.arr, layer.edge_curve, c, 150.0)] for c in layer.arr.face_cycles(fi)]
        if fid in face_ids:
            polys.append(Poly(rings, "#e08214", "rgba(0,0,0,0.8)", 0.8, hover=f"{group}<br>{fid}"))
        elif group.startswith("MA::pad::band"):
            polys.append(Poly(rings, "#c6dbef", "rgba(0,0,0,0.5)", 0.4, hover=f"{group}<br>{fid}"))
        elif group.startswith("PORT") or group.startswith("MA::Ground"):
            polys.append(Poly(rings, "#e8e8e8", "rgba(0,0,0,0.3)", 0.3, hover=f"{group}<br>{fid}"))
        else:
            polys.append(Poly(rings, "#f7f7f7", "rgba(0,0,0,0.2)", 0.3, hover=f"{group}<br>{fid}"))
    # the chosen edge, drawn from the arrangement
    index = layer.edge_ids.index(chosen["edge"])
    edge, curve = layer.arr.edges[index], layer.edge_curve[index]
    pts = [edge.a, edge.b] if curve is None else sample_arc(curve, edge.a, edge.b, 40)
    xs, ys = [p[0] * G for p in pts], [p[1] * G for p in pts]
    # the support primitive the carrier was generated from (exact arc, distance origin)
    family = spec.families[0]
    ring, prim_index = 0, 9
    prim = family.support.rings[ring][prim_index]
    t0, sweep = prim.sweep()
    cx, cy, r = float(prim.centre[0]), float(prim.centre[1]), float(prim.radius)
    arc = [((cx + r * math.cos(t0 + sweep * i / 60)) * G, (cy + r * math.sin(t0 + sweep * i / 60)) * G) for i in range(61)]
    mid = pts[len(pts) // 2]
    deviation = distance_to_primitive(prim, mid[0], mid[1]) * G
    legend = [Pts([-1e4], [-1e4], "#e08214", 8, name=f"faces of {TARGET_GROUP}", symbol="square"),
              Pts([-1e4], [-1e4], "#c6dbef", 8, name="other MA::pad band faces", symbol="square")]
    left = Panel("pad top face at z = 0.2 um (Route B)", (-55, 55), (-35, 35), xlabel="x (um)", ylabel="y (um)",
                 items=polys + [Line(xs, ys, kit.PORT, 3.0, name="traced edge", hover=chosen["edge"])] + legend)
    zoom = Panel("zoom: the traced edge, its support curve and its distance", (-52, -38), (-32, -20), xlabel="x (um)", ylabel="y (um)",
                 items=polys + [Line([p[0] for p in arc], [p[1] for p in arc], kit.INK, 2.4, name=f"support curve {prim.id} (ring {ring}, primitive {prim_index})"),
                                Line(xs, ys, kit.PORT, 3.0, name=f"traced edge (carrier {chosen['carriers'][0]})", hover=chosen["edge"])])
    kit.write_both([left, zoom], "identity-trace-plan", ncols=2, svg_size=(12, 4.8), html_size=(1250, 560), shared_legend=True,
                   title="Faces named by one physical group, and one of their edges traced back to the source arc")
    trace["traced_edge"] = {"edge": chosen["edge"], "carrier": chosen["carriers"][0], "distance_to_support_um_at_midpoint": round(deviation, 4),
                            "support_curve": prim.id, "support_lineage": prim.lineage.describe()}


# --------------------------------------------------------------------------- determinism evidence

DIGEST_SCRIPT = (
    "import sys; sys.path.insert(0, {proto!r})\n"
    "from sgbproto.pipeline import build, ids_digest\n"
    "from sgbproto.scenes import pad_in_ground, random_layout\n"
    "tiles, workers = {tiles}, {workers}\n"
    "a = ids_digest(build(pad_in_ground('B', bands_um=(0.1, 0.5, 2.0)), tiles=tiles, workers=workers))\n"
    "b = ids_digest(build(random_layout(60, seed=5, route='B'), tiles=tiles, workers=workers))\n"
    "print(a, b)\n"
)


def determinism() -> list[dict]:
    rows = []
    for tiles, workers in ((1, 1), (2, 1), (3, 2), (4, 4)):
        for hashseed in ("0", "1", "12345"):
            code = DIGEST_SCRIPT.format(proto=str(kit.PROTOTYPE), tiles=tiles, workers=workers)
            env = {**os.environ, "PYTHONHASHSEED": hashseed}
            out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.split()
            rows.append({"tiles": f"{tiles}x{tiles}", "workers": workers, "pythonhashseed": hashseed, "pad_in_ground_B_bands": out[0], "random_layout_60_seed5_B": out[1]})
    return rows


def stability() -> dict:
    """Ids of the Route B example with bands, with and without the lumped port (an unrelated edit for every band boundary)."""
    with_port = build(pad_in_ground("B", bands_um=MARGINS))
    without = build(pad_in_ground("B", bands_um=MARGINS, with_port=False))
    out = {"same_group_names": sorted(with_port.topology.groups) == sorted(without.topology.groups) or "differ"}
    for kind in ("vertices", "edges", "faces", "volumes"):
        a, b = set(getattr(with_port.topology, kind)), set(getattr(without.topology, kind))
        out[kind] = {"with_port": len(a), "without_port": len(b), "shared": len(a & b), "only_with_port": len(a - b), "only_without_port": len(b - a)}
    groups = {}
    for name, g in with_port.topology.groups.items():
        other = without.topology.groups.get(name)
        groups[name] = {"members_with_port": len(g["ids"]), "members_without_port": len(other["ids"]) if other else None,
                        "identical_members": bool(other) and sorted(other["ids"]) == sorted(g["ids"])}
    out["groups"] = groups
    return out


def tables(trace, rows, stab) -> str:
    out = ["## Determinism (identity-determinism.json)", "",
           "| tiles | workers | PYTHONHASHSEED | pad_in_ground B with bands | random layout (60 features, seed 5, Route B) |", "|---|---|---|---|---|"]
    out += [f"| {r['tiles']} | {r['workers']} | {r['pythonhashseed']} | `{r['pad_in_ground_B_bands']}` | `{r['random_layout_60_seed5_B']}` |" for r in rows]
    out += ["", f"distinct digests per model: {len({r['pad_in_ground_B_bands'] for r in rows})} and {len({r['random_layout_60_seed5_B'] for r in rows})}", "",
            "## Stability under an unrelated edit (identity-stability.json)", "", "| kind | with port | without port | shared | only with port | only without port |", "|---|---|---|---|---|---|"]
    out += [f"| {k} | {stab[k]['with_port']} | {stab[k]['without_port']} | {stab[k]['shared']} | {stab[k]['only_with_port']} | {stab[k]['only_without_port']} |"
            for k in ("vertices", "edges", "faces", "volumes")]
    out += ["", "| group | members with port | members without port | identical members |", "|---|---|---|---|"]
    out += [f"| `{n}` | {g['members_with_port']} | {g['members_without_port']} | {g['identical_members']} |" for n, g in sorted(stab["groups"].items())]
    return "\n".join(out) + "\n"


def main() -> None:
    kit.ensure_plotly_js()
    spec = pad_in_ground("B", bands_um=MARGINS)
    result = build(spec)
    trace = build_trace(spec, result)
    figure_trace(spec, result, trace)
    (kit.FIGURES / "identity-trace.json").write_text(json.dumps(trace, indent=1))
    rows = determinism()
    (kit.FIGURES / "identity-determinism.json").write_text(json.dumps(rows, indent=1))
    stab = stability()
    (kit.FIGURES / "identity-stability.json").write_text(json.dumps(stab, indent=1))
    (kit.FIGURES / "identity-tables.md").write_text(tables(trace, rows, stab))
    print("digest of the example:", ids_digest(result))
    print("wrote identity-*")


if __name__ == "__main__":
    main()
