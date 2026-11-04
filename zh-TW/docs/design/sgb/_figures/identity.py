"""Figures and tables for ``docs/design/sgb/identity.qmd``.

Run, with the Python of the checkout that contains the production SGB core (``scgsim.geometry.sgb``)::

    SGB_CORE=<sgb-core> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/identity.py

Computes, with the production package, a real mapping-chain lookup on the pad-in-ground running example (Route B with
inset bands, its lumped port on the support plane z = 0) and the determinism evidence quoted by the page.  The model is
built, written with ``emit_model`` into a scratch folder and the lookup reads the written sidecar
(``scgsim.sgb.sidecar.v1``) exactly as a consumer would.  Outputs go to ``docs/design/sgb/figures/`` (prefix ``identity-``):

* ``identity-trace.json``        every step of the lookup  Palace attribute -> source curve (also quoted in the page)
* ``identity-trace-plan.html``   the traced faces and edge in the plan view (+ ``.svg``)
* ``identity-determinism.json``  one model digest per (tile grid, workers, PYTHONHASHSEED); all must agree
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
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
from inset_kit import G_UM as G
from inset_kit import MARGINS, Line, Panel, Poly, Pts  # noqa: E402
from scgsim.geometry.sgb.build import emit_model  # noqa: E402
from scgsim.geometry.sgb.records.carriers import ArcCarrier, base_carrier_id  # noqa: E402
from scgsim.geometry.sgb.records.support import distance_to_primitive  # noqa: E402
from scgsim.geometry.sgb.sidecar.files import read_sidecar  # noqa: E402

CASE = (
    "pad_b_bands"  # tests/sgb_emit_cases.py name of the case; the port is on z = 0 here
)
TARGET_GROUP = "MA::pad::top::band=2"
ALT_GROUP = "MA::pad::top::band=0"
WALL_GROUP = "MA::pad::sidewall"
CARRIER = re.compile(
    r"^(?P<family>[^/]+)/ring(?P<ring>\d+)/(?P<kind>edge|corner)(?P<index>\d+)(?:\.(?P<piece>\d+))?@m(?P<margin>\d+(?:\.\d+)?)$"
)


def describe_edge(model, sidecar, edge_id: str) -> dict:
    """One edge with its carriers read from the sidecar lineage, resolved to the generating support curve."""
    supports = {
        f.family_id: f.support for f in model.labelled.catalogue.families if f.support
    }
    arcs = model.labelled.catalogue.arc_registry()
    carriers = list(
        sidecar["lineage"]["edges"].get(edge_id)
        or sidecar["lineage"]["vertical_edges"].get(edge_id, ())
    )
    entry = {"edge": edge_id, "carriers": carriers, "resolved": []}
    for carrier in carriers:
        match = CARRIER.match(carrier)
        if match:
            family = supports[match["family"]]
            ring, index = int(match["ring"]), int(match["index"])
            record = {
                "kind": "family member",
                "lineage_record": sidecar["lineage"]["carriers"][carrier],
                "family": match["family"],
                "margin_nm": int(float(match["margin"])),
                "support_ring": ring,
                "member": f"{match['family']}/ring{ring}/{match['kind']}{index}",
            }
            if match["kind"] == "edge":
                primitive = family.rings[ring][index]
                if isinstance(primitive, ArcCarrier):
                    record.update(
                        {
                            "support_curve": primitive.id,
                            "source": primitive.lineage.describe(),
                            "centre_um": [float(c) * G for c in primitive.centre],
                            "radius_um": float(primitive.radius) * G,
                        }
                    )
                else:
                    record.update(
                        {"support_curve": "straight edge", "support_primitive": index}
                    )
            else:
                record["support_curve"] = f"reflex-corner arc {index} of ring {ring}"
            entry["resolved"].append(record)
        elif "/polygon" in carrier:
            entity, polygon, ring, edge_index = carrier.split("/")
            entry["resolved"].append(
                {
                    "kind": "source edge",
                    "entity": entity,
                    "polygon": polygon,
                    "ring": ring,
                    "edge": edge_index,
                    "carrier": carrier,
                }
            )
        elif base_carrier_id(carrier) in arcs:
            arc = arcs[base_carrier_id(carrier)]
            entry["resolved"].append(
                {
                    "kind": "source arc carrier",
                    "carrier": carrier,
                    "source": arc.lineage.describe(),
                    "centre_um": [float(c) * G for c in arc.centre],
                    "radius_um": float(arc.radius) * G,
                }
            )
        else:
            entry["resolved"].append(
                {"kind": "source entity (vertical edge)", "entity": carrier}
            )
    return entry


def face_loops(model, sidecar, face_id: str) -> list[list[dict]]:
    """The boundary loops of a face, every edge described (the loop order and orientation come from the planned topology)."""
    return [
        [
            describe_edge(model, sidecar, eid) | {"orientation": sign}
            for eid, sign in loop
        ]
        for loop in model.topology.faces[face_id].loops
    ]


def _build_trace(model, sidecar) -> dict:
    topo = model.topology
    ids_by_tag = sidecar["binding"]["ids_by_tag"]
    native = {fid: 1 + i for i, fid in enumerate(ids_by_tag["2"])}
    trace = {
        "case": CASE,
        "route": "B",
        "margins_um": list(MARGINS),
        "schema": sidecar["schema"],
    }
    group = sidecar["physical_groups"][TARGET_GROUP]
    attribute = group["tag"]
    by_tag = {
        g["tag"]: name
        for name, g in sidecar["physical_groups"].items()
        if g["dim"] == 2
    }
    trace["step1_palace_attribute"] = {
        "attribute": attribute,
        "resolved_name": by_tag[attribute],
        "source": "sidecar: physical_groups[*].tag",
    }
    assert [native[f] for f in group["ids"]] == group["native_tags"]
    trace["step2_physical_group"] = {
        "name": TARGET_GROUP,
        "record": {k: group[k] for k in sorted(group)},
        "union_membership": [
            f"{kind}.{u}"
            for kind, unions in sidecar["unions"].items()
            for u, members in unions.items()
            if TARGET_GROUP in members
        ],
        "surface_epr_row": sidecar["surface_epr_rows"][TARGET_GROUP],
    }
    faces = []
    for fid in group["ids"]:
        face = topo.faces[fid]
        tag = native[fid]
        edge_tags = sidecar["expected"]["face_edges"][tag - 1]
        faces.append(
            {
                "face": fid,
                "native_tag": tag,
                "record": sidecar["faces"][fid],
                "plane": face.plane,
                "label_bands": [list(b) for b in face.bands],
                "sources_lineage": sidecar["lineage"]["faces"][fid],
                "membership": sidecar["lineage"]["membership"][fid],
                "edge_tags": edge_tags,
                "edge_ids": [ids_by_tag["1"][t - 1] for t in edge_tags],
                "loops": face_loops(model, sidecar, fid),
            }
        )
    trace["step3_atomic_faces"] = faces
    alt_face = sidecar["physical_groups"][ALT_GROUP]["ids"][0]
    trace["alt_group"] = {
        "name": ALT_GROUP,
        "face": alt_face,
        "sources_lineage": sidecar["lineage"]["faces"][alt_face],
        "membership": sidecar["lineage"]["membership"][alt_face],
        "loops": face_loops(model, sidecar, alt_face),
    }
    wall = next(
        fid
        for fid in sidecar["physical_groups"][WALL_GROUP]["ids"]
        if topo.faces[fid].kind == "wall" and not topo.faces[fid].curved
    )
    wall_face = topo.faces[wall]
    trace["wall"] = {
        "name": WALL_GROUP,
        "face": wall,
        "z_range": [str(z) for z in wall_face.z_range],
        "sources_lineage": sidecar["lineage"]["faces"][wall],
        "membership": sidecar["lineage"]["membership"][wall],
        "loops": face_loops(model, sidecar, wall),
    }
    touched = [
        (fid, m)
        for fid, m in sidecar["lineage"]["membership"].items()
        if "p1" in m and sidecar["faces"].get(fid, {}).get("kind") == "wall"
    ]
    trace["walls_with_port_membership"] = [
        {"face": fid, "group": sidecar["faces"][fid]["group"], "membership": m}
        for fid, m in sorted(touched)
    ]
    request = next(i for i in sidecar["insets"] if i["request_id"] == "band_pad_top")
    trace["step7_inset_request"] = {
        k: request[k]
        for k in ("request_id", "support", "margins_um", "support_area_um2")
    }
    return trace


def rounded_corner_index(model) -> int:
    """The primitive of the pad's outer ring that is the 5 um rounded corner (the support ring starts at an arc)."""
    support = next(
        f.support
        for f in model.labelled.catalogue.families
        if f.family_id == "band_pad_top"
    )
    return next(i for i, p in enumerate(support.rings[0]) if isinstance(p, ArcCarrier))


def _figure_trace(model, trace) -> None:
    layer = next(
        a for a in model.labelled.level_layers if a.plane == "level[1/5]"
    )  # the pad's top face, z = 0.2 um
    face_ids = {f["face"] for f in trace["step3_atomic_faces"]}
    prim_index = rounded_corner_index(model)
    traced = f"band_pad_top/ring0/edge{prim_index}@m500"  # the rounded corner's 0.5 um offset arc
    chosen = next(
        e
        for f in trace["step3_atomic_faces"]
        for loop in f["loops"]
        for e in loop
        if any(c.startswith(traced) for c in e["carriers"])
    )
    polys = []
    for fi in range(1, len(layer.arr.faces)):
        fid = layer.face_ids[fi]
        group = model.routed.face_group.get(fid) or "(not a face)"
        rings = kit.face_rings_um(layer, fi)
        if fid in face_ids:
            polys.append(
                Poly(
                    rings, "#e08214", "rgba(0,0,0,0.8)", 0.8, hover=f"{group}<br>{fid}"
                )
            )
        elif group.startswith("MA::pad::top::band"):
            polys.append(
                Poly(
                    rings, "#c6dbef", "rgba(0,0,0,0.5)", 0.4, hover=f"{group}<br>{fid}"
                )
            )
        elif group.startswith(("PORT", "MA::ground")):
            polys.append(
                Poly(
                    rings, "#e8e8e8", "rgba(0,0,0,0.3)", 0.3, hover=f"{group}<br>{fid}"
                )
            )
        else:
            polys.append(
                Poly(
                    rings, "#f7f7f7", "rgba(0,0,0,0.2)", 0.3, hover=f"{group}<br>{fid}"
                )
            )
    # the chosen edge, drawn from the arrangement
    index = layer.edge_ids.index(chosen["edge"])
    pts = kit.edge_polyline(layer.arr, layer.edge_curve, index, 40.0)
    xs, ys = [p[0] * G for p in pts], [p[1] * G for p in pts]
    # the support primitive the carrier was generated from (exact arc, distance origin)
    support = next(
        f.support
        for f in model.labelled.catalogue.families
        if f.family_id == "band_pad_top"
    )
    ring = 0
    prim = support.rings[ring][prim_index]
    t0, sweep = (float(v) for v in prim.sweep())
    cx, cy, r = float(prim.centre[0]), float(prim.centre[1]), float(prim.radius)
    arc = [
        (
            (cx + r * math.cos(t0 + sweep * i / 60)) * G,
            (cy + r * math.sin(t0 + sweep * i / 60)) * G,
        )
        for i in range(61)
    ]
    mid = pts[len(pts) // 2]
    deviation = distance_to_primitive(prim, mid[0], mid[1]) * G
    legend = [
        Pts(
            [-1e4],
            [-1e4],
            "#e08214",
            8,
            name=f"faces of {TARGET_GROUP}",
            symbol="square",
        ),
        Pts(
            [-1e4],
            [-1e4],
            "#c6dbef",
            8,
            name="other MA::pad::top band faces",
            symbol="square",
        ),
    ]
    left = Panel(
        "pad top face at z = 0.2 um (Route B)",
        (-55, 55),
        (-35, 35),
        xlabel="x (um)",
        ylabel="y (um)",
        items=polys
        + [Line(xs, ys, kit.PORT, 3.0, name="traced edge", hover=chosen["edge"])]
        + legend,
    )
    zoom = Panel(
        "zoom: the traced edge, its support curve and its distance",
        (-52, -38),
        (-32, -20),
        xlabel="x (um)",
        ylabel="y (um)",
        items=polys
        + [
            Line(
                [p[0] for p in arc],
                [p[1] for p in arc],
                kit.INK,
                2.4,
                name=f"support curve {prim.id} (ring {ring}, primitive {prim_index})",
            ),
            Line(
                xs,
                ys,
                kit.PORT,
                3.0,
                name=f"traced edge (carrier {chosen['carriers'][0]})",
                hover=chosen["edge"],
            ),
        ],
    )
    kit.write_both(
        [left, zoom],
        "identity-trace-plan",
        ncols=2,
        svg_size=(12, 4.8),
        html_size=(1250, 560),
        shared_legend=True,
        title="Faces named by one physical group, and one of their edges traced back to the source arc",
    )
    trace["traced_edge"] = {
        "edge": chosen["edge"],
        "carrier": chosen["carriers"][0],
        "distance_to_support_um_at_midpoint": round(deviation, 4),
        "support_curve": prim.id,
    }


# --------------------------------------------------------------------------- determinism evidence

DIGEST_SCRIPT = (
    "import sys; sys.path.insert(0, {kit_dir!r})\n"
    "import inset_kit as kit\n"
    "from sgb_scenes import random_layout\n"
    "from scgsim.geometry.sgb.model import build_model_from_catalogue\n"
    "tiles, workers = {tiles}, {workers}\n"
    "a = kit.pad_model('B', kit.MARGINS, tiles=tiles, workers=workers).identity_digest()\n"
    "b = build_model_from_catalogue(random_layout(60, seed=5, route='B'), 'B', tiles=tiles, workers=workers).identity_digest()\n"
    "print(a, b)\n"
)


def determinism() -> list[dict]:
    """The model digest of two models in a fresh interpreter for several tile grids, worker counts and hash seeds."""
    rows = []
    for tiles, workers in ((1, 1), (2, 1), (3, 2), (4, 4)):
        for hashseed in ("0", "1", "12345"):
            code = DIGEST_SCRIPT.format(
                kit_dir=str(Path(__file__).resolve().parent),
                tiles=tiles,
                workers=workers,
            )
            env = {**os.environ, "PYTHONHASHSEED": hashseed}
            out = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
            rows.append(
                {
                    "tiles": f"{tiles}x{tiles}",
                    "workers": workers,
                    "pythonhashseed": hashseed,
                    "pad_in_ground_B_bands": out[0],
                    "random_layout_60_seed5_B": out[1],
                }
            )
    return rows


def stability() -> dict:
    """Ids of the Route B example with bands, with and without the lumped port (an unrelated edit for every band boundary)."""
    with_port = kit.pad_model("B", MARGINS)
    without = kit.pad_model("B", MARGINS, with_port=False)
    out = {
        "same_group_names": sorted(with_port.routed.groups)
        == sorted(without.routed.groups)
        or "differ"
    }
    for kind in ("vertices", "edges", "faces", "volumes"):
        a, b = (
            set(getattr(with_port.topology, kind)),
            set(getattr(without.topology, kind)),
        )
        out[kind] = {
            "with_port": len(a),
            "without_port": len(b),
            "shared": len(a & b),
            "only_with_port": len(a - b),
            "only_without_port": len(b - a),
        }
    groups = {}
    for name, g in with_port.routed.groups.items():
        other = without.routed.groups.get(name)
        groups[name] = {
            "members_with_port": len(g.ids),
            "members_without_port": len(other.ids) if other else None,
            "identical_members": bool(other) and sorted(other.ids) == sorted(g.ids),
        }
    out["groups"] = groups
    out["groups_only_in_one_build"] = sorted(
        set(with_port.routed.groups) ^ set(without.routed.groups)
    )
    return out


def _tables(rows, stab) -> str:
    out = [
        "## Determinism (identity-determinism.json)",
        "",
        "| tiles | workers | PYTHONHASHSEED | pad_in_ground B with bands | random layout (60 features, seed 5, Route B) |",
        "|---|---|---|---|---|",
    ]
    out += [
        f"| {r['tiles']} | {r['workers']} | {r['pythonhashseed']} | `{r['pad_in_ground_B_bands']}` | `{r['random_layout_60_seed5_B']}` |"
        for r in rows
    ]
    out += [
        "",
        f"distinct digests per model: {len({r['pad_in_ground_B_bands'] for r in rows})} and {len({r['random_layout_60_seed5_B'] for r in rows})}",
        "",
        "## Stability under an unrelated edit (identity-stability.json)",
        "",
        "| kind | with port | without port | shared | only with port | only without port |",
        "|---|---|---|---|---|---|",
    ]
    out += [
        f"| {k} | {stab[k]['with_port']} | {stab[k]['without_port']} | {stab[k]['shared']} | {stab[k]['only_with_port']} | {stab[k]['only_without_port']} |"
        for k in ("vertices", "edges", "faces", "volumes")
    ]
    out += [
        "",
        f"groups in only one of the two builds: {', '.join(stab['groups_only_in_one_build']) or 'none'}",
        "",
        "| group | members with port | members without port | identical members |",
        "|---|---|---|---|",
    ]
    out += [
        f"| `{n}` | {g['members_with_port']} | {g['members_without_port']} | {g['identical_members']} |"
        for n, g in sorted(stab["groups"].items())
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    """Write the identity trace figure, the determinism and stability tables and their numbers."""
    kit.ensure_plotly_js()
    model = kit.pad_model("B", MARGINS)
    with tempfile.TemporaryDirectory(prefix="identity-") as scratch:
        sidecar = read_sidecar(emit_model(model, Path(scratch)).sidecar)
    trace = _build_trace(model, sidecar)
    _figure_trace(model, trace)
    trace["model_digest"] = model.identity_digest()
    trace["counts"] = {
        **model.topology.counts(),
        "groups": len(model.routed.groups),
        "emitted": sidecar["counts"],
    }
    (kit.FIGURES / "identity-trace.json").write_text(json.dumps(trace, indent=1))
    rows = determinism()
    (kit.FIGURES / "identity-determinism.json").write_text(json.dumps(rows, indent=1))
    stab = stability()
    (kit.FIGURES / "identity-stability.json").write_text(json.dumps(stab, indent=1))
    (kit.FIGURES / "identity-tables.md").write_text(_tables(rows, stab))
    print("model digest of the example:", trace["model_digest"])
    print("wrote identity-*")


if __name__ == "__main__":
    main()
