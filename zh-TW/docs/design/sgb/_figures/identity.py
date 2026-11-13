"""Figures and tables for ``docs/design/sgb/exact/identity.qmd``.

Run, with the Python of the checkout that contains the production SGB core (``scgsim.geometry.sgb``)::

    SGB_CORE=<sgb-core> <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/identity.py

Computes, with the production package, a real mapping-chain lookup on the pad-in-ground running example (Route B with
inset bands, its lumped port on the support plane z = 0) and the determinism evidence quoted by the page.  The model is
built, written with ``emit_model`` into a scratch folder and the lookup reads the written sidecar
(``scgsim.sgb.sidecar.v3``) through the contract's reader and views, exactly as a consumer would: every step follows a
typed reference of one record to another, and no id is parsed.  Outputs go to ``docs/design/sgb/figures/`` (prefix ``identity-``):

* ``identity-trace.json``        every step of the lookup  Palace attribute -> source curve (also quoted in the page)
* ``identity-trace-plan.html``   the traced faces and edge in the plan view (+ ``.svg``)
* ``identity-determinism.json``  one model digest per PYTHONHASHSEED; all must agree
* ``identity-stability.json``    which atomic ids survive an unrelated edit (removing the lumped port)
* ``identity-tables.md``         markdown tables quoted by the page
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
from inset_kit import G_UM as G
from inset_kit import MARGINS, Line, Panel, Poly, Pts  # noqa: E402

from scgsim.contracts.sgb_sidecar import read_sidecar, views  # noqa: E402
from scgsim.geometry.sgb.build import emit_model  # noqa: E402

CASE = (
    "pad_b_bands"  # tests/sgb_emit_cases.py name of the case; the port is on z = 0 here
)
TARGET_GROUP = "MA::pad__top::band=2"
ALT_GROUP = "MA::pad__top::band=0"
WALL_GROUP = "MA::pad__sidewall"
TRACED_FAMILY = "band_pad_top"
TRACED_MARGIN_UM = (
    0.5  # the traced edge is the 0.5 um offset of the pad's rounded corner
)


def describe_edge(sidecar, edge_id: str) -> dict:
    """One written edge and each of its carriers, followed by reference to what it comes from.

    A horizontal or refinement edge names its carriers (``edge.carriers`` -> ``carrier``); a band member leads to the
    support primitive it follows, the primitive's ring and band family, and the source curves drawn along it; a source
    carrier leads to its source curve and Entity.  A vertical edge names the Entities whose walls stand on it.
    """
    edge = sidecar["edge"][edge_id]
    return {
        "edge": edge_id,
        "kind": edge["kind"],
        "carriers": list(edge["carriers"]),
        "wall_entities": list(edge["entities"]),
        "resolved": [resolve_carrier(sidecar, c) for c in edge["carriers"]],
    }


def resolve_carrier(sidecar, carrier_id: str) -> dict:
    """What one carrier record says it comes from, every reference followed."""
    carrier = sidecar["carrier"][carrier_id]
    record: dict = {"carrier": carrier_id, "record": carrier}
    if carrier["kind"] == "source":
        record["source_curve"] = sidecar["source_curve"][carrier["source_curve"]]
    elif carrier["kind"] == "band_member":
        primitive = sidecar["support_primitive"][carrier["primitive"]]
        ring = sidecar["support_ring"][primitive["ring"]]
        record.update(
            margin_um=carrier["margin_grid"] * sidecar["model"]["grid_um"],
            support_primitive=primitive,
            position_in_ring=ring["primitives"].index(carrier["primitive"]),
            ring_size=len(ring["primitives"]),
            support_ring={
                "id": primitive["ring"],
                "family": ring["family"],
                "index": ring["index"],
            },
            source_curves={c: sidecar["source_curve"][c] for c in primitive["sources"]},
        )
    elif carrier["kind"] == "refinement":
        record["refinement_request"] = sidecar["refinement_request"][carrier["request"]]
    elif carrier["kind"] == "family_member":
        record["cut_family"] = sidecar["cut_family"][carrier["family"]]
    return record


def face_loops(sidecar, face_id: str) -> list[list[dict]]:
    """The written loops of a face (``face.loops``), every edge described with the direction the loop travels it."""
    return [
        [
            describe_edge(sidecar, eid) | {"reversed": backwards}
            for eid, backwards in zip(loop["edges"], loop["reversed"], strict=True)
        ]
        for loop in sidecar["face"][face_id]["loops"]
    ]


def _build_trace(model, sidecar) -> dict:
    members = views.group_members(sidecar)
    trace = {
        "case": CASE,
        "route": "B",
        "margins_um": list(MARGINS),
        "schema": sidecar["schema"],
    }
    group = sidecar["group"][TARGET_GROUP]
    attribute = group["tag"]
    by_tag = {g["tag"]: name for name, g in sidecar["group"].items() if g["dim"] == 2}
    trace["step1_palace_attribute"] = {
        "attribute": attribute,
        "resolved_name": by_tag[attribute],
        "source": "sidecar: group[*].tag",
    }
    trace["step2_physical_group"] = {
        "name": TARGET_GROUP,
        "record": {k: group[k] for k in sorted(group)},
        "faces": members[TARGET_GROUP],
        "native_tags": views.native_tags(sidecar, 2, members[TARGET_GROUP]),
        "surface_epr_rows": {
            row_id: row
            for row_id, row in sidecar["surface_epr_row"].items()
            if row["group"] == TARGET_GROUP
        },
    }
    trace["step3_atomic_faces"] = [
        _face_entry(model, sidecar, fid) for fid in members[TARGET_GROUP]
    ]
    alt_face = members[ALT_GROUP][0]
    trace["alt_group"] = {
        "name": ALT_GROUP,
        "face": alt_face,
        "entities": sidecar["face"][alt_face]["entities"],
        "families": sidecar["face"][alt_face]["families"],
        "membership": sidecar["face"][alt_face]["membership"],
        "loops": face_loops(sidecar, alt_face),
    }
    wall = next(
        fid
        for fid in members[WALL_GROUP]
        if sidecar["face"][fid]["kind"] == "wall" and not sidecar["face"][fid]["curved"]
    )
    trace["wall"] = {
        "name": WALL_GROUP,
        "face": wall,
        "z_um": sidecar["face"][wall]["z_um"],
        "entities": sidecar["face"][wall]["entities"],
        "membership": sidecar["face"][wall]["membership"],
        "loops": face_loops(sidecar, wall),
    }
    port_entities = {port["entity"] for port in sidecar["port"].values()}
    trace["walls_with_port_membership"] = [
        {"face": fid, "group": face["group"], "membership": face["membership"]}
        for fid, face in sorted(sidecar["face"].items())
        if face["kind"] == "wall" and port_entities & set(face["membership"])
    ]
    trace["step7_band_family"] = {
        "band_family": sidecar["band_family"][TRACED_FAMILY],
        "inset_request": sidecar["inset_request"][TRACED_FAMILY],
    }
    return trace


def _face_entry(model, sidecar, fid: str) -> dict:
    """One face of the target group: its tag, its record and its loops with every edge followed to its sources."""
    record = sidecar["face"][fid]
    edge_ids = [e for loop in record["loops"] for e in loop["edges"]]
    return {
        "face": fid,
        "native_tag": views.native_tags(sidecar, 2, [fid])[0],
        "record": {k: record[k] for k in sorted(record) if k != "loops"},
        "plane": model.topology.faces[fid].plane,
        "edge_ids": edge_ids,
        "edge_tags": views.native_tags(sidecar, 1, edge_ids),
        "loops": face_loops(sidecar, fid),
    }


def traced_edge(sidecar, trace) -> tuple[dict, dict]:
    """The edge of the target group on the 0.5 um offset of the pad's rounded corner, and its band-member carrier.

    The corner is the one arc among the primitives of the family's outer support ring (ring index 0); the edge is the
    one whose carrier offsets that primitive at the traced margin.  Everything is read from the sidecar's records.
    """
    ring_id = next(
        r
        for r, ring in sidecar["support_ring"].items()
        if ring["family"] == TRACED_FAMILY and ring["index"] == 0
    )
    corner = next(
        p
        for p in sidecar["support_ring"][ring_id]["primitives"]
        if sidecar["support_primitive"][p]["curve"]["kind"] == "arc"
    )
    margin_grid = round(TRACED_MARGIN_UM / sidecar["model"]["grid_um"])
    for face in trace["step3_atomic_faces"]:
        for loop in face["loops"]:
            for edge in loop:
                for resolved in edge["resolved"]:
                    carrier = resolved["record"]
                    if (
                        carrier["kind"] == "band_member"
                        and carrier["primitive"] == corner
                        and carrier["margin_grid"] == margin_grid
                    ):
                        return edge, resolved
    raise SystemExit("no edge of the target group offsets the rounded corner")


def distance_to_curve(curve: dict, x: float, y: float) -> float:
    """Distance (um) from a point to the circle of a sidecar arc record, or to the line of a line record."""
    if curve["kind"] == "arc":
        cx, cy = curve["centre_um"]
        return abs(math.hypot(x - cx, y - cy) - curve["radius_um"])
    (x0, y0), (x1, y1) = curve["start_um"], curve["end_um"]
    return abs((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) / math.hypot(
        x1 - x0, y1 - y0
    )


def _figure_trace(model, sidecar, trace) -> None:
    layer = next(
        a for a in model.labelled.level_layers if a.plane == "level[1/5]"
    )  # the pad's top face, z = 0.2 um
    face_ids = {f["face"] for f in trace["step3_atomic_faces"]}
    chosen, followed = traced_edge(sidecar, trace)
    primitive = followed["support_primitive"]
    polys = []
    for fi in range(1, len(layer.arr.faces)):
        fid = layer.face_ids[fi]
        face = sidecar["face"].get(fid)
        group = face["group"] if face and face["group"] else "(not a face)"
        facts = sidecar["group"].get(group, {})
        rings = kit.face_rings_um(layer, fi)
        if fid in face_ids:
            polys.append(
                Poly(
                    rings, "#e08214", "rgba(0,0,0,0.8)", 0.8, hover=f"{group}<br>{fid}"
                )
            )
        elif (facts.get("band") or {}).get("family") == TRACED_FAMILY:
            polys.append(
                Poly(
                    rings, "#c6dbef", "rgba(0,0,0,0.5)", 0.4, hover=f"{group}<br>{fid}"
                )
            )
        elif facts.get("interface_type") == "PORT" or "ground" in facts.get(
            "entities", ()
        ):
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
    # the support primitive the carrier offsets, as the sidecar records it (exact arc, distance origin)
    curve = primitive["curve"]
    arc = sampled_arc(curve, 60)
    mid = pts[len(pts) // 2]
    deviation = distance_to_curve(curve, mid[0] * G, mid[1] * G)
    ring = followed["support_ring"]["index"]
    position = followed["position_in_ring"]
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
            name="other MA::pad__top band faces",
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
                name=f"support curve (ring {ring}, primitive {position})",
            ),
            Line(
                xs,
                ys,
                kit.PORT,
                3.0,
                name=f"traced edge (carrier {followed['carrier']})",
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
        "carrier": followed["carrier"],
        "carrier_record": followed["record"],
        "margin_um": followed["margin_um"],
        "support_primitive": followed["record"]["primitive"],
        "support_ring": followed["support_ring"],
        "position_in_ring": position,
        "support_curve": curve,
        "source_curves": followed["source_curves"],
        "distance_to_support_um_at_midpoint": round(deviation, 4),
    }


def sampled_arc(curve: dict, steps: int) -> list[tuple[float, float]]:
    """Points (um) along a sidecar arc record from its start to its end, in its sense of travel."""
    cx, cy = curve["centre_um"]
    r = curve["radius_um"]
    t0 = math.atan2(curve["start_um"][1] - cy, curve["start_um"][0] - cx)
    t1 = math.atan2(curve["end_um"][1] - cy, curve["end_um"][0] - cx)
    sweep = (t1 - t0) % (2 * math.pi) if curve["ccw"] else -((t0 - t1) % (2 * math.pi))
    return [
        (
            cx + r * math.cos(t0 + sweep * i / steps),
            cy + r * math.sin(t0 + sweep * i / steps),
        )
        for i in range(steps + 1)
    ]


# --------------------------------------------------------------------------- determinism evidence

DIGEST_SCRIPT = (
    "import sys; sys.path.insert(0, {kit_dir!r})\n"
    "import inset_kit as kit\n"
    "from sgb_scenes import random_layout\n"
    "from scgsim.geometry.sgb.model import build_model_from_catalogue\n"
    "a = kit.pad_model('B', kit.MARGINS).identity_digest()\n"
    "b = build_model_from_catalogue(random_layout(60, seed=5, route='B'), 'B').identity_digest()\n"
    "print(a, b)\n"
)


def determinism() -> list[dict]:
    """The model digest of two models in a fresh interpreter for several hash seeds."""
    rows = []
    for hashseed in ("0", "1", "12345"):
        code = DIGEST_SCRIPT.format(kit_dir=str(Path(__file__).resolve().parent))
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
        "| PYTHONHASHSEED | pad_in_ground B with bands | random layout (60 features, seed 5, Route B) |",
        "|---|---|---|",
    ]
    out += [
        f"| {r['pythonhashseed']} | `{r['pad_in_ground_B_bands']}` | `{r['random_layout_60_seed5_B']}` |"
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
    _figure_trace(model, sidecar, trace)
    trace["model_digest"] = model.identity_digest()
    trace["counts"] = {
        **model.topology.counts(),
        "groups": len(model.routed.groups),
        "emitted": sidecar["summary"]["counts"],
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
