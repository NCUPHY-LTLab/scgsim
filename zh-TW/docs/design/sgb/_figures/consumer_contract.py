"""Excerpts and tables for ``consumer-contract.qmd`` and ``consumer-mapping.qmd``.

Run, after ``experiments/sgb-redesign/consumer-contract/generate_excerpt.py`` has written the files of a
tiny model with the sgb-core package (the instructions are in ``_figures/README.md``)::

    <sgb-core>/.artifacts/env/bin/python docs/design/sgb/_figures/consumer_contract.py

Nothing here builds geometry.  It reads the sidecars ``out/route_b`` and ``out/route_c`` and the
``out/consumer_runs.json`` that the generator wrote, and writes markdown blocks to ``../figures/`` with the
prefix ``consumer-``.  Every JSON excerpt is a selection of real keys; a list that is cut is shown with the
string ``"..."`` as its last item.  The Palace group dict is rebuilt by the sidecar test's own function
``palace_style_groups`` (sgb-core ``tests/test_sgb_sidecar.py``); the AEDT rows apply the rule of the sidecar
test ``test_aedt_can_rebuild_its_contribution_ledger_from_the_sidecar``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
ROOT = HERE.parents[4]
OUT = ROOT / "experiments" / "sgb-redesign" / "consumer-contract" / "out"
# The checkout that contains scgsim.geometry.sgb and its tests (defaults to this one).
SGB_CORE_TESTS = Path(os.environ.get("SGB_CORE", str(ROOT))) / "tests"
STEM = "semantic_geometry_route_{}"


def load(route: str) -> dict[str, Any]:
    """The sidecar of ``route`` ("b" or "c")."""
    path = OUT / f"route_{route}" / f"{STEM.format(route)}.sidecar.json"
    return json.loads(path.read_text(encoding="utf-8"))


def shorten(value: Any, keep: int = 2) -> Any:
    """Cut every list reached through dictionaries to ``keep`` items and mark the cut with ``"..."``; kept items stay whole."""
    if isinstance(value, list):
        return value[:keep] + ["..."] if len(value) > keep else value
    if isinstance(value, dict):
        return {k: shorten(v, keep) for k, v in value.items()}
    return value


def pretty(value: Any, indent: int = 0) -> str:
    """JSON with sorted keys; dictionaries break into lines, lists of plain values stay on one line."""
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f'{pad} "{k}": {pretty(v, indent + 1)}' for k, v in sorted(value.items())
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        items = [f"{pad} {pretty(v, indent + 1)}" for v in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=False)


def json_block(title: str, payload: Any) -> str:
    """A fenced, titled JSON block."""
    return f'```{{.json filename="{title}"}}\n{pretty(payload)}\n```\n'


def write(name: str, text: str) -> None:
    """Write one markdown include."""
    (FIGURES / name).write_text(text, encoding="utf-8")
    print("wrote", name)


def excerpts(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any]) -> None:
    """The JSON excerpts of Route B (and the one Route C difference)."""
    title = "semantic_geometry_route_b.sidecar.json（節錄）"
    head = {
        k: b[k]
        for k in (
            "schema",
            "route",
            "route_policy",
            "grid_um",
            "tolerance_um",
            "model",
            "files",
            "counts",
        )
    }
    write("consumer-excerpt-head.md", json_block(title, head))
    groups = {
        n: b["physical_groups"][n]
        for n in (
            "MA::pad::top::band=0",
            "MS::pad::bottom",
            "PORT::PORT_SHEET__P1__0000",
            "substrate",
        )
    }
    write("consumer-excerpt-groups.md", json_block(title, {"physical_groups": groups}))
    rows = {
        n: b["surface_epr_rows"][n]
        for n in ("MA::pad::top::band=0", "SA::substrate|VACUUM_REGION::interface")
    }
    write(
        "consumer-excerpt-unions.md",
        json_block(
            title,
            {"unions": b["unions"], "surface_epr_rows": rows, "insets": b["insets"]},
        ),
    )
    write(
        "consumer-excerpt-embeds.md",
        json_block(
            title,
            {
                "embedded_faces": b["embedded_faces"],
                "refinement_edges": b["refinement_edges"],
                "mesh_hints": [shorten(h, 3) for h in b["mesh_hints"]],
                "ports": b["ports"],
            },
        ),
    )
    # a ground sidewall (a wall between the substrate and the vacuum) and the substrate volume, found by name:
    # ids change whenever the geometry does, the group and entity names do not
    face = next(
        fid
        for fid, record in b["faces"].items()
        if record["group"] == "MA::ground_0000::sidewall"
    )
    volume = next(
        vid for vid, record in b["volumes"].items() if record["entity"] == "substrate"
    )
    lineage = b["lineage"]
    pick_edge = next(iter(lineage["edges"]))
    pick_carrier = lineage["edges"][pick_edge][0]
    body = {
        "binding": {
            "rule": b["binding"]["rule"],
            "ids_by_tag": shorten(b["binding"]["ids_by_tag"]),
        },
        "volumes": {volume: b["volumes"][volume]},
        "faces": {face: b["faces"][face]},
        "lineage": {
            "edges": {pick_edge: lineage["edges"][pick_edge]},
            "carriers": {pick_carrier: lineage["carriers"][pick_carrier]},
            "faces": {face: lineage["faces"][face]},
            "membership": {face: lineage["membership"][face]},
        },
        "slivers": b["slivers"],
        "arrangement": b["arrangement"],
        "expected": shorten(b["expected"]),
    }
    body["expected"]["solid_faces"] = [
        row[:8] + ["..."] if len(row) > 8 else row
        for row in body["expected"]["solid_faces"]
    ]
    write("consumer-excerpt-body.md", json_block(title, body))
    route_a = {
        "route_policy": a["route_policy"],
        "physical_groups": {
            "chip/left::band=0": a["physical_groups"]["chip/left::band=0"]
        },
        "surface_epr_rows": {
            n: a["surface_epr_rows"][n]
            for n in ("MA::chip/left::sheet::band=0", "MS::chip/left::sheet::band=0")
        },
        "unions": a["unions"],
        "embedded_faces": a["embedded_faces"],
        "ports": [
            {
                k: a["ports"][0][k]
                for k in (
                    "group",
                    "host_entity_ids",
                    "host_volumes",
                    "adjacent_volumes",
                )
            }
        ],
    }
    write(
        "consumer-excerpt-routea.md",
        json_block("semantic_geometry_route_a.sidecar.json（節錄）", route_a),
    )
    conductor = {n: c["physical_groups"][n] for n in ("pad",)}
    write(
        "consumer-excerpt-routec.md",
        json_block(
            "semantic_geometry_route_c.sidecar.json（節錄）",
            {
                "route": c["route"],
                "route_policy": c["route_policy"],
                "physical_groups": conductor,
            },
        ),
    )


def palace_rebuild(b: dict[str, Any]) -> None:
    """The Palace-style group dict rebuilt by the sidecar test's function."""
    sys.path.insert(0, str(SGB_CORE_TESTS))
    from test_sgb_sidecar import palace_style_groups  # noqa: PLC0415 -- needs the sgb-core test folder

    groups = palace_style_groups(b)
    selection = {
        "volumes": {"substrate": groups["volumes"]["substrate"]},
        "boundary_surfaces": {
            "MA::pad::top::band=0": groups["boundary_surfaces"]["MA::pad::top::band=0"]
        },
        "pec_surfaces": {
            "MA::pad::top::band=0": groups["pec_surfaces"]["MA::pad::top::band=0"]
        },
        "port_surfaces": groups["port_surfaces"],
    }
    sizes = {k: len(v) for k, v in groups.items()}
    text = json_block("palace_style_groups(sidecar)（節錄）", selection)
    text += (
        "\n各鍵的項目數：" + "、".join(f"`{k}` {n}" for k, n in sizes.items()) + "。\n"
    )
    write("consumer-palace-rebuild.md", text)


def aedt_rows(b: dict[str, Any]) -> None:
    """The AEDT contribution rows of the groups, by the rule of the sidecar test."""
    by_group: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for fid, face in b["faces"].items():
        group = b["physical_groups"].get(face["group"]) if face["group"] else None
        if group and group["interface_type"] in ("MA", "MS", "SA", "MM"):
            by_group.setdefault(face["group"], []).append((fid, face))
    lines = [
        "| 群組（`faces[*].group`） | 面數 | `interface_type` | `entities` | `face_kind` | `z_um` 範圍 | `adjacent_domains` |",
        "|:---|---:|:---|:---|:---|:---|:---|",
    ]
    for name, faces in sorted(by_group.items()):
        record = b["physical_groups"][name]
        kinds = ", ".join(f"`{k}`" for k in sorted({f["face_kind"] for _, f in faces}))
        assert kinds == f"`{record['face_kind']}`", (
            name
        )  # one face kind per group, equal to the group's own
        low = min(f["z_um"][0] for _, f in faces)
        high = max(f["z_um"][1] for _, f in faces)
        z_text = f"{low:g}" if low == high else f"{low:g} 到 {high:g}"
        entities = "、".join(f"`{e}`" for e in record["entities"])
        domains = "、".join(
            f"`{d['name']}`（{d['kind']}）" for d in record["adjacent_domains"]
        )
        lines.append(
            f"| {_code(name)} | {len(faces)} | `{record['interface_type']}` | {entities} | {kinds} | {z_text} | {domains} |"
        )
    write("consumer-aedt-rows-table.md", "\n".join(lines) + "\n")


def _code(name: str) -> str:
    """Inline code for a group name; ``|`` inside a table cell is written as an HTML entity."""
    return "<code>" + name.replace("|", "&#124;") + "</code>"


def run_tables(b: dict[str, Any]) -> None:
    """What a consumer measured on the tiny model: the load check, the embeddings, the STEP guard, the file sizes."""
    runs = json.loads((OUT / "consumer_runs.json").read_text(encoding="utf-8"))
    lines = [
        "| 項目 | Route B | Route C |",
        "|:---|:---|:---|",
    ]

    def pair(fmt) -> tuple[str, str]:
        return fmt(runs["B"]), fmt(runs["C"])

    rows = [
        ("`verify_loaded` 是否通過", lambda r: str(r["verify_loaded"]["ok"])),
        (
            "載入後與規劃的實體數（vertex／edge／face／solid）",
            lambda r: " / ".join(
                str(r["verify_loaded"]["counts"]["native"][d]) for d in "0123"
            ),
        ),
        (
            "physical group（規劃／載入）",
            lambda r: (
                f"{r['verify_loaded']['groups']['planned']} / {r['verify_loaded']['groups']['loaded']}"
            ),
        ),
        (
            "vertex 最大偏移 (µm)",
            lambda r: f"{r['verify_loaded']['max_vertex_offset_um']:g}",
        ),
        (
            "邊界不符的數量（edge、face、solid）",
            lambda r: ", ".join(
                str(r["verify_loaded"]["boundaries_mismatching"][k])
                for k in ("edges", "faces", "solids")
            ),
        ),
        (
            "face 面積最大相對誤差",
            lambda r: f"{r['verify_loaded']['max_relative_face_area_error']:.2e}",
        ),
        (
            "solid 體積最大相對誤差",
            lambda r: f"{r['verify_loaded']['max_relative_volume_error']:.2e}",
        ),
        (
            "`apply_sidecar_embeds` 嵌入的 face／edge",
            lambda r: (
                f"{r['apply_sidecar_embeds']['faces']} / {r['apply_sidecar_embeds']['edges']}"
            ),
        ),
        (
            "粗 mesh（`Mesh.MeshSizeMax` = 60 µm）的節點數",
            lambda r: str(r["mesh_nodes"]),
        ),
        (
            "`export_aedt_step` 的 object 名稱",
            lambda r: "、".join(f"`{n}`" for n in r["step_export"]["object_names"]),
        ),
        (
            "各 solid 的 guard face 數",
            lambda r: "、".join(
                f"{n} {k}" for n, k in r["step_export"]["faces_per_solid"].items()
            ),
        ),
    ]
    for label, fmt in rows:
        first, second = pair(fmt)
        lines.append(f"| {label} | {first} | {second} |")
    write("consumer-run-table.md", "\n".join(lines) + "\n")

    files = ["xao", "brep", "sidecar", "sliver_report", "validation_report"]
    lines = ["| `files` 的鍵 | 檔名（Route B） | 大小 (kB) |", "|:---|:---|---:|"]
    for key in files:
        name = b["files"][key]
        size = (OUT / "route_b" / name).stat().st_size / 1000
        lines.append(f"| `{key}` | `{name}` | {size:.1f} |")
    write("consumer-files-table.md", "\n".join(lines) + "\n")


def main() -> None:
    """Write every include."""
    a, b, c = load("a"), load("b"), load("c")
    excerpts(a, b, c)
    palace_rebuild(b)
    aedt_rows(b)
    run_tables(b)


if __name__ == "__main__":
    main()
