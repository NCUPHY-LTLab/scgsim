"""Excerpts and tables for ``consumer-contract.qmd`` and ``consumer-mapping.qmd``.

Run from the repository root with the package's environment (gmsh comes with the ``palace`` extra, OCP with the
``occt`` group)::

    uv run --extra palace --group dev --group occt python zh-TW/docs/design/sgb/_figures/consumer_contract.py

It builds a tiny model with ``build_sgb_geometry`` from the real inputs of the tests (``tests/sgb_inputs.py``):
``gds_pad_in_ground`` under Routes B and C (a 100 um die, a ground plane, a 20 um pad, one lumped port; bands of 0.5 and
1 um on the pad top) and ``thin_film_pair`` under Route A (bands on the sheet ``chip/left``).
The files go to a scratch folder.  On Routes B and C it then runs what a consumer runs: the Palace side (open the XAO in
Gmsh, ``verify_loaded``, ``apply_sidecar_embeds``, a coarse mesh) and the AEDT side (``export_aedt_step`` and
``read_step_guard``).  Every sidecar is read with the contract's ``read_sidecar`` and indexed by its tables and views.

It writes markdown blocks to ``../figures/`` with the prefix ``consumer-``.  Every JSON excerpt is a selection of real
records and keys of the ``scgsim.sgb.sidecar.v4`` sidecar; a list that is cut is shown with the string ``"..."`` as its
last item.  The Palace group dict is the one Palace builds, ``scgsim.palace.mesh.physical_groups.groups_from_sidecar``;
the AEDT rows group the written faces by ``face.group``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "tests"))

import gmsh  # noqa: E402
from sgb_inputs import gds_pad_in_ground, thin_film_pair  # noqa: E402

from scgsim.contracts.sgb_sidecar import read_sidecar, views  # noqa: E402
from scgsim.geometry.sgb import build_sgb_geometry  # noqa: E402
from scgsim.geometry.sgb.bands.requests import (  # noqa: E402
    ConductorSurface,
    InsetRequest,
)
from scgsim.geometry.sgb.checks.verification.loaded import (  # noqa: E402
    apply_sidecar_embeds,
    verify_loaded,
)
from scgsim.geometry.sgb.output.exchange.guard import read_step_guard  # noqa: E402
from scgsim.geometry.sgb.output.exchange.step import export_aedt_step  # noqa: E402
from scgsim.palace.mesh.physical_groups import groups_from_sidecar  # noqa: E402

STEM = "semantic_geometry_route_{}"
TITLE = "semantic_geometry_route_{}.sidecar.json（節錄）"
BAND_GROUP = "MA::pad__top::band=0"


# --------------------------------------------------------------------------- the model and what a consumer does with it


def build_models(out: Path) -> dict[str, Path]:
    """Write the three routes of the tiny model into ``out``; the folder of each route."""
    folders = {route: out / f"route_{route.lower()}" for route in "ABC"}
    effective = gds_pad_in_ground(out)
    for route in "BC":
        build_sgb_geometry(
            effective.build_input,
            effective.representation,
            route=route,
            insets=(
                InsetRequest("pad_top", ConductorSurface("pad", "top"), (0.5, 1.0)),
            ),
            output_dir=folders[route],
        )
    films = thin_film_pair()
    build_sgb_geometry(
        films.build_input,
        films.representation,
        route="A",
        insets=(
            InsetRequest(
                "band_left", ConductorSurface("chip/left", "sheet"), (0.5, 1.0)
            ),
        ),
        output_dir=folders["A"],
    )
    return folders


def consumer_run(folder: Path, route: str) -> dict[str, Any]:
    """The Palace-side sequence (open, synchronize, verify, embed, mesh) and the AEDT-side export and its guard."""
    stem = STEM.format(route.lower())
    sidecar = read_sidecar(folder / f"{stem}.sidecar.json")
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
        gmsh.open(str(folder / f"{stem}.xao"))
        gmsh.model.occ.synchronize()
        report = verify_loaded(sidecar)
        embedded = apply_sidecar_embeds(sidecar)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 60.0)
        gmsh.model.mesh.generate(3)
        mesh_nodes = len(gmsh.model.mesh.getNodes()[0])
    finally:
        gmsh.finalize()
    step = export_aedt_step(
        folder / f"{stem}.brep",
        folder / f"{stem}.sidecar.json",
        folder / "aedt_model.step",
    )
    guard = read_step_guard(step.guard_path)  # what the importer does first
    return {
        "verify_loaded": report,
        "apply_sidecar_embeds": embedded,
        "mesh_nodes": mesh_nodes,
        "object_names": sorted(step.object_names),
        "faces_per_solid": {s["aedt_name"]: len(s["faces"]) for s in guard["solids"]},
    }


# --------------------------------------------------------------------------- writing


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


def pick(table: dict[str, Any], *ids: str) -> dict[str, Any]:
    """The records ``ids`` of one table."""
    return {i: table[i] for i in ids}


# --------------------------------------------------------------------------- the excerpts


def excerpts(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any]) -> None:
    """The JSON excerpts of Route B, and the Route A and Route C differences."""
    title = TITLE.format("b")
    head = {k: b[k] for k in ("schema", "model", "summary")}
    write("consumer-excerpt-head.md", json_block(title, head))
    groups = pick(
        b["group"],
        BAND_GROUP,
        "MS::pad__bottom",
        "PORT::PORT_SHEET__P1__0000",
        "substrate",
    )
    write("consumer-excerpt-groups.md", json_block(title, {"group": groups}))
    rows = pick(
        b["surface_epr_row"], BAND_GROUP, "SA::substrate|VACUUM_REGION__interface"
    )
    write(
        "consumer-excerpt-requests.md",
        json_block(
            title,
            {
                "surface_epr_row": rows,
                "inset_request": b["inset_request"],
                "band_family": b["band_family"],
                "cut_family": b["cut_family"],
            },
        ),
    )
    write("consumer-excerpt-embeds.md", json_block(title, embeds(b)))
    write("consumer-excerpt-body.md", json_block(title, body(b)))
    route_a = {
        "model": {
            "route": a["model"]["route"],
            "route_policy": a["model"]["route_policy"],
        },
        "group": pick(a["group"], "chip/left::band=0"),
        "surface_epr_row": pick(
            a["surface_epr_row"],
            "MA::chip/left__sheet::band=0",
            "MS::chip/left__sheet::band=0",
        ),
        "port": {
            port_id: {
                k: port[k]
                for k in (
                    "group",
                    "host_entity_ids",
                    "host_volumes",
                    "adjacent_volumes",
                )
            }
            for port_id, port in views.ports(a)
        },
    }
    write("consumer-excerpt-routea.md", json_block(TITLE.format("a"), route_a))
    route_c = {
        "model": {
            "route": c["model"]["route"],
            "route_policy": c["model"]["route_policy"],
        },
        "group": pick(c["group"], "pad"),
    }
    write("consumer-excerpt-routec.md", json_block(TITLE.format("c"), route_c))


def embeds(b: dict[str, Any]) -> dict[str, Any]:
    """The port and the first mesh feature of each kind."""
    by_kind: dict[str, tuple[str, Any]] = {}
    for feature_id, feature in sorted(
        b["mesh_feature"].items(), key=lambda item: (item[1]["kind"], item[0])
    ):
        by_kind.setdefault(feature["kind"], (feature_id, shorten(feature, 3)))
    return {"port": b["port"], "mesh_feature": dict(by_kind.values())}


def body(b: dict[str, Any]) -> dict[str, Any]:
    """The binding, one volume, one face with its loop, one of its edges followed to its carrier, source curve and Entity."""
    # a ground sidewall (a wall between the substrate and the vacuum) and the substrate volume, found by their group
    # and Entity: ids change whenever the geometry does, the group and entity names do not
    face_id = next(
        fid
        for fid, record in sorted(b["face"].items())
        if record["group"] == "MA::ground_0000__sidewall"
    )
    volume_id = next(
        vid for vid, record in b["volume"].items() if record["entity"] == "substrate"
    )
    face = b["face"][face_id]
    edge_id = next(
        e
        for loop in face["loops"]
        for e in loop["edges"]
        if b["edge"][e]["kind"] == "horizontal"
    )
    edge = b["edge"][edge_id]
    (carrier_id,) = edge["carriers"]
    carrier = b["carrier"][carrier_id]
    source = b["source_curve"][carrier["source_curve"]]
    volume = shorten(b["volume"][volume_id], 4)
    return {
        "binding": {
            "rule": b["binding"]["rule"],
            "ids_by_tag": shorten(b["binding"]["ids_by_tag"]),
        },
        "volume": {volume_id: volume},
        "face": {face_id: face},
        "edge": {edge_id: edge},
        "vertex": pick(b["vertex"], *edge["vertices"]),
        "carrier": {carrier_id: carrier},
        "source_curve": {carrier["source_curve"]: source},
        "entity": pick(b["entity"], source["entity"]),
    }


# --------------------------------------------------------------------------- the consumer tables


def palace_rebuild(b: dict[str, Any]) -> None:
    """The group dict Palace builds from the sidecar, one record of each key."""
    groups = groups_from_sidecar(b)
    band = groups["boundary_surfaces"][BAND_GROUP]
    selection = {
        "volumes": {"substrate": groups["volumes"]["substrate"]},
        "boundary_surfaces": {
            BAND_GROUP: {k: shorten(v) for k, v in band.items() if k != "parts"}
        },
        "pec_surfaces": {
            BAND_GROUP: {
                k: shorten(v)
                for k, v in groups["pec_surfaces"][BAND_GROUP].items()
                if k != "parts"
            }
        },
        "port_surfaces": groups["port_surfaces"],
    }
    sizes = {k: len(v) for k, v in groups.items()}
    text = json_block("groups_from_sidecar(sidecar)（節錄）", selection)
    text += (
        "\n各鍵的項目數：" + "、".join(f"`{k}` {n}" for k, n in sizes.items()) + "。\n"
    )
    write("consumer-palace-rebuild.md", text)


def aedt_rows(b: dict[str, Any]) -> None:
    """The AEDT contribution rows of the groups: the written faces of each conductor and gap group, by ``face.group``."""
    by_group: dict[str, list[dict[str, Any]]] = {}
    for face in b["face"].values():
        group = b["group"][face["group"]] if face["group"] else None
        if group and group["interface_type"] in ("MA", "MS", "SA", "MM"):
            by_group.setdefault(face["group"], []).append(face)
    lines = [
        "| 群組（`face[*].group`） | 面數 | `interface_type` | `entities` | `face_kind` | `z_um` 範圍 | `adjacent_domains` |",
        "|:---|---:|:---|:---|:---|:---|:---|",
    ]
    for name, faces in sorted(by_group.items()):
        record = b["group"][name]
        kinds = ", ".join(f"`{k}`" for k in sorted({f["face_kind"] for f in faces}))
        assert kinds == f"`{record['face_kind']}`", (
            name
        )  # one face kind per group, equal to the group's own
        low = min(f["z_um"][0] for f in faces)
        high = max(f["z_um"][1] for f in faces)
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


def run_tables(
    runs: dict[str, dict[str, Any]], b: dict[str, Any], folder: Path
) -> None:
    """What a consumer measured on the tiny model: the load check, the embeddings, the STEP guard, the file sizes."""
    lines = ["| 項目 | Route B | Route C |", "|:---|:---|:---|"]
    rows = [
        ("`verify_loaded` 是否通過", lambda r: str(r["verify_loaded"].ok)),
        (
            "載入後與規劃的實體數（vertex／edge／face／solid）",
            lambda r: " / ".join(
                str(r["verify_loaded"].counts["native"][d]) for d in "0123"
            ),
        ),
        (
            "physical group（規劃／載入）",
            lambda r: (
                f"{r['verify_loaded'].groups['planned']} / {r['verify_loaded'].groups['loaded']}"
            ),
        ),
        (
            "vertex 最大偏移 (µm)",
            lambda r: f"{r['verify_loaded'].max_vertex_offset_um:g}",
        ),
        (
            "邊界不符的數量（edge、face、solid）",
            lambda r: ", ".join(
                str(r["verify_loaded"].boundaries_mismatching[k])
                for k in ("edges", "faces", "solids")
            ),
        ),
        (
            "face 面積最大相對誤差",
            lambda r: f"{r['verify_loaded'].max_relative_face_area_error:.2e}",
        ),
        (
            "solid 體積最大相對誤差",
            lambda r: f"{r['verify_loaded'].max_relative_volume_error:.2e}",
        ),
        (
            "`apply_sidecar_embeds` 嵌入的 sheet face",
            lambda r: str(r["apply_sidecar_embeds"]),
        ),
        (
            "粗 mesh（`Mesh.MeshSizeMax` = 60 µm）的節點數",
            lambda r: str(r["mesh_nodes"]),
        ),
        (
            "`export_aedt_step` 的 object 名稱",
            lambda r: "、".join(f"`{n}`" for n in r["object_names"]),
        ),
        (
            "各 solid 的 guard face 數",
            lambda r: "、".join(f"{n} {k}" for n, k in r["faces_per_solid"].items()),
        ),
    ]
    for label, fmt in rows:
        lines.append(f"| {label} | {fmt(runs['B'])} | {fmt(runs['C'])} |")
    write("consumer-run-table.md", "\n".join(lines) + "\n")

    lines = ["| `model.files` 的鍵 | 檔名（Route B） | 大小 (kB) |", "|:---|:---|---:|"]
    for key in ("xao", "brep", "sidecar", "sliver_report", "validation_report"):
        name = b["model"]["files"][key]
        size = (folder / name).stat().st_size / 1000
        lines.append(f"| `{key}` | `{name}` | {size:.1f} |")
    write("consumer-files-table.md", "\n".join(lines) + "\n")


def main() -> None:
    """Build the tiny model, run the consumers on it and write every include."""
    with tempfile.TemporaryDirectory(prefix="consumer-contract-") as scratch:
        folders = build_models(Path(scratch))
        a, b, c = (
            read_sidecar(folders[r] / f"{STEM.format(r.lower())}.sidecar.json")
            for r in "ABC"
        )
        runs = {route: consumer_run(folders[route], route) for route in "BC"}
        excerpts(a, b, c)
        palace_rebuild(b)
        aedt_rows(b)
        run_tables(runs, b, folders["B"])


if __name__ == "__main__":
    main()
