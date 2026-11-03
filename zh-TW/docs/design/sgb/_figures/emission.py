"""Figures and tables for ``docs/design/sgb/emission.qmd``.

Run (from anywhere)::

    /home/ili/Githubs/SCQ_Design/packages/scgsim/.venv/bin/python docs/design/sgb/_figures/emission.py

Everything is computed from the read-only prototype (``experiments/sgb-redesign/prototype``): the E3 outputs
under ``outputs/e3_brep/`` (``timing.json``, ``sidecar_timing.json``, per-case ``validation.json`` /
``model.sidecar.json`` / ``model.xao``) and a fresh build of the pad-in-ground example (Route B) whose numbering is
checked against the ``ids_by_tag`` of the written sidecar.
Outputs go to ``docs/design/sgb/figures/`` with the prefix ``emission-``:

* ``emission-timing``         write and load cost of the gmsh-API emitter against the direct writer (html + svg)
* ``emission-shared-face``    one shared face referenced + by a shell and - by the other (svg; schematic plus real references)
* ``emission-groups-3d``      the reopened XAO meshed on its surfaces, coloured by physical group (html + svg)
* ``emission-*-table.md``     markdown tables included by the page (counts, timing, sidecar, tags, groups, validation, OCCT, bands, step)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
PROTOTYPE = HERE.parents[4] / "experiments" / "sgb-redesign" / "prototype"
E3 = PROTOTYPE / "outputs" / "e3_brep"
OCCT = PROTOTYPE / "outputs" / "occt"
OCCT_CASES = ("airbridge_B", "pad_in_ground_A", "pad_in_ground_A_bands", "pad_in_ground_B", "pad_in_ground_B_bands")
RUNNING_EXAMPLE = "pad_in_ground_B"
CASES = ("pad_in_ground_A", "pad_in_ground_B", "airbridge_B")
BAND_CASES = ("pad_in_ground_A_bands", "pad_in_ground_B_bands")

sys.path.insert(0, str(PROTOTYPE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from sgbproto.brep_writer import plan_brep  # noqa: E402
from sgbproto.pipeline import build  # noqa: E402
from sgbproto.scenes import pad_in_ground  # noqa: E402

INK = "#1b1b1b"
STAGE_COLOURS = {  # one colour per named stage, shared by the plotly and matplotlib versions
    "OCC build (addPoint ... addVolume)": "#D55E00",
    "synchronize": "#E69F00",
    "physical groups + embed": "#F0E442",
    "write XAO": "#CC79A7",
    "other (start-up, bookkeeping)": "#BBBBBB",
    "plan (serial)": "#0072B2",
    "render (16 workers)": "#56B4E9",
    "load into Gmsh": "#444444",
}
GROUP_COLOURS = {
    "MA::pad": "#0072B2", "MS::pad": "#56B4E9", "MA::Ground": "#8C8C8C", "MS::Ground": "#CFCFCF",
    "SA::substrate|vacuum": "#E69F00", "PORT::p1": "#D55E00",
    "EXT::substrate": "#009E73", "EXT::vacuum": "#CC79A7",
}
GROUP_OPACITY = {"EXT::substrate": 0.12, "EXT::vacuum": 0.12}
MESH_SIZE_MAX, MESH_SIZE_MIN = 40.0, 2.0  # um; coarse on purpose: the view only needs the surfaces


# --------------------------------------------------------------------------- data


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


def _running_example():
    """The pad-in-ground example (Route B), its BREP plan, and the check that its numbering equals ``tags.json``."""
    topo = build(pad_in_ground("B")).topology
    plan = plan_brep(topo)
    sidecar = _load_json(E3 / RUNNING_EXAMPLE / "model.sidecar.json")
    recorded = {pid: (int(dim), i + 1) for dim, ids in sidecar["ids_by_tag"].items() for i, pid in enumerate(ids)}
    if recorded != plan.tags():
        raise SystemExit("the prototype's numbering no longer equals ids_by_tag of outputs/e3_brep/pad_in_ground_B/model.sidecar.json; rerun the E3 example")
    return topo, plan


def _sec(x: float) -> str:
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}" if x >= 1 else f"{x:.3f}"


def _table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    rule = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(rule) + "|"]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _write(name: str, text: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / name).write_text(text)
    print("wrote", name)


# --------------------------------------------------------------------------- timing


def _timing_rows() -> list[dict]:
    """One row per N with the stacked-bar segments of both emitters (seconds)."""
    rows = []
    for r in _load_json(E3 / "timing.json"):
        api = r["gmsh_api"]
        stages = api["stages_s"]
        other = api["total_s"] - sum(stages.values())
        direct = r["direct"]["xao_w16"]
        rows.append({
            "n": r["n_features"],
            "api": {"OCC build (addPoint ... addVolume)": stages["occ_build"], "synchronize": stages["synchronize"],
                    "physical groups + embed": stages["physical_groups"] + stages["embed"], "write XAO": stages["write_xao"],
                    "other (start-up, bookkeeping)": other, "load into Gmsh": r["load"]["gmsh_api_xao"]["load_s"]},
            "direct": {"plan (serial)": direct["plan"], "render (16 workers)": direct["render"],
                       "load into Gmsh": r["load"]["direct_xao"]["load_s"]},
            "raw": r,
        })
    return rows


def _timing_figures(rows: list[dict]) -> None:
    labels = ["gmsh API", "direct writer"]
    fig = make_subplots(rows=1, cols=len(rows), subplot_titles=[f"N = {r['n']:,} features" for r in rows],
                        horizontal_spacing=0.07)
    seen: set[str] = set()
    for col, r in enumerate(rows, start=1):
        for side, label in zip(("api", "direct"), labels):
            for stage, value in r[side].items():
                fig.add_trace(go.Bar(
                    x=[label], y=[value], name=stage, marker_color=STAGE_COLOURS[stage], legendgroup=stage,
                    showlegend=stage not in seen, hovertemplate=f"{stage}: %{{y:.3f}} s<extra>{label}, N = {r['n']:,}</extra>",
                ), row=1, col=col)
                seen.add(stage)
        for label, side in zip(labels, ("api", "direct")):
            fig.add_annotation(x=label, y=sum(r[side].values()), text=f"{sum(r[side].values()):.1f} s", showarrow=False,
                               yshift=10, row=1, col=col, font=dict(size=11, color=INK))
    fig.update_layout(barmode="stack", width=1000, height=470, paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color=INK, size=12), legend=dict(orientation="h", y=-0.12), margin=dict(l=55, r=15, t=45, b=20))
    fig.update_yaxes(title_text="seconds (write + load)", row=1, col=1, gridcolor="#e5e5e5")
    for col in range(2, len(rows) + 1):
        fig.update_yaxes(gridcolor="#e5e5e5", row=1, col=col)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIGURES / "emission-timing.html", include_plotlyjs="directory", config={"displaylogo": False})
    print("wrote emission-timing.html")

    sfig, axes = plt.subplots(1, len(rows), figsize=(10.5, 4.4))
    for ax, r in zip(axes, rows):
        for i, side in enumerate(("api", "direct")):
            bottom = 0.0
            for stage, value in r[side].items():
                ax.bar(i, value, bottom=bottom, color=STAGE_COLOURS[stage], width=0.7, edgecolor="white", linewidth=0.5,
                       label=stage)
                bottom += value
            ax.text(i, bottom, f"{bottom:.1f} s", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1], labels, fontsize=8)
        ax.set_title(f"N = {r['n']:,} features", fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("seconds (write + load)", fontsize=8)
    handles, names = axes[-1].get_legend_handles_labels()
    unique = dict(zip(names, handles))
    sfig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    sfig.tight_layout(rect=(0, 0.1, 1, 1))
    sfig.savefig(FIGURES / "emission-timing.svg", facecolor="white")
    plt.close(sfig)
    print("wrote emission-timing.svg")


def _timing_tables(rows: list[dict]) -> None:
    api_rows, direct_rows, load_rows, worker_rows = [], [], [], []
    for r in rows:
        raw, d = r["raw"], r["raw"]["direct"]
        api, stages = raw["gmsh_api"], raw["gmsh_api"]["stages_s"]
        ld = raw["load"]
        api_rows.append([
            f"{r['n']:,}", f"{raw['counts']['faces']:,}", _sec(raw["build_s"]), _sec(stages["occ_build"]), _sec(stages["synchronize"]),
            _sec(stages["physical_groups"] + stages["embed"]), _sec(stages["write_xao"]), _sec(api["total_s"]),
        ])
        direct_rows.append([
            f"{r['n']:,}", _sec(d["xao_w16"]["plan"]), _sec(d["xao_w16"]["render"]), _sec(d["xao_w16"]["total"]), _sec(d["xao_w1"]["total"]),
            f"{api['total_s'] / d['xao_w16']['total']:.1f}x", f"{api['total_s'] / d['xao_w1']['total']:.1f}x",
        ])
        load_rows.append([
            f"{r['n']:,}", f"{api['xao_bytes'] / 1e6:.1f} / {d['xao_bytes'] / 1e6:.1f} / {d['brep_bytes'] / 1e6:.1f}",
            _sec(ld["gmsh_api_xao"]["load_s"]), _sec(ld["direct_xao"]["load_s"]), _sec(ld["direct_brep"]["load_s"]),
            _sec(api["total_s"] + ld["gmsh_api_xao"]["load_s"]), _sec(d["xao_w16"]["total"] + ld["direct_xao"]["load_s"]),
        ])
        for workers in (1, 8, 16):
            t = d[f"xao_w{workers}"]
            pc = d["plan_counts"]
            per_entity = t["total"] / (pc["vertices"] + pc["edges"] + pc["faces"]) * 1e6
            worker_rows.append([f"{r['n']:,}", str(workers), _sec(t["plan_walk_s"]), _sec(t["plan_arrays_s"]), _sec(t["plan"]),
                                _sec(t["render"]), _sec(t["total"]), f"{per_entity:.1f}"])
    _write("emission-timing-api-table.md", _table(
        ["N", "規劃的 face 數", "推導 (s)", "OCC build (s)", "synchronize (s)", "groups 與 embed (s)", "寫出 XAO (s)", "API 合計 (s)"],
        api_rows))
    _write("emission-timing-direct-table.md", _table(
        ["N", "plan (s)", "算繪，16 worker (s)", "合計，16 worker (s)", "合計，1 worker (s)", "加速倍數，16 worker", "加速倍數，1 worker"],
        direct_rows))
    _write("emission-timing-load-table.md", _table(
        ["N", "檔案大小：API XAO / 直接 XAO / 直接 BREP (MB)", "載入 API XAO (s)", "載入直接 XAO (s)", "載入直接 BREP (s)",
         "寫出加載入，API (s)", "寫出加載入，直接 (s)"], load_rows))
    _write("emission-timing-workers-table.md", _table(
        ["N", "worker 數", "plan 走訪 (s)", "plan 陣列與檢查 (s)", "plan (s)", "算繪 (s)", "合計 (s)",
         "每個寫出元素的 µs"], worker_rows))


# --------------------------------------------------------------------------- running example: counts, tags, groups, validation


def _counts_table(plan) -> None:
    validation = _load_json(E3 / RUNNING_EXAMPLE / "validation.json")
    c = validation["plan_counts"]
    uses = {}
    for shell in plan.shells:
        for ref in shell:
            uses[abs(ref)] = uses.get(abs(ref), 0) + 1
    shared = sum(1 for n in uses.values() if n == 2)
    rows = [
        ["vertex", str(c["vertices"]), "`Ve`", "edge"],
        ["edge", str(c["edges"]), "`Ed`", "wire"],
        ["wire（迴圈）", str(c["wires"]), "`Wi`", "face"],
        ["face", f"{c['faces']}（其中 {c['free_faces']} 個為 free face：port sheet）", "`Fa`", "shell；free face 另由 compound 引用"],
        ["shell", str(c["shells"]), "`Sh`", "solid"],
        ["solid", str(c["solids"]), "`So`", "compound"],
        ["compound", "1", "`Co`", "無（根節點）"],
    ]
    _write("emission-counts-table.md", _table(["元素", "pad_in_ground_B 中的數量", "record", "被誰引用"], rows, "lrll"))
    print(f"  faces used by two shells: {shared}")


def _tag_rows(topo, plan) -> list[list[str]]:
    """A hand-picked but fully computed selection of planned ids with their native tags and XAO references."""
    tags = plan.tags()
    refs = plan.refs
    group_of_face = {plan.face_ids[i]: (name, tag) for name, kind, tag, elements in plan.groups if kind == "face" for i in elements}
    group_of_solid = {plan.solid_ids[i]: (name, tag) for name, kind, tag, elements in plan.groups if kind == "solid" for i in elements}
    picks = []  # (what, planned id, dim, index)
    for i, vid in enumerate(plan.solid_ids):
        picks.append((f"solid，{topo.volumes[vid].domain.name}", vid, 3, i))
    first_face_of = {}
    for i, fid in enumerate(plan.face_ids):
        first_face_of.setdefault(group_of_face[fid][0], i)
    for name in ("SA::substrate|vacuum", "MS::pad", "MA::pad", "PORT::p1", "MA::Ground"):
        picks.append((f"face，{name} 的第一個", plan.face_ids[first_face_of[name]], 2, first_face_of[name]))
    curved = next(i for i, fid in enumerate(plan.face_ids) if topo.faces[fid].curved and plan.face_flipped[i])
    picks.append(("face，curved wall (cylinder)，wire 已反向", plan.face_ids[curved], 2, curved))
    picks.append(("edge，第一個寫出", plan.edge_ids[0], 1, 0))
    picks.append(("vertex，第一個寫出", plan.vertex_ids[0], 0, 0))
    kind_key = {3: "solid", 2: "face", 1: "edge", 0: "vertex"}
    rows = []
    for what, pid, dim, index in picks:
        dim_tag = tags[pid]
        assert dim_tag == (dim, index + 1)
        group = (group_of_solid if dim == 3 else group_of_face).get(pid) if dim >= 2 else None
        rows.append([f"`{pid}`", what, f"{group[0]} (tag {group[1]})" if group else "無", f"({dim_tag[0]}, {dim_tag[1]})",
                     str(index), str(refs[kind_key[dim]][index])])
    return rows


def _tags_and_groups_tables(topo, plan) -> None:
    _write("emission-tags-table.md", _table(
        ["planned id", "元素", "primary physical group", "Gmsh (dim, tag)", "XAO index", "XAO reference"],
        _tag_rows(topo, plan), "llllrr"))
    rows, covered = [], 0
    for name, kind, tag, elements in plan.groups:
        rows.append([name, kind, str(tag), str(len(elements))])
        covered += len(elements) if kind == "face" else 0
    faces_emitted = len(plan.face_ids)
    assert covered == faces_emitted, "physical groups do not partition the emitted faces"
    assert len({i for _, kind, _, el in plan.groups if kind == "face" for i in el}) == covered
    _write("emission-groups-table.md", _table(["primary group", "維度", "physical tag", "成員數"], rows, "llrr"))
    print(f"  face groups cover {covered} of {faces_emitted} emitted faces, each exactly once")


def _bands_table() -> None:
    rows = []
    for case in BAND_CASES:
        r = _load_json(E3 / case / "validation.json")["verify_xao"]
        mesh = r["mesh"]
        bands = mesh["bands"]
        rows.append([f"`{case}`", f"{r['groups']['loaded']} of {r['groups']['planned']}", str(len(bands)),
                     f"{r['max_relative_face_area_error']:.2g}", f"{r['max_relative_volume_error_vs_plan']:.2g}",
                     f"{mesh['nodes']:,} nodes，{mesh['tets']:,} tets，conformal：{'是' if mesh['conformal'] else '否'}",
                     f"{sum(1 for b in bands.values() if b['all_shared_by_tets'] and b['centroids_in_band'])} of {len(bands)}"])
    _write("emission-bands-table.md", _table(
        ["case", "已載入 groups", "band group 數", "face 面積最大誤差", "volume 最大誤差", "粗略 3-D mesh",
         "triangle 全為 tet face 且落在 band 內的 band group 數"], rows, "lrrrrll"))


def _sidecar_table() -> None:
    data = _load_json(E3 / "sidecar_timing.json")
    rows = []
    for n, row in data.items():
        plain, full = row["no_lineage"], row["lineage"]
        rows.append([f"{int(n):,}", _sec(plain["build_s"]), _sec(plain["json_s"]), f"{plain['MB']:.1f}",
                     _sec(full["build_s"]), _sec(full["json_s"]), f"{full['MB']:.1f}"])
    _write("emission-sidecar-table.md", _table(
        ["N", "建立 (s)", "輸出 JSON (s)", "大小 (MB)", "含 lineage 建立 (s)", "含 lineage 輸出 JSON (s)", "含 lineage 大小 (MB)"], rows))


def _validation_table() -> None:
    reports = {case: _load_json(E3 / case / "validation.json") for case in CASES}

    def counts(case: str, which: str) -> str:
        n = reports[case][which]["counts"]["native"]
        return " / ".join(str(n[str(d)]) for d in range(4))

    def mismatches(case: str, which: str) -> str:
        m = reports[case][which]["boundaries_mismatching"]
        return f"{m['edges']} / {m['faces']} / {m['solids']}"

    def groups(case: str) -> str:
        g = reports[case]["verify_xao"]["groups"]
        return f"{g['loaded']} of {g['planned']}"

    def volume_error(case: str, which: str) -> str:
        return f"{reports[case][which]['max_relative_volume_error_vs_plan']:.2g}"

    def mesh(case: str) -> str:
        m = reports[case]["verify_xao"]["mesh"]
        return f"{m['nodes']} nodes，{m['tets']} tets，{m['duplicate_nodes']} 個重複 node，conformal：{'是' if m['conformal'] else '否'}"

    analytic = reports["pad_in_ground_B"]["analytic_volumes_xao"]
    analytic_text = "；".join(f"{k} {v['loaded']:.9g}，解析值 {v['expected']:.9g} µm³" for k, v in analytic.items())
    rows = [
        ["Gmsh 中的元素數（vertex / edge / face / solid），與規劃相符", *(counts(c, "verify_xao") for c in CASES)],
        ["以 `getBoundary` 檢查的邊界不一致數（edge / face / solid）", *(mismatches(c, "verify_xao") for c in CASES)],
        ["vertex 與規劃位置的最大偏差 (µm)", *(f"{reports[c]['verify_xao']['max_vertex_offset_um']:g}" for c in CASES)],
        ["從 XAO 載入的 physical group 數（BREP 不含 group）", *(groups(c) for c in CASES)],
        ["volume 對規劃的最大相對誤差（XAO）", *(volume_error(c, "verify_xao") for c in CASES)],
        ["載入後 XAO 的粗略 3-D mesh", *(mesh(c) for c in CASES)],
        ["face 面積對規劃的最大相對誤差", *(f"{reports[c]['verify_xao']['max_relative_face_area_error']:.2g}" for c in CASES)],
        ["sidecar 與 gmsh-API oracle 的 sidecar 相同（groups、unions、embeds、各維度 ids）", *("是" if reports[c]["sidecar_vs_gmsh_api_sidecar"]["equal"] else "否" for c in CASES)],
        ["解析 volume（封閉形式）", "未計算", analytic_text, "未計算"],
    ]
    _write("emission-validation-table.md", _table(["載入後的檢查", "pad_in_ground_A", "pad_in_ground_B", "airbridge_B"], rows, "llll"))

    _bands_table()
    step_rows = []
    for case in CASES + BAND_CASES:
        r = reports.get(case) or _load_json(E3 / case / "validation.json")
        step = r["step"]
        before, after = step["entities_before"], step["entities_after_reload"]
        shared = r["verify_xao"]["mesh"]["shared_interior_faces"]
        step_rows.append([f"`{case}`", " / ".join(map(str, before)), " / ".join(map(str, after)), str(after[2] - before[2]), str(shared),
                          f"{step['bytes'] / 1e3:.1f}", ", ".join(f"{v:.9g}" for v in step["volumes_after_reload"])])
    _write("emission-step-table.md", _table(
        ["case", "寫出的元素數（V / E / F / S）", "STEP 重新載入後的元素數", "多出的 face 數", "shared interior face 數",
         "STEP 大小 (kB)", "重新載入後各 solid 的 volume (µm³)"], step_rows, "lllrrrl"))


# --------------------------------------------------------------------------- shared face figure


def _face_reference(plan, face_index: int) -> int:
    """The number a shell uses for this face: its position counted from the end of the TShapes section."""
    return plan._bases()["F"] - face_index


def _shared_face_figure(topo, plan) -> None:
    """Schematic cross-section (conceptual) with the real reference numbers of pad_in_ground_B."""
    shared = [abs(r) for r in plan.shells[0] if abs(r) in {abs(x) for x in plan.shells[1]}]
    face_no = shared[0]  # 1-based face number (index + 1) of the first shared face
    ref = _face_reference(plan, face_no - 1)
    face_id = plan.face_ids[face_no - 1]
    group = next(name for name, kind, _, el in plan.groups if kind == "face" and face_no - 1 in el)
    signs = {}
    for k, shell in enumerate(plan.shells):
        signs[k] = next(1 if r > 0 else -1 for r in shell if abs(r) == face_no)
    domain = {k: topo.volumes[plan.solid_ids[k]].domain.name for k in range(2)}
    upper = next(k for k in range(2) if signs[k] < 0)  # the shell that uses the face with "-" lies above it
    lower = 1 - upper
    sym = {1: "+", -1: "-"}

    both_shared = sorted(_face_reference(plan, f - 1) for f in shared)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.add_patch(Rectangle((0.2, 2.1), 4.8, 1.7, facecolor="#f4dcea", edgecolor="#999", linewidth=1))
    ax.add_patch(Rectangle((0.2, 0.4), 4.8, 1.7, facecolor="#d6efe5", edgecolor="#999", linewidth=1))
    ax.plot([0.2, 5.0], [2.1, 2.1], color="#E69F00", linewidth=4, solid_capstyle="butt")
    ax.text(2.6, 3.35, f"solid {domain[upper]}\nits shell uses the face as {sym[signs[upper]]}{ref}", ha="center", va="center", fontsize=10, color=INK)
    ax.text(2.6, 0.95, f"solid {domain[lower]}\nits shell uses the face as {sym[signs[lower]]}{ref}", ha="center", va="center", fontsize=10, color=INK)
    ax.text(3.3, 2.2, f"one face, written once ({group})", ha="center", va="bottom", fontsize=8.5, color="#7a4a00")
    for y1, text, label_y in ((2.9, "outward\nfrom upper\nsolid", 2.78), (1.3, "outward\nfrom lower\nsolid", 1.65)):
        ax.add_patch(FancyArrowPatch((0.8, 2.1), (0.8, y1), arrowstyle="-|>", mutation_scale=14, color="#444", linewidth=1.4))
        ax.text(0.95, label_y, text, fontsize=7.5, va="center", color="#444")
    ax.add_patch(FancyArrowPatch((4.6, 2.1), (4.6, 2.9), arrowstyle="-|>", mutation_scale=14, color="#b00020", linewidth=1.8))
    ax.text(4.45, 2.6, "face normal\n(plane +z)", fontsize=7.5, ha="right", va="center", color="#b00020")
    ax.text(0.2, 4.0, "Conceptual cross-section at the substrate / vacuum interface", fontsize=9, color="#555")

    def record(shell_index: int) -> str:
        """The shell record with the two shared references in brackets; the rest is elided after the first eight."""
        refs = plan.shells[shell_index]
        items = [f"{'+' if r > 0 else '-'}{_face_reference(plan, abs(r) - 1)}" for r in refs]
        keep = [i for i, r in enumerate(refs) if i < 8 or abs(r) in shared]
        out, last = [], -1
        for i in keep:
            if i != last + 1:
                out.append("...")
            out.append(f"[{items[i]}]" if abs(refs[i]) in shared else items[i])
            last = i
        if last != len(refs) - 1:
            out.append("...")
        return "Sh  " + " ".join(out) + f"   ({len(refs)} faces)"

    ax.text(5.4, 3.85, "Shell records in pad_in_ground_B/model.brep (real)", fontsize=9, color="#555", va="top")
    ax.text(5.4, 3.35, f"shell of the {domain[upper]} solid:", fontsize=9, va="top", color=INK)
    ax.text(5.4, 3.05, record(upper), fontsize=7.5, va="top", family="monospace", color=INK)
    ax.text(5.4, 2.5, f"shell of the {domain[lower]} solid:", fontsize=9, va="top", color=INK)
    ax.text(5.4, 2.2, record(lower), fontsize=7.5, va="top", family="monospace", color=INK)
    ax.text(5.4, 1.5, f"Brackets mark the two faces both shells use (references {both_shared[0]} and {both_shared[1]});\n"
            f"each appears once with + and once with -. The face drawn is {face_id}.", fontsize=8, va="top", color="#444")

    fig.savefig(FIGURES / "emission-shared-face.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote emission-shared-face.svg")
    print(f"  shared face: planned id {face_id}, reference {ref}, group {group}, upper {domain[upper]} ({sym[signs[upper]]}), lower {domain[lower]} ({sym[signs[lower]]})")


# --------------------------------------------------------------------------- reopened XAO, coloured by group


def _group_triangles(plan):
    """Mesh the surfaces of the XAO and return per face entity: (group name, tag, points, triangles)."""
    import gmsh

    out = []
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
        gmsh.open(str(E3 / RUNNING_EXAMPLE / "model.xao"))
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_MAX)
        gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE_MIN)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.model.mesh.generate(2)
        group_of = {}
        for dim, ptag in gmsh.model.getPhysicalGroups(2):
            name = gmsh.model.getPhysicalName(dim, ptag)
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
                group_of[int(entity)] = name
        for _, tag in gmsh.model.getEntities(2):
            node_tags, coords, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True, returnParametricCoord=False)
            types, _, element_nodes = gmsh.model.mesh.getElements(2, tag)
            triangles = np.concatenate([np.asarray(e).reshape(-1, 3) for t, e in zip(types, element_nodes) if t == 2]) if len(types) else np.zeros((0, 3), int)
            lookup = {int(n): i for i, n in enumerate(node_tags)}
            points = np.asarray(coords).reshape(-1, 3)
            index = np.vectorize(lookup.get)(triangles) if len(triangles) else triangles
            out.append((group_of.get(tag, "(none)"), int(tag), points, index))
    finally:
        gmsh.finalize()
    return out


def _groups_3d_figures(plan) -> None:
    entities = _group_triangles(plan)
    total = sum(len(t) for _, _, _, t in entities)
    print(f"  surface mesh of the reopened XAO: {len(entities)} faces, {total} triangles")
    fig = go.Figure()
    shown: set[str] = set()
    for name, tag, points, tri in entities:
        if not len(tri):
            continue
        fig.add_trace(go.Mesh3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2], color=GROUP_COLOURS[name],
            opacity=GROUP_OPACITY.get(name, 1.0), name=name, legendgroup=name, showlegend=name not in shown, flatshading=True,
            hovertemplate=f"{name}<br>Gmsh face tag {tag}<br>planned id {plan.face_ids[tag - 1]}<extra></extra>"))
        shown.add(name)
    fig.update_layout(width=980, height=620, paper_bgcolor="white", font=dict(color=INK, size=12),
                      scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.45), xaxis_title="x (µm)", yaxis_title="y (µm)",
                                 zaxis_title="z (µm)", camera=dict(eye=dict(x=1.5, y=-1.6, z=1.0))),
                      margin=dict(l=0, r=0, t=10, b=0), legend=dict(itemsizing="constant"))
    fig.write_html(FIGURES / "emission-groups-3d.html", include_plotlyjs="directory", config={"displaylogo": False})
    print("wrote emission-groups-3d.html")

    sfig = plt.figure(figsize=(10.5, 4.6))
    ax = sfig.add_subplot(1, 2, 1, projection="3d")
    for name, _, points, tri in entities:
        if not len(tri):
            continue
        ax.add_collection3d(Poly3DCollection(points[tri], facecolor=GROUP_COLOURS[name], alpha=GROUP_OPACITY.get(name, 1.0),
                                             edgecolor="none", rasterized=True))
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(-100, 100)
    ax.set_box_aspect((1, 1, 0.45))
    ax.view_init(elev=24, azim=-55)
    ax.set_xlabel("x (µm)", fontsize=8)
    ax.set_ylabel("y (µm)", fontsize=8)
    ax.set_zlabel("z (µm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("whole model (z stretched)", fontsize=9)

    plan_ax = sfig.add_subplot(1, 2, 2)  # top view of the horizontal faces, lowest first so the metal top paints last
    horizontal = [(float(points[tri][:, :, 2].mean()), name, points[tri][:, :, :2]) for name, _, points, tri in entities
                  if len(tri) and not name.startswith("EXT") and np.ptp(points[tri][:, :, 2]) < 1e-6]
    for _, name, xy in sorted(horizontal, key=lambda item: item[0]):
        plan_ax.add_collection(PolyCollection(xy, facecolor=GROUP_COLOURS[name], edgecolor="white", linewidth=0.2, rasterized=True))
    plan_ax.set_xlim(-95, 95)
    plan_ax.set_ylim(-72, 72)
    plan_ax.set_aspect("equal")
    plan_ax.set_xlabel("x (µm)", fontsize=8)
    plan_ax.set_ylabel("y (µm)", fontsize=8)
    plan_ax.tick_params(labelsize=7)
    plan_ax.set_title("top view of the cutout (horizontal faces; walls and outer box not drawn)", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=max(GROUP_OPACITY.get(n, 1.0), 0.35)) for n, c in GROUP_COLOURS.items()]
    sfig.legend(handles, GROUP_COLOURS.keys(), fontsize=7.5, loc="lower center", ncol=4, frameon=False)
    sfig.tight_layout(rect=(0, 0.08, 1, 1))
    sfig.savefig(FIGURES / "emission-groups-3d.svg", facecolor="white")
    plt.close(sfig)
    print("wrote emission-groups-3d.svg")


# --------------------------------------------------------------------------- main


def _ensure_plotly_js() -> None:
    target = FIGURES / "plotly.min.js"
    if not target.exists():
        import plotly

        FIGURES.mkdir(parents=True, exist_ok=True)
        target.write_bytes((Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_bytes())


def _occt_tables() -> None:
    """Official OCCT checks over the five running-example fixtures, and the negative controls (outputs/occt/)."""
    rows = []
    for case in OCCT_CASES:
        report = _load_json(OCCT / "reports" / f"{case}.json")
        for fmt in ("brep", "xao"):
            r = report[fmt]
            shape, props = r["shape_analysis"], r["properties"]
            rows.append([f"`{case}`", fmt.upper(), "是" if r["ok"] else "否", "是" if r["brepcheck"]["valid"] else "否",
                         str(r["brepcheck"]["faulty_count"]), "有" if r["bop_argument_analyzer"]["faulty"] else "無",
                         str(r["bop_argument_analyzer"]["result_count"]), str(r["bop_argument_analyzer"]["excluded_free_faces"]),
                         f"{shape['shells_with_free_edges']} / {shape['shells_with_bad_edges']}", f"{shape['shortest_edge_um']:.3g}",
                         f"{props['faces']['max_relative_error']:.2g}", f"{max(x['relative_error'] for x in props['solids']):.2g}"])
    _write("emission-occt-table.md", _table(
        ["case", "格式", "ok", "BRepCheck valid", "faulty 元素數", "BOP 偵測到 fault", "BOP result 數", "BOP 排除的 free face 數",
         "有 free／bad edge 的 shell 數", "最短 edge (µm)", "face 面積最大相對誤差", "solid volume 最大相對誤差"], rows, "llllrlrrlrrr"))
    controls = _load_json(OCCT / "negative_controls.json")
    rows = []
    for defect, c in controls.items():
        def caught(flag, statuses):
            return ("抓到（" + "，".join(statuses) + "）") if flag and statuses else ("抓到" if flag else "未抓到")
        brep_statuses = list(c["BRepCheck_statuses"])
        rows.append([f"`{defect}`", caught(c["BRepCheck_Analyzer"], brep_statuses), caught(c["BOPAlgo_ArgumentAnalyzer"], c["BOP_statuses"]),
                     caught(c["ShapeAnalysis_Shell"], []), caught(c["plan_cross_check_area_volume"], []), f"{c['check_ms']:.1f}"])
    _write("emission-occt-controls-table.md", _table(
        ["注入的缺陷", "BRepCheck_Analyzer", "BOPAlgo_ArgumentAnalyzer", "ShapeAnalysis_Shell", "plan 面積／volume 比對", "全部檢查耗時 (ms)"],
        rows, "lllllr"))


def main() -> None:
    if "--occt-only" in sys.argv:
        _occt_tables()
        return
    _ensure_plotly_js()
    topo, plan = _running_example()
    rows = _timing_rows()
    _timing_figures(rows)
    _timing_tables(rows)
    _counts_table(plan)
    _tags_and_groups_tables(topo, plan)
    _validation_table()
    _occt_tables()
    _sidecar_table()
    _shared_face_figure(topo, plan)
    _groups_3d_figures(plan)


if __name__ == "__main__":
    main()
