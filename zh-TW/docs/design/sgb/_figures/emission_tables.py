"""Tables of the emission pages: counts, tags and groups, timings and the validator."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emission_common import markdown_table, seconds_text, write_figure_file
from emission_measure import shared_rows


def counts_table(model, plan) -> None:
    """The table of entity counts of the plan: what each count is and where it comes from."""
    c = plan.counts
    rows = [
        ["vertex", str(c["vertices"]), "`Ve`", "edge"],
        ["edge", str(c["edges"]), "`Ed`", "wire"],
        ["wire（迴圈）", str(c["wires"]), "`Wi`", "face"],
        [
            "face",
            f"{c['faces']}（其中 {c['free_faces']} 個為 free face）",
            "`Fa`",
            "shell",
        ],
        ["shell", str(c["shells"]), "`Sh`", "solid"],
        ["solid", str(c["solids"]), "`So`", "compound"],
        ["compound", "1", "`Co`", "無（根節點）"],
    ]
    write_figure_file(
        "emission-counts-table.md",
        markdown_table(
            ["元素", "pad-in-ground（Route B）中的數量", "record", "被誰引用"],
            rows,
            "lrll",
        ),
    )


def _tag_rows(topo, plan, sidecar) -> list[list[str]]:
    """A hand-picked but fully computed selection of planned ids with their native tags and XAO references."""
    tags = plan.tags()
    found = plan.found
    group_of = {}
    for g in plan.groups:
        for i in g.elements:
            group_of[(g.dim, i)] = (g.name, g.tag)
    refs = {
        3: found.solid_ref,
        2: found.face_ref,
        1: found.edge_ref,
        0: found.vertex_ref,
    }
    picks = []  # (what, planned id, dim, index)
    for i, vid in enumerate(plan.solid_ids):
        picks.append((f"solid，{topo.volumes[vid].domain.name}", vid, 3, i))
    first_face_of = {}
    for i in range(len(plan.face_ids)):
        first_face_of.setdefault(group_of[(2, i)][0], i)
    for name in (
        "SA::substrate|vacuum__interface",
        "MS::pad__bottom",
        "MA::pad__top",
        "PORT::p1",
        "MA::ground__top",
    ):
        picks.append(
            (
                f"face，{name} 的第一個",
                plan.face_ids[first_face_of[name]],
                2,
                first_face_of[name],
            )
        )
    curved = next(
        i
        for i, fid in enumerate(plan.face_ids)
        if topo.faces[fid].curved and found.face_flipped[i]
    )
    picks.append(
        ("face，curved wall (cylinder)，wire 已反向", plan.face_ids[curved], 2, curved)
    )
    picks.append(("edge，第一個寫出", plan.edge_ids[0], 1, 0))
    picks.append(("vertex，第一個寫出", plan.vertex_ids[0], 0, 0))
    rows = []
    for what, pid, dim, index in picks:
        assert tags[pid] == (dim, index + 1)
        group = group_of.get((dim, index)) if dim >= 2 else None
        rows.append(
            [
                f"`{pid}`",
                what,
                f"{group[0]} (tag {group[1]})" if group else "無",
                f"({dim}, {index + 1})",
                str(index),
                str(refs[dim][index]),
            ]
        )
    return rows


def tags_and_groups_tables(topo, plan, sidecar) -> None:
    """The tables of the tag numbering and of the physical groups, read from the plan and the written sidecar."""
    write_figure_file(
        "emission-tags-table.md",
        markdown_table(
            [
                "planned id",
                "元素",
                "primary physical group",
                "Gmsh (dim, tag)",
                "XAO index",
                "XAO reference",
            ],
            _tag_rows(topo, plan, sidecar),
            "llllrr",
        ),
    )
    rows, covered = [], 0
    for g in plan.groups:
        kind = "face" if g.dim == 2 else "solid"
        rows.append([g.name, kind, str(g.tag), str(len(g.elements))])
        covered += len(g.elements) if kind == "face" else 0
    assert covered == len(plan.face_ids), (
        "physical groups do not partition the emitted faces"
    )
    assert len({i for g in plan.groups if g.dim == 2 for i in g.elements}) == covered
    write_figure_file(
        "emission-groups-table.md",
        markdown_table(
            ["primary group", "維度", "physical tag", "成員數"], rows, "llrr"
        ),
    )
    print(
        f"  face groups cover {covered} of {len(plan.face_ids)} emitted faces, each exactly once"
    )


def timing_tables(head: list[dict], validator: list[dict]) -> dict:
    """The head-to-head tables (this script's run) and the direct-writer tail (``why-scaling-measured.json``)."""
    shared = shared_rows()
    api_rows, load_rows, head_numbers = _head_to_head_rows(head)
    load_rows += _shared_load_rows(shared)
    direct_rows, sidecar_rows, tail_numbers = _direct_writer_rows(shared)
    _write_timing_tables(api_rows, load_rows, direct_rows, sidecar_rows)
    return {"head": head_numbers, "tail": tail_numbers}


def _head_to_head_rows(head: list[dict]):
    """The Gmsh-API oracle against the direct writer: table rows and the numbers the page quotes, by N."""
    api_rows, load_rows = [], []
    head_numbers: dict = {}
    for r in head:
        api, d, ld = r["api"], r["direct"], r["load"]
        direct_total = d["plan_s"] + d["render_brep_s"] + d["render_xao_s"]
        api_rows.append(
            [
                f"{r['n']:,}",
                f"{d['plan_counts']['faces']:,}",
                f"{r['groups']:,}",
                seconds_text(api["occ_build_s"]),
                seconds_text(api["synchronize_s"]),
                seconds_text(api["physical_groups_s"]),
                seconds_text(api["write_xao_s"]),
                seconds_text(api["total_s"]),
                seconds_text(direct_total),
                f"{api['total_s'] / direct_total:.1f}x",
            ]
        )
        load_rows.append(
            [
                f"{r['n']:,}",
                f"{api['xao_bytes'] / 1e6:.1f} / {d['xao_bytes'] / 1e6:.1f} / {d['brep_bytes'] / 1e6:.1f}",
                seconds_text(ld["api_xao_s"]),
                seconds_text(ld["direct_xao_s"]),
                seconds_text(ld["direct_brep_s"]),
            ]
        )
        head_numbers[r["n"]] = {
            "api_total_s": api["total_s"],
            "direct_total_s": direct_total,
            "ratio": api["total_s"] / direct_total,
            "groups_share": api["physical_groups_s"] / api["total_s"],
            "groups": r["groups"],
            "faces": d["plan_counts"]["faces"],
            "load_api_s": ld["api_xao_s"],
            "load_direct_xao_s": ld["direct_xao_s"],
            "load_direct_brep_s": ld["direct_brep_s"],
        }
    return api_rows, load_rows, head_numbers


def _shared_load_rows(shared: dict) -> list:
    """Load rows of the sizes where only the direct writer ran (the API replay did not finish)."""
    load_rows = []
    for n in (5000, 20000):
        e = shared[n]["emission"]
        load_rows.append(
            [
                f"{n:,}",
                f"— / {e['file_bytes']['xao'] / 1e6:.1f} / {e['file_bytes']['brep'] / 1e6:.1f}",
                "未執行",
                seconds_text(e["gmsh_load_s"]),
                "未量測",
            ]
        )
    return load_rows


def _direct_writer_rows(shared: dict):
    """The direct writer and the sidecar tail, by N."""
    direct_rows, sidecar_rows = [], []
    tail_numbers: dict = {}
    for n, row in sorted(shared.items()):
        e = row["emission"]
        one = e["direct"]
        total = one["plan"] + one["write_brep"] + one["write_xao"]
        entities = (
            e["written_counts"]["vertices"]
            + e["written_counts"]["edges"]
            + e["written_counts"]["faces"]
        )
        direct_rows.append(
            [
                f"{n:,}",
                f"{e['written_counts']['faces']:,}",
                seconds_text(one["plan"]),
                seconds_text(one["write_brep"] + one["write_xao"]),
                seconds_text(total),
            ]
        )
        sidecar_rows.append(
            [
                f"{n:,}",
                seconds_text(one["reports"]),
                seconds_text(one["sidecar"]),
                f"{e['file_bytes']['sidecar'] / 1e6:.1f}",
                f"{e['file_bytes']['xao'] / 1e6:.1f}",
                f"{e['file_bytes']['brep'] / 1e6:.1f}",
                seconds_text(e["gmsh_load_s"]),
            ]
        )
        tail_numbers[n] = {
            "write_s": total,
            "plan_s": one["plan"],
            "plan_share": one["plan"] / total,
            "render_s": one["write_brep"] + one["write_xao"],
            "reports_s": one["reports"],
            "sidecar_s": one["sidecar"],
            "gmsh_load_s": e["gmsh_load_s"],
            "verify_loaded_s": e["verify_loaded_s"],
            "file_mb": {k: v / 1e6 for k, v in e["file_bytes"].items()},
            "build_s": shared[n]["build_wall_s"],
            "per_entity_us": total / entities * 1e6,
            "written_counts": e["written_counts"],
        }
    return direct_rows, sidecar_rows, tail_numbers


def _write_timing_tables(api_rows, load_rows, direct_rows, sidecar_rows) -> None:
    """Write the four timing tables the emission page includes."""
    write_figure_file(
        "emission-timing-api-table.md",
        markdown_table(
            [
                "N",
                "face 數",
                "physical group 數",
                "OCC build (s)",
                "synchronize (s)",
                "physical groups (s)",
                "寫出 XAO (s)",
                "Gmsh-API 合計 (s)",
                "direct writer：plan 加算繪 (s)",
                "Gmsh-API 是 direct 的幾倍",
            ],
            api_rows,
        ),
    )
    write_figure_file(
        "emission-timing-load-table.md",
        markdown_table(
            [
                "N",
                "檔案大小：API XAO / 直接 XAO / 直接 BREP (MB)",
                "載入 API XAO (s)",
                "載入直接 XAO (s)",
                "載入直接 BREP (s)",
            ],
            load_rows,
        ),
    )
    write_figure_file(
        "emission-timing-direct-table.md",
        markdown_table(
            [
                "N",
                "寫出的 face 數",
                "plan (s)",
                "算繪 (s)",
                "合計 (s)",
            ],
            direct_rows,
        ),
    )
    write_figure_file(
        "emission-sidecar-table.md",
        markdown_table(
            [
                "N",
                "sliver 與 validation report (s)",
                "sidecar (s)",
                "sidecar (MB)",
                "XAO (MB)",
                "BREP (MB)",
                "Gmsh 載入 XAO (s)",
            ],
            sidecar_rows,
        ),
    )


VALIDATOR_ABOUT = {  # what each timed check or load-side phase loops over (checks of validation/ and verification/loaded.py)
    "face_areas": "載入側：每個 face 一次 gmsh mass 呼叫，O(F)",
    "planarity": "S9：各 arrangement 的 edge，用 noding kernel",
    "shells": "S9：每個 shell 的 face",
    "boundaries": "載入側：每個 edge、face、solid 一次 getBoundary，O(E + F)",
    "volumes": "載入側：每個 solid 一次 mass 積分，O(solid)",
    "counts_and_vertex_placement": "載入側：每個 vertex 一次 gmsh 呼叫，O(V)",
    "physical_groups": "載入側：每個 entity 問一次 group，O(entity)",
    "edges_in_faces": "S9：每個 face 的 edge 與 vertical edge，O(E + F)",
    "atomic_ids": "S9：各 arrangement 的 vertex、edge、face id 的唯一性，O(V + E + F)",
    "groups": "S9：每個 face 與 volume 的 group 歸屬，O(F)",
    "grid_and_lengths": "S9：每條 edge 的長度與 vertex 的 grid 位置，O(E)",
    "face_sides": "S9：每個 face 的兩側，O(F)",
    "areas": "S9：各 arrangement 的面積守恆，O(E)",
    "prism_areas": "S9：prism 面積在 cell 上守恆，O(E)",
    "arcs": "S9：arc edge 互不相交與偏差，O(arc edge)",
    "cylinders": "S9：curved wall 是單一 cylinder，O(wall)",
    "bands": "S9：band 面積單調，O(含 band 的 level 的 face)",
}


def validator_tables(validator: list[dict]) -> dict:
    """The tables of the time of every check, structural and load-side, against the number of faces."""
    raw = {r["n"]: r for r in validator}
    ns = sorted(raw)
    faces = {n: raw[n]["counts"]["faces"] for n in ns}
    seconds: dict[str, dict[int, float]] = {}
    for n in ns:
        for name, value in raw[n]["validator_s"].items():
            seconds.setdefault(name, {})[n] = value
        for name, value in raw[n]["verify_phases_s"].items():
            seconds.setdefault(name, {})[n] = value
        for name in ("contact_check", "manifold_check"):
            seconds.setdefault(name, {})[n] = raw[n]["stage_s"][name]
    about = {
        **VALIDATOR_ABOUT,
        "contact_check": "S5：different-Net 接觸，每個 vertex 與 face，O(V + F)",
        "manifold_check": "S5：non-manifold pinch，O(E)",
    }
    missing = sorted(set(seconds) - set(about))
    assert not missing, f"checks without a description: {missing}"
    order = sorted(seconds, key=lambda k: -seconds[k][ns[-1]])
    table, slopes = [], {}
    for name in order:
        t = seconds[name]
        per = [t[n] / faces[n] * 1e6 for n in ns]
        slope = (
            math.log(t[ns[-1]] / t[ns[0]]) / math.log(faces[ns[-1]] / faces[ns[0]])
            if t[ns[0]] > 0 and t[ns[-1]] > 0
            else float("nan")
        )
        slopes[name] = slope
        table.append(
            [
                f"`{name}`",
                about[name],
                *[f"{t[n]:.3f} s" for n in ns],
                f"{per[0]:.2f} -> {per[-1]:.2f}",
                f"{slope:.2f}" if slope == slope else "n/a",
            ]
        )
    header = [
        "檢查",
        "做什麼",
        *[f"N = {n:,} (F = {faces[n]:,})" for n in ns],
        "每個 face 的 µs（首項到末項）",
        "對 F 的 slope",
    ]
    write_figure_file(
        "emission-validator-table.md", markdown_table(header, table, "llrrrrr")
    )
    run_rows = []
    for n in ns:
        r = raw[n]
        structural = (
            sum(r["validator_s"].values())
            + r["stage_s"]["contact_check"]
            + r["stage_s"]["manifold_check"]
        )
        run_rows.append(
            [
                f"{n:,}",
                f"{r['counts']['faces']:,}",
                f"{r['written']['faces']:,}",
                f"{structural:.2f} s",
                f"{r['verify_s']:.2f} s",
                f"{r['peak_rss_mb']:,.0f} MB",
            ]
        )
    write_figure_file(
        "emission-validator-run-table.md",
        markdown_table(
            [
                "N",
                "規劃的 face 數",
                "寫出的 face 數",
                "各項結構檢查的時間相加（S5 與 S9）",
                "`verify_loaded`",
                "整個 process 的 peak RSS",
            ],
            run_rows,
        ),
    )
    big = ns[-1]
    heavy = [
        k for k in order if seconds[k][big] >= 0.1 and slopes[k] == slopes[k]
    ]  # the checks that cost 0.1 s or more at the largest N
    return {
        "slope_range_of_checks_over_0.1s": [
            min(slopes[k] for k in heavy),
            max(slopes[k] for k in heavy),
        ],
        "per_check_at_largest": {k: seconds[k][big] for k in order},
        "verify_total_s": {n: raw[n]["verify_s"] for n in ns},
        "structural_total_s": {
            n: sum(raw[n]["validator_s"].values())
            + raw[n]["stage_s"]["contact_check"]
            + raw[n]["stage_s"]["manifold_check"]
            for n in ns
        },
        "slopes": slopes,
    }
