"""Checks and tables of the emission pages: the oracle, the loaded model, the validation defects, the bands and the OCCT controls."""

from __future__ import annotations

import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gmsh
import inset_kit as kit  # noqa: E402  (also puts the production core's tests on sys.path)
import sgb_occt_mutations as mutations  # noqa: E402
from emission_common import bundle, case_models, markdown_table, write_figure_file
from sgb_extreme_cases import build_model
from sgb_extreme_cases import cases as extreme_cases  # noqa: E402
from sgb_gmsh_oracle import (  # noqa: E402
    band_mesh_report,
    check_mesh_conformity,
    emit_with_gmsh_api,
    gmsh_session,
    mesh_coarsely,
    refine_near,
)

from scgsim.geometry.sgb.verification.loaded import apply_sidecar_embeds, verify_loaded  # noqa: E402
from scgsim.geometry.sgb.verification.occt import check_file, check_shape, read_brep  # noqa: E402

# --------------------------------------------------------------------------- loaded checks of the examples


def _analytic_volumes(loaded: dict[str, float]) -> str:
    """The pad's closed-form volumes: a 300 x 300 x 100 um substrate and the vacuum box minus the 0.2 um metal."""
    pad_area = (
        100 * 60 - 20 * 15 - 12 * 12 / 2 - 25 * (1 - math.pi / 4) - math.pi * 64
    )  # the scene's pad: notch, chamfer, rounded corner, hole
    ground_area = 300 * 300 - 160 * 120
    expected = {
        "substrate": 300 * 300 * 100.0,
        "vacuum": 300 * 300 * 100.0 - 0.2 * (ground_area + pad_area),
    }
    for name, want in expected.items():
        assert abs(loaded[name] - want) <= 1e-12 * want, (name, loaded[name], want)
    return "；".join(
        f"{k} {loaded[k]:.9g}，解析值 {v:.9g} µm³" for k, v in expected.items()
    )


def _oracle_agrees(model, plan, sidecar) -> dict:
    """The Gmsh-API oracle (an independent bottom-up replay) against the plan: entity counts, group members and the face areas.

    The oracle builds a curved wall with Gmsh's own surface construction, not as the exact cylinder the writer prints (the test side allows 1e-3
    for it), so the area error of the curved walls is reported separately from the planar faces.
    """

    with gmsh_session():
        oracle = emit_with_gmsh_api(model)
        counts = {d: len(gmsh.model.getEntities(d)) for d in range(4)}
        inverse = {tag: pid for pid, tag in oracle.tags.items()}
        groups = {
            name: sorted(inverse[(dim, t)] for t in tags)
            for name, (dim, tags) in oracle.groups.items()
        }
        areas = {
            pid: gmsh.model.occ.getMass(2, tag)
            for pid, (dim, tag) in oracle.tags.items()
            if dim == 2
        }
    planned = plan.counts
    same_counts = counts == {
        0: planned["vertices"],
        1: planned["edges"],
        2: planned["faces"],
        3: planned["solids"],
    }
    same_groups = groups == {
        name: sorted(g.ids) for name, g in model.routed.groups.items() if name in groups
    } and set(groups) == set(model.routed.groups)
    want = {
        pid: sidecar["expected"]["face_area_um2"][i]
        for i, pid in enumerate(sidecar["binding"]["ids_by_tag"]["2"])
    }
    worst = {"planar": 0.0, "curved": 0.0}
    for pid, area in want.items():
        kind = "curved" if model.topology.faces[pid].curved else "planar"
        worst[kind] = max(worst[kind], abs(areas[pid] - area) / area)
    return {
        "entities_and_groups": same_counts and same_groups,
        "planar_area_error": worst["planar"],
        "curved_area_error": worst["curved"],
        "curved_faces": sum(1 for pid in want if model.topology.faces[pid].curved),
    }


def _loaded_check(model, folder: Path, mesh_size: float) -> dict:
    """Write a model, load it in a fresh Gmsh session and run the mesher's checks and a coarse mesh."""

    artifacts, plan, sidecar = bundle(model, folder)
    with gmsh_session(artifacts.xao):
        report = verify_loaded(sidecar)
        masses = {
            model.topology.volumes[vid].domain.name: gmsh.model.occ.getMass(3, i + 1)
            for i, vid in enumerate(plan.solid_ids)
        }
        apply_sidecar_embeds(sidecar)
        if model.insets:
            refine_near(
                model,
                plan.tags(),
                ("band_pad_top", "band_pad_bottom", "band_gap"),
                mesh_size,
                1.0,
                (1.0, 20.0),
            )
        mesh_coarsely(mesh_size)
        conformity = check_mesh_conformity(model, plan.tags())
        bands = (
            band_mesh_report(
                model,
                plan.tags(),
                {
                    f.family_id: f.support
                    for f in model.labelled.catalogue.families
                    if f.support
                },
            )
            if model.insets
            else {}
        )
    return {
        "report": report,
        "masses": masses,
        "conformity": conformity,
        "bands": bands,
        "plan": plan,
        "sidecar": sidecar,
    }


def validation_table() -> None:
    """The table of what the load-side checks, the Gmsh-API oracle and the mesher report for the example models."""
    cases = ("pad_in_ground_A", "pad_in_ground_B", "airbridge_B")
    sizes = {"pad_in_ground_A": 40.0, "pad_in_ground_B": 60.0, "airbridge_B": 60.0}
    builders = case_models()
    results = {}
    for case in cases:
        model = builders[case]()
        with tempfile.TemporaryDirectory(prefix="emission-validation-") as scratch:
            checked = _loaded_check(model, Path(scratch), sizes[case])
            results[case] = {
                **checked,
                "oracle": _oracle_agrees(model, checked["plan"], checked["sidecar"]),
            }

    def counts(case: str) -> str:
        n = results[case]["report"].counts["native"]
        return " / ".join(str(n[str(d)]) for d in range(4))

    def mismatches(case: str) -> str:
        m = results[case]["report"].boundaries_mismatching
        return f"{m['edges']} / {m['faces']} / {m['solids']}"

    def groups(case: str) -> str:
        g = results[case]["report"].groups
        return f"{g['loaded']} of {g['planned']}"

    def mesh(case: str) -> str:
        c = results[case]["conformity"]
        return f"{c['nodes']} nodes，{c['tets']} tets，{c['duplicate_nodes']} 個重複 node，conformal：{'是' if c['conformal'] else '否'}"

    analytic = _analytic_volumes(results["pad_in_ground_B"]["masses"])
    rows = [
        [
            "Gmsh 中的元素數（vertex / edge / face / solid），與規劃相符",
            *(counts(c) for c in cases),
        ],
        [
            "以 `getBoundary` 檢查的邊界不一致數（edge / face / solid）",
            *(mismatches(c) for c in cases),
        ],
        [
            "vertex 與規劃位置的最大偏差 (µm)",
            *(f"{results[c]['report'].max_vertex_offset_um:g}" for c in cases),
        ],
        [
            "從 XAO 載入的 physical group 數（BREP 不含 group）",
            *(groups(c) for c in cases),
        ],
        [
            "volume 對規劃的最大相對誤差（XAO）",
            *(f"{results[c]['report'].max_relative_volume_error:.2g}" for c in cases),
        ],
        ["載入後 XAO 的粗略 3-D mesh", *(mesh(c) for c in cases)],
        [
            "face 面積對規劃的最大相對誤差",
            *(
                f"{results[c]['report'].max_relative_face_area_error:.2g}"
                for c in cases
            ),
        ],
        [
            "與 Gmsh-API oracle 的 entity 數與 group 成員相同",
            *(
                "是" if results[c]["oracle"]["entities_and_groups"] else "否"
                for c in cases
            ),
        ],
        [
            "與 oracle 的 face 面積最大相對差（平面 face；curved wall）",
            *(
                f"{results[c]['oracle']['planar_area_error']:.2g}；{results[c]['oracle']['curved_area_error']:.2g}（{results[c]['oracle']['curved_faces']} 面）"
                if results[c]["oracle"]["curved_faces"]
                else f"{results[c]['oracle']['planar_area_error']:.2g}；無 curved wall"
                for c in cases
            ),
        ],
        ["解析 volume（封閉形式）", "未計算", analytic, "未計算"],
        [
            "`verify_loaded` 是否通過",
            *("是" if results[c]["report"].ok else "否" for c in cases),
        ],
    ]
    write_figure_file(
        "emission-validation-table.md",
        markdown_table(["載入後的檢查", *cases], rows, "llll"),
    )
    _bands_table()


def _bands_table() -> None:
    rows = []
    builders = case_models()
    for case, size in (
        ("pad_in_ground_A_bands", 40.0),
        ("pad_in_ground_B_bands", 60.0),
        ("pad_in_ground_C_bands", 60.0),
    ):
        model = builders[case]()
        with tempfile.TemporaryDirectory(prefix="emission-bands-") as scratch:
            r = _loaded_check(model, Path(scratch), size)
        report, conf, bands = r["report"], r["conformity"], r["bands"]
        passing = sum(
            1
            for b in bands.values()
            if b["all_shared_by_tets"] and b["centroids_in_band"]
        )
        rows.append(
            [
                f"`{case}`",
                f"{report.groups['loaded']} of {report.groups['planned']}",
                str(len(bands)),
                f"{report.max_relative_face_area_error:.2g}",
                f"{report.max_relative_volume_error:.2g}",
                f"{conf['nodes']:,} nodes，{conf['tets']:,} tets，conformal：{'是' if conf['conformal'] else '否'}",
                f"{passing} of {len(bands)}",
            ]
        )
    write_figure_file(
        "emission-bands-table.md",
        markdown_table(
            [
                "case",
                "已載入 groups",
                "band group 數",
                "face 面積最大誤差",
                "volume 最大誤差",
                "粗略 3-D mesh",
                "triangle 全為 tet face 且落在 band 內的 band group 數",
            ],
            rows,
            "lrrrrll",
        ),
    )


# --------------------------------------------------------------------------- OCCT checks and the catalogue


def occt_tables() -> dict:
    """Official OCCT checks over the seven running-example models (BREP and XAO), and the negative controls."""
    rows = []
    seen = {
        "brep_check_ms": [],
        "bop_ms": [],
        "total_ms": [],
        "reports": 0,
        "all_ok": True,
        "max_face_error": 0.0,
        "max_volume_error": 0.0,
        "excluded_free_faces": 0,
    }
    for case, build in case_models().items():
        model = build()
        with tempfile.TemporaryDirectory(prefix="emission-occt-") as scratch:
            artifacts, plan, _ = bundle(model, Path(scratch))
            for fmt, path in (("brep", artifacts.brep), ("xao", artifacts.xao)):
                r = check_file(path, model.topology, plan).to_dict()
                shape, props = r["shape_analysis"], r["properties"]
                seen["brep_check_ms"].append(r["timings_s"]["brep_check"] * 1000)
                seen["bop_ms"].append(r["timings_s"]["bop_argument_analyzer"] * 1000)
                seen["total_ms"].append(sum(r["timings_s"].values()) * 1000)
                seen["reports"] += 1
                seen["all_ok"] &= r["ok"]
                seen["max_face_error"] = max(
                    seen["max_face_error"], props["faces"]["max_relative_error"]
                )
                seen["max_volume_error"] = max(
                    seen["max_volume_error"],
                    max(x["relative_error"] for x in props["solids"]),
                )
                seen["excluded_free_faces"] += r["bop_argument_analyzer"][
                    "excluded_free_faces"
                ]
                rows.append(
                    [
                        f"`{case}`",
                        fmt.upper(),
                        "是" if r["ok"] else "否",
                        "是" if r["brepcheck"]["valid"] else "否",
                        str(r["brepcheck"]["faulty_count"]),
                        "有" if r["bop_argument_analyzer"]["faulty"] else "無",
                        str(r["bop_argument_analyzer"]["result_count"]),
                        str(r["bop_argument_analyzer"]["excluded_free_faces"]),
                        f"{shape['shells_with_free_edges']} / {shape['shells_with_bad_edges']}",
                        f"{shape['shortest_edge_um']:.3g}",
                        f"{props['faces']['max_relative_error']:.2g}",
                        f"{max(x['relative_error'] for x in props['solids']):.2g}",
                    ]
                )
    write_figure_file(
        "emission-occt-table.md",
        markdown_table(
            [
                "case",
                "格式",
                "ok",
                "BRepCheck valid",
                "faulty 元素數",
                "BOP 偵測到 fault",
                "BOP result 數",
                "BOP 排除的 free face 數",
                "有 free／bad edge 的 shell 數",
                "最短 edge (µm)",
                "face 面積最大相對誤差",
                "solid volume 最大相對誤差",
            ],
            rows,
            "llllrlrrlrrr",
        ),
    )
    model = kit.pad_model("B", None)
    with tempfile.TemporaryDirectory(prefix="emission-controls-") as scratch:
        artifacts, plan, _ = bundle(model, Path(scratch))
        rows = []
        for defect, mutate in mutations.CONTROLS.items():
            started = time.perf_counter()
            report = check_shape(
                mutate(read_brep(artifacts.brep)), model.topology, plan, source=defect
            )
            millis = (time.perf_counter() - started) * 1000
            d = report.to_dict()
            statuses = sorted(d["brepcheck"]["histogram"])
            bop = sorted({x["status"] for x in d["bop_argument_analyzer"]["results"]})
            seen_shape = bool(
                d["shape_analysis"]["shells_with_free_edges"]
                or d["shape_analysis"]["shells_with_bad_edges"]
            )

            def caught(flag: bool, labels: list[str]) -> str:
                return (
                    ("抓到（" + "，".join(labels) + "）")
                    if flag and labels
                    else ("抓到" if flag else "未抓到")
                )

            rows.append(
                [
                    f"`{defect}`",
                    caught(not d["brepcheck"]["valid"], statuses),
                    caught(d["bop_argument_analyzer"]["faulty"], bop),
                    caught(seen_shape, []),
                    caught(bool(d["properties"]["mismatches"]), []),
                    f"{millis:.1f}",
                ]
            )
    write_figure_file(
        "emission-occt-controls-table.md",
        markdown_table(
            [
                "注入的缺陷",
                "BRepCheck_Analyzer",
                "BOPAlgo_ArgumentAnalyzer",
                "ShapeAnalysis_Shell",
                "plan 面積／volume 比對",
                "全部檢查耗時 (ms)",
            ],
            rows,
            "lllllr",
        ),
    )
    return {
        key: ([min(v), max(v)] if isinstance(v, list) else v) for key, v in seen.items()
    }


def extreme_tables() -> dict:
    """The extreme-case catalogue under every route: build, write, load, coarse mesh and the OCCT checks, with timings."""

    results = []
    for fixture, route in extreme_cases():
        row = {"fixture": fixture.name, "route": route}
        expected = fixture.expect_error.get(route)
        started = time.perf_counter()
        try:
            model = build_model(fixture, route)
        except Exception as error:  # noqa: BLE001 -- a refusal is a measurement
            row["refused"] = type(error).__name__
            row["refusal_ms"] = (time.perf_counter() - started) * 1000
            row["expected"] = expected is not None and isinstance(error, expected)
            results.append(row)
            continue
        assert expected is None, f"{fixture.name} {route} was expected to be refused"
        with tempfile.TemporaryDirectory(prefix="emission-extreme-") as scratch:
            artifacts, plan, sidecar = bundle(model, Path(scratch))
            with gmsh_session(artifacts.xao):
                row["load_ok"] = verify_loaded(sidecar).ok
                apply_sidecar_embeds(sidecar)
                mesh_coarsely(4.0)
                row["mesh_conformal"] = check_mesh_conformity(model, plan.tags())[
                    "conformal"
                ]
            report = check_file(
                artifacts.xao, model.topology, plan, self_interference=True
            )
            t = report.timings
            row.update(
                {
                    "occt_ok": report.ok,
                    "brep_check_ms": t["brep_check"] * 1000,
                    "bop_ms": t["bop_argument_analyzer"] * 1000,
                    "shape_plan_ms": (
                        t["shape_analysis"] + t.get("plan_properties", 0.0)
                    )
                    * 1000,
                    "occt_total_ms": sum(t.values()) * 1000,
                }
            )
        results.append(row)
    fixtures = {r["fixture"] for r in results}
    passed = [
        r
        for r in results
        if "refused" not in r and r["load_ok"] and r["mesh_conformal"] and r["occt_ok"]
    ]
    built = [r for r in results if "refused" not in r]
    refused = [r for r in results if "refused" in r]
    assert len(passed) == len(built) and all(r["expected"] for r in refused)
    count = lambda kind: sum(r["refused"] == kind for r in refused)  # noqa: E731
    rows = [
        [
            "通過全部檢查",
            str(len(passed)),
            "build 與結構檢查、寫出 BREP 與 XAO、Gmsh 載入與 `verify_loaded`、conformal mesh、`BRepCheck_Analyzer`、`BOPAlgo_ArgumentAnalyzer`（全部模式，含 self-interference）、`ShapeAnalysis`、與 plan 的比對",
        ],
        [
            "預期的拒絕：build 階段",
            str(len(refused)),
            "、".join(
                f"`{k}` {count(k)} 個" for k in sorted({r["refused"] for r in refused})
            )
            + "；在 build 開始後數毫秒內丟出",
        ],
    ]
    write_figure_file(
        "emission-extreme-table.md",
        markdown_table(
            [
                "結果",
                f"組合數（共 {len(results)} 個，{len(fixtures)} 個 fixture）",
                "內容",
            ],
            rows,
            "lrl",
        ),
    )
    spans = [
        ("`BRepCheck_Analyzer`", "brep_check_ms"),
        ("`BOPAlgo_ArgumentAnalyzer`（全部模式）", "bop_ms"),
        ("`ShapeAnalysis` 加 plan 比對", "shape_plan_ms"),
        ("OCCT 檢查合計（含讀檔）", "occt_total_ms"),
    ]
    trows = [
        [
            name,
            f"{min(r[key] for r in passed):.1f}",
            f"{max(r[key] for r in passed):.1f}",
        ]
        for name, key in spans
    ]
    trows.append(
        [
            "build 階段的拒絕（丟出錯誤所需時間）",
            f"{min(r['refusal_ms'] for r in refused):.1f}",
            f"{max(r['refusal_ms'] for r in refused):.1f}",
        ]
    )
    write_figure_file(
        "emission-extreme-timing-table.md",
        markdown_table(["檢查", "最短 (ms)", "最長 (ms)"], trows),
    )
    return {
        "combinations": len(results),
        "fixtures": len(fixtures),
        "passed": len(passed),
        "refused": {k: count(k) for k in sorted({r["refused"] for r in refused})},
        "occt_ms": {
            key: [min(r[key] for r in passed), max(r[key] for r in passed)]
            for _, key in spans
        },
        "refusal_ms": [
            min(r["refusal_ms"] for r in refused),
            max(r["refusal_ms"] for r in refused),
        ],
    }
