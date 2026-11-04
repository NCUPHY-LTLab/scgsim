"""Surface-EPR inset bands on the pad-in-ground example (phase 2).

Writes under ``outputs/inset/`` and ``outputs/pad_in_ground_{A,B}_bands/``:

* ``klayout_comparison.json`` / ``.md`` -- exact band areas vs a KLayout erosion of the same regions;
* ``palace_check.json`` -- every band is its own physical group; triangles and tets per band, shared on both sides;
* ``topology_change.json`` -- a neck splits and a band vanishes (recorded, nothing fixed);
* ``port_invariance.json`` -- adding the port changes no band boundary.
"""

import json

from _common import OUTPUTS, run_case

from sgbproto.klayout_check import comparison_rows, exact_band_areas
from sgbproto.pipeline import build
from sgbproto.scenes import pad_in_ground

MARGINS = (0.1, 0.5, 2.0)
REFINE = {"refine_families": ("band_pad_top", "band_pad_bottom", "band_gap"), "refine_size": 1.0, "refine_distance": (1.0, 20.0)}


def klayout_table(spec, result) -> list[dict]:
    rows = []
    for fam in spec.families:
        level = result.levels.levels.index(fam.target_levels[0])
        rows += comparison_rows(result, fam.support, fam.family_id, level)
    return rows


def markdown(rows: list[dict]) -> str:
    lines = ["| region | margin (um) | exact area (um^2) | KLayout area (um^2) | difference (um^2) | relative | oracle rounding bound (um^2) | components |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['region']} | {r['margin_units'] / 1000:g} | {r['exact_um2']:.4f} | {r['klayout_um2']:.4f} | "
                     f"{r['difference_um2']:+.5f} | {r['relative']:+.1e} | {r['oracle_rounding_bound_um2']:.3f} | {r['components_klayout']} |")
    return "\n".join(lines)


def main() -> None:
    out = OUTPUTS / "inset"
    out.mkdir(parents=True, exist_ok=True)

    spec_a = pad_in_ground("A", bands_um=MARGINS)
    summary = run_case(spec_a, "pad_in_ground_A_bands", mesh_size=40.0,
                       supports={f.family_id: f.support for f in spec_a.families}, **REFINE)
    result_a = build(spec_a)
    rows = klayout_table(spec_a, result_a)
    (out / "klayout_comparison.json").write_text(json.dumps(rows, indent=1))
    (out / "klayout_comparison.md").write_text(markdown(rows) + "\n")
    print(markdown(rows))

    spec_b = pad_in_ground("B", bands_um=MARGINS)
    run_case(spec_b, "pad_in_ground_B_bands", mesh_size=60.0, mesh_size_min=2.0)
    rows_b = klayout_table(spec_b, build(spec_b))
    (out / "klayout_comparison_route_B.json").write_text(json.dumps(rows_b, indent=1))

    stage = json.loads((OUTPUTS / "pad_in_ground_A_bands" / "stage_summary.json").read_text())
    (out / "palace_check.json").write_text(json.dumps(stage["stages"]["S10_emission"]["mesh"]["bands"], indent=1))

    split = pad_in_ground("A", bands_um=(4.0, 9.0, 30.0), band_regions=("pad_top",))
    res = build(split)
    layer = res.level_layers[1]
    per_band: dict[int, list[float]] = {}
    for fi in range(1, len(layer.arr.faces)):
        for _, k in layer.labels[fi].bands:
            from sgbproto.arcs import face_exact_area
            per_band.setdefault(k, []).append(round(face_exact_area(layer.arr, layer.edge_curve, fi) * 1e-6, 4))
    wide = pad_in_ground("A", bands_um=(4.0, 9.0, 21.0, 23.0), band_regions=("pad_top",))
    wide_result = build(wide)
    wide_rows = comparison_rows(wide_result, wide.families[0].support, "band_pad_top", 1)
    (out / "topology_change.json").write_text(json.dumps({
        "margins_um": [4, 9, 30],
        "faces_per_band_um2": {str(k): v for k, v in sorted(per_band.items())},
        "note": "B_3 (>= 30 um) is empty: R_30 vanished because the pad top has no point 30 um from its boundary "
                "(R_23 still exists, R_30 does not). R_9 splits into 3 components (KLayout) and B_2 into the 3 matching faces.",
        "validation_passed": not res.report.failures,
        "components_of_R_m_klayout": {str(r["margin_units"] / 1000): r["components_klayout"] for r in wide_rows},
        "wide_margins_um": [4, 9, 21, 23],
    }, indent=1))

    base = {k: exact_band_areas(result_a, k, 1) for k in ("band_pad_top", "band_gap")}
    no_port = build(pad_in_ground("A", bands_um=MARGINS, with_port=False))
    alt = {k: exact_band_areas(no_port, k, 1) for k in ("band_pad_top", "band_gap")}
    (out / "port_invariance.json").write_text(json.dumps({
        "families_identical": pad_in_ground("A", bands_um=MARGINS).families == pad_in_ground("A", bands_um=MARGINS, with_port=False).families,
        "max_band_area_difference_grid2": max(abs(base[f][k] - alt[f][k]) for f in base for k in base[f]),
    }, indent=1))
    print(json.dumps(summary["mesh"]))


if __name__ == "__main__":
    main()
