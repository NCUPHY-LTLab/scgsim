# ruff: noqa: PLC0415 -- the PDK and SCGSim are imported where used, after the PDK is activated
"""Are the drawing artifacts of orcapen-sc-pdk#47 still in the Disk Qubit at this PDK revision?

    python check_pdk47.py OUT.json

#47 measured, at PDK 8d18025: eight 1 nm tunable-coupler (TC) jogs, two per claw polygon, at (+-212.424,
+-87.9885) and (+-169.0415, +-59.289) um and their quarter-turn images (C2/C3); and a 9.9 nm step where the
lead fan meets the disk, at (77.36, -78.78) um and its mirror (C1).  For each Disk Qubit build this lists
(a) every edge of the flattened D1 metal polygons shorter than 25 nm and (b) every edge of the built SGB model
shorter than 25 grid steps whose lineage is source curves only, each with its location, and says which lie
within 3 um of a #47 location.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fine_structure  # noqa: E402
from run_case import configured_simulation, prepare_plan, surface_epr_specs  # noqa: E402

TC_JOGS = [(212.424, 87.9885), (169.0415, 59.289)]
C1_STEP = [(77.36, -78.78)]
NEAR_UM = 3.0
SHORT_UM = 0.025
BUILDS = (
    "disk_qubit_default",
    "disk_qubit_floating",
    "disk_qubit_floating_span0",
    "disk_qubit_grounded",
)


def _images(points):
    """The points under the cell's symmetries: the four quarter turns and the mirror in x = y."""
    out = set()
    for x, y in points:
        for a, b in ((x, y), (y, x)):
            for _ in range(4):
                a, b = -b, a
                out.add((round(a, 4), round(b, 4)))
                out.add((round(a, 4), round(-b, 4)))
    return sorted(out)


def _near(location, targets):
    return [t for t in targets if math.dist(location[:2], t) <= NEAR_UM]


def _polygon_short_edges(component, layer=(2, 0)):
    rows = []
    # gdsfactory keys the result by its layer index, one key for the one layer asked for.
    by_index = component.get_polygons_points(layers=[layer], merge=False)
    for polygon in (polygon for polygons in by_index.values() for polygon in polygons):
        points = [tuple(map(float, p)) for p in polygon]
        for a, b in zip(points, points[1:] + points[:1], strict=True):
            length = math.dist(a, b)
            if length < SHORT_UM:
                rows.append(
                    {
                        "length_nm": round(length * 1000, 2),
                        "at_um": [
                            round((a[0] + b[0]) / 2, 4),
                            round((a[1] + b[1]) / 2, 4),
                        ],
                    }
                )
    return rows


def main() -> None:
    """Measure every Disk Qubit build and write the findings to the file given."""
    import gdsfactory as gf
    import orcapen_sc_pdk
    from cases import LAYOUTS
    from orcapen_sc_pdk import get_material_records

    orcapen_sc_pdk.activate()
    jogs, step = _images(TC_JOGS), _images(C1_STEP)
    result = {}
    for name in BUILDS:
        gf.clear_cache()
        layout = LAYOUTS[name]()
        flat = layout.component.copy()
        flat.flatten()
        polygon_edges = _polygon_short_edges(flat)
        for row in polygon_edges:
            row["near_tc_jog"] = bool(_near(row["at_um"], jogs))
            row["near_c1_step"] = bool(_near(row["at_um"], step))
        snapshot = prepare_plan(layout, get_material_records())
        with TemporaryDirectory() as folder, fine_structure.capture() as captured:
            sim = configured_simulation(layout, snapshot, surface_epr_specs(False))
            # The geometry build mesh() and simulation_drc share; no public call keeps its files.
            sim._build_geometry(folder)  # noqa: SLF001
        report = captured[-1]["report"]
        g = report["grid_um"]
        model_edges = [
            {
                "length_nm": round(row["value_um"] * 1000, 2),
                "at_um": row["location_um"],
                "kind": row["kind"],
                "near_tc_jog": bool(_near(row["location_um"], jogs)),
                "near_c1_step": bool(_near(row["location_um"], step)),
                "sources": row["sources"][:3],
            }
            for row in report["shortest_edges"]
            if row["value_um"] < 25 * g
            and row["kind"] == "horizontal"
            and not any(fine_structure.is_band_source(s) for s in row["sources"])
        ]
        result[name] = {
            "d1_metal_polygon_edges_under_25_nm": polygon_edges,
            "sgb_source_edges_under_25_steps_no_bands": model_edges,
        }
        print(
            name,
            len(polygon_edges),
            "polygon edges <25 nm;",
            len(model_edges),
            "model source edges <25 steps",
            flush=True,
        )
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
