# ruff: noqa: PLC0415 -- the PDK and SCGSim are imported where used, after the PDK is activated
"""Markdown tables of a baseline run: stages per layout, fine structure, and the classes of issue 125.

    python tables.py RECORDS_DIR

Owners of the stops are not printed: attributing a stop reads the stage contract of issue 141 against
the evidence, which the experiment issue does in words.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

STAGES = ("plan", "sgb_build", "simulation_drc", "mesh")


def main() -> None:
    """Print the three tables of a records folder."""
    folder = Path(sys.argv[1])
    records = {
        path.stem: json.loads(path.read_text())
        for path in sorted(folder.glob("*.json"))
    }
    layouts = {k: r for k, r in records.items() if k.startswith("layout__")}
    scenes = {k: r for k, r in records.items() if k.startswith("scene__")}
    print(stage_table(layouts))
    print()
    print(fine_table({**layouts, **scenes}))
    print()
    print(class_table({**layouts, **scenes}))


def _cell(stage) -> str:
    if stage is None:
        return "–"
    if stage["result"] == "pass":
        text = f"pass, {stage['seconds']:.1f} s"
    elif stage["result"] in ("timeout", "crashed"):
        text = f"**{stage['result']}**"
    else:
        text = f"**{stage['error_class']}**, {stage['seconds']:.1f} s"
    return text


def _mesh_cell(stage) -> str:
    if stage is None:
        return "–"
    parts = [_cell(stage)]
    if stage.get("tetrahedra") is not None:
        parts.append(
            f"{stage['tetrahedra']:,} tets, minSICN {stage['min_sicn']:.4f}, {stage['bad_count']} bad, "
            f"{stage['rounds']} rounds, stop `{stage['stop']}`, meshing {stage['mesh_seconds']:.0f} s"
        )
    return "; ".join(parts)


def stage_table(layouts) -> str:
    """One row per layout run: each stage's outcome and the mesh summary."""
    rows = [
        "| Layout | Bands | Vacuum | Simulation | Plan | SGB build | simulation_drc | mesh() |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for record in layouts.values():
        stages = record["stages"]
        rows.append(
            f"| {record['case']} | {'yes' if record['bands'] else 'no'} | {_vacuum(record)} | "
            f"{record.get('simulation', '–')} | "
            + " | ".join(
                _cell(stages.get(s)) if s != "mesh" else _mesh_cell(stages.get(s))
                for s in STAGES
            )
            + " |"
        )
    return "\n".join(rows)


def _vacuum(record) -> str:
    return "notebook" if record.get("notebook_vacuum") else "1 um"


def _pair(block, cutoff) -> str:
    return f"{block['source']['count_below'][cutoff]} / {block['band']['count_below'][cutoff]}"


def _smallest(block, scale=1000.0, unit="nm") -> str:
    values = [
        v for v in (block["source"]["smallest"][:1] + block["band"]["smallest"][:1])
    ]
    src = block["source"]["smallest"][:1]
    band = block["band"]["smallest"][:1]
    fmt = lambda v: f"{v * scale:.1f} {unit}" if v is not None else "–"  # noqa: E731
    return (
        f"{fmt(src[0]) if src else '–'} / {fmt(band[0]) if band else '–'}"
        if values
        else "– / –"
    )


def fine_table(records) -> str:
    """Counts and smallest values of the fine structure, source curves / band curves."""
    rows = [
        "| Case | Bands | edges < 25 g (src / band) | shortest edge (src / band) | faces 2A/P < 25 g | smallest 2A/P | "
        "faces LFS < 25 g | wall gaps < 25 g | angles < 10° | smallest angle (src / band) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in records.values():
        fs = (record["stages"].get("sgb_build") or {}).get("fine_structure")
        if fs is None or record.get("notebook_vacuum"):
            continue
        e, f, a = fs["edges"], fs["faces"], fs["angles"]
        angle_smallest = (
            f"{a['source']['smallest'][0] if a['source']['smallest'] else '–'}° / "
            f"{a['band']['smallest'][0] if a['band']['smallest'] else '–'}°"
        )
        rows.append(
            f"| {'scene ' if record.get('scene') else ''}{record['case']} | {'yes' if record['bands'] else 'no'} | "
            f"{_pair(e, '25')} | {_smallest(e)} | {_pair(f['horizontal_2a_over_p'], '25')} | "
            f"{_smallest(f['horizontal_2a_over_p'])} | {_pair(f['horizontal_local_feature_size'], '25')} | "
            f"{_pair(f['wall_column_gap'], '25')} | {_pair(a, '10.0')} | {angle_smallest} |"
        )
    return "\n".join(rows)


def class_table(records) -> str:
    """The classes of issue 125 at 2, 10 and 25 grid steps, beside each mesh outcome."""
    rows = [
        "| Case | Bands | A wall gap < 2 / 10 / 25 g | B 2A/P < 2 / 10 / 25 g | B angle < 1° | C free edge < 2 / 10 / 25 g "
        "| mesh, 1 um vacuum | mesh, notebook vacuum |",
        "|---|---|---|---|---|---|---|---|",
    ]
    notebook = {
        (r["case"], r["bands"]): r for r in records.values() if r.get("notebook_vacuum")
    }
    for record in records.values():
        fs = (record["stages"].get("sgb_build") or {}).get("fine_structure")
        if fs is None or record.get("notebook_vacuum"):
            continue
        c = fs["issue125_classes"]

        def triple(block):
            return " / ".join(
                f"{block[s]['source'] + block[s]['band']}" for s in ("2", "10", "25")
            )

        def mesh_text(rec):
            mesh = None if rec is None else rec["stages"].get("mesh")
            if mesh is None:
                return "–"
            return (
                mesh.get("stop")
                if mesh["result"] == "pass"
                else f"{mesh.get('stop') or ''} {mesh.get('error_class', '')}".strip()
            )

        rows.append(
            f"| {'scene ' if record.get('scene') else ''}{record['case']} | {'yes' if record['bands'] else 'no'} | "
            f"{triple(c['A_wall_gap_below_steps'])} | {triple(c['B_face_2a_over_p_below_steps'])} | "
            f"{c['B_angle_below_1_degree']['source'] + c['B_angle_below_1_degree']['band']} | "
            f"{triple(c['C_free_edge_below_steps'])} | {mesh_text(record)} | "
            f"{mesh_text(notebook.get((record['case'], record['bands'])))} |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
