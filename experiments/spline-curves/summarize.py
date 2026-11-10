"""One table row per scene from the JSON files ``measure.py`` wrote.

Usage::

    uv run --locked --no-sync python experiments/spline-curves/summarize.py OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = (
    "scene | pieces (lines / arcs) | fit dev nm | kinks | build s | emit s | short band edges | bands | "
    "tets | minSICN | rounds | stop | mesh s"
)


def row(r: dict) -> str:
    """The table row of one scene's record."""
    label = _label(r)
    if "crash" in r:
        return f"{label} | crash: {r['crash'].splitlines()[-1][:120]}"
    pieces = [v for k, v in r["boundary"].items() if k != "fitting"]
    lines = sum(p["straight_edges"] for p in pieces)
    arcs = sum(p["arc_pieces"] for p in pieces)
    fit = r["boundary"].get("fitting", {})
    build = r["build"]
    if build["status"] != "built":
        return f"{label} | {lines} / {arcs} | | | refused ({build['seconds']} s): {build['error'][:110]}"
    t, b, m = r["topology"], r.get("bands", {}), r.get("mesh", {})
    bands = (
        "not asked"
        if not r.get("bands_requested", True)
        else "ok"
        if b.get("same_families")
        and b.get("worst_excess_um2", 1) <= b.get("witness_allowance_um2", 0)
        else b
    )
    return " | ".join(
        str(x)
        for x in (
            label,
            f"{lines} / {arcs}",
            fit.get("max_deviation_nm", ""),
            fit.get("kinks_over_1e-9_rad", ""),
            build["build_seconds"],
            build["emit_seconds"],
            t["short_band_edges"],
            bands,
            m.get("tetrahedra"),
            None if m.get("min_sicn") is None else round(m["min_sicn"], 4),
            m.get("rounds"),
            m.get("stop"),
            m.get("seconds"),
        )
    )


def _label(r: dict) -> str:
    """The scene, and its mode when it is not the default (bands and a mesh)."""
    mode = [
        flag
        for flag, asked in (
            ("no bands", r.get("bands_requested", True)),
            ("no mesh", r.get("mesh_requested", True)),
        )
        if not asked
    ]
    return f"{r['scene']} ({', '.join(mode)})" if mode else r["scene"]


def main(argv: list[str]) -> int:
    """Print the table of every record in the directory."""
    print(COLUMNS)
    for path in sorted(Path(argv[0]).glob("*.json")):
        print(row(json.loads(path.read_text(encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
