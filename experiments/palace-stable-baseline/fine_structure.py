# ruff: noqa: PLC0415 -- the PDK and SCGSim are imported where used, after the PDK is activated
"""Fine structure of a built SGB model, read from SGB's own sliver report.

SGB writes the ten worst entries of each sliver measure (``*.slivers.json``).  Counting needs every entry, so
:func:`capture` wraps the report function SGB's emission calls: it asks for every entry, keeps the full
report and the facts the classes of issue 125 need (which edges bound a wall, each wall's column gap), and
hands SGB the same ten-entry report it would have written.  Nothing SGB builds or writes changes.

Attribution follows issue 141: an entry whose lineage names a band curve (a carrier id with ``@m``, the margin
of an inset offset) involves a band; an entry whose lineage names source curves only comes from the layout.

SGB's report gives every edge as the distance between its end points.  An arc edge, and the column gap of a
curved wall, is measured here along its arc instead (``2 r asin(chord / 2r)``, as
``tests/sgb_supported_kit.py`` measures a band edge): the chord is shorter, and would count an arc of 25 grid
steps or more as short.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

# Counting cut-offs.  They are measurement bins, not proposed limits.
EDGE_STEPS = (2, 10, 25)
FACE_STEPS = (2, 10, 25)
ANGLE_DEGREES = (1.0, 2.8125, 10.0)
WORST = 5


def is_band_source(source: str) -> bool:
    """Whether a lineage id is a band curve: an inset offset carrier is named ``<family>@m<margin>``."""
    return "@m" in source


@contextmanager
def capture():
    """Collect the full sliver measurements of every model SGB emits inside the block."""
    import scgsim.geometry.sgb.build as sgb_build

    original = sgb_build.sliver_report
    captured: list[dict[str, Any]] = []

    def full_then_written(
        labelled, topology, routed, top: int = 10, *, drift: bool = True
    ):
        full = original(labelled, topology, routed, top=10**9, drift=drift)
        model = SimpleNamespace(labelled=labelled, topology=topology)
        captured.append({"report": full, "facts": _topology_facts(model)})
        return _as_written(full, top)

    sgb_build.sliver_report = full_then_written
    try:
        yield captured
    finally:
        sgb_build.sliver_report = original


def capture_model(model) -> dict[str, Any]:
    """The same measurements for a model built in this process (the test scenes)."""
    from scgsim.geometry.sgb.checks.slivers.report import sliver_report

    return {
        "report": sliver_report(
            model.labelled, model.topology, model.routed, top=10**9
        ),
        "facts": _topology_facts(model),
    }


def summarize(captured: dict[str, Any]) -> dict[str, Any]:
    """Counts, smallest values and worst locations of short edges, narrow faces and small angles."""
    report, facts = captured["report"], captured["facts"]
    g = report["grid_um"]
    return {
        "grid_um": g,
        "edges": _edges(report["shortest_edges"], facts, g),
        "faces": _faces(report["narrowest_faces"], facts, g),
        "angles": _angles(report["smallest_angles"]),
        "issue125_classes": _classes(report, facts, g),
    }


def _as_written(full: dict[str, Any], top: int) -> dict[str, Any]:
    written = dict(full)
    for key in ("shortest_edges", "narrowest_faces", "smallest_angles"):
        written[key] = full[key][:top]
    for key in ("snap_drift", "arc_edges"):
        if key in full:
            written[key] = {**full[key], "worst": full[key]["worst"][:top]}
    written["top"] = top
    return written


def edge_length_um(row: dict[str, Any], facts: dict[str, Any]) -> float:
    """The length of a sliver-report edge: along its arc for an arc edge, else the report's own value."""
    return facts["arc_length_um"].get(row["id"], row["value_um"])


def _along_arc(chord: float, radius: float) -> float:
    """The length of the minor arc of ``radius`` over ``chord`` (same unit)."""
    return 2 * radius * math.asin(min(1.0, chord / (2 * radius)))


def _topology_facts(model) -> dict[str, Any]:
    """Which edges bound a wall face, each wall's column gap, each face's band involvement, and arc lengths."""
    topo = model.topology
    g = model.labelled.catalogue.grid.size_um
    wall_edges: set[str] = set()
    wall_gap_um: dict[str, float] = {}
    face_band: dict[str, bool] = {}
    arc_length_um = {
        eid: _along_arc(
            math.dist(
                (topo.vertices[edge.v0].x, topo.vertices[edge.v0].y),
                (topo.vertices[edge.v1].x, topo.vertices[edge.v1].y),
            ),
            edge.curve.radius,
        )
        * g
        for eid, edge in topo.edges.items()
        if edge.kind == "horizontal" and edge.curve is not None
    }
    for fid, face in topo.faces.items():
        loop_edges = [eid for loop in face.loops for eid, _ in loop]
        face_band[fid] = any(
            is_band_source(source)
            for eid in loop_edges
            for source in (topo.lineage.get(eid) or topo.edges[eid].sources)
        )
        if face.kind == "horizontal":
            continue
        wall_edges.update(loop_edges)
        # As SGB's wall measure does: the distance between the wall's vertical edges (its columns), along
        # the arc for a curved wall, whose horizontal edges carry its circle.
        columns = {
            (topo.vertices[topo.edges[eid].v0].x, topo.vertices[topo.edges[eid].v0].y)
            for eid, _ in face.loops[0]
            if topo.edges[eid].kind == "vertical"
        }
        circle = next(
            (
                topo.edges[eid].curve
                for eid, _ in face.loops[0]
                if topo.edges[eid].curve is not None
            ),
            None,
        )
        points = list(columns)
        gaps = [
            math.dist(p, q)
            if circle is None
            else _along_arc(math.dist(p, q), circle.radius)
            for i, p in enumerate(points)
            for q in points[i + 1 :]
        ]
        wall_gap_um[fid] = min(gaps, default=math.inf) * g
    return {
        "wall_edges": wall_edges,
        "wall_gap_um": wall_gap_um,
        "face_band": face_band,
        "arc_length_um": arc_length_um,
    }


def _origin(sources) -> str:
    return "band" if any(is_band_source(s) for s in sources) else "source"


def _worst(rows, value_key: str) -> list[dict[str, Any]]:
    return [
        {
            "value": row[value_key],
            "kind": row.get("kind"),
            "location_um": row["location_um"],
            "plane": row.get("plane"),
            "group": row.get("group"),
            "sources": row["sources"][:4],
        }
        for row in rows[:WORST]
    ]


def _binned(rows, value_of, cutoffs, unit_scale: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for origin in ("source", "band"):
        mine = [row for row in rows if row["_origin"] == origin]
        mine.sort(key=value_of)
        out[origin] = {
            "count_below": {
                str(c): sum(1 for row in mine if value_of(row) < c * unit_scale)
                for c in cutoffs
            },
            "smallest": [round(value_of(row), 6) for row in mine[:WORST]],
            "worst": _worst(mine, "_value"),
        }
    return out


def _edges(rows, facts, g) -> dict[str, Any]:
    horizontal = []
    for row in rows:
        if row["kind"] != "horizontal":
            continue
        horizontal.append(
            {
                **row,
                "_origin": _origin(row["sources"]),
                "_value": edge_length_um(row, facts),
            }
        )
    vertical = [row["value_um"] for row in rows if row["kind"] == "vertical"]
    return {
        "measure": "horizontal edge length along the edge, um (counts below N grid steps)",
        "total_horizontal": len(horizontal),
        **_binned(horizontal, lambda row: row["_value"], EDGE_STEPS, g),
        "shortest_vertical_um": min(vertical, default=None),
    }


def _faces(rows, facts, g) -> dict[str, Any]:
    horizontal = []
    walls = []
    for row in rows:
        origin = "band" if facts["face_band"].get(row["id"]) else "source"
        if row["kind"] == "horizontal":
            width = row["width_2a_over_p_um"]
            horizontal.append(
                {
                    **row,
                    "_origin": origin,
                    "_value": row["value_um"],
                    "_2ap": math.inf if width is None else width,
                }
            )
        else:
            gap = facts["wall_gap_um"].get(row["id"], math.inf)
            walls.append({**row, "_origin": origin, "_value": gap})
    return {
        "measure": "horizontal faces: local feature size and 2A/P, um; walls: column gap, um",
        "horizontal_local_feature_size": _binned(
            horizontal, lambda r: r["_value"], FACE_STEPS, g
        ),
        "horizontal_2a_over_p": _binned(
            [{**r, "_value": r["_2ap"]} for r in horizontal],
            lambda r: r["_2ap"],
            FACE_STEPS,
            g,
        ),
        "wall_column_gap": _binned(walls, lambda r: r["_value"], FACE_STEPS, g),
    }


def _angles(rows) -> dict[str, Any]:
    tagged = [
        {**row, "_origin": _origin(row["sources"]), "_value": row["value_deg"]}
        for row in rows
    ]
    return {
        "measure": "wedge angle at arrangement vertices, degrees",
        **_binned(tagged, lambda r: r["value_deg"], ANGLE_DEGREES, 1.0),
    }


def _classes(report, facts, g) -> dict[str, Any]:
    """The three classes of issue 125, counted at each bin (A: wall gap; B: face 2A/P or angle; C: free edge)."""
    walls = [
        (facts["wall_gap_um"].get(row["id"], math.inf), row)
        for row in report["narrowest_faces"]
        if row["kind"] != "horizontal"
    ]
    faces = [
        (row["width_2a_over_p_um"], row)
        for row in report["narrowest_faces"]
        if row["kind"] == "horizontal" and row["width_2a_over_p_um"] is not None
    ]
    free_edges = [
        (edge_length_um(row, facts), row)
        for row in report["shortest_edges"]
        if row["kind"] == "horizontal" and row["id"] not in facts["wall_edges"]
    ]

    def at(items, steps):
        return {
            str(s): {
                "source": sum(
                    1
                    for v, r in items
                    if v < s * g and _row_origin(r, facts) == "source"
                ),
                "band": sum(
                    1 for v, r in items if v < s * g and _row_origin(r, facts) == "band"
                ),
            }
            for s in steps
        }

    return {
        "A_wall_gap_below_steps": at(walls, FACE_STEPS),
        "B_face_2a_over_p_below_steps": at(faces, FACE_STEPS),
        "B_angle_below_1_degree": {
            "source": sum(
                1
                for r in report["smallest_angles"]
                if r["value_deg"] < 1 and _origin(r["sources"]) == "source"
            ),
            "band": sum(
                1
                for r in report["smallest_angles"]
                if r["value_deg"] < 1 and _origin(r["sources"]) == "band"
            ),
        },
        "C_free_edge_below_steps": at(free_edges, EDGE_STEPS),
    }


def _row_origin(row, facts) -> str:
    if "id" in row and row["id"] in facts["face_band"]:
        return "band" if facts["face_band"][row["id"]] else "source"
    return _origin(row["sources"])
