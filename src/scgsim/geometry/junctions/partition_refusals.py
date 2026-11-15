"""The located refusals of a junction partition: where in the layout each problem lies, and who acts.

Each reads the planner of the junction being cut (``partition._JunctionPartitionPlanner``),
whose ``where`` places local points in the layout.
"""

from __future__ import annotations

from typing import Any

from scgsim.geometry.junctions.core import PRECISION_UM
from scgsim.geometry.junctions.selection import (
    END_REFUSALS,
    EndSection,
    contact_points,
    end_region,
    nearest_meetings,
)
from scgsim.geometry.models.grid import WALL_COLUMN_LIMIT_STEPS
from scgsim.geometry.models.junctions import junction_net_id


def conductor(planner: Any, entity_id: str) -> str:
    """A conductor Entity with the source layers of its polygons."""
    polygons = planner.entities[entity_id].polygon_ids
    layers = sorted(
        {planner.originals[p].layer for p in polygons if p in planner.originals}
    )
    return f"{entity_id} on layer {', '.join(layers)}" if layers else entity_id


def conductor_contacts(planner: Any, of_net) -> list[str]:
    """Where each eligible conductor of a Net that ``of_net`` accepts touches or overlaps the closed sheet."""
    found = []
    for entity_id, polygons in planner.by_entity.items():
        net = junction_net_id(planner.entities[entity_id])
        points = (
            contact_points(planner.selection, polygons, planner.span, planner.width)
            if of_net(net)
            else []
        )
        if points:
            found.append(
                f"{conductor(planner, entity_id)}, of Net {net}, meets the sheet at "
                f"{planner.where.place(points)}"
            )
    return found


def no_gap_refusal(planner: Any, lower, upper) -> str:
    """The terminals' contacts with the sheet overlap along the port, so no clear gap lies between them."""
    a, b = planner.junction.terminal_a_net, planner.junction.terminal_b_net
    arcs = (
        planner.outline_a.arc_reach is not None
        or planner.outline_b.arc_reach is not None
    )
    return planner.where.refusal(
        "unsupported junction contact ordering or clear corridor",
        [
            *conductor_contacts(planner, lambda net: net in (a, b)),
            f"the conductors of Net {a}{', with the exact arcs SGB builds,' if arcs else ''} "
            f"reach along the sheet to {planner.where.centre(lower)}, past where those "
            f"of Net {b} begin, {planner.where.centre(upper)}",
        ],
        "Fix the layout: draw the sheet so that it crosses one clear gap from one of its "
        "conductors to the other, and touches each only at its own end.",
    )


def short_steps_refusal(planner: Any, side: Any, first) -> str:
    """No cut beside one terminal leaves its edges along the sheet sides zero or at least the limit."""
    starts = nearest_meetings(
        side.outline.meetings, first, planner.span, left=side.left
    )
    return planner.where.refusal(
        "junction corridor has no cut whose terminal steps are zero or at least "
        f"{WALL_COLUMN_LIMIT_STEPS} grid steps",
        [
            f"{planner.where.gap(planner.lower, planner.upper)}; beside Net {planner.net_of(side)} the end piece runs along the sheet "
            f"sides from {', '.join(planner.where.place([start]) for start in starts)}, where "
            "the conductor's outline meets them, and every cut from "
            f"{planner.where.centre(first)} across the gap leaves such an edge "
            f"shorter than {WALL_COLUMN_LIMIT_STEPS} grid steps"
        ],
        "Fix the layout: lengthen the gap the sheet crosses, or give the conductor a "
        "straight edge across the sheet.",
    )


def no_span_refusal(planner: Any) -> str:
    """Each end's first cut whose steps pass lies past the other's, so no rectangle is left."""
    a, b = planner.junction.terminal_a_net, planner.junction.terminal_b_net
    return planner.where.refusal(
        "junction DBU grid has no positive central span",
        [
            f"{planner.where.gap(planner.lower, planner.upper)}; the first cut beside Net {a} whose edges along the sheet "
            f"sides are zero or at least {WALL_COLUMN_LIMIT_STEPS} grid steps long lies "
            f"at {planner.where.centre(planner.a)}, and the first beside Net {b} at "
            f"{planner.where.centre(planner.b)}, so no rectangle is left between them"
        ],
        "Fix the layout: lengthen the gap the sheet crosses, or give the conductors "
        "straight edges across the sheet.",
    )


def end_finding(planner: Any, side: Any, cut, reason: str) -> str:
    """Why the end piece beyond one candidate cut was refused, and where its pieces lie."""
    section = EndSection(cut, planner.span, planner.width, left=side.left)
    pieces = [
        planner.where.place(
            [(x / PRECISION_UM, y / PRECISION_UM) for x, y in piece.points]
        )
        for piece in end_region(planner.selection, planner.end_source(side), section)
    ]
    words = END_REFUSALS.get(reason, "has parts that meet only at a point")
    where = f", at {', '.join(pieces)}" if pieces else ""
    return (
        f"beside Net {planner.net_of(side)}, the end piece beyond the cut at "
        f"{planner.where.centre(cut)} {words}{where}"
    )


def lineage_refusal(planner: Any, end, cut, net: str, owners: list[str]) -> str:
    """An end piece that no one conductor of its Net owns, or that several do."""
    names = (
        ", ".join((conductor(planner, owner) for owner in owners))
        or "none of them alone"
    )
    # An empty end piece is the cut itself, which its conductor must cover alone.
    points = [p for rings in end for ring in rings for p in ring]
    what = "its end piece at {} meets" if end else "its cut at {} is covered by"
    where = what.format(planner.where.place(points or [(cut, 0), (cut, planner.width)]))
    return planner.where.refusal(
        "junction end lacks an unambiguous source Entity/material/Z lineage",
        [f"{where} these conductors of Net {net}: {names}"],
        "Fix the layout: land each end of the sheet on one conductor, or draw the "
        "conductors it lands on there as one.",
    )
