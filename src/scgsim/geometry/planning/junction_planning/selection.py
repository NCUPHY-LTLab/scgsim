"""Select terminal-owned residual ends on the authored corridor and grid."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from fractions import Fraction

from scgsim.geometry.planning.junction_planning.boolean_geometry import (
    boolean,
    build_polygon_region,
    read_boolean_rings,
    rectangle,
)
from scgsim.geometry.planning.junction_planning.core import PRECISION_UM
from scgsim.geometry.planning.junction_planning.exact_geometry import (
    closed_contact,
    covered,
    positive_boundary_contact,
    section,
)


@dataclass(frozen=True)
class EndSection:
    """Carry one selected end cut, authored rectangle extent and terminal side."""

    cut: int
    span: int
    width: int
    left: bool


def contacts(polygons, span, width):
    """Find extrema of area, segment and point contact with closed authored S."""
    region = boolean(
        build_polygon_region(polygons),
        build_polygon_region(rectangle(0, 0, span, width)),
        "and",
    )
    positions: list[int | Fraction] = [
        round(x / PRECISION_UM) for p in region for x, _ in p.points
    ]
    for rings in polygons:
        for axis, c, low, high in (
            (0, 0, 0, width),
            (0, span, 0, width),
            (1, 0, 0, span),
            (1, width, 0, span),
        ):
            intervals, points = section(rings, axis, c, low, high)
            if axis == 0 and (intervals or points):
                positions.append(c)
            elif axis == 1:
                positions.extend(points)
                positions.extend((v for pair in intervals for v in pair))
    return (min(positions), max(positions)) if positions else None


def bound_decimal(bound):
    """Convert a rational corridor bound into its existing decimal grid value."""
    rational = Fraction(bound)
    return (
        Decimal(rational.numerator)
        / Decimal(rational.denominator)
        * Decimal(str(PRECISION_UM))
    )


def candidate_indices(bound, step, *, left):
    """Read the existing ordered candidate grid indices beside one bound."""
    q = bound_decimal(bound) / Decimal(str(step))
    floor = int(q.to_integral_value(rounding=ROUND_FLOOR))
    aligned = q == floor
    if left:
        return ([floor] if aligned else []) + [floor + 1]
    return ([floor] if aligned else []) + [floor if not aligned else floor - 1]


def grid_cut(index, step):
    """Resolve the requested and realized position of one grid index."""
    requested = float(Decimal(index) * Decimal(str(step)))
    return (requested, round(requested / PRECISION_UM))


def end_candidate(free, own, foreign, section: EndSection):
    """Check the selected terminal end against exact closed-contact conditions."""
    clip = (
        rectangle(0, 0, section.cut, section.width)
        if section.left
        else rectangle(section.cut, 0, section.span, section.width)
    )
    region = (
        boolean(free, build_polygon_region(clip), "and")
        if (section.cut > 0 if section.left else section.cut < section.span)
        else []
    )
    try:
        end = read_boolean_rings(region)
    except ValueError as exc:
        return (None, {"valid": False, "reason": str(exc)})
    if len(end) > 1:
        return (
            None,
            {
                "valid": False,
                "reason": "nonempty end has disconnected components",
                "components": len(end),
            },
        )
    if end and (not positive_boundary_contact(end, own)):
        return (
            None,
            {"valid": False, "reason": "end lacks positive-length own-arm contact"},
        )
    if closed_contact(end, foreign):
        return (
            None,
            {"valid": False, "reason": "end contacts wrong terminal or third net"},
        )
    if not covered([*own, *end], 0, section.cut, 0, section.width):
        return (
            None,
            {"valid": False, "reason": "designated end edge is not wholly covered"},
        )
    return (end, {"valid": True, "empty_residual": not end, "components": len(end)})
