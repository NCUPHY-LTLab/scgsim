"""Where a meshing stop leaves its bad elements, and the fine structure recorded at those places.

Owns the answer to "who must act, and where" when the repair loop stops without a clean mesh.  Each remaining
bad element is placed where the loop refined it, with the reach of that refinement: the fine core of the local
size that holds it, or the size it would ask for of its own (:func:`~scgsim.meshing.repair.refinement_reach`).
Elements held by one local size, or whose own reaches overlap, are one place.  The meshing-stage findings
(``fine_structure``) whose box lies within that reach of an element are the fine structure at that place, each
with its owner; a place with none is the mesher's to answer for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .built_model import Point3
from .findings import BadElement
from .fine_structure import FineFeature
from .repair import RepairPolicy, refinement_reach
from .sizing import LocalSize


@dataclass
class StopPlace:
    """One place where bad elements remain: its centre, the refinement's reach, the elements and the findings there."""

    centre: Point3
    radius_um: float
    elements: list[BadElement] = field(default_factory=list)
    spots: list[tuple[Point3, float]] = field(default_factory=list)
    findings: list[FineFeature] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """The place as the quality report records it."""
        return {
            "centre_um": list(self.centre),
            "radius_um": self.radius_um,
            "bad_elements": len(self.elements),
            "worst_quality": min(e.quality for e in self.elements),
            "elements": sorted(e.element for e in self.elements),
            "fine_structure": [
                {
                    "rule": f.rule,
                    "measure": f.measure,
                    "action": f.action,
                    "owner": f.owner,
                    "entity": f.entity,
                    "groups": list(f.groups),
                    "sources": list(f.sources),
                    "location_um": list(f.location_um),
                    "value": f.value,
                    "value_steps": f.value_steps,
                    "fix": f.fix,
                }
                for f in self.findings
            ],
        }


def stop_places(
    bad: tuple[BadElement, ...],
    local: tuple[LocalSize, ...],
    threshold: float,
    policy: RepairPolicy,
    findings: tuple[FineFeature, ...],
) -> list[StopPlace]:
    """The places of the remaining bad elements, most elements first, each with the findings recorded there."""
    held: dict[int, StopPlace] = {}
    own: list[StopPlace] = []
    for element in sorted(bad, key=lambda e: e.element):
        holder, centre, reach = refinement_reach(element, local, threshold, policy)
        if holder is not None:
            place = held.setdefault(holder, StopPlace(centre, reach))
        else:
            place = _overlapping(own, centre, reach)
            if place is None:
                place = StopPlace(centre, reach)
                own.append(place)
        place.elements.append(element)
        place.spots.append((centre, reach))
    places = [*held.values(), *own]
    for place in places:
        place.findings = [
            finding
            for finding in findings
            if any(
                _point_box_distance(spot, finding.bounds_um) <= reach
                for spot, reach in place.spots
            )
        ]
    places.sort(key=lambda p: (-len(p.elements), min(e.element for e in p.elements)))
    return places


def _overlapping(
    places: list[StopPlace], centre: Point3, reach: float
) -> StopPlace | None:
    """The first place one of whose elements' reaches overlaps this one."""
    for place in places:
        if any(math.dist(spot, centre) <= reach + r for spot, r in place.spots):
            return place
    return None


def _point_box_distance(point: Point3, box: tuple[Point3, Point3]) -> float:
    low, high = box
    return math.sqrt(
        sum(max(low[k] - point[k], point[k] - high[k], 0.0) ** 2 for k in range(3))
    )
