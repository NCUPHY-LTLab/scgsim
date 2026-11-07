"""Rule 1: every source polygon of the flattened component is a usable ring.

Owns the per-polygon checks: self-intersection, zero area and repeated vertices. It works on the
component's own GDS, never on a later stage's result, so it reports whatever the stages after it do.

Validity is judged on the component flattened by gdsfactory, which rounds every vertex to the database
unit; a ring that collapses or crosses itself in that rounding is invalid here, as it would be when the
source stage rounds it. Vertices off the grid are not findings: the source stage rounds them, and a ring
that rounding breaks is reported here or by the physical-interpretation rule.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from scgsim.geometry.drc.report import POLYGON_VALIDITY, Layer, LayoutFinding, Location

_REASON_POINT = re.compile(r"\[(\S+) (\S+)\]")


def polygon_findings(
    component: Any, entities_by_layer: Mapping[Layer, Sequence[str]]
) -> list[LayoutFinding]:
    """Findings for every polygon of ``component`` after its references are flattened.

    ``entities_by_layer`` names the semantic Entities declared on a layer, so a finding can say what
    the polygon belongs to; a layer with none is reported against the cell.
    """

    def owner(layer: Layer) -> str:
        return ", ".join(entities_by_layer.get(layer, ())) or component.name

    flat = component.dup()
    flat.flatten()
    findings: list[LayoutFinding] = []
    for polygon in _gds_polygons(flat):
        layer = (polygon.layer, polygon.datatype)
        findings.extend(_ring_findings(np.asarray(polygon.points), layer, owner(layer)))
    return findings


def _gds_polygons(component: Any) -> list[Any]:
    """The component's polygons with references expanded."""
    import gdstk  # noqa: PLC0415 - optional layout dependency, needed only by a layout check

    with TemporaryDirectory(prefix="scgsim-drc-") as folder:
        gds_file = Path(folder) / "component.gds"
        component.write_gds(gds_file)
        library = gdstk.read_gds(str(gds_file))
    # KLayout also writes a ``$$$CONTEXT_INFO$$$`` metadata cell with one zero-area polygon on 0/0, and
    # gdstk lists it as the top-level cell; the component's own cell is found by name.
    top = next(
        cell
        for cell in library.cells
        if isinstance(cell, gdstk.Cell) and cell.name == component.name
    )
    return top.get_polygons()


def _ring_findings(points: Any, layer: Layer, entity: str) -> Iterator[LayoutFinding]:
    """The findings of one ring, at most one per kind of defect."""
    import shapely  # noqa: PLC0415 - optional layout dependency, needed only by a layout check

    ring = shapely.Polygon(points)
    if ring.area == 0:
        # A collapsed ring repeats vertices as a consequence; the zero area is the finding.
        yield _finding(
            layer,
            entity,
            _vertex(points[0]),
            "polygon has zero area",
            "delete the polygon or give it an area",
        )
        return
    repeated = _first_repeated_vertex(points)
    if repeated is not None:
        yield _finding(
            layer,
            entity,
            repeated,
            "polygon repeats a vertex (two consecutive points are equal)",
            "remove the duplicate point from the polygon's point list",
        )
    reason = shapely.is_valid_reason(ring)
    if reason != "Valid Geometry":
        yield _finding(
            layer,
            entity,
            _reason_location(reason),
            f"polygon is invalid: {reason}",
            "redraw the outline so that it does not cross itself",
        )


def _reason_location(reason: str) -> Location | None:
    """The point shapely names in its reason, such as ``Self-intersection[89.78 -64.4]``.

    Shapely gives the point only inside this text; it has no structured accessor for it.
    """
    match = _REASON_POINT.search(reason)
    return None if match is None else (float(match[1]), float(match[2]))


def _first_repeated_vertex(points: Any) -> Location | None:
    """The first vertex equal to its predecessor, the ring being closed from the last point to the first."""
    same = np.all(points == np.roll(points, 1, axis=0), axis=1)
    return _vertex(points[int(np.argmax(same))]) if same.any() else None


def _vertex(point: Any) -> Location:
    return float(point[0]), float(point[1])


def _finding(
    layer: Layer, entity: str, location: Location | None, message: str, hint: str
) -> LayoutFinding:
    return LayoutFinding(POLYGON_VALIDITY, layer, entity, location, message, hint)
