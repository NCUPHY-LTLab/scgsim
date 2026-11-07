"""Rule 1: every source polygon of the flattened component is a usable ring.

Owns the per-polygon checks: self-intersection, zero area, repeated vertices and off-grid vertices. It
works on the component's own GDS, never on a later stage's result, so it reports whatever the stages
after it do.

Two views of the same component are read. Validity is judged on the component flattened by
gdsfactory, which rounds every vertex to the database unit; a ring that collapses or crosses itself in
that rounding is invalid here, as it would be when the source stage rounds it. Off-grid vertices are
judged on the references expanded exactly, before any rounding.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from scgsim.geometry.drc.report import POLYGON_VALIDITY, Layer, LayoutFinding, Location
from scgsim.geometry.sgb.records.grid import Grid

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
    _, flat_polygons = _gds_polygons(flat)
    findings: list[LayoutFinding] = []
    for polygon in flat_polygons:
        layer = (polygon.layer, polygon.datatype)
        findings.extend(_ring_findings(np.asarray(polygon.points), layer, owner(layer)))
    grid, exact_polygons = _gds_polygons(component)
    findings.extend(_off_grid_findings(exact_polygons, grid, owner))
    return findings


def _gds_polygons(component: Any) -> tuple[Grid, list[Any]]:
    """The component's polygons with references expanded, and the grid of its database unit."""
    import gdstk  # noqa: PLC0415 - optional layout dependency, needed only by a layout check

    with TemporaryDirectory(prefix="scgsim-drc-") as folder:
        gds_file = Path(folder) / "component.gds"
        component.write_gds(gds_file)
        library = gdstk.read_gds(str(gds_file))
    # The pipeline takes the database unit as its model grid (``source_grid_context``).
    grid = Grid(float(Decimal(str(library.precision)) * Decimal("1000000")))
    # KLayout also writes a ``$$$CONTEXT_INFO$$$`` metadata cell with one zero-area polygon on 0/0, and
    # gdstk lists it as the top-level cell; the component's own cell is found by name.
    top = next(
        cell
        for cell in library.cells
        if isinstance(cell, gdstk.Cell) and cell.name == component.name
    )
    return grid, top.get_polygons()


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


def _off_grid_findings(
    polygons: Sequence[Any], grid: Grid, owner: Any
) -> Iterator[LayoutFinding]:
    """One finding per layer that has vertices off the grid; instances rotated off 90 degrees cause them."""
    off_grid: dict[Layer, list[Location]] = defaultdict(list)
    for polygon in polygons:
        layer = (polygon.layer, polygon.datatype)
        off_grid[layer].extend(_off_grid_vertices(polygon.points, grid))
    for layer, vertices in off_grid.items():
        if vertices:
            yield _finding(
                layer,
                owner(layer),
                vertices[0],
                f"{len(vertices)} vertices lie off the {grid.size_um} um source grid; the source stage rounds them",
                "place instances at multiples of the grid and rotate them by multiples of 90 degrees",
            )


def _off_grid_vertices(points: Any, grid: Grid) -> list[Location]:
    return [
        _vertex(point)
        for point in points
        if any(grid.lattice_units(float(value)) is None for value in point)
    ]


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
