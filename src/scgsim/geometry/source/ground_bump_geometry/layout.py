"""Translate PDK ground-bump shapes to layout footprints and GDS records."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from scgsim.geometry.primitives.grid import polygon_boolean
from scgsim.geometry.source.polygon_ops import (
    flattened_polygons,
    without_repeated_vertices,
)


def polygon_center(polygon: Any) -> tuple[float, float]:
    """Return the axis-aligned center of one authored polygon."""
    points = polygon_points(polygon)
    return (
        (min(point[0] for point in points) + max(point[0] for point in points)) / 2,
        (min(point[1] for point in points) + max(point[1] for point in points)) / 2,
    )


def polygon_points(polygon: Any) -> list[list[float]]:
    """Serialize a polygon boundary as numeric XY coordinate pairs."""
    return [[float(x), float(y)] for x, y in polygon.points]


def intersecting_blocker_ids(
    footprint: Sequence[Any], blockers: Sequence[tuple[str, Sequence[Any]]]
) -> tuple[str, ...]:
    """Return source IDs for blockers intersecting the requested footprint."""
    return tuple(
        source_id
        for source_id, polygons in blockers
        if polygons and polygon_boolean(footprint, polygons, "and", precision=1e-9)
    )


def point2(value: Any, name: str) -> tuple[float, float]:
    """Convert one two-item source coordinate to a numeric XY pair."""
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise TypeError(f"{name} must be a two-item numeric sequence.")
    return float(value[0]), float(value[1])


def layer_pair(value: Any, name: str) -> tuple[int, int]:
    """Read the exact integer layer/datatype pair from PDK input."""
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise TypeError(f"{name} must be a two-item integer layer pair.")
    return int(value[0]), int(value[1])


def relative_polygon(value: Any, name: str) -> tuple[tuple[float, float], ...]:
    """Read one PDK-authored polygon relative to its placement origin."""
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) < 3
    ):
        raise TypeError(f"{name} must contain at least three 2D points.")
    points = tuple(point2(point, name) for point in value)
    area = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, (*points[1:], points[0]), strict=True)
    )
    if not isfinite(area) or abs(area) <= 1e-12:
        raise ValueError(f"{name} must define a finite nondegenerate polygon.")
    return points


def translate_polygon(
    points: Sequence[tuple[float, float]], x: float, y: float
) -> tuple[tuple[float, float], ...]:
    """Place relative polygon vertices at one lattice center."""
    return tuple((px + x, py + y) for px, py in points)


def gdstk_cell(library: Any, name: str) -> Any:
    """Select the named source cell or the sole top-level cell."""
    matches = [cell for cell in library.cells if cell.name == name] if name else []
    if len(matches) == 1:
        return matches[0]
    tops = library.top_level()
    if len(tops) != 1:
        raise ValueError("generated component GDS needs one exact top cell.")
    return tops[0]


def gds_layer_polygons(cell: Any, layer: tuple[int, int]) -> list[Any]:
    """Read repeated source polygons from one exact GDS layer pair."""
    return without_repeated_vertices(
        polygon
        for polygon in flattened_polygons(cell)
        if (polygon.layer, polygon.datatype) == layer
    )


def gdstk_polygons(polygons: Sequence[Any]) -> list[Any]:
    """Lower adapter polygons to gdstk polygons while subtracting holes."""
    import gdstk  # noqa: PLC0415  # Optional dependency used only for GDS audits.

    result = []
    for polygon in polygons:
        outer = [gdstk.Polygon(polygon.exterior)]
        holes = [gdstk.Polygon(hole) for hole in polygon.holes]
        result.extend(
            outer if not holes else polygon_boolean(outer, holes, "not", precision=1e-9)
        )
    return result


def component_with_canonical_bumps(
    component: Any,
    spec: Mapping[str, Any],
    accepted: Sequence[tuple[int, int, float, float]],
    pitch: float,
) -> Any:
    """Place accepted PDK bump bodies as canonical rows or references."""
    import gdsfactory as gf  # noqa: PLC0415  # Optional layout dependency.

    derived = gf.Component()
    derived.add_ref(component)
    bump = gf.Component()
    bump.add_polygon(
        relative_polygon(spec["body_polygon_um"], "body_polygon_um"),
        layer=layer_pair(spec["body_layer"], "body_layer"),
    )
    bump.add_polygon(
        relative_polygon(spec["contact_polygon_um"], "contact_polygon_um"),
        layer=layer_pair(spec["contact_layer"], "contact_layer"),
    )
    rows: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for i, j, x, y in accepted:
        rows.setdefault((j, 0), []).append((i, x, y))
    for (_, _), sites in sorted(rows.items()):
        sites.sort()
        start = 0
        while start < len(sites):
            end = start + 1
            while end < len(sites) and sites[end][0] == sites[end - 1][0] + 1:
                end += 1
            run = sites[start:end]
            if len(run) == 1:
                reference = derived << bump
            else:
                reference = derived << gf.components.array(
                    component=bump,
                    columns=len(run),
                    rows=1,
                    column_pitch=pitch,
                    row_pitch=pitch,
                )
            reference.move((run[0][1], run[0][2]))
            start = end
    derived.ports.clear()
    return derived


def assert_generated_center_selectors(
    component: Any,
    bump_layer: tuple[int, int],
    accepted: Sequence[tuple[int, int, float, float]],
) -> None:
    """Verify every accepted center selects exactly one emitted body."""
    import gdstk  # noqa: PLC0415  # Optional dependency used only during verification.

    with TemporaryDirectory(prefix="scgsim-indium-fill-verify-") as temporary:
        path = Path(temporary) / "derived.gds"
        component.write_gds(path)
        cell = gdstk_cell(
            gdstk.read_gds(str(path)), str(getattr(component, "name", "") or "")
        )
        polygons = gds_layer_polygons(cell, bump_layer)
        for i, j, x, y in accepted:
            matches = [
                polygon
                for polygon in polygons
                if _point_in_polygon((x, y), polygon.points)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"generated bump at lattice index {(i, j)!r} matched "
                    f"{len(matches)} canonical body polygons."
                )


def _point_in_polygon(point: tuple[float, float], vertices: Any) -> bool:
    x, y = point
    inside = False
    points = list(vertices)
    for first, second in zip(points, [*points[1:], points[0]], strict=True):
        x1, y1 = first
        x2, y2 = second
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside
