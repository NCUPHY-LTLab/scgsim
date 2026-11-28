"""Load optional Shapely operations used by tagged-grid construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapely.geometry import GeometryCollection, Point, Polygon
    from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class ShapelyGridOperations:
    """Hold the native geometry operations needed by tagged-grid stages."""

    line_string: Callable[..., BaseGeometry]
    point: Callable[..., Point]
    polygon: Callable[..., Polygon]
    get_coordinates: Callable[..., Iterable[tuple[float, ...]]]
    get_precision: Callable[[BaseGeometry], float]
    points: Callable[..., object]
    polygonize: Callable[..., GeometryCollection]
    set_precision: Callable[..., Sequence[Point]]
    union_all: Callable[..., BaseGeometry]


def load_shapely_grid_operations() -> ShapelyGridOperations:
    """Load Shapely only when a tagged-grid operation is requested."""
    from shapely import (  # noqa: PLC0415 -- preserve optional dependency loading at the grid-operation boundary.
        LineString,
        Point,
        Polygon,
        get_coordinates,
        get_precision,
        points,
        polygonize,
        set_precision,
        union_all,
    )

    return ShapelyGridOperations(
        LineString,
        Point,
        Polygon,
        get_coordinates,
        get_precision,
        points,
        polygonize,
        set_precision,
        union_all,
    )
