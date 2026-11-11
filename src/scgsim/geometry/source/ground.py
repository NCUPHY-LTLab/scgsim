"""Source-owned construction of positive Ground regions from local source facts.

Both the stack adapter and GeometryPlan occurrence lowering use this recipe.
The caller supplies the actual local plane outline when one exists; occurrence
transforms remain at the occurrence boundary and are applied exactly once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import wraps
from types import ModuleType
from typing import Any, cast

from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec
from scgsim.geometry.primitives.grid import (
    current_grid_context,
    named_source,
    polygon_boolean,
    polygon_rings,
    preserve_declared_source,
    source_place,
    source_refusal,
)
from scgsim.geometry.primitives.loops import canonical_loop_sort_key
from scgsim.geometry.primitives.planar_loops import PlanarLoop
from scgsim.geometry.source.polygon_ops import (
    FIX_SELECTOR,
    fused_gdstk_components,
    point_in_layout_polygon,
    realized_on_lattice,
    selector_place,
)
from scgsim.geometry.source.polygon_ops import (
    layout_polygon_region as layout_polygon_region,
)


def _mask_key(entity: SemanticEntitySpec) -> tuple[int, int]:
    """The GDS layer a Ground's mask is drawn on, which its derived polygons carry."""
    mask_layer = entity.geometry.get("mask_layer")
    if mask_layer is None:
        return (int(entity.geometry["gds_layer"]), int(entity.geometry["gds_datatype"]))
    return (int(mask_layer[0]), int(mask_layer[1]))


def _named_by_mask_layer(
    function: Callable[..., tuple[LayoutPolygonSpec, ...]],
) -> Callable[..., tuple[LayoutPolygonSpec, ...]]:
    """Name the Ground and its mask layer in the refusals of the drawn polygons it reads."""

    @wraps(function)
    def named(
        entity: SemanticEntitySpec, **kwargs: Any
    ) -> tuple[LayoutPolygonSpec, ...]:
        mask = _mask_key(entity)
        with named_source(entity.semantic_id, f"{mask[0]}/{mask[1]}"):
            return function(entity, **kwargs)

    return named


@preserve_declared_source
@_named_by_mask_layer
def derived_ground_polygons(  # noqa: PLR0913 -- the Ground recipe: the entity, three layout inputs, and the plane outline with its layer
    entity: SemanticEntitySpec,
    *,
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
    cell_bounds_um: Mapping[str, float],
    domain_bounds_by_semantic_id: Mapping[str, Mapping[str, Any]],
    plane_polygons: Sequence[Any] | None = None,
    plane_layer: Sequence[int] | None = None,
) -> tuple[LayoutPolygonSpec, ...]:
    """Derive one positive Ground from its plane, mask and explicit includes.

    ``plane_layer`` is the GDS layer ``plane_polygons`` were read from; a refusal of the plane
    outline names it.
    """
    import gdstk  # noqa: PLC0415 -- Ground construction is the boundary that requires the optional native GDS kernel

    mask_key = _mask_key(entity)
    if plane_polygons is None:
        exterior = _rectangle_ring(
            _ground_plane_bounds(entity, domain_bounds_by_semantic_id, cell_bounds_um)
        )
        base = (gdstk.Polygon(exterior),)
    else:
        base = tuple(realized_on_lattice(plane_polygons))
        if plane_layer is not None:
            _read_rings_on_layer(entity, plane_layer, base)
    without_mask = polygon_boolean(
        base,
        _merged_gdstk_polygons(
            realized_on_lattice(polygons_by_layer.get(mask_key, ()))
        ),
        "not",
        precision=1e-9,
    )
    include_layer = entity.geometry.get("include_layer")
    include_points = entity.geometry.get("include_selector_points_um", ())
    if include_layer is None and include_points:
        raise ValueError(
            f"{entity.semantic_id} include_selector_points_um requires include_layer"
        )
    included: list[Any] = []
    source_includes = entity.geometry.get("source_occurrence_includes", ())
    if source_includes and (include_layer is not None or include_points):
        raise ValueError(
            f"{entity.semantic_id} mixes source contributions with global includes"
        )
    _append_source_includes(included, gdstk, entity, mask_key, source_includes)
    if include_layer is not None:
        if (
            not isinstance(include_layer, Sequence)
            or isinstance(include_layer, str | bytes)
            or len(include_layer) != 2
        ):
            raise TypeError(
                f"{entity.semantic_id} include_layer must be a GDS layer pair"
            )
        include_key = (int(include_layer[0]), int(include_layer[1]))
        # The include polygons are drawn on the include layer, so their refusals name it.
        with named_source(entity.semantic_id, f"{include_key[0]}/{include_key[1]}"):
            included.extend(
                _selected_includes(
                    entity,
                    gdstk,
                    polygons_by_layer.get(include_key, ()),
                    include_points,
                )
            )
    regions = polygon_boolean(without_mask or (), tuple(included), "or", precision=1e-9)
    normalized_regions = sorted(
        (_region_loops(region) for region in regions or ()),
        key=lambda region: (
            canonical_loop_sort_key(region[0]),
            tuple(canonical_loop_sort_key(hole) for hole in region[1]),
        ),
    )
    if not normalized_regions:
        plane = [corner for polygon in base for corner in polygon.bounding_box()]
        raise ValueError(
            source_refusal(
                f"{entity.semantic_id} derived ground must lower to at least one planar region",
                [
                    f"the ground mask on this layer covers its whole plane, at {source_place(plane)}"
                ],
                "Fix the layout or the declaration: leave part of the plane outside the "
                "ground mask, or include the metal the Ground keeps.",
            )
        )
    return tuple(
        LayoutPolygonSpec(
            polygon_id=f"{entity.semantic_id}__P{index:04d}",
            layer=f"{mask_key[0]}/{mask_key[1]}",
            exterior=outer,
            holes=holes,
            object_name=entity.semantic_id,
            net_name=entity.net_id,
            metadata={
                "gds_layer": mask_key[0],
                "gds_datatype": mask_key[1],
                "source": "die_face_minus_ground_mask",
                "include_layer": include_layer,
            },
        )
        for index, (outer, holes) in enumerate(normalized_regions)
    )


def _read_rings_on_layer(
    entity: SemanticEntitySpec, layer: Sequence[int], polygons: Sequence[Any]
) -> None:
    """Read each polygon's rings as the grid-bound Boolean that follows will, naming ``layer``.

    The Boolean reads the plane and the mask together under the mask's name; reading the
    plane's rings first refuses exactly the same outlines, on the layer they are drawn on.
    Without a bound grid the Boolean runs natively and reads no rings, so neither does this.
    """
    if current_grid_context() is None:
        return
    with named_source(entity.semantic_id, f"{int(layer[0])}/{int(layer[1])}"):
        for polygon in polygons:
            polygon_rings(polygon)


def _selected_includes(
    entity: SemanticEntitySpec,
    gdstk: ModuleType,
    drawn: Sequence[Any],
    include_points: Sequence[Any],
) -> list[Any]:
    """The include-layer components the Ground's selector points pick, one per point."""
    candidates = realized_on_lattice(drawn)
    # Fuse exact contacts before applying an explicit selector; ownership
    # still comes from the authored layer and selector, not proximity.
    components = fused_gdstk_components(candidates)
    included: list[Any] = []
    for point in include_points:
        if (
            not isinstance(point, Sequence)
            or isinstance(point, str | bytes)
            or len(point) != 2
        ):
            raise TypeError(
                f"{entity.semantic_id} include_selector_points_um requires 2D points"
            )
        selector = (float(point[0]), float(point[1]))
        matches = [
            polygon
            for polygon, _, _ in components
            if point_in_layout_polygon(selector, polygon)
        ]
        if len(matches) != 1:
            raise ValueError(
                source_refusal(
                    f"{entity.semantic_id} include selector {tuple(point)!r} matched "
                    f"{len(matches)} source polygons",
                    [
                        selector_place(
                            selector,
                            [polygon.exterior for polygon in matches],
                            [polygon.exterior for polygon, _, _ in components],
                        )
                    ],
                    FIX_SELECTOR,
                )
            )
        included.extend(layout_polygon_region(gdstk, matches[0]))
    return included


def _append_source_includes(
    included: list[Any],
    native: ModuleType,
    entity: SemanticEntitySpec,
    mask_key: tuple[int, int],
    contributions: Iterable[object],
) -> None:
    """Append declared occurrence polygons in their original source order."""
    for contribution in contributions:
        polygon = (
            contribution.get("polygon") if isinstance(contribution, Mapping) else None
        )
        if not isinstance(polygon, Mapping):
            raise ValueError(
                f"{entity.semantic_id} has an incomplete source contribution"
            )
        included.extend(
            layout_polygon_region(
                native,
                LayoutPolygonSpec(
                    polygon_id=f"{entity.semantic_id}__INCLUDE__{len(included):04d}",
                    layer=f"{mask_key[0]}/{mask_key[1]}",
                    exterior=polygon["exterior"],
                    holes=polygon["holes"],
                ),
            )
        )


def _merged_gdstk_polygons(polygons: Sequence[Any]) -> tuple[Any, ...]:
    if not polygons:
        return ()

    merged = polygon_boolean(
        polygons,
        (),
        "or",
        precision=1e-9,
    )
    return tuple(merged or ())


def _region_loops(region: Any) -> tuple[PlanarLoop, tuple[PlanarLoop, ...]]:
    """Read one Boolean region's exterior and holes as planar loops.

    Under a grid the Boolean already returns each region as its exterior and
    holes, so they are read as they are. Rebuilding them from the region's
    cutline walk loses that split: a hole that touches the exterior or another
    hole at one vertex, and the unrounded source reference's vertex pairs one
    float step apart, no longer form simple loops (Issue #87). Only a native
    gdstk polygon, from an unbound Boolean, has its walk split.
    """
    # The grid's ring type admits points of any length; ground regions are XY.
    return cast("tuple[PlanarLoop, tuple[PlanarLoop, ...]]", polygon_rings(region))


def _ground_plane_bounds(
    entity: SemanticEntitySpec,
    domain_bounds_by_semantic_id: Mapping[str, Mapping[str, Any]],
    cell_bounds_um: Mapping[str, float],
) -> Mapping[str, Any]:
    plane_bounds_ref = entity.geometry.get("plane_bounds_ref")
    if plane_bounds_ref is None:
        return cell_bounds_um
    if not isinstance(plane_bounds_ref, str):
        raise TypeError(f"{entity.semantic_id} plane_bounds_ref must be a string")
    try:
        return domain_bounds_by_semantic_id[plane_bounds_ref]
    except KeyError as exc:
        raise ValueError(
            f"{entity.semantic_id} plane_bounds_ref {plane_bounds_ref!r} "
            "does not match a solution region"
        ) from exc


def _rectangle_ring(bounds: Mapping[str, float]) -> tuple[tuple[float, float], ...]:
    return (
        (float(bounds["x_min_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_max_um"])),
        (float(bounds["x_min_um"]), float(bounds["y_max_um"])),
    )
