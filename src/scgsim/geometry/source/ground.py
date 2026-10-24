"""Source-owned construction of positive Ground regions from local source facts.

Both the stack adapter and GeometryPlan occurrence lowering use this recipe.
The caller supplies the actual local plane outline when one exists; occurrence
transforms remain at the occurrence boundary and are applied exactly once.
"""

from __future__ import annotations

from scgsim.geometry._primitives.grid import polygon_boolean, preserve_declared_source

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec
from scgsim.geometry.source._polygon_ops import (
    _fused_gdstk_components,
    _point_in_layout_polygon,
    _ring_from_gdstk_polygon,
)


@preserve_declared_source
def _derived_ground_polygons(
    entity: SemanticEntitySpec,
    *,
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
    cell_bounds_um: Mapping[str, float],
    domain_bounds_by_semantic_id: Mapping[str, Mapping[str, Any]],
    plane_polygons: Sequence[Any] | None = None,
) -> tuple[LayoutPolygonSpec, ...]:
    """Derive one positive Ground from its plane, mask and explicit includes."""
    import gdstk

    from scgsim.geometry._primitives.loops import (
        _canonical_loop_sort_key,
        _split_gdstk_cutline_loop,
    )

    mask_layer = entity.geometry.get("mask_layer")
    if mask_layer is None:
        mask_key = (
            int(entity.geometry["gds_layer"]),
            int(entity.geometry["gds_datatype"]),
        )
    else:
        mask_key = (int(mask_layer[0]), int(mask_layer[1]))
    if plane_polygons is None:
        exterior = _rectangle_ring(
            _ground_plane_bounds(entity, domain_bounds_by_semantic_id, cell_bounds_um)
        )
        base = (gdstk.Polygon(exterior),)
    else:
        base = tuple(plane_polygons)
    without_mask = polygon_boolean(
        base,
        _merged_gdstk_polygons(polygons_by_layer.get(mask_key, ())),
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
    for contribution in source_includes:
        polygon = (
            contribution.get("polygon") if isinstance(contribution, Mapping) else None
        )
        if not isinstance(polygon, Mapping):
            raise ValueError(
                f"{entity.semantic_id} has an incomplete source contribution"
            )
        included.extend(
            _layout_polygon_region(
                gdstk,
                LayoutPolygonSpec(
                    polygon_id=f"{entity.semantic_id}__INCLUDE__{len(included):04d}",
                    layer=f"{mask_key[0]}/{mask_key[1]}",
                    exterior=polygon["exterior"],
                    holes=polygon["holes"],
                ),
            )
        )
    if include_layer is not None:
        if (
            not isinstance(include_layer, Sequence)
            or isinstance(include_layer, str | bytes)
            or len(include_layer) != 2
        ):
            raise TypeError(
                f"{entity.semantic_id} include_layer must be a GDS layer pair"
            )
        candidates = polygons_by_layer.get(
            (int(include_layer[0]), int(include_layer[1])), ()
        )
        # Fuse exact contacts before applying an explicit selector; ownership
        # still comes from the authored layer and selector, not proximity.
        components = _fused_gdstk_components(candidates)
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
                if _point_in_layout_polygon(selector, polygon)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{entity.semantic_id} include selector {tuple(point)!r} matched {len(matches)} source polygons"
                )
            included.extend(_layout_polygon_region(gdstk, matches[0]))
    regions = polygon_boolean(without_mask or (), tuple(included), "or", precision=1e-9)
    normalized_regions = sorted(
        (
            _split_gdstk_cutline_loop(_ring_from_gdstk_polygon(region))
            for region in regions or ()
        ),
        key=lambda region: (
            _canonical_loop_sort_key(region[0]),
            tuple(_canonical_loop_sort_key(hole) for hole in region[1]),
        ),
    )
    if not normalized_regions:
        raise ValueError(
            f"{entity.semantic_id} derived ground must lower to at least one planar region"
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


def _layout_polygon_region(gdstk: Any, polygon: LayoutPolygonSpec) -> tuple[Any, ...]:
    outer = gdstk.Polygon(polygon.exterior)
    holes = tuple(gdstk.Polygon(hole) for hole in polygon.holes)
    if not holes:
        return (outer,)
    return tuple(polygon_boolean((outer,), holes, "not", precision=1e-9) or ())


def _rectangle_ring(bounds: Mapping[str, float]) -> tuple[tuple[float, float], ...]:
    return (
        (float(bounds["x_min_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_max_um"])),
        (float(bounds["x_min_um"]), float(bounds["y_max_um"])),
    )
