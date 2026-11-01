"""Create semantic source records for accepted PDK ground-bump sites."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scgsim.geometry.source.ground_bump_geometry.lattice import fill_id
from scgsim.geometry.source.ground_bump_geometry.layout import (
    translate_polygon,
)

if TYPE_CHECKING:
    from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec


@dataclass(frozen=True)
class FillRecordContext:
    """Carry one fill's accepted sites, source contract and model constructors."""

    accepted: Sequence[tuple[int, int, float, float]]
    fill_prefix: str
    template_semantic_id: str
    bump_layer: tuple[int, int]
    body_polygon_um: Sequence[tuple[float, float]]
    result_stack: Mapping[str, Any]
    semantic_contract: Mapping[str, Any]
    layout_polygon_spec: Callable[..., LayoutPolygonSpec]
    semantic_entity_spec: Callable[..., SemanticEntitySpec]


def fill_semantic_records(context: FillRecordContext) -> tuple[list[Any], list[Any]]:
    """Build source-facing Entity and polygon records for accepted fill sites."""
    import gdstk  # noqa: PLC0415  # Optional dependency used only for emitted GDS records.

    accepted = context.accepted
    fill_prefix = context.fill_prefix
    template_semantic_id = context.template_semantic_id
    bump_layer = context.bump_layer
    body_polygon_um = context.body_polygon_um
    result_stack = context.result_stack
    semantic_contract = context.semantic_contract
    layout_polygon_spec = context.layout_polygon_spec
    semantic_entity_spec = context.semantic_entity_spec

    template = next(
        record
        for record in result_stack["layers"]
        if record.get("semantic_id") == template_semantic_id
    )
    net_id = semantic_contract.get("net_id")
    equipotential_id = semantic_contract.get("equipotential_id")
    host_solution_volume_id = semantic_contract.get("host_solution_volume_id")
    owner_semantic_ids = semantic_contract.get("owner_semantic_ids")
    if (
        not isinstance(net_id, str)
        or not net_id
        or not isinstance(equipotential_id, str)
        or not equipotential_id
        or not isinstance(host_solution_volume_id, str)
        or not host_solution_volume_id
        or isinstance(owner_semantic_ids, str | bytes)
        or not isinstance(owner_semantic_ids, Sequence)
        or len(owner_semantic_ids) != 2
        or any(
            not isinstance(owner_id, str) or not owner_id
            for owner_id in owner_semantic_ids
        )
    ):
        raise ValueError("indium fill requires an exact audited semantic contract.")
    semantic_group_id = str(
        template.get("metadata", {}).get("semantic_group_id")
        or template.get("semantic_id")
        or ""
    )
    if not semantic_group_id:
        raise ValueError("indium fill requires an exact authored bump family id.")
    entities, polygons = [], []
    for i, j, x, y in accepted:
        semantic_id = fill_id(fill_prefix, i, j)
        polygon_id = f"{semantic_id}__P0000"
        body = gdstk.Polygon(translate_polygon(body_polygon_um, x, y))
        polygon = layout_polygon_spec(
            polygon_id=polygon_id,
            layer=f"{bump_layer[0]}/{bump_layer[1]}",
            exterior=tuple((float(px), float(py)) for px, py in body.points),
            object_name=semantic_id,
            net_name=net_id,
            metadata={
                "source": "pdk_ground_bump_fill",
                "lattice_index": (i, j),
                "center_um": (x, y),
            },
        )
        entities.append(
            semantic_entity_spec(
                semantic_id=semantic_id,
                role="metal",
                material_id=str(template["material_id"]),
                material_kind="conductor",
                priority=int(template.get("priority", 0)),
                geometry_kind="layout_extrusion",
                part_role="bump_body",
                net_id=net_id,
                polygon_ids=(polygon_id,),
                host_void_semantic_id=host_solution_volume_id,
                route_representations={
                    "A": "cutout_boundary_shell",
                    "B": "cutout_boundary_shell",
                },
                geometry={
                    **{
                        key: value
                        for key, value in dict(template["geometry"]).items()
                        if key
                        not in {
                            "include_selector_points_um",
                            "exclude_selector_points_um",
                        }
                    },
                    "outer_loop": polygon.exterior,
                    "source_provenance": "pdk_ground_bump_fill",
                },
                metadata={
                    "semantic_group_id": semantic_group_id,
                    "source_layer_name": str(template["metadata"]["source_layer_name"]),
                    "source_provenance": "pdk_ground_bump_fill",
                    "source_kind": "fill",
                    "source_semantic_id": semantic_id,
                    "owner_semantic_ids": tuple(owner_semantic_ids),
                    "equipotential_id": equipotential_id,
                    "lattice_index": (i, j),
                },
            )
        )
        polygons.append(polygon)
    return entities, polygons
