"""Project generated sheet interface intent from current source polygons. Compiler refresh uses this exact source projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext


def _route_a_sheet_interfaces(
    entities: Sequence[SemanticEntitySpec],
    polygons: Sequence[LayoutPolygonSpec],
    *,
    representation_key: str = "A",
    representation_context: RepresentationContext | None = None,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Build generated sheet intents from one explicit representation authority."""
    polygons_by_id = {polygon.polygon_id: polygon for polygon in polygons}
    interfaces: list[Mapping[str, Any]] = []
    for entity in entities:
        if representation_context is None:
            if entity.route_representations.get(representation_key) != "surface_sheet":
                continue
            intent_origin = "generated_route_a_surface_sheet"
            recognition_rule = "route_a_surface_sheet_polygon"
            valid_routes = (representation_key,)
            z_um = float(entity.geometry.get("z_um", 0.0))
        else:
            representation = representation_context.entities.get(entity.semantic_id)
            if representation is None or representation.representation != "sheet":
                continue
            intent_origin = "generated_representation_context_surface_sheet"
            recognition_rule = "representation_context_surface_sheet_polygon"
            valid_routes = None
            z_um = float(representation.effective_z_interval_um[0])
        for index, polygon_id in enumerate(entity.polygon_ids):
            polygon = polygons_by_id.get(polygon_id)
            if polygon is None:
                raise ValueError(
                    f"{entity.semantic_id} sheet intent references missing "
                    f"polygon {polygon_id!r}"
                )
            interfaces.append(
                {
                    "interface_id": f"MA__{entity.semantic_id}__AIR__{index:04d}",
                    "kind": "MA",
                    "owner_semantic_ids": (entity.semantic_id, "AIR"),
                    "interface_kinds": ("MS", "MA"),
                    "recognition_rule": recognition_rule,
                    "intent_origin": intent_origin,
                    "source_polygon_ids": (polygon_id,),
                    **(
                        {"valid_routes": valid_routes}
                        if valid_routes is not None
                        else {}
                    ),
                    "plane": {"axis": "z", "value_um": z_um},
                    "outer_loop": polygon.exterior,
                    "hole_loops": polygon.holes,
                }
            )
    return {"interfaces": tuple(interfaces)}
