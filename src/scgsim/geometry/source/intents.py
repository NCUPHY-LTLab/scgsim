"""Project generated sheet-interface intent from source-owned geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict, cast

from scgsim.geometry.models.common import PolygonRing, RouteLiteral
from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec
from scgsim.geometry.models.representation import RepresentationContext


class _SheetInterface(TypedDict):
    interface_id: str
    kind: str
    owner_semantic_ids: tuple[str, str]
    interface_kinds: tuple[str, str]
    recognition_rule: str
    intent_origin: str
    source_polygon_ids: tuple[str, ...]
    valid_routes: NotRequired[tuple[RouteLiteral, ...]]
    plane: dict[str, str | float]
    outer_loop: PolygonRing
    hole_loops: tuple[PolygonRing, ...]


@dataclass(frozen=True)
class _SheetSupport:
    """One source polygon or derived junction support for a sheet interface."""

    source_polygon_ids: tuple[str, ...]
    exterior: PolygonRing
    holes: tuple[PolygonRing, ...]


def _route_a_sheet_interfaces(
    entities: Sequence[SemanticEntitySpec],
    polygons: Sequence[LayoutPolygonSpec],
    *,
    representation_key: RouteLiteral = "A",
    representation_context: RepresentationContext | None = None,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Build generated sheet intents using the selected representation authority."""
    polygons_by_id = {polygon.polygon_id: polygon for polygon in polygons}
    interfaces: list[_SheetInterface] = []
    for entity in entities:
        selection = _sheet_representation(
            entity, representation_key, representation_context
        )
        if selection is None:
            continue
        intent_origin, recognition_rule, valid_routes, z_um = selection
        supports = _sheet_supports(entity, polygons_by_id)
        for index, support in enumerate(supports):
            # Conditional expansion retains the established serialized key order.
            interface = cast(
                _SheetInterface,
                {
                    "interface_id": f"MA__{entity.semantic_id}__AIR__{index:04d}",
                    "kind": "MA",
                    "owner_semantic_ids": (entity.semantic_id, "AIR"),
                    "interface_kinds": ("MS", "MA"),
                    "recognition_rule": recognition_rule,
                    "intent_origin": intent_origin,
                    "source_polygon_ids": support.source_polygon_ids,
                    **(
                        {"valid_routes": valid_routes}
                        if valid_routes is not None
                        else {}
                    ),
                    "plane": {"axis": "z", "value_um": z_um},
                    "outer_loop": support.exterior,
                    "hole_loops": support.holes,
                },
            )
            interfaces.append(interface)
    return {"interfaces": tuple(interfaces)}


def _sheet_representation(
    entity: SemanticEntitySpec,
    representation_key: str,
    representation_context: RepresentationContext | None,
) -> tuple[str, str, tuple[RouteLiteral, ...] | None, float] | None:
    """Select the existing sheet declaration and preserve its provenance keys."""
    if representation_context is None:
        # Keep the string parameter broad while narrowing the typed route-map lookup.
        if (
            entity.route_representations.get(cast(RouteLiteral, representation_key))
            != "surface_sheet"
        ):
            return None
        return (
            "generated_route_a_surface_sheet",
            "route_a_surface_sheet_polygon",
            (cast(RouteLiteral, representation_key),),
            float(entity.geometry.get("z_um", 0.0)),
        )
    representation = representation_context.entities.get(entity.semantic_id)
    if representation is None or representation.representation != "sheet":
        return None
    return (
        "generated_representation_context_surface_sheet",
        "representation_context_surface_sheet_polygon",
        None,
        float(representation.effective_z_interval_um[0]),
    )


def _sheet_supports(
    entity: SemanticEntitySpec,
    polygons_by_id: Mapping[str, LayoutPolygonSpec],
) -> tuple[_SheetSupport, ...]:
    """Resolve authored polygons or the existing junction-derived support."""
    if entity.metadata.get("junction_derived_support"):
        return (
            _SheetSupport(
                entity.polygon_ids,
                entity.geometry["outer_loop"],
                entity.geometry.get("hole_loops", ()),
            ),
        )
    supports = []
    for polygon_id in entity.polygon_ids:
        polygon = polygons_by_id.get(polygon_id)
        if polygon is None:
            raise ValueError(
                f"{entity.semantic_id} sheet intent references missing polygon {polygon_id!r}"
            )
        supports.append(_SheetSupport((polygon_id,), polygon.exterior, polygon.holes))
    return tuple(supports)
