"""Bind each authored lumped support to its level, its terminals and the one GDS polygon it names, for the plan."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scgsim.geometry.grid.booleans import split_gdstk_cutline_loop
from scgsim.geometry.grid.polygon_ops import ring_from_gdstk_polygon
from scgsim.geometry.grid.scope import GridScope
from scgsim.geometry.models.input import LayoutPolygonSpec
from scgsim.geometry.models.lumped import LumpedSupport
from scgsim.geometry.models.refusals import Declared
from scgsim.geometry.source.occurrence import read_polygons_by_layer


def _lumped_ring_identity(ring):
    """Exact source ring identity, independent only of start and traversal."""
    points = tuple(tuple(point) for point in ring)
    if points[0] == points[-1]:
        points = points[:-1]
    reverse = tuple(reversed(points))
    return min(
        order[index:] + order[:index]
        for order in (points, reverse)
        for index in range(len(order))
    )


def occurrence_lumped_polygon(
    scope: GridScope,
    support,
    *,
    cell,
    excluded_reference_indexes=(),
    transform=(1, 0, 0, 1, 0, 0),
):
    """Bind the complete authored local ring to actual original GDS geometry.

    The rings are read in the cell's own coordinates; ``transform`` places the
    occurrence in the layout, where a refusal locates the ring.
    """

    declared = (
        _lumped_ring_identity(support.exterior),
        sorted(_lumped_ring_identity(ring) for ring in support.holes),
    )
    selected = []
    for polygon in read_polygons_by_layer(
        cell, excluded_reference_indexes=excluded_reference_indexes
    ).get(support.source_layer, ()):
        exterior, holes = split_gdstk_cutline_loop(
            scope, ring_from_gdstk_polygon(scope, polygon)
        )
        local = LayoutPolygonSpec(support.source_polygon_id, "", exterior, holes)
        actual = (
            _lumped_ring_identity(exterior),
            sorted(_lumped_ring_identity(ring) for ring in holes),
        )
        if actual == declared:
            selected.append(local)
    if len(selected) != 1:
        layer = f"{support.source_layer[0]}/{support.source_layer[1]}"
        named = scope.placed(transform).naming(support.support_id, layer)
        raise ValueError(
            named.refusal(
                f"lumped support {support.support_id!r} matched {len(selected)} "
                "source polygons",
                [
                    f"its authored ring lies at {named.place(support.exterior)}, and "
                    f"{len(selected)} polygons drawn on the layer have exactly that outline"
                ],
                "Fix the declaration or the layout: give the lumped support the exact "
                "outline and holes of one polygon drawn on its layer.",
            )
        )
    return selected[0]


def check_lumped_terminals(
    support: LumpedSupport, entities: Mapping[str, Any], owner: Declared
) -> None:
    """Refuse a lumped support whose terminal Entities are not conductors with a final Net."""
    for entity_id in (
        *support.terminal_a_entity_ids,
        *support.terminal_b_entity_ids,
    ):
        if (
            entity_id not in entities
            or entities[entity_id].material_kind != "conductor"
        ):
            raise ValueError(
                owner.refusal(f"its terminal Entity {entity_id!r} is not a conductor")
            )
        if entities[entity_id].net_id is None:
            raise ValueError(
                owner.refusal(
                    f"its terminal Entity {entity_id!r} lacks final Net ownership; "
                    "give it a Net with set_nets"
                )
            )


def lumped_support_plane(
    support: LumpedSupport, layer_stack: Any, owner: Declared
) -> float:
    """The Z of the support's level, which its layer, its declared plane and its contacts must match."""
    if support.source_level not in layer_stack.layers:
        raise ValueError(
            owner.refusal(
                f"its level {support.source_level!r} is not declared in the layer stack"
            )
        )
    level = layer_stack.layers[support.source_level]
    from gdsfactory.pdk import get_layer_tuple  # noqa: PLC0415 - needed only for authored lumped supports

    if get_layer_tuple(level.layer.layer) != support.source_layer:
        raise ValueError(
            owner.refusal("its source level/layer binding differs from the layer stack")
        )
    source_z_um = float(level.zmin)
    if support.source_z_um is not None and support.source_z_um != source_z_um:
        raise ValueError(
            owner.expected(
                "source_z_um",
                f"its level's plane {source_z_um}",
                support.source_z_um,
            )
        )
    if any(point[2] != source_z_um for point in support.contact_points_um):
        raise ValueError(
            owner.refusal(
                f"its contacts differ from its level's plane z = {source_z_um} um"
            )
        )
    return source_z_um
