"""Refuse a Ground contribution whose include region also takes a named local conductor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.models.input import LayoutPolygonSpec
from scgsim.geometry.primitives.grid import (
    current_grid_size_um,
    named_source,
    polygon_boolean,
    source_place,
    source_refusal,
)
from scgsim.geometry.source.ground import layout_polygon_region


def validate_ground_contribution_ownership(
    raw_layers: Sequence[Mapping[str, Any]],
) -> None:
    """A ground include cannot consume a separately named local conductor."""
    import gdstk  # noqa: PLC0415 - load optional native geometry only at ownership validation.

    for plane in raw_layers:
        for contribution in plane.get("geometry", {}).get(
            "source_occurrence_includes", ()
        ):
            region = contribution["polygon"]
            included = layout_polygon_region(
                gdstk,
                LayoutPolygonSpec(
                    polygon_id="source_ground_contribution",
                    layer="0/0",
                    exterior=region["exterior"],
                    holes=region["holes"],
                ),
            )
            for other in raw_layers:
                if (
                    other.get("metadata", {}).get("source_occurrence_path")
                    != contribution["source_occurrence_path"]
                    or other["metadata"].get("logical_layer_id")
                    != contribution["level"]
                    or other.get("geometry", {}).get("geometry_source", "gds_polygon")
                    != "gds_polygon"
                ):
                    continue
                for polygon in other["geometry"]["source_occurrence_polygons_um"]:
                    claimed = layout_polygon_region(
                        gdstk,
                        LayoutPolygonSpec(
                            polygon_id="named_local_conductor",
                            layer="0/0",
                            exterior=polygon["exterior"],
                            holes=polygon["holes"],
                        ),
                    )
                    overlap = polygon_boolean(
                        included, claimed, "and", precision=current_grid_size_um()
                    )
                    if overlap:
                        raise ValueError(
                            _contribution_overlap(contribution, other, overlap)
                        )


def _contribution_overlap(
    contribution: Mapping[str, Any], other: Mapping[str, Any], overlap: Sequence[Any]
) -> str:
    """The refusal of a Ground contribution whose include region overlaps a named conductor, at the overlap."""
    layer = contribution["layer"]
    with named_source(
        str(contribution["source_local_id"]), f"{int(layer[0])}/{int(layer[1])}"
    ):
        return source_refusal(
            f"ground contribution {contribution['source_local_id']!r} "
            f"also contains local Entity {other['semantic_id']!r}",
            [
                f"its include region overlaps {other['semantic_id']} at "
                f"{source_place([point for part in overlap for point in part.points])}"
            ],
            "Fix the layout or the declaration: select an include region clear of the "
            "named conductors, or draw them apart.",
        )
