"""Preserve source error aggregation across input validation phases."""

import pytest

from scgsim.geometry.compiler import validate_geometry_input
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.regions import (
    PortSheetOverlapRecord,
    PortSheetRegionRecord,
)


def test_input_errors_keep_polygon_entity_port_and_vacuum_order():
    """Report independent source defects together in their original order."""
    polygon = LayoutPolygonSpec("outline", "1/0", ((0, 0), (1, 0)))
    metal = SemanticEntitySpec(
        "metal",
        "conductor",
        "PEC",
        "conductor",
        1,
        "polygon",
        polygon_ids=("missing",),
        host_void_semantic_id="missing_host",
    )
    overlap = PortSheetOverlapRecord(
        "overlap",
        "port",
        "source",
        "unknown",
        "missing",
        (),
    )
    port = PortSheetRegionRecord(
        "port",
        "2/0",
        "source",
        (),
        overlaps=(overlap,),
    )
    source = GeometryBuildInput(
        (polygon, polygon), (metal,), port_sheet_regions=(port,)
    )

    with pytest.raises(ValueError) as error:
        validate_geometry_input(source)

    assert str(error.value).split("; ") == [
        "Invalid GeometryBuildInput: outline exterior requires at least 3 points",
        "duplicate polygon_id: outline",
        "outline exterior requires at least 3 points",
        "metal references unknown polygon_id: missing",
        "metal host_void_semantic_id must resolve a solution_region",
        "port must declare source 'palace_lumped_port_sheet'",
        "port requires source_name",
        "port requires target_layer",
        "port requires direction_sign_convention",
        "port requires 1-based port_index",
        "port requires a 3D direction",
        "port requires raw direction provenance",
        "port exterior requires 3 points",
        "overlap references unknown host semantic id unknown",
        "overlap references unknown host polygon id missing",
        "overlap overlap_loop requires 3 points",
        "GeometryBuildInput requires a vacuum solution_region",
    ]


def test_valid_input_is_returned_without_replacement():
    """Keep validation from replacing the caller's valid source record."""
    source = GeometryBuildInput(
        (),
        (SemanticEntitySpec("air", "solution_region", "Air", "vacuum", 0, "box"),),
    )

    assert validate_geometry_input(source) is source
