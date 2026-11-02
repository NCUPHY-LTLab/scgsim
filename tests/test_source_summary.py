"""Contracts for detached source summaries and their notebook display."""

from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    PortSheetRegionRecord,
    SemanticEntitySpec,
)
from scgsim.geometry.models.regions import PortSheetOverlapRecord
from scgsim.geometry.source.summary import InputSummary, summarize_geometry_input


def test_input_summary_data_is_detached_and_html_escapes_source_values() -> None:
    """Edits to returned data stay local, and source text remains escaped."""
    summary = InputSummary(
        {
            "source": {"top_cell_name": "<chip>"},
            "backend": {},
            "entities": [],
            "nets": [],
            "materials": [],
            "ports": [],
        }
    )

    detached = summary.data
    detached["source"]["top_cell_name"] = "changed"
    rendered = summary._repr_html_()

    assert summary.data["source"]["top_cell_name"] == "<chip>"
    assert "&lt;chip&gt;" in rendered
    assert "<chip>" not in rendered


def test_geometry_summary_deduplicates_overlapping_host_nets() -> None:
    """Repeated overlap cells on one Net appear once in detached summary data."""
    host_polygons = (
        LayoutPolygonSpec("host-a", "metal", ((0, 0), (2, 0), (2, 1))),
        LayoutPolygonSpec("host-b", "metal", ((3, 0), (5, 0), (5, 1))),
    )
    port_polygon = LayoutPolygonSpec("port-sheet", "port", ((0, 0), (1, 0), (1, 1)))
    host = SemanticEntitySpec(
        "metal",
        "conductor",
        "copper",
        "conductor",
        1,
        "polygon",
        net_id="N1",
        polygon_ids=("host-a", "host-b"),
        host_void_semantic_id="vacuum-domain",
        geometry={"z_um": 0.0, "thickness_um": 1.0},
    )
    vacuum = SemanticEntitySpec(
        "vacuum-domain",
        "solution_region",
        "vacuum",
        "vacuum",
        1,
        "domain",
        geometry={"z_um": -1.0, "thickness_um": 3.0},
    )
    port = PortSheetRegionRecord(
        "port-1",
        "port",
        "port-sheet",
        port_polygon.exterior,
        overlaps=(
            PortSheetOverlapRecord(
                "overlap-a",
                "port-1",
                "port-sheet",
                "metal",
                "host-a",
                ((0, 0), (1, 0), (1, 1)),
            ),
            PortSheetOverlapRecord(
                "overlap-b",
                "port-1",
                "port-sheet",
                "metal",
                "host-b",
                ((3, 0), (4, 0), (4, 1)),
            ),
        ),
        metadata={
            "source": "palace_lumped_port_sheet",
            "source_name": "P1",
            "target_layer": "signal",
            "direction_sign_convention": "outward",
            "port_index": 1,
            "direction": (1.0, 0.0, 0.0),
            "direction_raw": (1.0, 0.0, 0.0),
        },
    )
    build_input = GeometryBuildInput(
        polygons=(*host_polygons, port_polygon),
        entities=(host, vacuum),
        port_sheet_regions=(port,),
    )

    summary = summarize_geometry_input(
        build_input,
        prepared_stack={
            "materials": {
                "copper": {"kind": "conductor"},
                "vacuum": {"kind": "vacuum"},
            }
        },
    )

    assert summary.data["ports"][0]["overlap_nets"] == ["N1"]
