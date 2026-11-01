"""Keep source port region ownership and dependency direction explicit."""

import ast
from dataclasses import asdict
from pathlib import Path

import gdstk
import pytest

from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec
from scgsim.geometry.source.ports import port_sheet_regions_from_stack_metadata


def test_port_regions_preserve_ids_metadata_and_overlap_ownership() -> None:
    """Build stable port IDs and preserve source layer and host overlap facts."""
    host = SemanticEntitySpec(
        "metal",
        "conductor",
        "Cu",
        "conductor",
        0,
        "polygon",
        polygon_ids=("host-poly",),
        geometry={"gds_layer": 5, "gds_datatype": 0},
        metadata={"logical_layer_id": "M5"},
    )
    host_polygon = LayoutPolygonSpec(
        "host-poly", "5/0", ((0, 0), (4, 0), (4, 4), (0, 4))
    )
    source_polygon = gdstk.Polygon([(1, 1), (3, 1), (3, 3), (1, 3)], layer=9)
    source_polygon.datatype = 2
    metadata = {
        "port_sheet_source_layers": (
            {
                "source": "palace_lumped_port_sheet",
                "layer": 9,
                "datatype": 2,
                "name": "P1",
                "port_index": 1,
                "target_layer": "M5",
                "direction": (3, 4, 0),
                "direction_sign_convention": "authored",
            },
        )
    }

    regions = port_sheet_regions_from_stack_metadata(
        metadata,
        polygons_by_layer={(9, 2): (source_polygon,)},
        host_entities=(host,),
        host_polygons=(host_polygon,),
    )

    assert len(regions) == 1
    region = regions[0]
    assert asdict(region) == {
        "port_sheet_id": "PORT_SHEET__P1__0000",
        "source_layer": "9/2",
        "source_polygon_id": "PORT_SHEET_9_2__P0000",
        "exterior": ((1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)),
        "holes": (),
        "overlaps": (
            {
                "overlap_id": "PORT_SHEET_OVERLAP__PORT_SHEET__P1__0000__metal__0000",
                "port_sheet_id": "PORT_SHEET__P1__0000",
                "port_polygon_id": "PORT_SHEET_9_2__P0000",
                "host_semantic_id": "metal",
                "host_polygon_id": "host-poly",
                "overlap_loop": ((3.0, 3.0), (1.0, 3.0), (1.0, 1.0), (3.0, 1.0)),
                "operation": "local_fragment_required",
                "metadata": {"overlap_polygon_index": 0},
            },
        ),
        "metadata": {
            "source_index": 0,
            "source_name": "P1",
            "port_index": 1,
            "target_layer": "M5",
            "direction": (0.6, 0.8, 0.0),
            "direction_raw": (3.0, 4.0, 0.0),
            "direction_sign_convention": "authored",
            "source": "palace_lumped_port_sheet",
        },
    }


def test_port_regions_keep_validation_error_order() -> None:
    """Reject a bad source kind before later missing field validation."""
    with pytest.raises(ValueError, match="currently supports only"):
        port_sheet_regions_from_stack_metadata(
            {"port_sheet_source_layers": ({"source": "other"},)},
            polygons_by_layer={},
            host_entities=(),
            host_polygons=(),
        )


def test_port_regions_keep_selector_and_occurrence_polygon_paths() -> None:
    """Retain indexed GDS selection and occurrence-authored source geometry."""
    first = gdstk.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=9)
    second = gdstk.Polygon([(3, 0), (4, 0), (4, 1), (3, 1)], layer=9)
    common = {
        "source": "palace_lumped_port_sheet",
        "layer": 9,
        "datatype": 0,
        "name": "selected",
        "port_index": 1,
        "target_layer": "M5",
        "direction": (1, 0, 0),
    }
    selected = port_sheet_regions_from_stack_metadata(
        {"port_sheet_source_layers": ({**common, "selector_point_um": (3.5, 0.5)},)},
        polygons_by_layer={(9, 0): (first, second)},
        host_entities=(),
        host_polygons=(),
    )
    assert selected[0].source_polygon_id == "PORT_SHEET_9_0__P0001"

    occurrence = port_sheet_regions_from_stack_metadata(
        {
            "port_sheet_source_layers": (
                {
                    **common,
                    "source_occurrence_path": "top/child",
                    "source_occurrence_polygon_um": {
                        "exterior": ((5, 0), (6, 0), (6, 1), (5, 1)),
                        "holes": (),
                    },
                },
            )
        },
        polygons_by_layer={},
        host_entities=(),
        host_polygons=(),
    )
    assert occurrence[0].source_polygon_id == "PORT_SHEET__selected__SOURCE__P0000"
    assert occurrence[0].metadata["source_occurrence_path"] == "top/child"


def test_occurrence_port_overlaps_only_owned_and_contributed_hosts() -> None:
    """Exclude same-XY sibling hosts while retaining nested and Ground sources."""
    occurrence_path = "top/intended"
    ring = ((0, 0), (2, 0), (2, 1), (0, 1))
    sources = (
        (
            "intended",
            "intended-poly",
            {"source_occurrence_path": occurrence_path},
            {},
        ),
        (
            "unrelated",
            "unrelated-poly",
            {"source_occurrence_path": "top/sibling"},
            {},
        ),
        (
            "nested",
            "nested-poly",
            {"source_occurrence_path": f"{occurrence_path}/child"},
            {},
        ),
        (
            "ground-carrier",
            "ground-poly",
            {
                "ground_plane_contribution_sources": (
                    {"source_occurrence_path": occurrence_path},
                )
            },
            {"geometry_source": "die_face_minus_ground_mask"},
        ),
    )
    entities = tuple(
        SemanticEntitySpec(
            semantic_id,
            "conductor",
            "Cu",
            "conductor",
            0,
            "polygon",
            polygon_ids=(polygon_id,),
            geometry={"gds_layer": 5, "gds_datatype": 0, **geometry},
            metadata={"logical_layer_id": "M5", **metadata},
        )
        for semantic_id, polygon_id, metadata, geometry in sources
    )
    host_polygons = tuple(
        LayoutPolygonSpec(polygon_id, "5/0", ring) for _, polygon_id, _, _ in sources
    )
    port_record = {
        "source": "palace_lumped_port_sheet",
        "layer": 9,
        "datatype": 0,
        "name": "port",
        "port_index": 1,
        "target_layer": "M5",
        "direction": (1, 0, 0),
        "source_occurrence_path": occurrence_path,
        "source_occurrence_polygon_um": {"exterior": ring, "holes": ()},
    }

    region = port_sheet_regions_from_stack_metadata(
        {"port_sheet_source_layers": (port_record,)},
        polygons_by_layer={},
        host_entities=entities,
        host_polygons=host_polygons,
    )[0]

    assert tuple(asdict(overlap) for overlap in region.overlaps) == (
        {
            "overlap_id": "PORT_SHEET_OVERLAP__PORT_SHEET__port__0000__intended__0000",
            "port_sheet_id": "PORT_SHEET__port__0000",
            "port_polygon_id": "PORT_SHEET__port__SOURCE__P0000",
            "host_semantic_id": "intended",
            "host_polygon_id": "intended-poly",
            "overlap_loop": ((2.0, 1.0), (0.0, 1.0), (0.0, 0.0), (2.0, 0.0)),
            "operation": "local_fragment_required",
            "metadata": {"overlap_polygon_index": 0},
        },
        {
            "overlap_id": "PORT_SHEET_OVERLAP__PORT_SHEET__port__0000__nested__0001",
            "port_sheet_id": "PORT_SHEET__port__0000",
            "port_polygon_id": "PORT_SHEET__port__SOURCE__P0000",
            "host_semantic_id": "nested",
            "host_polygon_id": "nested-poly",
            "overlap_loop": ((2.0, 1.0), (0.0, 1.0), (0.0, 0.0), (2.0, 0.0)),
            "operation": "local_fragment_required",
            "metadata": {"overlap_polygon_index": 0},
        },
        {
            "overlap_id": "PORT_SHEET_OVERLAP__PORT_SHEET__port__0000__ground-carrier__0002",
            "port_sheet_id": "PORT_SHEET__port__0000",
            "port_polygon_id": "PORT_SHEET__port__SOURCE__P0000",
            "host_semantic_id": "ground-carrier",
            "host_polygon_id": "ground-poly",
            "overlap_loop": ((2.0, 1.0), (0.0, 1.0), (0.0, 0.0), (2.0, 0.0)),
            "operation": "local_fragment_required",
            "metadata": {"overlap_polygon_index": 0},
        },
    )


def test_source_port_modules_have_no_import_cycle() -> None:
    """Keep adapter, grid and junction source imports acyclic."""
    dependencies = _source_import_dependencies()
    _assert_acyclic_imports(dependencies)


def _source_import_dependencies() -> dict[str, set[str]]:
    module_paths = {
        "adapter": Path("src/scgsim/geometry/source/adapter.py"),
        "adapter_records": Path("src/scgsim/geometry/source/adapter_records.py"),
        "occurrence_validation": Path(
            "src/scgsim/geometry/source/occurrence_validation.py"
        ),
        "grid": Path("src/scgsim/geometry/source/grid.py"),
        "junctions": Path("src/scgsim/geometry/source/junctions.py"),
        "ports": Path("src/scgsim/geometry/source/ports.py"),
    }
    dependencies = {name: set() for name in module_paths}
    for name, path in module_paths.items():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            imported = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for module in imported:
                for dependency in module_paths:
                    qualified_name = f"scgsim.geometry.source.{dependency}"
                    if module == qualified_name or module.startswith(
                        qualified_name + "."
                    ):
                        dependencies[name].add(dependency)

    return dependencies


def _assert_acyclic_imports(dependencies: dict[str, set[str]]) -> None:
    def visit(name: str, active: set[str], finished: set[str]) -> None:
        if name in active:
            raise AssertionError(f"source import cycle includes {name}")
        if name in finished:
            return
        active.add(name)
        for dependency in dependencies[name]:
            visit(dependency, active, finished)
        active.remove(name)
        finished.add(name)

    finished: set[str] = set()
    for module in dependencies:
        visit(module, set(), finished)
