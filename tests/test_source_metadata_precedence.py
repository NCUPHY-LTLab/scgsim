"""Keep stack, caller and fixed adapter metadata ownership separate from planes."""

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import gdstk
import pytest

from scgsim.geometry import build_gds_stack_geometry_input


def _native_source(
    folder: Path, *, plane_count: int, schema: int = 2
) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("METADATA_GROUND")
    cell.add(gdstk.rectangle((0, 0), (10, 10), layer=99))
    count = plane_count or 1
    for index in range(count):
        start = 2 + 4 * index
        cell.add(gdstk.rectangle((start, 2), (start + 2, 4), layer=index + 1))
    gds = folder / "ground.gds"
    library.write_gds(gds, timestamp=datetime(2000, 1, 1))
    layers = [_source_layer(index, derived=bool(plane_count)) for index in range(count)]
    stack = {
        "layers": layers,
        "materials": {"PEC": {"kind": "conductor"}, "Si": {"kind": "dielectric"}},
        "solution_regions": {
            "substrate": {
                "material_id": "Si",
                "geometry": {"z_um": -1, "thickness_um": 1},
            }
        },
        "metadata": {
            "component_semantics_schema_version": schema,
            "top_cell_name": cell.name,
            "same_key": "stack",
            "stack_note": {"authored": [1, 2]},
            "gds_sha256": hashlib.sha256(gds.read_bytes()).hexdigest(),
            "source_occurrences": [
                {
                    "path": "coupon",
                    "parent_path": None,
                    "cell_name": cell.name,
                    "reference_index": None,
                    "transform": [1, 0, 0, 1, 0, 0],
                }
            ],
        },
    }
    path = folder / "ground.json"
    path.write_text(json.dumps(stack), encoding="utf-8")
    return gds, path


def _source_layer(index: int, *, derived: bool) -> dict[str, object]:
    start = 2 + 4 * index
    geometry = {"z_um": 2 * index, "thickness_um": 1}
    source_geometry: dict[str, object] = dict(geometry)
    if derived:
        source_geometry.update(
            geometry_source="die_face_minus_ground_mask", plane_bounds_ref="substrate"
        )
    else:
        source_geometry["source_occurrence_polygons_um"] = [
            {
                "exterior": [[start, 2], [start + 2, 2], [start + 2, 4], [start, 4]],
                "holes": [],
            }
        ]
    return {
        "semantic_id": f"ground_{index}",
        "role": "conductor",
        "material_id": "PEC",
        "layer": index + 1,
        "datatype": 0,
        "z_um": 2 * index,
        "thickness_um": 1,
        "geometry": source_geometry,
        "metadata": {
            "logical_layer_id": f"metal_{index}",
            "source_occurrence_path": "coupon",
            "same_key": f"plane_{index}",
            "plane_note": index,
        },
    }


@pytest.mark.parametrize("plane_count", (1, 2))
@pytest.mark.parametrize(
    "caller", (None, {}, {"same_key": "caller", "caller_note": {"nested": [3, 4]}})
)
def test_derived_planes_preserve_global_metadata_precedence(
    tmp_path: Path, plane_count: int, caller: Mapping[str, object] | None
) -> None:
    """Plane metadata stays on its own entity for absent, empty and authored callers."""
    gds, stack = _native_source(tmp_path, plane_count=plane_count)
    before = deepcopy(caller)
    value = build_gds_stack_geometry_input(
        gds_file=gds, stack_file=stack, metadata=caller
    )
    assert caller == before
    assert value.metadata["same_key"] == (caller or {}).get("same_key", "stack")
    assert value.metadata["stack_note"] == {"authored": [1, 2]}
    assert value.metadata.get("caller_note") == (caller or {}).get("caller_note")
    assert "plane_note" not in value.metadata
    assert "logical_layer_id" not in value.metadata
    for index in range(plane_count):
        plane = next(
            entity
            for entity in value.entities
            if entity.semantic_id == f"ground_{index}"
        )
        assert plane.metadata["same_key"] == f"plane_{index}"
        assert plane.metadata["plane_note"] == index


@pytest.mark.parametrize(("schema", "plane_count"), ((1, 1), (1, 0), (2, 0), (2, 2)))
def test_fixed_fields_win_and_benign_metadata_preserves_geometry(
    tmp_path: Path, schema: int, plane_count: int
) -> None:
    """Keep fixed adapter facts authoritative and geometry independent of caller notes."""
    gds, stack = _native_source(tmp_path, plane_count=plane_count, schema=schema)
    plain = build_gds_stack_geometry_input(gds_file=gds, stack_file=stack)
    caller = {
        "same_key": "caller",
        "caller_note": {"nested": [3, 4]},
        "adapter": "caller",
        "gds_file": "caller",
        "stack_file": "caller",
        "selected_cell_name": "caller",
        "cell_bounds_um": "caller",
        "source_dbu_um": "caller",
    }
    before = deepcopy(caller)
    value = build_gds_stack_geometry_input(
        gds_file=gds, stack_file=stack, metadata=caller
    )
    assert caller == before
    assert value.metadata["same_key"] == "caller"
    assert value.metadata["caller_note"] == caller["caller_note"]
    assert value.metadata["adapter"] == "gds_stack"
    assert value.metadata["gds_file"] == str(gds)
    assert value.metadata["stack_file"] == str(stack)
    assert value.metadata["selected_cell_name"] == "METADATA_GROUND"
    assert value.metadata["cell_bounds_um"] == plain.metadata["cell_bounds_um"]
    assert value.metadata["source_dbu_um"] == plain.metadata["source_dbu_um"]
    assert value.polygons == plain.polygons
    assert value.entities == plain.entities
    assert value.solution_regions == plain.solution_regions
    assert value.port_sheet_regions == plain.port_sheet_regions
    assert value.boundary_curves == plain.boundary_curves
    assert value.boundary_reconstruction == plain.boundary_reconstruction
    assert value.lumped_supports == plain.lumped_supports
    assert (
        value.metadata["unquantized_source_sha256"]
        == plain.metadata["unquantized_source_sha256"]
    )
