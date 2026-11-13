"""Real native GDS regressions for the public source-adapter pipeline."""

import hashlib
import inspect
import json
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import gdstk
import pytest

from scgsim.geometry import build_gds_stack_geometry_input
from scgsim.preparation import preparation_resources


def _layer(name: str, number: int, **updates: Any) -> dict[str, Any]:
    record = {
        "semantic_id": name,
        "role": "conductor",
        "material_id": "PEC",
        "layer": number,
        "datatype": 0,
        "z_um": 0,
        "thickness_um": 1,
    }
    record.update(updates)
    return record


def _write_stack(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def native_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Write genuine holed, disconnected, curved and port source geometry."""
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("ADAPTER_PIPELINE")
    holed = gdstk.boolean(
        gdstk.rectangle((0, 0), (8, 8)),
        gdstk.rectangle((2, 2), (4, 4)),
        "not",
        precision=1e-9,
    )[0]
    holed.layer = 1
    cell.add(holed)
    cell.add(gdstk.rectangle((10, 0), (12, 2), layer=2))
    cell.add(gdstk.rectangle((14, 0), (16, 2), layer=2))
    cell.add(gdstk.ellipse((20, 4), 2, tolerance=0.001, layer=3))
    cell.add(gdstk.rectangle((0.5, 0.5), (1.5, 1.5), layer=9))
    gds = tmp_path / "pipeline.gds"
    library.write_gds(gds, timestamp=datetime(2000, 1, 1))
    stack = {
        "layers": [
            _layer(
                "metal",
                1,
                geometry={
                    "z_um": 0,
                    "thickness_um": 1,
                    "route_ab_fused_selector_mode": True,
                },
            ),
            _layer(
                "split",
                2,
                geometry={
                    "z_um": 0,
                    "thickness_um": 1,
                    "split_polygons_as_entities": True,
                },
            ),
            _layer("circle", 3),
        ],
        "solution_regions": {},
        "materials": {"PEC": {"kind": "conductor"}},
        "metadata": {
            "top_cell_name": cell.name,
            "same_key": "stack",
            "port_sheet_source_layers": [
                {
                    "source": "palace_lumped_port_sheet",
                    "name": "P1",
                    "layer": 9,
                    "datatype": 0,
                    "port_index": 1,
                    "target_layer": "metal",
                    "direction": [1, 0, 0],
                }
            ],
            "boundary_curves": [
                {
                    "boundary_id": "circle_outer",
                    "entity_id": "circle",
                    "selector_point_um": [20, 4],
                    "curves": [
                        {
                            "kind": "circular_arc",
                            "points_um": [[22, 4], [20, 6], [18, 4]],
                        }
                    ],
                }
            ],
            "boundary_reconstruction": [
                {
                    "boundary_id": "circle_reconstruction",
                    "entity_id": "circle",
                    "selector_point_um": [20, 4],
                    "kind": "bspline",
                    "closed": True,
                    "degree": 3,
                }
            ],
        },
    }
    stack_path = tmp_path / "pipeline.json"
    _write_stack(stack_path, stack)
    return gds, stack_path, stack


def test_public_native_pipeline_preserves_holes_ports_curves_and_metadata(
    native_inputs,
) -> None:
    """Keep complete native results and authored source identity across workers."""
    gds, stack, authored = native_inputs
    before = deepcopy(authored)
    results = []
    for workers in (None, 2):
        scope = (
            nullcontext()
            if workers is None
            else preparation_resources(max_workers=workers)
        )
        with scope:
            results.append(
                build_gds_stack_geometry_input(
                    gds_file=gds,
                    stack_file=stack,
                    metadata={"same_key": "caller", "adapter": "caller"},
                )
            )
    serial, parallel = results
    assert serial == parallel
    assert authored == before
    assert serial.metadata["same_key"] == "caller"
    assert serial.metadata["adapter"] == "gds_stack"
    assert serial.metadata["source_dbu_um"] == 0.001
    assert serial.metadata["selected_cell_name"] == "ADAPTER_PIPELINE"
    assert len(serial.polygons) == 4
    assert (
        len(next(p for p in serial.polygons if p.object_name == "metal_0000").holes)
        == 1
    )
    assert len(serial.port_sheet_regions) == 1
    assert serial.port_sheet_regions[0].metadata["source_name"] == "P1"
    assert len(serial.boundary_curves) == len(serial.boundary_reconstruction) == 1
    assert serial.boundary_curves[0].boundary_id == "circle_outer"
    assert serial.boundary_reconstruction[0].boundary_id == "circle_reconstruction"


@pytest.mark.parametrize("workers", (None, 2))
@pytest.mark.parametrize(
    ("updates", "error", "message"),
    [
        (
            {"layers": None, "solution_regions": None},
            TypeError,
            "stack_file must define sequence 'layers'",
        ),
        (
            {"solution_regions": None, "materials": None},
            TypeError,
            "stack_file must define mapping 'solution_regions'",
        ),
        (
            {"materials": None, "metadata": None},
            TypeError,
            "stack_file must define mapping 'materials'",
        ),
        (
            {"metadata": None},
            TypeError,
            "stack_file 'metadata' must be a mapping when provided",
        ),
    ],
)
def test_stack_container_failures_keep_priority(
    native_inputs, workers, updates, error, message
) -> None:
    """Report the first original structural failure before reading later containers."""
    gds, stack_path, stack = native_inputs
    stack.update(updates)
    _write_stack(stack_path, stack)
    scope = (
        nullcontext() if workers is None else preparation_resources(max_workers=workers)
    )
    with scope, pytest.raises(error) as failure:
        build_gds_stack_geometry_input(gds_file=gds, stack_file=stack_path)
    assert str(failure.value) == message


@pytest.mark.parametrize("workers", (None, 2))
def test_schema_two_identity_failure_precedes_occurrence_validation(
    native_inputs, workers
) -> None:
    """Compare the actual GDS digest before malformed occurrence metadata."""
    gds, stack_path, stack = native_inputs
    stack["metadata"].update(
        component_semantics_schema_version=2,
        gds_sha256="invalid",
        source_occurrences=None,
    )
    _write_stack(stack_path, stack)
    scope = (
        nullcontext() if workers is None else preparation_resources(max_workers=workers)
    )
    with scope, pytest.raises(ValueError) as failure:
        build_gds_stack_geometry_input(gds_file=gds, stack_file=stack_path)
    assert (
        str(failure.value) == "GeometryPlan GDS identity differs from its paired stack"
    )


@pytest.mark.parametrize("version", (1, 3, "2"))
def test_an_unknown_component_semantics_version_is_refused(
    native_inputs, version
) -> None:
    """Only schema 2 carries occurrence lineage; another version is not skipped."""
    gds, stack_path, stack = native_inputs
    stack["metadata"]["component_semantics_schema_version"] = version
    _write_stack(stack_path, stack)
    with pytest.raises(ValueError) as failure:
        build_gds_stack_geometry_input(gds_file=gds, stack_file=stack_path)
    assert str(failure.value) == (
        f"stack metadata component_semantics_schema_version is {version!r}; "
        "only 2, which GeometryPlan writes, is supported"
    )


def test_public_adapter_wrappers_keep_introspection() -> None:
    """The cache and diagnostics scopes retain the public signature and documentation."""
    function = build_gds_stack_geometry_input
    original = inspect.unwrap(function)
    assert function.__name__ == original.__name__ == "build_gds_stack_geometry_input"
    assert function.__doc__ == original.__doc__
    assert inspect.signature(function) == inspect.signature(original)
    wrapped = getattr(function, "__wrapped__", None)
    assert callable(wrapped)
    assert wrapped is not original
    assert set(inspect.signature(function).parameters) == {
        "gds_file",
        "stack_file",
        "top_cell_name",
        "metadata",
        "grid_size_um",
    }


def test_schema_two_real_ground_keeps_geometry_across_workers(tmp_path: Path) -> None:
    """Preserve the genuine die-face subtraction and original source-plane metadata."""
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("GROUND")
    cell.add(gdstk.rectangle((0, 0), (10, 10), layer=99))
    cell.add(gdstk.rectangle((2, 2), (4, 4), layer=1))
    gds = tmp_path / "ground.gds"
    library.write_gds(gds, timestamp=datetime(2000, 1, 1))
    stack = {
        "layers": [
            _layer(
                "ground",
                1,
                geometry={
                    "z_um": 0,
                    "thickness_um": 1,
                    "geometry_source": "die_face_minus_ground_mask",
                    "plane_bounds_ref": "substrate",
                },
                metadata={
                    "logical_layer_id": "metal",
                    "source_occurrence_path": "coupon",
                },
            )
        ],
        "materials": {"PEC": {"kind": "conductor"}, "Si": {"kind": "dielectric"}},
        "solution_regions": {
            "substrate": {
                "material_id": "Si",
                "geometry": {"z_um": -1, "thickness_um": 1},
            }
        },
        "metadata": {
            "component_semantics_schema_version": 2,
            "top_cell_name": "GROUND",
            "gds_sha256": hashlib.sha256(gds.read_bytes()).hexdigest(),
            "source_occurrences": [
                {
                    "path": "coupon",
                    "parent_path": None,
                    "cell_name": "GROUND",
                    "reference_index": None,
                    "transform": [1, 0, 0, 1, 0, 0],
                }
            ],
        },
    }
    path = tmp_path / "ground.json"
    _write_stack(path, stack)
    serial = build_gds_stack_geometry_input(gds_file=gds, stack_file=path)
    with preparation_resources(max_workers=2):
        parallel = build_gds_stack_geometry_input(gds_file=gds, stack_file=path)
    assert parallel == serial
    assert len(serial.polygons) == 1
    assert len(serial.polygons[0].holes) == 1
    assert serial.entities[-1].semantic_id == "ground"
    assert serial.entities[-1].metadata["source_occurrence_path"] == "coupon"
