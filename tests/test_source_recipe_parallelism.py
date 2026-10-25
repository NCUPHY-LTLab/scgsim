"""Offline source recipe equivalence and ownership, without solver/mesh work."""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from threading import Barrier, get_ident
from types import SimpleNamespace

import gdstk
import pytest

from scgsim.geometry import (
    GeometryPlan,
    PhysicalLayerDeclaration,
    build_gds_stack_geometry_input,
)
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry._primitives.grid import (
    geometry_grid_scope,
    source_grid_audits,
    _source_recipe_map,
)
from scgsim.geometry.source import _occurrence as occurrence
from scgsim.preparation import preparation_resources


@pytest.fixture
def native_source():
    cell = gdstk.Cell("SOURCE")
    cell.add(gdstk.rectangle((0, 0), (2, 2), layer=1))
    cell.add(gdstk.rectangle((4, 0), (6, 2), layer=1))
    return cell


def entity(name, point):
    return occurrence._entity_from_layer_record(
        {
            "semantic_id": name,
            "role": "metal",
            "material_id": "PEC",
            "layer": 1,
            "datatype": 0,
            "z_um": 0,
            "thickness_um": 1,
            "geometry": {
                "selector_point_um": point,
                "route_ab_fused_selector_mode": True,
            },
        },
        materials={"PEC": {"kind": "conductor"}},
    )


def test_recipe_results_context_audits_and_serial_extraction_match(
    native_source, monkeypatch
):
    # Source selectors are world-frame facts; both execution paths use this transform.
    def prepared_jobs():
        for index, point in enumerate(((11, 21), (15, 21))):
            yield occurrence._source_recipe_job(
                occurrence._occurrence_polygons_for_entity,
                entity(f"metal_{index}", point),
                source_cell=native_source,
                transform=(1, 0, 0, 1, 10, 20),
            )

    caller = get_ident()
    extraction_threads = []
    original = occurrence._extract_polygons_by_layer

    def extract(*args, **kwargs):
        extraction_threads.append(get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(occurrence, "_extract_polygons_by_layer", extract)
    with geometry_grid_scope(GeometryGridContext(0.001)):
        serial = _source_recipe_map(occurrence._run_source_recipe, prepared_jobs())
        serial_audits = deepcopy(source_grid_audits())
    with geometry_grid_scope(GeometryGridContext(0.001)):
        with preparation_resources(max_workers=2):
            parallel = _source_recipe_map(
                occurrence._run_source_recipe, prepared_jobs()
            )
        parallel_audits = source_grid_audits()
    assert parallel == serial
    assert parallel_audits == serial_audits
    assert [row["source_entity_id"] for row in parallel_audits] == [
        "metal_0",
        "metal_1",
    ]
    assert set(extraction_threads) == {caller}


def test_job_native_objects_and_record_inputs_are_isolated(native_source):
    original_points = [p.points.tolist() for p in native_source.polygons]
    record = {"nested": [1]}

    def mutate(value, *, cell, polygons_by_layer):
        assert cell is None
        value["nested"].append(2)
        polygons_by_layer[(1, 0)][0].translate(100, 0)
        return value, polygons_by_layer[(1, 0)][0].points.tolist()

    def make_jobs():
        for _ in range(2):
            yield occurrence._source_recipe_job(
                mutate, record, source_cell=native_source
            )

    with preparation_resources(max_workers=2):
        results = _source_recipe_map(occurrence._run_source_recipe, make_jobs())
    expected_points = [[x + 100, y] for x, y in original_points[0]]
    assert results == [({"nested": [1, 2]}, expected_points)] * 2
    assert record == {"nested": [1]}
    assert [p.points.tolist() for p in native_source.polygons] == original_points


def test_first_comparison_failure_keeps_audit_prefix_and_allows_clean_next_map(
    native_source,
):
    def invalid_jobs():
        for index, point in enumerate(((1, 1), (99, 99))):
            yield occurrence._source_recipe_job(
                occurrence._occurrence_polygons_for_entity,
                entity(f"metal_{index}", point),
                source_cell=native_source,
                transform=(1, 0, 0, 1, 0, 0),
                expected=[],
                comparison_error=f"source mismatch {index}",
            )

    errors = []

    def observed_run(job):
        try:
            return occurrence._run_source_recipe(job)
        except ValueError as error:
            errors.append(str(error))
            raise

    with geometry_grid_scope(GeometryGridContext(0.001)):
        with preparation_resources(max_workers=2):
            with pytest.raises(ValueError, match="source mismatch 0"):
                _source_recipe_map(observed_run, invalid_jobs())
            assert "source mismatch 0" in errors
            assert len(errors) == 2
            assert any("source mismatch" not in error for error in errors)
            assert [r["source_entity_id"] for r in source_grid_audits()] == ["metal_0"]
            result = _source_recipe_map(
                occurrence._run_source_recipe,
                [
                    occurrence._source_recipe_job(
                        occurrence._occurrence_polygons_for_entity,
                        entity("next", (1, 1)),
                        source_cell=native_source,
                        transform=(1, 0, 0, 1, 0, 0),
                    )
                ],
            )
            assert result
            assert [r["source_entity_id"] for r in source_grid_audits()] == [
                "metal_0",
                "next",
            ]


def test_ground_include_and_port_recipe_outputs_match():
    cell = gdstk.Cell("GROUND_SOURCE")
    cell.add(gdstk.rectangle((1, 1), (2, 2), layer=1))
    cell.add(gdstk.rectangle((0, 0), (4, 4), layer=2))
    cell.add(gdstk.rectangle((5, 0), (6, 1), layer=3))
    cell.add(gdstk.rectangle((0, 0), (1, 1), layer=4))
    ground = occurrence._entity_from_layer_record(
        {
            "semantic_id": "ground",
            "role": "metal",
            "material_id": "PEC",
            "layer": 1,
            "datatype": 0,
            "geometry": {
                "z_um": 0,
                "thickness_um": 1,
                "geometry_source": "die_face_minus_ground_mask",
            },
        },
        materials={"PEC": {"kind": "conductor"}},
    )

    def prepared_jobs():
        yield occurrence._source_recipe_job(
            occurrence._occurrence_ground_polygons_for_entity,
            ground,
            source_cell=cell,
            transform=(1, 0, 0, 1, 0, 0),
            plane_layer=(2, 0),
        )
        yield occurrence._source_recipe_job(
            occurrence._occurrence_include_polygon,
            {
                "source_local_id": "include",
                "layer": [3, 0],
                "selector_point_um": [5.5, 0.5],
            },
            source_cell=cell,
            transform=(1, 0, 0, 1, 0, 0),
        )
        yield occurrence._source_recipe_job(
            occurrence._occurrence_port_polygon,
            {
                "name": "port",
                "layer": 4,
                "datatype": 0,
                "selector_point_um": [0.5, 0.5],
            },
            source_cell=cell,
            transform=(1, 0, 0, 1, 0, 0),
        )

    with geometry_grid_scope(GeometryGridContext(0.001)):
        serial = _source_recipe_map(occurrence._run_source_recipe, prepared_jobs())
        serial_audits = deepcopy(source_grid_audits())
    with geometry_grid_scope(GeometryGridContext(0.001)):
        with preparation_resources(max_workers=2):
            parallel = _source_recipe_map(
                occurrence._run_source_recipe, prepared_jobs()
            )
        assert source_grid_audits() == serial_audits
    assert parallel == serial
    assert len(parallel[0][0]["holes"]) == 1


def test_public_adapter_exact_metadata_and_audit_order(tmp_path):
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("ADAPTER")
    cell.add(gdstk.rectangle((0, 0), (2, 2), layer=1))
    cell.add(gdstk.rectangle((4, 0), (6, 2), layer=2))
    gds = tmp_path / "source.gds"
    library.write_gds(gds)
    stack = tmp_path / "stack.json"
    stack.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "semantic_id": "first",
                        "layer": 1,
                        "datatype": 0,
                        "role": "metal",
                        "material_id": "PEC",
                        "z_um": 0,
                        "thickness_um": 1,
                    },
                    {
                        "semantic_id": "second",
                        "layer": 2,
                        "datatype": 0,
                        "role": "metal",
                        "material_id": "PEC",
                        "z_um": 0,
                        "thickness_um": 1,
                    },
                ],
                "materials": {"PEC": {"kind": "conductor"}},
                "solution_regions": {},
            }
        )
    )
    serial = build_gds_stack_geometry_input(gds_file=gds, stack_file=stack)
    with preparation_resources(max_workers=2):
        parallel = build_gds_stack_geometry_input(gds_file=gds, stack_file=stack)
    assert parallel == serial
    assert parallel.metadata == serial.metadata
    assert [
        row["source_entity_id"]
        for row in parallel.metadata["geometry_grid_source_audits"]
    ] == ["first", "second"]


def test_public_plan_exact_detached_snapshot_for_identical_emitted_fixture(
    tmp_path, monkeypatch
):
    import gdsfactory as gf
    from scgsim.geometry.source import plan as plan_module

    # Restore process-local PDK state after this public generic fixture.
    monkeypatch.setattr(gf.pdk, "_ACTIVE_PDK", gf.pdk._ACTIVE_PDK)
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon(((0, 0), (2, 0), (2, 2), (0, 2)), layer=(1, 0))
    component.add_polygon(((4, 0), (6, 0), (6, 2), (4, 2)), layer=(2, 0))
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {"semantic_id": "first", "level": "metal", "gds_layer": [1, 0]},
            {"semantic_id": "second", "level": "metal", "gds_layer": [2, 0]},
        ],
    }

    def level(name, material, interval, role):
        declaration = PhysicalLayerDeclaration(
            name, name, {"level": name}, interval, material, "finite_volume", {}, {}
        )
        return SimpleNamespace(
            material=material,
            zmin=interval[0],
            thickness=interval[1] - interval[0],
            mesh_order=1,
            info={
                "simulation_role": role,
                "include_in_component_simulation": True,
                "part_role": "signal",
                "physical_layer": declaration.to_payload(),
            },
        )

    stack = SimpleNamespace(
        layers={
            "VACUUM_REGION": level(
                "VACUUM_REGION", "vacuum", (-1, 2), "solution_region"
            ),
            "metal": level("metal", "PEC", (0, 1), "conductor"),
        }
    )
    plan = GeometryPlan(
        component,
        root_id="coupon",
        layer_stack=stack,
        material_records={"PEC": {"kind": "conductor"}, "vacuum": {"kind": "vacuum"}},
        coupon_padding_um=1,
        grid_size_um=0.001,
    )
    # Freeze genuine fixture emission bytes/path for full metadata equality;
    # this does not normalize timestamps or substitute geometry in product code.
    emitted = tmp_path / "fixture.gds"
    component.write_gds(emitted)
    emitted_bytes = emitted.read_bytes()
    original_write = type(component).write_gds

    def fixed_fixture_write(self, path, *args, **kwargs):
        if self is component:
            Path(path).write_bytes(emitted_bytes)
            return Path(path)
        return original_write(self, path, *args, **kwargs)

    @contextmanager
    def fixed_fixture_directory(**kwargs):
        directory = tmp_path / "source-inputs"
        directory.mkdir(exist_ok=True)
        yield str(directory)

    monkeypatch.setattr(type(component), "write_gds", fixed_fixture_write)
    monkeypatch.setattr(plan_module, "TemporaryDirectory", fixed_fixture_directory)
    serial = plan.prepare()
    from scgsim.geometry.source import adapter as adapter_module

    barrier = Barrier(2)
    worker_threads = []
    original_selector = plan_module._occurrence_polygons_for_entity

    def observed_selector(*args, **kwargs):
        # Observe genuine selectors, without substituting their outputs.
        if kwargs.get("cell") is None:
            barrier.wait(timeout=5)
            worker_threads.append(get_ident())
        return original_selector(*args, **kwargs)

    with monkeypatch.context() as watched:
        watched.setattr(
            plan_module, "_occurrence_polygons_for_entity", observed_selector
        )
        watched.setattr(
            adapter_module, "_occurrence_polygons_for_entity", observed_selector
        )
        with preparation_resources(max_workers=2):
            parallel = plan.prepare()
    assert (
        len(worker_threads) == 4
    )  # Plan extraction plus adapter expected-value batch.
    assert len(set(worker_threads)) == 2
    assert get_ident() not in worker_threads
    assert parallel.gds_bytes == serial.gds_bytes == emitted_bytes
    assert parallel.stack == serial.stack
    assert parallel.source_occurrences == serial.source_occurrences
    assert parallel.geometry_input == serial.geometry_input
    copied = parallel.geometry_input
    copied.metadata["detached"] = True
    assert "detached" not in parallel.geometry_input.metadata

    # Keep current expected-record checks and source-order error priority.
    saved_stack = tmp_path / "saved.stack.json"
    broken = deepcopy(serial.stack)
    broken["layers"][0]["geometry"]["source_occurrence_polygons_um"] = []
    broken["layers"][1]["metadata"]["source_occurrence_path"] = "missing"
    saved_stack.write_text(json.dumps(broken))
    with preparation_resources(max_workers=2):
        with pytest.raises(
            ValueError, match="polygon source differs for 'coupon/first'"
        ):
            build_gds_stack_geometry_input(gds_file=emitted, stack_file=saved_stack)
        assert occurrence._EXTRACTION_CACHE.get() is None
        saved_stack.write_text(json.dumps(serial.stack))
        assert build_gds_stack_geometry_input(
            gds_file=emitted, stack_file=saved_stack
        ).polygons
