"""Offline source recipe equivalence and ownership, without solver/mesh work."""

import json
from contextlib import contextmanager
from copy import deepcopy
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
from scgsim.geometry.primitives.grid import (
    geometry_grid_scope,
    source_grid_audits,
    source_recipe_map,
)
from scgsim.geometry.source import occurrence as occurrence
from scgsim.geometry.source import plan as plan_module
from scgsim.preparation import preparation_resources


@pytest.fixture
def native_source():
    """Build native source geometry used by serial and parallel recipes."""
    cell = gdstk.Cell("SOURCE")
    cell.add(gdstk.rectangle((0, 0), (2, 2), layer=1))
    cell.add(gdstk.rectangle((4, 0), (6, 2), layer=1))
    return cell


def entity(name, point):
    """Create a conductor selector record at a world-frame point."""
    return occurrence.entity_from_layer_record(
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
    """Preserve recipe output, audit order, and caller-thread extraction."""

    # Source selectors are world-frame facts; both execution paths use this transform.
    def prepared_jobs():
        for index, point in enumerate(((11, 21), (15, 21))):
            yield occurrence.source_recipe_job(
                occurrence.occurrence_polygons_for_entity,
                entity(f"metal_{index}", point),
                source_cell=native_source,
                transform=(1, 0, 0, 1, 10, 20),
            )

    caller = get_ident()
    extraction_threads = []
    # White-box observation verifies worker recipes reuse caller-thread extraction.
    original = occurrence._extract_polygons_by_layer  # noqa: SLF001 - observe private extraction boundary

    def extract(*args, **kwargs):
        extraction_threads.append(get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(occurrence, "_extract_polygons_by_layer", extract)
    with geometry_grid_scope(GeometryGridContext(0.001)):
        serial = source_recipe_map(occurrence.run_source_recipe, prepared_jobs())
        serial_audits = deepcopy(source_grid_audits())
    with geometry_grid_scope(GeometryGridContext(0.001)):
        with preparation_resources(max_workers=2):
            parallel = source_recipe_map(occurrence.run_source_recipe, prepared_jobs())
        parallel_audits = source_grid_audits()
    assert parallel == serial
    assert parallel_audits == serial_audits
    assert [row["source_entity_id"] for row in parallel_audits] == [
        "metal_0",
        "metal_1",
    ]
    assert set(extraction_threads) == {caller}


def test_job_native_objects_and_record_inputs_are_isolated(native_source):
    """Give parallel recipes detached records and native polygon copies."""
    original_points = [p.points.tolist() for p in native_source.polygons]
    record = {"nested": [1]}

    def mutate(value, *, cell, polygons_by_layer):
        assert cell is None
        value["nested"].append(2)
        polygons_by_layer[(1, 0)][0].translate(100, 0)
        return value, polygons_by_layer[(1, 0)][0].points.tolist()

    def make_jobs():
        for _ in range(2):
            yield occurrence.source_recipe_job(
                mutate, record, source_cell=native_source
            )

    with preparation_resources(max_workers=2):
        results = source_recipe_map(occurrence.run_source_recipe, make_jobs())
    expected_points = [[x + 100, y] for x, y in original_points[0]]
    assert results == [({"nested": [1, 2]}, expected_points)] * 2
    assert record == {"nested": [1]}
    assert [p.points.tolist() for p in native_source.polygons] == original_points


def refused_after_selection(value, **kwargs):
    """Select genuinely, then refuse, so the selection's audit is kept before the error."""
    occurrence.occurrence_polygons_for_entity(value, **kwargs)
    raise ValueError(f"source mismatch {value.semantic_id[-1]}")


def test_first_recipe_failure_keeps_audit_prefix_and_allows_clean_next_map(
    native_source,
):
    """Retain source-order failure and audit behavior across the next map."""

    def invalid_jobs():
        for index, point in enumerate(((1, 1), (99, 99))):
            yield occurrence.source_recipe_job(
                refused_after_selection,
                entity(f"metal_{index}", point),
                source_cell=native_source,
                transform=(1, 0, 0, 1, 0, 0),
            )

    errors = []

    def observed_run(job):
        try:
            return occurrence.run_source_recipe(job)
        except ValueError as error:
            errors.append(str(error))
            raise

    with geometry_grid_scope(GeometryGridContext(0.001)):
        with preparation_resources(max_workers=2):
            with pytest.raises(ValueError, match="source mismatch 0"):
                source_recipe_map(observed_run, invalid_jobs())
            assert "source mismatch 0" in errors
            assert len(errors) == 2
            assert any("source mismatch" not in error for error in errors)
            assert [r["source_entity_id"] for r in source_grid_audits()] == ["metal_0"]
            result = source_recipe_map(
                occurrence.run_source_recipe,
                [
                    occurrence.source_recipe_job(
                        occurrence.occurrence_polygons_for_entity,
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
    """Match serial and parallel outputs for ground, include, and port recipes."""
    cell = gdstk.Cell("GROUND_SOURCE")
    cell.add(gdstk.rectangle((1, 1), (2, 2), layer=1))
    cell.add(gdstk.rectangle((0, 0), (4, 4), layer=2))
    cell.add(gdstk.rectangle((5, 0), (6, 1), layer=3))
    cell.add(gdstk.rectangle((0, 0), (1, 1), layer=4))
    ground = occurrence.entity_from_layer_record(
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
        yield occurrence.source_recipe_job(
            occurrence.occurrence_ground_polygons_for_entity,
            ground,
            source_cell=cell,
            transform=(1, 0, 0, 1, 0, 0),
            plane_layer=(2, 0),
        )
        yield occurrence.source_recipe_job(
            occurrence.occurrence_include_polygon,
            {
                "source_local_id": "include",
                "layer": [3, 0],
                "selector_point_um": [5.5, 0.5],
            },
            source_cell=cell,
            transform=(1, 0, 0, 1, 0, 0),
        )
        yield occurrence.source_recipe_job(
            occurrence.occurrence_port_polygon,
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
        serial = source_recipe_map(occurrence.run_source_recipe, prepared_jobs())
        serial_audits = deepcopy(source_grid_audits())
    with geometry_grid_scope(GeometryGridContext(0.001)):
        with preparation_resources(max_workers=2):
            parallel = source_recipe_map(occurrence.run_source_recipe, prepared_jobs())
        assert source_grid_audits() == serial_audits
    assert parallel == serial
    assert len(parallel[0][0]["holes"]) == 1


def test_public_adapter_exact_metadata_and_audit_order(tmp_path):
    """Preserve exact adapter metadata and source audit ordering."""
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
    """Compare detached plan snapshots from serial and parallel preparation."""
    plan, plan_module, emitted, emitted_bytes = _build_plan_fixture(
        tmp_path, monkeypatch
    )
    serial, parallel, worker_threads = _prepare_serial_and_parallel_plan(
        plan, plan_module, monkeypatch
    )
    _assert_parallel_snapshot(serial, parallel, emitted_bytes, worker_threads)
    _assert_error_priority_and_recovery(serial, emitted, tmp_path)


def _build_plan_fixture(tmp_path, monkeypatch):
    """Create a public plan and freeze its genuine GDS fixture bytes."""
    # Keep optional gdsfactory loading inside the public-plan fixture test path.
    import gdsfactory as gf  # noqa: PLC0415 - keep optional dependency local to this fixture

    # Restore process-local PDK state after this public generic fixture.
    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 - preserve active PDK during test setup
    )
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
    return plan, plan_module, emitted, emitted_bytes


def _prepare_serial_and_parallel_plan(plan, plan_module, monkeypatch):
    """Prepare serially, then observe genuine selector work on worker threads."""
    serial = plan.prepare()
    barrier = Barrier(2)
    worker_threads = []
    original_selector = plan_module.occurrence_polygons_for_entity

    def observed_selector(*args, **kwargs):
        # Observe genuine selectors, without substituting their outputs.
        if kwargs.get("cell") is None:
            barrier.wait(timeout=5)
            worker_threads.append(get_ident())
        return original_selector(*args, **kwargs)

    with monkeypatch.context() as watched:
        watched.setattr(
            plan_module, "occurrence_polygons_for_entity", observed_selector
        )
        with preparation_resources(max_workers=2):
            parallel = plan.prepare()
    return serial, parallel, worker_threads


def _assert_parallel_snapshot(serial, parallel, emitted_bytes, worker_threads):
    """Check thread use, exact plan equality, and detached metadata."""
    assert len(worker_threads) == 2  # One extraction per occurrence Entity.
    assert len(set(worker_threads)) == 2
    assert get_ident() not in worker_threads
    assert parallel.gds_bytes == serial.gds_bytes == emitted_bytes
    assert parallel.stack == serial.stack
    assert parallel.source_occurrences == serial.source_occurrences
    assert parallel.geometry_input == serial.geometry_input
    copied = parallel.geometry_input
    copied.metadata["detached"] = True
    assert "detached" not in parallel.geometry_input.metadata


def _assert_error_priority_and_recovery(serial, emitted, tmp_path):
    """Refuse the saved stack with another GDS, then read the saved pair as prepare() lowered it."""
    saved_stack = tmp_path / "saved.stack.json"
    other_gds = tmp_path / "other.gds"
    other_gds.write_bytes(emitted.read_bytes() + b"\0")
    saved_stack.write_text(json.dumps(serial.stack))
    with preparation_resources(max_workers=2):
        with pytest.raises(ValueError, match="written for the GDS with SHA-256"):
            build_gds_stack_geometry_input(gds_file=other_gds, stack_file=saved_stack)
        assert occurrence._EXTRACTION_CACHE.get() is None  # noqa: SLF001 - verify failed adapter scope cleanup
        reloaded = build_gds_stack_geometry_input(
            gds_file=emitted, stack_file=saved_stack
        )
    prepared = serial.geometry_input

    def shapes(source):
        return [(p.polygon_id, p.layer, p.exterior, p.holes) for p in source.polygons]

    assert shapes(reloaded) == shapes(prepared)
    assert [e.semantic_id for e in reloaded.entities] == [
        e.semantic_id for e in prepared.entities
    ]
