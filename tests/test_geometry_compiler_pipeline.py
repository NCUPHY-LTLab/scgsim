"""Test compiler orchestration with doubles; native geometry is checked separately."""

import inspect
import json
from dataclasses import dataclass, field, replace

import pytest

from scgsim.geometry.compiler import builder, dispatch
from scgsim.geometry.compiler.stages import StageTimings
from scgsim.geometry.models.construction import (
    ConstructionPlanRecord,
    RouteABConstructionPlanRecord,
)
from scgsim.geometry.models.grid import GeometryGridContext
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
)
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.models.representation import RepresentationContext
from scgsim.geometry.primitives.grid_cache import (  # private-import: exercise the existing build cache boundary through final timing and failure cleanup.
    cached_subdivision,
)

BUILD_STAGES = (
    "validate_geometry_input",
    "write_01_validate_geometry_input",
    "build_route_construction_plan",
    "write_02_build_route_construction_plan",
    "engine_gate_volume_adjacency_conformality",
    "write_engine_gate_volume_adjacency_conformality",
    "assert_engine_gate_volume_adjacency_conformality",
    "write_occ_geometry_from_plan",
    "write_03_build_occ_geometry",
    "write_engine_gate_gmsh_brep_conformality",
    "assert_engine_gate_gmsh_brep_conformality",
    "validate_backend_tag_ledger",
    "export_physical_group_records",
    "write_04_export_physical_groups",
)
POLYGON_STAGES = (
    "validate_selected_route",
    "recognize_route_interfaces",
    "plan_conductor_contact_patches",
    "plan_mm_contact_records",
    "plan_surface_partitions",
    "plan_route_construction_bodies",
    "plan_route_surfaces",
    "reconcile_solution_domain_boundaries",
    "lower_port_sheet_regions",
    "validate_surface_sheet_interface_coverage",
    "merge_solution_sidewall_interfaces",
    "complete_interface_plan_from_surfaces",
    "plan_canonical_topology",
    "validate_route_b_port_sheet_sidewall_topology",
    "validate_curve_plan_coverage",
    "validate_surface_deduplication",
    "validate_no_surface_overlap",
    "plan_route_volumes",
    "validate_route_volume_surface_refs",
    "validate_volume_surface_closure",
    "validate_surface_use_counts",
    "plan_cut_host_operations",
    "plan_route_tags",
    "validate_route_operation_coverage",
    "validate_surface_partition_coverage",
    "validate_interface_surface_source_of_truth",
    "validate_tag_plan_coverage",
)


class _StageFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class _BuildScenario:
    fail_stage: str | None = None
    fail_timing: bool = False
    exception_type: type[BaseException] = _StageFailure


@dataclass
class _BuildObservation:
    events: list[str] = field(default_factory=list)
    timing: dict = field(default_factory=dict)


def _install_builder_stages(monkeypatch, scenario):
    observed = _BuildObservation()
    plan = ConstructionPlanRecord("B", metadata={"source_path": "/private/source.gds"})
    native_report = {"status": "pass", "engine_gate": "native"}
    built = replace(
        plan, metadata={"engine_gate_gmsh_brep_conformality": native_report}
    )

    def called(stage, result):
        observed.events.append(stage)
        if stage == scenario.fail_stage:
            raise scenario.exception_type(stage)
        return result

    def operation(stage, result):
        def execute(*_args, **_kwargs):
            return called(stage, result)

        return execute

    for name, result in (
        ("validate_geometry_input", GeometryBuildInput((), ())),
        ("build_route_construction_plan", plan),
        (
            "engine_gate_volume_adjacency_conformality",
            {"status": "pass", "engine_gate": "planned"},
        ),
        ("write_occ_geometry_from_plan", built),
        ("validate_backend_tag_ledger", None),
        ("export_physical_group_records", ()),
    ):
        monkeypatch.setattr(builder, name, operation(name, result))

    def assert_gate(report):
        stage = (
            "assert_engine_gate_volume_adjacency_conformality"
            if report["engine_gate"] == "planned"
            else "assert_engine_gate_gmsh_brep_conformality"
        )
        called(stage, None)

    monkeypatch.setattr(builder, "assert_engine_gate_pass", assert_gate)
    write_sidecar = builder._write_stage_sidecar  # noqa: SLF001 -- Keep the real serializer while doubling only orchestration.

    def write(path, name, payload):
        if name == "00_timing":
            observed.events.append(name)
            observed.timing = payload
            if scenario.fail_timing:
                raise OSError("timing sidecar failed")
        else:
            called("write_" + name, None)
        return write_sidecar(path, name, payload)

    monkeypatch.setattr(builder, "_write_stage_sidecar", write)
    return observed


@pytest.mark.parametrize("failed_stage", [None, *BUILD_STAGES])
def test_build_keeps_stage_order_and_final_timing_on_each_exit(
    tmp_path, monkeypatch, failed_stage
):
    """Keep completed progress and the failed stage before writing final timing."""
    observed = _install_builder_stages(monkeypatch, _BuildScenario(failed_stage))
    source = GeometryBuildInput((), ())
    context = RepresentationContext(entities={})
    if failed_stage is None:
        assert (
            builder.SemanticGeometryBuilder().build(
                source, route="B", run_folder=tmp_path, representation_context=context
            )
            == ()
        )
        expected = list(BUILD_STAGES)
        statuses = ["done"] * len(expected)
    else:
        with pytest.raises(_StageFailure, match=failed_stage):
            builder.SemanticGeometryBuilder().build(
                source, route="B", run_folder=tmp_path, representation_context=context
            )
        expected = list(BUILD_STAGES[: BUILD_STAGES.index(failed_stage) + 1])
        statuses = ["done"] * (len(expected) - 1) + ["failed"]
    assert observed.events == [*expected, "00_timing"]
    assert [record["stage"] for record in observed.timing["stages"]] == expected
    assert [record["status"] for record in observed.timing["stages"]] == statuses
    metadata = tmp_path / "metadata/semantic_geometry"
    for name, size in observed.timing["sidecar_bytes"].items():
        assert (metadata / f"{name}.json").stat().st_size == size
    plan_completed = (
        "build_route_construction_plan" in expected
        and failed_stage != "build_route_construction_plan"
    )
    assert bool(observed.timing["counts"]) is plan_completed
    if (
        "write_02_build_route_construction_plan" in expected
        and failed_stage != "write_02_build_route_construction_plan"
    ):
        assert (
            json.loads(
                (metadata / "02_build_route_construction_plan.json").read_text()
            )["metadata"]["source_path"]
            == "<external>/source.gds"
        )


@pytest.mark.parametrize(
    "primary_stage", [None, "validate_geometry_input", "write_occ_geometry_from_plan"]
)
def test_final_timing_write_failure_preserves_existing_exception_priority(
    tmp_path, monkeypatch, primary_stage
):
    """Let a final-sidecar error replace success or an earlier stage error."""
    observed = _install_builder_stages(
        monkeypatch, _BuildScenario(primary_stage, fail_timing=True)
    )
    with pytest.raises(OSError, match="timing sidecar failed") as error:
        builder.SemanticGeometryBuilder().build(
            GeometryBuildInput((), ()),
            route="B",
            run_folder=tmp_path,
            representation_context=RepresentationContext(entities={}),
        )
    assert observed.events[-1] == "00_timing"
    if primary_stage is None:
        assert error.value.__context__ is None
    else:
        assert isinstance(error.value.__context__, _StageFailure)
        assert str(error.value.__context__) == primary_stage


def test_base_exception_still_writes_final_timing_without_an_exception_stage(
    tmp_path, monkeypatch
):
    """Retain the original Exception boundary rather than catching BaseException."""
    observed = _install_builder_stages(
        monkeypatch,
        _BuildScenario("validate_geometry_input", exception_type=KeyboardInterrupt),
    )
    with pytest.raises(KeyboardInterrupt):
        builder.SemanticGeometryBuilder().build(
            GeometryBuildInput((), ()),
            route="B",
            run_folder=tmp_path,
            representation_context=RepresentationContext(entities={}),
        )
    assert observed.events == ["validate_geometry_input", "00_timing"]
    assert observed.timing["stages"] == []


def test_missing_context_is_rejected_before_creating_build_directories(tmp_path):
    """Retain the builder's pre-transaction physical-context requirement."""
    run = tmp_path / "not-created"
    with pytest.raises(ValueError, match="interpret_physical_geometry"):
        builder.SemanticGeometryBuilder().build(
            GeometryBuildInput((), ()), route="B", run_folder=run
        )
    assert not run.exists()


def test_real_sidecar_serialization_error_still_writes_final_timing(
    tmp_path, monkeypatch
):
    """Record a real JSON failure without counting its unwritten sidecar."""
    observed = _install_builder_stages(monkeypatch, _BuildScenario())
    unsupported = GeometryBuildInput((), (), metadata={"unsupported": object()})
    monkeypatch.setattr(builder, "validate_geometry_input", lambda _source: unsupported)
    with pytest.raises(TypeError, match="not JSON serializable"):
        builder.SemanticGeometryBuilder().build(
            GeometryBuildInput((), ()),
            route="B",
            run_folder=tmp_path,
            representation_context=RepresentationContext(entities={}),
        )
    assert observed.timing["stages"][-1]["stage"] == "write_01_validate_geometry_input"
    assert observed.timing["stages"][-1]["status"] == "failed"
    assert observed.timing["sidecar_bytes"] == {}
    metadata = tmp_path / "metadata/semantic_geometry"
    assert not (metadata / "01_validate_geometry_input.json").exists()
    assert (metadata / "00_timing.json").is_file()


@dataclass(frozen=True)
class _CacheOperand:
    operand_id: str = "pipeline"
    exterior: tuple = ((0, 0), (1, 0), (1, 1))
    holes: tuple = ()
    physical: bool = True
    contact: bool = False


@pytest.mark.parametrize("fail_validation", [False, True])
def test_build_cache_covers_final_timing_then_cleans_up(
    tmp_path, monkeypatch, fail_validation
):
    """Keep reuse active through final timing and release it after either exit."""
    _install_builder_stages(monkeypatch, _BuildScenario())
    created = []

    def compute():
        value = object()
        created.append(value)
        return value

    def cached():
        return cached_subdivision(
            (_CacheOperand(),),
            GeometryGridContext(0.01),
            apply_grid=True,
            build=compute,
        )

    def validate(source):
        assert cached() is cached()
        if fail_validation:
            raise _StageFailure("validate")
        return source

    write_sidecar = builder._write_stage_sidecar  # noqa: SLF001 -- Check cache lifetime at the actual final-sidecar boundary.

    def write(path, name, payload):
        if name == "00_timing":
            assert cached() is created[0]
        return write_sidecar(path, name, payload)

    monkeypatch.setattr(builder, "validate_geometry_input", validate)
    monkeypatch.setattr(builder, "_write_stage_sidecar", write)
    if fail_validation:
        with pytest.raises(_StageFailure, match="validate"):
            builder.SemanticGeometryBuilder().build(
                GeometryBuildInput((), ()),
                route="B",
                run_folder=tmp_path,
                representation_context=RepresentationContext(entities={}),
            )
    else:
        builder.SemanticGeometryBuilder().build(
            GeometryBuildInput((), ()),
            route="B",
            run_folder=tmp_path,
            representation_context=RepresentationContext(entities={}),
        )
    assert len(created) == 1
    assert cached() is not created[0]
    assert len(created) == 2


@dataclass(frozen=True)
class _NativeResultWithoutTags:
    metadata: dict


def test_native_tag_attribute_error_is_measured_inside_its_stage(tmp_path, monkeypatch):
    """Include call argument errors within the original timed boundary."""
    observed = _install_builder_stages(monkeypatch, _BuildScenario())
    result = _NativeResultWithoutTags(
        {"engine_gate_gmsh_brep_conformality": {"engine_gate": "native"}}
    )
    monkeypatch.setattr(
        builder, "write_occ_geometry_from_plan", lambda *_args, **_kwargs: result
    )
    with pytest.raises(AttributeError, match="backend_entity_tags"):
        builder.SemanticGeometryBuilder().build(
            GeometryBuildInput((), ()),
            route="B",
            run_folder=tmp_path,
            representation_context=RepresentationContext(entities={}),
        )
    assert observed.timing["stages"][-1]["stage"] == "validate_backend_tag_ledger"
    assert observed.timing["stages"][-1]["status"] == "failed"


@pytest.mark.parametrize("report", [None, {}], ids=["missing", "failed"])
def test_native_report_rejection_remains_owned_by_the_existing_gate_validator(
    tmp_path, monkeypatch, report
):
    """Retain native report errors after writing their exact evidence sidecar."""
    validate_gate = builder.assert_engine_gate_pass
    observed = _install_builder_stages(monkeypatch, _BuildScenario())
    monkeypatch.setattr(builder, "assert_engine_gate_pass", validate_gate)
    native = ConstructionPlanRecord(
        "B", metadata={"engine_gate_gmsh_brep_conformality": report}
    )
    monkeypatch.setattr(
        builder, "write_occ_geometry_from_plan", lambda *_args, **_kwargs: native
    )
    with pytest.raises(TypeError if report is None else ValueError):
        builder.SemanticGeometryBuilder().build(
            GeometryBuildInput((), ()),
            route="B",
            run_folder=tmp_path,
            representation_context=RepresentationContext(entities={}),
        )
    evidence = (
        tmp_path / "metadata/semantic_geometry/engine_gate_gmsh_brep_conformality.json"
    )
    assert json.loads(evidence.read_text()) == report
    assert (
        observed.timing["stages"][-1]["stage"]
        == "assert_engine_gate_gmsh_brep_conformality"
    )
    assert observed.timing["stages"][-1]["status"] == "failed"


def _install_polygon_stages(monkeypatch, failed_stage=None):
    events = []
    results = {
        "plan_mm_contact_records": ((), ()),
        "plan_canonical_topology": ((), (), (), ()),
    }
    identity = {
        "project_native_route_representations",
        "prepare_auto_vacuum_solution_regions",
    }
    names = (
        "project_native_route_representations",
        "prepare_auto_vacuum_solution_regions",
        "build_semantic_evidence_facade",
        *POLYGON_STAGES,
        "bind_route_b_port_volume_plans",
        "reconcile_construction_body_surface_ids",
    )
    renamed = {
        "reconcile_solution_domain_boundaries": "reconcile_solution_domain_boundaries",
        "lower_port_sheet_regions": "lower_port_sheet_regions",
        "merge_solution_sidewall_interfaces": "merge_solution_sidewall_interfaces",
        "validate_route_b_port_sheet_sidewall_topology": "validate_route_b_port_sheet_sidewall_topology",
    }

    def operation(name):
        def execute(*args, **_kwargs):
            stage = name.removeprefix("_")
            events.append(stage)
            if stage == failed_stage:
                raise _StageFailure(stage)
            return args[0] if name in identity else results.get(stage, ())

        return execute

    for name in names:
        actual = renamed.get(name, name)
        monkeypatch.setattr(dispatch, actual, operation(actual))
    return events


def _polygon_source(with_port=False):
    port = PortSheetRegionRecord("port", "port", "polygon", ((0, 0), (1, 0), (1, 1)))
    return GeometryBuildInput((), (), port_sheet_regions=(port,) if with_port else ())


@pytest.mark.parametrize(
    "route,with_port", [("A", False), ("B", False), ("B", True), ("C", False)]
)
def test_polygon_phase_order_keeps_route_specific_steps(monkeypatch, route, with_port):
    """Preserve conditional contacts, port lowering and route-specific record type."""
    events = _install_polygon_stages(monkeypatch)
    plan = dispatch.build_route_construction_plan(
        _polygon_source(with_port), route=route
    )
    stages = [
        stage
        for stage in POLYGON_STAGES
        if not (
            (stage == "plan_mm_contact_records" and route == "C")
            or (stage == "lower_port_sheet_regions" and not with_port)
        )
    ]
    assert [entry["stage"] for entry in plan.metadata["timings"]] == stages
    assert all(entry["status"] == "done" for entry in plan.metadata["timings"])
    expected = [
        "project_native_route_representations",
        "prepare_auto_vacuum_solution_regions",
        "validate_selected_route",
        "build_semantic_evidence_facade",
        *stages[1:],
    ]
    if route == "B":
        expected.insert(
            expected.index("plan_route_volumes") + 1, "bind_route_b_port_volume_plans"
        )
    expected.insert(
        expected.index("validate_surface_use_counts") + 1,
        "reconcile_construction_body_surface_ids",
    )
    assert events == expected
    assert isinstance(plan, RouteABConstructionPlanRecord) is (route in {"A", "B"})
    assert plan.metadata["port_sheet_region_layer"]["lowering_status"] == (
        "lowered" if with_port else "none"
    )


@pytest.mark.parametrize("failed_stage", POLYGON_STAGES)
def test_polygon_failure_stops_before_following_stage(monkeypatch, failed_stage):
    """Preserve the first stage failure without running later planning work."""
    events = _install_polygon_stages(monkeypatch, failed_stage)
    with pytest.raises(_StageFailure, match=failed_stage):
        dispatch.build_route_construction_plan(_polygon_source(True), route="B")
    assert events[-1] == failed_stage
    expected_prefix = list(POLYGON_STAGES[: POLYGON_STAGES.index(failed_stage) + 1])
    assert [event for event in events if event in POLYGON_STAGES] == expected_prefix


@pytest.mark.parametrize(
    "operation,malformed",
    [("plan_mm_contact_records", ((),)), ("plan_canonical_topology", ((), (), ()))],
)
def test_result_unpacking_stays_outside_the_measured_function(
    monkeypatch, operation, malformed
):
    """Retain completed timing when a returned tuple cannot be unpacked."""
    _install_polygon_stages(monkeypatch)
    timings = StageTimings(diagnostic_prefix="geometry.compiler.")
    monkeypatch.setattr(dispatch, "StageTimings", lambda **_kwargs: timings)
    monkeypatch.setattr(dispatch, operation, lambda *_args, **_kwargs: malformed)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        dispatch.build_route_construction_plan(_polygon_source(), route="B")
    assert timings.records[-1]["stage"] == operation
    assert timings.records[-1]["status"] == "done"


@pytest.mark.parametrize("route", ["A", "B", "C"])
@pytest.mark.parametrize("boundary_kind", ["authored", "reconstructed"])
def test_curve_dispatch_precedes_polygon_preparation(monkeypatch, route, boundary_kind):
    """Project first, then dispatch curves or reject Route C before polygon work."""
    events = _install_polygon_stages(monkeypatch)
    source = _polygon_source()
    if boundary_kind == "authored":
        source = replace(
            source,
            boundary_curves=(BoundaryCurveChainSpec("boundary", "entity", (0, 0), ()),),
        )
    else:
        source = replace(
            source,
            boundary_reconstruction=(
                GdsBoundaryReconstructionSpec("boundary", "entity", (0, 0), "circle"),
            ),
        )
    sentinel = ConstructionPlanRecord(route)

    def lower_curves(received, *, route, representation_context):
        assert received is source
        assert representation_context is None
        events.append("curved")
        return sentinel

    monkeypatch.setattr(dispatch, "build_curved_construction_plan", lower_curves)
    if route == "C":
        with pytest.raises(NotImplementedError, match="not Route C"):
            dispatch.build_route_construction_plan(source, route=route)
        assert events == ["project_native_route_representations"]
    else:
        assert dispatch.build_route_construction_plan(source, route=route) is sentinel
        assert events == ["project_native_route_representations", "curved"]


def test_withdrawn_bands_fail_before_projection(monkeypatch):
    """Retain the earliest rejection before any planning or projection calls."""
    events = _install_polygon_stages(monkeypatch)
    source = replace(
        _polygon_source(), metadata={"surface_distance_bands": ["withdrawn"]}
    )
    with pytest.raises(ValueError, match="withdrawn"):
        dispatch.build_route_construction_plan(source, route="B")
    assert events == []


@pytest.mark.parametrize("error_type", [_StageFailure, KeyboardInterrupt])
def test_timing_records_keep_the_original_exception_boundary(error_type):
    """Record failed Exception stages while leaving BaseException unrecorded."""
    timings = StageTimings()
    with pytest.raises(error_type):
        with timings.measure("failure"):
            raise error_type("failure")
    assert [record["status"] for record in timings.records] == (
        ["failed"] if issubclass(error_type, Exception) else []
    )


def test_public_wrappers_retain_function_metadata_and_signatures():
    """Keep the names, keyword-only arguments and original wrapper layering."""
    build = builder.SemanticGeometryBuilder.build
    plan = dispatch.build_route_construction_plan
    assert build.__name__ == "build"
    assert plan.__name__ == "build_route_construction_plan"
    assert list(inspect.signature(build).parameters) == [
        "self",
        "build_input",
        "route",
        "run_folder",
        "representation_context",
    ]
    assert list(inspect.signature(plan).parameters) == [
        "build_input",
        "route",
        "representation_context",
    ]
    assert build.__wrapped__.__wrapped__.__name__ == "build"
    assert plan.__wrapped__.__wrapped__.__name__ == "build_route_construction_plan"
