"""Semantic route inputs preserve existing compiler and artifact identities."""

from dataclasses import asdict, replace
import json

import pytest

from scgsim.geometry import GeometryRoute, ApproximationProfile, interpret_physical_geometry
from scgsim.geometry.compiler import (
    GeometryRoute as CompilerGeometryRoute,
    SemanticGeometryBuilder,
    build_route_construction_plan,
)
from scgsim.geometry.models import GeometryRoute as ModelsGeometryRoute
from scgsim.geometry.models.common import _normalize_geometry_route
from scgsim.geometry.models.tags import BackendEntityTagRecord
from scgsim.geometry.construction.export import export_physical_group_records
from scgsim.palace.preparation.eigenmode import EigenmodeSim
from scgsim.palace.preparation.electrostatic import ElectrostaticSim
from scgsim.palace.preparation.transforms import normalize_route_a_thin_film
from scgsim.palace.mesh.build import build_route_mesh
from test_geometry_context_propagation import conductor_input
from sgb_method_cases import junction_hole_case


def test_enum_is_one_public_type_with_legacy_string_and_json_values():
    assert GeometryRoute is CompilerGeometryRoute is ModelsGeometryRoute
    assert [(member.name, str(member), member.display_name) for member in GeometryRoute] == [
        ("THIN_FILM", "A", "Thin film"),
        ("PEC_SHELL", "B", "PEC shell"),
        ("RETAINED_SOLID", "C", "Retained solid"),
    ]
    assert json.dumps(list(GeometryRoute)) == '["A", "B", "C"]'


@pytest.mark.parametrize("route", list(GeometryRoute))
def test_aliases_have_one_canonical_serialized_code(route):
    for alias in (route, route.value, route.value.lower(), route.name,
                  f" {route.name.lower().replace('_', '-')} "):
        result = _normalize_geometry_route(alias)
        assert result == route.value
        assert type(result) is str


@pytest.mark.parametrize("route", ["", "unknown", "Route A", None, 1])
def test_unknown_names_rejected(route):
    with pytest.raises(ValueError, match="thin_film"):
        _normalize_geometry_route(route)


def _serialized_groups(plan):
    plan = replace(plan, backend_entity_tags=tuple(
        BackendEntityTagRecord(tag.source_record_kind, tag.source_record_id,
                               (tag.dimension, index + 1))
        for index, tag in enumerate(plan.tags)
    ))
    return json.dumps([asdict(group) for group in export_physical_group_records(plan)],
                      sort_keys=True)


@pytest.mark.parametrize("route", list(GeometryRoute))
def test_semantic_names_enum_and_codes_have_identical_pure_plans_and_groups(route):
    if route is GeometryRoute.THIN_FILM:
        source, stack, profile = junction_hole_case()
    else:
        source, stack = conductor_input()
        profile = ApproximationProfile("route_b" if route.value == "B" else "solid", "v1", {})
    effective, _, physical = interpret_physical_geometry(source, stack, profile)
    plans = [build_route_construction_plan(effective, route=value,
                                          representation_context=physical.context)
             for value in (route.value, route, route.name.lower())]
    untimed = [replace(plan, metadata={k: v for k, v in plan.metadata.items() if k != "timings"})
               for plan in plans]
    assert untimed[0] == untimed[1] == untimed[2]
    assert plans[0].route == route.value
    assert type(plans[0].route) is str
    assert _serialized_groups(plans[0]) == _serialized_groups(plans[1]) == _serialized_groups(plans[2])
    if route is GeometryRoute.RETAINED_SOLID:
        assert any(volume.material_id == "metal_material" for volume in plans[0].volumes)


@pytest.mark.parametrize("sim_type", [EigenmodeSim, ElectrostaticSim])
@pytest.mark.parametrize("route", [GeometryRoute.THIN_FILM, GeometryRoute.PEC_SHELL])
def test_frontend_accepts_semantic_names_and_keeps_code(sim_type, route):
    sim = sim_type()
    sim._materials = {}
    for value in (route, route.name.lower(), route.value):
        sim.set_surface_epr(representation=value, specs={})
        assert sim.route == route.value
        assert type(sim.route) is str


@pytest.mark.parametrize("sim_type", [EigenmodeSim, ElectrostaticSim])
@pytest.mark.parametrize("route", ["retained_solid", GeometryRoute.RETAINED_SOLID, "unknown"])
def test_frontend_rejects_unsupported_or_unknown_route(sim_type, route):
    with pytest.raises(ValueError):
        sim_type().set_surface_epr(representation=route, specs={})


@pytest.mark.parametrize("route", ["thin_film", GeometryRoute.THIN_FILM, "A"])
def test_thin_film_lowering_profile_is_independent_and_unchanged(route):
    for profile in ("substrate_face", "metal_gap_equivalent"):
        assert normalize_route_a_thin_film(route, profile) == profile
    with pytest.raises(ValueError, match="requires"):
        normalize_route_a_thin_film(route, None)


def test_pec_shell_transform_and_palace_mesh_keep_capability_boundary(tmp_path):
    assert normalize_route_a_thin_film("pec_shell", None) is None
    for route in ("retained_solid", "C", "unknown"):
        with pytest.raises(ValueError):
            normalize_route_a_thin_film(route, None)
        with pytest.raises(ValueError):
            build_route_mesh(component=None, stack={}, route=route,
                             output_dir=tmp_path, source_geometry_input=object(),
                             representation_context=object())
    assert not list(tmp_path.iterdir())


def test_builder_normalizes_before_native_or_filesystem_work(tmp_path, monkeypatch):
    import scgsim.geometry.compiler.builder as builder
    source, _ = conductor_input()
    received = []

    def stop_at_planning(build_input, *, route, representation_context):
        received.append(route)
        raise RuntimeError("stop before native")

    monkeypatch.setattr(builder, "build_route_construction_plan", stop_at_planning)
    for alias in (GeometryRoute.PEC_SHELL, "pec_shell", "B"):
        with pytest.raises(RuntimeError, match="stop before native"):
            SemanticGeometryBuilder().build(source, route=alias,
                run_folder=tmp_path, representation_context=object())
    assert received == ["B", "B", "B"]
    with pytest.raises(ValueError, match="geometry route"):
        SemanticGeometryBuilder().build(source, route="bad", run_folder=tmp_path)


@pytest.mark.parametrize("route", [GeometryRoute.THIN_FILM, GeometryRoute.PEC_SHELL])
def test_mesh_entry_normalizes_before_native_work(route, tmp_path, monkeypatch):
    import scgsim.palace.mesh.build as mesh
    received = []

    def stop_before_native(stack, code):
        received.append(code)
        raise RuntimeError("stop before native")

    monkeypatch.setattr(mesh, "_route_a_thin_film_provenance", stop_before_native)
    for alias in (route, route.name.lower(), route.value):
        with pytest.raises(RuntimeError, match="stop before native"):
            build_route_mesh(component=None, stack={}, route=alias,
                output_dir=tmp_path, source_geometry_input=object(),
                representation_context=object())
    assert received == [route.value] * 3
    assert not list(tmp_path.iterdir())


def test_strict_artifact_route_reader_does_not_accept_semantic_aliases():
    from scgsim.palace.mesh.build import _sgb_records_route_context
    assert _sgb_records_route_context([{"route": "A"}]) == "A"
    with pytest.raises(ValueError, match="explicit route"):
        _sgb_records_route_context([{"route": "thin_film"}])
