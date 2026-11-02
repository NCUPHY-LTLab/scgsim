"""Preserve shell membership, reference order and fail-first volume planning."""

from dataclasses import replace

import pytest
from test_geometry_context_propagation import conductor_input

from scgsim.geometry.models.construction import ConstructionBodyPlanRecord
from scgsim.geometry.models.topology import (
    MMContactRecord,
    RouteABVolumePlanRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.volumes import (
    plan_cut_host_operations,
    plan_route_construction_bodies,
    plan_route_volumes,
    reconcile_construction_body_surface_ids,
)


def _face(surface_id, *, owner="metal", z_um=0, role="cutout_boundary_shell"):
    return SurfacePlanRecord(
        surface_id,
        owner,
        role,
        {"outer_loop": ((0, 0), (4, 0), (4, 4), (0, 4)), "z_um": z_um},
        normal_hint=(0, 0, 1),
        metadata={"boundary_volume_ids": ("vacuum",), "conductor_component_id": "PEC"},
    )


def _contact(contact_id):
    return MMContactRecord(
        contact_id=contact_id,
        lower_entity_id="metal",
        upper_entity_id="upper",
        lower_source_face_id="metal-top",
        upper_source_face_id="upper-bottom",
        lower_source_fragment_ids=(),
        upper_source_fragment_ids=(),
        outer_loop=((0, 0), (4, 0), (4, 4), (0, 4)),
        area_um2=16,
        normal=(0, 0, 1),
        conductor_component_id="PEC",
        net_id="signal",
    )


def test_inner_shell_keeps_contact_multiplicity_and_surface_order():
    """A contacted component retains its shell members and repeated contact IDs."""
    source, _ = conductor_input()
    metal, vacuum = source.entities
    upper = replace(
        metal,
        semantic_id="upper",
        geometry={**metal.geometry, "z_min_um": 1, "z_max_um": 2},
    )
    source = replace(source, entities=(metal, upper, vacuum))
    first, second = _face("bottom"), _face("bottom-child")
    exterior = _face("exterior", owner="vacuum", z_um=6, role="domain_boundary")
    (volume,) = plan_route_volumes(
        source,
        route="B",
        surfaces=(first, exterior, second),
        mm_contacts=(_contact("second"), _contact("first"), _contact("second")),
    )

    assert isinstance(volume, RouteABVolumePlanRecord)
    assert tuple(ref.surface_id for ref in volume.surface_refs) == (
        "bottom",
        "exterior",
        "bottom-child",
    )
    assert tuple(ref.surface_id for ref in volume.exterior_surface_refs) == (
        "exterior",
    )
    (shell,) = volume.inner_pec_void_shells
    assert shell.shell_id == "VOID__vacuum__PEC"
    assert shell.owner_semantic_ids == ("metal", "upper")
    assert tuple((ref.surface_id, ref.orientation) for ref in shell.surface_refs) == (
        ("bottom", "forward"),
        ("bottom-child", "forward"),
    )
    assert volume.metadata["inner_pec_void_shell_contact_ids"] == {
        "PEC": ("second", "first", "second")
    }


def test_boundary_attached_component_stays_in_exterior_shell():
    """One boundary-attached face moves the entire component to the exterior."""
    source, _ = conductor_input()
    metal, vacuum = source.entities
    vacuum = replace(
        vacuum,
        geometry={
            **vacuum.geometry,
            "z_max_um": 1,
            "domain_bounds_um": {**vacuum.geometry["domain_bounds_um"], "z_max_um": 1},
        },
    )
    source = replace(source, entities=(metal, vacuum))
    (volume,) = plan_route_volumes(
        source, route="B", surfaces=(_face("bottom"), _face("top", z_um=1))
    )
    assert isinstance(volume, RouteABVolumePlanRecord)
    assert volume.inner_pec_void_shells == ()
    assert volume.exterior_surface_refs == volume.surface_refs


@pytest.mark.parametrize(
    "geometry,normal_hint,message",
    (
        (
            {"quad_points": ((0, 0, 0), (1, 0, 0), (2, 0, 0))},
            None,
            "broken has no nondegenerate normal",
        ),
        ({}, (0, 0, 1), "broken requires outer_loop or quad_points"),
    ),
)
def test_surface_error_precedes_missing_solution_bounds(geometry, normal_hint, message):
    """Normal and ring failures occur before accessing malformed volume bounds."""
    source, _ = conductor_input()
    vacuum = replace(source.entities[1], geometry={})
    source = replace(source, entities=(vacuum,))
    surface = replace(_face("broken"), geometry_ref=geometry, normal_hint=normal_hint)
    with pytest.raises(ValueError, match=f"^{message}$"):
        plan_route_volumes(source, route="B", surfaces=(surface,))


@pytest.mark.parametrize("route", ("A", "B", "C"))
def test_cut_groups_preserve_host_encounter_order_and_duplicate_surface_ids(route):
    """Host exclusions keep body order and every declared surface reference."""
    bodies = tuple(
        ConstructionBodyPlanRecord(
            body_id, "metal", host, "cutout_boundary_shell", {}, ids
        )
        for body_id, host, ids in (
            ("first", "second-host", ("shared", "shared")),
            ("second", "first-host", ("other",)),
            ("third", "second-host", ("last",)),
        )
    )
    operations = plan_cut_host_operations(route=route, construction_bodies=bodies)
    if route == "C":
        assert operations == ()
        return
    assert tuple(operation.host_semantic_id for operation in operations) == (
        "second-host",
        "first-host",
    )
    assert operations[0].construction_body_ids == ("first", "third")
    assert operations[0].exposed_surface_ids == ("shared", "shared", "last")


def test_reconcile_sorts_children_but_keeps_unmatched_body():
    """Final local children replace expected IDs while untouched bodies stay intact."""
    body = ConstructionBodyPlanRecord(
        "body", "metal", "vacuum", "cutout_boundary_shell", {}
    )
    unmatched = replace(
        body, construction_body_id="unmatched", expected_surface_ids=("old",)
    )
    surfaces = tuple(
        replace(_face(surface_id), geometry_ref={"construction_body_id": "body"})
        for surface_id in ("z", "a", "z")
    )
    first, second = reconcile_construction_body_surface_ids(
        (body, unmatched), surfaces=surfaces
    )
    assert first.expected_surface_ids == ("a", "z", "z")
    assert second is unmatched


def test_route_c_has_no_construction_body_dependencies():
    """Retained material routes return before host or interface processing."""
    source, _ = conductor_input()
    metal = replace(source.entities[0], host_void_semantic_id=None)
    assert (
        plan_route_construction_bodies(replace(source, entities=(metal,)), route="C")
        == ()
    )


@pytest.mark.parametrize(
    "normal,orientation", (((1, 0, 0), "reversed"), ((-1, 0, 0), "forward"))
)
def test_center_aligned_void_face_uses_local_occupied_region(normal, orientation):
    """An L-shaped conductor's reentrant wall is oriented by native inside probes."""
    source, _ = conductor_input()
    metal, vacuum = source.entities
    metal = replace(
        metal,
        geometry={
            **metal.geometry,
            "outer_loop": ((0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)),
        },
    )
    vacuum = replace(
        vacuum,
        geometry={
            **vacuum.geometry,
            "outer_loop": ((-4, -2), (6, -2), (6, 6), (-4, 6)),
            "domain_bounds_um": {**vacuum.geometry["domain_bounds_um"], "x_min_um": -4},
        },
    )
    source = replace(source, entities=(metal, vacuum))
    wall = replace(
        _face("reentrant"),
        geometry_ref={"quad_points": ((2, 2, 0), (2, 4, 0), (2, 4, 1), (2, 2, 1))},
        normal_hint=normal,
    )
    (volume,) = plan_route_volumes(source, route="B", surfaces=(wall,))
    assert isinstance(volume, RouteABVolumePlanRecord)
    (shell,) = volume.inner_pec_void_shells
    assert shell.surface_refs[0].orientation == orientation
