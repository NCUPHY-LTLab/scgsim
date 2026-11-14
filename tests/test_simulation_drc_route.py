"""``simulation_drc`` checks the route the simulation will use: Route A or Route B, with ``set_route``'s rule.

``route`` and ``approximation_profile`` are the arguments of ``set_route``.  A Route A check builds the model
``mesh()`` builds for the same plan and ``set_route`` call, and a route ``set_route`` refuses is refused by
the check, naming the check's call.  No test here runs Gmsh: ``mesh()`` stops where the mesher would start.
"""

from __future__ import annotations

import json

import pytest
from test_palace_route_and_surface_epr import ROUTE_B, SUBSTRATE_FACE, THIN_FILM
from test_simulation_drc import (
    LAYER_STACK,
    MATERIALS,
    SIMULATIONS,
    MeshingReached,
    check,
    configured,
    two_pads,
)

import scgsim.palace.mesh.build as build_module
from scgsim.geometry import GeometryPlan
from scgsim.palace.simulation import core

# The Route A profiles the check is compared on: the default, and one with film bindings.
ROUTE_A_PROFILES = [None, SUBSTRATE_FACE]
ROUTE_A_IDS = ["thin_film", "substrate_face"]


def recorded_sidecars(monkeypatch):
    """The sidecar of every real SGB build that ``mesh()`` and ``simulation_drc`` make, read as it is built.

    Both reach the build through ``build_route_geometry``; the record is read before the check removes its folder.
    """
    sidecars = []
    real = core.build_route_geometry

    def recording(*args, **kwargs):
        geometry = real(*args, **kwargs)
        sidecars.append(json.loads(geometry.artifacts.sidecar.read_bytes()))
        return geometry

    monkeypatch.setattr(core, "build_route_geometry", recording)
    return sidecars


def meshed_until_the_mesher(  # noqa: PLR0913 -- the layout, the test fixtures, and the set_route call
    simulation, component, tmp_path, monkeypatch, *, route, approximation_profile
):
    """``mesh()`` of the plan and settings ``simulation_drc`` uses, after ``set_route``; it stops where meshing starts.

    Returns the error ``mesh()`` raised: ``MeshingReached`` when the geometry and meshing-stage check passed.
    """

    def meshing_reached(*_args, **_kwargs):
        raise MeshingReached

    plan = GeometryPlan(
        component,
        root_id="chip",
        layer_stack=LAYER_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    sim = configured(simulation, plan.prepare(), None)
    sim.set_route(route, approximation_profile=approximation_profile)
    sim.set_output_dir(tmp_path / "run")
    monkeypatch.setattr(build_module, "mesh_sgb", meshing_reached)
    with pytest.raises(Exception) as error:  # noqa: PT011 -- the type is what the test compares
        sim.mesh()
    return error.value


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_check_without_a_route_is_route_b(monkeypatch, simulation):
    """No ``route`` gives the Route B check: the same built model and the same report as ``route="B"``."""
    component = two_pads(monkeypatch, gap_um=10.0, port=True)
    sidecars = recorded_sidecars(monkeypatch)
    default = check(simulation, component)
    explicit = check(simulation, component, route="B")
    assert default == explicit
    assert default.passed
    first, second = sidecars
    assert first == second


@pytest.mark.parametrize("simulation", SIMULATIONS)
@pytest.mark.parametrize("profile", ROUTE_A_PROFILES, ids=ROUTE_A_IDS)
def test_a_route_a_check_builds_the_model_mesh_builds(
    monkeypatch, tmp_path, simulation, profile
):
    """The check and ``mesh()`` build one model for the same plan and ``set_route("A", ...)`` call.

    The two sidecars are equal, and the Route A model differs from the Route B one, so the route reached the build.
    """
    component = two_pads(monkeypatch, gap_um=10.0, port=True)
    sidecars = recorded_sidecars(monkeypatch)
    reached = meshed_until_the_mesher(
        simulation,
        component,
        tmp_path,
        monkeypatch,
        route="A",
        approximation_profile=profile,
    )
    assert isinstance(reached, MeshingReached)
    assert check(simulation, component, route="A", approximation_profile=profile).passed
    check(simulation, component)
    meshed, checked, route_b = sidecars
    assert checked == meshed
    assert route_b != meshed


@pytest.mark.parametrize("simulation", SIMULATIONS)
@pytest.mark.parametrize(
    ("gap_um", "setting"),
    [
        (-5.0, {"route": "A", "approximation_profile": None}),
        (10.0, {"route": "A", "approximation_profile": ROUTE_B}),
        (10.0, {"route": "B", "approximation_profile": THIN_FILM}),
    ],
    ids=[
        "route-a-overlapping-electrodes",
        "route-a-route-b-profile",
        "route-b-thin-film-profile",
    ],
)
def test_a_layout_mesh_refuses_on_the_route_is_refused_by_the_check_with_the_same_error(
    monkeypatch, tmp_path, simulation, gap_um, setting
):
    """Overlapping electrodes on Route A, and a profile that does not belong to the route.

    ``set_route`` takes each, since whether a profile suits the route depends on the plan's films; ``mesh()``
    refuses it before meshing, and the check raises that error.
    """
    route, profile = setting["route"], setting["approximation_profile"]
    component = two_pads(monkeypatch, gap_um=gap_um)
    expected = meshed_until_the_mesher(
        simulation,
        component,
        tmp_path,
        monkeypatch,
        route=route,
        approximation_profile=profile,
    )
    assert not isinstance(expected, MeshingReached)
    with pytest.raises(type(expected)) as raised:
        check(simulation, component, route=route, approximation_profile=profile)
    assert str(raised.value) == str(expected)


@pytest.mark.parametrize("simulation", SIMULATIONS)
@pytest.mark.parametrize(
    ("route", "says"),
    [
        (
            "C",
            "Route C: is not a Palace route: SGB builds Route C for backends that keep conductor volumes. "
            "Call simulation_drc(..., route='A') or simulation_drc(..., route='B'). "
            "Change it in the call to simulation_drc.",
        ),
        (
            "D",
            "Unknown route 'D': call simulation_drc(..., route='A') for films as zero-thickness sheets or "
            "simulation_drc(..., route='B') for 3D conductors. Change it in the call to simulation_drc.",
        ),
    ],
)
def test_a_route_set_route_refuses_is_refused_by_the_check_naming_the_check(
    monkeypatch, simulation, route, says
):
    """Route C and an unknown route are refused with the route rule's messages, said for this call."""
    component = two_pads(monkeypatch, gap_um=10.0)
    with pytest.raises(ValueError) as raised:  # noqa: PT011 -- the message is compared whole
        check(simulation, component, route=route)
    assert str(raised.value) == says
    with pytest.raises(ValueError) as by_set_route:  # noqa: PT011 -- the message is compared whole
        simulation().set_route(route)
    assert str(by_set_route.value) == says.replace(
        "simulation_drc(..., route=", "set_route("
    ).replace("the call to simulation_drc", "the call to set_route")


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_profile_that_is_not_an_approximation_profile_is_refused_naming_the_check(
    monkeypatch, simulation
):
    """The profile is refused as ``set_route`` refuses it, and the refusal points at the check's call."""
    component = two_pads(monkeypatch, gap_um=10.0)
    with pytest.raises(TypeError) as raised:
        check(simulation, component, route="A", approximation_profile="thin_film")
    message = str(raised.value)
    assert "simulation_drc(..., approximation_profile=ApproximationProfile(" in message
    assert message.endswith("found 'str'. Change it in the call to simulation_drc.")
    assert "set_route" not in message


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_refused_route_is_refused_before_the_layout_is_read(monkeypatch, simulation):
    """The route is checked first, so a refused route costs no preparation of the layout."""
    component = two_pads(monkeypatch, gap_um=10.0)

    def layout_read(*_args, **_kwargs):
        raise AssertionError("the layout was prepared before the route was checked")

    monkeypatch.setattr(core, "prepare_layout_snapshot", layout_read)
    with pytest.raises(ValueError, match="Route C"):
        check(simulation, component, route="C")


def test_thin_film_is_the_default_route_a_profile_of_the_check(monkeypatch):
    """``route="A"`` and ``route="A", approximation_profile=thin_film`` build one model."""
    component = two_pads(monkeypatch, gap_um=10.0, port=True)
    sidecars = recorded_sidecars(monkeypatch)
    check(SIMULATIONS[0], component, route="A")
    check(SIMULATIONS[0], component, route="A", approximation_profile=THIN_FILM)
    default, explicit = sidecars
    assert default == explicit
