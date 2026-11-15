"""The two geometry settings of a Palace simulation: the route (``set_route``) and Surface EPR (``set_surface_epr``).

Route B is the default; one ``set_route`` call states the whole route setting.  Surface EPR is computed only
when ``set_surface_epr`` was called with specs, by the same rule for Eigenmode and Electrostatic: without it
``mesh()`` asks for no bands, the config has no Surface-EPR rows and ``input_summary()`` says it is not
requested.  No test here runs the mesher.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from palace_sgb_support import FILM
from test_palace_index_map_v3 import Run
from test_palace_input_summary import eigenmode_sim
from test_palace_mesh_controls import STACK, stand_in_mesh_stages, two_pad_snapshot
from test_palace_simulation_lifecycle import handed_off
from test_simulation_drc import LAYER_STACK

from scgsim.geometry import ApproximationProfile
from scgsim.palace import EigenmodeSim, ElectrostaticSim
from scgsim.palace.config.index import SurfaceEprEntry
from scgsim.palace.simulation import core
from scgsim.palace.simulation.lifecycle import Stage
from scgsim.palace.source import layout_ports
from scgsim.palace.source.workflow import route_profile

SIMS = [EigenmodeSim, ElectrostaticSim]
THIN_FILM = ApproximationProfile("thin_film", "v1", {})
SUBSTRATE_FACE = ApproximationProfile(
    "substrate_face",
    "v1",
    {
        "film_bindings": {
            "metal": {
                "supporting_entity_id": "substrate",
                "supporting_side": "upper",
            }
        }
    },
)
ROUTE_B = ApproximationProfile("route_b", "v1", {})
SOLID = ApproximationProfile("solid", "v1", {})
ROUTE_SETTING = ("route", "approximation_profile")


def route_setting(sim):
    """The two values one ``set_route`` call states."""
    return tuple(getattr(sim, name) for name in ROUTE_SETTING)


def profile_used(sim):
    """The profile the preparation interprets the plan with, as ``input_summary()`` names it."""
    return route_profile(sim.route, sim.approximation_profile)


# ---- The route ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sim_type", SIMS)
def test_a_simulation_that_never_calls_set_route_is_route_b(
    sim_type, monkeypatch, tmp_path
):
    """Route B is the default, with the identity profile ``route_b`` left to the preparation."""
    sim = sim_type()
    assert route_setting(sim) == ("B", None)
    assert profile_used(sim) == ROUTE_B
    calls = stand_in_mesh_stages(monkeypatch, core, tmp_path)
    monkeypatch.setattr(layout_ports, "prepare_mesh_input", _prepared_stand_in)
    sim.set_plan(two_pad_snapshot(monkeypatch))
    sim.set_output_dir(tmp_path)
    sim.mesh()
    (_, geometry), *_ = calls
    assert geometry.route == "B"


@pytest.mark.parametrize("sim_type", SIMS)
def test_each_route_has_a_default_profile_and_a_given_profile_is_used_as_given(
    sim_type,
):
    """``set_route("A")`` lowers with ``thin_film`` and ``set_route("B")`` keeps ``route_b``."""
    sim = sim_type()
    sim.set_route("A")
    assert route_setting(sim) == ("A", None)
    assert profile_used(sim) == THIN_FILM
    sim.set_route("A", approximation_profile=SUBSTRATE_FACE)
    assert route_setting(sim) == ("A", SUBSTRATE_FACE)
    assert profile_used(sim) == SUBSTRATE_FACE
    sim.set_route("B")
    assert route_setting(sim) == ("B", None)
    assert profile_used(sim) == ROUTE_B
    with pytest.raises(TypeError, match="ApproximationProfile"):
        sim.set_route("A", approximation_profile="thin_film")  # type: ignore[arg-type]
    assert route_setting(sim) == ("B", None)


@pytest.mark.parametrize("sim_type", SIMS)
@pytest.mark.parametrize(
    ("route", "profile"),
    [("B", THIN_FILM), ("B", SUBSTRATE_FACE), ("A", ROUTE_B), ("A", SOLID)],
)
def test_a_profile_the_route_does_not_lower_with_is_used_as_given(
    sim_type, route, profile
):
    """Whether a profile suits the route depends on the plan's films, so SGB's route check refuses it at
    ``mesh()``, with the finite films or the sheets it finds; ``set_route`` does not guess."""
    sim = sim_type()
    sim.set_route(route, approximation_profile=profile)
    assert route_setting(sim) == (route, profile)
    assert profile_used(sim) == profile


@pytest.mark.parametrize("sim_type", SIMS)
def test_set_route_has_no_thin_film_keyword(sim_type):
    """The profile's name says the lowering, so ``thin_film=`` is an unknown keyword."""
    with pytest.raises(TypeError, match="thin_film"):
        sim_type().set_route("A", thin_film="substrate_face")  # type: ignore[call-arg]
    assert not hasattr(sim_type(), "thin_film")


@pytest.mark.parametrize("sim_type", SIMS)
def test_route_a_without_a_profile_prepares_the_same_input_as_the_thin_film_profile(
    sim_type, monkeypatch
):
    """``set_route("A")`` and ``set_route("A", approximation_profile=thin_film)`` prepare one geometry input."""
    snapshot = two_pad_snapshot(monkeypatch)
    prepared = []
    for profile in (None, THIN_FILM):
        sim = sim_type()
        sim.set_plan(snapshot)
        sim.set_route("A", approximation_profile=profile)
        prepared.append(sim._geometry_request(snapshot).source)  # noqa: SLF001
    default, explicit = prepared
    assert default.geometry_input == explicit.geometry_input
    assert default.stack == explicit.stack
    assert default.representation == explicit.representation


@pytest.mark.parametrize("sim_type", SIMS)
@pytest.mark.parametrize(
    ("calls", "profile"),
    [
        ((), "route_b"),
        ((("B", None),), "route_b"),
        ((("A", None),), "thin_film"),
        ((("A", SUBSTRATE_FACE),), "substrate_face"),
        ((("A", SUBSTRATE_FACE), ("B", None)), "route_b"),
        ((("B", None), ("A", None)), "thin_film"),
    ],
)
def test_input_summary_names_the_profile_the_preparation_uses(
    sim_type, calls, profile, monkeypatch, tmp_path
):
    """The route and the profile's name, the route's default included; a second call replaces both."""
    sim = sim_type()
    sim.set_plan(two_pad_snapshot(monkeypatch))
    for route, given in calls:
        sim.set_route(route, approximation_profile=given)
    backend = sim.input_summary().data["backend"]
    assert backend["route"] == (calls[-1][0] if calls else "B")
    assert backend["approximation_profile"]["name"] == profile
    assert "thin_film" not in backend


@pytest.mark.parametrize("sim_type", SIMS)
@pytest.mark.parametrize(
    ("route", "says"),
    [
        (
            "C",
            "Route C: is not a Palace route: SGB builds Route C for backends that keep conductor volumes",
        ),
        ("D", "Unknown route 'D'"),
        (None, "Unknown route None"),
    ],
)
def test_any_other_route_is_refused_with_a_and_b_named(sim_type, route, says):
    """A Palace simulation takes Route A or Route B; the refusal says why and names both calls."""
    with pytest.raises(ValueError, match=says) as refused:
        sim_type().set_route(route)  # type: ignore[arg-type]
    assert "set_route('A')" in str(refused.value)
    assert "set_route('B')" in str(refused.value)


@pytest.mark.parametrize("sim_type", SIMS)
def test_the_constructor_states_a_route_setting_by_the_same_rule(sim_type):
    """A route setting given to the constructor is checked as ``set_route`` checks it, and names the constructor."""
    name = sim_type.__name__
    assert sim_type(route=" a ").route == "A"
    assert (
        sim_type(route="A", approximation_profile=THIN_FILM).approximation_profile
        == THIN_FILM
    )
    with pytest.raises(TypeError) as not_a_profile:
        sim_type(route="A", approximation_profile={"name": "thin_film"})
    message = str(not_a_profile.value)
    assert f"{name}(..., approximation_profile=ApproximationProfile(" in message
    assert message.endswith(f"found 'dict'. Change it in the call to {name}.")
    with pytest.raises(ValueError, match="Route C") as route_c:
        sim_type(route="C")
    assert str(route_c.value).endswith(
        f"Call {name}(route='A') or {name}(route='B'). Change it in the call to {name}."
    )


@pytest.mark.parametrize("sim_type", SIMS)
def test_a_second_set_route_replaces_the_route_and_the_profile(sim_type):
    """One call states the whole route setting: nothing survives from the call before."""
    sim = sim_type()
    sim.set_route("A", approximation_profile=SUBSTRATE_FACE)
    sim.set_route("A")
    assert route_setting(sim) == ("A", None)
    sim.set_route("B")
    assert route_setting(sim) == ("B", None)


@pytest.mark.parametrize("sim_type", SIMS)
def test_set_route_discards_the_mesh_config_and_handoff(sim_type, monkeypatch):
    """The route is geometry: ``set_route`` falls back to ``BOUND`` like the other geometry setters."""
    sim = sim_type()
    sim.set_plan(two_pad_snapshot(monkeypatch))
    handed_off(sim)
    sim.set_route("B")
    assert sim._stage is Stage.BOUND  # noqa: SLF001
    assert (sim._mesh_result, sim.config_path, sim.handoff_plan) == (None, None, None)  # noqa: SLF001


# ---- Surface EPR ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("sim_type", SIMS)
def test_without_set_surface_epr_mesh_requests_no_bands(
    sim_type, monkeypatch, tmp_path
):
    """Both problems mesh without Surface EPR, and the geometry build is given no specs."""
    calls = stand_in_mesh_stages(monkeypatch, core, tmp_path)
    monkeypatch.setattr(layout_ports, "prepare_mesh_input", _prepared_stand_in)
    sim = sim_type()
    sim.set_plan(two_pad_snapshot(monkeypatch))
    sim.set_output_dir(tmp_path)
    sim.mesh()
    (_, geometry), *_ = calls
    assert geometry.surface_epr_specs is None


@pytest.mark.parametrize("route", ["A", "B"])
@pytest.mark.parametrize("problem", ["Eigenmode", "Electrostatic"])
def test_without_surface_epr_the_config_has_no_surface_epr_rows(
    route, problem, tmp_path
):
    """Specs of None write no ``Dielectric`` row and no Surface-EPR index entry, on either route."""
    run = Run(route, problem, tmp_path)
    config, entries, _extra = (
        run._eigenmode(None) if problem == "Eigenmode" else run._electrostatic(None)  # noqa: SLF001
    )
    assert "Postprocessing" not in config["Boundaries"]
    assert not [entry for entry in entries if isinstance(entry, SurfaceEprEntry)]


@pytest.mark.parametrize("sim_type", SIMS)
def test_an_empty_mapping_is_refused_with_the_next_step(sim_type, monkeypatch):
    """ "No Surface EPR" has one spelling: no call, or ``None`` to turn it off again."""
    sim = sim_type()
    sim.set_plan(two_pad_snapshot(monkeypatch))
    with pytest.raises(
        ValueError,
        match=r"do not call set_surface_epr\(\), or call set_surface_epr\(None\)",
    ):
        sim.set_surface_epr({})
    assert sim.surface_epr_specs is None


@pytest.mark.parametrize("sim_type", SIMS)
def test_set_surface_epr_none_clears_the_specs_and_discards_later_stages(sim_type):
    """``None`` turns Surface EPR off again, and the mesh cut with its bands is discarded."""
    sim = sim_type()
    sim._materials = STACK["materials"]  # noqa: SLF001 -- stands for set_plan()
    sim._stage = Stage.BOUND  # noqa: SLF001 -- stands for set_plan()
    sim.set_surface_epr({"MA": {**FILM, "inset_margins_um": [0.5]}})
    assert sim.surface_epr_specs is not None
    handed_off(sim)
    sim.set_surface_epr(None)
    assert sim.surface_epr_specs is None
    assert sim._stage is Stage.BOUND  # noqa: SLF001
    assert (sim._mesh_result, sim.config_path, sim.handoff_plan) == (None, None, None)  # noqa: SLF001


def test_input_summary_shows_surface_epr_not_requested_then_the_specs(monkeypatch):
    """The summary holds None (shown as "not requested") until specs are set, then the specs."""
    sim = eigenmode_sim(monkeypatch, LAYER_STACK)
    summary = sim.input_summary()
    assert summary.data["backend"]["surface_epr_specs"] is None
    assert "not requested" in _surface_section(summary._repr_html_())  # noqa: SLF001

    sim.set_surface_epr({"MA": {**FILM, "inset_margins_um": [0.5, 1.0]}})
    summary = sim.input_summary()
    specs = summary.data["backend"]["surface_epr_specs"]
    assert list(specs) == ["MA"]
    assert specs["MA"]["inset_margins_um"] == [0.5, 1.0]
    assert "not requested" not in _surface_section(summary._repr_html_())  # noqa: SLF001


def _surface_section(html: str) -> str:
    """The HTML of the summary's Surface table, up to the next heading."""
    start = html.index("Surface films and requested margins")
    return html[start : html.index("<h4>", start)]


def _prepared_stand_in(**_kwargs):
    """A prepared mesh input that stands in for the source stage: the mesh stages are stand-ins too."""
    return SimpleNamespace(
        stack=STACK,
        materials=None,
        geometry_input=object(),
        physical_plan=SimpleNamespace(context=object()),
    )
