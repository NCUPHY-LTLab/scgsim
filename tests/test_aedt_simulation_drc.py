"""``scgsim.aedt.EigenmodeSim.simulation_drc`` runs the geometry half of its own AEDT pipeline on a plan, without AEDT.

Every test builds a real SGB model and its STEP (OCP), as ``prepare_handoff()`` does; none starts AEDT.  Where
the check is compared with the simulation, the simulation is set up from the settings a user would write and
its ``prepare_handoff()`` or the import's first step raises the error compared.
"""

from __future__ import annotations

import pytest
from test_simulation_drc import (
    BANDED,
    LAYER_STACK,
    NO_VACUUM_STACK,
    layout_plan,
    two_pads,
)
from test_simulation_drc_meshing_stage import PADS, scene

pytest.importorskip("OCP")

import scgsim.aedt.drc as drc_module  # noqa: E402
import scgsim.palace as palace  # noqa: E402
from scgsim.aedt import EigenmodeSim  # noqa: E402
from scgsim.aedt.drc import SimulationDrcReport  # noqa: E402
from scgsim.aedt.runtime.families.sgb_boundaries import BoundaryPlanError  # noqa: E402
from scgsim.aedt.runtime.families.sgb_import import planned_import  # noqa: E402
from scgsim.aedt.specs.common import LumpedRlc  # noqa: E402
from scgsim.contracts.sgb_sidecar import read_sidecar  # noqa: E402
from scgsim.geometry import LumpedElement, LumpedPort  # noqa: E402
from scgsim.geometry.sgb.output.exchange.guard import read_step_guard  # noqa: E402

MODELINGS = ["solid", "thin_film"]


@pytest.mark.parametrize("modeling", MODELINGS)
def test_the_planned_import_holds_each_guarded_face_once_per_solid_and_each_free_face_as_its_sheet(
    monkeypatch, modeling
):
    """``planned_import`` is the import the guard plans: the copies a Surface-EPR row selects from."""
    seen = []

    def recording(model):
        guard = read_step_guard(model.path("guard"))
        seen.append((guard, planned_import(guard)))

    import scgsim.aedt.eigenmode as eigenmode_module  # noqa: PLC0415 -- the module whose import is replaced

    monkeypatch.setattr(eigenmode_module, "check_eigenmode_model", recording)
    check(two_pads(monkeypatch, gap_um=10.0, port=True), modeling=modeling)
    ((guard, planned),) = seen
    for solid in guard["solids"]:
        for face in solid["faces"]:
            assert solid["aedt_name"] in planned.face_copies[face["face"]]
    assert set(planned.sheets) == {face["face"] for face in guard["free_faces"]}
    for face, name in planned.sheets.items():
        assert planned.face_copies[face] == {name: 0}


def check(  # noqa: PLR0913 -- the layout, each plan setting an author writes, and the check's own options
    component,
    *,
    layer_stack=LAYER_STACK,
    nets=None,
    lumped_elements=None,
    surface_epr_specs=None,
    **options,
):
    """``EigenmodeSim.simulation_drc`` on the plan a layout author builds for ``component``.

    ``lumped_elements`` None declares every authored sheet as a 1 nH element, as ``layout_plan`` does.
    """
    plan = layout_plan(
        component,
        layer_stack=layer_stack,
        nets=nets,
        lumped_elements=lumped_elements,
        surface_epr_specs=surface_epr_specs,
    )
    return EigenmodeSim.simulation_drc(plan, **options)


def handoff_refusal(plan, modeling, tmp_path):
    """The error ``prepare_handoff()`` raises for ``plan``, with the plan's lumped elements, as ``simulation_drc`` has them."""
    snapshot = plan.prepare()
    sim = EigenmodeSim(modeling=modeling, aedt_version="2024.2")
    sim.set_plan(snapshot)
    sim.set_output_dir(tmp_path / "run")
    sim.set_eigenmode(project_name="check", design_name="eigenmode")
    with pytest.raises(Exception) as error:  # noqa: PT011 -- the type is what the test compares
        sim.prepare_handoff()
    return error.value


@pytest.mark.parametrize("modeling", MODELINGS)
@pytest.mark.parametrize("port", [False, True])
def test_a_layout_that_models_passes_on_both_routes(monkeypatch, modeling, port):
    """Two separate pads, with or without an authored port, build as the Route A and the Route C model.

    The port, a 1 nH element on the plan, is an EPR junction, and the report says so.
    """
    report = check(two_pads(monkeypatch, gap_um=10.0, port=port), modeling=modeling)
    assert isinstance(report, SimulationDrcReport)
    assert report.passed
    junction = (
        "lumped elements:\n"
        "  chip/P1: EPR junction 'chip/P1', Lumped RLC L 1e-09 H, C 0.0 F\n"
    )
    assert report.render_text() == "geometry stage: passed\n" + (
        junction if port else ""
    )


def built_models(monkeypatch, **options):
    """The sidecar of every model ``simulation_drc`` builds, read before its folder is removed."""
    sidecars = []
    real = drc_module.check_eigenmode_model

    def recording(model):
        sidecars.append(read_sidecar(model.path("sidecar")))
        real(model)

    monkeypatch.setattr(drc_module, "check_eigenmode_model", recording)
    import scgsim.aedt.eigenmode as eigenmode_module  # noqa: PLC0415 -- the module whose import is replaced

    monkeypatch.setattr(eigenmode_module, "check_eigenmode_model", recording)
    check(two_pads(monkeypatch, gap_um=10.0, port=True), **options)
    return sidecars


@pytest.mark.parametrize(("modeling", "route"), [("solid", "C"), ("thin_film", "A")])
def test_the_check_builds_the_route_its_modeling_realizes(monkeypatch, modeling, route):
    """``solid`` is checked as the Route C model and ``thin_film`` as the Route A model (decision 7)."""
    (sidecar,) = built_models(monkeypatch, modeling=modeling)
    assert sidecar["model"]["route"] == route


def test_surface_epr_specs_become_bands_and_every_row_is_selected(monkeypatch):
    """With specs, the MA rows carry their bands; without, no row has a band."""
    (plain,) = built_models(monkeypatch, modeling="solid")
    assert all(row["band"] is None for row in plain["surface_epr_row"].values())
    (banded,) = built_models(monkeypatch, modeling="solid", surface_epr_specs=BANDED)
    bands = {
        row["interface_type"]
        for row in banded["surface_epr_row"].values()
        if row["band"] is not None
    }
    assert bands == {"MA"}


@pytest.mark.parametrize("modeling", MODELINGS)
def test_a_refused_layout_raises_the_error_prepare_handoff_raises(
    monkeypatch, tmp_path, modeling
):
    """Overlapping electrodes are refused with the type and message of ``prepare_handoff()``."""
    component = two_pads(monkeypatch, gap_um=-5.0)
    expected = handoff_refusal(layout_plan(component), modeling, tmp_path)
    with pytest.raises(type(expected)) as raised:
        check(component, modeling=modeling)
    assert str(raised.value) == str(expected)


def test_the_check_builds_the_vacuum_the_plan_sets(monkeypatch):
    """A stack with no ``VACUUM_REGION`` and the plan's own vacuum region: the Region box is the plan's."""
    (sidecar,) = built_models(
        monkeypatch, modeling="solid", layer_stack=NO_VACUUM_STACK
    )
    vacuum = [
        volume for volume in sidecar["volume"].values() if volume["kind"] == "vacuum"
    ]
    assert vacuum


@pytest.mark.parametrize("modeling", MODELINGS)
def test_a_plan_without_a_vacuum_is_refused_as_prepare_handoff_refuses_it(
    monkeypatch, tmp_path, modeling
):
    """The check adds no vacuum (Issue #266): a plan with none is refused with ``prepare_handoff()``'s error."""
    plan = layout_plan(
        two_pads(monkeypatch, gap_um=10.0),
        layer_stack=NO_VACUUM_STACK,
        vacuum_padding_um=None,
    )
    expected = handoff_refusal(plan, modeling, tmp_path)
    assert "set_vacuum_region" in str(expected)
    with pytest.raises(type(expected)) as raised:
        EigenmodeSim.simulation_drc(plan, modeling=modeling)
    assert str(raised.value) == str(expected)


def test_the_check_leaves_the_callers_plan_as_it_was(monkeypatch):
    """The plan prepares to the same snapshot after the check as before it."""
    plan = layout_plan(two_pads(monkeypatch, gap_um=10.0, port=True))
    before = plan.prepare()
    assert EigenmodeSim.simulation_drc(plan, modeling="solid").passed
    after = plan.prepare()
    assert (after.gds_sha256, after.stack) == (before.gds_sha256, before.stack)


def test_a_first_argument_that_is_not_a_plan_is_refused_with_the_plan_to_build(
    monkeypatch,
):
    """A Component or a prepared snapshot is refused before the modeling is read, with the plan to build."""
    component = two_pads(monkeypatch, gap_um=10.0)
    for value in (component, layout_plan(component).prepare()):
        with pytest.raises(TypeError) as raised:
            EigenmodeSim.simulation_drc(value, modeling="route_b")
        message = str(raised.value)
        assert message.startswith(
            "the first argument of EigenmodeSim.simulation_drc: must be the GeometryPlan the "
            f"simulation is given, not {type(value).__name__}; build it as plan = GeometryPlan("
            "component=..., root_id=..., layer_stack=..., material_records=..., "
            "coupon_padding_um=...); plan.set_nets({...})"
        )


# The two pads' Nets: the pad (A) and the electrode (B) the authored port sheet joins.
JUNCTION_NETS = {"A": ("chip/pad",), "B": ("chip/electrode",)}
SHEET_EDGES = {
    f"PORT_SHEET__chip/P1__0000/polygon0/ring0/edge{index}" for index in range(4)
}


def port_sheet_carriers(sidecar):
    """The sheet curves that bound the model's one port face."""
    (face,) = [
        face for face in sidecar["face"].values() if face["group"].startswith("PORT::")
    ]
    return {
        carrier
        for loop in face["loops"]
        for edge in loop["edges"]
        for carrier in sidecar["edge"][edge]["carriers"]
        if carrier.startswith("PORT_SHEET__")
    }


@pytest.mark.parametrize("modeling", MODELINGS)
def test_every_lumped_element_is_checked_with_its_port_sheet_cut(monkeypatch, modeling):
    """Each kind of lumped element is in the model on its sheet cut to the rectangle between its conductors.

    An inductance alone is an EPR junction named by its sheet, from the pad (A) at the start of the authored
    direction to the electrode (B); a resistance is a parallel Lumped RLC; an excitation port is a parallel
    Lumped RLC of its impedance.  The port face is bounded by the sheet's four cut edges in every case.  A
    sheet that carries no element is not in the model.
    """
    component = two_pads(monkeypatch, gap_um=10.0, port=True)
    models = []
    real = drc_module.check_eigenmode_model

    def recording(model):
        models.append((model, read_sidecar(model.path("sidecar"))))
        real(model)

    import scgsim.aedt.eigenmode as eigenmode_module  # noqa: PLC0415 -- the module whose import is replaced

    monkeypatch.setattr(eigenmode_module, "check_eigenmode_model", recording)
    sheet = "PORT_SHEET__chip/P1__0000"
    elements = {
        "junction": LumpedElement("chip/P1", inductance_h=1e-8, capacitance_f=2e-15),
        "load": LumpedElement("chip/P1", resistance_ohm=50.0),
        "port": LumpedPort("chip/P1", reference_net="B", impedance_ohm=50.0),
    }
    for element in elements.values():
        assert check(
            component, nets=JUNCTION_NETS, lumped_elements=[element], modeling=modeling
        ).passed
    report = check(component, nets=JUNCTION_NETS, lumped_elements=[], modeling=modeling)
    assert report.lumped_elements == (
        "chip/P1: no lumped element on the plan; not modeled",
    )
    (junction, _), (load, _), (port, _), (empty, unmodeled) = models
    for _, sidecar in models[:3]:
        assert port_sheet_carriers(sidecar) == SHEET_EDGES
    ((bound,),) = [junction.junctions]
    assert (bound.port_group, bound.junction.junction_id) == (
        f"PORT::{sheet}",
        "chip/P1",
    )
    assert (bound.junction.terminal_a_net, bound.junction.terminal_b_net) == ("A", "B")
    assert (bound.junction.inductance_h, bound.junction.capacitance_f) == (1e-8, 2e-15)
    assert load.lumped_rlcs == (LumpedRlc(sheet, "parallel", resistance_ohm=50.0),)
    assert port.lumped_rlcs == (LumpedRlc(sheet, "parallel", resistance_ohm=50.0),)
    assert (empty.junctions, empty.lumped_rlcs) == ((), ())
    assert not any(group.startswith("PORT::") for group in unmodeled["group"])


def test_a_modeling_that_is_not_solid_or_thin_film_is_refused_before_the_plan_is_prepared(
    monkeypatch,
):
    """The modeling selector is checked before the plan is prepared, with the message ``EigenmodeSim`` gives."""
    plan = layout_plan(two_pads(monkeypatch, gap_um=10.0))

    def prepared(*_args, **_kwargs):
        raise AssertionError("the plan was prepared")

    monkeypatch.setattr(plan, "prepare", prepared)
    with pytest.raises(ValueError, match="explicitly 'solid' or 'thin_film'"):
        EigenmodeSim.simulation_drc(plan, modeling="route_b")


def test_the_check_has_no_surface_epr_argument_and_the_simulation_no_setter(
    monkeypatch,
):
    """Surface EPR is the plan's (``GeometryPlan.set_surface_epr``): the check reads it from the plan it is given."""
    plan = layout_plan(two_pads(monkeypatch, gap_um=10.0))
    with pytest.raises(TypeError, match="surface_epr_specs"):
        EigenmodeSim.simulation_drc(plan, modeling="solid", surface_epr_specs=BANDED)  # type: ignore[call-arg]
    assert not hasattr(EigenmodeSim, "set_surface_epr")


def test_nets_that_become_one_aedt_name_are_refused_naming_both(monkeypatch):
    """Two Nets whose IDs sanitise alike are refused by the preparation the check runs."""
    nets = {"chip/A": ("chip/pad",), "chip_A": ("chip/electrode",)}
    with pytest.raises(ValueError, match="chip/A") as raised:
        check(two_pads(monkeypatch, gap_um=10.0), modeling="solid", nets=nets)
    assert "chip_A" in str(raised.value)


def test_the_import_plan_refusal_is_the_checks_refusal(monkeypatch):
    """A refusal of the import's first step is raised by the check unchanged."""

    def refused(*_args, **_kwargs):
        raise BoundaryPlanError("port sheet 'PORT::P1' has no face in the sidecar")

    monkeypatch.setattr(drc_module, "plan_import", refused)
    with pytest.raises(BoundaryPlanError, match="PORT::P1"):
        check(two_pads(monkeypatch, gap_um=10.0, port=True), modeling="solid")


@pytest.mark.parametrize("modeling", MODELINGS)
def test_a_palace_pass_does_not_stand_for_the_aedt_check(monkeypatch, modeling):
    """Two Net IDs that become one AEDT name: Palace models the plan, AEDT's import cannot name its Nets."""
    plan = layout_plan(
        two_pads(monkeypatch, gap_um=10.0),
        nets={"chip/A": ("chip/pad",), "chip_A": ("chip/electrode",)},
    )
    assert palace.EigenmodeSim.simulation_drc(plan).passed
    with pytest.raises(ValueError, match="chip_A"):
        EigenmodeSim.simulation_drc(plan, modeling=modeling)


@pytest.mark.parametrize("modeling", MODELINGS)
def test_an_aedt_pass_does_not_stand_for_the_palace_check(monkeypatch, modeling):
    """A 1 nm jog in a pad's edge: AEDT builds and imports it, Palace's meshing stage refuses it."""
    plan = layout_plan(scene(monkeypatch, PADS["jog"]), layer_stack=NO_VACUUM_STACK)
    assert EigenmodeSim.simulation_drc(plan, modeling=modeling).passed
    with pytest.raises(palace.SimulationDrcError):
        palace.EigenmodeSim.simulation_drc(plan)
