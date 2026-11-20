"""``GeometryPlan.set_lumped_elements``: one declaration of every lumped element, resolved by ``prepare()`` (issue #283).

The layout is the two pads of ``test_simulation_drc`` with the authored sheet ``chip/P1`` from the pad at x
15 um to the electrode at x 35 um, along +x.  Every refusal is compared whole: it names the element, the
sheet's layer and its place in micrometres, and who changes it.  Then each Palace problem's realization of the
records, read from the snapshot.
"""

from __future__ import annotations

import math

import pytest
from test_simulation_drc import MATERIALS, NO_VACUUM_STACK, two_pads

from scgsim.geometry import GeometryPlan, LumpedElement, LumpedPort
from scgsim.palace import EigenmodeSim, ElectrostaticSim
from scgsim.palace.source.lumped_elements import eigenmode_ports, eigenmode_report

NETS = {"pad": ("chip/pad",), "electrode": ("chip/electrode",)}
PLACE = "its sheet on layer 9/0 at (15.000, 8.000)..(35.000, 12.000) um"
SET_LUMPED = "Change it in the call to GeometryPlan.set_lumped_elements."


def plan_of(component, *, nets=NETS, elements=()):
    """The plan of ``component`` in a thin vacuum, with ``nets`` and the lumped ``elements``."""
    plan = GeometryPlan(
        component,
        root_id="chip",
        layer_stack=NO_VACUUM_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    plan.set_vacuum_region((0.0, 0.0, 10.0))
    if nets is not None:
        plan.set_nets(nets)
    plan.set_lumped_elements(elements)
    return plan


@pytest.mark.parametrize(
    ("make", "error", "message"),
    [
        (
            lambda: LumpedElement("chip/P1"),
            ValueError,
            "LumpedElement('chip/P1'): give at least one of resistance_ohm, inductance_h and capacitance_f. "
            "Change it in the record given to GeometryPlan.set_lumped_elements.",
        ),
        (
            lambda: LumpedElement("chip/P1", inductance_h=-1e-9),
            ValueError,
            "LumpedElement('chip/P1'): inductance_h must be a finite number above zero, not -1e-09. "
            "Change it in the record given to GeometryPlan.set_lumped_elements.",
        ),
        (
            lambda: LumpedElement("chip/P1", capacitance_f=math.nan),
            ValueError,
            "LumpedElement('chip/P1'): capacitance_f must be a finite number above zero, not nan. "
            "Change it in the record given to GeometryPlan.set_lumped_elements.",
        ),
        (
            lambda: LumpedElement("chip/P1", resistance_ohm=True),
            TypeError,
            "LumpedElement('chip/P1'): resistance_ohm must be a number, not True. "
            "Change it in the record given to GeometryPlan.set_lumped_elements.",
        ),
        (
            lambda: LumpedElement("", inductance_h=1e-9),
            ValueError,
            "LumpedElement: sheet must be the nonempty qualified name of an authored port sheet, not ''. "
            "Change it in the record given to GeometryPlan.set_lumped_elements.",
        ),
        (
            lambda: LumpedPort("chip/P1", reference_net="", impedance_ohm=50.0),
            ValueError,
            "LumpedPort('chip/P1'): reference_net must be the nonempty ID of a Net, not ''. "
            "Change it in the record given to GeometryPlan.set_lumped_elements.",
        ),
        (
            lambda: LumpedPort("chip/P1", reference_net="pad", impedance_ohm=0.0),
            ValueError,
            "LumpedPort('chip/P1'): impedance_ohm must be a finite number above zero, not 0.0. "
            "Change it in the record given to GeometryPlan.set_lumped_elements.",
        ),
    ],
    ids=[
        "no-value",
        "negative",
        "nan",
        "bool",
        "no-sheet",
        "no-reference",
        "zero-impedance",
    ],
)
def test_a_record_with_a_value_no_solver_can_realize_is_refused_when_made(
    make, error, message
):
    """Each value is a finite number above zero, an element gives at least one, and every name is nonempty."""
    with pytest.raises(error) as raised:
        make()
    assert str(raised.value) == message


def test_the_setter_refuses_what_is_not_one_list_of_records_and_keeps_the_earlier_list(
    monkeypatch,
):
    """Not a sequence, an entry of another type, or a sheet named twice; each refused call leaves the list."""
    plan = plan_of(two_pads(monkeypatch, gap_um=10.0, port=True))
    element = LumpedElement("chip/P1", inductance_h=1e-9)
    plan.set_lumped_elements([element])
    refusals = [
        (
            "chip/P1",
            TypeError,
            "elements must be a sequence of LumpedElement and LumpedPort, not 'chip/P1'",
        ),
        (
            {"chip/P1": 1e-9},
            TypeError,
            "elements must be a sequence of LumpedElement and LumpedPort",
        ),
        (
            [1e-9],
            TypeError,
            "every entry must be a LumpedElement or a LumpedPort, not 1e-09",
        ),
        (
            [element, LumpedPort("chip/P1", reference_net="pad", impedance_ohm=50.0)],
            ValueError,
            "sheets ['chip/P1'] are declared more than once; a sheet carries one element",
        ),
    ]
    for value, error, message in refusals:
        with pytest.raises(error) as raised:
            plan.set_lumped_elements(value)
        assert str(raised.value).startswith(f"lumped elements: {message}")
        assert str(raised.value).endswith(SET_LUMPED)
    (record,) = plan.prepare().lumped_elements
    assert record.declaration == element


@pytest.mark.parametrize(
    ("case", "message"),
    [
        (
            "unknown sheet",
            "lumped element 'chip/P2': the layout authors no port sheet of that name; its port sheets are "
            f"['chip/P1']. {SET_LUMPED}",
        ),
        (
            "one conductor",
            f"lumped element 'chip/P1', {PLACE}: it lands on 1 conductor, ['chip/pad_0000']; draw the sheet so "
            "that it lands on exactly two conductors of its target layer 'metal', one at each end. Change it in "
            "the layout.",
        ),
        (
            "one Net",
            f"lumped element 'chip/P1', {PLACE}: both conductors it lands on, ['chip/pad_0000', "
            "'chip/electrode_0000'], are in Net 'n'; a lumped element joins two Nets. Change it in the layout or "
            "the call to GeometryPlan.set_nets.",
        ),
        (
            "reference Net",
            f"lumped port 'chip/P1', {PLACE}: reference_net 'ground' is not the Net of either conductor it lands "
            f"on (chip/pad_0000 in 'pad', chip/electrode_0000 in 'electrode'). {SET_LUMPED}",
        ),
    ],
)
def test_prepare_refuses_an_element_the_layout_cannot_carry_where_it_is(
    monkeypatch, case, message
):
    """A name the layout does not author, a sheet on one conductor, two conductors of one Net, a wrong reference."""
    one_conductor = ((60, 0), (90, 0), (90, 20), (60, 20))
    component = two_pads(
        monkeypatch,
        gap_um=10.0,
        port=True,
        electrode=one_conductor if case == "one conductor" else None,
    )
    nets = {"n": ("chip/pad", "chip/electrode")} if case == "one Net" else NETS
    element = {
        "unknown sheet": LumpedElement("chip/P2", inductance_h=1e-9),
        "reference Net": LumpedPort(
            "chip/P1", reference_net="ground", impedance_ohm=50.0
        ),
    }.get(case, LumpedElement("chip/P1", inductance_h=1e-9))
    plan = plan_of(component, nets=nets, elements=[element])
    with pytest.raises(ValueError) as raised:
        plan.prepare()
    assert str(raised.value) == message
    sim = EigenmodeSim
    with pytest.raises(ValueError) as checked:
        sim.simulation_drc(plan)
    assert str(checked.value) == message


def test_the_records_carry_the_sheet_its_conductors_in_direction_order_and_the_ports_first(
    monkeypatch,
):
    """The snapshot's records: the port numbered first, each conductor with its Net, start before end."""
    port = LumpedPort("chip/P1", reference_net="electrode", impedance_ohm=50.0)
    plan = plan_of(two_pads(monkeypatch, gap_um=10.0, port=True), elements=[port])
    (record,) = plan.prepare().lumped_elements
    assert (record.number, record.name, record.port_sheet_id) == (
        1,
        "chip/P1",
        "PORT_SHEET__chip/P1__0000",
    )
    assert (record.source_layer, record.target_layer, record.direction) == (
        "9/0",
        "metal",
        (1.0, 0.0, 0.0),
    )
    assert (record.start.source_entity_id, record.start.net_id) == (
        "chip/pad_0000",
        "pad",
    )
    assert (record.end.source_entity_id, record.end.net_id) == (
        "chip/electrode_0000",
        "electrode",
    )
    assert (record.signal, record.reference) == (record.start, record.end)
    assert record.place() == PLACE.removeprefix("its sheet ")


def test_eigenmode_realizes_each_element_as_a_palace_port_and_a_port_as_its_impedance(
    monkeypatch,
):
    """Eigenmode's ``LumpedPort`` keys are R, L and C in that order, a port's impedance as R; the report says so."""
    component = two_pads(monkeypatch, gap_um=10.0, port=True)
    element = LumpedElement(
        "chip/P1", capacitance_f=1e-15, inductance_h=1e-8, resistance_ohm=1e6
    )
    (port,) = eigenmode_ports(plan_of(component, elements=[element]).prepare())
    assert (port.name, port.layer, list(port.circuit.items())) == (
        "chip/P1",
        "metal",
        [("R", 1e6), ("L", 1e-8), ("C", 1e-15)],
    )
    load = LumpedPort("chip/P1", reference_net="pad", impedance_ohm=50.0)
    snapshot = plan_of(component, elements=[load]).prepare()
    (port,) = eigenmode_ports(snapshot)
    assert dict(port.circuit) == {"R": 50.0}
    assert eigenmode_report(snapshot) == (
        "P1 chip/P1: LumpedPort, R 50.0 ohm: the excitation port as its impedance, a load (the Q is the "
        "loaded Q)",
    )


def test_each_palace_check_says_what_it_does_with_the_lumped_elements(monkeypatch):
    """Eigenmode cuts the element's sheet as a port; Electrostatic realizes none; a sheet without one is left out."""
    component = two_pads(monkeypatch, gap_um=10.0, port=True)
    plan = plan_of(component, elements=[LumpedElement("chip/P1", inductance_h=1e-8)])
    assert EigenmodeSim.simulation_drc(plan).lumped_elements == (
        "P1 chip/P1: LumpedPort, L 1e-08 H",
    )
    assert ElectrostaticSim.simulation_drc(plan).lumped_elements == (
        "lumped element 'chip/P1' (L 1e-08 H): not realized: the capacitance matrix is the conductors', with "
        "every lumped element open",
    )
    empty = plan_of(component)
    report = EigenmodeSim.simulation_drc(empty)
    assert report.lumped_elements == (
        "chip/P1: no lumped element on the plan; not modeled",
    )
    assert report.render_text().endswith(
        "lumped elements:\n  chip/P1: no lumped element on the plan; not modeled\n"
    )
    assert ElectrostaticSim.simulation_drc(empty).lumped_elements == ()


def test_the_input_summary_shows_each_element_as_the_simulation_realizes_it(
    monkeypatch,
):
    """``input_summary()`` gives each record with what the problem makes of it, and no simulation ports of its own."""
    component = two_pads(monkeypatch, gap_um=10.0, port=True)
    snapshot = plan_of(
        component, elements=[LumpedElement("chip/P1", inductance_h=1e-8)]
    ).prepare()
    for simulation, realization in (
        (EigenmodeSim, "LumpedPort, L 1e-08 H"),
        (
            ElectrostaticSim,
            "not realized: the capacitance matrix is the conductors', with every lumped element open",
        ),
    ):
        sim = simulation()
        sim.set_plan(snapshot)
        backend = sim.input_summary().data["backend"]
        (entry,) = backend["lumped_elements"]
        assert (entry["name"], entry["number"], entry["inductance_h"]) == (
            "chip/P1",
            1,
            1e-8,
        )
        assert entry["realization"] == realization
        assert "ports" not in backend
