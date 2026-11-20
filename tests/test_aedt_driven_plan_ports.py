"""HFSS Driven Terminal reads its two ports from the plan's lumped elements (issue #283), without AEDT.

The layout is the two-port source of ``sgb_driven_inputs`` drawn as a component: two pads beside one ground
strip, and an authored port sheet from the strip up to each pad, along +y.  Every test prepares the plan and
writes the real SGB model; none starts AEDT.
"""

from __future__ import annotations

import json

import pytest
from test_simulation_drc import LAYER_STACK, MATERIALS

pytest.importorskip("OCP")

from scgsim.aedt import (  # noqa: E402
    FrequencySweepSpec,
    HfssRunControl,
    prepare_hfss_driven_from_geometry,
)
from scgsim.aedt.preparation.lumped_elements import hfss_driven_inputs  # noqa: E402
from scgsim.aedt.specs.common import LumpedTerminalPort  # noqa: E402
from scgsim.aedt.specs.parse import parse_aedt_spec  # noqa: E402
from scgsim.contracts.sgb_sidecar import read_sidecar  # noqa: E402
from scgsim.geometry import GeometryPlan, LumpedElement, LumpedPort  # noqa: E402

pytestmark = pytest.mark.occt

RUN = HfssRunControl("Setup", "Sweep", FrequencySweepSpec(1, 2, points=5))
NETS = {"P1": ("chip/pad_1",), "P2": ("chip/pad_2",), "G": ("chip/ground",)}
# Each conductor: its Entity, its GDS layer and its outline in um.
CONDUCTORS = (
    ("pad_1", (1, 0), ((0, 0), (5, 0), (5, 3), (0, 3))),
    ("pad_2", (2, 0), ((10, 0), (15, 0), (15, 3), (10, 3))),
    ("ground", (3, 0), ((-2, -6), (17, -6), (17, -3), (-2, -3))),
)
# Each sheet: its port name and x extent; it runs from y = -4 in the strip to y = 1 in its pad.
SHEETS = (("port_1", 1, 2), ("port_2", 11, 14), ("port_3", 3, 4))


def two_port_layout(monkeypatch, sheets=SHEETS[:2]):
    """The two pads, the ground strip and the authored port sheets ``sheets``, each pointing from the strip to a pad."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    component = gf.Component()
    for _, layer, outline in CONDUCTORS:
        component.add_polygon(outline, layer=layer)
    for name, x0, x1 in sheets:
        component.add_polygon(((x0, -4), (x1, -4), (x1, 1), (x0, 1)), layer=(9, 0))
        component.add_port(
            name=name,
            center=((x0 + x1) / 2, -1.5),
            width=x1 - x0,
            orientation=90,
            layer=(9, 0),
        )
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {"semantic_id": name, "level": "metal", "gds_layer": list(layer)}
            for name, layer, _ in CONDUCTORS
        ],
        "ports": [
            {"name": name, "layer": [9, 0], "target_layer": "metal"}
            for name, _, _ in sheets
        ],
    }
    return component


def two_port_plan(monkeypatch, elements, *, sheets=SHEETS[:2]):
    """The plan of :func:`two_port_layout` with ``elements`` declared."""
    plan = GeometryPlan(
        two_port_layout(monkeypatch, sheets),
        root_id="chip",
        layer_stack=LAYER_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    plan.set_nets(NETS)
    plan.set_lumped_elements(elements)
    return plan


PORTS = [
    LumpedPort("chip/port_2", reference_net="G", impedance_ohm=50.0),
    LumpedPort("chip/port_1", reference_net="G", impedance_ohm=25.0),
]


def handoff(plan, root, modeling="solid"):
    """The Driven Terminal handoff of ``plan``, prepared as a study prepares it."""
    return prepare_hfss_driven_from_geometry(
        plan.prepare(),
        aedt_version="2024.2",
        modeling=modeling,
        output_dir=root,
        project_name="Driven",
        design_name="Design",
        run_control=RUN,
    )


@pytest.mark.parametrize("modeling", ["solid", "thin_film"])
def test_the_driven_ports_are_the_plans_ports_in_its_order(
    monkeypatch, tmp_path, modeling
):
    """Port 1 is the plan's first ``LumpedPort``: its sheet, its signal pad, the ground as reference, its impedance.

    The handoff's request carries exactly these two lumped Terminal ports, and its model has exactly their two
    sheets, each cut to the rectangle between the strip and its pad.
    """
    plan = handoff(two_port_plan(monkeypatch, PORTS), tmp_path / "run", modeling)
    spec = parse_aedt_spec(
        json.loads(plan.spec_path.read_text()), base_dir=plan.run_dir
    )
    assert spec.ports == (
        LumpedTerminalPort(
            1,
            "chip_port_2",
            "PORT_SHEET__chip/port_2__0000",
            ("chip/pad_2_0000",),
            ("chip/ground_0000",),
            50.0,
        ),
        LumpedTerminalPort(
            2,
            "chip_port_1",
            "PORT_SHEET__chip/port_1__0000",
            ("chip/pad_1_0000",),
            ("chip/ground_0000",),
            25.0,
        ),
    )
    sidecar = read_sidecar(spec.model.path("sidecar"))
    assert sorted(port["group"] for port in sidecar["port"].values()) == [
        "PORT::PORT_SHEET__chip/port_1__0000",
        "PORT::PORT_SHEET__chip/port_2__0000",
    ]


def test_a_sheet_without_a_lumped_element_is_left_out_of_the_driven_model(monkeypatch):
    """A third authored sheet that the plan does not declare is not in the model, and the realization says so."""
    plan = two_port_plan(monkeypatch, PORTS, sheets=SHEETS)
    lumped = hfss_driven_inputs(plan.prepare())
    assert sorted(
        region.port_sheet_id for region in lumped.build_input.port_sheet_regions
    ) == [
        "PORT_SHEET__chip/port_1__0000",
        "PORT_SHEET__chip/port_2__0000",
    ]
    assert (
        lumped.report[-1] == "chip/port_3: no lumped element on the plan; not modeled"
    )
    assert lumped.report[0] == (
        "chip/port_2: lumped Terminal port 1 'chip_port_2', Terminal on chip/pad_2_0000, reference chip/ground_0000"
    )


def test_a_lumped_element_in_a_driven_plan_is_refused_by_name(monkeypatch, tmp_path):
    """HFSS Driven Terminal does not realize an element: the refusal names it, its sheet's layer and place."""
    elements = [*PORTS, LumpedElement("chip/port_3", inductance_h=1e-8)]
    plan = two_port_plan(monkeypatch, elements, sheets=SHEETS)
    with pytest.raises(ValueError) as raised:
        handoff(plan, tmp_path / "run")
    assert str(raised.value) == (
        "lumped element 'chip/port_3' (L 1e-08 H), its sheet on layer 9/0 at (3.000, -4.000)..(4.000, 1.000) um: "
        "HFSS Driven Terminal does not realize a lumped element yet: its import assigns Lumped RLCs outside "
        "a Driven design only, and none was checked in AEDT; leave it out of the plan of this calculation. "
        "Change it in the call to GeometryPlan.set_lumped_elements."
    )
    assert not (tmp_path / "run").exists()


def test_a_driven_plan_without_two_ports_is_refused_naming_its_ports(
    monkeypatch, tmp_path
):
    """HFSS Driven Terminal V1 takes two ports; a plan that declares one is refused before anything is built."""
    with pytest.raises(ValueError) as raised:
        handoff(two_port_plan(monkeypatch, PORTS[:1]), tmp_path / "run")
    assert str(raised.value) == (
        "the plan's excitation ports: HFSS Driven Terminal V1 takes two lumped ports; the plan declares 1: "
        "['chip/port_2']. Change it in the call to GeometryPlan.set_lumped_elements."
    )
