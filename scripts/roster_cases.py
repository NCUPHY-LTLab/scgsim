# ruff: noqa: PLC0415 -- gdsfactory and the PDK are imported where a layout is built: the PDK is not in SCGSim's own lock, and the runner's record handling loads without it
"""The roster of issue 141 and the simulation settings it is tested with.

The roster is the list of OrcaPen SC PDK layouts that define "Palace stable": SimplePad, the circular pad, the
airbridge coupon (single die), the flip-chip pad, ``disk_flip_chip_qubit`` at its default settings (flip
chip), and, for Palace Driven, the launcher-to-launcher layout of issue 187.  Each is a :class:`Layout`: the
component, drawn at ``curve_sampling="simulation"``, the PDK layer stack, the registered instances, the Net
assignment and the lumped elements its simulation's plan is given.

The settings: the plan (:func:`geometry_plan`) has root ``chip``, zero coupon padding, when the layout's stack
has no vacuum of its own, the vacuum the Human set for testing layouts (issue 182), padding 0 um in x and y and
1000 um below and above, and Surface EPR with the PDK's literature-central MA (metal-air), MS (metal-substrate)
and SA (substrate-air) presets, cut as bands at margins 0.5 and 1.0 um by type; an Eigenmode layout's plan
declares every authored port as a 10 nH lumped element (issue 283), and the Driven layout's plan declares its
two launcher sheets as 50 ohm ports; the simulation runs on Route B.  ``simulation_drc`` and ``mesh()`` are
given the same plan (issues 266, 268).

The launcher-to-launcher layout is composed here from PDK cells (:func:`launcher_to_launcher`) until the PDK
draws it as one cell with sheets that can be ports (orcapen-sc-pdk issue 67).  The Driven tests and the
acceptance record ``experiments/palace-driven`` build it from here, so the nightly check, the tests and the
recorded Palace runs share one layout.

``scripts/check_roster.py`` runs the roster every night.  The Palace baseline experiment
(``experiments/palace-stable-baseline``) builds its other layouts from the same helpers and measures them with
the same settings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from scgsim.geometry import GeometryPlan, LumpedElement, LumpedPort
from scgsim.palace import DrivenSim, EigenmodeSim, ElectrostaticSim

SAMPLING = "simulation"
MARGINS_UM = [0.5, 1.0]
PORT_INDUCTANCE_H = 10e-9
# (x, y, z) padding in um, on both sides of each axis: the vacuum ends at the die's planar edge and extends
# 1000 um below and above it (the Human, issue 182).
VACUUM_PADDING_UM = (0.0, 0.0, 1000.0)

# The launcher-to-launcher layout of issue 187 (the run of its plan's fact 9): two PDK launchers, each a 150 um
# pad tapering over 150 um to the CPW with 85 um of gap behind and beside the pad, joined by a 900 um
# ``cpw_6_7_6`` straight on a 1870 x 900 um die with no dicing frame.
LINE_UM = 900.0
DIE_UM = (1870.0, 900.0)
PAD_UM = 150.0
TAPER_UM = 150.0
BACK_GAP_UM = 85.0
# How far each port sheet reaches into the pad and into the ground: 10 grid steps, which the meshing stage
# accepts for a sheet as wide as the pad; 1 grid step leaves 1 nm wall columns it refuses (issue 187, fact 5).
LANDING_UM = 0.01
PORT_IMPEDANCE_OHM = 50.0
LAUNCHER_NETS = {"signal": ("chip/SIGNAL",), "ground": ("chip/GROUND",)}
# Left first, so the left launcher is port 1: row and column 1 of the S-matrix.
LAUNCHER_PORT_SHEETS = ("chip/o_lumped_left", "chip/o_lumped_right")
# A GDS layer no stack level reads: the launcher's own sheet goes there and is not modeled.
_UNUSED_LAYER = (250, 0)

Sense = Literal["pdk", "reversed"]


@dataclass(frozen=True)
class Layout:
    """One layout as its simulation receives it.

    ``lumped_elements`` are the plan's lumped elements when the layout states them; otherwise an Eigenmode
    layout's plan declares every authored port as a ``PORT_INDUCTANCE_H`` element (:func:`geometry_plan`).
    """

    component: Any
    layer_stack: Any
    # "eigenmode" (its plan declares a junction element), "electrostatic" or "driven" (its plan declares ports)
    simulation: str
    instances: Mapping[str, Any] = field(default_factory=dict)
    nets: Mapping[str, Sequence[str]] | None = None
    lumped_elements: Sequence[LumpedElement | LumpedPort] | None = None


def single_die(
    name: str,
    *,
    nets=None,
    airbridges: bool = False,
    simulation: str = "eigenmode",
    **params,
) -> Callable[[], Layout]:
    """The builder of a PDK cell on the single-die layer stack, with ``params`` passed to the cell."""

    def build() -> Layout:
        import gdsfactory as gf
        from orcapen_sc_pdk.tech import get_single_die_layer_stack

        component = gf.get_component(name, curve_sampling=SAMPLING, **params)
        stack = get_single_die_layer_stack(include_airbridges=airbridges)
        return Layout(component, stack, simulation, nets=nets)

    return build


def flip_chip_pad() -> Layout:
    """The flip-chip pad coupon with its four ground bumps."""
    import gdsfactory as gf
    from orcapen_sc_pdk.tech import get_two_die_flip_chip_layer_stack

    component = gf.get_component("flip_chip_pad_coupon", curve_sampling=SAMPLING)
    references = {reference.name: reference for reference in component.insts}
    bumps = ("BUMP_MM", "BUMP_MP", "BUMP_PM", "BUMP_PP")
    instances = {f"chip/{name}": references[name] for name in ("LOWER_PAD_DIE", *bumps)}
    nets = {
        "pad": ("chip/LOWER_PAD_DIE/PAD",),
        "ground": (
            "chip/LOWER_PAD_DIE/GROUND",
            "chip/UPPER_GROUND",
            *(f"chip/{name}/D0_D1_INDIUM_BUMP" for name in bumps),
        ),
    }
    return Layout(
        component,
        get_two_die_flip_chip_layer_stack(),
        "electrostatic",
        instances,
        nets,
    )


def airbridge_coupon() -> Layout:
    """The CPW airbridge coupon, its bridge on the ground Net."""
    import gdsfactory as gf
    from orcapen_sc_pdk.tech import get_single_die_layer_stack

    component = gf.get_component("cpw_airbridge_coupon", curve_sampling=SAMPLING)
    references = {reference.name: reference for reference in component.insts}
    nets = {
        "signal": ("chip/SIGNAL",),
        "ground": (
            "chip/GROUND_MINUS",
            "chip/GROUND_PLUS",
            "chip/AIRBRIDGE/DECK",
            "chip/AIRBRIDGE/PIER_MINUS",
            "chip/AIRBRIDGE/PIER_PLUS",
        ),
    }
    stack = get_single_die_layer_stack(include_airbridges=True)
    return Layout(
        component,
        stack,
        "electrostatic",
        {"chip/AIRBRIDGE": references["AIRBRIDGE"]},
        nets,
    )


def disk_qubit(plane_bounds_um=None, **params) -> Callable[[], Layout]:
    """The builder of ``disk_flip_chip_qubit`` as ``chip/device`` under the PDK's common ground planes (orcapen-sc-pdk#64)."""

    def build() -> Layout:
        import gdsfactory as gf
        from orcapen_sc_pdk.helpers.assembly import (
            author_common_flip_chip_ground_planes,
        )
        from orcapen_sc_pdk.tech import LAYER_STACK_FLIP_CHIP

        qubit = gf.get_component(
            "disk_flip_chip_qubit", curve_sampling=SAMPLING, **params
        )
        parent = gf.Component()
        device = parent << qubit
        device.name = "device"
        parent.add_ports(device.ports)
        author_common_flip_chip_ground_planes(parent, plane_bounds_um=plane_bounds_um)
        references = {reference.name: reference for reference in qubit.insts}
        bumps = sorted(qubit.info["bump_instance_names"])
        instances = {
            "chip/device": device,
            **{f"chip/device/{name}": references[name] for name in bumps},
        }
        nets = {
            "Ground": (
                "chip/D0_TOP_GROUND_PLANE",
                "chip/D1_BOTTOM_GROUND_PLANE",
                *(f"chip/device/{name}/D0_D1_INDIUM_BUMP" for name in bumps),
            )
        }
        return Layout(
            parent,
            LAYER_STACK_FLIP_CHIP,
            "eigenmode",
            instances,
            nets,
        )

    return build


def launcher_pair(*, right_sense: Sense = "pdk") -> Layout:
    """The launcher-to-launcher layout with its two sheets as ``PORT_IMPEDANCE_OHM`` ports to ground, left first."""
    from orcapen_sc_pdk.tech import get_single_die_layer_stack

    ports = [
        LumpedPort(sheet, reference_net="ground", impedance_ohm=PORT_IMPEDANCE_OHM)
        for sheet in LAUNCHER_PORT_SHEETS
    ]
    return Layout(
        launcher_to_launcher(right_sense=right_sense),
        get_single_die_layer_stack(),
        "driven",
        nets=LAUNCHER_NETS,
        lumped_elements=ports,
    )


def launcher_to_launcher(*, right_sense: Sense = "pdk") -> Any:
    """The launcher-to-launcher component; ``right_sense`` is the sense the right port sheet is drawn in.

    The parent authors ``component_semantics`` v2: ``SIGNAL`` (the one DRAW polygon the launchers and the line
    flatten into), ``GROUND`` (the die face less the ground mask) and the two port sheets, ``o_lumped_left`` and
    ``o_lumped_right``.  The launcher's own sheet abuts the pad and the ground with no overlap, so it lands on
    no conductor and cannot be a port (orcapen-sc-pdk issue 67): it is drawn on an unused layer, and each port
    sheet is drawn here, 85 x 150 um behind its pad, landing ``LANDING_UM`` into the pad and the ground.  The
    PDK draws both sheets from the ground to the pad; ``right_sense="reversed"`` draws the right one from the
    pad to the ground.
    """
    import gdsfactory as gf
    from orcapen_sc_pdk import activate
    from orcapen_sc_pdk.ports import add_driven_lumped_port
    from orcapen_sc_pdk.tech import LAYER

    activate()
    component = gf.Component()
    launcher = gf.get_component(
        "launcher",
        cpw_xs="cpw_6_7_6",
        sim_boundary_layer=_UNUSED_LAYER,
        curve_sampling=SAMPLING,
    )
    left = component << launcher
    left.move(origin=left.ports["o_neck"].center, destination=(-LINE_UM / 2, 0.0))
    right = component << launcher
    right.rotate(180)
    right.move(origin=right.ports["o_neck"].center, destination=(LINE_UM / 2, 0.0))
    line = component << gf.components.straight(
        length=LINE_UM, cross_section=gf.get_cross_section("cpw_6_7_6")
    )
    line.move(origin=line.ports["o1"].center, destination=left.ports["o_neck"].center)
    component << gf.components.rectangle(
        size=DIE_UM, centered=True, layer=LAYER.D0_SUBSTRATE_AREA
    )
    pad_back_um = LINE_UM / 2 + TAPER_UM + PAD_UM
    for name, side in (("o_lumped_left", -1.0), ("o_lumped_right", 1.0)):
        x0, x1 = sorted(
            (
                side * (pad_back_um - LANDING_UM),
                side * (pad_back_um + BACK_GAP_UM + LANDING_UM),
            )
        )
        half = PAD_UM / 2
        component.add_polygon(
            [(x0, -half), (x1, -half), (x1, half), (x0, half)],
            layer=LAYER.D0_TOP_SIM_BOUNDARY,
        )
        # The orientation points along the sheet from its start conductor to its end conductor.
        toward_pad = 0.0 if side < 0 else 180.0
        drawn_reversed = name == "o_lumped_right" and right_sense == "reversed"
        add_driven_lumped_port(
            component,
            name=name,
            center=(side * (pad_back_um + BACK_GAP_UM / 2), 0.0),
            orientation=(toward_pad + 180.0) % 360.0 if drawn_reversed else toward_pad,
            layer=LAYER.D0_TOP_SIM_BOUNDARY,
        )
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {
                "semantic_id": "SIGNAL",
                "level": "D0_TOP_M1",
                "gds_layer": tuple(LAYER.D0_TOP_M1_DRAW),
                "geometry": {
                    "geometry_source": "gds_polygon",
                    "selector_point_um": (0.0, 0.0),
                },
            },
            {
                "semantic_id": "GROUND",
                "level": "D0_TOP_M1",
                "gds_layer": tuple(LAYER.D0_TOP_GROUND_MASK),
            },
        ],
        "ports": [
            {
                "name": name,
                "layer": tuple(LAYER.D0_TOP_SIM_BOUNDARY),
                "target_layer": "D0_TOP_M1",
            }
            for name in ("o_lumped_left", "o_lumped_right")
        ],
    }
    return component


PAD_NETS = {"pad": ("chip/PAD",), "ground": ("chip/GROUND",)}

# The roster the Human accepted (issue 141), with the Driven case of issue 187's plan (slice 6).
ROSTER: dict[str, Callable[[], Layout]] = {
    "simple_pad": single_die("square_pad_capacitor", nets=PAD_NETS),
    "circular_pad": single_die("circular_pad_capacitor", nets=PAD_NETS),
    "airbridge_coupon": airbridge_coupon,
    "flip_chip_pad": flip_chip_pad,
    "disk_qubit_default": disk_qubit(),
    "launcher_to_launcher": launcher_pair,
}


def surface_epr_specs(bands: bool) -> dict[str, dict[str, Any]]:
    """MA, MS and SA from the PDK's literature-central presets; with bands, margins (0.5, 1.0) um by type."""
    from orcapen_sc_pdk import (
        get_interface_preset_records,
        validate_interface_preset_records,
    )

    records = validate_interface_preset_records(get_interface_preset_records())
    specs = {}
    for kind in ("MA", "MS", "SA"):
        record = records[f"LiteratureCentral_{kind}"]
        specs[kind] = {
            key: record[key]
            for key in ("thickness", "permittivity", "loss_tangent", "source")
        }
        if bands:
            specs[kind]["inset_margins_um"] = list(MARGINS_UM)
    return specs


def geometry_plan(layout: Layout, material_records, specs) -> GeometryPlan:
    """The plan ``simulation_drc`` and the simulation are given: root ``chip``, no coupon padding, the roster's vacuum, ``specs``.

    The vacuum is set on the plan only when the layout's stack has no ``VACUUM_REGION`` of its own.  ``specs`` are
    the Surface-EPR specs of :func:`surface_epr_specs`, or None for none.  The plan's lumped elements are the
    layout's when it states them; otherwise an Eigenmode layout's plan declares every authored port as a lumped
    element (:func:`every_authored_port`).
    """
    plan = GeometryPlan(
        layout.component,
        root_id="chip",
        layer_stack=layout.layer_stack,
        material_records=material_records,
        coupon_padding_um=0.0,
    )
    for path, reference in layout.instances.items():
        plan.register_instance(path, reference)
    if layout.nets is not None:
        plan.set_nets(layout.nets)
    if "VACUUM_REGION" not in layout.layer_stack.layers:
        plan.set_vacuum_region(VACUUM_PADDING_UM)
    plan.set_surface_epr(specs)
    if layout.lumped_elements is not None:
        plan.set_lumped_elements(layout.lumped_elements)
    elif layout.simulation == "eigenmode":
        plan.set_lumped_elements(every_authored_port(plan))
    return plan


def every_authored_port(plan: GeometryPlan) -> list[LumpedElement]:
    """Every port sheet the layout authors, as a ``PORT_INDUCTANCE_H`` lumped element, in the order it authors them.

    The names are read from a snapshot of the plan, as a study reads the sheets its layout draws.
    """
    authored = plan.prepare().stack["metadata"].get("port_sheet_source_layers", ())
    return [
        LumpedElement(port["name"], inductance_h=PORT_INDUCTANCE_H) for port in authored
    ]


def prepare_plan(layout: Layout, material_records, specs):
    """The prepared :func:`geometry_plan`, as ``set_plan`` takes it."""
    return geometry_plan(layout, material_records, specs).prepare()


_SIMULATIONS: dict[str, type[EigenmodeSim | ElectrostaticSim | DrivenSim]] = {
    "eigenmode": EigenmodeSim,
    "electrostatic": ElectrostaticSim,
    "driven": DrivenSim,
}


def simulation_class(
    layout: Layout,
) -> type[EigenmodeSim | ElectrostaticSim | DrivenSim]:
    """The SCGSim simulation the layout is meant for."""
    return _SIMULATIONS[layout.simulation]


def configured_simulation(layout: Layout, snapshot):
    """The simulation the roster is meshed with: the plan's snapshot, with its Surface EPR and lumped elements, on Route B."""
    sim = simulation_class(layout)()
    sim.set_plan(snapshot)
    return sim
