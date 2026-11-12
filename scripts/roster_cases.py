# ruff: noqa: PLC0415 -- gdsfactory and the PDK are imported where a layout is built: the PDK is not in SCGSim's own lock, and the runner's record handling loads without it
"""The roster of issue 141 and the simulation settings it is tested with.

The roster is the list of OrcaPen SC PDK layouts that define "Palace stable": SimplePad, the circular pad, the
airbridge coupon (single die), the flip-chip pad, and ``disk_flip_chip_qubit`` at its default settings (flip
chip).  Each is a :class:`Layout`: the component, drawn at ``curve_sampling="simulation"``, the PDK layer
stack, the registered instances and the Net assignment its simulation's plan is given.

The settings: plan root ``chip`` and zero coupon padding, as ``simulation_drc`` builds its plan; the vacuum
the Human set for testing layouts (issue 182), padding 0 um in x and y and 1000 um below and above, when the
layout has none of its own; every authored port at 10 nH; and Route B Surface EPR with the PDK's
literature-central MA (metal-air), MS (metal-substrate) and SA (substrate-air) presets, cut as bands at margins
0.5 and 1.0 um by type.  ``simulation_drc`` sets its own vacuum.

``scripts/check_roster.py`` runs the roster every night.  The Palace baseline experiment
(``experiments/palace-stable-baseline``) builds its other layouts from the same helpers and measures them with
the same settings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry import GeometryPlan
from scgsim.palace import EigenmodeSim, ElectrostaticSim

SAMPLING = "simulation"
MARGINS_UM = [0.5, 1.0]
PORT_INDUCTANCE_H = 10e-9
# (x, y, z) padding in um, on both sides of each axis: the vacuum ends at the die's planar edge and extends
# 1000 um below and above it (the Human, issue 182).
VACUUM_PADDING_UM = (0.0, 0.0, 1000.0)


@dataclass(frozen=True)
class Layout:
    """One layout as its simulation receives it."""

    component: Any
    layer_stack: Any
    simulation: str  # "eigenmode" (a junction port is declared) or "electrostatic"
    instances: Mapping[str, Any] = field(default_factory=dict)
    nets: Mapping[str, Sequence[str]] | None = None


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


PAD_NETS = {"pad": ("chip/PAD",), "ground": ("chip/GROUND",)}

# The roster the Human accepted (issue 141).
ROSTER: dict[str, Callable[[], Layout]] = {
    "simple_pad": single_die("square_pad_capacitor", nets=PAD_NETS),
    "circular_pad": single_die("circular_pad_capacitor", nets=PAD_NETS),
    "airbridge_coupon": airbridge_coupon,
    "flip_chip_pad": flip_chip_pad,
    "disk_qubit_default": disk_qubit(),
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


def prepare_plan(layout: Layout, material_records):
    """The prepared plan as ``simulation_drc`` builds it: root ``chip``, no coupon padding, no vacuum of its own."""
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
    return plan.prepare()


def simulation_class(layout: Layout) -> type[EigenmodeSim] | type[ElectrostaticSim]:
    """The SCGSim simulation the layout is meant for."""
    return EigenmodeSim if layout.simulation == "eigenmode" else ElectrostaticSim


def configured_simulation(layout: Layout, snapshot, specs):
    """The simulation the roster is meshed with: the (0, 0, 1000) um vacuum, every authored port, Route B Surface EPR."""
    sim = simulation_class(layout)()
    sim.set_plan(snapshot)
    if "VACUUM_REGION" not in snapshot.stack["solution_regions"]:
        sim.set_vacuum_region(VACUUM_PADDING_UM)
    if layout.simulation == "eigenmode":
        for port in snapshot.stack["metadata"].get("port_sheet_source_layers", ()):
            sim.add_port(
                port["name"],
                layer=port["target_layer"],
                layout_sheet=True,
                inductance=PORT_INDUCTANCE_H,
            )
    sim.set_surface_epr(representation="B", specs=specs)
    return sim
