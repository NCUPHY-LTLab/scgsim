# ruff: noqa: PLC0415 -- the PDK and SCGSim are imported where used, after the PDK is activated
"""The layouts of the baseline: how each OrcaPen SC PDK layout is built and which simulation it is meant for.

Every layout is drawn at ``curve_sampling="simulation"``.  A layout is a :class:`Layout`: the component, the
PDK layer stack, the registered instances and the Net assignment its simulation's plan is given.  The plan
settings are those ``simulation_drc`` fixes (root ``chip``, zero coupon padding, a 1 um vacuum region when the
layout has none), so the SGB build, the check and the mesh all see one model.

The supported banded scenes of ``tests/sgb_supported_*`` are listed in :data:`SCENES`; they are SGB
catalogues, not layouts, and are measured for fine structure only.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SAMPLING = "simulation"
# SimplePad and CircularPad notebooks pad a single die by (250, 250, 1000) um; the Kosen Xmon notebook pads a
# flip-chip coupon by (0, 0, OUTER_VACUUM_THICKNESS_UM = 1000) um.
FLIP_CHIP_VACUUM_UM = (0.0, 0.0, 1000.0)
# The site extent of one Disk Qubit, the plane bounds orcapen-sc-pdk#64 used for include_ground=False builds.
DISK_SITE_BOUNDS_UM = (-500.0, -500.0, 500.0, 500.0)


@dataclass(frozen=True)
class Layout:
    """One layout as its simulation receives it."""

    component: Any
    layer_stack: Any
    simulation: str  # "eigenmode" (a junction port is declared) or "electrostatic"
    instances: Mapping[str, Any] = field(default_factory=dict)
    nets: Mapping[str, Sequence[str]] | None = None
    # The plan vacuum of the PDK's Palace notebooks: (x, y, z) padding in um.
    notebook_vacuum_padding_um: tuple[float, float, float] = (250.0, 250.0, 1000.0)


def _gf():
    import gdsfactory as gf

    return gf


def _single_die(
    name: str,
    *,
    nets=None,
    airbridges: bool = False,
    simulation: str = "eigenmode",
    **params,
):
    def build() -> Layout:
        from orcapen_sc_pdk.tech import get_single_die_layer_stack

        component = _gf().get_component(name, curve_sampling=SAMPLING, **params)
        stack = get_single_die_layer_stack(include_airbridges=airbridges)
        return Layout(component, stack, simulation, nets=nets)

    return build


def _without_semantics_key(
    name: str, key: str, *, nets=None, simulation: str = "electrostatic"
):
    """The cell with one ``component_semantics`` entry removed: without ``boundary_curves`` SCGSim reads the polygon."""

    def build() -> Layout:
        from orcapen_sc_pdk.tech import get_single_die_layer_stack

        # The cached cell itself is changed (a copy gets a new cell name, which the plan then cannot find in
        # the GDS it writes); every case runs in its own process, so no other case sees the change.
        component = _gf().get_component(name, curve_sampling=SAMPLING)
        semantics = copy.deepcopy(dict(component.info["component_semantics"]))
        semantics.pop(key, None)
        component.info["component_semantics"] = semantics
        return Layout(component, get_single_die_layer_stack(), simulation, nets=nets)

    return build


def flip_chip_pad() -> Layout:
    """The flip-chip pad coupon with its four ground bumps."""
    from orcapen_sc_pdk.tech import get_two_die_flip_chip_layer_stack

    component = _gf().get_component("flip_chip_pad_coupon", curve_sampling=SAMPLING)
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
        FLIP_CHIP_VACUUM_UM,
    )


def airbridge_coupon() -> Layout:
    """The CPW airbridge coupon, its bridge on the ground Net."""
    from orcapen_sc_pdk.tech import get_single_die_layer_stack

    component = _gf().get_component("cpw_airbridge_coupon", curve_sampling=SAMPLING)
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


def kosen_xmon() -> Layout:
    """The Kosen Xmon coupon of the PDK's Route B eigenmode notebook (corner-anchored ground shorts)."""
    from orcapen_sc_pdk import LAYER_STACK
    from orcapen_sc_pdk.helpers.assembly import place_flip_chip_ground_short_bumps

    device = _gf().get_component(
        "kosen2024_flip_chip_xmon_qubit",
        bump_ring_count_per_side=0,
        qubit_pad_length=309.5,
        qubit_pad_width=24.65,
        qubit_gap=20.0,
        curve_sampling=SAMPLING,
    )
    coupon = place_flip_chip_ground_short_bumps(
        device,
        coupon_padding_um=75.0,
        clearance_um=30.0,
        placement_mode="corner_anchors",
    )
    instances = {
        f"chip/{name}": reference for name, reference in coupon.named_instances.items()
    }
    shorts = [name for name in coupon.named_instances if name.startswith("short_")]
    nets = {
        "xmon_pad": ("chip/device/D1_XMON_PAD",),
        "Ground": (
            "chip/D0_TOP_GROUND_PLANE",
            "chip/D1_BOTTOM_GROUND_PLANE",
            *(f"chip/device/D1_COUPLER_{index}" for index in range(1, 5)),
            *(f"chip/{name}/D0_D1_INDIUM_BUMP" for name in shorts),
        ),
    }
    return Layout(
        coupon.component, LAYER_STACK, "eigenmode", instances, nets, FLIP_CHIP_VACUUM_UM
    )


def _disk_qubit(plane_bounds_um=None, **params):
    """``disk_flip_chip_qubit`` as ``chip/device`` under the PDK's common ground planes (orcapen-sc-pdk#64)."""

    def build() -> Layout:
        from orcapen_sc_pdk.helpers.assembly import (
            author_common_flip_chip_ground_planes,
        )
        from orcapen_sc_pdk.tech import LAYER_STACK_FLIP_CHIP

        gf = _gf()
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
            FLIP_CHIP_VACUUM_UM,
        )

    return build


def sample_airbridge_chip() -> Layout:
    """The PDK's sample airbridge chip (it declares no component semantics)."""
    from orcapen_sc_pdk.samples.simulation_demos import small_airbridge_chip
    from orcapen_sc_pdk.tech import get_single_die_layer_stack

    component = small_airbridge_chip(curve_sampling=SAMPLING)
    return Layout(
        component, get_single_die_layer_stack(include_airbridges=True), "electrostatic"
    )


PAD_NETS = {"pad": ("chip/PAD",), "ground": ("chip/GROUND",)}

# The roster the Human accepted (issue 141), then the other layouts.
ROSTER: dict[str, Callable[[], Layout]] = {
    "simple_pad": _single_die("square_pad_capacitor", nets=PAD_NETS),
    "circular_pad": _single_die("circular_pad_capacitor", nets=PAD_NETS),
    "airbridge_coupon": airbridge_coupon,
    "flip_chip_pad": flip_chip_pad,
    "disk_qubit_default": _disk_qubit(),
}
OTHERS: dict[str, Callable[[], Layout]] = {
    "kosen_xmon": kosen_xmon,
    "purcell_filter": _single_die(
        "capacitive_coupling_intrinsic_individual_purcell_filter_readout_resonators",
        simulation="electrostatic",
    ),
    "tsv_concentric": _single_die(
        "tsv_concentric_signal_coupon", simulation="electrostatic"
    ),
    "tsv_stitched": _single_die("tsv_ground_stitched_cpw", simulation="electrostatic"),
    "bend_euler": _single_die("bend_euler", simulation="electrostatic"),
    "sample_airbridge_chip": sample_airbridge_chip,
    "bspline_strip_curves": _single_die("bspline_strip", simulation="electrostatic"),
    "bspline_strip_polygon": _without_semantics_key("bspline_strip", "boundary_curves"),
    "interpolation_strip_curves": _single_die(
        "interpolation_spline_strip", simulation="electrostatic"
    ),
    "interpolation_strip_polygon": _without_semantics_key(
        "interpolation_spline_strip", "boundary_curves"
    ),
    # orcapen-sc-pdk#63 concerns non-default Disk Qubit settings: one variant with grounded claws, one floating.
    "disk_qubit_grounded": _disk_qubit(DISK_SITE_BOUNDS_UM, include_ground=False),
    "disk_qubit_floating": _disk_qubit(
        DISK_SITE_BOUNDS_UM, include_ground=False, tc_grounded_sides=()
    ),
    # The variant of issues 100 and 125 (floating claws, no lead fan), for continuity with their measurements.
    "disk_qubit_floating_span0": _disk_qubit(
        DISK_SITE_BOUNDS_UM,
        include_ground=False,
        tc_grounded_sides=(),
        junction_lead_span=0.0,
    ),
}
LAYOUTS: dict[str, Callable[[], Layout]] = {**ROSTER, **OTHERS}


def scene_builders() -> dict[str, Callable[[], Any]]:
    """The supported banded scenes of the test suite (``tests`` must be on ``sys.path``).

    ``cell24_sheet_circular_pad`` left the supported list after the baseline (its files cannot be written);
    it still builds, so it stays here and its fine-structure row can be measured again.
    """
    import sgb_supported_cells
    import sgb_supported_pdk_cells

    return {
        **sgb_supported_cells.SCENES,
        **sgb_supported_pdk_cells.SCENES,
        **sgb_supported_pdk_cells.NOT_WRITTEN,
    }
