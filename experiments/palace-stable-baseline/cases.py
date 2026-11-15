# ruff: noqa: PLC0415 -- the PDK and SCGSim are imported where used, after the PDK is activated
"""The layouts of the baseline: how each OrcaPen SC PDK layout is built and which simulation it is meant for.

The roster of issue 141 and the helpers that build it are in ``scripts/roster_cases.py``, which the nightly
roster check runs too; :data:`LAYOUTS` is that roster followed by the other layouts, built here with the same
helpers.  Every layout is drawn at ``curve_sampling="simulation"``.  A layout is a :class:`Layout`: the
component, the PDK layer stack, the registered instances and the Net assignment its simulation's plan is given.
The plan settings are those ``simulation_drc`` fixes (root ``chip``, zero coupon padding, no vacuum of its
own); the vacuum and the Surface EPR are set on the simulation (``roster_cases.configured_simulation``).

The supported banded scenes of ``tests/sgb_supported_*`` are listed in :data:`SCENES`; they are SGB
catalogues, not layouts, and are measured for fine structure only.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from roster_cases import (  # noqa: E402
    ROSTER,
    SAMPLING,
    Layout,
    disk_qubit,
    single_die,
)

# The site extent of one Disk Qubit, the plane bounds orcapen-sc-pdk#64 used for include_ground=False builds.
DISK_SITE_BOUNDS_UM = (-500.0, -500.0, 500.0, 500.0)


def _without_semantics_key(
    name: str, key: str, *, nets=None, simulation: str = "electrostatic"
):
    """The cell with one ``component_semantics`` entry removed: without ``boundary_curves`` SCGSim reads the polygon."""

    def build() -> Layout:
        import gdsfactory as gf
        from orcapen_sc_pdk.tech import get_single_die_layer_stack

        # The cached cell itself is changed (a copy gets a new cell name, which the plan then cannot find in
        # the GDS it writes); every case runs in its own process, so no other case sees the change.
        component = gf.get_component(name, curve_sampling=SAMPLING)
        semantics = copy.deepcopy(dict(component.info["component_semantics"]))
        semantics.pop(key, None)
        component.info["component_semantics"] = semantics
        return Layout(component, get_single_die_layer_stack(), simulation, nets=nets)

    return build


def kosen_xmon() -> Layout:
    """The Kosen Xmon coupon of the PDK's Route B eigenmode notebook (corner-anchored ground shorts)."""
    import gdsfactory as gf
    from orcapen_sc_pdk import LAYER_STACK
    from orcapen_sc_pdk.helpers.assembly import place_flip_chip_ground_short_bumps

    device = gf.get_component(
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
    return Layout(coupon.component, LAYER_STACK, "eigenmode", instances, nets)


def sample_airbridge_chip() -> Layout:
    """The PDK's sample airbridge chip (it declares no component semantics)."""
    from orcapen_sc_pdk.samples.simulation_demos import small_airbridge_chip
    from orcapen_sc_pdk.tech import get_single_die_layer_stack

    component = small_airbridge_chip(curve_sampling=SAMPLING)
    return Layout(
        component, get_single_die_layer_stack(include_airbridges=True), "electrostatic"
    )


# The other layouts, after the roster the Human accepted (issue 141).
OTHERS: dict[str, Callable[[], Layout]] = {
    "kosen_xmon": kosen_xmon,
    "purcell_filter": single_die(
        "capacitive_coupling_intrinsic_individual_purcell_filter_readout_resonators",
        simulation="electrostatic",
    ),
    "tsv_concentric": single_die(
        "tsv_concentric_signal_coupon", simulation="electrostatic"
    ),
    "tsv_stitched": single_die("tsv_ground_stitched_cpw", simulation="electrostatic"),
    "bend_euler": single_die("bend_euler", simulation="electrostatic"),
    "sample_airbridge_chip": sample_airbridge_chip,
    "bspline_strip_curves": single_die("bspline_strip", simulation="electrostatic"),
    "bspline_strip_polygon": _without_semantics_key("bspline_strip", "boundary_curves"),
    "interpolation_strip_curves": single_die(
        "interpolation_spline_strip", simulation="electrostatic"
    ),
    "interpolation_strip_polygon": _without_semantics_key(
        "interpolation_spline_strip", "boundary_curves"
    ),
    # orcapen-sc-pdk#63 concerns non-default Disk Qubit settings: one variant with grounded claws, one floating.
    "disk_qubit_grounded": disk_qubit(DISK_SITE_BOUNDS_UM, include_ground=False),
    "disk_qubit_floating": disk_qubit(
        DISK_SITE_BOUNDS_UM, include_ground=False, tc_grounded_sides=()
    ),
    # The variant of issues 100 and 125 (floating claws, no lead fan), for continuity with their measurements.
    "disk_qubit_floating_span0": disk_qubit(
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
