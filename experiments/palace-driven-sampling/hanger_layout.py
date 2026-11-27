"""The hanger-resonator layout of issue 307: a quarter-wave resonator beside a launcher-to-launcher feedline.

It is composed from public OrcaPen SC PDK cells as ``launcher_to_launcher()`` in ``scripts/roster_cases.py`` is:
two PDK ``launcher`` cells joined by an 800 um ``cpw_6_10_6`` feedline along y = 0, a hand-drawn 85 x 150 um port
sheet behind each pad that lands ``LANDING_UM`` into the pad and into the ground, and the PDK ``resonator``
(hanger, meander and open end) at its defaults, its coupling straight parallel to the feedline above it.

The resonator's coupling straight starts at its port ``o1`` without an end cap, so its centre conductor ends
flush with the end of its own ground mask and meets the ground plane there: the shorted end of the quarter wave.
The far end, ``o2``, carries the PDK's open-end cap.  The resonator metal is therefore part of Net ``ground``:
the ``GROUND`` region keeps the resonator's DRAW polygon (``include_layer`` with a selector point on the coupling
straight), and ``SIGNAL`` is the feedline and the two launchers only.

The plan has the roster's settings (``geometry_plan``): root ``chip``, the PDK single-die stack, vacuum padding
(0, 0, 1000) um, MA, MS and SA Surface EPR from the PDK's literature-central presets without bands, and the two
sheets as 50 ohm ports to ``ground``, left first, so the left launcher is port 1.  The Human accepted this
geometry on issue 307 (its SGIP, round 1).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gdsfactory as gf
from orcapen_sc_pdk import activate
from orcapen_sc_pdk.materials import get_material_records
from orcapen_sc_pdk.ports import add_driven_lumped_port
from orcapen_sc_pdk.tech import LAYER, get_single_die_layer_stack

from scgsim.geometry import GeometryPlan, LumpedPort

# The roster lives in scripts/, beside this folder's parent; `python -I` does not put it on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from roster_cases import (  # noqa: E402
    LANDING_UM,
    PORT_IMPEDANCE_OHM,
    SAMPLING,
    Layout,
    geometry_plan,
    surface_epr_specs,
)

# The feedline uses the resonator's own default cross-section: a 10 um signal with 6 um gaps, about 50 ohm on Si.
FEED_XS = "cpw_6_10_6"
FEED_SIGNAL_UM, FEED_GAP_UM = 10.0, 6.0
RESONATOR_SIGNAL_UM, RESONATOR_GAP_UM = 10.0, 6.0
LINE_UM = 800.0  # the feedline straight between the two launcher necks
PAD_UM, TAPER_UM, BACK_GAP_UM = 150.0, 150.0, 85.0  # the PDK launcher's defaults
# The ground strip between the feedline's upper gap and the coupling straight's lower gap sets the coupling.
COUPLING_STRIP_UM = 5.0
COUPLING_Y_UM = (
    FEED_SIGNAL_UM / 2
    + FEED_GAP_UM
    + COUPLING_STRIP_UM
    + RESONATOR_GAP_UM
    + RESONATOR_SIGNAL_UM / 2
)  # 27 um: the coupling straight's centre line
MARGIN_UM = 150.0  # ground from every ground-mask edge to the die edge
NETS = {"signal": ("chip/SIGNAL",), "ground": ("chip/GROUND",)}
# Left first, so the left launcher is port 1: row and column 1 of the S-matrix.
PORT_SHEETS = ("chip/o_lumped_left", "chip/o_lumped_right")
# A GDS layer no stack level reads: the launcher's own sheet goes there and is not modeled (as in the roster).
_UNUSED_LAYER = (250, 0)


def hanger_plan(*, surface_epr: bool = True) -> GeometryPlan:
    """The plan every run of the study is built from; ``surface_epr=False`` leaves Surface EPR unset."""
    return geometry_plan(
        hanger_layout(),
        get_material_records(),
        surface_epr_specs(bands=False) if surface_epr else None,
    )


def hanger_layout() -> Layout:
    """The layout with its two sheets as 50 ohm ports to ground, left first."""
    ports = [
        LumpedPort(sheet, reference_net="ground", impedance_ohm=PORT_IMPEDANCE_OHM)
        for sheet in PORT_SHEETS
    ]
    return Layout(
        hanger_on_feedline(),
        get_single_die_layer_stack(),
        "driven",
        nets=NETS,
        lumped_elements=ports,
    )


def hanger_on_feedline() -> Any:
    """The component; ``component.info["hanger"]`` records the resonator's lengths and the die."""
    activate()
    component = gf.Component()
    _place_feedline(component)
    resonator = gf.get_component("resonator", curve_sampling=SAMPLING)
    hanger = component << resonator
    # The PDK anchors a hanger at the centre of its coupling straight; the body grows to +y from it.
    hanger.move(
        origin=hanger.ports["o_hanger_center"].center, destination=(0.0, COUPLING_Y_UM)
    )
    die = _add_die(component)
    _add_port_sheets(component)
    component.info["component_semantics"] = _semantics()
    component.info["hanger"] = {
        "resonator": {
            key: resonator.info[key]
            for key in ("length", "hanger_length", "meander_length")
        },
        "die_um": die,
        "coupling_y_um": COUPLING_Y_UM,
    }
    return component


def _place_feedline(component: Any) -> None:
    """Two PDK launchers, the right one turned to face the left, joined by the ``LINE_UM`` feedline."""
    launcher = gf.get_component(
        "launcher",
        cpw_xs=FEED_XS,
        sim_boundary_layer=_UNUSED_LAYER,
        curve_sampling=SAMPLING,
    )
    left = component << launcher
    left.move(origin=left.ports["o_neck"].center, destination=(-LINE_UM / 2, 0.0))
    right = component << launcher
    right.rotate(180)
    right.move(origin=right.ports["o_neck"].center, destination=(LINE_UM / 2, 0.0))
    line = component << gf.components.straight(
        length=LINE_UM, cross_section=gf.get_cross_section(FEED_XS)
    )
    line.move(origin=line.ports["o1"].center, destination=left.ports["o_neck"].center)


def _add_die(component: Any) -> tuple[float, float, float, float]:
    """The die, ``MARGIN_UM`` of ground beyond every ground-mask edge; returns (x0, y0, x1, y1) in um."""
    masks = component.get_polygons_points(by="tuple")[tuple(LAYER.D0_TOP_GROUND_MASK)]
    x0 = min(float(mask[:, 0].min()) for mask in masks) - MARGIN_UM
    x1 = max(float(mask[:, 0].max()) for mask in masks) + MARGIN_UM
    y0 = min(float(mask[:, 1].min()) for mask in masks) - MARGIN_UM
    y1 = max(float(mask[:, 1].max()) for mask in masks) + MARGIN_UM
    component.add_polygon(
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=LAYER.D0_SUBSTRATE_AREA
    )
    return (x0, y0, x1, y1)


def _add_port_sheets(component: Any) -> None:
    """Each port sheet, ``BACK_GAP_UM`` x ``PAD_UM`` behind its pad, drawn from the ground to the pad as the PDK does."""
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
        # The orientation points along the sheet from its start conductor (ground) to its end conductor (pad).
        add_driven_lumped_port(
            component,
            name=name,
            center=(side * (pad_back_um + BACK_GAP_UM / 2), 0.0),
            orientation=0.0 if side < 0 else 180.0,
            layer=LAYER.D0_TOP_SIM_BOUNDARY,
        )


def _semantics() -> dict[str, Any]:
    """``component_semantics`` v2: ``SIGNAL`` (feedline and launchers), ``GROUND`` (with the resonator) and the sheets."""
    return {
        "schema_version": 2,
        "conductor_regions": [
            {
                "semantic_id": "SIGNAL",
                "level": "D0_TOP_M1",
                "gds_layer": tuple(LAYER.D0_TOP_M1_DRAW),
                # A point on the feedline's centre conductor selects the one DRAW polygon it and the launchers form.
                "geometry": {
                    "geometry_source": "gds_polygon",
                    "selector_point_um": (0.0, 0.0),
                },
            },
            {
                "semantic_id": "GROUND",
                "level": "D0_TOP_M1",
                "gds_layer": tuple(LAYER.D0_TOP_GROUND_MASK),
                # The die face less the ground mask, plus the resonator's DRAW polygon, selected on its coupling
                # straight: the resonator is shorted to the ground plane at o1.
                "geometry": {
                    "include_layer": tuple(LAYER.D0_TOP_M1_DRAW),
                    "include_selector_points_um": [(0.0, COUPLING_Y_UM)],
                },
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
