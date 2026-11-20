"""The finite-ground CPW coupon of the public OrcaPen SC PDK, cut across at x = 0: issue 188's worked example.

The plan has the settings of the example: Nets ``signal`` and ``ground``, the vacuum padded (0, 0, 1000) um,
and MA, MS and SA from the PDK's literature-central presets, banded at 0.5 and 1.0 um.  The plan's 3D model is
built in memory, cut and classified, with no solver: ground [0, 80], gap [80, 86], signal [86, 96],
gap [96, 102], ground [102, 182] um along the cut, Al 0.1 um on Si 500 um.  Route B and Route C give one record.
"""

import pytest

from scgsim.geometry import CutLine, GeometryPlan
from scgsim.geometry.models.section import LineConductor, LineGap
from scgsim.geometry.sections import cross_section_at

MARGINS_UM = [0.5, 1.0]
INTERFACE_KINDS = ("MA", "MS", "SA")  # metal-air, metal-substrate, substrate-air
# The vacuum ends at the die's planar edge and extends 1000 um below and above it (issue 182).
VACUUM_PADDING_UM = (0.0, 0.0, 1000.0)
CUT = CutLine((0.0, -91.0), (0.0, 91.0))


def coupon_snapshot():
    """The prepared plan of ``finite_ground_cpw_coupon`` at the PDK's defaults (W 10, s 6, ground 80 um)."""
    pytest.importorskip(
        "orcapen_sc_pdk", reason="public OrcaPen SC PDK is not installed"
    )
    from orcapen_sc_pdk import (  # noqa: PLC0415 -- optional PDK, imported only when installed.
        activate,
        get_interface_preset_records,
        validate_interface_preset_records,
    )
    from orcapen_sc_pdk.cells.simulation.scgsim import (  # noqa: PLC0415 -- optional PDK, imported only when installed.
        finite_ground_cpw_coupon,
    )
    from orcapen_sc_pdk.materials import get_material_records  # noqa: PLC0415 -- optional PDK, imported only when installed.
    from orcapen_sc_pdk.tech import get_single_die_layer_stack  # noqa: PLC0415 -- optional PDK, imported only when installed.

    activate()
    plan = GeometryPlan(
        component=finite_ground_cpw_coupon(curve_sampling="simulation"),
        root_id="chip",
        layer_stack=get_single_die_layer_stack(),
        material_records=get_material_records(),
        coupon_padding_um=0.0,
    )
    plan.set_nets(
        {
            "signal": ("chip/SIGNAL",),
            "ground": ("chip/GROUND_PLUS", "chip/GROUND_MINUS"),
        }
    )
    plan.set_vacuum_region(padding=VACUUM_PADDING_UM)
    presets = validate_interface_preset_records(get_interface_preset_records())
    plan.set_surface_epr(
        {
            kind: {
                **{
                    key: presets[f"LiteratureCentral_{kind}"][key]
                    for key in ("thickness", "permittivity", "loss_tangent", "source")
                },
                "inset_margins_um": MARGINS_UM,
            }
            for kind in INTERFACE_KINDS
        }
    )
    return plan.prepare()


@pytest.mark.pdk
def test_the_coupon_cut_at_x_0_is_the_worked_example():
    """The section of the plan's worked example, exactly, and the same record on Route C."""
    snapshot = coupon_snapshot()
    line = cross_section_at(snapshot, CUT)
    film = (0.0, 0.1)
    assert line.conductors == (
        LineConductor(
            "chip/GROUND_MINUS", "ground", "reference", "Al", (0.0, 80.0), film
        ),
        LineConductor("chip/SIGNAL", "signal", "signal", "Al", (86.0, 96.0), film),
        LineConductor(
            "chip/GROUND_PLUS", "ground", "reference", "Al", (102.0, 182.0), film
        ),
    )
    assert line.gaps == (
        LineGap((80.0, 86.0), film, ("ground", "signal")),
        LineGap((96.0, 102.0), film, ("signal", "ground")),
    )
    assert [(d.entity, d.material, d.u_um, d.z_um) for d in line.dielectrics] == [
        ("VACUUM_REGION", "vacuum", (0.0, 182.0), (-1500.0, -500.0)),
        ("D0_SUBSTRATE", "Si", (0.0, 182.0), (-500.0, 0.0)),
        ("VACUUM_REGION", "vacuum", (0.0, 182.0), (0.0, 1000.1)),
    ]
    assert line.describe() == (
        "ground 80 | 6 | signal 10 | 6 | ground 80 um; Al 0.1 um on Si 500 um; cut at x = 0 um"
    )
    on_c = cross_section_at(snapshot, CUT, route="C")
    assert on_c == line
    assert on_c.digest == line.digest
