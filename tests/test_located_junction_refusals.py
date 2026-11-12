"""The junction partition's layout refusals name the lumped port, its layer and the place in um.

The third slice of the located layout refusals (``test_located_refusals.py`` and
``test_located_layout_refusals.py`` hold the first two).  Each test draws a pad, an electrode
and a lumped-port sheet with one defect at a known place, prepares the port as ``mesh()`` and
``simulation_drc`` do (``EigenmodeSim``'s own preparation), and reads every coordinate back out
of the refusal: each must lie on the defect.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import gdstk
import pytest
from test_located_refusals import on, places
from test_lumped_ports_at_any_angle import placed
from test_palace_rectangular_ports import prepared_port
from test_simulation_drc import LAYER_STACK, MATERIALS, NO_VACUUM_STACK, check, two_pads

from scgsim.geometry import GeometryPlan
from scgsim.geometry.models.physical import PhysicalLayerDeclaration
from scgsim.palace import EigenmodeSim

PORT = "chip/device/P1"
JUNCTION = f"junction 'PORT_SHEET__{PORT}__0000' (lumped port {PORT!r}) on layer 9/0"
NETS = {"pad": ("chip/device/pad",), "electrode": ("chip/device/electrode",)}
# A pad ending at x = 0, an electrode from x = 2 and a 1 um wide sheet from x = -1 to 3 between them.
PAD = ("pad", 1, [(-20, -10), (0, -10), (0, 10), (-20, 10)], (-10, 0))
ELECTRODE = ("electrode", 2, [(2, -10), (20, -10), (20, 10), (2, 10)], (10, 0))
SHEET = [(-1, -0.5), (3, -0.5), (3, 0.5), (-1, 0.5)]


def device(monkeypatch, regions, sheet=SHEET, centre=(1.0, 0.0), width_um=1.0):
    """The conductor ``regions`` ``(name, GDS layer, outline, selector point)`` and a sheet P1 along x."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    component = gf.Component()
    for _, layer, outline, _ in regions:
        component.add_polygon(outline, layer=(layer, 0))
    component.add_polygon(sheet, layer=(9, 0))
    # gdsfactory keeps a port width in whole pairs of grid steps; SCGSim reads the sheet itself.
    width = 2 * round(width_um * 500) / 1000
    component.add_port(
        name="P1", center=centre, width=width, orientation=0, layer=(9, 0)
    )
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {
                "semantic_id": name,
                "level": "metal",
                "gds_layer": [layer, 0],
                "geometry": {
                    "geometry_source": "gds_polygon",
                    "selector_point_um": point,
                },
            }
            for name, layer, _, point in regions
        ],
        "ports": [{"name": "P1", "layer": [9, 0], "target_layer": "metal"}],
    }
    return component


def device_plan(component, nets=NETS, turn_degrees=0.0) -> GeometryPlan:
    """The plan of a chip that places the device as ``chip/device``, turned by ``turn_degrees``."""
    chip, reference = placed(component, turn_degrees)
    plan = GeometryPlan(
        chip,
        root_id="chip",
        layer_stack=NO_VACUUM_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    plan.register_instance("chip/device", reference)
    plan.set_nets(nets)
    return plan


def refusal(component, nets=NETS) -> str:
    """The refusal of the port's preparation, the one ``mesh()`` and ``simulation_drc`` run."""
    sim = EigenmodeSim()
    sim.set_plan(device_plan(component, nets).prepare())
    sim.add_port(PORT, layer="metal", layout_sheet=True, inductance=1e-9)
    with pytest.raises(ValueError) as error:
        prepared_port(sim)
    return str(error.value)


def circle(cx, cy, radius, count, turn=0.0):
    """A polygon circle of ``count`` vertices, each rounded to the 1 nm grid."""
    return [
        (
            round(cx + radius * math.cos(2 * math.pi * k / count + turn), 3),
            round(cy + radius * math.sin(2 * math.pi * k / count + turn), 3),
        )
        for k in range(count)
    ]


def point(x, y):
    """A defect that is one point, as a box."""
    return (x, y, x, y)


# ----------------------------------------------------------- the sheet's own conductors


def test_a_third_conductor_touching_the_sheet_is_located_where_it_touches(monkeypatch):
    """A conductor of a third Net, on layer 3/0, touches the sheet's side along x = 0.5..1.5 um, y = 0.5 um."""
    other = ("other", 3, [(0.5, 0.5), (1.5, 0.5), (1.5, 3), (0.5, 3)], (1, 2))
    message = refusal(device(monkeypatch, [PAD, ELECTRODE, other]))
    assert message.startswith(
        f"third-net closed contact with authored junction sheet: {JUNCTION} "
        "(grid g = 0.001 um): chip/device/other on layer 3/0"
    )
    assert message.endswith(
        "Fix the layout: keep the sheet clear of every conductor but the two it lands on."
    )
    (place,) = places(message)
    assert on(place, (0.5, 0.5, 1.5, 0.5))


def test_a_terminal_net_touching_the_far_end_leaves_no_gap_and_is_located(monkeypatch):
    """A second conductor of the pad's Net touches the sheet's far end at x = 3 um, past the electrode."""
    electrode = ("electrode", 2, [(2, -10), (2.5, -10), (2.5, 10), (2, 10)], (2.2, 0))
    far = ("pad2", 3, [(3, -0.5), (6, -0.5), (6, 0.5), (3, 0.5)], (4, 0))
    nets = {**NETS, "pad": ("chip/device/pad", "chip/device/pad2")}
    message = refusal(device(monkeypatch, [PAD, electrode, far]), nets)
    assert message.startswith(
        f"unsupported junction contact ordering or clear corridor: {JUNCTION}"
    )
    assert "chip/device/pad2 on layer 3/0, of Net pad, meets the sheet at" in message
    assert "Fix the layout: draw the sheet so that it crosses one clear gap" in message
    pad, electrode_place, pad2, reach, begins = places(message)
    assert on(pad, (-1, -0.5, 0, 0.5)) and on(electrode_place, (2, -0.5, 2.5, 0.5))
    assert on(pad2, (3, -0.5, 3, 0.5))
    assert on(reach, point(3, 0)) and on(begins, point(2, 0))


def test_a_sheet_landing_twice_on_one_conductor_is_located_at_each_landing(monkeypatch):
    """A C-shaped pad under both ends of the sheet: it lands at x = -1..0 and x = 5..6 um."""
    pad = (
        "pad",
        1,
        [(-20, -10), (8, -10), (8, 10), (5, 10), (5, -6), (0, -6), (0, 10), (-20, 10)],
        (-10, 0),
    )
    electrode = ("electrode", 2, [(2, -2), (4, -2), (4, 2), (2, 2)], (3, 0))
    sheet = [(-1, -0.5), (6, -0.5), (6, 0.5), (-1, 0.5)]
    message = refusal(device(monkeypatch, [pad, electrode], sheet, centre=(3, 0)))
    assert message.startswith(
        "rectangular junction requires one actual overlap per explicit terminal: "
        f"lumped port {PORT!r} on layer 9/0 (grid g = 0.001 um): it overlaps "
        "chip/device/pad on layer 1/0 2 times"
    )
    assert "Fix the layout: draw the sheet so that it overlaps each" in message
    first, second, electrode_place = places(message)
    assert {
        on(first, (-1, -0.5, 0, 0.5)) or on(first, (5, -0.5, 6, 0.5)),
        on(second, (-1, -0.5, 0, 0.5)) or on(second, (5, -0.5, 6, 0.5)),
    } == {True}
    assert first != second
    assert on(electrode_place, (2, -0.5, 4, 0.5))


# --------------------------------------------------------------- the cuts at each end


def round_terminals(monkeypatch, gap_nm):
    """Two polygon circles of radius 1 um, ``gap_nm`` apart at x = 0, under a sheet 0.2 um wide.

    Each circle meets the sheet's sides 5 nm before its tip, so a cut at the tip leaves a 5 nm
    edge along each side, shorter than the 10 grid-step limit.
    """
    gap = gap_nm / 1000
    pad = ("pad", 1, circle(-1, 0, 1, 256), (-1, 0))
    electrode = ("electrode", 2, circle(1 + gap, 0, 1, 256, math.pi), (1 + gap, 0))
    sheet = [(-0.5, -0.1), (0.5, -0.1), (0.5, 0.1), (-0.5, 0.1)]
    return device(monkeypatch, [pad, electrode], sheet, centre=(0, 0), width_um=0.2)


def test_a_gap_too_short_for_the_terminal_steps_is_located(monkeypatch):
    """Round terminals 4 nm apart: no cut beside the pad leaves its side edges zero or 10 grid steps long."""
    message = refusal(round_terminals(monkeypatch, 4))
    assert message.startswith(
        "junction corridor has no cut whose terminal steps are zero or at least 10 grid "
        f"steps: {JUNCTION}"
    )
    assert "Fix the layout: lengthen the gap the sheet crosses" in message
    start, end, low_side, high_side, cut = places(message)
    assert on(start, point(0, 0)) and on(end, point(0.004, 0))
    # The pad's outline meets the sheet's sides 5 nm before its tip.
    assert on(low_side, (-0.006, -0.1, -0.004, -0.1))
    assert on(high_side, (-0.006, 0.1, -0.004, 0.1))
    assert on(cut, (0, 0, 0.004, 0))


def test_a_gap_whose_two_cuts_cross_is_located(monkeypatch):
    """Round terminals 8 nm apart: each end's first passing cut lies past the other's."""
    message = refusal(round_terminals(monkeypatch, 8))
    assert message.startswith(
        f"junction DBU grid has no positive central span: {JUNCTION}"
    )
    start, end, cut_a, cut_b = places(message)
    assert on(start, point(0, 0)) and on(end, point(0.008, 0))
    assert all(on(cut, (0, 0, 0.008, 0)) for cut in (cut_a, cut_b))
    assert cut_a[0] > cut_b[0]


def test_an_end_piece_split_by_a_hole_under_the_sheet_is_located_at_the_hole(
    monkeypatch,
):
    """A round pad with a hole at x = -0.6..-0.4 um under the sheet: the end piece falls apart there."""
    (pad_ring,) = gdstk.boolean(
        gdstk.Polygon(circle(-5, 0, 5, 1024)),
        gdstk.rectangle((-0.6, -0.2), (-0.4, 0.2)),
        "not",
    )
    pad = ("pad", 1, [tuple(p) for p in pad_ring.points], (-5, 0))
    message = refusal(device(monkeypatch, [pad, ELECTRODE]))
    assert message.startswith(
        f"unsupported junction end topology on verified corridor/grid: {JUNCTION}"
    )
    assert "beside Net pad, the end piece beyond the cut at" in message
    assert "falls apart into separate pieces" in message
    assert "Fix the layout: draw the sheet's end so that" in message
    found = places(message)
    assert any(on(place, (-0.6, -0.2, -0.4, 0.2)) for place in found)


def test_an_end_piece_two_conductors_of_its_net_meet_is_located(monkeypatch):
    """Two conductors of the pad's Net meet the sheet's free end at x = -1..0 um, y = 0..0.5 um."""
    pad = ("pad", 1, [(-20, -10), (0, -10), (0, 0), (-20, 0)], (-10, -5))
    pad2 = ("pad2", 3, [(-20, 0.5), (0, 0.5), (0, 10), (-20, 10)], (-10, 5))
    nets = {**NETS, "pad": ("chip/device/pad", "chip/device/pad2")}
    message = refusal(device(monkeypatch, [pad, pad2, ELECTRODE]), nets)
    assert message.startswith(
        "junction end lacks an unambiguous source Entity/material/Z lineage: "
        f"{JUNCTION}"
    )
    assert (
        "these conductors of Net pad: chip/device/pad on layer 1/0, "
        "chip/device/pad2 on layer 3/0"
    ) in message
    assert "Fix the layout: land each end of the sheet on one conductor" in message
    (place,) = places(message)
    assert on(place, (-1, 0, 0, 0.5)) and on((-1, 0, 0, 0.5), place)


# -------------------------------------------------------------- the terminal's outline


def test_a_terminal_outline_with_a_spline_is_located_by_its_ends(monkeypatch):
    """The pad's authored outline holds a B-spline from (0, -10) um to (0, 10) um under the sheet."""
    component = device(monkeypatch, [PAD, ELECTRODE])
    component.info["component_semantics"]["boundary_curves"] = [
        {
            "boundary_id": "PAD_OUTER",
            "entity_id": "pad",
            "selector_point_um": (-10.0, 0.0),
            "role": "outer",
            "curves": [
                {"kind": "line_segment", "points_um": [[-20, -10], [0, -10]]},
                {
                    "kind": "bspline",
                    "points_um": [[0, -10], [1, 0], [0, 10]],
                    "degree": 2,
                },
                {"kind": "line_segment", "points_um": [[0, 10], [-20, 10]]},
                {"kind": "line_segment", "points_um": [[-20, 10], [-20, -10]]},
            ],
        }
    ]
    message = refusal(component)
    assert message.startswith(
        "junction terminal outline curve 'bspline' is not supported; a terminal is built "
        "from lines and circular arcs: chip/device/pad on layer 1/0 (grid g = 0.001 um)"
    )
    assert message.endswith(
        "Fix the layout: draw the curve as lines and circular arcs."
    )
    start, end = places(message)
    assert on(start, point(0, -10)) and on(end, point(0, 10))


# ------------------------------------------------------ hosts with no common film plane


def test_a_port_on_conductors_without_a_film_plane_names_the_declaration(monkeypatch):
    """The metal level declared a finite volume gives the hosts no film plane to attach the sheet to."""
    metal = LAYER_STACK.layers["metal"]
    volume = PhysicalLayerDeclaration(
        "metal",
        "metal",
        {"level": "metal"},
        (0, 0.2),
        "PEC",
        "finite_volume",
        {"axial_span_um": 0.2},
        {},
        {},
    )
    stack = SimpleNamespace(
        layers={
            **LAYER_STACK.layers,
            "metal": SimpleNamespace(
                **{
                    **vars(metal),
                    "info": {**metal.info, "physical_layer": volume.to_payload()},
                }
            ),
        }
    )
    with pytest.raises(ValueError) as error:
        check(
            EigenmodeSim,
            two_pads(monkeypatch, gap_um=10.0, port=True),
            layer_stack=stack,
        )
    message = str(error.value)
    assert message.startswith(
        "junction port hosts lack one common authored support plane: lumped port "
        "'chip/P1' on layer 9/0 (grid g = 0.001 um): the sheet at"
    )
    assert "chip/pad_0000 (z 0..0.2 um, no film plane)" in message
    assert (
        "Fix the declaration: land the port on conductors of one film level" in message
    )
    (place,) = places(message)
    # ``two_pads`` draws the sheet from x = 15 um to 35 um, y = 8..12 um.
    assert on(place, (15, 8, 35, 12)) and on((15, 8, 35, 12), place)
