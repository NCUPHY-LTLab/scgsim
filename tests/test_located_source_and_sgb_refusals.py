"""Selector-point and SGB refusals name the object, its layer and the place in um, and who acts.

The third slice of the located layout refusals, beside ``test_located_junction_refusals.py``.
Each test draws the defect at a known place, reaches the refusal from a public entry point
(``GeometryPlan.prepare``, ``simulation_drc``, the source recipes the plan and the stack adapter
run, or the SGB catalogue both simulations build) and reads every coordinate back out of the
message: each must lie on the defect.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace

import gdstk
import pytest
from sgb_inputs import gds_pad_in_ground
from test_located_refusals import GRID, on, places, rectangle
from test_rectangular_junction_geometry import fixture as junction_fixture
from test_sgb_prisms import with_boundaries
from test_simulation_drc import LAYER_STACK, MATERIALS, check, level, two_pads

from scgsim.geometry import GeometryPlan
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    SourceCurveSpec,
)
from scgsim.geometry.primitives.grid import geometry_grid_scope
from scgsim.geometry.sgb.errors import (
    InputError,
    NoGapEdgeError,
    OverlapError,
    SgbError,
)
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue
from scgsim.geometry.source.ground import derived_ground_polygons
from scgsim.geometry.source.occurrence import (
    gds_polygons_for_entity,
    occurrence_ground_polygons_for_entity,
    occurrence_include_polygon,
)
from scgsim.geometry.source.occurrence_validation import (
    validate_ground_contribution_ownership,
)
from scgsim.geometry.source.ports import port_sheet_regions_from_stack_metadata
from scgsim.palace import EigenmodeSim

FIX_SELECTOR = (
    "Fix the layout or the declaration: draw one region under the selector point"
)
# The occurrence transform of a component placed at (100, 50) um.
PLACED = (1, 0, 0, 1, 100, 50)


def point(x, y):
    """A defect that is one point, as a box."""
    return (x, y, x, y)


def pads_with_selectors(monkeypatch, pad_selector=(10.0, 10.0), **options):
    """``two_pads`` with selector points: the pad's (default inside it) and the electrode's."""
    component = two_pads(monkeypatch, gap_um=10.0, **options)
    pad, electrode = component.info["component_semantics"]["conductor_regions"]
    for region, selector in ((pad, pad_selector), (electrode, (45.0, 10.0))):
        region["geometry"] = {
            "geometry_source": "gds_polygon",
            "selector_point_um": selector,
        }
    return component


def refused(error_type, **options):
    """``simulation_drc``'s refusal of a layout, as text."""
    with pytest.raises(error_type) as error:
        check(EigenmodeSim, **options)
    return str(error.value)


# ---------------------------------------------------------------- conductor selector points


def test_a_selector_point_in_a_placed_component_is_located_in_the_layout(monkeypatch):
    """The pad's selector (25, 25) um of the component, placed at (100, 50) um, misses the pad at (125, 75) um."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    device = pads_with_selectors(monkeypatch, pad_selector=(25.0, 25.0))
    chip = gf.Component()
    reference = chip << device
    reference.move((100, 50))
    chip.info["component_semantics"] = {"schema_version": 2, "conductor_regions": []}
    plan = GeometryPlan(
        chip,
        root_id="chip",
        layer_stack=LAYER_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    plan.register_instance("chip/device", reference)
    with pytest.raises(ValueError) as error:
        plan.prepare()
    message = str(error.value)
    assert message.startswith(
        "chip/device/pad selector point (25.0, 25.0) matched 0 fused polygons: "
        "chip/device/pad on layer 1/0 (grid g = 0.001 um): the selector point lies at"
    )
    assert FIX_SELECTOR in message
    at, extent = places(message)
    assert on(at, point(125, 75))
    assert on(extent, (100, 50, 120, 70)) and on((100, 50, 120, 70), extent)


def pad_entity(**geometry):
    """A conductor Entity reading GDS layer 1/0, with the selector settings given."""
    build, _ = junction_fixture()
    return replace(
        build.entities[0],
        semantic_id="chip/pad",
        geometry={"gds_layer": 1, "gds_datatype": 0, **geometry},
    )


PAD_LAYER = {(1, 0): (gdstk.rectangle((0, 0), (20, 20)),)}


def test_two_selector_points_in_one_region_are_located():
    """A selector point and an excluded point both lie in the 20 um pad at (5, 5) and (6, 6) um."""
    entity = pad_entity(
        route_ab_fused_selector_mode=True,
        selector_point_um=(5.0, 5.0),
        exclude_selector_points_um=[(6.0, 6.0)],
    )
    with geometry_grid_scope(GRID), pytest.raises(ValueError) as error:
        gds_polygons_for_entity(entity, polygons_by_layer=PAD_LAYER)
    message = str(error.value)
    assert message.startswith(
        "chip/pad selector points must match distinct polygons: chip/pad on layer 1/0"
    )
    assert "Fix the declaration or the layout: give each selector point" in message
    first, second, region = places(message)
    assert on(first, point(5, 5)) and on(second, point(6, 6))
    assert on(region, (0, 0, 20, 20))


def test_an_unfused_selector_point_that_misses_is_located():
    """A stack-file selector at (30, 30) um misses the 20 um pad on layer 1/0."""
    entity = pad_entity(selector_point_um=(30.0, 30.0))
    with geometry_grid_scope(GRID), pytest.raises(ValueError) as error:
        gds_polygons_for_entity(entity, polygons_by_layer=PAD_LAYER)
    message = str(error.value)
    assert message.startswith(
        "chip/pad selector_point_um matched 0 polygons: chip/pad on layer 1/0"
    )
    assert FIX_SELECTOR in message
    at, extent = places(message)
    assert on(at, point(30, 30)) and on(extent, (0, 0, 20, 20))


# ----------------------------------------------------------------------- port selector points


def test_a_port_whose_centre_misses_its_sheet_is_located(monkeypatch):
    """The port's centre (25, 30) um lies off its sheet, drawn at x = 15..35 um, y = 8..12 um."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    component = pads_with_selectors(monkeypatch)
    component.add_polygon(((15, 8), (35, 8), (35, 12), (15, 12)), layer=(9, 0))
    component.add_port(name="P1", center=(25, 30), width=4, orientation=0, layer=(9, 0))
    component.info["component_semantics"]["ports"] = [
        {"name": "P1", "layer": [9, 0], "target_layer": "metal"}
    ]
    assert isinstance(component, gf.Component)
    message = refused(ValueError, component=component)
    assert message.startswith(
        "GeometryPlan port 'chip/P1' matched 0 local sheet polygons: chip/P1 on layer 9/0"
    )
    assert "the port's centre lies at" in message
    assert "Fix the layout: draw the port's sheet as one polygon" in message
    at, extent = places(message)
    assert on(at, point(25, 30)) and on(extent, (15, 8, 35, 12))


def test_a_stack_port_selector_that_misses_is_located():
    """A stack-file port on layer 9/2 with its selector at (10, 10) um, off the sheet at (1, 1)..(3, 3) um."""
    record = {
        "source": "palace_lumped_port_sheet",
        "layer": 9,
        "datatype": 2,
        "name": "P1",
        "port_index": 1,
        "target_layer": "M5",
        "direction": (1, 0, 0),
        "direction_sign_convention": "authored",
        "selector_point_um": (10.0, 10.0),
    }
    sheet = gdstk.Polygon([(1, 1), (3, 1), (3, 3), (1, 3)], layer=9, datatype=2)
    with geometry_grid_scope(GRID), pytest.raises(ValueError) as error:
        port_sheet_regions_from_stack_metadata(
            {"port_sheet_source_layers": (record,)},
            polygons_by_layer={(9, 2): (sheet,)},
            host_entities=(),
            host_polygons=(),
        )
    message = str(error.value)
    assert message.startswith(
        "port sheet 'P1' selector matched 0 polygons: P1 on layer 9/2"
    )
    assert "Fix the layout or the declaration: draw the port's sheet" in message
    at, extent = places(message)
    assert on(at, point(10, 10)) and on(extent, (1, 1, 3, 3))


# -------------------------------------------------------------------------- Grounds


def ground_entity(**geometry):
    """A Ground masked on layer 1/0, with the geometry given."""
    build, _ = junction_fixture()
    return replace(
        build.entities[0],
        semantic_id="chip/ground",
        net_id="Ground",
        geometry={"mask_layer": (1, 0), **geometry},
    )


BOUNDS = {"x_min_um": 0, "x_max_um": 20, "y_min_um": 0, "y_max_um": 20}


def test_a_ground_include_selector_that_misses_is_located():
    """The include selector (10, 10) um misses the include polygon at (2, 15)..(4, 17) um on 3/0."""
    ground = ground_entity(
        include_layer=(3, 0), include_selector_points_um=((10.0, 10.0),)
    )
    with geometry_grid_scope(GRID), pytest.raises(ValueError) as error:
        derived_ground_polygons(
            ground,
            polygons_by_layer={(3, 0): (gdstk.rectangle((2, 15), (4, 17)),)},
            cell_bounds_um=BOUNDS,
            domain_bounds_by_semantic_id={},
            plane_polygons=(gdstk.rectangle((0, 0), (20, 20)),),
        )
    message = str(error.value)
    assert message.startswith(
        "chip/ground include selector (10.0, 10.0) matched 0 source polygons: "
        "chip/ground on layer 3/0"
    )
    assert FIX_SELECTOR in message
    at, extent = places(message)
    assert on(at, point(10, 10)) and on(extent, (2, 15, 4, 17))


def test_a_ground_its_mask_covers_whole_is_located_at_the_plane():
    """A mask over the whole 20 um plane leaves the Ground nothing."""
    with geometry_grid_scope(GRID), pytest.raises(ValueError) as error:
        derived_ground_polygons(
            ground_entity(),
            polygons_by_layer={(1, 0): (gdstk.rectangle((-1, -1), (21, 21)),)},
            cell_bounds_um=BOUNDS,
            domain_bounds_by_semantic_id={},
            plane_polygons=(gdstk.rectangle((0, 0), (20, 20)),),
        )
    message = str(error.value)
    assert message.startswith(
        "chip/ground derived ground must lower to at least one planar region: "
        "chip/ground on layer 1/0"
    )
    assert "Fix the layout or the declaration: leave part of the plane" in message
    (place,) = places(message)
    assert on(place, (0, 0, 20, 20)) and on((0, 0, 20, 20), place)


def test_an_occurrence_ground_without_its_plane_is_located_where_the_occurrence_draws():
    """A Ground occurrence placed at (100, 50) um draws only on 1/0, nothing on its plane layer 2/0."""
    with geometry_grid_scope(GRID), pytest.raises(ValueError) as error:
        occurrence_ground_polygons_for_entity(
            ground_entity(),
            cell=None,
            transform=PLACED,
            plane_layer=(2, 0),
            polygons_by_layer={(1, 0): (gdstk.rectangle((2, 3), (4, 5)),)},
        )
    message = str(error.value)
    assert message.startswith(
        "chip/ground occurrence has no declared plane-bound geometry: chip/ground on layer 2/0"
    )
    assert "Fix the layout or the declaration: draw the Ground's plane" in message
    (place,) = places(message)
    assert on(place, (102, 53, 104, 55)) and on((102, 53, 104, 55), place)


def test_a_ground_contribution_that_misses_is_located_in_the_layout():
    """A contribution's selector at (110, 60) um misses the local fill at (2, 3)..(4, 5) of a component at (100, 50) um."""
    record = {
        "layer": [3, 0],
        "selector_point_um": (110.0, 60.0),
        "source_local_id": "fill",
    }
    with geometry_grid_scope(GRID), pytest.raises(ValueError) as error:
        occurrence_include_polygon(
            record,
            cell=None,
            transform=PLACED,
            polygons_by_layer={(3, 0): (gdstk.rectangle((2, 3), (4, 5)),)},
        )
    message = str(error.value)
    assert message.startswith(
        "ground contribution 'fill' matched 0 local source polygons: fill on layer 3/0"
    )
    assert FIX_SELECTOR in message
    at, extent = places(message)
    assert on(at, point(110, 60)) and on(extent, (102, 53, 104, 55))


def test_a_ground_contribution_over_a_named_conductor_is_located_at_the_overlap():
    """A contribution's region (0, 0)..(10, 10) um overlaps the local Entity drawn at (8, 8)..(12, 12) um."""
    plane = {
        "geometry": {
            "source_occurrence_includes": [
                {
                    "polygon": {"exterior": rectangle(0, 0, 10, 10), "holes": []},
                    "source_occurrence_path": "chip/device",
                    "source_local_id": "fill",
                    "level": "metal",
                    "layer": [3, 0],
                }
            ]
        }
    }
    other = {
        "semantic_id": "chip/device/pad",
        "metadata": {
            "source_occurrence_path": "chip/device",
            "logical_layer_id": "metal",
        },
        "geometry": {
            "source_occurrence_polygons_um": [
                {"exterior": rectangle(8, 8, 12, 12), "holes": []}
            ]
        },
    }
    with geometry_grid_scope(GRID), pytest.raises(ValueError) as error:
        validate_ground_contribution_ownership([plane, other])
    message = str(error.value)
    assert message.startswith(
        "ground contribution 'fill' also contains local Entity 'chip/device/pad': fill on layer 3/0"
    )
    assert "Fix the layout or the declaration: select an include region" in message
    (place,) = places(message)
    assert on(place, (8, 8, 10, 10)) and on((8, 8, 10, 10), place)


# --------------------------------------------------------------- boundary curves (SGB)


SQUARE = [
    {"kind": "line_segment", "points_um": [[0, 0], [20, 0]]},
    {"kind": "line_segment", "points_um": [[20, 0], [20, 20]]},
    {"kind": "line_segment", "points_um": [[20, 20], [0, 20]]},
    {"kind": "line_segment", "points_um": [[0, 20], [0, 0]]},
]


def with_pad_boundary(monkeypatch, curves=SQUARE, selector=(10.0, 10.0), extra=()):
    """The pads of ``two_pads`` with an authored boundary of the pad, and ``extra`` more records."""
    component = pads_with_selectors(monkeypatch)
    record = {
        "boundary_id": "PAD_OUTER",
        "entity_id": "pad",
        "selector_point_um": selector,
        "role": "outer",
        "curves": curves,
    }
    component.info["component_semantics"]["boundary_curves"] = [record, *extra]
    return component


def test_a_boundary_selector_that_misses_its_entity_is_located(monkeypatch):
    """The boundary's selector (25, 25) um lies outside the pad, drawn at (0, 0)..(20, 20) um on 1/0."""
    message = refused(
        InputError, component=with_pad_boundary(monkeypatch, selector=(25.0, 25.0))
    )
    assert message.startswith(
        "chip/PAD_OUTER selector must bind one Entity-owned logical polygon of "
        "'chip/pad'; it binds 0: the selector point lies at"
    )
    assert "on layer 1/0" in message
    assert "Fix the declaration: put the boundary's selector point inside" in message
    at, extent = places(message)
    assert on(at, point(25, 25)) and on(extent, (0, 0, 20, 20))


def test_a_chain_with_a_gap_between_curves_is_located_at_both_ends(monkeypatch):
    """Curve 1 starts at (20, 1) um where curve 0 ended at (20, 0) um."""
    curves = [dict(curve) for curve in SQUARE]
    curves[1] = {"kind": "line_segment", "points_um": [[20, 1], [20, 20]]}
    message = refused(InputError, component=with_pad_boundary(monkeypatch, curves))
    assert message.startswith(
        "chip/PAD_OUTER (Entity 'chip/pad', layer 1/0) curve 1 does not start where the "
        "previous curve ends: it starts at"
    )
    assert message.endswith(
        "Fix the layout: start each curve where the one before it ends."
    )
    start, previous_end = places(message)
    assert on(start, point(20, 1)) and on(previous_end, point(20, 0))


def test_a_chain_that_does_not_close_is_located_at_both_ends(monkeypatch):
    """The last curve ends at (0, 1) um, not at the chain's start (0, 0) um."""
    curves = [dict(curve) for curve in SQUARE]
    curves[3] = {"kind": "line_segment", "points_um": [[0, 20], [0, 1]]}
    message = refused(InputError, component=with_pad_boundary(monkeypatch, curves))
    assert message.startswith(
        "chip/PAD_OUTER (Entity 'chip/pad') curves do not form one closed chain: the "
        "chain on layer 1/0 starts at"
    )
    assert message.endswith(
        "Fix the layout: end the last curve where the first starts."
    )
    start, end = places(message)
    assert on(start, point(0, 0)) and on(end, point(0, 1))


def test_a_curve_end_off_the_grid_names_who_acts(monkeypatch):
    """Curve 0 ends at (20.0004, 0) um, between grid points."""
    curves = [
        {"kind": "line_segment", "points_um": [[0, 0], [20.0004, 0]]},
        {"kind": "line_segment", "points_um": [[20.0004, 0], [20, 20]]},
        SQUARE[2],
        SQUARE[3],
    ]
    message = refused(InputError, component=with_pad_boundary(monkeypatch, curves))
    assert message.startswith(
        "chip/PAD_OUTER (Entity 'chip/pad', layer 1/0) curve 0: end point "
        "(20.0004, 0.0) um is off the 0.001 um model grid"
    )
    assert message.endswith("Fix the layout: draw the ends of each curve on the grid.")
    (place,) = places(message)
    assert on(place, point(20.0004, 0), slack=1e-9)


def test_two_boundaries_of_one_ring_are_located_at_the_ring(monkeypatch):
    """Two complete boundaries of the pad both select its ring (0, 0)..(20, 20) um."""
    second = {
        "boundary_id": "PAD_OUTER_2",
        "entity_id": "pad",
        "selector_point_um": (10.0, 10.0),
        "role": "outer",
        "curves": SQUARE,
    }
    message = refused(
        InputError, component=with_pad_boundary(monkeypatch, extra=(second,))
    )
    assert message.startswith(
        "multiple complete authored boundaries claim ring ('chip/pad', 0, 0): "
        "chip/PAD_OUTER and chip/PAD_OUTER_2 (Entity 'chip/pad' on layer 1/0)"
    )
    assert message.endswith(
        "Fix the declaration: give each ring one complete boundary record."
    )
    (place,) = places(message)
    assert on(place, (0, 0, 20, 20)) and on((0, 0, 20, 20), place)


def square_chain(boundary_id, role="outer", hole_index=None):
    """The pad of ``gds_pad_in_ground``, (40, 40)..(60, 60) um, as an authored chain of lines."""
    corners = ((40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0))
    return BoundaryCurveChainSpec(
        boundary_id,
        "pad",
        (50.0, 50.0),
        tuple(
            SourceCurveSpec("line_segment", (corners[k], corners[(k + 1) % 4]))
            for k in range(4)
        ),
        role=role,
        hole_index=hole_index,
    )


def test_an_authored_and_a_reconstructed_boundary_of_one_ring_are_located(tmp_path):
    """An authored chain and a reconstruction both select the pad's ring (40, 40)..(60, 60) um.

    The reconstruction is declared in the stack file, on the pad as drawn: its first two sides.
    """
    reconstruction = {
        "boundary_id": "pad_corner",
        "entity_id": "pad",
        "selector_point_um": [50.0, 50.0],
        "kind": "circular_arc",
        "segment_indices": [0, 1],
        "anchor_indices": [0, 1, 2],
    }
    pad = gds_pad_in_ground(
        tmp_path,
        with_port=False,
        extra_metadata={"boundary_reconstruction": [reconstruction]},
    )
    curved = with_boundaries(pad, boundary_curves=(square_chain("pad_outline"),))
    with pytest.raises(InputError) as error:
        build_prism_catalogue(curved.build_input, curved.representation)
    message = str(error.value)
    assert message.startswith(
        "authored and reconstructed boundaries both claim ring ('pad', 0, 0): "
        "pad_outline and pad_corner (Entity 'pad'"
    )
    assert "Fix the declaration: describe the ring by one authored boundary" in message
    (place,) = places(message)
    assert on(place, (40, 40, 60, 60)) and on((40, 40, 60, 60), place)


def test_a_boundary_of_a_hole_the_polygon_lacks_is_located_at_the_polygon(tmp_path):
    """A hole boundary of the pad, which has no hole, names the pad at (40, 40)..(60, 60) um."""
    pad = gds_pad_in_ground(tmp_path, with_port=False)
    curved = with_boundaries(
        pad, boundary_curves=(square_chain("pad_hole", role="hole", hole_index=0),)
    )
    with pytest.raises(InputError) as error:
        build_prism_catalogue(curved.build_input, curved.representation)
    message = str(error.value)
    assert message.startswith("pad_hole: 'pad' polygon 0 has no hole 0: the polygon")
    assert (
        "has 0. Fix the declaration: give the hole_index of one of its holes" in message
    )
    (place,) = places(message)
    assert on(place, (40, 40, 60, 60)) and on((40, 40, 60, 60), place)


# ------------------------------------------------------------- port sheets and bands (SGB)


def test_two_port_sheets_over_each_other_name_both_ports_and_their_layers(monkeypatch):
    """Port P2's sheet on 10/0, at (16, 10)..(36, 14) um, overlaps P1's on 9/0 at (15, 8)..(35, 12) um."""
    component = pads_with_selectors(monkeypatch, port=True)
    component.add_polygon(((16, 10), (36, 10), (36, 14), (16, 14)), layer=(10, 0))
    component.add_port(
        name="P2", center=(26, 13), width=4, orientation=0, layer=(10, 0)
    )
    component.info["component_semantics"]["ports"].append(
        {"name": "P2", "layer": [10, 0], "target_layer": "metal"}
    )
    message = refused(OverlapError, component=component)
    assert message.startswith("a port sheet overlaps another sheet near")
    assert "PORT_SHEET__chip/P1__0000 (port 'chip/P1') on layer 9/0" in message
    assert "PORT_SHEET__chip/P2__0000 (port 'chip/P2') on layer 10/0" in message
    assert message.endswith(
        "Fix the layout: draw each port sheet clear of every other sheet but the "
        "conductors it lands on."
    )
    (place,) = places(message)
    assert on(place, (16, 10, 35, 12))


def test_a_banded_face_without_a_gap_edge_is_located(monkeypatch):
    """The electrode abuts the pad of its own Net and fills the die's right part: its top has no gap edge."""
    specs = {
        "MA": {
            "film_thickness_m": 2e-9,
            "permittivity": 4,
            "inset_margins_um": [0.5],
            "inset_margins_by_entity_um": {"chip/electrode": [0.5]},
        }
    }
    component = two_pads(monkeypatch, gap_um=0.0)
    pad, electrode = component.info["component_semantics"]["conductor_regions"]
    for region, selector in ((pad, (10.0, 10.0)), (electrode, (35.0, 10.0))):
        region["geometry"] = {
            "geometry_source": "gds_polygon",
            "selector_point_um": selector,
        }
    message = refused(
        NoGapEdgeError,
        component=component,
        nets={"G": ("chip/pad", "chip/electrode")},
        surface_epr_specs=specs,
    )
    assert message.startswith(
        "inset request 'epr::chip/electrode::top': the top of 'chip/electrode' has no gap "
        "edge to measure the bands from"
    )
    assert "the face on layer metal lies at z = 0.2 um within" in message
    assert "Fix the declaration or the layout: drop the band margins" in message
    (place,) = places(message)
    assert on(place, (20, 0, 50, 20)) and on((20, 0, 50, 20), place)


# ---------------------------------------------------------------- emission (SGB)


def test_a_film_thinner_than_the_emission_tolerance_is_located(monkeypatch):
    """A metal film 1e-7 um thick leaves vertical edges of that length at the electrode's corners."""
    stack = SimpleNamespace(
        layers={
            **LAYER_STACK.layers,
            "metal": level("metal", "PEC", (0, 1e-7), "conductor"),
        }
    )
    message = refused(
        SgbError, component=pads_with_selectors(monkeypatch), layer_stack=stack
    )
    assert message.startswith("edge shorter than 10 x tolerance (1e-07 um): edge ")
    assert message.endswith(
        "Fix the layer stack: this edge is a film thickness or a gap between levels; "
        "make each at least 1e-06 um."
    )
    ends = re.findall(r"\((-?[\d.]+), (-?[\d.]+), ([-\d.e]+)\) um", message)
    (x0, y0, z0), (x1, y1, z1) = ((float(v) for v in end) for end in ends)
    assert (x0, y0) == (x1, y1)
    assert (x0, y0) in {
        (30, 0),
        (30, 20),
        (60, 0),
        (60, 20),
        (0, 0),
        (0, 20),
        (20, 0),
        (20, 20),
    }
    assert {z0, z1} == {0.0, 1e-7}
