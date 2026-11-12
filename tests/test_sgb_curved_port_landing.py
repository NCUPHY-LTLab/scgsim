"""A lumped port landing on an exactly reconstructed circular pad shares the pad's arc edges (review R20-02).

The adapter's host overlap is the host's raw GDS polyline; SGB has rebuilt the pad as an exact circle.  The port
footprint is therefore the authored region whole, and labelling reads the landing from the host footprint SGB
built, so the port and the pad meet along one arc edge that carries the pad's arc only.
"""

from __future__ import annotations

import math
from dataclasses import replace

import gdstk
import pytest
from sgb_inputs import gds_pad_in_ground
from test_sgb_band_labels import PORT_CASES, all_interface_insets

from scgsim.geometry.sgb.cutfamilies.requests import RefinementRequest
from scgsim.geometry.sgb.emission.measures import planned_face_area_um2
from scgsim.geometry.sgb.model import build_conformal_model

PORT_GROUP = "PORT::PORT_SHEET__P1__0000"
PAD_CENTRE_UM, PAD_RADIUS_UM = (50.0, 50.0), 10.0
PORT_WIDTH_UM, PORT_HEIGHT_UM = 12.0, 4.0  # the authored rectangle (29, 48) - (41, 52)
GROUND_LANDING_UM2 = (
    1.0 * PORT_HEIGHT_UM
)  # 1 um on the straight edge of the ground cutout
PAD_LANDING_X_UM = (
    41.0  # the port ends 1 um inside the pad, whose left-most point is x = 40
)
AREA_TOLERANCE_UM2 = (
    0.01  # the lattice is 0.001 um: chord drift of the reconstructed circle
)


def circular_pad_input(directory):
    """The pad-in-ground input with the pad's GDS polyline reconstructed as one exact circle.

    The stack file declares the reconstruction on the polygon as gdstk draws it: every drawn
    segment, and drawn vertices 0, n/3 and 2n/3 as anchors.
    """
    drawn = gdstk.ellipse(PAD_CENTRE_UM, PAD_RADIUS_UM, tolerance=0.01)
    count = len(drawn.points)
    reconstruction = {
        "boundary_id": "pad_circle",
        "entity_id": "pad",
        "selector_point_um": list(PAD_CENTRE_UM),
        "kind": "circular_arc",
        "closed": True,
        "segment_indices": list(range(count)),
        "anchor_indices": [0, count // 3, 2 * count // 3],
    }
    effective = gds_pad_in_ground(
        directory,
        pad_polygon=drawn,
        extra_metadata={"boundary_reconstruction": [reconstruction]},
    )
    return effective, effective.build_input


def pad_landing_area_um2() -> float:
    """The part of the port rectangle inside the circle: the strip between the circle's left arc and x = 41."""
    half_height = PORT_HEIGHT_UM / 2
    r = PAD_RADIUS_UM
    # integral of sqrt(r^2 - t^2) over |t| <= half_height: the circle's left arc is x = cx - sqrt(r^2 - t^2)
    arc_integral = 2 * (
        half_height / 2 * math.sqrt(r * r - half_height**2)
        + r * r / 2 * math.asin(half_height / r)
    )
    return PORT_HEIGHT_UM * (PAD_LANDING_X_UM - PAD_CENTRE_UM[0]) + arc_integral


@pytest.fixture(scope="module")
def circular_pad(tmp_path_factory):
    """The effective input and the same input with the circular reconstruction."""
    return circular_pad_input(tmp_path_factory.mktemp("circular_pad"))


@pytest.mark.parametrize("route", ["B", "C"])
def test_the_circular_pad_without_its_port_builds(circular_pad, route):
    """The positive control: every source field except the port sheet region."""
    effective, curved = circular_pad
    model = build_conformal_model(
        replace(curved, port_sheet_regions=()), effective.representation, route
    )
    assert model.report.failures == []
    assert "curved_walls_are_single_cylinders" in {c.name for c in model.report.checks}


@pytest.mark.parametrize("route", ["B", "C"])
def test_the_port_landing_on_the_circular_pad_builds_and_validates(circular_pad, route):
    """With the port both routes build and pass every validation, the cylinder check included."""
    effective, curved = circular_pad
    model = build_conformal_model(curved, effective.representation, route)
    assert model.report.failures == []
    assert "curved_walls_are_single_cylinders" in {c.name for c in model.report.checks}
    # one cylinder per quarter arc of the pad, split at the vertices the port and ground add on it
    for face in model.topology.faces.values():
        if face.curved:
            bottom, top = (model.topology.edges[e] for e in face.cylinder)
            assert bottom.curve is not None and top.curve is not None


@pytest.mark.parametrize("route", ["B", "C"])
def test_the_port_area_is_the_authored_rectangle_without_its_landings(
    circular_pad, route
):
    """12 x 4 um minus the 1 um landing on the ground cutout minus the circular-segment landing on the pad."""
    effective, curved = circular_pad
    model = build_conformal_model(curved, effective.representation, route)
    (face_id,) = model.routed.groups[PORT_GROUP].ids
    area = planned_face_area_um2(model.topology, model.topology.faces[face_id])
    expected = (
        PORT_WIDTH_UM * PORT_HEIGHT_UM - GROUND_LANDING_UM2 - pad_landing_area_um2()
    )
    assert expected == pytest.approx(40.2683, abs=1e-4)
    assert area == pytest.approx(expected, abs=AREA_TOLERANCE_UM2)


@pytest.mark.parametrize("route", ["B", "C"])
def test_the_port_and_the_pad_share_arc_edges_carrying_the_pad_arc_only(
    circular_pad, route
):
    """The port's boundary on the circle is the pad's own arc: the same edges, with no port or polyline carrier."""
    effective, curved = circular_pad
    model = build_conformal_model(curved, effective.representation, route)
    topology = model.topology
    (port_id,) = model.routed.groups[PORT_GROUP].ids
    port_edges = [topology.edges[e] for e, _ in topology.faces[port_id].loops[0]]
    arcs = [edge for edge in port_edges if edge.curve is not None]
    assert arcs, "the port keeps an arc boundary where it leaves the pad"
    assert all(s.startswith("pad/") for edge in arcs for s in edge.sources)
    pad_bottom = {
        e
        for face_id in model.routed.groups["MS::pad__bottom"].ids
        for e, _ in topology.faces[face_id].loops[0]
    }
    assert {edge.id for edge in arcs} <= pad_bottom
    # the pad edges outside the port carry no port source: only the port's own straight sides do
    assert all(
        not any(s.startswith("PORT_SHEET") for s in edge.sources) for edge in arcs
    )


@pytest.mark.parametrize("route", ["B", "C"])
def test_bands_on_the_circular_pad_and_its_port_keep_one_port_face(circular_pad, route):
    """Band seams stop at the active port boundary, and continue through the landing on the pad."""
    effective, curved = circular_pad
    insets = all_interface_insets(effective, route, PORT_CASES["B"][1])
    model = build_conformal_model(
        curved, effective.representation, route, insets=insets
    )
    assert model.report.failures == []
    (face_id,) = model.routed.groups[PORT_GROUP].ids
    area = planned_face_area_um2(model.topology, model.topology.faces[face_id])
    expected = (
        PORT_WIDTH_UM * PORT_HEIGHT_UM - GROUND_LANDING_UM2 - pad_landing_area_um2()
    )
    assert area == pytest.approx(expected, abs=AREA_TOLERANCE_UM2)


@pytest.mark.parametrize("route", ["B", "C"])
def test_a_refinement_curve_across_the_arc_builds(tmp_path, route):
    """A refinement curve that crosses the circular pad cuts its wall into arcs travelled either way; it still builds."""
    effective, curved = circular_pad_input(tmp_path)
    refinement = RefinementRequest("across_arc", 0.0, ((35.0, 49.0), (45.0, 49.0)))
    model = build_conformal_model(
        replace(curved, port_sheet_regions=()),
        effective.representation,
        route,
        refinements=(refinement,),
    )
    assert not model.report.failures
    assert "curved_walls_are_single_cylinders" in {c.name for c in model.report.checks}
