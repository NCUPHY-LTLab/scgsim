"""A GDS boundary reconstruction names the ring as drawn in the GDS, whatever the source stage does to it (Issue #163).

Segment index i is the edge from drawn vertex i to i + 1 of the GDS polygon walk, in its own start
vertex and direction, through any instance transform.  The source stage emits the ring from another
start, possibly the other way round, and with the crossings of a port sheet noded into it; the
records' segments and anchors still land on the drawn ones.  Each GeometryPlan layout goes through
``EigenmodeSim``'s preparation and the SGB build that ``simulation_drc`` runs, and the arc a build
carries is read from the sidecar SGB writes; a record declared in a stack file goes through the GDS
stack adapter and SGB.  No mesher or solver runs.
"""

from __future__ import annotations

import math
from dataclasses import replace

import gdstk
import pytest
from sgb_inputs import gds_pad_in_ground
from test_located_refusals import on, places
from test_lumped_ports_at_any_angle import built_sidecar as built_device_sidecar
from test_lumped_ports_at_any_angle import plan_snapshot as device_plan_snapshot
from test_palace_rectangular_ports import built_sidecar, eigenmode_sim, prepared_port
from test_simulation_drc import MATERIALS, NO_VACUUM_STACK

from scgsim.geometry import GeometryPlan
from scgsim.geometry.construction.curved_geometry.boundaries import (
    circle_through_points,
    source_boundaries,
    source_rings,
)
from scgsim.geometry.models.input import (
    DRAWN_RING_METADATA,
    GdsBoundaryReconstructionSpec,
    RingChain,
    drawn_chain_on_ring,
)
from scgsim.geometry.models.junctions import is_junction_end_piece
from scgsim.geometry.sgb.errors import InputError
from scgsim.geometry.sgb.model import build_conformal_model
from scgsim.geometry.sgb.prisms.catalogue import build_prism_catalogue

GRID_UM = 0.001
PAD_RADIUS_UM = 100.0
# The PDK's circular pad draws its circle with 256 points.
PAD_VERTICES = 256
NETS = {"pad": ("chip/pad",), "electrode": ("chip/electrode",)}
# A rounded pad corner: a quarter circle of radius 20 um about (80, 80) um, drawn with 16 chords.
CORNER_RADIUS_UM = 20.0
CORNER_CHORDS = 16
CORNER = [
    (
        80.0 + CORNER_RADIUS_UM * math.cos(math.pi / 2 * k / CORNER_CHORDS),
        80.0 + CORNER_RADIUS_UM * math.sin(math.pi / 2 * k / CORNER_CHORDS),
    )
    for k in range(CORNER_CHORDS + 1)
]


def round_pad_with_port(monkeypatch, sheet_x_um, sheet_y_um):
    """A 256-point pad of radius 100 um declared as one closed circle, and a port sheet to an electrode.

    The sheet spans ``sheet_y_um`` and runs from ``sheet_x_um``, inside the pad, to 1 um inside
    an electrode whose straight edge lies at x = 120 um.  The reconstruction selects every drawn
    segment; its three anchors are drawn vertices 0, 85 and 170.
    """
    low, high = sheet_y_um
    pad = [
        (
            PAD_RADIUS_UM * math.cos(2 * math.pi * i / PAD_VERTICES),
            PAD_RADIUS_UM * math.sin(2 * math.pi * i / PAD_VERTICES),
        )
        for i in range(PAD_VERTICES)
    ]
    reconstruction = {
        "boundary_id": "PAD_CIRCLE",
        "entity_id": "pad",
        "selector_point_um": (0.0, 0.0),
        "role": "outer",
        "kind": "circular_arc",
        "closed": True,
        "segment_indices": list(range(PAD_VERTICES)),
        "anchor_indices": [0, PAD_VERTICES // 3, 2 * PAD_VERTICES // 3],
    }
    return pad_layout(
        monkeypatch,
        pad,
        electrode=[(120, -30), (150, -30), (150, 30), (120, 30)],
        sheet=[(sheet_x_um, low), (121, low), (121, high), (sheet_x_um, high)],
        port_orientation=0,
        reconstruction=reconstruction,
    )


def cornered_pad(monkeypatch, *, port: bool, reconstruction=None):
    """The square [-100, 100]^2 um pad, its top-right corner rounded, its bottom-left one cut off.

    The cut corner, from (-80, -100) to (-100, -80) um, makes the GDS walk start at a vertex
    the source grid does not start its ring at.  An electrode lies above the pad; with
    ``port`` a 2 um sheet runs from 1 um inside the pad's straight top edge to 1 um inside the
    electrode, whose straight edge lies at y = 120 um.
    """
    return pad_layout(
        monkeypatch,
        [(-80, -100), (100, -100), *CORNER, (-100, 100), (-100, -80)],
        electrode=[(-30, 120), (30, 120), (30, 150), (-30, 150)],
        sheet=[(-1, 99), (1, 99), (1, 121), (-1, 121)] if port else None,
        port_orientation=90,
        reconstruction=reconstruction,
    )


def drawn_pad_ring(component):
    """The pad polygon as its cell stores it in the GDS: the walk the indices name."""
    (polygon,) = component.get_polygons(by="tuple")[(1, 0)]
    dbu = component.kcl.dbu
    return [(point.x * dbu, point.y * dbu) for point in polygon.each_point_hull()]


def corner_reconstruction(drawn):
    """An open arc over the rounded corner's 16 drawn segments, anchored at its ends and middle."""
    corner = {grid_key(point) for point in CORNER}
    count = len(drawn)
    (start,) = [
        index
        for index in range(count)
        if all(
            grid_key(drawn[(index + k) % count]) in corner
            for k in range(CORNER_CHORDS + 1)
        )
    ]
    return {
        "boundary_id": "PAD_CORNER",
        "entity_id": "pad",
        "selector_point_um": (0.0, 0.0),
        "role": "outer",
        "kind": "circular_arc",
        "closed": False,
        "segment_indices": [(start + k) % count for k in range(CORNER_CHORDS)],
        "anchor_indices": [
            start,
            (start + CORNER_CHORDS // 2) % count,
            (start + CORNER_CHORDS) % count,
        ],
    }


def grid_key(point):
    """A point as whole grid steps."""
    return (round(point[0] / GRID_UM), round(point[1] / GRID_UM))


def pad_layout(  # noqa: PLR0913 -- the drawn pad, its electrode, the sheet and the record
    monkeypatch, pad, *, electrode, sheet, port_orientation, reconstruction
):
    """The pad and the electrode as conductors, the sheet as port P1, and the reconstruction record."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon(pad, layer=(1, 0))
    component.add_polygon(electrode, layer=(2, 0))
    # An undeclared die outline keeps the substrate edge clear of the pad.
    component.add_polygon(
        [(-110, -110), (160, -110), (160, 160), (-110, 160)], layer=(50, 0)
    )
    electrode_x = sum(x for x, _ in electrode) / len(electrode)
    electrode_y = sum(y for _, y in electrode) / len(electrode)
    semantics = {
        "schema_version": 2,
        "conductor_regions": [
            {
                "semantic_id": "pad",
                "level": "metal",
                "gds_layer": [1, 0],
                "geometry": {
                    "geometry_source": "gds_polygon",
                    "selector_point_um": (0.0, 0.0),
                },
            },
            {
                "semantic_id": "electrode",
                "level": "metal",
                "gds_layer": [2, 0],
                "geometry": {
                    "geometry_source": "gds_polygon",
                    "selector_point_um": (electrode_x, electrode_y),
                },
            },
        ],
        "ports": [],
    }
    if sheet is not None:
        xs, ys = [x for x, _ in sheet], [y for _, y in sheet]
        component.add_polygon(sheet, layer=(9, 0))
        component.add_port(
            name="P1",
            center=((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2),
            width=(max(ys) - min(ys)) if port_orientation == 0 else max(xs) - min(xs),
            orientation=port_orientation,
            layer=(9, 0),
        )
        semantics["ports"] = [{"name": "P1", "layer": [9, 0], "target_layer": "metal"}]
    if reconstruction is not None:
        semantics["boundary_reconstruction"] = [reconstruction]
    component.info["component_semantics"] = semantics
    return component


def emitted_pad_ring(component):
    """The pad's outer ring as the plan's source grid stage emits it."""
    plan = GeometryPlan(
        component,
        root_id="chip",
        layer_stack=NO_VACUUM_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    plan.set_nets(NETS)
    (pad,) = [p for p in plan.prepare().geometry_input.polygons if p.layer == "1/0"]
    return pad.exterior


def pad_arcs(sidecar):
    """The pad's arc features and the carriers on its authored ring (polygon 0, outer ring)."""
    arcs = [
        feature
        for feature in sidecar["mesh_features"]
        if feature["kind"] == "arc" and feature["source"] == "chip/pad"
    ]
    authored_ring = [
        carrier_id
        for carrier_id, carrier in sidecar["lineage"]["carriers"].items()
        if (carrier.get("entity"), carrier.get("polygon"), carrier.get("ring"))
        == ("chip/pad", 0, 0)
    ]
    return arcs, authored_ring


def noded_pad_vertices(sim):
    """How many vertices the prepared pad ring has: the drawn ones and the crossings noded into it."""
    prepared, _, _ = prepared_port(sim)
    (pad,) = [
        p
        for p in prepared.polygons
        if p.layer == "1/0" and not is_junction_end_piece(p)
    ]
    return len(pad.exterior)


@pytest.mark.parametrize(
    ("sheet_x_um", "sheet_y_um", "noded_vertices"),
    [
        # 1 nm inside the pad's arc where it meets y = +-1 um, as the PDK's circular pad starts.
        (math.sqrt(PAD_RADIUS_UM**2 - 1.0) - GRID_UM, (-1.0, 1.0), PAD_VERTICES + 2),
        (99.9, (-1.0, 2.454), PAD_VERTICES + 1),
    ],
    ids=["two-noded-crossings", "upper-side-through-a-drawn-vertex"],
)
def test_a_closed_reconstruction_crossed_by_a_port_sheet_builds_its_drawn_circle(
    monkeypatch, sheet_x_um, sheet_y_um, noded_vertices
):
    """The port sheet nodes the pad ring, and the whole ring is still one circle through the drawn anchors.

    In the first case the sheet's left edge crosses two chords of the pad.  In the second the
    sheet starts well inside the pad and its upper side, y = 2.454 um, runs through drawn vertex
    1 (99.970, 2.454) um, the end of a declared segment, so only the lower side's crossing is
    noded in.  SGB builds the pad's authored ring from arcs alone, of the drawn radius: an anchor
    carried onto a noded crossing would move the circle.
    """
    component = round_pad_with_port(monkeypatch, sheet_x_um, sheet_y_um)
    assert noded_pad_vertices(eigenmode_sim(component, NETS)) == noded_vertices
    arcs, authored_ring = pad_arcs(built_sidecar(monkeypatch, component, NETS))
    (arc,) = arcs
    assert arc["facts"]["radius_um"] == pytest.approx(PAD_RADIUS_UM, abs=GRID_UM)
    assert authored_ring
    assert all("/arc" in carrier_id for carrier_id in authored_ring)


@pytest.mark.parametrize("port", [False, True], ids=["alone", "with-port"])
def test_an_open_reconstruction_in_drawn_order_lands_on_its_drawn_corner(
    monkeypatch, port
):
    """The record names the rounded corner in the GDS walk's order, and SGB builds the 20 um arc there.

    The source grid emits the pad ring from another vertex than the walk starts at; with the
    port it also nodes the two crossings on the top edge into it, ahead of the corner.  Read
    along the emitted ring, the same indices would name a run one vertex off the corner.
    """
    drawn = drawn_pad_ring(cornered_pad(monkeypatch, port=False))
    component = cornered_pad(
        monkeypatch, port=port, reconstruction=corner_reconstruction(drawn)
    )
    emitted = emitted_pad_ring(component)
    assert grid_key(emitted[0]) != grid_key(drawn[0])
    assert len(emitted) == len(drawn) + (2 if port else 0)
    arcs, _ = pad_arcs(built_sidecar(monkeypatch, component, NETS))
    (arc,) = arcs
    assert arc["facts"]["radius_um"] == pytest.approx(CORNER_RADIUS_UM, abs=GRID_UM)


@pytest.mark.parametrize("port", [False, True], ids=["alone", "with-port"])
def test_an_open_reconstruction_through_a_mirrored_occurrence_lands_on_its_drawn_corner(
    monkeypatch, port
):
    """A device placed mirrored keeps the corner its record names in the device cell's drawn order.

    The mirror about y = 0 carries each drawn vertex to its place and reverses the walk's
    direction, and the source grid emits the ring from another vertex than the mirrored walk
    starts at.  The record's indices still name the device cell's walk, and SGB builds the
    20 um arc on the mirrored corner.
    """
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    drawn = drawn_pad_ring(cornered_pad(monkeypatch, port=False))
    device = cornered_pad(
        monkeypatch, port=port, reconstruction=corner_reconstruction(drawn)
    )
    chip = gf.Component()
    reference = chip << device
    reference.dmirror_y(0)
    chip.add_polygon(
        [(-120, -170), (170, -170), (170, 120), (-120, 120)], layer=(50, 0)
    )
    chip.info["component_semantics"] = {"schema_version": 2, "conductor_regions": []}
    (pad,) = [
        p
        for p in device_plan_snapshot(chip, reference).geometry_input.polygons
        if p.layer == "1/0"
    ]
    mirrored = [grid_key((x, -y)) for x, y in drawn]
    assert {grid_key(point) for point in pad.exterior} >= set(mirrored)
    assert grid_key(pad.exterior[0]) != mirrored[0]
    sidecar = built_device_sidecar(monkeypatch, chip, reference)
    (arc,) = [
        feature
        for feature in sidecar["mesh_features"]
        if feature["kind"] == "arc" and feature["source"] == "chip/device/pad"
    ]
    assert arc["facts"]["radius_um"] == pytest.approx(CORNER_RADIUS_UM, abs=GRID_UM)


def test_a_closed_reconstruction_takes_its_anchors_in_drawn_order(monkeypatch):
    """The circle passes through the three drawn vertices the record names, not three others.

    The pad is an ellipse of semi-axes 100 and 80 um drawn with 128 points, so three of its
    vertices fix one circle.  The source grid emits the ring from another vertex than the GDS
    walk starts at.
    """
    count = 128
    pad = [
        (
            100.0 * math.cos(2 * math.pi * i / count),
            80.0 * math.sin(2 * math.pi * i / count),
        )
        for i in range(count)
    ]
    anchors = [0, count // 3, 2 * count // 3]
    record = {
        "boundary_id": "PAD_CIRCLE",
        "entity_id": "pad",
        "selector_point_um": (0.0, 0.0),
        "role": "outer",
        "kind": "circular_arc",
        "closed": True,
        "segment_indices": list(range(count)),
        "anchor_indices": anchors,
    }
    component = pad_layout(
        monkeypatch,
        pad,
        electrode=[(120, -30), (150, -30), (150, 30), (120, 30)],
        sheet=None,
        port_orientation=0,
        reconstruction=record,
    )
    drawn = drawn_pad_ring(component)
    assert len(drawn) == count
    assert grid_key(emitted_pad_ring(component)[0]) != grid_key(drawn[0])
    _, radius = circle_through_points([drawn[i] for i in anchors])
    arcs, _ = pad_arcs(built_sidecar(monkeypatch, component, NETS))
    (arc,) = arcs
    assert arc["facts"]["radius_um"] == pytest.approx(radius, abs=GRID_UM)


@pytest.mark.parametrize("route", ["B", "C"])
def test_a_closed_reconstruction_in_a_stack_file_is_carried_by_the_grid_stage(
    tmp_path, route
):
    """The adapter's grid stage nodes the port into the drawn circle and carries the stack's record onto it.

    The pad is a polygon circle of radius 10 um in a ground cutout, and the port sheet ends 1 um
    inside it.  The record, declared in the stack file on the drawn polygon, names every one of
    its segments; SGB rebuilds the pad as one exact circle.
    """
    drawn = gdstk.ellipse((50.0, 50.0), 10.0, tolerance=0.01)
    count = len(drawn.points)
    record = {
        "boundary_id": "pad_circle",
        "entity_id": "pad",
        "selector_point_um": [50.0, 50.0],
        "kind": "circular_arc",
        "closed": True,
        "segment_indices": list(range(count)),
        "anchor_indices": [0, count // 3, 2 * count // 3],
    }
    effective = gds_pad_in_ground(
        tmp_path,
        pad_polygon=drawn,
        extra_metadata={"boundary_reconstruction": [record]},
    )
    (pad,) = [p for p in effective.build_input.polygons if p.object_name == "pad"]
    assert len(pad.exterior) == count + 2
    model = build_conformal_model(
        effective.build_input, effective.representation, route
    )
    assert model.report.failures == []
    assert "curved_walls_are_single_cylinders" in {c.name for c in model.report.checks}
    assert any(face.curved for face in model.topology.faces.values())


def test_drawn_segments_are_found_on_a_ring_that_runs_the_other_way():
    """The shared mapping reads a ring that starts elsewhere, runs backwards and holds an inserted vertex.

    Drawn a, b, c, d, e; the ring holds them as c, x, b, a, e, d, with x noded in between c
    and b.  The chain comes back in the ring's order, and an open chain's anchors with it.
    """
    drawn = ["a", "b", "c", "d", "e"]
    ring = ["c", "x", "b", "a", "e", "d"]
    # Drawn segments b-c and c-d, anchored at b, c and d: on the ring d-c, c-x, x-b.
    assert drawn_chain_on_ring((1, 2), (1, 2, 3), False, drawn, ring) == RingChain(
        (5, 0, 1), (5, 0, 2)
    )
    # The whole ring, anchored at a, c and e: a closed chain keeps its first anchor first.
    assert drawn_chain_on_ring(
        (0, 1, 2, 3, 4), (0, 2, 4), True, drawn, ring
    ) == RingChain((3, 4, 5, 0, 1, 2), (3, 0, 4))
    # A ring that meets the drawn vertices out of order, or lacks one, holds no drawn chain.
    assert (
        drawn_chain_on_ring((0,), (0, 1), False, drawn, ["a", "c", "b", "d", "e"])
        is None
    )
    assert drawn_chain_on_ring((0,), (0, 1), False, drawn, ["a", "b", "c", "d"]) is None


# ------------------------------------------------------------------ located refusals

# The square pad of ``gds_pad_in_ground``: its ring and the GDS layer it is drawn on.
SQUARE_PAD = (40.0, 40.0, 60.0, 60.0)
PAD_LAYER = "2/0"


def square_pad_corner(**metadata):
    """An open arc over the square pad's first two sides, anchored at its first three vertices."""
    return GdsBoundaryReconstructionSpec(
        "pad_corner",
        "pad",
        (50.0, 50.0),
        "circular_arc",
        segment_indices=(0, 1),
        anchor_indices=(0, 1, 2),
        metadata=metadata,
    )


def refusal(stage, effective, record):
    """The refusal of the binding stage that binds the record on the effective input."""
    build_input = replace(effective.build_input, boundary_reconstruction=(record,))
    if stage == "sgb":
        with pytest.raises(InputError) as error:
            build_prism_catalogue(build_input, effective.representation)
    else:
        with pytest.raises(ValueError) as error:
            source_boundaries(build_input, source_rings(build_input))
    return str(error.value)


@pytest.mark.parametrize("stage", ["construction", "sgb"])
def test_a_record_attached_after_the_source_stage_is_located_at_its_ring(
    tmp_path, stage
):
    """A record attached to the gridded input carries no drawn ring: the refusal names the ring.

    The pad ring is (40, 40)..(60, 60) um, on layer 2/0.
    """
    effective = gds_pad_in_ground(tmp_path, with_port=False)
    message = refusal(stage, effective, square_pad_corner())
    assert message.startswith(
        f"pad_corner (Entity 'pad', layer {PAD_LAYER}): the ring it selects has no "
        "drawn GDS ring recorded for it: the ring at "
    )
    assert "Fix the declaration: declare the reconstruction with the layout" in message
    (place,) = places(message)
    assert on(place, SQUARE_PAD) and on(SQUARE_PAD, place)


@pytest.mark.parametrize("stage", ["construction", "sgb"])
@pytest.mark.parametrize(
    ("drawn", "faults"),
    [
        # The pad's corner (40, 60) drawn 1 um higher: the ring lacks it.
        (
            ((40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 61.0)),
            {(40.0, 61.0): "is missing from it"},
        ),
        # Two corners swapped: the ring meets them out of drawn order.
        (
            ((40.0, 40.0), (60.0, 60.0), (60.0, 40.0), (40.0, 60.0)),
            {
                (40.0, 40.0): "is out of drawn order",
                (60.0, 60.0): "is out of drawn order",
                (60.0, 40.0): "is out of drawn order",
            },
        ),
    ],
    ids=["missing-vertex", "out-of-order"],
)
def test_a_ring_that_lost_its_drawn_order_is_located_at_the_drawn_vertex(
    tmp_path, stage, drawn, faults
):
    """The refusal names each drawn vertex the ring lacks or meets out of order, where it was drawn."""
    effective = gds_pad_in_ground(tmp_path, with_port=False)
    record = square_pad_corner(**{DRAWN_RING_METADATA: drawn})
    message = refusal(stage, effective, record)
    assert message.startswith(
        f"pad_corner (Entity 'pad', layer {PAD_LAYER}): the ring it selects no longer "
        "holds each vertex of its drawn ring once, in drawn order: "
    )
    assert "Fix the layout: draw the ring's vertices at least one grid step apart" in (
        message
    )
    found = places(message)
    assert len(found) == len(faults)
    for (x, y), fault in faults.items():
        assert any(on(place, (x, y, x, y)) for place in found)
        assert f" um {fault}" in message
