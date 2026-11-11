"""A GDS boundary reconstruction keeps its drawn segments where a lumped-port sheet crosses its ring (Issue #163).

A reconstruction record names segments of the drawn ring by index.  The source grid stage nodes the
ring where the port sheet's boundary crosses it, inserting each crossing as a vertex, so the
record's segments are carried onto the chains of noded segments they became.  Each GeometryPlan
layout goes through ``EigenmodeSim``'s preparation and the SGB build that ``simulation_drc`` runs,
and the arc a build carries is read from the sidecar SGB writes; a record declared in a stack file
goes through the GDS stack adapter and SGB.  No mesher or solver runs.
"""

from __future__ import annotations

import math

import gdstk
import pytest
from sgb_inputs import gds_pad_in_ground
from test_palace_rectangular_ports import built_sidecar, eigenmode_sim, prepared_port
from test_simulation_drc import MATERIALS, NO_VACUUM_STACK

from scgsim.geometry import GeometryPlan
from scgsim.geometry.models.junctions import is_junction_end_piece
from scgsim.geometry.sgb.model import build_conformal_model

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
    """The square [-100, 100]^2 um pad with its top-right corner rounded, and an electrode above it.

    With ``port`` a 2 um sheet runs from 1 um inside the pad's straight top edge to 1 um inside
    the electrode, whose straight edge lies at y = 120 um.
    """
    return pad_layout(
        monkeypatch,
        [(-100, -100), (100, -100), *CORNER, (-100, 100)],
        electrode=[(-30, 120), (30, 120), (30, 150), (-30, 150)],
        sheet=[(-1, 99), (1, 99), (1, 121), (-1, 121)] if port else None,
        port_orientation=90,
        reconstruction=reconstruction,
    )


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


def test_an_open_reconstruction_after_a_port_crossing_keeps_its_drawn_segments(
    monkeypatch,
):
    """The rounded corner keeps its radius when the port's crossings come before it along the ring.

    The record's indices are read from the ring the plan emits for the pad alone.  With the port,
    the two crossings on the pad's top edge are noded in ahead of the corner's chain; the chain
    and its anchors are still the corner's, so SGB builds a 20 um arc there.
    """
    ring = emitted_pad_ring(cornered_pad(monkeypatch, port=False))
    keys = [(round(x / GRID_UM), round(y / GRID_UM)) for x, y in ring]
    ends = sorted(
        keys.index((round(x / GRID_UM), round(y / GRID_UM)))
        for x, y in (CORNER[0], CORNER[-1])
    )
    # The corner is one run of 16 drawn segments along the ring, travelled either way.
    assert ends[1] - ends[0] == CORNER_CHORDS
    first = ends[0]
    reconstruction = {
        "boundary_id": "PAD_CORNER",
        "entity_id": "pad",
        "selector_point_um": (0.0, 0.0),
        "role": "outer",
        "kind": "circular_arc",
        "closed": False,
        "segment_indices": list(range(first, first + CORNER_CHORDS)),
        "anchor_indices": [first, first + CORNER_CHORDS // 2, first + CORNER_CHORDS],
    }
    component = cornered_pad(monkeypatch, port=True, reconstruction=reconstruction)
    assert noded_pad_vertices(eigenmode_sim(component, NETS)) == len(ring) + 2
    arcs, _ = pad_arcs(built_sidecar(monkeypatch, component, NETS))
    (arc,) = arcs
    assert arc["facts"]["radius_um"] == pytest.approx(CORNER_RADIUS_UM, abs=GRID_UM)


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
