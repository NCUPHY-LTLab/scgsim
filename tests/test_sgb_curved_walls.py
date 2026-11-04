"""Curved walls are split at every vertex on their arc, so each piece is one exact cylinder (decision after review R3)."""

from __future__ import annotations

import math

import pytest
from sgb_extreme_cases import FIXTURES
from sgb_scenes import pad_in_ground

from scgsim.geometry.sgb.model import build_model_from_catalogue

DISK_VOLUME_UM3_GRID2 = math.pi * 5000**2 * 0.2  # grid^2 * um
SNAP_VOLUME_TOLERANCE = 5e-5  # lattice cut points move arcs by < 1 g; the relative volume error stays far below this


def build(name: str, route: str):
    """Build a fixture of the extreme-case catalogue."""
    return build_model_from_catalogue(FIXTURES[name].build(route), route)


def cylinders(model):
    """The curved wall faces of a model."""
    return [f for f in model.topology.faces.values() if f.curved]


def arcs_of(model, face):
    """The bottom and top arc edges of a cylinder wall."""
    return tuple(model.topology.edges[e] for e in face.cylinder)


@pytest.mark.parametrize(
    "name", ["arc_with_foreign_vertex", "arc_with_foreign_vertex_from_post"]
)
def test_a_foreign_vertex_on_an_arc_wall_gives_two_cylinders_not_a_five_edge_wall(name):
    """Four quarter arcs, one of them split at the vertex: five cylinder walls, none with two top arcs."""
    model = build(name, "B")
    walls = cylinders(model)
    assert len(walls) == 5
    for wall in walls:
        bottom, top = arcs_of(model, wall)
        assert bottom.curve is not None and top.curve is not None
        # one bottom arc and one top arc through the same end points on one circle
        assert (bottom.curve.centre, bottom.curve.radius, bottom.curve.ccw) == (
            top.curve.centre,
            top.curve.radius,
            top.curve.ccw,
        )
        assert {bottom.v0, bottom.v1} != {top.v0, top.v1} or bottom.plane != top.plane
    assert "curved_walls_are_single_cylinders" in {c.name for c in model.report.checks}
    assert not model.report.failures


def test_the_split_pieces_share_the_vertical_edge_through_the_foreign_vertex():
    """The two walls on either side of the vertex use the same vertical edge on that column."""
    model = build("arc_with_foreign_vertex", "B")
    topology = model.topology
    column = {
        e.id
        for e in topology.edges.values()
        if e.kind == "vertical"
        and (int(topology.vertices[e.v0].x), int(topology.vertices[e.v0].y))
        == (4330, 2500)
    }
    assert column
    users = [f for f in cylinders(model) if any(eid in column for eid, _ in f.loops[0])]
    assert len(users) == 2  # the two pieces of the split quarter arc
    assert set.intersection(*({eid for eid, _ in f.loops[0]} & column for f in users))


def test_the_horizontal_faces_above_and_below_carry_the_split_so_edges_stay_shared_whole():
    """The bottom (MS) and top (MA) faces of the disk both split the quarter arc at the foreign vertex."""
    model = build("arc_with_foreign_vertex", "B")
    topology = model.topology

    def arc_edges(group):
        """Arc edges."""
        (face,) = (
            topology.faces[i]
            for i in model.routed.groups[group].ids
            if topology.faces[i].kind == "horizontal"
        )
        return [e for e, _ in face.loops[0] if topology.edges[e].curve is not None]

    assert len(arc_edges("MA::disk::top")) == len(arc_edges("MS::disk::bottom")) == 5
    assert model.report.failures == []


def test_the_disk_volume_still_matches_the_closed_form_after_splitting():
    """Chord polyhedron plus the segment-area bulge of every cylinder piece."""
    for name in (
        "arc_with_foreign_vertex",
        "arc_with_foreign_vertex_from_post",
        "port_crossing_round_pad",
    ):
        model = build(name, "B")
        disk = next(
            v for v in model.topology.volumes.values() if v.domain.name == "disk"
        )
        volume = float(sum(s.signed_volume for s in disk.shells))
        assert volume == pytest.approx(
            DISK_VOLUME_UM3_GRID2, rel=SNAP_VOLUME_TOLERANCE
        ), name


def test_a_port_crossing_a_round_pad_twice_splits_every_stacked_run_consistently():
    """Two foreign vertices at the port plane cascade through the wall runs below and above it."""
    model = build("port_crossing_round_pad", "B")
    walls = cylinders(model)
    assert len(walls) == 9
    z_ranges = {(float(f.z_range[0]), float(f.z_range[1])) for f in walls}
    assert z_ranges == {
        (0.0, 0.1),
        (0.1, 0.2),
        (0.0, 0.2),
    }  # the port plane cuts the wall it crosses
    assert not model.report.failures


def test_route_c_splits_curved_walls_the_same_way():
    """The wall split does not depend on the route."""
    b = build("arc_with_foreign_vertex", "B")
    c = build("arc_with_foreign_vertex", "C")
    assert len(cylinders(b)) == len(cylinders(c)) == 5
    assert {f.id for f in cylinders(b)} == {f.id for f in cylinders(c)}


def test_a_circle_approximated_by_a_polygon_never_produces_cylinder_walls():
    """Polygon-only circles have planar walls; only declared arcs are cylinders."""
    model = build_model_from_catalogue(
        pad_in_ground("B", hole_sides=32, rounded_corner=False), "B"
    )
    assert cylinders(model) == []


def test_straight_walls_keep_more_than_four_boundary_edges_unsplit():
    """A planar wall may have any number of boundary edges: a straight wall is not split at foreign vertices."""
    model = build("port_edge_on_metal_sidewall", "B")
    topology = model.topology
    walls = [f for f in topology.faces.values() if f.kind == "wall"]
    assert any(len(f.loops[0]) > 4 for f in walls)
    assert all(f.planar for f in walls)
