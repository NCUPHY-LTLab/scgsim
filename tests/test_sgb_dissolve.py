"""Labelling, required edges and dissolution (design brief section 4)."""

from __future__ import annotations

import pytest
from sgb_scenes import GRID

from scgsim.geometry.sgb.arrangement.dcel import build_arrangement, interior_point
from scgsim.geometry.sgb.errors import OverlapError
from scgsim.geometry.sgb.labelling.dissolve import FamilyPolicies, dissolve
from scgsim.geometry.sgb.labelling.prism_set import PrismSet, resolve_domain
from scgsim.geometry.sgb.noding.snap_rounding import Segment, snap_round
from scgsim.geometry.sgb.records.families import CurveFamilyRecord
from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import Footprint, PrismRecord


def rect(x0, y0, x1, y1):
    """Rect."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def prism(  # noqa: PLR0913 - a test builder
    name, points, net="A", kind="conductor", material="Al", priority=0, polygon=0
):
    """A prism of an entity; two polygons of one entity share its name and differ in ``polygon``."""
    return PrismRecord(
        name,
        Footprint.from_rings(points),
        exact_z(0),
        exact_z(1),
        material,
        kind,
        net_id=net if kind == "conductor" else None,
        priority=priority,
        polygon=polygon,
    )


def label_cells(arr, prisms):
    """Domain label of every face of an arrangement (index 0: the unbounded face)."""
    cells = PrismSet(prisms, range(len(prisms)), GRID)
    labels = [None]
    for face in range(1, len(arr.faces)):
        point = interior_point(arr, face)
        labels.append(
            resolve_domain([prisms[k] for k in cells.hit(point)], point, GRID)
        )
    return labels


def arrange(prisms, extra=()):
    """Noded arrangement of every prism boundary plus extra segments."""
    segments = [
        Segment(c.p, c.q, frozenset([c.id])) for p in prisms for c in p.carriers()
    ]
    return build_arrangement(snap_round([*segments, *extra]).edges)


def label(prisms):
    """Label."""
    arr = arrange(prisms)
    return arr, label_cells(arr, prisms)


def test_same_net_overlap_dissolves_into_one_cell_with_union_outline():
    """Same net overlap dissolves into one cell with union outline."""
    prisms = [
        prism("p", rect(0, 0, 100, 100)),
        prism("p", rect(50, 50, 150, 150), polygon=1),
    ]
    arr, labels = label(prisms)
    assert len(arr.faces) - 1 == 3  # two private L-shaped parts and the overlap
    dissolved = dissolve(arr, labels)
    assert len(dissolved.arrangement.faces) - 1 == 1
    assert dissolved.arrangement.face_twice_area(1) == 2 * (
        100 * 100 + 100 * 100 - 50 * 50
    )
    assert (
        len(dissolved.arrangement.edges) == 8
    )  # union outline of two overlapping squares


def test_entities_of_one_net_keep_their_seam_and_the_higher_priority_owns_the_overlap():
    """Decision D5: surfaces are attributed per Entity, so the seam between two entities stays."""
    prisms = [
        prism("p", rect(0, 0, 100, 100), priority=1),
        prism("q", rect(50, 50, 150, 150), priority=2),
    ]
    arr, labels = label(prisms)
    dissolved = dissolve(arr, labels)
    area_of = {
        dissolved.labels[face].owner: dissolved.arrangement.face_twice_area(face)
        for face in range(1, len(dissolved.arrangement.faces))
    }
    assert area_of == {
        "p": 2 * (100 * 100 - 50 * 50),
        "q": 2 * 100 * 100,
    }  # q owns the 50 x 50 overlap


def test_equal_priority_overlap_of_two_entities_of_one_net_fails_naming_both():
    """No silent winner: equal priority means the source did not say who owns the overlap."""
    prisms = [prism("p", rect(0, 0, 100, 100)), prism("q", rect(50, 50, 150, 150))]
    with pytest.raises(
        OverlapError,
        match=r"same-Net entities p and q overlap near \(.*\) um with equal priority 0: declare distinct entity priorities",
    ):
        label(prisms)


def test_different_net_overlap_raises_and_names_both_entities_and_the_location():
    """Different net overlap raises and names both entities and the location."""
    with pytest.raises(OverlapError, match=r"p, q|q, p") as error:
        label(
            [
                prism("p", rect(0, 0, 100, 100), net="A"),
                prism("q", rect(50, 50, 150, 150), net="B"),
            ]
        )
    assert "um" in str(error.value)


def test_dielectric_priority_resolves_overlap_and_equal_priority_raises():
    """Dielectric priority resolves overlap and equal priority raises."""
    air = prism("air", rect(0, 0, 100, 100), kind="vacuum", material="vacuum")
    substrate = prism(
        "sub", rect(0, 0, 100, 100), kind="dielectric", material="substrate", priority=0
    )
    # a dielectric wins over vacuum even at equal priority
    _, labels = label([air, substrate])
    assert labels[1].material == "substrate" and labels[1].owner == "sub"
    other = prism(
        "other", rect(0, 0, 100, 100), kind="dielectric", material="oxide", priority=0
    )
    with pytest.raises(OverlapError, match="equal-priority"):
        label([substrate, other])
    higher = prism(
        "other", rect(0, 0, 100, 100), kind="dielectric", material="oxide", priority=2
    )
    assert label([substrate, higher])[1][1].material == "oxide"


@pytest.mark.parametrize(("order_a", "order_b", "winner"), ((1, 2, "a"), (2, 1, "b")))
def test_the_lower_mesh_order_dielectric_owns_the_overlap_and_beats_a_ranked_vacuum(
    order_a, order_b, winner
):
    """Priority is the negated PDK ``mesh_order``: the lower order wins, and vacuum loses even when ranked first."""
    # The vacuum carries the lowest mesh_order (highest priority); material kind decides first.
    air = prism(
        "air", rect(0, 0, 100, 100), kind="vacuum", material="vacuum", priority=0
    )
    a = prism(
        "a",
        rect(0, 0, 100, 100),
        kind="dielectric",
        material="silicon",
        priority=-order_a,
    )
    b = prism(
        "b",
        rect(0, 0, 100, 100),
        kind="dielectric",
        material="sapphire",
        priority=-order_b,
    )
    _, labels = label([air, a, b])
    assert labels[1].owner == winner


def test_required_family_edge_survives_dissolution_but_plain_edges_do_not():
    """Required family edge survives dissolution but plain edges do not."""
    prisms = [
        prism("p", rect(0, 0, 100, 100)),
        prism("p", rect(0, 0, 100, 100), polygon=1),
    ]
    arr = arrange(prisms, [Segment((50, 0), (50, 100), frozenset(["fam"]))])
    labels = label_cells(arr, prisms)
    assert len(arr.faces) - 1 == 2
    kept = dissolve(arr, labels, FamilyPolicies(required=frozenset(["fam"])))
    assert len(kept.arrangement.faces) - 1 == 2
    gone = dissolve(arr, labels)
    assert len(gone.arrangement.faces) - 1 == 1
    assert (
        len(gone.arrangement.edges) == 4
    )  # degree-2 vertices of the seam were removed


def test_refinement_only_edges_are_kept_aside_without_splitting_the_cell():
    """Refinement only edges are kept aside without splitting the cell."""
    prisms = [prism("p", rect(0, 0, 100, 100))]
    arr = arrange(prisms, [Segment((50, 10), (50, 90), frozenset(["ref"]))])
    labels = label_cells(arr, prisms)
    dissolved = dissolve(
        arr, labels, FamilyPolicies(refinement_only=frozenset(["ref"]))
    )
    assert len(dissolved.arrangement.faces) - 1 == 1
    assert len(dissolved.refinement) == 1 and dissolved.refinement[0][1] == 1


def test_degree_two_vertex_with_different_carriers_is_kept():
    """Degree two vertex with different carriers is kept."""
    prisms = [
        prism("p", rect(0, 0, 100, 100)),
        prism("p", rect(100, 0, 200, 100), polygon=1),
    ]
    arr, labels = label(prisms)
    dissolved = dissolve(arr, labels)
    # the shared wall dissolves; the outline of the union keeps the vertices where carriers change
    assert len(dissolved.arrangement.faces) - 1 == 1
    assert dissolved.arrangement.face_twice_area(1) == 2 * 200 * 100
    assert len(dissolved.arrangement.edges) == 6


def test_an_unknown_edge_policy_is_rejected():
    """An unknown edge policy is rejected."""
    with pytest.raises(ValueError, match="unknown edge policy"):
        CurveFamilyRecord("f", "seam", (), (exact_z(0),), "dissolve")  # type: ignore[arg-type]
