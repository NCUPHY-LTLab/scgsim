"""Labelling, required edges and dissolution (brief section 4)."""

from __future__ import annotations

import pytest

from sgbproto.arrangement import build_arrangement
from sgbproto.dissolve import FamilyPolicies, dissolve
from sgbproto.errors import OverlapError
from sgbproto.labels import PrismSet, label_slab
from sgbproto.records import Footprint, PrismRecord, z_value
from sgbproto.snap_rounding import Segment, snap_round


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def prism(name, pts, net="A", kind="conductor", material="Al", priority=0):
    return PrismRecord(name, Footprint.from_rings(pts), z_value(0), z_value(1), material, kind,
                       net_id=net if kind == "conductor" else None, priority=priority)


def label(prisms):
    segments = [Segment(c.p, c.q, frozenset([c.id])) for p in prisms for c in p.carriers()]
    arr = build_arrangement(snap_round(segments).edges)
    labels, _ = label_slab(arr, PrismSet(prisms, range(len(prisms))))
    return arr, labels


def test_same_net_overlap_dissolves_into_one_cell_with_union_outline():
    prisms = [prism("p", rect(0, 0, 100, 100)), prism("q", rect(50, 50, 150, 150))]
    arr, labels = label(prisms)
    assert len(arr.faces) - 1 == 3  # two private L-shaped parts and the overlap
    d = dissolve(arr, labels)
    assert len(d.arrangement.faces) - 1 == 1
    assert d.arrangement.face_twice_area(1) == 2 * (100 * 100 + 100 * 100 - 50 * 50)
    assert len(d.arrangement.edges) == 8  # union outline of two overlapping squares


def test_different_net_overlap_raises():
    with pytest.raises(OverlapError):
        label([prism("p", rect(0, 0, 100, 100), net="A"), prism("q", rect(50, 50, 150, 150), net="B")])


def test_dielectric_priority_resolves_overlap_and_equal_priority_raises():
    air = prism("air", rect(0, 0, 100, 100), kind="vacuum", material="vacuum")
    sub = prism("sub", rect(0, 0, 100, 100), kind="dielectric", material="substrate", priority=1)
    _, labels = label([air, sub])
    assert labels[1].material == "substrate"
    sub0 = prism("sub", rect(0, 0, 100, 100), kind="dielectric", material="substrate", priority=0)
    with pytest.raises(OverlapError):
        label([air, sub0])


def test_required_family_edge_survives_dissolution_but_plain_edges_do_not():
    prisms = [prism("p", rect(0, 0, 100, 100)), prism("q", rect(0, 0, 100, 100))]
    segments = [Segment(c.p, c.q, frozenset([c.id])) for p in prisms for c in p.carriers()]
    segments.append(Segment((50, 0), (50, 100), frozenset(["fam"])))
    arr = build_arrangement(snap_round(segments).edges)
    labels, _ = label_slab(arr, PrismSet(prisms, range(2)))
    assert len(arr.faces) - 1 == 2
    kept = dissolve(arr, labels, FamilyPolicies(required=frozenset(["fam"])))
    assert len(kept.arrangement.faces) - 1 == 2
    gone = dissolve(arr, labels)
    assert len(gone.arrangement.faces) - 1 == 1
    assert len(gone.arrangement.edges) == 4  # degree-2 vertices of the seam were removed


def test_refinement_only_edges_are_kept_aside_without_splitting_the_cell():
    p = [prism("p", rect(0, 0, 100, 100))]
    segments = [Segment(c.p, c.q, frozenset([c.id])) for c in p[0].carriers()]
    segments.append(Segment((50, 10), (50, 90), frozenset(["ref"])))
    arr = build_arrangement(snap_round(segments).edges)
    labels, _ = label_slab(arr, PrismSet(p, [0]))
    d = dissolve(arr, labels, FamilyPolicies(refinement_only=frozenset(["ref"])))
    assert len(d.arrangement.faces) - 1 == 1
    assert len(d.refinement) == 1 and d.refinement[0][1] == 1


def test_degree_two_vertex_with_different_carriers_is_kept():
    prisms = [prism("p", rect(0, 0, 100, 100)), prism("q", rect(100, 0, 200, 100))]
    arr, labels = label(prisms)
    d = dissolve(arr, labels)
    # the shared wall dissolves; the outline of the union keeps the vertices where carriers change
    assert len(d.arrangement.faces) - 1 == 1
    assert d.arrangement.face_twice_area(1) == 2 * 200 * 100
    assert len(d.arrangement.edges) == 6
