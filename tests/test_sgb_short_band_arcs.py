"""A reflex vertex whose band arc would be a few grid steps long gets the intersection of its two offsets.

The pad's top edge dips to a vertex at ``(0, 10 um - d)`` that turns the outline by ``turn`` degrees, a reflex
vertex of the pad's MA band support.  Its arc of radius ``m`` would be ``m * turn`` long.  Below
``SHORT_ARC_GRID_STEPS`` (25) the two offset lines meet at their intersection instead, which lies
``m (1 / cos(turn / 2) - 1)`` beyond the arc: less than one grid step, or the arc stays (Issue #125).
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from sgb_scenes import GRID, chip_prisms, conductor, units

from scgsim.geometry.sgb.cutfamilies.inset import SHORT_ARC_GRID_STEPS
from scgsim.geometry.sgb.cutfamilies.requests import ConductorSurface, InsetRequest
from scgsim.geometry.sgb.model import build_model_from_catalogue
from scgsim.geometry.sgb.records.prisms import PrismCatalogue

HALF_WIDTH_UM = 20.0


def dipped_pad(turn_degrees: float) -> tuple[PrismCatalogue, tuple[int, int]]:
    """A 40 x 20 um pad whose top edge dips to a reflex vertex turning ``turn_degrees``; and that vertex."""
    dip = HALF_WIDTH_UM * math.tan(math.radians(turn_degrees / 2))
    outline = [
        (-20.0, -10.0),
        (20.0, -10.0),
        (20.0, 10.0),
        (0.0, 10.0 - dip),
        (-20.0, 10.0),
    ]
    pad = conductor("pad", outline, 0, "0.2", "pad")
    vertex = (0, units(10.0 - dip))
    return PrismCatalogue(GRID, (*chip_prisms(GRID, 40.0), pad), name="dipped"), vertex


def pad_top_family(catalogue: PrismCatalogue, margins_um: tuple[float, ...]):
    """The MA band family of the pad's top face, built under Route B."""
    request = InsetRequest("band", ConductorSurface("pad", "top"), margins_um)
    model = build_model_from_catalogue(catalogue, "B", insets=(request,))
    family = next(f for f in model.labelled.catalogue.families if f.family_id == "band")
    return model, family


def arcs_about(family, vertex):
    """The arc members of a family centred at a lattice point."""
    centre = (Fraction(vertex[0]), Fraction(vertex[1]))
    return [a for a in family.arcs if a.centre == centre]


def lattice_turn(vertex: tuple[int, int]) -> float:
    """The turn of the drawn outline at the vertex, in radians, after the outline is put on the lattice."""
    a = math.atan2(vertex[1] - units(10.0), -units(HALF_WIDTH_UM))
    return 2 * abs(math.pi - abs(a))


@pytest.mark.parametrize("turn_degrees", [0.15, 1.0])
def test_a_short_reflex_arc_becomes_the_intersection_of_its_offsets(turn_degrees):
    """At 0.5 and 1.0 um the arc would be under 25 grid steps: no arc, and the offsets meet within one step."""
    catalogue, vertex = dipped_pad(turn_degrees)
    model, family = pad_top_family(catalogue, (0.5, 1.0))
    assert arcs_about(family, vertex) == []
    turn = lattice_turn(vertex)
    points = {(v.x, v.y) for v in model.topology.vertices.values()}
    for margin in (500, 1000):
        assert margin * turn < SHORT_ARC_GRID_STEPS
        # The exact join of the two offsets is straight below the vertex at margin / cos(turn / 2).
        exact = vertex[1] - margin / math.cos(turn / 2)
        joined = [p for p in points if p[0] == 0 and abs(p[1] - exact) <= 0.5]
        assert joined, f"no vertex at the offsets' intersection for m = {margin}"
        assert margin / math.cos(turn / 2) - margin < 1


@pytest.mark.parametrize(
    ("turn_degrees", "kept"), [(2.0, {1000}), (5.0, {500, 1000})], ids=["2deg", "5deg"]
)
def test_a_reflex_arc_of_25_grid_steps_or_more_stays(turn_degrees, kept):
    """At 2 degrees only the 1.0 um arc (35 steps) stays; at 5 degrees both (44 and 87 steps), radius m."""
    catalogue, vertex = dipped_pad(turn_degrees)
    _, family = pad_top_family(catalogue, (0.5, 1.0))
    turn = lattice_turn(vertex)
    assert {m for m in (500, 1000) if m * turn >= SHORT_ARC_GRID_STEPS} == kept
    assert {a.radius for a in arcs_about(family, vertex)} == {Fraction(m) for m in kept}


def test_a_short_arc_whose_intersection_lies_a_grid_step_beyond_it_stays():
    """At m = 0.05 um a 25.8 degree turn gives a 22.5-step arc, but the intersection lies 1.3 steps beyond it."""
    catalogue, vertex = dipped_pad(25.8)
    turn = lattice_turn(vertex)
    assert 50 * turn < SHORT_ARC_GRID_STEPS
    assert 50 / math.cos(turn / 2) - 50 >= 1
    _, family = pad_top_family(catalogue, (0.05,))
    assert {a.radius for a in arcs_about(family, vertex)} == {Fraction(50)}
