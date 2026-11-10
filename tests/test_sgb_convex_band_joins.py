"""At a convex vertex between two lines, both band offsets end at their crossing, one shared lattice point.

The pad's top edge rises to a vertex at ``(0, 10 um + d)`` that turns the outline by ``turn`` degrees, a convex
vertex of the pad's MA and MS band supports.  The offsets of the two edges by ``m`` cross ``m tan(turn / 2)``
before their own ends, on the bisector ``m / cos(turn / 2)`` below the vertex; past the crossing each runs
nearer than ``m`` to the other edge, so the band boundary turns there.  Below about 2.6 degrees at 0.5 um the
two overlap pieces lie under half a grid step apart, and snapping used to fuse them into a seam piece of a few
grid steps (Issue #138).  Ending both offsets at the crossing removes it and adds no point to the curve.

Counterexample: a corner cut by a chamfer.  Where the chamfer is too short to bound the eroded corner, its
offset would lose more than its length to the crossings with its neighbours' offsets; it is not ended there,
and R_m keeps the plain corner.  Where it is long enough, R_m has the two corners at those crossings.  The
bands stay exact either way.
"""

from __future__ import annotations

import math

import pytest
from sgb_supported_kit import (
    CONTOUR_SNAP_UM,
    GRID,
    Conductor,
    Scene,
    assert_exact_bands,
    short_band_edges,
    to_lattice,
)

from scgsim.geometry.sgb.cutfamilies.inset import SHORT_ARC_GRID_STEPS

HALF_WIDTH_UM = 20.0
MARGINS_GRID = (500, 1000)
CONTOUR_SNAP_GRID = CONTOUR_SNAP_UM / GRID.size_um


def peaked_pad(turn_degrees: float) -> tuple[Scene, tuple[int, int]]:
    """A 40 x 20 um pad whose top edge rises to a convex vertex turning ``turn_degrees``; and that vertex."""
    rise = HALF_WIDTH_UM * math.tan(math.radians(turn_degrees / 2))
    outline = [
        (-20.0, -10.0),
        (20.0, -10.0),
        (20.0, 10.0),
        (0.0, 10.0 + rise),
        (-20.0, 10.0),
    ]
    ring = to_lattice(outline)
    scene = Scene(
        "peaked", (), (-40.0, -40.0, 40.0, 40.0), [Conductor("pad", "pad", ring)]
    )
    return scene, ring[3]


def lattice_turn(vertex: tuple[int, int]) -> float:
    """The turn of the lattice outline at the peak, in radians."""
    return 2 * math.atan2(vertex[1] - 10_000, 20_000)


@pytest.mark.parametrize("turn_degrees", [0.15, 0.30, 0.86, 1.72])
def test_a_near_straight_convex_vertex_ends_both_offsets_at_their_crossing(
    turn_degrees,
):
    """The turns of the Euler and taper joints of cells 14, 15 and 18: no band edge under 25 grid steps.

    At 1.72 degrees and 1.0 um the margin times the turn is 30 grid steps: the join has no cut-off on it.
    """
    scene, vertex = peaked_pad(turn_degrees)
    model = assert_exact_bands(scene)
    assert short_band_edges(model, SHORT_ARC_GRID_STEPS) == []
    turn = lattice_turn(vertex)
    points = {(v.x, v.y) for v in model.topology.vertices.values()}
    for margin in MARGINS_GRID:
        # The two offsets end at one lattice point, within half a grid step of their exact crossing...
        crossing = vertex[1] - margin / math.cos(turn / 2)
        assert any(p[0] == 0 and abs(p[1] - crossing) <= 0.5 for p in points), margin
        # ...and nothing remains of the overlap pieces: no vertex on the offset line near the junction.
        overlap = margin * math.tan(turn / 2) + 1
        offset_level = round(10_000 - margin)
        assert not [
            p for p in points if p[1] == offset_level and 0 < abs(p[0]) <= overlap
        ], margin


def chamfered_pad(chamfer_um: float) -> Scene:
    """A 40 x 20 um pad whose top right corner is cut by a 45 degree chamfer of legs ``chamfer_um``."""
    outline = [
        (-20.0, -10.0),
        (20.0, -10.0),
        (20.0, 10.0 - chamfer_um),
        (20.0 - chamfer_um, 10.0),
        (-20.0, 10.0),
    ]
    return Scene(
        "chamfered",
        (),
        (-40.0, -40.0, 40.0, 40.0),
        [Conductor("pad", "pad", to_lattice(outline))],
    )


@pytest.mark.parametrize("chamfer_um", [0.2, 0.5])
def test_a_chamfer_offset_is_kept_exactly_where_it_bounds_the_eroded_corner(chamfer_um):
    """A 45 degree chamfer of legs ``c`` cuts the eroded corner only when ``c > (2 - sqrt 2) m``.

    Then its offset runs between its crossings with the right and the top offsets and R_m has two corners there;
    otherwise the chamfer's offset would lose more than its length to the two crossings, it is not ended at
    them, and R_m keeps the plain corner ``(20 um - m, 10 um - m)``.  A 0.2 um chamfer is below the bound at both
    margins; a 0.5 um chamfer above it at 0.5 um and below it at 1.0 um.  The bands stay exact either way.

    A corner where the two offsets cross properly is the pixel of their crossing once their end points are on
    the lattice, as the noding has always placed it: on the side's offset, and within half a grid diagonal of
    the chamfer's exact offset line (the contour snap bound), not necessarily within half a step of the exact
    crossing.
    """
    model = assert_exact_bands(chamfered_pad(chamfer_um))
    points = {(v.x, v.y) for v in model.topology.vertices.values()}
    chamfer = round(chamfer_um * 1000)
    for margin in MARGINS_GRID:
        right, top = 20_000 - margin, 10_000 - margin
        level = (
            30_000 - chamfer - margin * math.sqrt(2)
        )  # x + y on the chamfer's offset
        cut = chamfer > (2 - math.sqrt(2)) * margin
        assert ((right, top) in points) is not cut, margin
        on_right = [p for p in points if p[0] == right and p[1] < top]
        on_top = [p for p in points if p[1] == top and p[0] < right]
        for side in (on_right, on_top):
            found = any(
                abs(p[0] + p[1] - level) / math.sqrt(2) <= CONTOUR_SNAP_GRID
                for p in side
            )
            assert found is cut, margin
