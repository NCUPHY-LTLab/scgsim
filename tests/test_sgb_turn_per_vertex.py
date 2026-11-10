"""The turn-per-vertex rule for curved outlines drawn as polygons: at most 128 points per full circle.

A curved outline is supported when every vertex turns by at least 360 / 128 = 2.8125 degrees.  Regular n-gon
disks show the rule: up to n = 128 they build with the bands of their exact erosion at every radius, also off
the lattice and rotated.  At n = 256 (1.41 degrees per vertex) they were refused at every radius, because
neighbouring offsets at such small turns could miss each other on the lattice; since the offsets of two lines
end at their crossing on both sides of a vertex (Issue #138), they build exact bands too.  Whether the rule
stays a layout rule is open on Issue #138; between 128 and 256 points nothing further is tested.

The rule says nothing about a single vertex.  The counterexamples below record what SGB does with one vertex
that turns less, on either side of 2.8125 degrees, in an otherwise coarse outline, and with a vertex that does
not turn at all.  Measured at margins 0.5 and 1.0 um only.
"""

from __future__ import annotations

import math

import pytest
from sgb_supported_kit import (
    Conductor,
    Scene,
    Um,
    assert_exact_bands,
    band_areas,
    catalogue,
    ngon,
    to_lattice,
    turns_degrees,
    type_bands,
)

from scgsim.geometry.sgb.model import build_model_from_catalogue

RULE_DEGREES = 360 / 128
RADII = (5.0, 10.0, 50.0, 150.0)


def disk(points: list[Um], centre: Um = (0.0, 0.0), half: float = 0.0) -> Scene:
    """One conductor on a die 20 um wider than it on every side."""
    reach = half or max(math.dist(p, centre) for p in points) + 20
    die = (centre[0] - reach, centre[1] - reach, centre[0] + reach, centre[1] + reach)
    return Scene("disk", (), die, [Conductor("disk", "disk", to_lattice(points))])


@pytest.mark.parametrize("r", RADII)
@pytest.mark.parametrize("n", [16, 24, 32, 64, 128])
def test_a_polygon_circle_of_at_most_128_points_builds_exact_bands(n, r):
    """22.5 down to 2.8125 degrees per vertex, radius 5 to 150 um."""
    assert_exact_bands(disk(ngon(r, n)))


@pytest.mark.parametrize("r", [5.0, 10.0, 37.3, 75.0, 150.0])
@pytest.mark.parametrize(
    ("centre", "phase"),
    [((0.3337, 0.1234), 0.0), ((12.0004, -7.7771), 1.7), ((0.0, 0.0), 0.73)],
    ids=["off-lattice", "off-lattice-rotated", "rotated"],
)
def test_a_128_point_circle_off_the_lattice_or_rotated_builds_exact_bands(
    centre, phase, r
):
    """The vertices round to the lattice in every direction, and the rule still holds."""
    assert_exact_bands(disk(ngon(r, 128, centre, phase), centre))


@pytest.mark.parametrize("r", RADII)
def test_a_256_point_circle_builds_exact_bands_at_every_radius(r):
    """1.41 degrees per vertex, outside the rule: the offsets meet at every vertex, so the bands are exact."""
    assert_exact_bands(disk(ngon(r, 256)))


def with_bumped_edge(r: float, turn: float) -> list[Um]:
    """A 16-gon (22.5 degrees per vertex) with one vertex added on an edge, turning ``turn`` degrees.

    The new vertex sits on the perpendicular bisector of the edge, outside it for a convex turn and inside it
    for a reflex (negative) one; its two neighbours then turn ``22.5 - turn / 2`` degrees.
    """
    points = ngon(r, 16)
    (ax, ay), (bx, by) = points[0], points[1]
    mx, my = (ax + bx) / 2, (ay + by) / 2
    rise = math.dist(points[0], points[1]) / 2 * math.tan(math.radians(turn) / 2)
    scale = rise / math.hypot(mx, my)
    return [points[0], (mx + scale * mx, my + scale * my), *points[1:]]


@pytest.mark.parametrize(
    "turn", [2.9, 2.7, 1.41, 0.3, 0.01, -2.7, -0.01], ids=lambda t: f"{t:+g}deg"
)
def test_one_vertex_turning_less_than_the_rule_in_a_coarse_outline_builds_exact_bands(
    turn,
):
    """Recorded behaviour: a single small turn between 9.8 um edges builds, convex or reflex, down to 0.01
    degrees.  Outside the rule this is observed, not guaranteed."""
    points = with_bumped_edge(50.0, turn)
    lattice_turn = turns_degrees(to_lattice(points))[1]
    assert lattice_turn == pytest.approx(turn, abs=0.01)
    assert (abs(lattice_turn) >= RULE_DEGREES) == (abs(turn) >= RULE_DEGREES)
    assert_exact_bands(disk(points))


def with_collinear_vertex(ring: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The ring with one lattice vertex added on the first edge that holds one, nearest that edge's middle."""
    for i, a in enumerate(ring):
        b = ring[(i + 1) % len(ring)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        steps = math.gcd(dx, dy)
        if steps > 1:
            half = steps // 2
            extra = (a[0] + dx // steps * half, a[1] + dy // steps * half)
            return [*ring[: i + 1], extra, *ring[i + 1 :]]
    raise AssertionError("no edge of the ring holds a lattice point between its ends")


@pytest.mark.parametrize("outline", ["square", "16-gon"])
def test_a_vertex_that_does_not_turn_builds_exact_bands(outline):
    """Recorded behaviour: a collinear lattice vertex (0 degrees) builds exact bands.

    On the square's axis-aligned edge every band keeps its area exactly.  On the 16-gon's oblique edge the
    offsets of the two halves end at a rounded lattice point at the new vertex, so the areas move by a few
    1e-4 um^2, within the snap bound.
    """
    plain = disk(ngon(10.0, 4, phase=45.0) if outline == "square" else ngon(10.0, 16))
    ring = with_collinear_vertex(plain.conductors[0].outer)
    split = Scene("split", (), plain.die, [Conductor("disk", "disk", ring)])
    assert sorted(turns_degrees(ring), key=abs)[0] == 0.0
    assert_exact_bands(split)
    if outline == "square":
        before, after = (
            band_areas(
                build_model_from_catalogue(catalogue(s), "B", insets=type_bands())
            )
            for s in (plain, split)
        )
        assert after == before
