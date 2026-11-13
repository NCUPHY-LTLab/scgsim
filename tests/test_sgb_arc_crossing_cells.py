"""The broad phase of the sampled arc-crossing check: no crossing is lost, and a long segment costs its length.

``sampled_crossings`` tests only the segment pairs that share one of its 256-unit cells and involve an arc;
two straight edges are left to ``check_planar``, which proves in exact arithmetic that chords never cross.
These tests compare its pairs with an all-pairs exact reference over seeded random edge sets, lock down that
a long oblique segment is listed in O(length) cells, not in every cell of its bounding box, and that each
kind of crossing is still refused by the check that owns it.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from scgsim.geometry.sgb.arrangement.arcs import (  # private-import: the broad phase and its exact test are the contract under test
    RealisedArc,
    _edge_polyline,
    _segment_cells,
    _segments_cross,
    sampled_crossings,
)
from scgsim.geometry.sgb.arrangement.dcel import Arrangement
from scgsim.geometry.sgb.arrangement.derivation.layer import Layer, RawMeasurements
from scgsim.geometry.sgb.arrangement.labelling.dissolve import Dissolved
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import NodedEdge
from scgsim.geometry.sgb.checks.validation.arrangements import check_arcs, check_planar
from scgsim.geometry.sgb.checks.validation.report import ValidationReport
from scgsim.geometry.sgb.exact.stopwatch import Stopwatch

CELL = 256  # the cell size sampled_crossings uses, in grid units
STEP = 8.0  # its default arc sampling step


def edges_only(segments: list[tuple[tuple[int, int], tuple[int, int]]]) -> Arrangement:
    """An arrangement holding just these edges: the crossing check reads nothing else.

    The edges may cross, which a built arrangement never does; that is the failure the check looks for.
    """
    edges = [NodedEdge(a, b, frozenset()) for a, b in segments]
    return Arrangement([], edges, [], [], [], [], [])


def arc_through(centre: tuple[int, int], radius: int, degrees0: float, sweep: float):
    """End points on the lattice and the realised arc of ``radius`` about ``centre`` between them."""
    a0, a1 = math.radians(degrees0), math.radians(degrees0 + sweep)
    a = (
        round(centre[0] + radius * math.cos(a0)),
        round(centre[1] + radius * math.sin(a0)),
    )
    b = (
        round(centre[0] + radius * math.cos(a1)),
        round(centre[1] + radius * math.sin(a1)),
    )
    ccw = sweep > 0
    curve = RealisedArc(
        "arc",
        "arc",
        (float(centre[0]), float(centre[1])),
        float(radius),
        ccw,
        not ccw,
        0.0,
        0.0,
    )
    return (a, b), curve


def random_layer(rng: random.Random):
    """A seeded edge set mixing the cases the cells must not lose.

    Lattice segments on multiples of 128 cross on cell lines and at cell corners; long segments span
    many cells in both directions; axis-aligned segments lie on cell lines; arcs are sampled chains.
    """
    segments, curves = [], []
    for _ in range(rng.randint(2, 18)):
        kind = rng.random()
        if kind < 0.3:
            a = (128 * rng.randint(-6, 6), 128 * rng.randint(-6, 6))
            b = (128 * rng.randint(-6, 6), 128 * rng.randint(-6, 6))
            curve = None
        elif kind < 0.55:
            span = rng.choice((600, 3000))
            a = (rng.randint(-span, span), rng.randint(-span, span))
            b = (rng.randint(-span, span), rng.randint(-span, span))
            curve = None
        elif kind < 0.7:
            line, lo, hi = (
                256 * rng.randint(-3, 3),
                rng.randint(-900, 0),
                rng.randint(1, 900),
            )
            a, b = (
                ((line, lo), (line, hi))
                if rng.random() < 0.5
                else ((lo, line), (hi, line))
            )
            curve = None
        else:
            centre = (rng.randint(-800, 800), rng.randint(-800, 800))
            sweep = rng.choice((-1, 1)) * rng.uniform(20, 170)
            (a, b), curve = arc_through(
                centre, rng.randint(40, 700), rng.uniform(0, 360), sweep
            )
        if a != b:
            segments.append((a, b))
            curves.append(curve)
    return edges_only(segments), curves


def all_pairs_crossings(
    arr: Arrangement, curves: list[RealisedArc | None]
) -> list[tuple[int, int]]:
    """The reference: every segment of every edge against every segment of every later edge, one of them an arc."""
    polylines = [_edge_polyline(arr, curves, e, STEP) for e in range(len(arr.edges))]
    found = set()
    for i, first in enumerate(polylines):
        for j in range(i + 1, len(polylines)):
            if curves[i] is None and curves[j] is None:
                continue  # two chords: check_planar's exact test, not this one
            second = polylines[j]
            if any(
                _segments_cross(p, q, r, s)
                for p, q in zip(first, first[1:], strict=False)
                for r, s in zip(second, second[1:], strict=False)
            ):
                found.add((i, j))
    return sorted(found)


def test_the_cells_lose_no_crossing_of_the_all_pairs_reference():
    """Seeded random edge sets, crossings on cell lines and corners included: the same pairs as all pairs."""
    rng = random.Random(115)
    crossing_pairs = 0
    for _ in range(400):
        arr, curves = random_layer(rng)
        expected = all_pairs_crossings(arr, curves)
        assert sampled_crossings(arr, curves) == expected
        crossing_pairs += len(expected)
    # the sets do cross an arc, so the comparison is not vacuous
    assert crossing_pairs > 900


@pytest.mark.parametrize(
    "first, second",
    [
        (
            ((0, 0), (512, 512)),
            ((0, 512), (512, 0)),
        ),  # cross at the cell corner (256, 256)
        (
            ((256, -300), (256, 300)),
            ((-300, 0), (700, 0)),
        ),  # both on cell lines, cross at (256, 0)
        (
            ((0, 0), (1024, 256)),
            ((512, -100), (512, 900)),
        ),  # cross at (512, 128) on a column line
    ],
)
def test_a_crossing_on_a_cell_line_or_corner_is_found(first, second):
    """A crossing point on the boundary between cells is in a cell both segments are listed in."""
    cells = _segment_cells([list(first), list(second)], float(CELL))
    assert any({(0, 0), (1, 0)} <= set(members) for members in cells.values())


@pytest.mark.parametrize("start", [(0, 0), (100, 37)])
def test_a_long_45_degree_segment_is_listed_in_cells_along_its_path(start):
    """A 0.58 mm diagonal at 1 nm grid: O(length) cells, not the 1601 x 1601 cells of its bounding box."""
    end = (start[0] + 409_839, start[1] + 409_839)
    cells = _segment_cells([[start, end]], float(CELL))
    columns = end[0] // CELL - start[0] // CELL + 1
    rows = end[1] // CELL - start[1] // CELL + 1
    # each column holds the rows its clipped piece spans, plus at most one more on each side for the pad
    assert len(cells) <= rows + 3 * columns
    assert len(cells) < columns * rows / 300


def layer_of(
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
    curves: list[RealisedArc | None],
) -> Layer:
    """A level layer holding just these edges, for the arrangement checks."""
    dissolved = Dissolved(edges_only(segments), [], [], [], {})
    return Layer(
        "level",
        0,
        "level[0]",
        Fraction(0),
        dissolved,
        None,
        RawMeasurements(0, 0),
        edge_ids=[f"e{k}" for k in range(len(segments))],
        edge_curve=curves,
    )


def outcome(report: ValidationReport, name: str) -> bool:
    """Whether the named check passed."""
    return next(c.passed for c in report.checks if c.name == name)


def test_a_realised_arc_crossing_a_straight_edge_is_refused():
    """The arc bulges across a chord between its end points' neighbours: the arc check refuses it."""
    (a, b), curve = arc_through((0, 0), 500, 0.0, 90.0)
    report = ValidationReport()
    check_arcs(report, [layer_of([(a, b), ((0, 0), (700, 700))], [curve, None])])
    assert not outcome(report, "arc_edges_do_not_cross")
    assert "e0 x e1" in next(
        c.detail for c in report.checks if c.name == "arc_edges_do_not_cross"
    )


def test_two_crossing_straight_edges_are_refused_by_the_planarity_check():
    """Two chords that cross are not the arc check's to find; the exact planarity check refuses them."""
    layer = layer_of([((0, 0), (512, 512)), ((0, 512), (512, 0))], [None, None])
    report = ValidationReport()
    check_arcs(report, [layer])
    check_planar(report, [layer], Stopwatch())
    assert outcome(report, "arc_edges_do_not_cross")
    assert not outcome(report, "arrangements_planar")
