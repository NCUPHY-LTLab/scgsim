"""Supported Geometry cells 13-24: the further unit cells the OrcaPen SC PDK cells use, as SGB-native scenes.

Dimensions are those of the PDK cells drawn with ``curve_sampling="simulation"``.  A CPW is a signal strip and a
ground plane less a mask strip ``w + 2g`` wide on the same centreline; a bend is sampled along its centreline and
both edges are offset from it the way an extruded cross-section is (:func:`offset_path`).  The two Euler
centrelines are given as sampled points because an Euler bend has no closed-form radius.  Every CPW runs on to
the die edge, so the ground plane does not close the gaps.
"""

from __future__ import annotations

import heapq
import math

from sgb_supported_kit import (
    Conductor,
    Expectation,
    Port,
    Scene,
    Um,
    conductor_shape,
    die_footprint,
    lattice,
    mirror_y,
    ngon,
    offset_path,
    planar_expectations,
    polar,
    rect,
    rotate,
    to_lattice,
)
from shapely.geometry import Polygon
from shapely.ops import unary_union

from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import Footprint, PrismRecord

CPW_GAP = 6.0
BEND_R = 100.0
CIRCLE = 32  # chords per full circle above 100 um diameter

# Cell 14: gdsfactory's 90-degree Euler bend, p = 0.5, R = 100, minimum radius 70.61 um; 14 chords.
EULER_90 = [
    (0.0, 0.0), (11.091, 0.0581), (22.1737, 0.4645), (33.2073, 1.5657), (44.0855, 3.7), (54.6071, 7.1796),
    (66.8144, 13.7046), (77.5143, 22.4857), (86.2954, 33.1856), (92.8204, 45.3929), (96.3, 55.9145),
    (98.4343, 66.7927), (99.5355, 77.8263), (99.9419, 88.909), (100.0, 100.0),
]  # fmt: skip
# Cell 15: the 180-degree Euler U-turn of the meanders, p = 0.5, R = 100, minimum radius 90.86 um; 30 chords,
# legs 200 um apart.
EULER_180 = [
    (0.0, 0.0), (14.2724, 0.0374), (28.5423, 0.2989), (42.7962, 1.0085), (57.0, 2.3887), (71.088, 4.6578),
    (84.9531, 8.025), (98.4382, 12.6814), (111.3289, 18.7886), (123.3506, 26.462), (134.1692, 35.751),
    (143.4289, 46.5928), (150.8787, 58.7496), (156.3349, 71.9222), (159.6633, 85.7861), (160.782, 100.0),
    (159.6633, 114.2139), (156.3349, 128.0778), (150.8787, 141.2504), (143.4289, 153.4072), (134.1692, 164.249),
    (123.3506, 173.538), (111.3289, 181.2114), (98.4382, 187.3186), (84.9531, 191.975), (71.088, 195.3422),
    (57.0, 197.6113), (42.7962, 198.9915), (28.5423, 199.7011), (14.2724, 199.9626), (0.0, 200.0),
]  # fmt: skip


def cpw_scene(  # noqa: PLR0913 - one CPW scene's parts
    name: str,
    cells: tuple[int, ...],
    die: tuple[float, float, float, float],
    pieces: list[list[Um]],
    w: float,
    corners: tuple[list[Um], list[Um]],
) -> Scene:
    """A CPW along consecutive centreline ``pieces`` (both ends on the die edge) with the two grounds beside it.

    Each piece (a straight, a bend) is offset on its own, so where two pieces meet both edges pass exactly
    ``w / 2`` and ``w / 2 + g`` from the joint, as when separately drawn pieces are merged.  ``corners`` are the
    die corners that close the left and the right ground after its mask edge (left: from the path's end back to
    its start; right: from its end on).
    """
    signal = [*piece_edge(pieces, w / 2), *reversed(piece_edge(pieces, -w / 2))]
    return Scene(
        name,
        cells,
        die,
        [
            Conductor("signal", "signal", to_lattice(signal)),
            Conductor(
                "ground_left",
                "Ground",
                to_lattice([*piece_edge(pieces, w / 2 + CPW_GAP), *corners[0]]),
            ),
            Conductor(
                "ground_right",
                "Ground",
                to_lattice([*piece_edge(pieces, -(w / 2 + CPW_GAP)), *corners[1]]),
            ),
        ],
    )


def piece_edge(pieces: list[list[Um]], d: float) -> list[Um]:
    """The edge ``d`` um to the left of a path of straights (two points) and sampled bends between straights.

    A bend's end vertices lie on its end cross-sections, which the straights beside it share, so they come from
    the straights; a bend contributes its interior (miter) vertices only.
    """
    out: list[Um] = []
    for piece in pieces:
        offsets = offset_path(piece, d)
        out += offsets if len(piece) == 2 else offsets[1:-1]
    return out


def arc_path(centre: Um, r: float, start: float, sweep: float, chords: int) -> list[Um]:
    """A sampled circular centreline: ``chords + 1`` points from angle ``start`` over ``sweep`` degrees."""
    return [polar(r, start + sweep * k / chords, centre) for k in range(chords + 1)]


def cell13_circular_bend() -> Scene:
    """Cell 13: a circular CPW bend, R 100, 8 chords per 90 degrees, w 7 / g 6 (edge radii 90.5 to 109.5)."""
    bend = arc_path((0.0, BEND_R), BEND_R, -90.0, 90.0, CIRCLE // 4)
    pieces = [[(-20.0, 0.0), bend[0]], bend, [bend[-1], (BEND_R, BEND_R + 20)]]
    die = (-20.0, -30.0, 130.0, 120.0)
    return cpw_scene(
        "cell13_circular_bend",
        (13,),
        die,
        pieces,
        7.0,
        ([(-20.0, 120.0)], [(130.0, 120.0), (130.0, -30.0), (-20.0, -30.0)]),
    )


def cell14_euler_bend() -> Scene:
    """Cell 14: a 90-degree Euler CPW bend, w 7 / g 6; its first chord turns 0.30 degrees."""
    pieces = [
        [(-20.0, 0.0), EULER_90[0]],
        EULER_90,
        [EULER_90[-1], (BEND_R, BEND_R + 20)],
    ]
    die = (-20.0, -30.0, 130.0, 120.0)
    return cpw_scene(
        "cell14_euler_bend",
        (14,),
        die,
        pieces,
        7.0,
        ([(-20.0, 120.0)], [(130.0, 120.0), (130.0, -30.0), (-20.0, -30.0)]),
    )


def cell15_euler_u_turn() -> Scene:
    """Cell 15: the 180-degree Euler U-turn of a meander, w 10 / g 6; its first chord turns 0.15 degrees.

    The ground inside the U, between the legs, is 178 um wide.
    """
    pieces = [[(-20.0, 0.0), EULER_180[0]], EULER_180, [EULER_180[-1], (-20.0, 200.0)]]
    die = (-20.0, -30.0, 190.0, 230.0)
    return cpw_scene(
        "cell15_euler_u_turn",
        (15,),
        die,
        pieces,
        10.0,
        ([], [(-20.0, 230.0), (190.0, 230.0), (190.0, -30.0), (-20.0, -30.0)]),
    )


MEANDER_TURNS = 12
MEANDER_R = 25.0
MEANDER_LEG = 150.0
MEANDER_POINTS_PER_CIRCLE = (
    48  # 7.5 degrees per vertex, 3.75 where a bend meets a straight
)


def meander_long_outline() -> Scene:
    """Cells 13 and 15 combined: a CPW meander, w 10 / g 6, whose outlines have 500 to 900 vertices.

    Twelve U-turns of radius 25, each drawn as a circular bend with 48 points per full circle, join thirteen
    straight legs 150 um long; both ends lie on the die edge.  Every vertex on a bend turns 7.5 degrees and
    every vertex where a bend meets a straight 3.75.
    """
    pitch, chords = 2 * MEANDER_R, MEANDER_POINTS_PER_CIRCLE // 2
    x_left, x_right = -50.0, MEANDER_LEG + 50.0
    pieces: list[list[Um]] = [[(x_left, 0.0), (MEANDER_LEG, 0.0)]]
    for k in range(MEANDER_TURNS):
        y = pitch * k
        heading_right = k % 2 == 0
        x = MEANDER_LEG if heading_right else 0.0
        sweep = (
            180.0 if heading_right else -180.0
        )  # turn left at the right end, right at the left end
        pieces.append(arc_path((x, y + MEANDER_R), MEANDER_R, -90.0, sweep, chords))
        end = (
            x_right
            if k == MEANDER_TURNS - 1
            else (0.0 if heading_right else MEANDER_LEG)
        )
        pieces.append([(x, y + pitch), (end, y + pitch)])
    top = pitch * MEANDER_TURNS + 30.0
    die = (x_left, -30.0, x_right, top)
    corners = ([(x_right, top), (x_left, top)], [(x_right, -30.0), (x_left, -30.0)])
    return cpw_scene("meander_long_outline", (13, 15), die, pieces, 10.0, corners)


def cell16_coupled_pair() -> Scene:
    """Cell 16: two 7 um traces with 6 um gaps on either side of a 3 um ground rib, leaving in Euler bends.

    The traces run coupled from the die edge at x = -150 to x = -250, then each turns away from the rib through
    the cell-14 bend; the rib widens continuously as they diverge and becomes the ground between the bends.
    """
    bend = [(-250.0 - x, 11.0 + y) for x, y in EULER_90]  # heading -x, turning to +y
    upper = [[(-150.0, 11.0), bend[0]], bend, [bend[-1], (-350.0, 121.0)]]
    lower = [mirror_y(piece) for piece in upper]
    reach = 3.5 + CPW_GAP
    rib_up, rib_lo = piece_edge(upper, reach), piece_edge(lower, -reach)
    outer_up, outer_lo = piece_edge(upper, -reach), piece_edge(lower, reach)
    die = (-370.0, -121.0, -150.0, 121.0)
    return Scene(
        "cell16_coupled_pair",
        (16,),
        die,
        [
            Conductor(
                "trace_upper",
                "trace_upper",
                to_lattice(
                    [*piece_edge(upper, 3.5), *reversed(piece_edge(upper, -3.5))]
                ),
            ),
            Conductor(
                "trace_lower",
                "trace_lower",
                to_lattice(
                    [*piece_edge(lower, 3.5), *reversed(piece_edge(lower, -3.5))]
                ),
            ),
            Conductor(
                "ground_rib",
                "Ground",
                to_lattice(
                    [*rib_up, (-370.0, 121.0), (-370.0, -121.0), *reversed(rib_lo)]
                ),
            ),
            Conductor(
                "ground_upper", "Ground", to_lattice([*outer_up, (-150.0, 121.0)])
            ),
            Conductor(
                "ground_lower", "Ground", to_lattice([*outer_lo, (-150.0, -121.0)])
            ),
        ],
    )


def cell17_t_junction() -> Scene:
    """Cell 17: a Manhattan CPW T-junction, w 10 / g 6, its three arms on the die edge."""
    signal = [
        (-100, -5),
        (-100, 5),
        (-5, 5),
        (-5, 100),
        (5, 100),
        (5, 5),
        (100, 5),
        (100, -5),
    ]
    return Scene(
        "cell17_t_junction",
        (17,),
        (-100.0, -60.0, 100.0, 100.0),
        [
            Conductor("signal", "signal", to_lattice(signal)),
            Conductor("ground_below", "Ground", to_lattice(rect(-100, -60, 100, -11))),
            Conductor("ground_left", "Ground", to_lattice(rect(-100, 11, -11, 100))),
            Conductor("ground_right", "Ground", to_lattice(rect(11, 11, 100, 100))),
        ],
    )


def cell17_shorted_end() -> Scene:
    """Cell 17b: a CPW, w 10 / g 6, whose signal joins the ground where it ends: two 6 um slots end flush."""
    outline = [
        (-150, -60), (50, -60), (50, -11), (-100, -11), (-100, -5), (50, -5),
        (50, 5), (-100, 5), (-100, 11), (50, 11), (50, 60), (-150, 60),
    ]  # fmt: skip
    return Scene(
        "cell17_shorted_end",
        (17,),
        (-150.0, -60.0, 50.0, 60.0),
        [Conductor("ground", "Ground", to_lattice(outline))],
    )


def cell18_handoff() -> Scene:
    """Cell 18: a CPW handoff from w 10 to w 7 over 50 um, gap 6: one edge kink of 1.72 degrees."""
    signal = [(0, -5), (150, -5), (200, -3.5), (200, 3.5), (150, 5), (0, 5)]
    upper = [(0, 11), (150, 11), (200, 9.5), (200, 40), (0, 40)]
    return Scene(
        "cell18_handoff",
        (18,),
        (0.0, -40.0, 200.0, 40.0),
        [
            Conductor("signal", "signal", to_lattice(signal)),
            Conductor("ground_upper", "Ground", to_lattice(upper)),
            Conductor("ground_lower", "Ground", to_lattice(mirror_y(upper))),
        ],
    )


def cell18_taper() -> Scene:
    """Cell 18: a linear taper from w 10 to w 7 over 100 um between two straights: kinks of 0.86 degrees."""
    signal = [
        (-50, -5),
        (0, -5),
        (100, -3.5),
        (150, -3.5),
        (150, 3.5),
        (100, 3.5),
        (0, 5),
        (-50, 5),
    ]
    upper = [(-50, 11), (0, 11), (100, 9.5), (150, 9.5), (150, 40), (-50, 40)]
    return Scene(
        "cell18_taper",
        (18,),
        (-50.0, -40.0, 150.0, 40.0),
        [
            Conductor("signal", "signal", to_lattice(signal)),
            Conductor("ground_upper", "Ground", to_lattice(upper)),
            Conductor("ground_lower", "Ground", to_lattice(mirror_y(upper))),
        ],
    )


def cell19_launcher() -> Scene:
    """Cell 19: a launcher, a 150 x 150 pad tapering at 25.0 degrees to a w 10 CPW (mask at 44.8 degrees).

    The 85 x 150 port sheet on the substrate surface fills the gap behind the pad, abutting the pad and the
    ground exactly.
    """
    signal = [
        (0, -75),
        (150, -75),
        (300, -5),
        (350, -5),
        (350, 5),
        (300, 5),
        (150, 75),
        (0, 75),
    ]
    ground = [
        (-120, -200), (350, -200), (350, -11), (300, -11), (150, -160), (-85, -160),
        (-85, 160), (150, 160), (300, 11), (350, 11), (350, 200), (-120, 200),
    ]  # fmt: skip
    return Scene(
        "cell19_launcher",
        (19,),
        (-120.0, -200.0, 350.0, 200.0),
        [
            Conductor("signal", "signal", to_lattice(signal)),
            Conductor("ground", "Ground", to_lattice(ground)),
        ],
        [
            Port(
                "launch",
                to_lattice(rect(-85, -75, 0, 75)),
                ("signal", "ground"),
                plane="0",
                direction=(1.0, 0.0, 0.0),
            )
        ],
    )


def _comb() -> list[Um]:
    """The left electrode of cell 20: funnel, 100 um terminal and 10 fingers 3.3 wide, 100 long (top finger)."""
    left, tip = -51.65, 48.35
    fingers: list[Um] = []
    for k in range(10):  # from the top finger down: y_top = 64.35 - k * 13.2
        top = 64.35 - 13.2 * k
        fingers += [(tip, top), (tip, top - 3.3), (left, top - 3.3), (left, top - 13.2)]
    return [
        (left, -64.35),
        (-201.65, -5),
        (-301.65, -5),
        (-301.65, 5),
        (-201.65, 5),
        (left, 64.35),
        *fingers[:-1],
    ]


def cell20_comb() -> Scene:
    """Cell 20: an interdigitated comb, 20 fingers 3.3 wide with 3.3 gaps, 100 long, funnel edges at 21.6 degrees.

    The mask opens 85 um round the comb; its funnel edges run at 42.7 degrees.
    """
    comb = _comb()
    upper = [
        (-301.65, 11),
        (-201.65, 11),
        (-51.65, 149.35),
        (51.65, 149.35),
        (201.65, 11),
        (301.65, 11),
        (301.65, 170),
        (-301.65, 170),
    ]
    return Scene(
        "cell20_comb",
        (20,),
        (-301.65, -170.0, 301.65, 170.0),
        [
            Conductor("comb_left", "comb_left", to_lattice(comb)),
            Conductor(
                "comb_right", "comb_right", to_lattice([rotate(p, 180.0) for p in comb])
            ),
            Conductor("ground_upper", "Ground", to_lattice(upper)),
            Conductor("ground_lower", "Ground", to_lattice(mirror_y(upper))),
        ],
    )


# --------------------------------------------------------------------------- cell 21: spline strips

SPLINE_HALF_WIDTH = 5.0  # the strip is its centreline moved by +-5 um in y


def hermite_centreline(per_span: int = 32) -> list[Um]:
    """The interpolation strip's centreline: a piecewise cubic Hermite y(x) through five points, sampled
    ``per_span`` times per span, uniform in x."""
    knots = [
        (-200.0, 0.0, 0.0),
        (-100.0, 60.0, -0.2),
        (0.0, -40.0, 0.0),
        (100.0, 60.0, 0.2),
        (200.0, 0.0, 0.0),
    ]
    out: list[Um] = []
    for (x0, y0, m0), (x1, y1, m1) in zip(knots, knots[1:], strict=False):
        h = x1 - x0
        for i in range(per_span + (1 if x1 == knots[-1][0] else 0)):
            t = i / per_span
            y = (2 * t**3 - 3 * t**2 + 1) * y0 + (t**3 - 2 * t**2 + t) * h * m0
            y += (-2 * t**3 + 3 * t**2) * y1 + (t**3 - t**2) * h * m1
            out.append((x0 + t * h, y))
    return out


def bspline_centreline(per_span: int = 64) -> list[Um]:
    """The B-spline strip's centreline: a clamped cubic B-spline, sampled ``per_span`` times per span, uniform
    in its parameter."""
    control = [
        (-200.0, 0.0),
        (-100.0, 0.0),
        (0.0, 150.0),
        (100.0, -30.0),
        (200.0, -30.0),
    ]
    knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    samples = 2 * per_span
    return [
        _de_boor(control, knots, min(i / samples, 1 - 1e-15))
        if i < samples
        else control[-1]
        for i in range(samples + 1)
    ]


def _de_boor(control: list[Um], knots: list[float], t: float, degree: int = 3) -> Um:
    span = max(i for i in range(degree, len(control)) if knots[i] <= t)
    d = [list(control[j + span - degree]) for j in range(degree + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            i = j + span - degree
            alpha = (t - knots[i]) / (knots[i + 1 + degree - r] - knots[i])
            d[j] = [(1 - alpha) * d[j - 1][c] + alpha * d[j][c] for c in (0, 1)]
    return (d[degree][0], d[degree][1])


def resample_by_turn(centreline: list[Um], min_turn: float) -> list[Um]:
    """A dense centreline thinned until every interior vertex turns by at least ``min_turn`` degrees.

    The vertex with the smallest turn is removed first, and its neighbours' turns are measured again; the two
    end points stay.  On a circle this leaves chords turning between ``min_turn`` and about twice it.
    """
    points = list(centreline)
    alive = [True] * len(points)
    before = list(range(-1, len(points) - 1))
    after = list(range(1, len(points) + 1))

    def turn_at(i: int) -> float:
        return abs(_turn(points[before[i]], points[i], points[after[i]]))

    queue = [(turn_at(i), i) for i in range(1, len(points) - 1)]
    heapq.heapify(queue)
    while queue:
        turn, i = heapq.heappop(queue)
        if not alive[i] or turn != turn_at(i):
            continue  # removed already, or measured again since
        if turn >= min_turn:
            break
        alive[i] = False
        a, b = before[i], after[i]
        after[a], before[b] = b, a
        for j in (a, b):
            if 0 < j < len(points) - 1:
                heapq.heappush(queue, (turn_at(j), j))
    return [p for p, keep in zip(points, alive, strict=True) if keep]


def _turn(a: Um, b: Um, c: Um) -> float:
    cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
    return math.degrees(math.atan2(cross, dot))


def cell21_spline_strip(name: str, centreline: list[Um]) -> Scene:
    """Cell 21: a 10 um spline strip with no ground (the mask covers the die); its ends lie on the die edge."""
    upper = [(x, y + SPLINE_HALF_WIDTH) for x, y in centreline]
    lower = [(x, y - SPLINE_HALF_WIDTH) for x, y in reversed(centreline)]
    return Scene(
        name,
        (21,),
        (-200.0, -100.0, 200.0, 150.0),
        [Conductor("strip", "strip", to_lattice([*upper, *lower]))],
    )


# --------------------------------------------------------------------------- cell 22: the hollow TSV

TSV_HEIGHT = 500.0
TSV_BORE_R, TSV_OUTER_R = 29.8, 30.0
TSV_PHASE = -90.0  # the 24-gons' first vertex lies straight below the centre


def _ring24(r: float) -> list[Um]:
    return ngon(r, 24, phase=TSV_PHASE)


def cell22_tsv_signal() -> Scene:
    """Cell 22: a concentric signal TSV, a 0.2 um wall between two 24-gons (r 29.8 / 30), 500 um tall.

    An Al landing annulus (r 29.8 to a 32-gon of r 100) on each face sits in a 120 um ground opening (a 20 um
    gap).  The landing's bands are measured from the gap only; the wall and the bore are no origins.
    """
    half = 150.0
    die = (-half, -half, half, half)
    bore = to_lattice(_ring24(TSV_BORE_R))
    outer = to_lattice(_ring24(TSV_OUTER_R))
    landing = to_lattice(ngon(100.0, CIRCLE, phase=TSV_PHASE))
    opening = to_lattice(ngon(120.0, CIRCLE, phase=TSV_PHASE))
    square = to_lattice(rect(*die))
    s = Scene("cell22_tsv_signal", (22,), die, [])
    fp = die_footprint(s)
    s.prisms = [
        PrismRecord(
            "vacuum_below", fp, exact_z(-600), exact_z(-TSV_HEIGHT), "vacuum", "vacuum"
        ),
        PrismRecord(
            "substrate",
            Footprint.from_rings(square, [outer]),
            exact_z(-TSV_HEIGHT),
            exact_z(0),
            "substrate",
            "dielectric",
        ),
        PrismRecord(
            "bore",
            Footprint.from_rings(bore),
            exact_z(-TSV_HEIGHT),
            exact_z(0),
            "vacuum",
            "vacuum",
        ),
        PrismRecord("vacuum_above", fp, exact_z(0), exact_z(100), "vacuum", "vacuum"),
    ]
    s.conductors = [
        Conductor("wall", "Signal", outer, [bore], (str(-TSV_HEIGHT), "0")),
        Conductor("top_signal", "Signal", landing, [bore], ("0", "0.1")),
        Conductor(
            "back_signal", "Signal", landing, [bore], ("-500.1", str(-TSV_HEIGHT))
        ),
        Conductor("top_ground", "Ground", square, [opening], ("0", "0.1")),
        Conductor(
            "back_ground", "Ground", square, [opening], ("-500.1", str(-TSV_HEIGHT))
        ),
    ]
    s.expectations = _tsv_expectations
    return s


def _tsv_expectations(scene: Scene) -> dict[str, Expectation]:
    """Both faces alike: landing and ground banded from the 20 um gap, MA toward vacuum, MS toward the substrate
    (the landing's MS face stops at the wall), SA on the gap from the metal round it."""
    shapes = {c.entity: conductor_shape(c) for c in scene.conductors}
    wall = shapes["wall"]
    out: dict[str, Expectation] = {}
    for face, ma, ms, z in (
        ("top", "top", "bottom", "0.0"),
        ("back", "bottom", "top", "-500.0"),
    ):
        landing, ground = shapes[f"{face}_signal"], shapes[f"{face}_ground"]
        gap = Polygon(ground.interiors[0]).difference(
            Polygon(landing.exterior)
        )  # the exposed substrate
        out[f"ma::{face}_signal::{ma}"] = (landing, gap)
        out[f"ms::{face}_signal::{ms}"] = (landing.difference(wall), gap)
        out[f"ma::{face}_ground::{ma}"] = (ground, gap)
        out[f"ms::{face}_ground::{ms}"] = (ground, gap)
        vacuum = "vacuum_above" if face == "top" else "vacuum_below"
        out[f"sa::substrate|{vacuum}::z={z}"] = (gap, unary_union([landing, ground]))
    return out


# --------------------------------------------------------------------------- cells 23 and 24

AIRBRIDGE_METAL = ("0", "0.1")


def cell23_airbridge() -> Scene:
    """Cell 23: an airbridge over a w 10 / g 6 CPW, a 12 x 84 deck (z 3.1..3.4) on two 14 x 14 piers.

    The piers (z 0.1..3.1) stand on the ground; a pier contact is no gap edge and no band face.
    """
    die = (-30.0, -60.0, 30.0, 60.0)
    s = Scene(
        "cell23_airbridge",
        (23,),
        die,
        [
            Conductor(
                "signal", "signal", to_lattice(rect(-30, -5, 30, 5)), z=AIRBRIDGE_METAL
            ),
            Conductor(
                "ground_lower",
                "Ground",
                to_lattice(rect(-30, -60, 30, -11)),
                z=AIRBRIDGE_METAL,
            ),
            Conductor(
                "ground_upper",
                "Ground",
                to_lattice(rect(-30, 11, 30, 60)),
                z=AIRBRIDGE_METAL,
            ),
            Conductor(
                "pier_lower",
                "Ground",
                to_lattice(rect(-7, -49, 7, -35)),
                z=("0.1", "3.1"),
            ),
            Conductor(
                "pier_upper",
                "Ground",
                to_lattice(rect(-7, 35, 7, 49)),
                z=("0.1", "3.1"),
            ),
            Conductor(
                "deck", "Ground", to_lattice(rect(-6, -42, 6, 42)), z=("3.1", "3.4")
            ),
        ],
    )
    s.expectations = _airbridge_expectations
    return s


def _airbridge_expectations(scene: Scene) -> dict[str, Expectation]:
    """The CPW's own bands; each ground's MA face less the pier standing on it.  Pier and deck are not banded."""
    plane = Scene(
        scene.name,
        scene.cells,
        scene.die,
        [c for c in scene.conductors if c.z == AIRBRIDGE_METAL],
    )
    out = planar_expectations(plane)
    piers = unary_union(
        [conductor_shape(c) for c in scene.conductors if c.entity.startswith("pier")]
    )
    for entity in ("ground_lower", "ground_upper"):
        support, origin = out[f"ma::{entity}::top"]
        out[f"ma::{entity}::top"] = (support.difference(piers), origin)
    return out


def cell24_sheet_across_gap(circular: bool) -> Scene:
    """Cell 24: a 2 x 20 um lumped sheet across the 20 um gap between a pad and its ground opening.

    The sheet lies on the substrate surface and enters the pad and the ground by 1 nm each, as the PDK draws it.
    The pad is a 200 um square in a 240 um opening, or a 32-gon of r 100 in one of r 120.
    """
    die = (-200.0, -200.0, 200.0, 200.0)
    if circular:
        pad, opening = to_lattice(ngon(100.0, CIRCLE)), to_lattice(ngon(120.0, CIRCLE))
    else:
        pad, opening = (
            to_lattice(rect(-100, -100, 100, 100)),
            to_lattice(rect(-120, -120, 120, 120)),
        )
    sheet = [(lattice(x), lattice(y)) for x, y in rect(99.999, -1, 120.001, 1)]
    return Scene(
        f"cell24_sheet_{'circular' if circular else 'square'}_pad",
        (24,),
        die,
        [
            Conductor("pad", "pad", pad),
            Conductor("ground", "Ground", to_lattice(rect(*die)), [opening]),
        ],
        [Port("sheet", sheet, ("pad", "ground"), plane="0", direction=(1.0, 0.0, 0.0))],
    )


SPLINES = {"interpolation": hermite_centreline, "bspline": bspline_centreline}
TURN_RULE_DEGREES = 360.0 / 128
# The resampled strips aim at 3 degrees so that the turns stay above the rule once rounded to the lattice.
RESAMPLE_TURN_DEGREES = 3.0

SCENES = {
    "cell13_circular_bend": cell13_circular_bend,
    "cell14_euler_bend": cell14_euler_bend,
    "cell15_euler_u_turn": cell15_euler_u_turn,
    "meander_long_outline": meander_long_outline,
    "cell16_coupled_pair": cell16_coupled_pair,
    "cell17_t_junction": cell17_t_junction,
    "cell17_shorted_end": cell17_shorted_end,
    "cell18_handoff": cell18_handoff,
    "cell18_taper": cell18_taper,
    "cell19_launcher": cell19_launcher,
    "cell20_comb": cell20_comb,
    "cell21_interpolation_resampled": lambda: cell21_spline_strip(
        "cell21_interpolation_resampled",
        resample_by_turn(hermite_centreline(1024), RESAMPLE_TURN_DEGREES),
    ),
    "cell21_bspline_resampled": lambda: cell21_spline_strip(
        "cell21_bspline_resampled",
        resample_by_turn(bspline_centreline(1024), RESAMPLE_TURN_DEGREES),
    ),
    "cell22_tsv_signal": cell22_tsv_signal,
    "cell23_airbridge": cell23_airbridge,
    "cell24_sheet_square_pad": lambda: cell24_sheet_across_gap(False),
    "cell24_sheet_circular_pad": lambda: cell24_sheet_across_gap(True),
}
# The spline strips as the PDK draws them (258 vertices, smallest turns 0.012 and 0.058 degrees).  They build
# with exact bands as drawn; the supported roster above carries the resampled strips.
DRAWN_SPLINES = {
    "cell21_interpolation_drawn": lambda: cell21_spline_strip(
        "cell21_interpolation_drawn", hermite_centreline()
    ),
    "cell21_bspline_drawn": lambda: cell21_spline_strip(
        "cell21_bspline_drawn", bspline_centreline()
    ),
}
