"""Draw the figures of the Supported Geometry and Known Geometry Problems pages from SGB-native scenes.

Owns the plan-view SVGs in ``docs/assets/supported-geometry/``: one per supported unit cell (``cell-NN.svg``),
drawn from the scenes the tests build (``tests/sgb_supported_cells.py`` and ``tests/sgb_supported_pdk_cells.py``),
one per known problem (``problem-*.svg``), drawn here on the 1 nm lattice, and one per unsupported case
(``unsupported-*.svg``), drawn from the samples the tests pin at their refusal (``tests/sgb_unsupported_cells.py``) with their band boundaries constructed exactly here.  Nothing is drawn from
PDK code, and no model is built.  A problem or unsupported figure marks the problem in red and adds zoomed
panels where it is too small to see; the figures carry no words, only scale bars, so both language editions use
the same files.

Run from the repository root: ``python scripts/draw_supported_geometry.py``.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import sgb_supported_cells as disk_qubit  # noqa: E402
import sgb_supported_pdk_cells as pdk  # noqa: E402
import sgb_unsupported_cells as unsupported  # noqa: E402
from sgb_supported_kit import (  # noqa: E402
    MARGINS,
    Conductor,
    Scene,
    Um,
    ngon,
    polar,
    rect,
    to_lattice,
)

OUTPUT = ROOT / "docs" / "assets" / "supported-geometry"
SINGLE_PANEL_PX = 360
ZOOMED_PANEL_PX = 260
PANEL_GAP_PX = 12
SUBSTRATE = "#f3efe6"
GROUND = "#9aabbd"
RAISED = (
    "#4f6479"  # metal standing above the plane metal: bumps, piers, airbridge decks
)
CONDUCTORS = ("#d9822b", "#3a7ca5", "#7b5ea7", "#4c9a5f", "#c0504d")
PORT = "#e9c46a"
INK = "#3d3d3d"
MARK = "#d62828"
OFF_DIE = "#ffffff"  # beyond the die edge, where a figure shows it
SCALE_STEPS = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
)

View = tuple[float, float, float, float]


@dataclass(frozen=True)
class Mark:
    """An overlay in micrometres: a polyline (closed or not), or its points as dots; red unless ``colour`` is given."""

    points: tuple[Um, ...]
    closed: bool = False
    dots: bool = False
    dashed: bool = False
    colour: str = MARK
    width: float = 1.6


@dataclass
class Drawing:
    """A scene to draw with the marks that point at its problem.

    ``die`` is the die outline, for a figure that shows the die edge: the window is then white beyond it.
    """

    scene: Scene
    marks: list[Mark] = field(default_factory=list)
    die: tuple[Um, ...] | None = None


@dataclass(frozen=True)
class Figure:
    """One figure: its file name, how to make the drawing, the window (um) and the zoomed windows."""

    name: str
    make: Callable[[], Drawing]
    view: View | None = None  # the whole die when None
    entities: tuple[str, ...] = ()  # every conductor when empty
    zooms: tuple[View, ...] = ()


def main() -> None:
    """Write one SVG per supported unit cell, per known problem and per unsupported case."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for figure in [*cell_figures(), *problem_figures(), *unsupported_figures()]:
        drawing = figure.make()
        path = OUTPUT / f"{figure.name}.svg"
        path.write_text(draw(drawing, figure))
        print(f"wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- supported unit cells


def cell_figures() -> list[Figure]:
    """The figure of every unit cell, with a narrower window where the feature is small against its die."""
    # Cell 12 is drawn from the test build of the qubit die, off the banded list for its straight lead (Issue #128).
    scenes = {
        **disk_qubit.SCENES,
        **disk_qubit.REFUSED_BEFORE_MESHING,
        **pdk.SCENES,
    }

    def cell(
        number: int,
        scene: str,
        view: View | None = None,
        entities: tuple[str, ...] = (),
    ) -> Figure:
        return Figure(
            f"cell-{number:02d}", lambda: Drawing(scenes[scene]()), view, entities
        )

    d = disk_qubit.rotate(disk_qubit.DISK_TERMINAL, disk_qubit.JUNCTION_ANGLE)
    g = disk_qubit.rotate(disk_qubit.GROUND_TERMINAL, disk_qubit.JUNCTION_ANGLE)
    junction = ((d[0] + g[0]) / 2, (d[1] + g[1]) / 2)
    return [
        cell(1, "cell01_disk"),
        cell(2, "cell02_clearance"),
        cell(3, "claw_floating", (150, -140, 430, 140)),
        cell(4, "claw_floating", (146, 54, 168, 76)),
        cell(5, "claw_floating", (300, -70, 440, 70)),
        cell(6, "cell06_cpw_straight"),
        cell(7, "cell07_cpw45_ro"),
        cell(8, "xy_tip", (-308, -308, -278, -278)),
        cell(9, "junction_fan", (60, -230, 260, -30)),
        cell(
            10,
            "junction_fan",
            (junction[0] - 7, junction[1] - 7, junction[0] + 7, junction[1] + 7),
        ),
        cell(
            11,
            "cell11_bumps_flip_chip",
            (230, 20, 470, 260),
            ("d0_ground", "bump_01", "bump_02", "bump_03"),
        ),
        cell(12, "d1_die_test_build"),
        cell(13, "cell13_circular_bend"),
        cell(14, "cell14_euler_bend"),
        cell(15, "cell15_euler_u_turn"),
        cell(16, "cell16_coupled_pair"),
        cell(17, "cell17_t_junction"),
        cell(18, "cell18_handoff"),
        cell(19, "cell19_launcher"),
        cell(20, "cell20_comb"),
        cell(21, "cell21_bspline_resampled"),
        cell(
            22,
            "cell22_tsv_signal",
            (-130, -130, 130, 130),
            ("top_ground", "top_signal"),
        ),
        cell(23, "cell23_airbridge"),
        cell(24, "cell24_sheet_square_pad", (88, -14, 132, 14)),
    ]


# --------------------------------------------------------------------------- known problems


def problem_figures() -> list[Figure]:
    """One figure per known problem: the case drawn at its dimensions, with zooms onto it."""
    return [
        Figure("problem-spline-curve", spline_curve_drawing, (-210, -60, 210, 160)),
        *boundary_figures(),
    ]  # fmt: skip


def _around(centre: Um, half: float) -> View:
    return (centre[0] - half, centre[1] - half, centre[0] + half, centre[1] + half)


def _metal(entity: str, points: list[Um], net: str | None = None) -> Conductor:
    return Conductor(entity, net or entity, to_lattice(points))


def spline_curve_drawing() -> Drawing:
    """The B-spline strip with its control polygon: the curve SGB would have to offset exactly."""
    scene = pdk.cell21_spline_strip("bspline_dense", pdk.bspline_centreline(256))
    scene.die = (-210.0, -60.0, 210.0, 160.0)
    control = (
        (-200.0, 0.0),
        (-100.0, 0.0),
        (0.0, 150.0),
        (100.0, -30.0),
        (200.0, -30.0),
    )
    return Drawing(scene, [Mark(control, dashed=True), Mark(control, dots=True)])


# A conductor on the model's planar boundary.  The die is 60 x 40 um; ground strips run across it, cut off at
# x = 0 and x = 60, and a centre conductor sits on the left boundary x = 0, where the vacuum ends.  The CPW's
# strip and the square pad occupy a segment of that boundary; the 64-gon and the diamond touch it at one vertex.
BOUNDARY_DIE = (0.0, 0.0, 60.0, 40.0)
BOUNDARY_VIEW = (-8.0, -4.0, 64.0, 44.0)
BOUNDARY_POINT = (0.0, 20.0)
BOUNDARY_GROUNDS = (rect(0, 0, 60, 12), rect(0, 28, 60, 40))
BOUNDARY_CENTRES: dict[str, tuple[list[Um], View]] = {
    "cpw": (rect(0, 18, 60, 22), _around(BOUNDARY_POINT, 4.0)),
    "square": (rect(0, 14, 12, 26), _around((0.0, 14.0), 4.0)),
    "round": (ngon(6.0, 64, (6.0, 20.0)), _around(BOUNDARY_POINT, 3.0)),
    "diamond": ([(0, 20), (6, 14), (12, 20), (6, 26)], _around(BOUNDARY_POINT, 3.0)),
}
# The fix: the ground drawn round the pad's left side, 4 um from it, so the ground sets the boundary x = -10.
BOUNDARY_FIXED_GROUNDS = (rect(-10, 0, 60, 12), rect(-10, 28, 60, 40), rect(-10, 12, -4, 28))  # fmt: skip


def boundary_figures() -> list[Figure]:
    """The four conductors on the boundary, each zoomed where it meets it, and the round pad with the fix."""
    return [
        *(Figure(f"problem-boundary-{name}", lambda name=name: boundary_drawing(name), BOUNDARY_VIEW,
                 zooms=(zoom,)) for name, (_, zoom) in BOUNDARY_CENTRES.items()),
        Figure("problem-boundary-fixed", boundary_fixed_drawing, (-18.0, -4.0, 64.0, 44.0),
               zooms=(_around(BOUNDARY_POINT, 6.0),)),
    ]  # fmt: skip


def boundary_drawing(name: str) -> Drawing:
    """One centre conductor between the ground strips; the stretches of boundary metal occupies in ink, a point in red."""
    outline, _ = BOUNDARY_CENTRES[name]
    conductors = [
        _metal(f"ground_{k}", ground, "Ground")
        for k, ground in enumerate(BOUNDARY_GROUNDS)
    ]
    conductors.append(_metal(name, outline))
    scene = Scene(f"boundary_{name}", (), BOUNDARY_DIE, conductors)
    occupied = [((0.0, 0.0), (0.0, 12.0)), ((0.0, 28.0), (0.0, 40.0))]
    if name == "cpw":
        occupied.append(((0.0, 18.0), (0.0, 22.0)))
    if name == "square":
        occupied.append(((0.0, 14.0), (0.0, 26.0)))
    marks = [_occupied(segment) for segment in occupied]
    if name in ("round", "diamond"):
        marks.append(Mark((BOUNDARY_POINT,), dots=True))
    return Drawing(scene, marks, die=tuple(rect(*BOUNDARY_DIE)))


def boundary_fixed_drawing() -> Drawing:
    """The round pad with the ground drawn round its left side: the ground occupies the whole boundary x = -10."""
    outline, _ = BOUNDARY_CENTRES["round"]
    conductors = [
        _metal(f"ground_{k}", ground, "Ground")
        for k, ground in enumerate(BOUNDARY_FIXED_GROUNDS)
    ]
    conductors.append(_metal("round", outline))
    die = (-10.0, 0.0, 60.0, 40.0)
    scene = Scene("boundary_fixed", (), die, conductors)
    return Drawing(
        scene, [_occupied(((-10.0, 0.0), (-10.0, 40.0)))], die=tuple(rect(*die))
    )


def _occupied(segment: tuple[Um, Um]) -> Mark:
    """A stretch of the boundary that metal occupies: a heavy ink line along the die edge."""
    return Mark(segment, colour=INK, width=4.0)


# --------------------------------------------------------------------------- unsupported geometry


def unsupported_figures() -> list[Figure]:
    """One figure per unsupported case: the sample SCGSim refuses, its band boundaries in ink, the problem in red."""
    outer_cut = unsupported.ARC_OUTER_CUT_UM
    return [
        Figure("unsupported-arc-die-cut", arc_die_cut_drawing, (-2.0, -1.5, 18.0, 12.0),
               zooms=(_around(outer_cut, 1.4),)),
        Figure("unsupported-notch-beside-cut", notch_drawing, (6.0, 8.0, 30.0, 32.0),
               zooms=(_around((21.7, 27.2), 1.4),)),
        Figure("unsupported-lead-core", lead_core_drawing, (8.0, -6.0, 32.0, 6.0),
               zooms=(_around((20.0, 0.0), 1.4), _around((20.0, 0.0), 0.01))),
    ]  # fmt: skip


def _band_line(points: list[Um]) -> Mark:
    return Mark(tuple(points), colour=INK, width=1.0)


def _arc_points(
    centre: Um, radius: float, start_deg: float, end_deg: float
) -> list[Um]:
    """Points on a circle from one angle to another, every quarter degree or closer."""
    steps = max(2, math.ceil(abs(end_deg - start_deg) * 4))
    return [
        polar(radius, start_deg + (end_deg - start_deg) * k / steps, centre)
        for k in range(steps + 1)
    ]


def arc_die_cut_drawing() -> Drawing:
    """The curved strip cut by the die edge, with the bands it would have if it ended there: caps about (0, 8) um.

    Measured from its two arcs and its straight end, the die edge being no origin, the boundary at margin ``m`` is
    the inner arc's offset (radius r_inner + m) from the die edge round to y = m, the straight end's offset y = m,
    and the outer arc's offset (radius r_outer - m) up to the ray through the outer cut.  Beyond that ray, nearer
    the die edge, the nearest point of the outer arc is its end on the die edge, so the boundary turns into the
    half circle of radius ``m`` about that end until it meets the die edge: the cap in red, which SCGSim does not
    build.
    """
    catalogue = unsupported.arc_cut_by_the_die_edge()
    strip = next(p for p in catalogue.prisms if p.entity_id == "strip")
    scene = Scene(
        "arc_die_cut",
        (),
        unsupported.ARC_DIE,
        [Conductor("strip", "strip", list(strip.footprint.outer))],
    )
    centre = unsupported.ARC_CENTRE_UM
    inner, outer = unsupported.ARC_INNER_R_UM, unsupported.ARC_OUTER_R_UM
    cut = unsupported.ARC_OUTER_CUT_UM
    cut_deg = math.degrees(math.atan2(cut[1] - centre[1], cut[0] - centre[0]))
    marks = []
    for m in MARGINS:
        # where the inner offset meets the die edge x = 0, and where each offset meets the straight end's y = m
        on_die_deg = math.degrees(math.acos((0.0 - centre[0]) / (inner + m)))
        inner_end_deg = math.degrees(math.asin(m / (inner + m)))
        outer_start_deg = math.degrees(math.asin(m / (outer - m)))
        boundary = [
            *_arc_points(centre, inner + m, on_die_deg, inner_end_deg),
            *_arc_points(centre, outer - m, outer_start_deg, cut_deg),
        ]
        marks.append(_band_line(boundary))
        # The cap: from the outer offset's end, on the ray from the centre through the cut, down to the die edge.
        marks.append(
            Mark(tuple(_arc_points(cut, m, cut_deg - 180.0, -90.0)), dashed=True)
        )
    return Drawing(scene, marks, die=tuple(rect(*unsupported.ARC_DIE)))


def notch_drawing() -> Drawing:
    """The strip cut on the notch floor: its right side's continuation past the die edge, and the notch wall beside it.

    The continuation runs on along the side's own line until it lies twice the largest margin past the floor's
    line, as SGB continues it; the notch wall it comes within the margin of is red.
    """
    scene = unsupported.strip_into_notch()
    # the right side, up to the notch floor at y = 26 um
    start, vertex = unsupported.NOTCH_STRIP[1:3]
    rise = 2 * MARGINS[-1] / (vertex[1] - start[1])
    end = (
        vertex[0] + rise * (vertex[0] - start[0]),
        vertex[1] + rise * (vertex[1] - start[1]),
    )
    die = tuple((float(x), float(y)) for x, y in unsupported.NOTCH_OUTLINE)
    wall = die[3:5]  # the notch's right wall, x = 22 um
    return Drawing(scene, [Mark((vertex, end), dashed=True), Mark(wall)], die=die)


LEAD_WIDTH_UM = 2.004  # twice the 1.0 um margin and 4 grid steps more


def lead_core_drawing() -> Drawing:
    """The isolated lead 2.004 um wide: the 0.5 um band boundary in ink, the 1.0 um one in red round the core.

    Every edge of the lead is a gap edge and every corner convex, so the boundary at margin ``m`` is the lead's
    rectangle moved in by ``m`` on all sides; inside the largest one lies the core, 4 grid steps wide.
    """
    scene = unsupported.straight_lead(LEAD_WIDTH_UM)
    x0, x1 = unsupported.LEAD_X_UM
    half = LEAD_WIDTH_UM / 2
    boundaries = [tuple(rect(x0 + m, m - half, x1 - m, half - m)) for m in MARGINS]
    marks = [Mark(b, closed=True, colour=INK, width=1.0) for b in boundaries[:-1]]
    marks.append(Mark(boundaries[-1], closed=True))
    return Drawing(scene, marks)


# --------------------------------------------------------------------------- SVG


def draw(drawing: Drawing, figure: Figure) -> str:
    """The SVG of a drawing: the main window and any zoomed windows side by side, each with a scale bar."""
    views = [figure.view or drawing.scene.die, *figure.zooms]
    width = SINGLE_PANEL_PX if len(views) == 1 else ZOOMED_PANEL_PX
    heights = [round(width * (v[3] - v[1]) / (v[2] - v[0])) for v in views]
    height = max(heights)
    panels = []
    for i, (view, panel_height) in enumerate(zip(views, heights, strict=True)):
        x = i * (width + PANEL_GAP_PX)
        y = (height - panel_height) // 2
        body = _panel(drawing, figure, view, (width, panel_height), framed=i == 0)
        panels.append(
            f'<svg x="{x}" y="{y}" width="{width}" height="{panel_height}">{body}</svg>'
        )
    total = len(views) * width + (len(views) - 1) * PANEL_GAP_PX
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="{height}" '
        f'viewBox="0 0 {total} {height}" font-family="sans-serif">\n'
        + "\n".join(panels)
        + "\n</svg>\n"
    )


def _panel(
    drawing: Drawing, figure: Figure, view: View, size: tuple[int, int], framed: bool
) -> str:
    """One window of a drawing; the main window (``framed``) also outlines the zoomed windows."""
    (width, height), entities = size, figure.entities
    frames = figure.zooms if framed else ()
    scale = width / (view[2] - view[0])
    to_px = _projection(view, scale)
    colours = _colours(drawing.scene)
    body = [f'<rect width="{width}" height="{height}" fill="{SUBSTRATE}"/>']
    if drawing.die is not None:
        body = [
            f'<rect width="{width}" height="{height}" fill="{OFF_DIE}"/>',
            _die(drawing.die, to_px),
        ]
    shown = [
        c for c in drawing.scene.conductors if not entities or c.entity in entities
    ]
    for c in sorted(shown, key=lambda c: float(c.z[0])):
        raised = float(c.z[0]) > 0.1
        rings = [
            [(x * 0.001, y * 0.001) for x, y in ring] for ring in (c.outer, *c.holes)
        ]
        body.append(
            _path(
                rings,
                to_px,
                RAISED if raised else colours[c.entity],
                0.6 if raised else 1.0,
            )
        )
    for p in drawing.scene.ports:
        body.append(
            _path([[(x * 0.001, y * 0.001) for x, y in p.outline]], to_px, PORT, 0.9)
        )
    body += [_mark(m, to_px) for m in drawing.marks]
    body += [
        _mark(Mark(tuple(rect(*f)), closed=True, dashed=True), to_px) for f in frames
    ]
    body.append(
        f'<rect width="{width}" height="{height}" fill="none" stroke="{INK}" stroke-width="1"/>'
    )
    body.append(_scale_bar(view[2] - view[0], scale, height))
    return "".join(body)


def _projection(view: View, scale: float) -> Callable[[Um], tuple[float, float]]:
    x0, _, _, y1 = view
    return lambda p: ((p[0] - x0) * scale, (y1 - p[1]) * scale)


def _colours(scene) -> dict[str, str]:
    nets = sorted({c.net for c in scene.conductors if c.net != "Ground"})
    by_net = {net: CONDUCTORS[i % len(CONDUCTORS)] for i, net in enumerate(nets)}
    return {
        c.entity: GROUND if c.net == "Ground" else by_net[c.net]
        for c in scene.conductors
    }


def _path(rings, to_px, fill: str, opacity: float) -> str:
    parts = []
    for i, ring in enumerate(rings):
        if (_twice_area(ring) > 0) == (
            i > 0
        ):  # outer ring counter-clockwise, holes clockwise
            ring = ring[::-1]
        parts.append(
            "M" + " L".join("{:.2f},{:.2f}".format(*to_px(p)) for p in ring) + " Z"
        )
    return (
        f'<path d="{" ".join(parts)}" fill="{fill}" fill-opacity="{opacity}" fill-rule="evenodd" '
        f'stroke="{INK}" stroke-width="0.6" stroke-opacity="0.6"/>'
    )


def _die(outline: tuple[Um, ...], to_px) -> str:
    """The die as exposed substrate, its edge drawn as a line."""
    points = " ".join("{:.2f},{:.2f}".format(*to_px(p)) for p in outline)
    return f'<polygon points="{points}" fill="{SUBSTRATE}" stroke="{INK}" stroke-width="1.2"/>'


def _mark(mark: Mark, to_px) -> str:
    pixels = [to_px(p) for p in mark.points]
    if mark.dots:
        return "".join(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{mark.colour}"/>'
            for x, y in pixels
        )
    dash = ' stroke-dasharray="5 3"' if mark.dashed else ""
    shape = "polygon" if mark.closed else "polyline"
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in pixels)
    return (
        f'<{shape} points="{points}" fill="none" stroke="{mark.colour}" '
        f'stroke-width="{mark.width:g}"{dash}/>'
    )


def _twice_area(ring) -> float:
    return sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(ring, [*ring[1:], ring[0]], strict=True)
    )


def _scale_bar(span_um: float, scale: float, height_px: int) -> str:
    length = max(s for s in SCALE_STEPS if s <= span_um / 4)
    px = length * scale
    x, y = 12.0, height_px - 12.0
    label = f"{length * 1000:g} nm" if length < 1 else f"{length:g} µm"
    return (
        f'<rect x="{x - 4}" y="{y - 22}" width="{px + 8:.2f}" height="28" fill="white" fill-opacity="0.8"/>'
        f'<line x1="{x}" y1="{y}" x2="{x + px:.2f}" y2="{y}" stroke="{INK}" stroke-width="2.5"/>'
        f'<text x="{x + px / 2:.2f}" y="{y - 6}" text-anchor="middle" font-size="12" fill="{INK}">{label}</text>'
    )


if __name__ == "__main__":
    main()
