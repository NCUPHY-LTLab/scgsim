"""Draw the figures of the Supported Geometry and Known Geometry Problems pages from SGB-native scenes.

Owns the plan-view SVGs in ``docs/assets/supported-geometry/``: one per supported unit cell (``cell-NN.svg``),
drawn from the scenes the tests build (``tests/sgb_supported_cells.py`` and ``tests/sgb_supported_pdk_cells.py``),
and one per known problem (``problem-*.svg``), redrawn here on the 1 nm lattice from the dimensions reported for
it.  Nothing is drawn from PDK code, and no model is built.  A problem figure marks the problem in red and adds
zoomed panels where it is too small to see; the figures carry no words, only scale bars, so both language
editions use the same files.

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
from sgb_supported_kit import (  # noqa: E402
    Conductor,
    Scene,
    Um,
    fillet,
    intersect,
    ngon,
    polar,
    rect,
    round_cap,
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
    """A red overlay in micrometres: a polyline (closed or not), or its points as dots."""

    points: tuple[Um, ...]
    closed: bool = False
    dots: bool = False
    dashed: bool = False


@dataclass
class Drawing:
    """A scene to draw with the marks that point at its problem."""

    scene: Scene
    marks: list[Mark] = field(default_factory=list)


@dataclass(frozen=True)
class Figure:
    """One figure: its file name, how to make the drawing, the window (um) and the zoomed windows."""

    name: str
    make: Callable[[], Drawing]
    view: View | None = None  # the whole die when None
    entities: tuple[str, ...] = ()  # every conductor when empty
    zooms: tuple[View, ...] = ()


def main() -> None:
    """Write one SVG per supported unit cell and per known problem."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for figure in [*cell_figures(), *problem_figures()]:
        drawing = figure.make()
        path = OUTPUT / f"{figure.name}.svg"
        path.write_text(draw(drawing, figure))
        print(f"wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- supported unit cells


def cell_figures() -> list[Figure]:
    """The figure of every unit cell, with a narrower window where the feature is small against its die."""
    scenes = {**disk_qubit.SCENES, **pdk.SCENES}

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
    """One figure per known problem: the case drawn at its reported dimensions, with zooms onto it."""
    mouth, step = _crack_points()
    return [
        Figure("problem-c1-crack", crack_drawing, (60, -60, 250, 60),
               zooms=(_around(mouth, 0.6), _around(step, 0.03))),
        Figure("problem-c2-c3-slits", slit_drawing, (150, -110, 250, 30),
               zooms=(_around(SLIT_C2_TIP, 1.6), _around(SLIT_C2_BASE, 0.006))),
        Figure("problem-c4-corner-bite", corner_bite_drawing, (185, 55, 245, 115),
               zooms=(_around(polar(disk_qubit.DISK_R + 120, 22.5), 4.0),)),
        Figure("problem-a1-detached-fingers", detached_fingers_drawing, (-10, -25, 140, 90),
               zooms=(_around((3.3, 0.0), 1.5), _around((3.3, 0.0), 0.006))),
        Figure("problem-a6-etch-frame", etch_frame_drawing, (-1200, -560, 1200, 560),
               zooms=((-1180, 330, -980, 530),)),
        Figure("problem-dense-circle", lambda: _dense_circle(256), (-12, -12, 12, 12),
               zooms=((9.4, -0.6, 10.2, 0.6),)),
        Figure("problem-spline-258", spline_drawing, (-200, -60, 200, 90),
               zooms=((-200, -8, -180, 8),)),
        Figure("problem-spline-curve", spline_curve_drawing, (-210, -60, 210, 160)),
        Figure("problem-small-turn-runs", euler_run_drawing, (-20, -30, 190, 230),
               zooms=((-5, -12, 45, 12),)),
        Figure("problem-single-vertex", single_vertex_drawing, (-55, -55, 55, 55),
               zooms=(_around(_added_vertex(), 0.1),)),
    ]  # fmt: skip


def _around(centre: Um, half: float) -> View:
    return (centre[0] - half, centre[1] - half, centre[0] + half, centre[1] + half)


def _metal(entity: str, points: list[Um], net: str | None = None) -> Conductor:
    return Conductor(entity, net or entity, to_lattice(points))


# C1: the radial lead fan of the disk is a three-point arc that shares only its middle vertex with the
# disk's 32-gon, so a thin wedge stays open between the fan's chord and the disk's chord on either side.
FAN_HALF_SPAN = 10.0  # degrees
CRACK_STEP_Y = (
    1.0  # the wedge closes to a nanometre step about 1 um from the shared vertex
)


def _crack_points() -> tuple[Um, Um]:
    r = disk_qubit.DISK_R
    disk_chord = (polar(r, -11.25), polar(r, 0.0))
    fan_end = polar(r, -FAN_HALF_SPAN)
    mouth = intersect(*disk_chord, fan_end, (fan_end[0] + 1.0, fan_end[1]))
    step = intersect(*disk_chord, (0.0, -CRACK_STEP_Y), (1.0, -CRACK_STEP_Y))
    return mouth, step


def crack_drawing() -> Drawing:
    """C1: the disk and its lead fan in the junction frame (junction axis on +x), both wedges open."""
    r = disk_qubit.DISK_R
    terminal = disk_qubit.DISK_TERMINAL
    outline: list[Um] = [
        polar(r, 360.0 * k / 32) for k in range(1, 32)
    ]  # 11.25 ... 348.75 degrees
    marks = []
    for side in (-1, 1):
        disk_chord = (polar(r, side * 11.25), polar(r, 0.0))
        fan_end = polar(r, side * FAN_HALF_SPAN)
        on_disk = intersect(
            *disk_chord, (0.0, side * CRACK_STEP_Y), (1.0, side * CRACK_STEP_Y)
        )
        on_fan = intersect(
            polar(r, 0.0),
            fan_end,
            (0.0, side * CRACK_STEP_Y),
            (1.0, side * CRACK_STEP_Y),
        )
        mouth = intersect(*disk_chord, fan_end, (fan_end[0] + 1.0, fan_end[1]))
        marks.append(Mark((mouth, on_disk, on_fan, fan_end), closed=True))
        wedge = [on_disk, on_fan, fan_end, (terminal[0], side * disk_qubit.LEAD_W / 2)]
        if side == -1:
            outline += wedge
            outline += round_cap(
                terminal, disk_qubit.LEAD_W / 2, 0.0, inscribed_steps=False
            )[1:-1]
        else:
            outline += wedge[::-1]
    scene = Scene("c1", (), (60.0, -60.0, 250.0, 60.0), [_metal("disk", outline)])
    return Drawing(scene, marks)


# C2 and C3: zero-width slits with a 1 nm base, spliced into the claw outline (reported on the west claw;
# drawn here on the east claw of cell 3).
SLIT_C2_BASE = (212.424, -87.989)
SLIT_C2_TIP = (212.923, -86.783)  # 1.305 um into the metal
SLIT_C3_LENGTH = 2.282
SLIT_BASE = 0.001


def slit_drawing() -> Drawing:
    """C2 on the radial edge beside the outer corner, C3 on a chord of the inner arc, each 1 nm wide at the base."""
    metal = disk_qubit.claw(disk_qubit.DISK_R).metal
    a, b = polar(disk_qubit.DISK_R + 60, -11.25), polar(disk_qubit.DISK_R + 60, 0.0)
    c3_base = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    inward = polar(1.0, -5.625)
    c3_tip = (
        c3_base[0] + SLIT_C3_LENGTH * inward[0],
        c3_base[1] + SLIT_C3_LENGTH * inward[1],
    )
    outline = _splice(metal, SLIT_C2_BASE, SLIT_C2_TIP)
    outline = _splice(outline, c3_base, c3_tip)
    marks = [Mark((SLIT_C2_BASE, SLIT_C2_TIP)), Mark((c3_base, c3_tip))]
    return Drawing(
        Scene("c2c3", (), (150.0, -110.0, 250.0, 30.0), [_metal("claw", outline)]),
        marks,
    )


def _splice(outline: list[Um], base: Um, tip: Um) -> list[Um]:
    """The outline with an out-and-back slit from ``base`` to ``tip`` inserted on the edge nearest ``base``."""
    nearest = min(
        range(len(outline)),
        key=lambda i: _distance_to_segment(
            base, outline[i], outline[(i + 1) % len(outline)]
        ),
    )
    a, b = outline[nearest], outline[(nearest + 1) % len(outline)]
    length = math.dist(a, b)
    along = ((b[0] - a[0]) / length * SLIT_BASE, (b[1] - a[1]) / length * SLIT_BASE)
    back = (base[0] + along[0], base[1] + along[1])
    return [*outline[: nearest + 1], base, tip, back, *outline[nearest + 1 :]]


def _distance_to_segment(p: Um, a: Um, b: Um) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(
        0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy))
    )
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def corner_bite_drawing() -> Drawing:
    """C4: the claw's outer corner rounded with r 5 first, then the taper's sharp 57.7-degree corner added back.

    What stays is a sharp tip, a short edge down the taper's chord to the rounding, the rest of the rounding,
    and a bite between them.
    """
    metal = disk_qubit.claw(disk_qubit.DISK_R).metal
    up = polar(disk_qubit.DISK_R + 120, 22.5)
    corner = metal.index(up)
    inward = metal[
        corner + 1
    ]  # the radial edge runs from the corner toward the inner arc
    along_arc = (
        up[0] + 10 * math.sin(math.radians(22.5)),
        up[1] - 10 * math.cos(math.radians(22.5)),
    )
    rounding = fillet(
        inward, up, along_arc, 5.0, 8
    )  # from the radial edge round to the outer arc
    crossing = next(
        i
        for i in range(len(rounding) - 1)
        if (rounding[i][0] - up[0]) * (rounding[i + 1][0] - up[0]) <= 0
    )
    meet = intersect(
        rounding[crossing], rounding[crossing + 1], up, (up[0], up[1] - 1.0)
    )
    bite = [up, meet, *rounding[crossing::-1]]
    outline = [*metal[:corner], *bite, *metal[corner + 1 :]]
    marks = [Mark((up, meet, *rounding[crossing::-1]), closed=True)]
    return Drawing(
        Scene("c4", (), (185.0, 55.0, 245.0, 115.0), [_metal("claw", outline)]), marks
    )


# A1: ten Purcell IDC fingers start 1 nm above the funnel they should grow from.
FINGER_W, FINGER_PITCH, FINGER_LENGTH, FINGER_GAP = 3.3, 13.2, 59.925, 3.3


def detached_fingers_drawing() -> Drawing:
    """A1: one comb's funnel edge at y = 0 and its fingers from y = 0.001; the other comb's fingers between."""
    top = FINGER_LENGTH + FINGER_GAP
    conductors = [_metal("funnel", rect(-10, -25, 140, 0), "comb_a")]
    marks = []
    for k in range(10):
        x = FINGER_PITCH * k
        conductors.append(
            _metal(
                f"finger_{k}",
                rect(x, 0.001, x + FINGER_W, FINGER_LENGTH + 0.001),
                "comb_a",
            )
        )
        marks.append(Mark(((x, 0.0), (x + FINGER_W, 0.0))))
    other = [(-10, top + 20), (-10, top)]
    for k in range(10):
        x = FINGER_PITCH * k + FINGER_W * 2
        other += [
            (x, top),
            (x, FINGER_GAP),
            (x + FINGER_W, FINGER_GAP),
            (x + FINGER_W, top),
        ]
    other += [(140, top), (140, top + 20)]
    conductors.append(_metal("comb_b", other))
    return Drawing(Scene("a1", (), (-10.0, -25.0, 140.0, 90.0), conductors), marks)


def etch_frame_drawing() -> Drawing:
    """A6: the ground derived from the mask covers a 50 um frame round the 2200 x 900 chip that the etch removes."""
    outer, inner = rect(-1150, -500, 1150, 500), rect(-1100, -450, 1100, 450)
    ground = [_metal("ground_upper", rect(-1150, 11, 1150, 500), "Ground"),
              _metal("ground_lower", rect(-1150, -500, 1150, -11), "Ground"),
              _metal("signal", rect(-1150, -5, 1150, 5))]  # fmt: skip
    marks = [
        Mark(tuple(outer), closed=True),
        Mark(tuple(inner), closed=True, dashed=True),
    ]
    return Drawing(Scene("a6", (), (-1150.0, -500.0, 1150.0, 500.0), ground), marks)


def _dense_circle(points: int) -> Drawing:
    """A polygon circle of radius 10 um with ``points`` vertices, the vertices near +x marked."""
    ring = ngon(10.0, points)
    near_zoom = tuple(p for p in ring if p[0] > 9.0 and abs(p[1]) < 1.0)
    return Drawing(Scene("dense", (), (-12.0, -12.0, 12.0, 12.0), [_metal("disk", ring)]),
                   [Mark(near_zoom, dots=True)])  # fmt: skip


def spline_drawing() -> Drawing:
    """The interpolation spline strip as the PDK draws it: 258 vertices, their turns down to 0.058 degrees."""
    scene = pdk.DRAWN_SPLINES["cell21_interpolation_drawn"]()
    points = tuple((x * 0.001, y * 0.001) for x, y in scene.conductors[0].outer)
    return Drawing(scene, [Mark(tuple(p for p in points if p[0] < -178.0), dots=True)])


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


def euler_run_drawing() -> Drawing:
    """The 180-degree Euler U-turn of cell 15: its first centreline vertices turn 0.15, 0.9, 1.8 and 2.7 degrees."""
    return Drawing(
        pdk.SCENES["cell15_euler_u_turn"](), [Mark(tuple(pdk.EULER_180[:5]), dots=True)]
    )


SINGLE_VERTEX_TURN = 0.3  # degrees


def _added_vertex() -> Um:
    """The vertex added on the first edge of a 16-gon of r 50 um so that it turns ``SINGLE_VERTEX_TURN``."""
    (ax, ay), (bx, by) = ngon(50.0, 16)[:2]
    mx, my = (ax + bx) / 2, (ay + by) / 2
    rise = (
        math.dist((ax, ay), (bx, by))
        / 2
        * math.tan(math.radians(SINGLE_VERTEX_TURN) / 2)
    )
    scale = rise / math.hypot(mx, my)
    return (mx + scale * mx, my + scale * my)


def single_vertex_drawing() -> Drawing:
    """A 16-gon of r 50 um with one vertex added on its first edge, turning 0.3 degrees."""
    points = ngon(50.0, 16)
    added = _added_vertex()
    ring = [points[0], added, *points[1:]]
    scene = Scene("single", (), (-55.0, -55.0, 55.0, 55.0), [_metal("disk", ring)])
    return Drawing(
        scene, [Mark((points[0], points[1]), dashed=True), Mark((added,), dots=True)]
    )


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


def _mark(mark: Mark, to_px) -> str:
    pixels = [to_px(p) for p in mark.points]
    if mark.dots:
        return "".join(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{MARK}"/>'
            for x, y in pixels
        )
    dash = ' stroke-dasharray="5 3"' if mark.dashed else ""
    shape = "polygon" if mark.closed else "polyline"
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in pixels)
    return f'<{shape} points="{points}" fill="none" stroke="{MARK}" stroke-width="1.6"{dash}/>'


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
