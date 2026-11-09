"""Draw the figures of the Supported Geometry page from the SGB-native sample scenes.

Owns one plan-view SVG per supported unit cell, written to ``docs/assets/supported-geometry/``.  The scenes are
the ones the tests build (``tests/sgb_supported_cells.py`` and ``tests/sgb_supported_pdk_cells.py``); nothing
is drawn from PDK code, and no model is built.  The figures carry no words, only a scale bar, so both language
editions use the same files.

Run from the repository root: ``python scripts/draw_supported_geometry.py``.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "supported-geometry"
WIDTH_PX = 360
SUBSTRATE = "#f3efe6"
GROUND = "#9aabbd"
RAISED = (
    "#4f6479"  # metal standing above the plane metal: bumps, piers, airbridge decks
)
CONDUCTORS = ("#d9822b", "#3a7ca5", "#7b5ea7", "#4c9a5f", "#c0504d")
PORT = "#e9c46a"
INK = "#3d3d3d"
SCALE_STEPS = (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500)

View = tuple[float, float, float, float]


@dataclass(frozen=True)
class Figure:
    """One figure: the cell it shows, the sample scene it draws and the window (um) it frames."""

    cell: int
    scene: str
    view: View | None = None  # the whole die when None
    entities: tuple[str, ...] = ()  # every conductor when empty


def main() -> None:
    """Load the sample scenes and write one SVG per figure."""
    scenes = _load_scenes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for figure in figures():
        scene = scenes[figure.scene]()
        path = OUTPUT / f"cell-{figure.cell:02d}.svg"
        path.write_text(draw(scene, figure.view or scene.die, figure.entities))
        print(f"wrote {path.relative_to(ROOT)}")


def _load_scenes() -> dict:
    sys.path.insert(0, str(ROOT / "tests"))
    disk_qubit = importlib.import_module("sgb_supported_cells")
    pdk = importlib.import_module("sgb_supported_pdk_cells")
    return {**disk_qubit.SCENES, **pdk.SCENES}


def figures() -> list[Figure]:
    """The figure of every unit cell, with a zoom where the feature is small against its die."""
    junction = _junction_centre()
    return [
        Figure(1, "cell01_disk"),
        Figure(2, "cell02_clearance"),
        Figure(3, "claw_floating", (150, -140, 430, 140)),
        Figure(4, "claw_floating", (146, 54, 168, 76)),
        Figure(5, "claw_floating", (300, -70, 440, 70)),
        Figure(6, "cell06_cpw_straight"),
        Figure(7, "cell07_cpw45_ro"),
        Figure(8, "xy_tip", (-308, -308, -278, -278)),
        Figure(9, "junction_fan", (60, -230, 260, -30)),
        Figure(
            10,
            "junction_fan",
            (junction[0] - 7, junction[1] - 7, junction[0] + 7, junction[1] + 7),
        ),
        Figure(
            11,
            "cell11_bumps_flip_chip",
            (230, 20, 470, 260),
            ("d0_ground", "bump_01", "bump_02", "bump_03"),
        ),
        Figure(12, "d1_die_test_build"),
        Figure(13, "cell13_circular_bend"),
        Figure(14, "cell14_euler_bend"),
        Figure(15, "cell15_euler_u_turn"),
        Figure(16, "cell16_coupled_pair"),
        Figure(17, "cell17_t_junction"),
        Figure(18, "cell18_handoff"),
        Figure(19, "cell19_launcher"),
        Figure(20, "cell20_comb"),
        Figure(21, "cell21_bspline_resampled"),
        Figure(
            22,
            "cell22_tsv_signal",
            (-130, -130, 130, 130),
            ("top_ground", "top_signal"),
        ),
        Figure(23, "cell23_airbridge"),
        Figure(24, "cell24_sheet_square_pad", (88, -14, 132, 14)),
    ]


def _junction_centre() -> tuple[float, float]:
    cells = importlib.import_module("sgb_supported_cells")
    d = cells.rotate(cells.DISK_TERMINAL, cells.JUNCTION_ANGLE)
    g = cells.rotate(cells.GROUND_TERMINAL, cells.JUNCTION_ANGLE)
    return ((d[0] + g[0]) / 2, (d[1] + g[1]) / 2)


def draw(scene, view: View, entities: tuple[str, ...]) -> str:
    """The SVG of a scene's conductors and ports inside ``view``: exposed substrate, metal, sheets, scale bar."""
    x0, y0, x1, y1 = view
    height_px = round(WIDTH_PX * (y1 - y0) / (x1 - x0))
    scale = WIDTH_PX / (x1 - x0)
    colours = _colours(scene)
    body = [f'<rect width="{WIDTH_PX}" height="{height_px}" fill="{SUBSTRATE}"/>']
    shown = [c for c in scene.conductors if not entities or c.entity in entities]
    for c in sorted(shown, key=lambda c: float(c.z[0])):
        raised = float(c.z[0]) > 0.1
        fill = RAISED if raised else colours[c.entity]
        body.append(
            _path([c.outer, *c.holes], view, scale, fill, 0.6 if raised else 1.0)
        )
    for p in scene.ports:
        body.append(_path([p.outline], view, scale, PORT, 0.9))
    body.append(_scale_bar(x1 - x0, scale, height_px))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_PX}" height="{height_px}" '
        f'viewBox="0 0 {WIDTH_PX} {height_px}" font-family="sans-serif">\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def _colours(scene) -> dict[str, str]:
    nets = sorted({c.net for c in scene.conductors if c.net != "Ground"})
    by_net = {net: CONDUCTORS[i % len(CONDUCTORS)] for i, net in enumerate(nets)}
    return {
        c.entity: GROUND if c.net == "Ground" else by_net[c.net]
        for c in scene.conductors
    }


def _path(rings, view: View, scale: float, fill: str, opacity: float) -> str:
    x0, _, _, y1 = view
    parts = []
    for i, ring in enumerate(rings):
        if (_twice_area(ring) > 0) == (
            i > 0
        ):  # outer ring counter-clockwise, holes clockwise
            ring = ring[::-1]
        points = [((x * 0.001 - x0) * scale, (y1 - y * 0.001) * scale) for x, y in ring]
        parts.append("M" + " L".join(f"{px:.2f},{py:.2f}" for px, py in points) + " Z")
    return (
        f'<path d="{" ".join(parts)}" fill="{fill}" fill-opacity="{opacity}" fill-rule="evenodd" '
        f'stroke="{INK}" stroke-width="0.6" stroke-opacity="0.6"/>'
    )


def _twice_area(ring) -> int:
    return sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(ring, [*ring[1:], ring[0]], strict=True)
    )


def _scale_bar(span_um: float, scale: float, height_px: int) -> str:
    length = max(s for s in SCALE_STEPS if s <= span_um / 4)
    px = length * scale
    x, y = 12.0, height_px - 12.0
    label = f"{length:g} µm"
    return (
        f'<rect x="{x - 4}" y="{y - 22}" width="{px + 8:.2f}" height="28" fill="white" fill-opacity="0.8"/>'
        f'<line x1="{x}" y1="{y}" x2="{x + px:.2f}" y2="{y}" stroke="{INK}" stroke-width="2.5"/>'
        f'<text x="{x + px / 2:.2f}" y="{y - 6}" text-anchor="middle" font-size="12" fill="{INK}">{label}</text>'
    )


if __name__ == "__main__":
    main()
