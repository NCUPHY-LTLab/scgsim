"""Model builders: the pad-in-ground running example, the airbridge variant and a
randomized layout generator (stage S1/S2 inputs).

Owns turning micrometre descriptions into :class:`PrismRecord` lists in grid
units. All geometry here is a *source* description; nothing is derived. The
running example's pad has exact arcs (a circular hole and a rounded corner).
"""

from __future__ import annotations

import math
import random

from fractions import Fraction

from sgbproto.geometry import ring_twice_area
from sgbproto.records import (
    ArcCarrier, Footprint, Grid, LineEdge, Lineage, ModelSpec, PrismRecord, circle_arcs, z_value,
)

Um = tuple[float, float]


def to_grid(grid: Grid, pts: list[Um]) -> list[tuple[int, int]]:
    return [(grid.snap(x), grid.snap(y)) for x, y in pts]


def rect_um(x0: float, y0: float, x1: float, y1: float) -> list[Um]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def regular_polygon_um(cx: float, cy: float, radius: float, sides: int) -> list[Um]:
    return [
        (cx + radius * math.cos(2 * math.pi * i / sides), cy + radius * math.sin(2 * math.pi * i / sides))
        for i in range(sides)
    ]


def pad_outline_um() -> list[Um]:
    """100x60 pad: chamfered top-right corner and a 20x15 reflex notch in the bottom edge."""
    return [
        (-50, -30), (-10, -30), (-10, -15), (10, -15), (10, -30), (50, -30),
        (50, 18), (38, 30), (-50, 30),
    ]


def _footprint(grid: Grid, outer: list[Um], holes: list[list[Um]] = ()) -> Footprint:
    return Footprint.from_rings(to_grid(grid, outer), [to_grid(grid, h) for h in holes])


def _chip_prisms(grid: Grid, half: float) -> list[PrismRecord]:
    box = _footprint(grid, rect_um(-half, -half, half, half))
    return [
        PrismRecord("substrate", box, z_value(-100), z_value(0), "substrate", "dielectric", role="body"),
        PrismRecord("vacuum", box, z_value(0), z_value(100), "vacuum", "vacuum", role="body"),
    ]


def pad_primitives(grid: Grid, rounded_corner: bool = True) -> list:
    """The pad outline (CCW) as exact primitives: lines plus, optionally, a 5 um quarter-circle rounded corner."""
    def pt(x, y):
        return (grid.snap(x), grid.snap(y))

    corner = -50.0 if not rounded_corner else -45.0
    ring = [
        LineEdge(pt(corner, -30), pt(-10, -30)), LineEdge(pt(-10, -30), pt(-10, -15)),
        LineEdge(pt(-10, -15), pt(10, -15)), LineEdge(pt(10, -15), pt(10, -30)),
        LineEdge(pt(10, -30), pt(50, -30)), LineEdge(pt(50, -30), pt(50, 18)),
        LineEdge(pt(50, 18), pt(38, 30)), LineEdge(pt(38, 30), pt(-50, 30)),
    ]
    if rounded_corner:
        r = grid.snap(5)
        lineage = Lineage("polygon_edge", "pad", 0, 0, 8)
        ring.append(LineEdge(pt(-50, 30), pt(-50, -25)))
        ring.append(ArcCarrier("pad/corner", (Fraction(pt(-45, -25)[0]), Fraction(pt(-45, -25)[1])), Fraction(r), True,
                               (Fraction(pt(-50, -25)[0]), Fraction(pt(-50, -25)[1])),
                               (Fraction(pt(-45, -30)[0]), Fraction(pt(-45, -30)[1])), lineage))
    else:
        ring.append(LineEdge(pt(-50, 30), pt(-50, -30)))
    return ring


def pad_hole_arcs(grid: Grid) -> list[ArcCarrier]:
    """The pad's hole as a true circle of radius 8 um (four quarter arcs, CW as a hole)."""
    return circle_arcs("pad/hole", (grid.snap(-25), grid.snap(5)), grid.snap(8), False, Lineage("polygon_edge", "pad", 0, 1, 0))


def pad_prism(grid: Grid, z0, z1, hole_sides: int | None, rounded_corner: bool) -> PrismRecord:
    outer = pad_primitives(grid, rounded_corner)
    if hole_sides is None:
        footprint, arcs = Footprint.from_primitives(outer, [pad_hole_arcs(grid)])
    else:
        hole = to_grid(grid, regular_polygon_um(-25, 5, 8, hole_sides))
        footprint, arcs = Footprint.from_primitives(outer, [_cw(hole)])
    return PrismRecord("pad", footprint, z0, z1, "Al", "conductor", net_id="pad", role="body", arcs=arcs)


def _cw(points: list) -> list:
    """A polygon given as points as clockwise line primitives."""
    pts = list(points)
    if ring_twice_area(pts) > 0:
        pts.reverse()
    return [LineEdge(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def pad_in_ground(
    route: str = "A",
    grid: Grid = Grid(),
    hole_sides: int | None = None,
    with_port: bool = True,
    rounded_corner: bool = True,
    bands_um: tuple[float, ...] | None = None,
    band_regions: tuple[str, ...] | None = None,
) -> ModelSpec:
    """The running example: ground with cutout, pad (chamfer, notch, 5 um rounded corner, true circular hole),
    lumped port. ``hole_sides`` replaces the circle by a regular polygon; ``bands_um`` adds Surface-EPR inset bands
    on the pad's top surface and the gap ring (Route A: the sheet, the gap) and, in Route B, also on the pad's bottom
    face (MS); ``band_regions`` selects ``pad_top``, ``pad_bottom`` and ``gap`` explicitly."""
    half = 150.0
    thickness = {"A": (0, 0), "B": (0, 0.2)}[route]
    z0, z1 = z_value(thickness[0]), z_value(thickness[1])
    prisms = _chip_prisms(grid, half)
    ground = _footprint(grid, rect_um(-half, -half, half, half), [rect_um(-80, -60, 80, 60)])
    prisms.append(PrismRecord("ground", ground, z0, z1, "Al", "conductor", net_id="Ground", role="body"))
    prisms.append(pad_prism(grid, z0, z1, hole_sides, rounded_corner))
    if with_port:
        port_z = z_value(0) if route == "A" else z_value("0.1")
        port = _footprint(grid, rect_um(50, -5, 80, 5))
        prisms.append(PrismRecord("p1", port, port_z, port_z, "lumped_port", "dielectric", role="port"))
    families = ()
    if bands_um:
        from sgbproto.inset import inset_family

        margins = tuple(grid.snap(m) for m in bands_um)
        pad_top_z = z_value(0) if route == "A" else z_value("0.2")
        regions = band_regions or (("pad_top", "gap") if route == "A" else ("pad_top", "pad_bottom", "gap"))
        fams = []
        if "pad_top" in regions:
            fams.append(inset_family("band_pad_top", pad_top_region(grid, margins, hole_sides, rounded_corner), (pad_top_z,)))
        if "pad_bottom" in regions:  # Route B: the metal-substrate face at the bottom of the pad
            fams.append(inset_family("band_pad_bottom", pad_top_region(grid, margins, hole_sides, rounded_corner), (z_value(0),)))
        if "gap" in regions:
            fams.append(inset_family("band_gap", gap_region(grid, margins, rounded_corner), (z_value(0),)))
        families = tuple(fams)
    suffix = "_bands" if bands_um else ""
    return ModelSpec(grid, tuple(prisms), families, name=f"pad_in_ground_{route}{suffix}", meta={"route": route})


def pad_top_region(grid: Grid, margins: tuple[int, ...], hole_sides: int | None = None, rounded_corner: bool = True):
    """Support region (a): the pad's top surface (outline minus the hole)."""
    from sgbproto.inset import SupportRegion

    outer = pad_primitives(grid, rounded_corner)
    if hole_sides is None:
        hole = pad_hole_arcs(grid)
    else:
        hole = _cw(to_grid(grid, regular_polygon_um(-25, 5, 8, hole_sides)))
    return SupportRegion((tuple(outer), tuple(hole)), margins)


def gap_region(grid: Grid, margins: tuple[int, ...], rounded_corner: bool = True):
    """Support region (b): the substrate-vacuum gap ring between the pad and the ground cutout."""
    from sgbproto.inset import SupportRegion, reverse_ring

    cutout = to_grid(grid, rect_um(-80, -60, 80, 60))
    outer = [LineEdge(cutout[i], cutout[(i + 1) % 4]) for i in range(4)]
    return SupportRegion((tuple(outer), tuple(reverse_ring(pad_primitives(grid, rounded_corner)))), margins)


def airbridge(grid: Grid = Grid()) -> ModelSpec:
    """Two ground halves split by a gap, bridged by two posts and a deck (Route B, multi-slab)."""
    half = 150.0
    g0, g1 = z_value(0), z_value("0.2")
    p0, p1 = g1, z_value(3)
    d0, d1 = p1, z_value("3.3")
    prisms = _chip_prisms(grid, half)
    left = _footprint(grid, rect_um(-half, -half, -20, half))
    right = _footprint(grid, rect_um(20, -half, half, half))
    prisms.append(PrismRecord("ground_left", left, g0, g1, "Al", "conductor", net_id="Ground"))
    prisms.append(PrismRecord("ground_right", right, g0, g1, "Al", "conductor", net_id="Ground"))
    prisms.append(PrismRecord("post_left", _footprint(grid, rect_um(-35, -10, -25, 10)), p0, p1, "Al", "conductor", net_id="Ground"))
    prisms.append(PrismRecord("post_right", _footprint(grid, rect_um(25, -10, 35, 10)), p0, p1, "Al", "conductor", net_id="Ground"))
    prisms.append(PrismRecord("deck", _footprint(grid, rect_um(-35, -10, 35, 10)), d0, d1, "Al", "conductor", net_id="Ground"))
    return ModelSpec(grid, tuple(prisms), name="airbridge_B", meta={"route": "B"})


# --------------------------------------------------------------------------- random layouts


def _rotated_rect(cx: float, cy: float, w: float, h: float, angle: float) -> list[Um]:
    c, s = math.cos(angle), math.sin(angle)
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def _feature_outline(rng: random.Random, cx: float, cy: float, size: float) -> list[Um]:
    """One of: rectangle, L shape, chamfered rectangle, rotated rectangle (all inside ``size``)."""
    w = size * rng.uniform(0.5, 1.0)
    h = size * rng.uniform(0.4, 0.9)
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    kind = rng.choice(["rect", "L", "chamfer", "rotated"])
    if kind == "rect":
        return rect_um(x0, y0, x1, y1)
    if kind == "L":
        nx, ny = w * rng.uniform(0.3, 0.6), h * rng.uniform(0.3, 0.6)
        return [(x0, y0), (x1, y0), (x1, y0 + ny), (x0 + nx, y0 + ny), (x0 + nx, y1), (x0, y1)]
    if kind == "chamfer":
        c = min(w, h) * rng.uniform(0.15, 0.4)
        return [(x0, y0), (x1, y0), (x1, y1 - c), (x1 - c, y1), (x0, y1)]
    angle = rng.uniform(0.15, 1.4)
    return _rotated_rect(cx, cy, w * 0.7, h * 0.7, angle)


def random_layout(
    n_features: int,
    seed: int = 0,
    route: str = "A",
    grid: Grid = Grid(),
    pitch: float = 260.0,
    overlap_fraction: float = 0.25,
) -> ModelSpec:
    """N features, each in its own ground cutout; a fraction get a second overlapping same-Net shape."""
    rng = random.Random(seed)
    cols = max(1, math.ceil(math.sqrt(n_features)))
    half = cols * pitch / 2
    z0, z1 = (z_value(0), z_value(0)) if route == "A" else (z_value(0), z_value("0.2"))
    prisms = _chip_prisms(grid, half)
    holes: list[list[Um]] = []
    features: list[PrismRecord] = []
    for k in range(n_features):
        col, row = k % cols, k // cols
        cx = -half + (col + 0.5) * pitch
        cy = -half + (row + 0.5) * pitch
        size = pitch * 0.45
        outline = _feature_outline(rng, cx, cy, size)
        net = f"net{k}"
        features.append(PrismRecord(f"f{k}", _footprint(grid, outline), z0, z1, "Al", "conductor", net_id=net))
        if rng.random() < overlap_fraction:
            extra = _feature_outline(rng, cx + size * rng.uniform(-0.2, 0.2), cy + size * rng.uniform(-0.2, 0.2), size * 0.7)
            features.append(PrismRecord(f"f{k}b", _footprint(grid, extra), z0, z1, "Al", "conductor", net_id=net))
            outline = outline + extra
        xs = [p[0] for p in outline]
        ys = [p[1] for p in outline]
        margin = pitch * 0.08
        holes.append(rect_um(min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin))
    ground = _footprint(grid, rect_um(-half, -half, half, half), holes)
    prisms.append(PrismRecord("ground", ground, z0, z1, "Al", "conductor", net_id="Ground"))
    prisms.extend(features)
    return ModelSpec(grid, tuple(prisms), name=f"random_{n_features}_{seed}_{route}", meta={"route": route})
