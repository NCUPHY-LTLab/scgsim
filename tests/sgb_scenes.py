"""Model builders for the SGB tests: hand-made prism catalogues in grid units.

These build :class:`PrismCatalogue` records directly, the way the arrangement stages
see them after stage S2, so the noding, labelling and dissolution tests do not depend
on the source adapter.  Tests of stage S2 itself build real ``GeometryBuildInput``
objects in ``sgb_inputs``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from fractions import Fraction

from scgsim.geometry.exact.carriers import ArcCarrier, Lineage, circle_arcs
from scgsim.geometry.exact.grid import Grid, exact_z
from scgsim.geometry.exact.plane import (
    Point,
    segment_point_distance_sq,
    segments_properly_cross,
)
from scgsim.geometry.sgb.arrangement.arcs import face_exact_area
from scgsim.geometry.sgb.arrangement.noding.snap_rounding import NodedEdge, Segment
from scgsim.geometry.sgb.arrangement.noding.spatial_hash import HotGrid
from scgsim.geometry.sgb.model import ConformalModel
from scgsim.geometry.sgb.records.prisms import (
    Footprint,
    LineEdge,
    PortIdentity,
    PrismCatalogue,
    PrismRecord,
)

GRID = Grid(0.001)
Um = tuple[float, float]


def units(value_um: float, grid: Grid = GRID) -> int:
    """Lattice coordinate of a micrometre length."""
    return math.floor(value_um / grid.size_um + 0.5)


def to_grid(points: list[Um], grid: Grid = GRID) -> list[Point]:
    """Micrometre points as lattice points."""
    return [(units(x, grid), units(y, grid)) for x, y in points]


def rect_um(x0: float, y0: float, x1: float, y1: float) -> list[Um]:
    """Counter-clockwise rectangle corners."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def footprint(
    outer: list[Um], holes: Sequence[list[Um]] = (), grid: Grid = GRID
) -> Footprint:
    """A polygonal footprint from micrometre rings."""
    return Footprint.from_rings(to_grid(outer, grid), [to_grid(h, grid) for h in holes])


def chip_prisms(grid: Grid, half: float) -> list[PrismRecord]:
    """Substrate below z = 0 and vacuum above, both ``2 * half`` wide."""
    box = footprint(rect_um(-half, -half, half, half), grid=grid)
    return [
        PrismRecord(
            "substrate", box, exact_z(-100), exact_z(0), "substrate", "dielectric"
        ),
        PrismRecord("vacuum", box, exact_z(0), exact_z(100), "vacuum", "vacuum"),
    ]


def conductor(  # noqa: PLR0913 - a test builder
    entity: str,
    outer: list[Um],
    z0: str | int,
    z1: str | int,
    net: str,
    grid: Grid = GRID,
    priority: int = 0,
) -> PrismRecord:
    """A conductor prism with a rectangular-or-polygonal footprint."""
    return PrismRecord(
        entity,
        footprint(outer, grid=grid),
        exact_z(z0),
        exact_z(z1),
        "Al",
        "conductor",
        net_id=net,
        priority=priority,
    )


def slab_model(  # noqa: PLR0913 - a test builder
    route: str,
    rects: list[tuple[str, str, tuple[float, float, float, float]]],
    grid: Grid = GRID,
    half: float = 40.0,
) -> PrismCatalogue:
    """Chip with conductor rectangles ``(entity, net, box)`` as sheets (A) or 0.2 um metal (B)."""
    z1 = 0 if route == "A" else "0.2"
    prisms = chip_prisms(grid, half)
    for entity, net, box in rects:
        prisms.append(conductor(entity, rect_um(*box), 0, z1, net, grid))
    return PrismCatalogue(grid, tuple(prisms), name=f"model_{route}")


LINEAGE = Lineage("polygon_edge", "disc")


def disc_model(
    radius: int, extra_rect: tuple[int, int, int, int] | None = None
) -> PrismCatalogue:
    """A circular sheet conductor (four quarter arcs), optionally overlapped by a same-Net rectangle."""
    prisms = chip_prisms(GRID, 10.0)
    arcs = circle_arcs("disc", (0, 0), radius, True, LINEAGE)
    footprint_, carried = Footprint.from_primitives(arcs)
    prisms.append(
        PrismRecord(
            "disc",
            footprint_,
            exact_z(0),
            exact_z(0),
            "Al",
            "conductor",
            net_id="N",
            arcs=carried,
        )
    )
    if extra_rect:
        x0, y0, x1, y1 = extra_rect
        bar = Footprint.from_rings([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        prisms.append(
            PrismRecord(
                "bar",
                bar,
                exact_z(0),
                exact_z(0),
                "Al",
                "conductor",
                net_id="N",
                priority=1,
            )
        )
    return PrismCatalogue(GRID, tuple(prisms), name="disc")


# --------------------------------------------------------------------------- the running example


def _pad_ring(rounded_corner: bool) -> list[LineEdge | ArcCarrier]:
    """The pad outline (CCW): chamfer, 20x15 notch and, optionally, a 5 um rounded corner."""

    def pt(x: float, y: float) -> Point:
        """Pt."""
        return (units(x), units(y))

    corner = -45.0 if rounded_corner else -50.0
    ring: list[LineEdge | ArcCarrier] = [
        LineEdge(pt(corner, -30), pt(-10, -30), 0),
        LineEdge(pt(-10, -30), pt(-10, -15), 1),
        LineEdge(pt(-10, -15), pt(10, -15), 2),
        LineEdge(pt(10, -15), pt(10, -30), 3),
        LineEdge(pt(10, -30), pt(50, -30), 4),
        LineEdge(pt(50, -30), pt(50, 18), 5),
        LineEdge(pt(50, 18), pt(38, 30), 6),
        LineEdge(pt(38, 30), pt(-50, 30), 7),
    ]
    if not rounded_corner:
        ring.append(LineEdge(pt(-50, 30), pt(-50, -30), 8))
        return ring
    ring.append(LineEdge(pt(-50, 30), pt(-50, -25), 8))
    centre = pt(-45, -25)
    ring.append(
        ArcCarrier(
            "pad/corner",
            (Fraction(centre[0]), Fraction(centre[1])),
            Fraction(units(5)),
            True,
            (Fraction(units(-50)), Fraction(units(-25))),
            (Fraction(units(-45)), Fraction(units(-30))),
            Lineage("polygon_edge", "pad", 0, 0, 9),
        )
    )
    return ring


def _polygon_hole(sides: int) -> list[LineEdge | ArcCarrier]:
    """The pad's hole as a regular polygon (clockwise, as a hole) instead of a true circle."""
    centre = (-25.0, 5.0)
    points = [
        (
            units(centre[0] + 8 * math.cos(-2 * math.pi * k / sides)),
            units(centre[1] + 8 * math.sin(-2 * math.pi * k / sides)),
        )
        for k in range(sides)
    ]
    return [LineEdge(points[k], points[(k + 1) % sides], k) for k in range(sides)]


def pad_in_ground(
    route: str = "A",
    *,
    with_port: bool = True,
    rounded_corner: bool = True,
    hole_sides: int | None = None,
) -> PrismCatalogue:
    """Ground with a cutout, a pad (chamfer, notch, rounded corner, circular hole) and a lumped port.

    ``hole_sides`` replaces the true circle by a regular polygon.
    """
    half = 150.0
    z0, z1 = exact_z(0), exact_z(0 if route == "A" else "0.2")
    prisms = chip_prisms(GRID, half)
    ground = footprint(rect_um(-half, -half, half, half), [rect_um(-80, -60, 80, 60)])
    prisms.append(
        PrismRecord("ground", ground, z0, z1, "Al", "conductor", net_id="Ground")
    )
    hole: list[LineEdge | ArcCarrier] = (
        _polygon_hole(hole_sides)
        if hole_sides
        else list(
            circle_arcs(
                "pad/hole",
                (units(-25), units(5)),
                units(8),
                False,
                Lineage("polygon_edge", "pad", 0, 1, 0),
            )
        )
    )
    pad_footprint, arcs = Footprint.from_primitives(_pad_ring(rounded_corner), [hole])
    prisms.append(
        PrismRecord(
            "pad", pad_footprint, z0, z1, "Al", "conductor", net_id="pad", arcs=arcs
        )
    )
    if with_port:
        prisms.append(lumped_port("p1", rect_um(50, -5, 80, 5), route))
    return PrismCatalogue(GRID, tuple(prisms), name=f"pad_in_ground_{route}")


def lumped_port(name: str, outline: list[Um], route: str) -> PrismRecord:
    """A lumped-port sheet on the sheet plane (Route A) or in the 0.2 um metal's mid-plane (Route B).

    A stage-level scene: the arrangement accepts a sheet at any level, and the mid-plane one exercises an
    embedded sheet.  The compiler's own ports (``prisms.ports``) lie on the hosts' support plane instead.
    """
    plane = exact_z(0) if route == "A" else exact_z("0.1")
    identity = PortIdentity(
        name, name.upper(), 1, (1.0, 0.0, 0.0), ("pad", "ground"), "metal", "port"
    )
    return PrismRecord(
        name, footprint(outline), plane, plane, "lumped_port", "port", port=identity
    )


def airbridge() -> PrismCatalogue:
    """Two ground halves split by a gap, bridged by two posts and a deck (finite metal, five slabs)."""
    half = 150.0
    prisms = chip_prisms(GRID, half)
    prisms += [
        conductor("ground_left", rect_um(-half, -half, -20, half), 0, "0.2", "Ground"),
        conductor("ground_right", rect_um(20, -half, half, half), 0, "0.2", "Ground"),
        conductor("post_left", rect_um(-35, -10, -25, 10), "0.2", 3, "Ground"),
        conductor("post_right", rect_um(25, -10, 35, 10), "0.2", 3, "Ground"),
        conductor("deck", rect_um(-35, -10, 35, 10), 3, "3.3", "Ground"),
    ]
    return PrismCatalogue(GRID, tuple(prisms), name="airbridge")


# --------------------------------------------------------------------------- random layouts


def _rotated_rect(cx: float, cy: float, w: float, h: float, angle: float) -> list[Um]:
    c, s = math.cos(angle), math.sin(angle)
    corners = ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in corners]


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
        return [
            (x0, y0),
            (x1, y0),
            (x1, y0 + ny),
            (x0 + nx, y0 + ny),
            (x0 + nx, y1),
            (x0, y1),
        ]
    if kind == "chamfer":
        c = min(w, h) * rng.uniform(0.15, 0.4)
        return [(x0, y0), (x1, y0), (x1, y1 - c), (x1 - c, y1), (x0, y1)]
    return _rotated_rect(cx, cy, w * 0.7, h * 0.7, rng.uniform(0.15, 1.4))


def random_layout(
    count: int,
    seed: int = 0,
    route: str = "A",
    pitch: float = 260.0,
    overlap_fraction: float = 0.25,
) -> PrismCatalogue:
    """``count`` features, each in its own ground cutout; a fraction get an overlapping same-Net shape."""
    rng = random.Random(seed)
    cols = max(1, math.ceil(math.sqrt(count)))
    half = cols * pitch / 2
    z1 = 0 if route == "A" else "0.2"
    prisms = chip_prisms(GRID, half)
    holes: list[list[Um]] = []
    features: list[PrismRecord] = []
    for k in range(count):
        col, row = k % cols, k // cols
        cx, cy = -half + (col + 0.5) * pitch, -half + (row + 0.5) * pitch
        size = pitch * 0.45
        outline = _feature_outline(rng, cx, cy, size)
        net = f"net{k}"
        features.append(conductor(f"f{k}", outline, 0, z1, net))
        if rng.random() < overlap_fraction:
            extra = _feature_outline(
                rng,
                cx + size * rng.uniform(-0.2, 0.2),
                cy + size * rng.uniform(-0.2, 0.2),
                size * 0.7,
            )
            features.append(conductor(f"f{k}b", extra, 0, z1, net, priority=1))
            outline = outline + extra
        xs, ys = [p[0] for p in outline], [p[1] for p in outline]
        margin = pitch * 0.08
        holes.append(
            rect_um(
                min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin
            )
        )
    ground = footprint(rect_um(-half, -half, half, half), holes)
    prisms.append(
        PrismRecord(
            "ground",
            ground,
            exact_z(0),
            exact_z(z1),
            "Al",
            "conductor",
            net_id="Ground",
        )
    )
    prisms.extend(features)
    return PrismCatalogue(GRID, tuple(prisms), name=f"random_{count}_{seed}_{route}")


# --------------------------------------------------------------------------- exact brute-force checks


def random_segments(seed: int, count: int, extent: int) -> list[Segment]:
    """Random integer segments with ids ``s0 ...``."""
    rng = random.Random(seed)
    return [
        Segment(
            (rng.randint(0, extent), rng.randint(0, extent)),
            (rng.randint(0, extent), rng.randint(0, extent)),
            frozenset([f"s{i}"]),
        )
        for i in range(count)
    ]


def crossing_pairs(edges: list[NodedEdge]) -> int:
    """Brute-force count of edge pairs crossing in both interiors (exact)."""
    return sum(
        1
        for i in range(len(edges))
        for j in range(i + 1, len(edges))
        if segments_properly_cross(edges[i].a, edges[i].b, edges[j].a, edges[j].b)
    )


def pass_through_vertices(edges: list[NodedEdge]) -> int:
    """Vertices lying inside the hot pixel of an edge that does not end there."""
    vertices = {p for e in edges for p in (e.a, e.b)}
    grid = HotGrid(vertices)
    return sum(1 for e in edges for c in grid.touching(e.a, e.b) if c not in (e.a, e.b))


def max_deviation(edges: list[NodedEdge], segments: list[Segment]) -> float:
    """Largest distance (grid units) from an output edge end to the nearest of its carriers."""
    by_id = {next(iter(s.carriers)): s for s in segments}
    worst = 0.0
    for e in edges:
        best = min(
            max(
                segment_point_distance_sq(by_id[c].a, by_id[c].b, e.a),
                segment_point_distance_sq(by_id[c].a, by_id[c].b, e.b),
            )
            for c in e.carriers
        )
        worst = max(worst, math.sqrt(float(best)))
    return worst


def exact_band_areas(
    model: ConformalModel, family_id: str, *, include_ports: bool = True
) -> dict[int, float]:
    """Area (grid^2) of every band of a family from the level arrangements, arcs realised exactly.

    A port face is no band's surface and carries band 0 nominally (its band seams stop at its boundary);
    ``include_ports=False`` leaves the port faces out, so the bands are what the groups carry.
    """
    areas: dict[int, float] = {}
    for layer in model.labelled.level_layers:
        for fi in range(1, len(layer.arr.faces)):
            sheet = layer.labels[fi].sheet
            if not include_ports and sheet is not None and sheet.kind == "port":
                continue
            for family, band in layer.labels[fi].bands:
                if family == family_id:
                    areas[band] = areas.get(band, 0.0) + face_exact_area(
                        layer.arr, layer.edge_curve, fi
                    )
    return areas
