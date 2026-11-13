"""Shared kit of the Supported Geometry sample set: lattice drawing, scenes, catalogues and reference bands.

The sample set (``sgb_supported_cells`` and ``sgb_supported_pdk_cells``) draws every unit cell SCGSim supports
directly as SGB records in lattice coordinates (grid 1 nm), the way ``sgb_scenes`` builds its scenes: no
gdsfactory, KQCircuits or PDK code.  This module owns what the cells share:

* drawing on the lattice (micrometre points rounded once, polygon circles, fillets, offset paths);
* the :class:`Scene` record and its Route B catalogue (0.2 um metal on a substrate at z = 0);
* the Surface-EPR requests: MA, MS and SA bands at ``inset_margins_um = (0.5, 1.0)``, asked for by type;
* the measured band areas of a built model and their reference, the exact erosion of the same lattice
  polygons (shapely), with the snap bound each band must meet.

MA is the metal-to-air surface (a conductor face toward vacuum), MS the metal-to-substrate surface and SA the
substrate-to-air surface.  A band is measured from gap edges only: the edges of a conductor footprint that face
exposed substrate.  The die edge, a port and a contact with another conductor are no distance origins.  The die
edge is a dicing line: a gap edge that runs onto it is measured as if its conductor continued past it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from shapely.geometry import LineString, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from scgsim.geometry.sgb.arrangement.arcs import face_exact_area
from scgsim.geometry.sgb.bands.requests import InsetRequest, InterfaceSurfaces
from scgsim.geometry.sgb.exact.grid import Grid, exact_z
from scgsim.geometry.sgb.exact.prisms import (
    Footprint,
    PortIdentity,
    PrismCatalogue,
    PrismRecord,
)
from scgsim.geometry.sgb.model import ConformalModel, build_model_from_catalogue
from scgsim.geometry.sgb.output.sidecar.lineage import CarrierLineage
from scgsim.geometry.sgb.topology.routes.naming import interface_type_of

GRID = Grid(0.001)
MARGINS = (0.5, 1.0)
METAL_TOP = "0.2"
SUBSTRATE_SA = "sa::substrate|vacuum::z=0.0"
# Each band contour is a chain of lattice segments within half a grid diagonal of the exact contour, so a
# band bounded by contours of total length L differs from the exact area by at most L * sqrt(2)/2 * grid
# (the bound of test_oblique_edges_give_the_hand_calculated_bands_within_the_snap_bound).
CONTOUR_SNAP_UM = math.sqrt(2) / 2 * GRID.size_um
# Double-precision sums of face areas: a band that vanishes comes out as +-4e-16 um^2 rather than 0, which a
# zero snap bound (no contour) must still accept.  Relative to the area, far below any snap bound.
FLOAT_ROUNDOFF = 1e-12
# The stadium end caps: 128 chords per quarter circle fall short of a quarter disk of radius 1 um by 2e-5 um^2,
# under 2 % of the snap bound of the 1.57 um arc that bounds it (1.1e-3 um^2).
BUFFER_QUAD_SEGMENTS = 128

Um = tuple[float, float]
Point = tuple[int, int]


# --------------------------------------------------------------------------- drawing on the lattice


def lattice(value_um: float) -> int:
    """Lattice coordinate of a micrometre length, rounded half up once."""
    return math.floor(value_um / GRID.size_um + 0.5)


def to_lattice(points: list[Um]) -> list[Point]:
    """Micrometre points as lattice points, consecutive repeats dropped."""
    out: list[Point] = []
    for x, y in points:
        point = (lattice(x), lattice(y))
        if not out or point != out[-1]:
            out.append(point)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def polar(r: float, degrees: float, centre: Um = (0.0, 0.0)) -> Um:
    """The point at radius ``r`` and angle ``degrees`` (counter-clockwise from +x) about ``centre``."""
    a = math.radians(degrees)
    return (centre[0] + r * math.cos(a), centre[1] + r * math.sin(a))


def rotate(point: Um, degrees: float) -> Um:
    """``point`` rotated about the origin."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return (c * point[0] - s * point[1], s * point[0] + c * point[1])


def quarter_turn(points: list[Point], quarters: int) -> list[Point]:
    """Exact lattice rotation by ``quarters`` x 90 degrees."""
    for _ in range(quarters % 4):
        points = [(-y, x) for x, y in points]
    return points


def mirror_y(points: list[Um]) -> list[Um]:
    """The mirror image in the x axis."""
    return [(x, -y) for x, y in points]


def intersect(p1: Um, p2: Um, p3: Um, p4: Um) -> Um:
    """The intersection of the lines p1-p2 and p3-p4."""
    d = (p2[0] - p1[0]) * (p4[1] - p3[1]) - (p2[1] - p1[1]) * (p4[0] - p3[0])
    t = ((p3[0] - p1[0]) * (p4[1] - p3[1]) - (p3[1] - p1[1]) * (p4[0] - p3[0])) / d
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))


def rect(x0: float, y0: float, x1: float, y1: float) -> list[Um]:
    """Counter-clockwise rectangle corners."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def ngon(r: float, n: int, centre: Um = (0.0, 0.0), phase: float = 0.0) -> list[Um]:
    """A polygon circle: ``n`` vertices on radius ``r``, at ``phase + k * 360 / n`` degrees."""
    return [polar(r, phase + 360.0 * k / n, centre) for k in range(n)]


def chord(r: float, degrees0: float, degrees1: float) -> tuple[Um, Um]:
    """The chord of the circle of radius ``r`` between two angles."""
    return polar(r, degrees0), polar(r, degrees1)


def round_cap(
    centre: Um, radius: float, facing: float, inscribed_steps: bool
) -> list[Um]:
    """The half of a 16-gon cap that faces ``facing`` degrees, from ``facing - 90`` to ``facing + 90``.

    ``inscribed_steps``: vertices at ``facing + k * 22.5`` (a circle ORed onto a strip end); otherwise at
    ``facing +- 11.25 + k * 22.5`` with half steps at both ends (a strip end with rounded corners).
    """
    if inscribed_steps:
        angles = [facing - 90 + 22.5 * k for k in range(9)]
    else:
        angles = [
            facing - 90,
            *(facing - 78.75 + 22.5 * k for k in range(8)),
            facing + 90,
        ]
    return [polar(radius, a, centre) for a in angles]


def fillet(
    previous: Um, corner: Um, following: Um, radius: float, segments: int
) -> list[Um]:
    """A corner replaced by ``segments`` chords of the tangent arc of ``radius`` (vertices on the arc)."""
    a = math.atan2(previous[1] - corner[1], previous[0] - corner[0])
    b = math.atan2(following[1] - corner[1], following[0] - corner[0])
    alpha = abs((b - a + math.pi) % (2 * math.pi) - math.pi)
    tangent = radius / math.tan(alpha / 2)
    t1 = (corner[0] + tangent * math.cos(a), corner[1] + tangent * math.sin(a))
    t2 = (corner[0] + tangent * math.cos(b), corner[1] + tangent * math.sin(b))
    bisector = math.atan2(math.sin(a) + math.sin(b), math.cos(a) + math.cos(b))
    reach = radius / math.sin(alpha / 2)
    centre = (
        corner[0] + reach * math.cos(bisector),
        corner[1] + reach * math.sin(bisector),
    )
    start = math.atan2(t1[1] - centre[1], t1[0] - centre[0])
    end = math.atan2(t2[1] - centre[1], t2[0] - centre[0])
    sweep = (end - start + math.pi) % (2 * math.pi) - math.pi
    return [
        polar(radius, math.degrees(start + sweep * k / segments), centre)
        for k in range(segments + 1)
    ]


def offset_path(path: list[Um], d: float) -> list[Um]:
    """The edge ``d`` um to the left of a sampled centreline, as an extruded cross-section draws it.

    The two end vertices lie exactly ``d`` from the centreline; an interior vertex lies on the miter, ``d /
    cos(turn / 2)`` along the bisector of its two normals.
    """
    out = []
    for i, point in enumerate(path):
        normals = []
        if i > 0:
            normals.append(_left_normal(path[i - 1], point))
        if i < len(path) - 1:
            normals.append(_left_normal(point, path[i + 1]))
        nx, ny = (sum(n[0] for n in normals), sum(n[1] for n in normals))
        length = math.hypot(nx, ny)
        nx, ny = nx / length, ny / length
        scale = d / (nx * normals[0][0] + ny * normals[0][1])
        out.append((point[0] + scale * nx, point[1] + scale * ny))
    return out


def _left_normal(a: Um, b: Um) -> Um:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    return (-dy / length, dx / length)


def turns_degrees(points: list[Point]) -> list[float]:
    """Signed turn (degrees, counter-clockwise positive) at every vertex of a closed lattice ring."""
    out = []
    for i, (bx, by) in enumerate(points):
        ax, ay = points[i - 1]
        cx, cy = points[(i + 1) % len(points)]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        dot = (bx - ax) * (cx - bx) + (by - ay) * (cy - by)
        out.append(math.degrees(math.atan2(cross, dot)))
    return out


# --------------------------------------------------------------------------- scenes


@dataclass
class Conductor:
    """One conductor prism: lattice rings, its Net and its z interval (Route B metal by default)."""

    entity: str
    net: str
    outer: list[Point]
    holes: list[list[Point]] = field(default_factory=list)
    z: tuple[str, str] = ("0", METAL_TOP)


@dataclass
class Port:
    """A lumped-port sheet: lattice outline, its hosts and its plane (the metal mid-plane by default)."""

    name: str
    outline: list[Point]
    hosts: tuple[str, str]
    plane: str = "0.1"
    direction: tuple[float, float, float] = (0.0, 1.0, 0.0)


# The reference of one band family: its support face and the region whose boundary is the distance origin
# (exposed substrate for a conductor face, metal for SA), both as micrometre shapely geometry.  Where the origin
# is no region's boundary (beside a contact between two conductors), it is given as the gap edges themselves.
Expectation = tuple[BaseGeometry, BaseGeometry]


@dataclass
class Scene:
    """A sample layout: conductors on one die box, optional ports, and how its reference is formed.

    ``prisms`` replaces the default substrate and vacuum (multi-level scenes).  ``expectations`` replaces the
    single-level reference of :func:`planar_expectations`; an empty dict means the scene has no band family.
    """

    name: str
    cells: tuple[int, ...]
    die: tuple[float, float, float, float]
    conductors: list[Conductor]
    ports: list[Port] = field(default_factory=list)
    prisms: list[PrismRecord] | None = None
    expectations: Callable[[Scene], dict[str, Expectation]] | None = None


def die_box(half: float) -> tuple[float, float, float, float]:
    """A square die centred on the origin."""
    return (-half, -half, half, half)


def ground(
    entity: str,
    die: tuple[float, float, float, float],
    holes: list[list[Point]],
    net: str = "Ground",
) -> Conductor:
    """A ground plane covering the die box, with holes."""
    return Conductor(entity, net, to_lattice(rect(*die)), holes)


# --------------------------------------------------------------------------- catalogue and bands


def die_footprint(scene: Scene) -> Footprint:
    """The die box as a lattice footprint."""
    x0, y0, x1, y1 = (lattice(v) for v in scene.die)
    return Footprint.from_rings([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def catalogue(scene: Scene) -> PrismCatalogue:
    """Route B prisms of a scene: substrate below z = 0, vacuum above, the conductors and the port sheets."""
    die = die_footprint(scene)
    prisms = (
        list(scene.prisms)
        if scene.prisms is not None
        else [
            PrismRecord(
                "substrate", die, exact_z(-100), exact_z(0), "substrate", "dielectric"
            ),
            PrismRecord("vacuum", die, exact_z(0), exact_z(100), "vacuum", "vacuum"),
        ]
    )
    for c in scene.conductors:
        prisms.append(
            PrismRecord(
                c.entity,
                Footprint.from_rings(c.outer, c.holes),
                exact_z(c.z[0]),
                exact_z(c.z[1]),
                "Al",
                "conductor",
                net_id=c.net,
            )
        )
    for p in scene.ports:
        identity = PortIdentity(
            p.name, p.name.upper(), 1, p.direction, p.hosts, "metal", "port"
        )
        plane = exact_z(p.plane)
        prisms.append(
            PrismRecord(
                p.name,
                Footprint.from_rings(p.outline),
                plane,
                plane,
                "lumped_port",
                "port",
                port=identity,
            )
        )
    return PrismCatalogue(GRID, tuple(prisms), name=scene.name)


def type_bands() -> tuple[InsetRequest, ...]:
    """MA, MS and SA bands at (0.5, 1.0) um, asked for by type."""
    return tuple(
        InsetRequest(t.lower(), InterfaceSurfaces(t), MARGINS)
        for t in ("MA", "MS", "SA")
    )


def band_areas(model: ConformalModel) -> dict[str, list[float]]:
    """Area (um^2) of every band of every inset family, as its band groups carry it, arcs realised exactly.

    Only the cells of the family's own interface type count: a port face is no band's surface, and a cell
    where one conductor stands on another (metal to metal) carries the band label but belongs to no band group.
    """
    grid_area = model.labelled.catalogue.grid.size_um**2
    out: dict[str, list[float]] = {}
    for layer in model.labelled.level_layers:
        for fi in range(1, len(layer.arr.faces)):
            label = layer.labels[fi]
            row_type = interface_type_of(label.below, label.above, label.sheet)
            for family, band in label.bands:
                if family.split("::", 1)[0].upper() != row_type:
                    continue
                area = face_exact_area(layer.arr, layer.edge_curve, fi) * grid_area
                out.setdefault(family, [0.0] * (len(MARGINS) + 1))[band] += area
    return out


@dataclass(frozen=True)
class ReferenceBands:
    """The exact band areas of one family (um^2) and the snap bound each measured area must meet."""

    areas: tuple[float, ...]
    bounds: tuple[float, ...]


def reference(scene: Scene) -> dict[str, ReferenceBands]:
    """Exact erosion of the scene's lattice polygons, one entry per band family the build must produce."""
    expectations = (scene.expectations or planar_expectations)(scene)
    die = die_shape(scene)
    return {
        family: exact_bands(
            support, continued_past_the_die(gap_edges(origin, die.boundary), die)
        )
        for family, (support, origin) in expectations.items()
    }


def gap_edges(origin: BaseGeometry, die_edge: BaseGeometry) -> BaseGeometry:
    """The distance origins of a band family: the origin region's boundary (or the given lines), less the die edge."""
    lines = (
        origin
        if origin.geom_type in ("LineString", "MultiLineString")
        else origin.boundary
    )
    return lines.difference(die_edge)


def continued_past_the_die(edges: BaseGeometry, die: Polygon) -> list[tuple[Um, Um]]:
    """The segments of the gap ``edges``, each one that the die edge cuts continued straight past it.

    The die edge is a dicing line, so a conductor it cuts continues past it.  A segment whose end lies on the die
    edge and ends a run of gap edges (no other segment meets it there) is continued along its own line to the box
    two largest margins outside the die: a point on that box lies farther than every margin from the die, so the
    stadium of the continued segment enters the die through its straight sides only, as the continuing line's
    neighbourhood does.  The bands keep only the part inside the support, which lies in the die.
    """
    pairs = segment_pairs(edges)
    meeting = Counter(point for pair in pairs for point in pair)
    x0, y0, x1, y1 = die.bounds
    reach = 2 * MARGINS[-1]
    outside = (x0 - reach, y0 - reach, x1 + reach, y1 + reach)

    def cut_end(point: Um) -> bool:
        x, y = point
        on_edge = (x in (x0, x1) and y0 <= y <= y1) or (y in (y0, y1) and x0 <= x <= x1)
        return meeting[point] == 1 and on_edge

    continued = []
    for a, b in pairs:
        if cut_end(b):
            b = _to_the_box(a, b, outside)
        if cut_end(a):
            a = _to_the_box(b, a, outside)
        continued.append((a, b))
    return continued


def _to_the_box(inner: Um, end: Um, outside: tuple[float, float, float, float]) -> Um:
    """``end`` moved on along the line from ``inner`` until it reaches the boundary of the ``outside`` box."""
    dx, dy = end[0] - inner[0], end[1] - inner[1]
    x0, y0, x1, y1 = outside
    steps = []
    if dx:
        steps.append(((x1 if dx > 0 else x0) - end[0]) / dx)
    if dy:
        steps.append(((y1 if dy > 0 else y0) - end[1]) / dy)
    t = min(steps)
    return (end[0] + t * dx, end[1] + t * dy)


def exact_bands(support: BaseGeometry, edges: list[tuple[Um, Um]]) -> ReferenceBands:
    """Band k of ``support``: the part within ``m_k`` of the gap ``edges`` and not within ``m_(k-1)``; band K the rest.

    The points within ``m`` of the edge segments are the union of one stadium per segment, which is the
    definition of that distance.  A stadium's straight sides are exact and only its end caps are polygonised,
    from inside, so a band that erodes away is covered by straight sides and has no reference area.  (Buffering
    a closed ring as a whole is not used: when its inner offset collapses, GEOS leaves a small hole, about
    0.0012 um^2 in a 16-gon of radius 1 um at a 1 um margin.)
    """
    segments = [LineString(pair) for pair in edges]
    grown = [
        unary_union([seg.buffer(m, quad_segs=BUFFER_QUAD_SEGMENTS) for seg in segments])
        for m in MARGINS
    ]
    near = [support.intersection(g).area for g in grown]
    contours = [g.boundary.intersection(support).length for g in grown]
    areas = (
        near[0],
        *(near[k] - near[k - 1] for k in range(1, len(near))),
        support.area - near[-1],
    )
    lengths = (
        contours[0],
        *(contours[k - 1] + contours[k] for k in range(1, len(contours))),
        contours[-1],
    )
    return ReferenceBands(areas, tuple(length * CONTOUR_SNAP_UM for length in lengths))


def segment_pairs(lines: BaseGeometry) -> list[tuple[Um, Um]]:
    """Every straight segment of a line geometry (a line string, a ring or a collection of them)."""
    parts = getattr(lines, "geoms", [lines])
    pairs = []
    for part in parts:
        coords = list(part.coords)
        pairs += list(zip(coords, coords[1:], strict=False))
    return pairs


def assert_exact_bands(scene: Scene) -> ConformalModel:
    """The build has the reference's band families, each band within its snap bound of the exact erosion.

    Returns the built model, for checks of the same build.
    """
    model = build_model_from_catalogue(catalogue(scene), "B", insets=type_bands())
    got = band_areas(model)
    expected = reference(scene)
    assert sorted(got) == sorted(expected)
    for family, ref in expected.items():
        for band, (area, exact, bound) in enumerate(
            zip(got[family], ref.areas, ref.bounds, strict=True)
        ):
            roundoff = FLOAT_ROUNDOFF * max(1.0, abs(exact))
            assert abs(area - exact) <= bound + roundoff, (
                family,
                band,
                area,
                exact,
                bound,
            )
    return model


def short_band_edges(model: ConformalModel, steps: int) -> list[str]:
    """The emitted horizontal edges that carry a band seam and are shorter than ``steps`` grid steps.

    Each is given as its length in grid steps, its two end points in um and the carriers it comes from.  An arc
    edge is measured along its arc.  A band seam is a carrier whose lineage is a band member.
    """
    topology = model.topology
    um = topology.grid_um
    lineage = CarrierLineage.of(model)
    found = []
    for edge in topology.edges.values():
        if edge.kind != "horizontal" or not any(
            (found_lineage := lineage.lineage(s)) is not None
            and found_lineage.band_member is not None
            for s in edge.sources
        ):
            continue
        a, b = topology.vertices[edge.v0], topology.vertices[edge.v1]
        length = math.hypot(a.x - b.x, a.y - b.y)
        if edge.curve is not None:
            length = (
                2
                * edge.curve.radius
                * math.asin(min(1.0, length / (2 * edge.curve.radius)))
            )
        if length < steps:
            found.append(
                f"{length:.1f} steps ({a.x * um:.4f}, {a.y * um:.4f})-"
                f"({b.x * um:.4f}, {b.y * um:.4f}) um {sorted(edge.sources)}"
            )
    return found


def shape(outer: list[Point], holes: Sequence[list[Point]] = ()) -> Polygon:
    """A lattice ring (with holes) as a micrometre shapely polygon."""
    um = GRID.size_um
    return Polygon(
        [(x * um, y * um) for x, y in outer],
        [[(x * um, y * um) for x, y in h] for h in holes],
    )


def die_shape(scene: Scene) -> Polygon:
    """The die box as a micrometre shapely polygon, on the lattice like the prisms."""
    um = GRID.size_um
    return box(*(lattice(v) * um for v in scene.die))


def conductor_shape(c: Conductor) -> Polygon:
    """A conductor's footprint as a micrometre shapely polygon."""
    return shape(c.outer, c.holes)


def planar_expectations(scene: Scene) -> dict[str, Expectation]:
    """Single-level scene: every conductor's top (MA) and bottom (MS) and the exposed substrate (SA).

    A conductor is banded from the exposed substrate around it, SA from the metal.  A port sheet on the
    substrate surface (z = 0) is no origin, and the part of it over the gap is a port face, not an SA face.
    """
    die = die_shape(scene)
    shapes = {c.entity: conductor_shape(c) for c in scene.conductors}
    metal = unary_union(list(shapes.values()))
    exposed = die.difference(metal)
    out: dict[str, Expectation] = {}
    for entity, polygon in shapes.items():
        out[f"ma::{entity}::top"] = (polygon, exposed)
        out[f"ms::{entity}::bottom"] = (polygon, exposed)
    surface_ports = [shape(p.outline) for p in scene.ports if p.plane == "0"]
    out[SUBSTRATE_SA] = (
        exposed.difference(unary_union(surface_ports)) if surface_ports else exposed,
        metal,
    )
    return out
