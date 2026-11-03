"""Catalogue of minimal extreme-case fixtures for the SGB model builder.

Every fixture is a handful of prisms (a few dozen faces) that targets exactly one risk, so the
structural validators stay cheap.  The catalogue is data: :data:`FIXTURES` maps a name to a
:class:`Fixture` (risk, builder, routes, expected failure per route).  Fixtures that must be
refused (a different-Net contact, a same-Net non-manifold pinch) carry the expected error class.

Inset-band fixtures belong to the cut-family sub-stage and are not part of this catalogue yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

from sgb_scenes import GRID, rect_um, to_grid

from scgsim.geometry.sgb.errors import ContactError, NonManifoldError
from scgsim.geometry.sgb.records.carriers import Lineage, LineCarrier, circle_arcs
from scgsim.geometry.sgb.records.families import CurveFamilyRecord
from scgsim.geometry.sgb.records.grid import exact_z
from scgsim.geometry.sgb.records.prisms import (
    Footprint,
    PortIdentity,
    PrismCatalogue,
    PrismRecord,
)

METAL_THICKNESS = "0.2"
Point = tuple[float, float]


def chip(half: float = 20.0, depth: float = 10.0) -> list[PrismRecord]:
    """Substrate below z = 0 and vacuum above, both ``2 half`` wide: a small box."""
    box = Footprint.from_rings(to_grid(rect_um(-half, -half, half, half)))
    return [
        PrismRecord(
            "substrate", box, exact_z(-depth), exact_z(0), "substrate", "dielectric"
        ),
        PrismRecord("vacuum", box, exact_z(0), exact_z(depth), "vacuum", "vacuum"),
    ]


def metal_z(route: str, base: float | str = 0) -> tuple[Fraction, Fraction]:
    """(z0, z1) of a conductor: a sheet at ``base`` in Route A, a 0.2 um prism in Routes B and C."""
    z0 = exact_z(base)
    return (z0, z0) if route == "A" else (z0, z0 + exact_z(METAL_THICKNESS))


def metal(  # noqa: PLR0913 - a fixture builder
    name: str,
    net: str,
    outer: list[Point],
    route: str,
    holes: tuple[list[Point], ...] = (),
    base: float | str = 0,
    z: tuple[Fraction, Fraction] | None = None,
    priority: int = 0,
) -> PrismRecord:
    """A conductor prism with a polygonal footprint."""
    z0, z1 = z or metal_z(route, base)
    footprint = Footprint.from_rings(to_grid(outer), [to_grid(h) for h in holes])
    return PrismRecord(
        name, footprint, z0, z1, "Al", "conductor", net_id=net, priority=priority
    )


def grid_metal(
    name: str, net: str, outline: list[tuple[int, int]], route: str
) -> PrismRecord:
    """A conductor whose outline is given in grid units (to place vertices 1, 2, 3 g apart)."""
    z0, z1 = metal_z(route)
    return PrismRecord(
        name, Footprint.from_rings(outline), z0, z1, "Al", "conductor", net_id=net
    )


def disk(  # noqa: PLR0913 - a fixture builder
    name: str,
    net: str,
    centre: tuple[int, int],
    radius: int,
    route: str,
    base: float | str = 0,
    priority: int = 0,
) -> PrismRecord:
    """A conductor disk of exact radius: four quarter arcs, no polygon approximation."""
    arcs = circle_arcs(
        f"{name}/c", centre, radius, True, Lineage("polygon_edge", name, 0, 0, 0)
    )
    footprint, carried = Footprint.from_primitives(arcs)
    z0, z1 = metal_z(route, base)
    return PrismRecord(
        name,
        footprint,
        z0,
        z1,
        "Al",
        "conductor",
        net_id=net,
        priority=priority,
        arcs=carried,
    )


def catalogue_of(
    name: str,
    route: str,
    prisms: list[PrismRecord],
    families: tuple[CurveFamilyRecord, ...] = (),
) -> PrismCatalogue:
    """The catalogue of a fixture under a route."""
    return PrismCatalogue(GRID, tuple(prisms), families, name=f"{name}_{route}")


def port(
    name: str, outline: list[Point], z: Fraction, hosts: tuple[str, ...] = ()
) -> PrismRecord:
    """A lumped-port sheet prism."""
    identity = PortIdentity(
        name, name.upper(), 1, (1.0, 0.0, 0.0), hosts, "metal", "port"
    )
    return PrismRecord(
        name,
        Footprint.from_rings(to_grid(outline)),
        z,
        z,
        "lumped_port",
        "port",
        port=identity,
    )


def refinement(
    name: str, p: tuple[int, int], q: tuple[int, int], z: Fraction
) -> CurveFamilyRecord:
    """A one-segment refinement-only family at height ``z``."""
    carrier = LineCarrier(
        f"{name}0", p, q, Lineage("family_member", family_id=name, member=f"{name}0")
    )
    return CurveFamilyRecord(name, "refinement", (carrier,), (z,), "refinement_only")


# --------------------------------------------------------------------------- fixtures


def baseline(route: str) -> PrismCatalogue:
    """Control: one rectangular pad on the chip."""
    return catalogue_of(
        "baseline", route, [*chip(), metal("pad", "N", rect_um(-8, -5, 8, 5), route)]
    )


def _tangent_pair(route: str, bar_bottom: float, name: str) -> PrismCatalogue:
    """A disk (r = 5 um) and a bar of the same Net whose lower edge cuts the circle ``5 - bar_bottom`` um deep."""
    bar = metal("bar", "N", rect_um(-12, bar_bottom, 12, 8), route, priority=1)
    return catalogue_of(
        name, route, [*chip(), disk("disk", "N", (0, 0), 5000, route), bar]
    )


def near_tangent_arc_line(route: str) -> PrismCatalogue:
    """Near-tangent arc and line: 2 nm secant depth, crossing angle 0.03 rad, chord 0.28 um."""
    return _tangent_pair(route, 4.998, "near_tangent_arc_line")


def exactly_tangent_arc_line(route: str) -> PrismCatalogue:
    """The bar touches the circle at one point (0, 5)."""
    return _tangent_pair(route, 5.0, "exactly_tangent_arc_line")


def arcs_crossing_shallow(route: str) -> PrismCatalogue:
    """Two same-Net disks (r = 5 um) whose centres are 9.999 um apart: the circles cross at about 1.6 degrees."""
    left = disk("left", "N", (-5000, 0), 5000, route)
    right = disk("right", "N", (4999, 0), 5000, route, priority=1)
    return catalogue_of("arcs_crossing_shallow", route, [*chip(), left, right])


def edges_1_to_3_g(route: str) -> PrismCatalogue:
    """A 10 um square whose right side carries jogs of 1, 2, 3 g and a 1.4 g diagonal."""
    outline = [
        (0, 0), (10000, 0), (10000, 4000), (10001, 4000), (10001, 4002), (10004, 4002), (10004, 6000),
        (10003, 6001), (10003, 6003), (10004, 6004), (10004, 10000), (0, 10000),
    ]  # fmt: skip
    shifted = [(x - 5000, y - 5000) for x, y in outline]
    return catalogue_of(
        "edges_1_to_3_g", route, [*chip(), grid_metal("jogged", "N", shifted, route)]
    )


def kissing_corners_same_net(route: str) -> PrismCatalogue:
    """Vertex-to-vertex contact of two same-Net pads."""
    return catalogue_of(
        "kissing_corners_same_net",
        route,
        [
            *chip(),
            metal("padL", "N", rect_um(-10, -10, 0, 0), route),
            metal("padR", "N", rect_um(0, 0, 10, 10), route),
        ],
    )


def kissing_corners_different_nets(route: str) -> PrismCatalogue:
    """Vertex-to-vertex contact of two different Nets."""
    return catalogue_of(
        "kissing_corners_different_nets",
        route,
        [
            *chip(),
            metal("padL", "N1", rect_um(-10, -10, 0, 0), route),
            metal("padR", "N2", rect_um(0, 0, 10, 10), route),
        ],
    )


def collinear_overlap(route: str) -> PrismCatalogue:
    """Three same-Net rectangles whose edges overlap collinearly (x = -4 and y = 4 each shared by a partial range)."""
    return catalogue_of(
        "collinear_overlap",
        route,
        [
            *chip(),
            metal("padL", "N", rect_um(-14, -6, -4, 4), route),
            metal("padR", "N", rect_um(-4, 0, 6, 10), route, priority=1),
            metal("padM", "N", rect_um(-10, 4, 0, 12), route, priority=2),
        ],
    )


def hole_touches_outer(route: str) -> PrismCatalogue:
    """A conductor whose diamond hole touches the outer boundary at the single point (10, 0)."""
    diamond = [(10.0, 0.0), (5.0, -5.0), (0.0, 0.0), (5.0, 5.0)]
    ring = metal("ring", "N", rect_um(-10, -10, 10, 10), route, (diamond,))
    return catalogue_of("hole_touches_outer", route, [*chip(), ring])


def _foreign_vertex_family(route: str) -> tuple[CurveFamilyRecord, ...]:
    """A refinement curve whose end point (4330, 2500) g lies 0.1 g from the circle r = 5000 g."""
    z = exact_z(0) if route == "A" else exact_z(METAL_THICKNESS)
    return (refinement("ref", (0, 0), (4330, 2500), z),)


def arc_with_foreign_vertex(route: str) -> PrismCatalogue:
    """The arc of a disk carries a vertex that belongs to another curve: a 5-edge cylinder wall before the split."""
    return catalogue_of(
        "arc_with_foreign_vertex",
        route,
        [*chip(), disk("disk", "N", (0, 0), 5000, route)],
        _foreign_vertex_family(route),
    )


def arc_with_foreign_vertex_from_post(route: str) -> PrismCatalogue:
    """Same, with the foreign vertex coming from the corner of a same-Net post standing on the disk."""
    post = metal(
        "post", "N", [(4.330, 2.5), (6.5, 2.5), (6.5, 6.0), (4.330, 6.0)], route,
        z=(exact_z(METAL_THICKNESS), exact_z(3)),
    )  # fmt: skip
    return catalogue_of(
        "arc_with_foreign_vertex_from_post",
        route,
        [*chip(), disk("disk", "N", (0, 0), 5000, route), post],
    )


def port_crossing_round_pad(route: str) -> PrismCatalogue:
    """A port sheet at the metal mid-plane whose edges cross the arc of a round pad twice (Route B: the part
    inside the excluded conductor is an embedded sheet)."""
    z = exact_z(0) if route == "A" else exact_z("0.1")
    lumped = port("p1", [(4.0, 2.0), (9.0, 2.0), (9.0, 4.0), (4.0, 4.0)], z)
    return catalogue_of(
        "port_crossing_round_pad",
        route,
        [*chip(), disk("disk", "N", (0, 0), 5000, route), lumped],
    )


def floating_conductor(route: str) -> PrismCatalogue:
    """A conductor enclosed by vacuum: Route B gives the vacuum an inner (void) shell, Route A an embedded sheet."""
    float_metal = metal("float", "floating", rect_um(-6, -4, 6, 4), route, base=3)
    return catalogue_of("floating_conductor", route, [*chip(), float_metal])


def nested_holes(route: str) -> PrismCatalogue:
    """Ring (hole +-8) around a ring island (+-6, hole +-3) around an island (+-1): three nested loops, three Nets."""
    return catalogue_of(
        "nested_holes",
        route,
        [
            *chip(),
            metal(
                "outer",
                "N1",
                rect_um(-14, -14, 14, 14),
                route,
                (rect_um(-8, -8, 8, 8),),
            ),
            metal(
                "middle", "N2", rect_um(-6, -6, 6, 6), route, (rect_um(-3, -3, 3, 3),)
            ),
            metal("core", "N3", rect_um(-1, -1, 1, 1), route),
        ],
    )


def _stack(route: str, name: str, upper: list[Point]) -> PrismCatalogue:
    lower = metal(
        "lower", "N", rect_um(-12, -6, -2, 4), route, z=(exact_z(0), exact_z(1))
    )
    top = metal("upper", "N", upper, route, z=(exact_z(1), exact_z(2)))
    return catalogue_of(name, route, [*chip(), lower, top])


def stacked_corner_touch(route: str) -> PrismCatalogue:
    """Two same-Net slabs stacked so that only their corners (-2, 4) meet at z = 1."""
    return _stack(route, "stacked_corner_touch", rect_um(-2, 4, 8, 14))


def stacked_vertex_on_wall_edge(route: str) -> PrismCatalogue:
    """The upper slab is a triangle whose apex (-2, 0) touches the lower slab's top edge x = -2 at one point."""
    return _stack(
        route, "stacked_vertex_on_wall_edge", [(-2.0, 0.0), (8.0, -5.0), (8.0, 5.0)]
    )


def port_edge_on_metal_sidewall(route: str) -> PrismCatalogue:
    """A lumped-port sheet at mid-height of a 0.2 um metal whose edge x = 8 runs along the metal sidewall partly."""
    ground = metal("ground", "G", rect_um(-16, -6, 8, 6), route)
    z = exact_z(0) if route == "A" else exact_z("0.1")
    lumped = port("p1", rect_um(8, -2, 14, 9), z)
    return catalogue_of("port_edge_on_metal_sidewall", route, [*chip(), ground, lumped])


def _layers_offset(route: str, name: str, offset_um: float) -> PrismCatalogue:
    lower = metal(
        "lower", "N", rect_um(-10, -4, 0, 4), route, z=(exact_z(0), exact_z(1))
    )
    upper = metal(
        "upper", "N", rect_um(-10, -4, offset_um, 4), route, z=(exact_z(1), exact_z(2))
    )
    return catalogue_of(name, route, [*chip(), lower, upper])


def near_coincident_layers_sub_grid(route: str) -> PrismCatalogue:
    """Near-coincident edges of two layers, 0.4 g apart: they merge, no sliver."""
    return _layers_offset(route, "near_coincident_layers_sub_grid", 0.0004)


def near_coincident_layers_2g(route: str) -> PrismCatalogue:
    """Near-coincident edges of two layers, 2 g apart: a real 2 nm exposed sliver of the level z = 1."""
    return _layers_offset(route, "near_coincident_layers_2g", 0.002)


# --------------------------------------------------------------------------- catalogue


@dataclass(frozen=True)
class Fixture:
    """One fixture: what it targets, how to build it and what must happen per route."""

    name: str
    risk: str
    build: Callable[[str], PrismCatalogue]
    routes: tuple[str, ...] = ("A", "B", "C")
    expect_error: dict[str, type[Exception]] = field(default_factory=dict)
    note: str = ""


FIXTURES: dict[str, Fixture] = {
    f.name: f
    for f in (
        Fixture("baseline", "control: one rectangular pad on the chip", baseline),
        Fixture(
            "near_tangent_arc_line",
            "near-tangent arc and line (secant 2 nm deep)",
            near_tangent_arc_line,
        ),
        Fixture(
            "exactly_tangent_arc_line",
            "arc and line touching at one point (same Net)",
            exactly_tangent_arc_line,
            expect_error={"B": NonManifoldError, "C": NonManifoldError},
            note="Route A: two sheets meeting at a point are legal",
        ),  # fmt: skip
        Fixture(
            "arcs_crossing_shallow",
            "two arcs crossing at 1.6 degrees",
            arcs_crossing_shallow,
        ),
        Fixture(
            "edges_1_to_3_g",
            "edges 1, 2, 3 g long and a 1.4 g diagonal",
            edges_1_to_3_g,
        ),
        Fixture(
            "kissing_corners_same_net",
            "vertex-to-vertex contact of two same-Net pads",
            kissing_corners_same_net,
            expect_error={"B": NonManifoldError, "C": NonManifoldError},
            note="Route A: two sheets sharing a corner are legal",
        ),  # fmt: skip
        Fixture(
            "kissing_corners_different_nets",
            "vertex-to-vertex contact of two different Nets",
            kissing_corners_different_nets,
            expect_error={"A": ContactError, "B": ContactError, "C": ContactError},
        ),  # fmt: skip
        Fixture(
            "collinear_overlap",
            "collinear overlapping edges from different prisms",
            collinear_overlap,
        ),
        Fixture(
            "hole_touches_outer",
            "a hole touching the outer boundary at one point",
            hole_touches_outer,
            expect_error={"B": NonManifoldError, "C": NonManifoldError},
            note="Route A: a sheet with a pinched hole is legal",
        ),  # fmt: skip
        Fixture(
            "arc_with_foreign_vertex",
            "arc whose edge carries a foreign hot pixel: cylinder split at the vertex",
            arc_with_foreign_vertex,
        ),  # fmt: skip
        Fixture(
            "arc_with_foreign_vertex_from_post",
            "same, vertex from a stacked same-Net post",
            arc_with_foreign_vertex_from_post,
            routes=("B", "C"),
        ),  # fmt: skip
        Fixture(
            "port_crossing_round_pad",
            "a port sheet crossing a round pad twice",
            port_crossing_round_pad,
            routes=("B",),
        ),  # fmt: skip
        Fixture(
            "floating_conductor",
            "inner void shell (B) / embedded sheet (A)",
            floating_conductor,
        ),
        Fixture(
            "nested_holes",
            "nested holes: an island in a hole in an island in a hole",
            nested_holes,
        ),
        Fixture(
            "stacked_corner_touch",
            "stacked slabs touching at one vertex of a level",
            stacked_corner_touch,
            routes=("B", "C"),
        ),  # fmt: skip
        Fixture(
            "stacked_vertex_on_wall_edge",
            "stacked slabs: a level vertex touching a wall's top edge",
            stacked_vertex_on_wall_edge,
            routes=("B", "C"),
        ),  # fmt: skip
        Fixture(
            "port_edge_on_metal_sidewall",
            "port sheet whose edge lies partly on a metal sidewall",
            port_edge_on_metal_sidewall,
            routes=("A", "B"),
        ),  # fmt: skip
        Fixture(
            "near_coincident_layers_sub_grid",
            "near-coincident layer edges 0.4 g apart: must merge, no sliver",
            near_coincident_layers_sub_grid,
            routes=("B", "C"),
        ),  # fmt: skip
        Fixture(
            "near_coincident_layers_2g",
            "near-coincident layer edges 2 g apart: a real 2 nm sliver face",
            near_coincident_layers_2g,
            routes=("B", "C"),
        ),  # fmt: skip
    )
}


def cases() -> list[tuple[Fixture, str]]:
    """Every (fixture, route) pair of the catalogue."""
    return [
        (fixture, route) for fixture in FIXTURES.values() for route in fixture.routes
    ]
