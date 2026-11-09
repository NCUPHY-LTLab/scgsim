"""The source grid nodes each polygon only with the polygons whose Z ranges meet its own.

The fixture is the flip-chip case of Issue #103: both dies' grounds have the same 32-gon clearance hole,
and a tongue of the D1 ground starts at two lattice points on one chord of that hole.  The D0 ground
never touches D1 metal, so its ring must keep exactly its 32 vertices.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
from geometry_grid_fixtures import GRID
from test_simulation_drc import MATERIALS, level

from scgsim.geometry import GeometryBuildInput, LayoutPolygonSpec, SemanticEntitySpec
from scgsim.geometry.source.grid import apply_geometry_grid
from scgsim.palace import EigenmodeSim

RADIUS_UM = 35.0
HALF_SIDE_UM = 50.0
TONGUE_UM = 2.0
D0_Z = {"z_min_um": 0.0, "z_max_um": 0.1}
D1_Z = {"z_min_um": 8.1, "z_max_um": 8.2}
BUMP_Z = {"z_min_um": 0.1, "z_max_um": 8.1}


def lattice(value):
    """The nearest 1 nm lattice coordinate, as GDS stores it."""
    return round(value * 1000) / 1000


def clearance():
    """The PDK's 32-gon clearance circle on the lattice, counter-clockwise from +x."""
    return [
        (
            lattice(RADIUS_UM * math.cos(2 * math.pi * k / 32)),
            lattice(RADIUS_UM * math.sin(2 * math.pi * k / 32)),
        )
        for k in range(32)
    ]


def lattice_point_on(start, end, near):
    """A lattice point within 0.1 nm of the chord start-end, close to parameter ``near``.

    The chord crosses that point's grid pixel, so snap rounding would route the chord through it.
    """
    for step in range(2000):
        for t in (near + step * 1e-4, near - step * 1e-4):
            x = lattice(start[0] + t * (end[0] - start[0]))
            y_on_chord = start[1] + (x - start[0]) * (end[1] - start[1]) / (
                end[0] - start[0]
            )
            if abs(y_on_chord - lattice(y_on_chord)) < 1e-4:
                return (x, lattice(y_on_chord))
    raise AssertionError("no lattice point on the chord")


def tongue_bases():
    """Two lattice points on the chord between -56.25 and -45 degrees."""
    circle = clearance()
    start, end = circle[27], circle[28]
    return (
        start,
        end,
        (lattice_point_on(start, end, 0.4), lattice_point_on(start, end, 0.6)),
    )


def square():
    """The 100 um die square, centred on the clearance circle."""
    s = HALF_SIDE_UM
    return ((-s, -s), (s, -s), (s, s), (-s, s))


def tongue_tips():
    """The tongue's inner corners, 2 um inward from its bases; they lie inside the clearance hole."""
    start, end, bases = tongue_bases()
    inward = (-(start[0] + end[0]) / 2, -(start[1] + end[1]) / 2)
    scale = TONGUE_UM / math.hypot(*inward)
    return tuple(
        (lattice(x + inward[0] * scale), lattice(y + inward[1] * scale))
        for x, y in bases
    )


def d1_hole():
    """The clearance hole with the D1 ground's tongue reaching inward from the chord."""
    _, _, (first, second) = tongue_bases()
    first_tip, second_tip = tongue_tips()
    circle = clearance()
    return (*circle[:28], first, first_tip, second_tip, second, *circle[28:])


def ground(name, z, hole):
    """One die's ground conductor: the square minus its clearance hole."""
    polygon = LayoutPolygonSpec(name, name, square(), (tuple(reversed(hole)),))
    entity = SemanticEntitySpec(
        "chip/" + name,
        "metal",
        "PEC",
        "conductor",
        1,
        "layout_extrusion",
        part_role="face_metal",
        net_id="Ground",
        polygon_ids=(name,),
        geometry={"geometry_source": "gds_polygon", **z},
    )
    return polygon, entity


def projected_rings(build):
    """Each projected polygon's rings, by polygon id."""
    projected = apply_geometry_grid(build, GRID)
    return {
        polygon.polygon_id: (polygon.exterior, *polygon.holes)
        for polygon in projected.polygons
    }, projected


def has_vertex_near(ring, point):
    """Whether a ring has a vertex within 2 nm of ``point``."""
    return any(math.dist(vertex, point) < 2e-3 for vertex in ring)


def two_die_build(*extra):
    """Both grounds, and any extra (polygon, Entity) pairs, as one source input."""
    d0 = ground("D0", D0_Z, tuple(clearance()))
    d1 = ground("D1", D1_Z, d1_hole())
    pairs = (d0, d1, *extra)
    return GeometryBuildInput(
        tuple(polygon for polygon, _ in pairs), tuple(entity for _, entity in pairs)
    )


def test_a_d1_vertex_on_a_shared_chord_does_not_split_the_d0_ring():
    """The D0 hole keeps its 32 vertices; the D1 hole keeps its tongue bases."""
    _, _, bases = tongue_bases()
    rings, projected = projected_rings(two_die_build())
    d0_hole, d1_hole_ring = rings["D0"][1], rings["D1"][1]
    assert len(d0_hole) == 32
    assert not any(has_vertex_near(d0_hole, base) for base in bases)
    assert all(has_vertex_near(d1_hole_ring, base) for base in bases)
    ranges = [
        group["z_range_um"] for group in projected.metadata["geometry_grid_arrangement"]
    ]
    assert ranges == [[0.0, 0.1], [8.1, 8.2]]


def test_metal_on_one_face_is_still_noded_together():
    """A bump spanning the gap meets both dies, so a D1 vertex on its edge still splits that edge."""
    x, y = tongue_tips()[0]
    # A 2 um bump inside the clearance hole whose lower edge runs through the tongue's first tip.
    bump = LayoutPolygonSpec(
        "bump", "bump", ((x - 1, y), (x + 1, y), (x + 1, y + 2), (x - 1, y + 2))
    )
    bump_entity = SemanticEntitySpec(
        "chip/bump",
        "metal",
        "PEC",
        "conductor",
        1,
        "layout_extrusion",
        part_role="bump_body",
        net_id="Ground",
        polygon_ids=("bump",),
        geometry={"geometry_source": "gds_polygon", **BUMP_Z},
    )
    rings, projected = projected_rings(two_die_build((bump, bump_entity)))
    assert has_vertex_near(rings["bump"][0], (x, y))
    assert len(rings["D0"][1]) == 32
    ranges = [
        group["z_range_um"] for group in projected.metadata["geometry_grid_arrangement"]
    ]
    assert ranges == [[0.0, 0.1], [0.1, 8.1], [8.1, 8.2]]


@pytest.mark.parametrize("z", [{}, D0_Z])
def test_one_z_range_keeps_a_single_arrangement(z):
    """Inputs on one Z range, or with no Z at all, are noded in one arrangement as before."""
    polygon = LayoutPolygonSpec("D0", "D0", square(), (tuple(reversed(clearance())),))
    entity = SemanticEntitySpec(
        "chip/D0",
        "metal",
        "PEC",
        "conductor",
        1,
        "layout_extrusion",
        polygon_ids=("D0",),
        geometry={"geometry_source": "gds_polygon", **z},
    )
    projected = apply_geometry_grid(GeometryBuildInput((polygon,), (entity,)), GRID)
    (group,) = projected.metadata["geometry_grid_arrangement"]
    assert group["z_range_um"] == ([0.0, 0.1] if z else None)


# Two dies 8 um apart, each with its ground on the face toward the gap.
FLIP_CHIP_STACK = SimpleNamespace(
    layers={
        "D0_SUBSTRATE": level("D0_SUBSTRATE", "Si", (-20, 0), "solution_region"),
        "GAP": level("GAP", "vacuum", (0, 8.2), "solution_region"),
        "D1_SUBSTRATE": level("D1_SUBSTRATE", "Si", (8.2, 28.2), "solution_region"),
        "D0_M": level("D0_M", "PEC", (0, 0.1), "conductor"),
        "D1_M": level("D1_M", "PEC", (8.1, 8.2), "conductor", face="upper"),
    }
)
FILM = {"film_thickness_m": 2e-9, "permittivity": 4, "inset_margins_um": [0.5, 1.0]}


def flip_chip(monkeypatch, *, with_d1):
    """The two grounds as a gdsfactory Component; without ``with_d1`` only the D0 ground is drawn."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it
    import klayout.db as kdb  # noqa: PLC0415 -- gdsfactory's polygon type, for a polygon with a hole

    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    component = gf.Component()
    regions = [("D0", "D0_M", (1, 0), clearance())]
    if with_d1:
        regions.append(("D1", "D1_M", (2, 0), d1_hole()))
    for _, _, layer, hole in regions:
        polygon = kdb.DPolygon([kdb.DPoint(*point) for point in square()])
        polygon.insert_hole([kdb.DPoint(*point) for point in hole])
        component.add_polygon(polygon, layer=layer)
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {"semantic_id": name, "level": level_name, "gds_layer": list(layer)}
            for name, level_name, layer, _ in regions
        ],
    }
    return component


def d0_band_areas(monkeypatch, component):
    """The D0 band areas of the one SGB build the geometry stage makes, by physical group."""
    sidecars = []
    real = EigenmodeSim._build_geometry  # noqa: SLF001 -- the geometry half the check runs

    def recording(self, output_dir):
        geometry = real(self, output_dir)
        sidecars.append(json.loads(geometry.artifacts.sidecar.read_bytes()))
        return geometry

    monkeypatch.setattr(EigenmodeSim, "_build_geometry", recording)
    refusal = EigenmodeSim.simulation_drc(
        component,
        layer_stack=FLIP_CHIP_STACK,
        material_records=MATERIALS,
        surface_epr_specs={"MA": FILM, "MS": FILM, "SA": FILM},
    )
    assert refusal is None
    (sidecar,) = sidecars
    return {
        name: group["area_um2"]
        for name, group in sidecar["physical_groups"].items()
        if "band=" in name and ("D0_0000" in name or "D0_SUBSTRATE" in name)
    }


def test_the_d0_bands_do_not_depend_on_the_d1_ground(monkeypatch):
    """The geometry stage passes, and D0's bands are the same with or without D1 metal above them."""
    alone = d0_band_areas(monkeypatch, flip_chip(monkeypatch, with_d1=False))
    paired = d0_band_areas(monkeypatch, flip_chip(monkeypatch, with_d1=True))
    assert len(alone) == 9
    assert paired == alone
