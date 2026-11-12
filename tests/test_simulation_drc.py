"""Each Palace simulation's ``simulation_drc`` runs its ``mesh()`` modeling stages on a layout, then the meshing-stage check, and never meshes.

No test here runs Gmsh or a solver: where ``mesh()`` is compared, its meshing stage is replaced by a
marker error, so what is compared is the geometry stage both share.
"""

from __future__ import annotations

import json
import math
import re
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

import scgsim.documentation as documentation
import scgsim.geometry.surface_epr.build as banded_build
import scgsim.palace.mesh.build as build_module
from scgsim.documentation import page_url
from scgsim.geometry import GeometryPlan
from scgsim.geometry.models.physical import PhysicalLayerDeclaration
from scgsim.geometry.sgb.errors import ContactError, NonManifoldError
from scgsim.palace import EigenmodeSim, ElectrostaticSim

MATERIALS = {
    "PEC": {"kind": "conductor"},
    "vacuum": {"kind": "vacuum"},
    "Si": {"kind": "dielectric", "permittivity": 11.45},
}
FILM = {"film_thickness_m": 2e-9, "permittivity": 4}
BANDED = {"MA": {**FILM, "inset_margins_um": [0.5, 1.0]}}


class MeshingReached(Exception):
    """Raised in place of meshing: the geometry stage had no refusal."""


def level(name, material, interval, role, *, face="lower"):
    """One layer-stack level carrying its neutral physical declaration.

    The conductor is a film attached by its ``face``; the lower face is what a port sheet needs to land on.
    """
    film = role == "conductor"
    declaration = PhysicalLayerDeclaration(
        name,
        name,
        {"level": name},
        interval,
        material,
        "stack_film" if film else "finite_volume",
        {"normal_thickness_um": interval[1] - interval[0]}
        if film
        else {"axial_span_um": interval[1] - interval[0]},
        {"layer_attachment_face": face} if film else {},
        {},
    )
    return SimpleNamespace(
        material=material,
        zmin=interval[0],
        thickness=interval[1] - interval[0],
        mesh_order=1,
        info={
            "simulation_role": role,
            "include_in_component_simulation": True,
            "part_role": "face_metal",
            "physical_layer": declaration.to_payload(),
        },
    )


LAYER_STACK = SimpleNamespace(
    layers={
        "VACUUM_REGION": level("VACUUM_REGION", "vacuum", (-10, 12), "solution_region"),
        "substrate": level("substrate", "Si", (-5, 0), "solution_region"),
        "metal": level("metal", "PEC", (0, 0.2), "conductor"),
    }
)


# The same stack with no vacuum level, so the layout owns no vacuum.
NO_VACUUM_STACK = SimpleNamespace(
    layers={
        name: layer
        for name, layer in LAYER_STACK.layers.items()
        if name != "VACUUM_REGION"
    }
)
# Pad outlines inside the 20 um square: a 64-sided polygon circle and a diamond, which touch their bounds at single points.
CIRCLE_PAD = tuple(
    (10 + 10 * math.cos(2 * math.pi * i / 64), 10 + 10 * math.sin(2 * math.pi * i / 64))
    for i in range(64)
)
DIAMOND_PAD = ((10, 0), (20, 10), (10, 20), (0, 10))


def ground_around(low, high):
    """A ground plane 5 um wide, drawn 5 um around the box ``low``..``high`` and open towards +x."""
    (x0, y0), (x1, y1) = low, high
    return (
        (x0 - 10, y0 - 10),
        (x1 + 20, y0 - 10),
        (x1 + 20, y0 - 5),
        (x0 - 5, y0 - 5),
        (x0 - 5, y1 + 5),
        (x1 + 20, y1 + 5),
        (x1 + 20, y1 + 10),
        (x0 - 10, y1 + 10),
    )


SIMULATIONS = [EigenmodeSim, ElectrostaticSim]


def two_pads(monkeypatch, *, gap_um, port=False, pad=None, electrode=None):
    """A 20 um pad and a 30 um electrode ``gap_um`` apart; a negative gap makes them overlap.

    With ``port`` a lumped-port sheet bridges the gap, as an authored port of the component.
    ``pad`` replaces the square pad's outline, drawn inside the same 20 um square; ``electrode`` replaces
    the electrode's outline.
    """
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    # Restore process-local PDK state after this public generic fixture.
    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    component = gf.Component()
    far = 20.0 + gap_um
    pad = pad or ((0, 0), (20, 0), (20, 20), (0, 20))
    component.add_polygon(pad, layer=(1, 0))
    electrode = electrode or ((far, 0), (far + 30, 0), (far + 30, 20), (far, 20))
    component.add_polygon(electrode, layer=(2, 0))
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {"semantic_id": "pad", "level": "metal", "gds_layer": [1, 0]},
            {"semantic_id": "electrode", "level": "metal", "gds_layer": [2, 0]},
        ],
    }
    if port:
        # The sheet reaches 5 um into each conductor, so it lands on both.
        component.add_polygon(
            ((15, 8), (far + 5, 8), (far + 5, 12), (15, 12)), layer=(9, 0)
        )
        component.add_port(
            name="P1",
            center=(20 + gap_um / 2, 10),
            width=4,
            orientation=0,
            layer=(9, 0),
        )
        component.info["component_semantics"]["ports"] = [
            {"name": "P1", "layer": [9, 0], "target_layer": "metal"}
        ]
    return component


def check(simulation, component, *, layer_stack=LAYER_STACK, **options):
    """``simulation.simulation_drc`` with the stand-in PDK data."""
    return simulation.simulation_drc(
        component, layer_stack=layer_stack, material_records=MATERIALS, **options
    )


def configured(simulation, snapshot, specs):
    """A ``simulation`` set up as ``simulation_drc`` sets it up, from the settings a user would write."""
    sim = simulation()
    sim.set_plan(snapshot)
    for port in snapshot.stack["metadata"]["port_sheet_source_layers"]:
        if simulation is EigenmodeSim:
            sim.add_port(
                port["name"],
                layer=port["target_layer"],
                layout_sheet=True,
                inductance=1e-9,
            )
    if specs is not None or simulation is ElectrostaticSim:
        sim.set_surface_epr(representation="B", specs=specs or {})
    return sim


def mesh_refusal(simulation, component, tmp_path, monkeypatch, specs=None):
    """The error ``mesh()`` raises before meshing, for the plan and settings ``simulation_drc`` builds."""

    def meshing_reached(*_args, **_kwargs):
        raise MeshingReached

    plan = GeometryPlan(
        component,
        root_id="chip",
        layer_stack=LAYER_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    sim = configured(simulation, plan.prepare(), specs)
    sim.set_output_dir(tmp_path / "run")
    monkeypatch.setattr(build_module, "mesh_sgb", meshing_reached)
    with pytest.raises(Exception) as error:  # noqa: PT011 -- the type is what the test compares
        sim.mesh()
    return error.value


@pytest.mark.parametrize("simulation", SIMULATIONS)
@pytest.mark.parametrize("port", [False, True])
def test_a_layout_that_models_passes_both_stages(monkeypatch, simulation, port):
    """Two separate pads on a substrate, with or without an authored port, model directly."""
    component = two_pads(monkeypatch, gap_um=10.0, port=port)
    assert check(simulation, component).passed


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_layout_with_bands_passes_both_stages(monkeypatch, simulation):
    """Bands cut from Surface-EPR margins fit the gap between the pads."""
    component = two_pads(monkeypatch, gap_um=10.0, port=True)
    assert check(simulation, component, surface_epr_specs=BANDED).passed


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_refused_layout_raises_the_error_mesh_raises(
    monkeypatch, tmp_path, simulation
):
    """Overlapping electrodes are refused by the geometry build, with the same type and message as ``mesh()``."""
    component = two_pads(monkeypatch, gap_um=-5.0)
    expected = mesh_refusal(simulation, component, tmp_path, monkeypatch)
    assert not isinstance(expected, MeshingReached)
    with pytest.raises(type(expected)) as raised:
        check(simulation, component)
    assert str(raised.value) == str(expected)


def geometry_inset_requests(monkeypatch, simulation, **options):
    """The ``insets`` of every SGB build ``simulation_drc`` makes."""
    builds = []
    real = banded_build.build_sgb_geometry

    def recording(*args, **kwargs):
        builds.append(kwargs.get("insets", ()))
        return real(*args, **kwargs)

    monkeypatch.setattr(banded_build, "build_sgb_geometry", recording)
    check(simulation, two_pads(monkeypatch, gap_um=10.0), **options)
    return builds


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_surface_epr_specs_reach_the_band_build(monkeypatch, simulation):
    """With specs, the SGB build asks for their bands; without, it asks for none."""
    assert geometry_inset_requests(monkeypatch, simulation) == [()]
    (requests,) = geometry_inset_requests(
        monkeypatch, simulation, surface_epr_specs=BANDED
    )
    assert [request.request_id for request in requests] == ["epr_MA"]
    assert requests[0].margins_um == (0.5, 1.0)


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_the_check_never_meshes_and_leaves_no_folder(monkeypatch, tmp_path, simulation):
    """Meshing is not reached, and the temporary run folder is removed."""

    def meshing_reached(*_args, **_kwargs):
        raise MeshingReached

    monkeypatch.setattr(build_module, "mesh_sgb", meshing_reached)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    component = two_pads(monkeypatch, gap_um=10.0)
    assert check(simulation, component, surface_epr_specs=BANDED).passed
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("simulation", SIMULATIONS)
@pytest.mark.parametrize("specs", [[], ""])
def test_invalid_surface_epr_specs_are_refused_as_set_surface_epr_refuses_them(
    monkeypatch, simulation, specs
):
    """A falsy non-mapping is not read as "no Surface EPR"; the setter refuses it, as it does in a simulation."""
    component = two_pads(monkeypatch, gap_um=10.0)
    with pytest.raises(TypeError, match="surface_epr specs must be a mapping"):
        check(simulation, component, surface_epr_specs=specs)


def check_sidecars(monkeypatch, simulation, component, **options):
    """The sidecar of every real SGB build ``simulation_drc`` makes, read before its folder is removed."""
    sidecars = []
    real = simulation._build_geometry  # noqa: SLF001 -- the geometry half the check runs

    def recording(self, output_dir):
        geometry = real(self, output_dir)
        sidecars.append(json.loads(geometry.artifacts.sidecar.read_bytes()))
        return geometry

    monkeypatch.setattr(simulation, "_build_geometry", recording)
    check(simulation, component, **options)
    return sidecars


def ma_band_groups(sidecar):
    """The MA band physical groups of a sidecar."""
    return [
        name
        for name in sidecar["physical_groups"]
        if name.startswith("MA") and "band=" in name
    ]


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_layout_with_no_vacuum_is_wrapped_so_ma_bands_are_cut(
    monkeypatch, simulation
):
    """With no vacuum in the layout the check adds one, so the metal has air faces to band."""
    component = two_pads(monkeypatch, gap_um=10.0)
    (sidecar,) = check_sidecars(
        monkeypatch,
        simulation,
        component,
        layer_stack=NO_VACUUM_STACK,
        surface_epr_specs=BANDED,
    )
    assert ma_band_groups(sidecar)


def built_bounds(sidecar):
    """The bounds in um of the built vacuum, and of every face not on the vacuum's outer boundary (the layout).

    Read from the sidecar's expected topology of the written model: solid, face, edge, vertex.
    """
    expected = sidecar["expected"]
    groups = sidecar["physical_groups"]

    def bounds(faces):
        vertices = {
            vertex
            for face in faces
            for edge in expected["face_edges"][face - 1]
            for vertex in expected["edge_vertices"][abs(edge) - 1]
        }
        axes = list(
            zip(
                *(expected["vertices_um"][vertex - 1] for vertex in vertices),
                strict=True,
            )
        )
        return tuple(map(min, axes)), tuple(map(max, axes))

    vacuum = {
        face
        for solid in groups["VACUUM_REGION"]["native_tags"]
        for face in expected["solid_faces"][solid - 1]
    }
    outer = {
        face
        for name, group in groups.items()
        if name.startswith("EXT::VACUUM_REGION__")
        for face in group["native_tags"]
    }
    every = set(range(1, len(expected["face_edges"]) + 1))
    return bounds(vacuum), bounds(every - outer)


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_layout_with_no_vacuum_is_wrapped_in_the_decided_padding(
    monkeypatch, simulation
):
    """The built vacuum ends at the layout's planar extent and reaches 1000 um below and above it (Issue #182).

    The layout is the two pads on their substrate: x 0 to 60 um, y 0 to 20 um, z from the substrate's
    bottom at -5 um to the metal's top at 0.2 um.
    """
    (sidecar,) = check_sidecars(
        monkeypatch,
        simulation,
        two_pads(monkeypatch, gap_um=10.0),
        layer_stack=NO_VACUUM_STACK,
    )
    vacuum, layout = built_bounds(sidecar)
    assert layout == (
        pytest.approx((0.0, 0.0, -5.0)),
        pytest.approx((60.0, 20.0, 0.2)),
    )
    below = [edge - wall for edge, wall in zip(layout[0], vacuum[0], strict=True)]
    above = [wall - edge for edge, wall in zip(layout[1], vacuum[1], strict=True)]
    assert below == pytest.approx([0.0, 0.0, 1000.0])
    assert above == pytest.approx([0.0, 0.0, 1000.0])


PINCH_AT_THE_EDGE = re.compile(
    r"non-manifold pinch: conductor (?P<conductor>\S+) on layer (?P<layer>\S+) of Net \S+ reaches the "
    r"edge of VACUUM_REGION only at \((?P<x>-?\d+\.\d+), (?P<y>-?\d+\.\d+)\) um, "
    r"z (?P<z0>-?\d+\.\d+)\.\.(?P<z1>-?\d+\.\d+) um \(slab\[[^]]*\]\), a point on the model's planar "
    r"boundary"
)
FIX_AT_THE_EDGE = "Fix the layout: draw the die outline or a ground plane around it, or set a vacuum region."
# The section that explains the refusal, on the published documentation root of the running version.
SEE_THE_EDGE_SECTION = re.compile(
    r" See https://ncuphy-ltlab\.github\.io/scgsim/v\d+\.\d+\.\d+(-dev)?/"
    r"docs/known-geometry-problems\.html#boundary-point-pinch$"
)


def pinch_of(error):
    """The conductor, layer, place in um and film height that a pinch at the vacuum's edge names.

    The refusal also says who must act, draw around the conductor or set a vacuum region, and ends with the
    link to the section that explains it.
    """
    message = str(error)
    match = PINCH_AT_THE_EDGE.search(message)
    assert match is not None, message
    link = SEE_THE_EDGE_SECTION.search(message)
    assert link is not None, message
    assert message[: link.start()].endswith(FIX_AT_THE_EDGE)
    assert link.group(0) == " See " + page_url(
        "docs/known-geometry-problems", "boundary-point-pinch"
    )
    return (
        match["conductor"],
        match["layer"],
        (float(match["x"]), float(match["y"])),
        (float(match["z0"]), float(match["z1"])),
    )


@pytest.mark.parametrize("simulation", SIMULATIONS)
@pytest.mark.parametrize("pad", [CIRCLE_PAD, DIAMOND_PAD], ids=["circle", "diamond"])
def test_a_pad_reaching_the_planar_extent_at_a_point_is_refused_there(
    monkeypatch, simulation, pad
):
    """The vacuum ends at the layout's planar extent, so a pad that reaches it only at a vertex pinches it there.

    The refusal says that the pad reaches the vacuum's edge only there, names the pad, its layer and the
    place (the pad's vertex on the extent's left edge, x = 0, across the film's height), and says that the
    layout must draw the die outline or a ground plane around it, or set a vacuum region.
    """
    component = two_pads(monkeypatch, gap_um=10.0, pad=pad)
    with pytest.raises(NonManifoldError) as raised:
        check(
            simulation,
            component,
            layer_stack=NO_VACUUM_STACK,
            surface_epr_specs=BANDED,
        )
    conductor, layer, place, film = pinch_of(raised.value)
    assert (conductor, layer, film) == ("chip/pad_0000", "metal", (0.0, 0.2))
    assert place == (0.0, 10.0)
    assert min(math.dist(place, vertex) for vertex in pad) < 0.001


def dielectric_island_stack(outline, interval, **more):
    """``NO_VACUUM_STACK`` with finite dielectric islands: solution regions whose authored outlines are given.

    The island ``island`` has ``outline``; ``more`` adds islands by name, at the same ``interval``.
    """
    islands = {}
    for name, loop in {"island": outline, **more}.items():
        islands[name] = level(name, "Si", interval, "solution_region")
        islands[name].info["geometry"] = {"outer_loop": [list(p) for p in loop]}
    return SimpleNamespace(layers={**NO_VACUUM_STACK.layers, **islands})


# Islands in the 10 um gap of ``two_pads``, against the layout's lower edge y = 0: a diamond whose tip is on it,
# a square whose edge lies along it, and a diamond 1 um clear of it.
ISLAND_TIP_ON_THE_EDGE = ((25, 0), (28, 3), (25, 6), (22, 3))
ISLAND_ALONG_THE_EDGE = ((22, 0), (28, 0), (28, 6), (22, 6))
ISLAND_CLEAR_OF_THE_EDGE = ((25, 1), (28, 4), (25, 7), (22, 4))
REGION_AT_THE_EDGE = re.compile(
    r"non-manifold pinch: dielectric region (?P<region>\S+) reaches the edge of VACUUM_REGION only at "
    r"\((?P<x>-?\d+\.\d+), (?P<y>-?\d+\.\d+)\) um, z (?P<z0>-?\d+\.\d+)\.\.(?P<z1>-?\d+\.\d+) um"
)
FIX_REGION_AT_THE_EDGE = (
    "Fix the layout: draw the die outline around it, or change that region's outline so it meets "
    "the boundary along an edge or not at all."
)


@pytest.mark.parametrize("simulation", SIMULATIONS)
@pytest.mark.parametrize(
    "interval", [(0.0, 0.2), (0.2, 1.0)], ids=["metal-height", "above-the-metal"]
)
def test_a_dielectric_island_reaching_the_planar_extent_at_a_point_is_named_as_itself(
    monkeypatch, simulation, interval
):
    """A dielectric island, not a conductor, pinches the vacuum at the boundary: the refusal names the island.

    No conductor reaches the point, so the refusal names the island, the place and a fix that fits a region
    (draw the die outline, or change its outline), and still links to the section that explains the pinch.
    """
    with pytest.raises(NonManifoldError) as raised:
        check(
            simulation,
            two_pads(monkeypatch, gap_um=10.0),
            layer_stack=dielectric_island_stack(ISLAND_TIP_ON_THE_EDGE, interval),
        )
    message = str(raised.value)
    match = REGION_AT_THE_EDGE.search(message)
    assert match is not None, message
    assert match["region"] == "island"
    assert (float(match["x"]), float(match["y"])) == (25.0, 0.0)
    assert (float(match["z0"]), float(match["z1"])) == interval
    assert "conductor" not in message
    assert message.endswith(
        f"{FIX_REGION_AT_THE_EDGE} See "
        + page_url("docs/known-geometry-problems", "boundary-point-pinch")
    )


@pytest.mark.parametrize(
    "outline",
    [ISLAND_ALONG_THE_EDGE, ISLAND_CLEAR_OF_THE_EDGE],
    ids=["along-the-edge", "clear-of-the-edge"],
)
def test_a_dielectric_island_along_the_edge_or_clear_of_it_passes(monkeypatch, outline):
    """The refusal's fix holds: the island meeting the boundary along an edge, or not at all, models."""
    assert check(
        EigenmodeSim,
        two_pads(monkeypatch, gap_um=10.0),
        layer_stack=dielectric_island_stack(outline, (0.0, 0.2)),
    ).passed


# Two islands in the gap, meeting only at their corner (25, 9) um, well inside the layout's extent.
ISLAND_BELOW_THE_CORNER = ((21, 5), (25, 5), (25, 9), (21, 9))
ISLAND_ABOVE_THE_CORNER = ((25, 9), (29, 9), (29, 13), (25, 13))
REGIONS_INSIDE = re.compile(
    r"non-manifold pinch: VACUUM_REGION \(pinched between dielectric regions (?P<regions>[^)]+)\) "
    r"touches itself along the vertical edge at \((?P<x>-?\d+\.\d+), (?P<y>-?\d+\.\d+)\) um, "
    r"z (?P<z0>-?\d+\.\d+)\.\.(?P<z1>-?\d+\.\d+) um"
)


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_two_dielectric_islands_meeting_at_a_corner_are_named_as_regions(
    monkeypatch, simulation
):
    """Inside the layout the vacuum between two islands that meet only at a corner is pinched: the refusal names them.

    No conductor is there, so it names the two regions, not "conductors ?", keeps the place and the corner fix,
    and links to the section on the same rule inside the layout.
    """
    stack = dielectric_island_stack(
        ISLAND_BELOW_THE_CORNER, (0.0, 0.2), island_b=ISLAND_ABOVE_THE_CORNER
    )
    with pytest.raises(NonManifoldError) as raised:
        check(simulation, two_pads(monkeypatch, gap_um=10.0), layer_stack=stack)
    message = str(raised.value)
    match = REGIONS_INSIDE.search(message)
    assert match is not None, message
    assert match["regions"] == "island, island_b"
    assert (float(match["x"]), float(match["y"])) == (25.0, 9.0)
    assert (float(match["z0"]), float(match["z1"])) == (0.0, 0.2)
    assert "conductor" not in message
    assert message.endswith(
        "Fix the layout: give them a shared edge or separate them. See "
        + page_url("docs/known-geometry-problems", "pinch-inside-the-layout")
    )


# Islands on top of the square pad of ``two_pads`` (z 0.2 to 1.0 um, the pad ends at 0.2 um): a diamond whose tip
# is on the layout's lower edge, which the pad runs along, and two squares meeting only at the corner (10, 10) um.
ISLAND_ON_THE_PAD_AT_THE_EDGE = ((10, 0), (13, 3), (10, 6), (7, 3))
ISLANDS_ON_THE_PAD_AT_A_CORNER = (((5, 5), (10, 5), (10, 10), (5, 10)), ((10, 10), (15, 10), (15, 15), (10, 15)))  # fmt: skip


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_an_island_on_a_pad_along_the_edge_is_named_not_the_pad(
    monkeypatch, simulation
):
    """The pinch is in the island's slab; the pad below ends where that slab begins, so the island is named.

    The pad runs along the boundary under the island's tip, so it touches the pinched point from below.
    """
    stack = dielectric_island_stack(ISLAND_ON_THE_PAD_AT_THE_EDGE, (0.2, 1.0))
    with pytest.raises(NonManifoldError) as raised:
        check(simulation, two_pads(monkeypatch, gap_um=10.0), layer_stack=stack)
    message = str(raised.value)
    match = REGION_AT_THE_EDGE.search(message)
    assert match is not None, message
    assert match["region"] == "island"
    assert (float(match["x"]), float(match["y"])) == (10.0, 0.0)
    assert (float(match["z0"]), float(match["z1"])) == (0.2, 1.0)
    assert "chip/pad_0000" not in message
    assert FIX_REGION_AT_THE_EDGE in message


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_islands_meeting_at_a_corner_on_a_pad_are_named_not_the_pad(
    monkeypatch, simulation
):
    """Inside the layout too: the pad under the corner ends where the islands' slab begins, so they are named."""
    below, above = ISLANDS_ON_THE_PAD_AT_A_CORNER
    stack = dielectric_island_stack(below, (0.2, 1.0), island_b=above)
    with pytest.raises(NonManifoldError) as raised:
        check(simulation, two_pads(monkeypatch, gap_um=10.0), layer_stack=stack)
    message = str(raised.value)
    match = REGIONS_INSIDE.search(message)
    assert match is not None, message
    assert match["regions"] == "island, island_b"
    assert (float(match["x"]), float(match["y"])) == (10.0, 10.0)
    assert (float(match["z0"]), float(match["z1"])) == (0.2, 1.0)
    assert "chip/pad_0000" not in message


@pytest.mark.parametrize("simulation", SIMULATIONS)
@pytest.mark.parametrize(
    "tip_x_um", [20.002, 20.001, 20.0], ids=["2-nm-off", "1-nm-off", "on-the-corner"]
)
def test_an_island_beside_the_pad_at_the_edge_is_named_not_the_pad(
    monkeypatch, simulation, tip_x_um
):
    """At the metal's height, an island's tip on the lower edge next to the pad's corner (20, 0) um is named.

    The pad runs along the edge up to its corner, so it pinches nothing there; within two grid units of the
    tip, or touching it at the corner, it is still not what separates the vacuum on the two sides of the tip.
    """
    island = ((tip_x_um, 0), (24, 4), (21, 4))
    stack = dielectric_island_stack(island, (0.0, 0.2))
    with pytest.raises(NonManifoldError) as raised:
        check(simulation, two_pads(monkeypatch, gap_um=10.0), layer_stack=stack)
    message = str(raised.value)
    match = REGION_AT_THE_EDGE.search(message)
    assert match is not None, message
    assert match["region"] == "island"
    assert (float(match["x"]), float(match["y"])) == (tip_x_um, 0.0)
    assert "chip/pad_0000" not in message
    assert FIX_REGION_AT_THE_EDGE in message


@pytest.mark.parametrize(
    "installed", ["2.1.0a1", None], ids=["unpublished-version", "no-metadata"]
)
def test_without_a_published_root_the_refusal_stays_and_leaves_the_link_out(
    monkeypatch, installed
):
    """A running SCGSim whose documentation the site does not publish still refuses the pinch, as before.

    Only the closing "See <URL>" is left out; the conductor, layer, place and who must act stay.
    """

    def running_version(_package):
        if installed is None:
            raise PackageNotFoundError("scgsim")
        return installed

    monkeypatch.setattr(documentation, "version", running_version)
    with pytest.raises(NonManifoldError) as raised:
        check(
            EigenmodeSim,
            two_pads(monkeypatch, gap_um=10.0, pad=CIRCLE_PAD),
            layer_stack=NO_VACUUM_STACK,
        )
    message = str(raised.value)
    match = PINCH_AT_THE_EDGE.search(message)
    assert match is not None, message
    assert (match["conductor"], match["layer"], match["x"], match["y"]) == (
        "chip/pad_0000",
        "metal",
        "0.0000",
        "10.0000",
    )
    assert message.endswith(FIX_AT_THE_EDGE)
    assert "See " not in message


@pytest.mark.parametrize("simulation", SIMULATIONS)
@pytest.mark.parametrize("pad", [CIRCLE_PAD, DIAMOND_PAD], ids=["circle", "diamond"])
def test_a_ground_plane_drawn_around_the_pad_removes_the_pinch(
    monkeypatch, simulation, pad
):
    """With a ground plane setting the planar extent, the circle and the diamond no longer reach it."""
    component = two_pads(
        monkeypatch,
        gap_um=10.0,
        pad=pad,
        electrode=ground_around((0, 0), (20, 20)),
    )
    assert check(
        simulation,
        component,
        layer_stack=NO_VACUUM_STACK,
        surface_epr_specs=BANDED,
    ).passed


@pytest.mark.parametrize("radius_um", [10.0, 50.0])
def test_a_gdsfactory_circle_conductor_prepares_and_passes(monkeypatch, radius_um):
    """One ``gf.components.circle`` conductor prepares and models (Issue #87).

    gdsfactory stores the circle's closing point too, so the polygon repeats a
    vertex; before the source reader dropped the repeat (Issue #107) its
    zero-length edge reached the cutline split and ``prepare()`` divided by zero.
    Alone, the circle sets the layout's planar extent and reaches it only at its
    vertices, so the check refuses the vacuum it pinches at its leftmost vertex;
    with a ground plane drawn around it, it models.
    """
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_ref(gf.components.circle(radius=radius_um, layer=(1, 0)))
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {"semantic_id": "disk", "level": "metal", "gds_layer": [1, 0]}
        ],
    }
    snapshot = GeometryPlan(
        component,
        root_id="chip",
        layer_stack=NO_VACUUM_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    ).prepare()
    assert len(snapshot.geometry_input.polygons) == 1
    with pytest.raises(NonManifoldError) as raised:
        check(EigenmodeSim, component, layer_stack=NO_VACUUM_STACK)
    conductor, layer, place, _film = pinch_of(raised.value)
    assert (conductor, layer, place) == ("chip/disk_0000", "metal", (-radius_um, 0.0))

    component.add_polygon(
        ground_around((-radius_um, -radius_um), (radius_um, radius_um)), layer=(2, 0)
    )
    component.info["component_semantics"]["conductor_regions"].append(
        {"semantic_id": "ground", "level": "metal", "gds_layer": [2, 0]}
    )
    assert check(EigenmodeSim, component, layer_stack=NO_VACUUM_STACK).passed


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_layout_with_its_own_vacuum_gets_no_vacuum_region(monkeypatch, simulation):
    """When the stack carries ``VACUUM_REGION`` the check adds none and does not raise."""
    added = []
    real = simulation.set_vacuum_region

    def recording(self, padding=0.0):
        added.append(padding)
        return real(self, padding)

    monkeypatch.setattr(simulation, "set_vacuum_region", recording)
    component = two_pads(monkeypatch, gap_um=10.0)
    assert check(simulation, component, surface_epr_specs=BANDED).passed
    assert added == []
    assert check(simulation, component, layer_stack=NO_VACUUM_STACK).passed
    assert added == [(0.0, 0.0, 1000.0)]


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_touching_conductors_of_one_net_pass(monkeypatch, simulation):
    """Two abutting pads model when ``nets`` puts both in one Net, as the simulation's plan would (Issue #125)."""
    component = two_pads(monkeypatch, gap_um=0.0)
    nets = {"Ground": ("chip/pad", "chip/electrode")}
    assert check(simulation, component, nets=nets).passed


@pytest.mark.parametrize("simulation", SIMULATIONS)
@pytest.mark.parametrize(
    "nets",
    [None, {"Ground": ("chip/pad",), "Signal": ("chip/electrode",)}],
    ids=["no-nets", "two-nets"],
)
def test_a_contact_between_two_nets_is_still_refused(monkeypatch, simulation, nets):
    """Abutting pads without a shared Net are refused as a different-Net contact, with or without ``nets``."""
    component = two_pads(monkeypatch, gap_um=0.0)
    options = {} if nets is None else {"nets": nets}
    with pytest.raises(
        ContactError, match=r"different-Net contact .* at \(20\.0000, 0\.0000\) um"
    ):
        check(simulation, component, **options)


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_nets_are_refused_as_set_nets_refuses_them(monkeypatch, simulation):
    """A Net naming an Entity the plan does not have is refused by ``GeometryPlan.set_nets``."""
    component = two_pads(monkeypatch, gap_um=0.0)
    with pytest.raises(
        ValueError, match="Net 'Ground' has unknown Entity ID 'chip/pads'"
    ):
        check(simulation, component, nets={"Ground": ("chip/pads",)})
