"""Each Palace simulation's ``simulation_drc`` runs its ``mesh()`` modeling stages on a layout and stops before meshing.

No test here runs Gmsh or a solver: where ``mesh()`` is compared, its meshing stage is replaced by a
marker error, so what is compared is the geometry stage both share.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

import scgsim.geometry.surface_epr.build as banded_build
import scgsim.palace.mesh.build as build_module
from scgsim.geometry import GeometryPlan
from scgsim.geometry.models.physical import PhysicalLayerDeclaration
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

SIMULATIONS = [EigenmodeSim, ElectrostaticSim]


def two_pads(monkeypatch, *, gap_um, port=False, pad=None):
    """A 20 um pad and a 30 um electrode ``gap_um`` apart; a negative gap makes them overlap.

    With ``port`` a lumped-port sheet bridges the gap, as an authored port of the component.
    ``pad`` replaces the square pad's outline, drawn inside the same 20 um square.
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
    component.add_polygon(
        ((far, 0), (far + 30, 0), (far + 30, 20), (far, 20)), layer=(2, 0)
    )
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
def test_a_layout_that_models_returns_none(monkeypatch, simulation, port):
    """Two separate pads on a substrate, with or without an authored port, model directly."""
    component = two_pads(monkeypatch, gap_um=10.0, port=port)
    assert check(simulation, component) is None


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_layout_with_bands_returns_none(monkeypatch, simulation):
    """Bands cut from Surface-EPR margins fit the gap between the pads."""
    component = two_pads(monkeypatch, gap_um=10.0, port=True)
    assert check(simulation, component, surface_epr_specs=BANDED) is None


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
    assert check(simulation, component, surface_epr_specs=BANDED) is None
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


@pytest.mark.parametrize("simulation", SIMULATIONS)
@pytest.mark.parametrize("pad", [CIRCLE_PAD, DIAMOND_PAD], ids=["circle", "diamond"])
def test_pads_touching_the_vacuum_at_a_point_pass(monkeypatch, simulation, pad):
    """The 1 um wrap leaves air around a polygon circle and a diamond, which a flush vacuum pinches."""
    component = two_pads(monkeypatch, gap_um=10.0, pad=pad)
    assert (
        check(
            simulation,
            component,
            layer_stack=NO_VACUUM_STACK,
            surface_epr_specs=BANDED,
        )
        is None
    )


@pytest.mark.parametrize("radius_um", [10.0, 50.0])
def test_a_gdsfactory_circle_conductor_prepares_and_passes(monkeypatch, radius_um):
    """One ``gf.components.circle`` conductor prepares and models (Issue #87).

    gdsfactory stores the circle's closing point too, so the polygon repeats a
    vertex; before the source reader dropped the repeat (Issue #107) its
    zero-length edge reached the cutline split and ``prepare()`` divided by zero.
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
    assert check(EigenmodeSim, component, layer_stack=NO_VACUUM_STACK) is None


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
    assert check(simulation, component, surface_epr_specs=BANDED) is None
    assert added == []
    assert check(simulation, component, layer_stack=NO_VACUUM_STACK) is None
    assert added == [1.0]
