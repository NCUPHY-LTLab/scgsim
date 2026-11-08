"""``simulation_drc`` runs the modeling stages of ``mesh()`` on a layout and stops before meshing.

No test here runs Gmsh or a solver: where ``mesh()`` is compared, its meshing stage is replaced by a
marker error, so what is compared is the geometry stage both share.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import scgsim.palace.mesh.build as build_module
from scgsim.geometry import GeometryPlan
from scgsim.geometry.models.physical import PhysicalLayerDeclaration
from scgsim.palace import EigenmodeSim, simulation_drc

MATERIALS = {
    "PEC": {"kind": "conductor"},
    "vacuum": {"kind": "vacuum"},
    "Si": {"kind": "dielectric", "permittivity": 11.45},
}
FILM = {"film_thickness_m": 2e-9, "permittivity": 4}
BANDED = {"MA": {**FILM, "inset_margins_um": [0.5, 1.0]}}


class MeshingReached(Exception):
    """Raised in place of meshing: the geometry stage had no refusal."""


def level(name, material, interval, role):
    """One layer-stack level carrying its neutral physical declaration."""
    declaration = PhysicalLayerDeclaration(
        name, name, {"level": name}, interval, material, "finite_volume", {}, {}
    )
    return SimpleNamespace(
        material=material,
        zmin=interval[0],
        thickness=interval[1] - interval[0],
        mesh_order=1,
        info={
            "simulation_role": role,
            "include_in_component_simulation": True,
            "part_role": "signal",
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


def two_pads(monkeypatch, *, gap_um):
    """A 20 um pad and a 30 um electrode ``gap_um`` apart; a negative gap makes them overlap."""
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
    component.add_polygon(((0, 0), (20, 0), (20, 20), (0, 20)), layer=(1, 0))
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
    return component


def check(component, **options):
    """``simulation_drc`` with the stand-in PDK data."""
    return simulation_drc(
        component, layer_stack=LAYER_STACK, material_records=MATERIALS, **options
    )


def mesh_refusal(component, tmp_path, monkeypatch, specs=None):
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
    sim = EigenmodeSim()
    sim.set_plan(plan.prepare())
    if specs is not None:
        sim.set_surface_epr(representation="B", specs=specs)
    sim.set_output_dir(tmp_path / "run")
    monkeypatch.setattr(build_module, "mesh_sgb", meshing_reached)
    with pytest.raises(Exception) as error:  # noqa: PT011 -- the type is what the test compares
        sim.mesh()
    return error.value


def test_a_layout_that_models_returns_none(monkeypatch):
    """Two separate pads on a substrate model directly."""
    assert check(two_pads(monkeypatch, gap_um=10.0)) is None


def test_a_refused_layout_raises_the_error_mesh_raises(monkeypatch, tmp_path):
    """Overlapping electrodes are refused by the geometry build, with the same type and message as ``mesh()``."""
    component = two_pads(monkeypatch, gap_um=-5.0)
    expected = mesh_refusal(component, tmp_path, monkeypatch)
    assert not isinstance(expected, MeshingReached)
    with pytest.raises(type(expected)) as raised:
        check(component)
    assert str(raised.value) == str(expected)


def geometry_inset_requests(monkeypatch, **options):
    """The ``insets`` of every SGB build ``simulation_drc`` makes."""
    builds = []
    real = build_module.build_sgb_geometry

    def recording(*args, **kwargs):
        builds.append(kwargs.get("insets", ()))
        return real(*args, **kwargs)

    monkeypatch.setattr(build_module, "build_sgb_geometry", recording)
    check(two_pads(monkeypatch, gap_um=10.0), **options)
    return builds


def test_surface_epr_specs_reach_the_band_build(monkeypatch):
    """With specs, the SGB build asks for their bands; without, it asks for none."""
    assert geometry_inset_requests(monkeypatch) == [()]
    (requests,) = geometry_inset_requests(monkeypatch, surface_epr_specs=BANDED)
    assert [request.request_id for request in requests] == ["epr_MA"]
    assert requests[0].margins_um == (0.5, 1.0)


def test_the_check_never_meshes_and_leaves_no_folder(monkeypatch, tmp_path):
    """Meshing is not reached, and the temporary run folder is removed."""

    def meshing_reached(*_args, **_kwargs):
        raise MeshingReached

    monkeypatch.setattr(build_module, "mesh_sgb", meshing_reached)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    component = two_pads(monkeypatch, gap_um=10.0)
    assert check(component, surface_epr_specs=BANDED) is None
    assert list(tmp_path.iterdir()) == []
