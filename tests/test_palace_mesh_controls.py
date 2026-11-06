"""The mesh controls of the public Sims: recipe, mesher, budgets, threshold; and Surface EPR as geometry (no Gmsh)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from palace_sgb_support import FILM

from scgsim.meshing import (
    Budgets,
    FeatureRecipe,
    GlobalSizeRecipe,
    MesherOptions,
    UniformBoundaryRecipe,
)
from scgsim.palace import EigenmodeSim, ElectrostaticSim
from scgsim.palace.mesh.controls import DEFAULT_QUALITY_THRESHOLD, MeshControls
from scgsim.palace.preparation import eigenmode, electrostatic

STACK = {
    "materials": {"Si": {"kind": "dielectric", "permittivity": 11.45}},
    "solution_regions": {"substrate": {"material_id": "Si"}},
    "layers": [{"semantic_id": "chip"}],
}


@pytest.mark.parametrize("sim_type", [ElectrostaticSim, EigenmodeSim])
def test_defaults_are_the_module_defaults_and_the_provisional_threshold(sim_type):
    """FeatureRecipe, MesherOptions, Budgets and 0.01 minSICN for both simulations."""
    controls = sim_type().mesh_controls
    assert controls == MeshControls(
        FeatureRecipe(), MesherOptions(), Budgets(), DEFAULT_QUALITY_THRESHOLD
    )
    assert DEFAULT_QUALITY_THRESHOLD == 0.01


@pytest.mark.parametrize("sim_type", [ElectrostaticSim, EigenmodeSim])
def test_set_mesh_retains_omitted_controls_and_invalidates_only_on_change(sim_type):
    """Each setter call changes the named controls only; an equal setting keeps the mesh."""
    sim = sim_type()
    sim.set_mesh(recipe=UniformBoundaryRecipe())
    sim.set_mesh(mesher=MesherOptions(algorithm_2d=5, geometry_order=2))
    sim.set_mesh(budgets=Budgets(max_rounds=2), quality_threshold=0.05)
    assert sim.mesh_controls == MeshControls(
        UniformBoundaryRecipe(),
        MesherOptions(algorithm_2d=5, geometry_order=2),
        Budgets(max_rounds=2),
        0.05,
    )
    sim._mesh_result = object()  # noqa: SLF001 -- stands for a generated mesh
    sim.set_mesh(recipe=UniformBoundaryRecipe())
    assert sim._mesh_result is not None  # noqa: SLF001
    sim.set_mesh(recipe=GlobalSizeRecipe())
    assert sim._mesh_result is None  # noqa: SLF001
    assert sim.mesh_controls.recipe == GlobalSizeRecipe()


@pytest.mark.parametrize("sim_type", [ElectrostaticSim, EigenmodeSim])
@pytest.mark.parametrize(
    "removed",
    (
        "size_policy",
        "refined_mesh_size",
        "max_mesh_size",
        "algorithm_3d",
        "threads",
        "surface_threads",
        "geometry_order",
        "high_order_optimize",
    ),
)
def test_the_old_mesh_options_are_gone(sim_type, removed):
    """No separate size or Gmsh options remain on ``set_mesh`` or ``set_numerical``, nor in the numerical dict."""
    sim = sim_type()
    with pytest.raises(TypeError):
        sim.set_mesh(**{removed: 1})
    if removed in ("refined_mesh_size", "max_mesh_size"):
        with pytest.raises(TypeError):
            sim.set_numerical(**{removed: 1.0})
    assert removed not in sim.numerical


class PlainRecipe:
    """A sizing recipe that is not a dataclass: only the two identity fields and the sizing method."""

    name = "plain"
    version = "1"

    def size_requests(self, features, bounding_box):
        """No sizes: the controls never call it."""
        return {}


@pytest.mark.parametrize("sim_type", [ElectrostaticSim, EigenmodeSim])
def test_a_plain_class_recipe_passes_set_mesh_and_the_input_summary(sim_type):
    """The manifest records a non-dataclass recipe by name and version with no parameters, as the report does."""
    sim = sim_type()
    sim.set_mesh(recipe=PlainRecipe())
    assert sim.mesh_controls.to_payload()["recipe"] == {
        "name": "plain",
        "version": "1",
        "parameters": {},
    }


def test_invalid_mesh_controls_are_refused():
    """Wrong types and a non-finite threshold fail before any state changes."""
    sim = ElectrostaticSim()
    before = sim.mesh_controls
    with pytest.raises(TypeError):
        sim.set_mesh(recipe="features")
    with pytest.raises(TypeError):
        sim.set_mesh(mesher={"algorithm_3d": 10})
    with pytest.raises(TypeError):
        sim.set_mesh(budgets=3)
    with pytest.raises(ValueError, match="finite"):
        sim.set_mesh(quality_threshold=float("nan"))
    assert sim.mesh_controls == before


@pytest.mark.parametrize("sim_type", [ElectrostaticSim, EigenmodeSim])
def test_mesh_passes_the_controls_and_the_surface_epr_specs_to_the_builder(
    sim_type, monkeypatch, tmp_path
):
    """``mesh()`` hands the current Surface-EPR specs (their margins become bands) and the controls on."""
    module = electrostatic if sim_type is ElectrostaticSim else eigenmode
    seen = {}

    def build(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(mesh_path=tmp_path / "palace.msh")

    prepared = SimpleNamespace(
        component=object(),
        stack=STACK,
        indium_ground_bump_fill=None,
        materials=None,
        geometry_input=object(),
        physical_plan=SimpleNamespace(context=object()),
    )
    monkeypatch.setattr(module, "build_route_mesh", build)
    monkeypatch.setattr(module, "prepare_mesh_input", lambda **kwargs: prepared)
    sim = sim_type()
    sim.set_stack(STACK)
    sim.set_geometry(SimpleNamespace(write_gds=lambda path: None))
    sim.set_output_dir(tmp_path / "run")
    sim.set_surface_epr(
        representation="B", specs={"MA": {**FILM, "inset_margins_um": [0.5]}}
    )
    sim.set_mesh(quality_threshold=0.02, budgets=Budgets(max_rounds=1))
    sim.mesh()
    assert seen["surface_epr_specs"]["MA"]["inset_margins_um"] == [0.5]
    assert seen["recipe"] == FeatureRecipe()
    assert seen["mesher"] == MesherOptions()
    assert seen["budgets"] == Budgets(max_rounds=1)
    assert seen["quality_threshold"] == 0.02


def test_surface_epr_set_after_mesh_invalidates_it():
    """Surface EPR is geometry: calling ``set_surface_epr`` again discards the mesh, config and handoff."""
    sim = ElectrostaticSim()
    sim.set_stack(STACK)
    sim._mesh_result = object()  # noqa: SLF001 -- stands for a generated mesh
    sim.config_path = "config.json"  # type: ignore[assignment]
    sim.set_surface_epr(representation="B", specs={"MA": {**FILM}})
    assert sim._mesh_result is None  # noqa: SLF001
    assert sim.config_path is None


def test_eigenmode_without_surface_epr_requests_no_insets(monkeypatch, tmp_path):
    """Eigenmode may skip Surface EPR: ``mesh()`` then passes no specs, so no bands are requested."""
    seen = {}
    prepared = SimpleNamespace(
        component=object(),
        stack=STACK,
        indium_ground_bump_fill=None,
        materials=None,
        geometry_input=object(),
        physical_plan=SimpleNamespace(context=object()),
    )
    monkeypatch.setattr(
        eigenmode,
        "build_route_mesh",
        lambda **kw: seen.update(kw) or SimpleNamespace(mesh_path=tmp_path),
    )
    monkeypatch.setattr(eigenmode, "prepare_mesh_input", lambda **kwargs: prepared)
    sim = EigenmodeSim()
    sim.set_stack(STACK)
    sim.set_geometry(SimpleNamespace(write_gds=lambda path: None))
    sim.set_output_dir(tmp_path)
    sim.mesh()
    assert seen["surface_epr_specs"] is None
