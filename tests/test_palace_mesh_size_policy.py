"""Public mesh policy/state and transport contracts; no native mesh proof."""

from copy import deepcopy
import json

import pytest

from scgsim.geometry import ApproximationProfile
from scgsim.palace import EigenmodeSim, ElectrostaticSim
from scgsim.palace.mesh.controls import MeshSizing
from scgsim.palace.preparation.config import configure_numerical_controls


@pytest.fixture(
    params=(EigenmodeSim, ElectrostaticSim), ids=("eigenmode", "electrostatic")
)
def sim(request):
    return request.param(
        route="B", approximation_profile=ApproximationProfile("route_b", "v1", {})
    )


def _sizing(sim):
    return tuple(
        sim.numerical[key]
        for key in ("size_policy", "refined_mesh_size", "max_mesh_size")
    )


def test_default_and_omitted_policy_preserve_legacy_custom_sizes(sim):
    assert _sizing(sim) == (
        ("gmsh_default", None, None)
        if isinstance(sim, EigenmodeSim)
        else ("scgsim_default", 5.0, 300.0)
    )
    sim.set_mesh(
        size_policy="scgsim_default", refined_mesh_size=10.0, max_mesh_size=200.0
    )
    sim.set_mesh(threads=1)
    assert _sizing(sim) == ("scgsim_default", 10.0, 200.0)
    sim.set_mesh(size_policy="scgsim_default")
    assert _sizing(sim) == ("scgsim_default", 10.0, 200.0)


def test_gmsh_policy_clears_stale_sizes_and_omission_retains_it(sim):
    sim.set_mesh(
        size_policy="scgsim_default", refined_mesh_size=10.0, max_mesh_size=200.0
    )
    sim.set_mesh(size_policy="gmsh_default")
    assert _sizing(sim) == ("gmsh_default", None, None)
    sim.set_mesh(threads=1)
    assert _sizing(sim) == ("gmsh_default", None, None)


@pytest.mark.parametrize(
    "sizes",
    (
        {"refined_mesh_size": 10.0},
        {"max_mesh_size": 200.0},
        {"refined_mesh_size": 10.0, "max_mesh_size": 200.0},
    ),
)
def test_gmsh_with_explicit_sizes_rejects_before_mutating_controls(sim, sizes):
    sim.set_mesh(
        size_policy="scgsim_default", refined_mesh_size=10.0, max_mesh_size=200.0
    )
    before = deepcopy(sim.numerical)
    with pytest.raises(ValueError):
        sim.set_mesh(size_policy="gmsh_default", algorithm_3d="HXT", **sizes)
    assert sim.numerical == before


def test_switch_to_scg_restores_defaults_or_explicit_sizes(sim):
    sim.set_mesh(
        size_policy="scgsim_default", refined_mesh_size=10.0, max_mesh_size=200.0
    )
    sim.set_mesh(size_policy="gmsh_default")
    sim.set_mesh(size_policy="scgsim_default")
    assert _sizing(sim) == ("scgsim_default", 5.0, 300.0)
    sim.set_mesh(size_policy="gmsh_default")
    sim.set_mesh(
        size_policy="scgsim_default", refined_mesh_size=10.0, max_mesh_size=200.0
    )
    assert _sizing(sim) == ("scgsim_default", 10.0, 200.0)


def test_numerical_without_sizes_preserves_gmsh_and_explicit_sizes_reject_atomically(
    sim,
):
    sim.set_mesh(size_policy="gmsh_default")
    sim.set_numerical(order=2, max_iterations=2000)
    assert _sizing(sim) == ("gmsh_default", None, None)
    assert sim.numerical["order"] == 2
    before = deepcopy(sim.numerical)
    with pytest.raises(ValueError):
        sim.set_numerical(order=1, refined_mesh_size=10.0, max_mesh_size=200.0)
    assert sim.numerical == before


@pytest.mark.parametrize(
    "policy,lower,upper",
    (("scgsim_default", 10.0, 200.0), ("gmsh_default", None, None)),
)
def test_mesh_sizing_and_numerical_json_codecs_retain_explicit_policy(
    policy, lower, upper
):
    sizing = MeshSizing(
        size_policy=policy, refined_mesh_size=lower, max_mesh_size=upper
    )
    payload = json.loads(json.dumps(sizing.to_payload()))
    assert payload == {
        "size_policy": policy,
        "refined_mesh_size": lower,
        "max_mesh_size": upper,
    }
    assert MeshSizing.from_payload(payload) == sizing
    controls = configure_numerical_controls(**payload)
    assert configure_numerical_controls(**json.loads(json.dumps(controls))) == controls


def test_native_sizing_leaves_groups_untouched_without_importing_gmsh(monkeypatch):
    import builtins
    from scgsim.palace.mesh.build import _apply_mesh_sizing

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "gmsh":
            raise AssertionError("native sizing must not import or configure Gmsh")
        return original_import(name, *args, **kwargs)

    groups = {"boundary_surfaces": {"surface": {"tags": (1,), "phys_group": [2]}}}
    before = deepcopy(groups)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _apply_mesh_sizing(groups, MeshSizing("gmsh_default", None, None))
    assert groups == before
