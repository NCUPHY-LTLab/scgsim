"""``EigenmodeSim.input_summary()`` summarises the input ``mesh()`` builds from, not the plan as it was bound.

A plan with no vacuum of its own gets its vacuum from the simulation, so the summary has to show that vacuum
as the model has it (Issue #193).  Each comparison runs the real preparation and SGB geometry build of
``mesh()``; only the mesher is replaced by a marker error, so no test here runs Gmsh or a solver.
``ElectrostaticSim`` has no ``input_summary()``.
"""

from __future__ import annotations

import copy
import json

import pytest
from test_simulation_drc import (
    LAYER_STACK,
    MATERIALS,
    NO_VACUUM_STACK,
    MeshingReached,
    built_bounds,
    two_pads,
)

import scgsim.palace.mesh.build as build_module
import scgsim.palace.preparation.eigenmode as eigenmode_module
from scgsim.geometry import ApproximationProfile, GeometryPlan
from scgsim.geometry.source.summary import summarize_geometry_input
from scgsim.palace import EigenmodeSim

# The two pads on their substrate span z from -5 um to the metal's top at 0.2 um; this padding, as
# ``set_vacuum_region`` takes it, adds nothing in the plane and 1000 um below and above.
VACUUM_PADDING_UM = (0.0, 0.0, 1000.0)
THIN_FILM = ApproximationProfile("thin_film", "v1", {})


def eigenmode_sim(monkeypatch, layer_stack, *, route="B"):
    """An ``EigenmodeSim`` on the two-pad plan of ``layer_stack`` with its authored port, as a user sets it up.

    A plan with no vacuum of its own is given ``VACUUM_PADDING_UM`` on the simulation.
    """
    plan = GeometryPlan(
        two_pads(monkeypatch, gap_um=10.0, port=True),
        root_id="chip",
        layer_stack=layer_stack,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    )
    plan.set_nets({"pad": ("chip/pad",), "electrode": ("chip/electrode",)})
    sim = EigenmodeSim()
    sim.set_plan(plan.prepare())
    if layer_stack is NO_VACUUM_STACK:
        sim.set_vacuum_region(VACUUM_PADDING_UM)
    sim.add_port("chip/P1", layer="metal", layout_sheet=True, inductance=1e-9)
    if route == "A":
        sim.set_surface_epr(
            representation="A", specs={}, approximation_profile=THIN_FILM
        )
    return sim


def built_by_mesh(monkeypatch, sim, output_dir):
    """The input ``mesh()`` hands its geometry build, and the sidecar of the model that build writes.

    The build is the real one, recorded as it runs; ``mesh()`` then stops at the mesher.
    """
    builds = []
    real = eigenmode_module.build_route_geometry

    def recording(**source):
        geometry = real(**source)
        builds.append((source, json.loads(geometry.artifacts.sidecar.read_bytes())))
        return geometry

    def meshing_reached(*_args, **_kwargs):
        raise MeshingReached

    monkeypatch.setattr(eigenmode_module, "build_route_geometry", recording)
    monkeypatch.setattr(build_module, "mesh_sgb", meshing_reached)
    sim.set_output_dir(output_dir)
    with pytest.raises(MeshingReached):
        sim.mesh()
    (built,) = builds
    return built


@pytest.mark.parametrize(
    ("layer_stack", "route", "vacuum_z_um"),
    [
        (LAYER_STACK, "B", (-10.0, 12.0)),
        (NO_VACUUM_STACK, "B", (-1005.0, 1000.2)),
        (NO_VACUUM_STACK, "A", (-1005.0, 1000.0)),
    ],
    ids=["plan-vacuum", "simulation-vacuum", "simulation-vacuum-route-a"],
)
def test_the_summary_shows_the_vacuum_mesh_builds(
    monkeypatch, tmp_path, layer_stack, route, vacuum_z_um
):
    """The summary's vacuum is the vacuum of the built model, and its source half is that of the input ``mesh()`` builds from.

    The plan's own vacuum spans z -10 to 12 um.  A plan without one gets the simulation's region, 1000 um
    below the substrate's bottom and above the metal's top; on Route A the film is lowered to a sheet at
    z = 0, so the region the model has ends 0.2 um lower than the bound plan and the padding give.
    """
    sim = eigenmode_sim(monkeypatch, layer_stack, route=route)

    summary = sim.input_summary().data

    assert summary["vacuum_domains"] == [
        {
            "semantic_id": "VACUUM_REGION",
            "z_min_um": vacuum_z_um[0],
            "z_max_um": vacuum_z_um[1],
        }
    ]
    source, sidecar = built_by_mesh(monkeypatch, sim, tmp_path / "run")
    (vacuum_low, vacuum_high), _layout = built_bounds(sidecar)
    assert (vacuum_low[2], vacuum_high[2]) == pytest.approx(vacuum_z_um)
    built_from = summarize_geometry_input(
        source["source_geometry_input"], prepared_stack=source["stack"]
    ).data
    assert {key: summary[key] for key in built_from} == built_from
    assert [port["source_name"] for port in summary["ports"]] == ["chip/P1"]


def test_the_summary_leaves_the_simulation_as_it_was(monkeypatch):
    """Preparing the summary keeps nothing: the materials the simulation holds are still the bound plan's.

    The prepared stack completes the vacuum material of the region the simulation set; only ``mesh()``
    keeps that.
    """
    sim = eigenmode_sim(monkeypatch, NO_VACUUM_STACK)
    before = copy.copy(sim)

    sim.input_summary()

    assert sim == before
