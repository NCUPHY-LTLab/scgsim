"""The meshing stage of ``simulation_drc``: fine edges and faces of the built model, located and attributed by lineage.

Every scene is a small synthetic layout built by the real SGB build: a 20 um pad beside a 30 um electrode, on a
substrate wrapped in the check's 1 um vacuum.  What changes is the pad's outline:

* ``clean`` -- a square;
* ``jog`` -- a 1 nm step in the edge facing the electrode (layout fine structure: a wall face 1 grid step
  wide);
* ``convex`` -- the top edge bent by 0.57 degrees at a convex vertex, so its left half is tilted by 0.29 degrees
  (no fine structure in the source; where that half's bands end at the pad's left edge, x = 0, their cap
  arcs are the margin times the tilt, 2 and 5 grid steps, an SGB defect);
* ``bar`` -- a bar 2.004 um wide inside a U-shaped electrode (with 1.0 um bands, the exact erosion leaves a
  core 4 nm wide).

``mesh()`` runs the same meshing stage on the model it built before the mesher starts; its tests replace only
the mesher, so the refusal comes from the real build and the real check.  The fast meshing-stop test stands in
for the Gmsh worker; the located finding it reports comes from the real build's sidecar.  The slow ones mesh
with Gmsh.
"""

from __future__ import annotations

import math

import pytest
from meshing_support import audit_of, outcome_of
from test_simulation_drc import (
    BANDED,
    MATERIALS,
    NO_VACUUM_STACK,
    SIMULATIONS,
)

import scgsim.meshing.entry as meshing_entry
import scgsim.meshing.repair as meshing_repair
import scgsim.palace.mesh.build as build_module
import scgsim.palace.preparation.drc as drc_module
from scgsim.geometry.sgb.errors import OverlapError
from scgsim.meshing import (
    Budgets,
    FineStructureLimits,
    MeshRepairError,
    check_fine_structure,
    mesh_sgb,
)
from scgsim.meshing.findings import BadElement
from scgsim.meshing.repair import RepairPolicy
from scgsim.meshing.sizing import BoundingBox
from scgsim.meshing.stop_places import stop_places
from scgsim.palace import EigenmodeSim, ElectrostaticSim, SimulationDrcError
from scgsim.palace.preparation.layout_snapshot import (
    DRC_VACUUM_PADDING_UM,
    prepare_layout_snapshot,
)

ELECTRODE = ((30, 0), (60, 0), (60, 25), (30, 25))
PADS = {
    "clean": ((0, 0), (20, 0), (20, 20), (0, 20)),
    "jog": ((0, 0), (20, 0), (20, 10), (20.001, 10), (20.001, 20), (0, 20)),
    # A jog of exactly ten grid steps whose coordinates subtract to 0.00999999999999801 um.
    "ten_step_jog": (
        (0, 0),
        (19.001, 0),
        (19.001, 10),
        (19.011, 10),
        (19.011, 20),
        (0, 20),
    ),
    "convex": ((0, 0), (20, 0), (20, 20), (10, 20.05), (0, 20)),
}
BAR = ((0, 10), (20, 10), (20, 12.004), (0, 12.004))
U_ELECTRODE = (
    (-5, 0),
    (30, 0),
    (30, 25),
    (-5, 25),
    (-5, 15),
    (25, 15),
    (25, 7),
    (-5, 7),
)
# Where the jog's wall stands: the 1 nm step at y = 10 um on the pad's edge facing the electrode, across the 0.2 um film.
JOG_WALL_UM = (20.0005, 10.0, 0.1)


def scene(monkeypatch, pad, electrode=ELECTRODE):
    """A pad and an electrode on layers (1, 0) and (2, 0), both conductors of the ``metal`` level."""
    import gdsfactory as gf  # noqa: PLC0415 -- optional layout stack, imported only by the tests that need it

    # Restore process-local PDK state after this public generic fixture.
    monkeypatch.setattr(
        gf.pdk,
        "_ACTIVE_PDK",
        gf.pdk._ACTIVE_PDK,  # noqa: SLF001 -- preserve the active PDK during test setup
    )
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon(pad, layer=(1, 0))
    component.add_polygon(electrode, layer=(2, 0))
    component.info["component_semantics"] = {
        "schema_version": 2,
        "conductor_regions": [
            {"semantic_id": "pad", "level": "metal", "gds_layer": [1, 0]},
            {"semantic_id": "electrode", "level": "metal", "gds_layer": [2, 0]},
        ],
    }
    return component


def check(simulation, component, **options):
    """``simulation.simulation_drc`` on a layout with no vacuum of its own, so the check wraps it in 1 um."""
    return simulation.simulation_drc(
        component,
        layer_stack=NO_VACUUM_STACK,
        material_records=MATERIALS,
        **options,
    )


def refusal(simulation, component, **options):
    """The meshing-stage refusal ``simulation_drc`` raises for ``component``."""
    with pytest.raises(SimulationDrcError) as raised:
        check(simulation, component, **options)
    return raised.value


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_layout_fine_structure_is_refused_in_the_meshing_stage_and_located(
    monkeypatch, simulation
):
    """The jog builds (geometry stage passes) and is refused in the meshing stage as the layout's, with its place."""
    error = refusal(simulation, scene(monkeypatch, PADS["jog"]))
    report = error.report
    assert report.geometry.passed
    assert report.geometry.findings == ()
    assert not report.meshing.passed
    (finding,) = [f for f in report.meshing.findings if f.action == "refuse"]
    assert (finding.stage, finding.rule, finding.measure) == (
        "meshing",
        "A",
        "wall column gap",
    )
    assert finding.owner == "layout"
    assert finding.location_um == pytest.approx(JOG_WALL_UM)
    assert finding.value == pytest.approx(0.001)
    assert finding.value_steps == pytest.approx(1)
    assert finding.groups == ("MA::chip/pad_0000__sidewall",)
    assert all(s.startswith("chip/pad_0000/polygon0/") for s in finding.sources)
    assert "remove the step from the polygon outline" in finding.fix
    assert "(20.0005, 10.0000, 0.1000) um" in str(error)


def test_the_bands_of_a_layout_jog_belong_to_the_layout(monkeypatch):
    """Band offsets that repeat a layout jog are the layout's: they lie within the band margin of the jog.

    The jog sits on the pad's top edge, which the 20 um electrode leaves flush with the model's edge, so the
    bands along it repeat the 1 nm step as seam pieces of one grid step.
    """
    top_jog = ((0, 0), (20, 0), (20, 20), (10, 20), (10, 20.001), (0, 20.001))
    flush = ((30, 0), (60, 0), (60, 20), (30, 20))
    error = refusal(
        EigenmodeSim, scene(monkeypatch, top_jog, flush), surface_epr_specs=BANDED
    )
    refusals = [f for f in error.report.meshing.findings if f.action == "refuse"]
    warnings = [f for f in error.report.meshing.findings if f.action == "warn"]
    assert [(f.rule, f.owner) for f in refusals] == [("A", "layout")]
    assert warnings
    assert {(f.rule, f.owner) for f in warnings} == {("C", "layout")}
    assert all(f.value_steps == pytest.approx(1) for f in warnings)
    assert all(
        any(source.startswith("epr_MA::chip/pad_0000::top/") for source in f.sources)
        for f in warnings
    )


def test_an_sgb_caused_fine_edge_is_attributed_to_sgb(monkeypatch):
    """The source is not fine anywhere; the short cap arcs where the tilted half's bands end are SGB's, and only warn."""
    report = check(
        EigenmodeSim,
        scene(monkeypatch, PADS["convex"]),
        surface_epr_specs=BANDED,
    )
    assert report.passed
    findings = report.meshing.findings
    assert findings
    assert {(f.rule, f.action, f.owner) for f in findings} == {("C", "warn", "sgb")}
    at_band_ends = [
        f for f in findings if f.location_um[0] == pytest.approx(0, abs=0.01)
    ]
    assert sorted(f.value_steps for f in at_band_ends) == pytest.approx([2, 5])
    assert all(f.value_steps is not None and f.value_steps < 25 for f in findings)
    assert all(
        f.sources
        and all(s.startswith("epr_MA::chip/pad_0000::top/") for s in f.sources)
        for f in findings
    )
    assert "SGB defect" in at_band_ends[0].fix


def test_a_core_the_exact_erosion_leaves_is_attributed_to_the_erosion(monkeypatch):
    """A bar 2.004 um wide keeps a 4 nm core beyond its 1.0 um bands: the offsets of its two long sides meet."""
    error = refusal(
        EigenmodeSim, scene(monkeypatch, BAR, U_ELECTRODE), surface_epr_specs=BANDED
    )
    (finding,) = [f for f in error.report.meshing.findings if f.action == "refuse"]
    assert (finding.rule, finding.measure, finding.owner) == (
        "B",
        "face width 2A/P",
        "erosion",
    )
    assert finding.value == pytest.approx(0.004, abs=1e-6)
    assert finding.groups == ("MA::chip/pad_0000__top::band=2",)
    assert "surface_epr_specs" in finding.fix


@pytest.mark.parametrize("simulation", SIMULATIONS)
@pytest.mark.parametrize("specs", [None, BANDED], ids=["no_bands", "bands"])
def test_a_clean_layout_has_no_meshing_finding(monkeypatch, simulation, specs):
    """A square pad and a rectangle electrode leave nothing below the limits, with or without bands."""
    options = {} if specs is None else {"surface_epr_specs": specs}
    report = check(simulation, scene(monkeypatch, PADS["clean"]), **options)
    assert report.passed
    assert report.geometry.passed and report.meshing.passed
    assert report.meshing.findings == ()
    assert "meshing stage: passed (0 refusing, 0 warning)" in report.render_text()


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_a_geometry_refusal_never_reaches_the_meshing_stage(monkeypatch, simulation):
    """Overlapping conductors raise the geometry stage's own error; the meshing stage does not run."""

    def meshing_stage_reached(*_args, **_kwargs):
        raise AssertionError("the meshing stage ran on a model that did not build")

    monkeypatch.setattr(drc_module, "check_fine_structure", meshing_stage_reached)
    overlapping = ((20 - 5, 0), (50, 0), (50, 20), (20 - 5, 20))
    with pytest.raises(OverlapError):
        check(simulation, scene(monkeypatch, PADS["clean"], overlapping))


def test_limits_hold_the_decided_values_and_reject_bad_ones():
    """The one settings object holds the Human's values; a non-positive limit or an unknown action is refused."""
    limits = FineStructureLimits()
    assert (
        limits.wall_column_gap_steps,
        limits.face_width_steps,
        limits.wedge_angle_deg,
        limits.free_edge_steps,
    ) == (10.0, 10.0, 1.0, 25.0)
    assert (limits.wall_action, limits.face_action, limits.free_edge_action) == (
        "refuse",
        "refuse",
        "warn",
    )
    with pytest.raises(ValueError, match="wall_column_gap_steps"):
        FineStructureLimits(wall_column_gap_steps=0)
    with pytest.raises(ValueError, match="free_edge_action"):
        FineStructureLimits(free_edge_action="ignore")  # type: ignore[arg-type]


class MesherStarted(Exception):
    """Raised in place of the mesher: ``mesh()`` got past the meshing-stage check."""


def stop_at_the_mesher(monkeypatch):
    """Replace the mesher ``mesh()`` starts by one that records the call and raises :class:`MesherStarted`."""
    started = []

    def mesher(xao, *_args, **_kwargs):
        started.append(xao)
        raise MesherStarted

    monkeypatch.setattr(build_module, "mesh_sgb", mesher)
    return started


def set_up_as_the_check(simulation, component, run_dir, specs=None):
    """A ``simulation`` set up as ``simulation_drc`` sets it up for ``component``, writing into ``run_dir``.

    The layout has no vacuum of its own, so it gets the check's vacuum region; ``ElectrostaticSim`` needs
    Surface EPR, so none given means ``{}``.
    """
    snapshot = prepare_layout_snapshot(
        component,
        layer_stack=NO_VACUUM_STACK,
        material_records=MATERIALS,
        instances=None,
        nets=None,
    )
    sim = simulation()
    sim.set_plan(snapshot)
    sim.set_vacuum_region(DRC_VACUUM_PADDING_UM)
    if specs is not None or simulation is ElectrostaticSim:
        sim.set_surface_epr(representation="B", specs=specs or {})
    sim.set_output_dir(run_dir)
    return sim


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_mesh_refuses_layout_fine_structure_before_the_mesher_starts(
    monkeypatch, tmp_path, simulation
):
    """``mesh()`` checks the model it built: the jog is refused with the report ``simulation_drc`` gives, unmeshed."""
    expected = refusal(simulation, scene(monkeypatch, PADS["jog"])).report
    started = stop_at_the_mesher(monkeypatch)
    sim = set_up_as_the_check(
        simulation, scene(monkeypatch, PADS["jog"]), tmp_path / "run"
    )
    with pytest.raises(SimulationDrcError) as raised:
        sim.mesh()
    assert started == []
    assert raised.value.report == expected
    assert "(20.0005, 10.0000, 0.1000) um" in str(raised.value)
    with pytest.raises(RuntimeError, match="requires a generated mesh"):
        sim.mesh_summary()


@pytest.mark.parametrize("simulation", SIMULATIONS)
def test_mesh_goes_on_to_the_mesher_past_warnings(monkeypatch, tmp_path, simulation):
    """The convex vertex leaves only class C warnings: ``simulation_drc`` passes with them, and ``mesh()`` meshes."""
    report = check(
        simulation, scene(monkeypatch, PADS["convex"]), surface_epr_specs=BANDED
    )
    assert report.passed
    assert report.meshing.findings
    assert {f.action for f in report.meshing.findings} == {"warn"}
    started = stop_at_the_mesher(monkeypatch)
    sim = set_up_as_the_check(
        simulation, scene(monkeypatch, PADS["convex"]), tmp_path / "run", BANDED
    )
    with pytest.raises(MesherStarted):
        sim.mesh()
    assert len(started) == 1


@pytest.mark.slow
def test_gmsh_meshes_a_model_with_only_warnings(monkeypatch, tmp_path):
    """With Gmsh, ``mesh()`` goes past the convex scene's class C warnings and returns a clean mesh."""
    pytest.importorskip("gmsh", reason="the meshing module needs Gmsh (extra: palace)")
    sim = set_up_as_the_check(
        EigenmodeSim, scene(monkeypatch, PADS["convex"]), tmp_path / "run", BANDED
    )
    assert sim.mesh().is_file()
    meshing = sim.mesh_summary()["meshing"]
    assert meshing["stop"] == "clean"
    assert meshing["linear_mesh"]["bad_count"] == 0


def built_jog(monkeypatch, tmp_path, pad="jog"):
    """A jog scene built by the geometry stage into ``tmp_path``: its SGB artifacts."""
    snapshot = prepare_layout_snapshot(
        scene(monkeypatch, PADS[pad]),
        layer_stack=NO_VACUUM_STACK,
        material_records=MATERIALS,
        instances=None,
        nets=None,
    )
    sim = EigenmodeSim()
    sim.set_plan(snapshot)
    sim.set_vacuum_region(1.0)
    return sim._build_geometry(tmp_path / "run").artifacts  # noqa: SLF001 -- the geometry half simulation_drc runs


def test_limits_change_what_is_found(monkeypatch, tmp_path):
    """Below a limit of one grid step the jog is no finding; the values are the only switch."""
    artifacts = built_jog(monkeypatch, tmp_path)
    assert not check_fine_structure(artifacts.sidecar).passed
    relaxed = FineStructureLimits(wall_column_gap_steps=1.0)
    assert check_fine_structure(artifacts.sidecar, relaxed).findings == ()


def test_a_wall_of_exactly_the_limit_is_not_below_it(monkeypatch, tmp_path):
    """Ten grid steps is not below ten, though its coordinates subtract to a hair under 0.01 um."""
    artifacts = built_jog(monkeypatch, tmp_path, pad="ten_step_jog")
    assert check_fine_structure(artifacts.sidecar).passed
    stricter = FineStructureLimits(wall_column_gap_steps=11.0)
    walls = [
        f
        for f in check_fine_structure(artifacts.sidecar, stricter).findings
        if f.value_steps == 10.0
    ]
    assert len(walls) == 1


def bad_at(element, point, *, edge_min, edge_max, quality=0.0005):
    """A needle-shaped bad element thinnest at ``point``."""
    return BadElement(
        element=element,
        quality=quality,
        edge_min=edge_min,
        edge_max=edge_max,
        centroid=point,
        shortest_edge_midpoint=point,
        location="point",
        points=(),
        curves=(),
        surfaces=(),
        volumes=(),
        sgb_ids=(),
        groups=(),
        feature_kinds=(),
        candidates=(),
    )


def test_a_meshing_stop_names_where_and_who_must_act(monkeypatch, tmp_path):
    """A stop with bad elements at the jog and in open space says the layout must act at the jog, and SCGSim elsewhere.

    The round stands in for the Gmsh worker: its audit has one needle on the jog's 1 nm edge and one far away.
    """
    artifacts = built_jog(monkeypatch, tmp_path)
    bad = (
        bad_at(1, (20.0005, 10.0, 0.2), edge_min=0.001, edge_max=2.0),
        bad_at(2, (45.0, 12.0, 6.0), edge_min=0.5, edge_max=1.0, quality=0.005),
    )
    monkeypatch.setattr(
        meshing_entry,
        "measure_bounding_box",
        lambda *_args, **_kwargs: BoundingBox((-1, -1, -6), (61, 26, 1.2)),
    )

    def round_with_bad_elements(job, *_args, **_kwargs):
        job.msh.write_text("", encoding="utf-8")  # the rejected mesh the stop keeps
        return outcome_of(audit_of(bad))

    monkeypatch.setattr(meshing_repair, "run_round", round_with_bad_elements)
    with pytest.raises(MeshRepairError) as raised:
        mesh_sgb(
            artifacts.xao,
            artifacts.sidecar,
            tmp_path / "mesh",
            threshold=0.01,
            budgets=Budgets(max_rounds=0),
        )
    places = raised.value.report["stop_places"]
    assert [place["elements"] for place in places] == [[1], [2]]
    at_jog, far = places
    (found,) = at_jog["fine_structure"]
    assert (found["rule"], found["owner"]) == ("A", "layout")
    assert found["location_um"] == pytest.approx(list(JOG_WALL_UM))
    assert far["fine_structure"] == []
    message = str(raised.value)
    assert "layout must act: A wall column gap" in message
    assert "the mesher's to answer for" in message


@pytest.mark.slow
def test_gmsh_stops_at_the_jog_and_names_the_layout(monkeypatch, tmp_path):
    """Meshed once without repair, the jog leaves bad elements; the stop places them at the jog, the layout's to fix."""
    pytest.importorskip("gmsh", reason="the meshing module needs Gmsh (extra: palace)")
    artifacts = built_jog(monkeypatch, tmp_path)
    with pytest.raises(MeshRepairError) as raised:
        mesh_sgb(
            artifacts.xao,
            artifacts.sidecar,
            tmp_path / "mesh",
            threshold=0.01,
            budgets=Budgets(max_rounds=0),
        )
    places = raised.value.report["stop_places"]
    assert places
    owners = {
        (found["rule"], found["owner"])
        for place in places
        for found in place["fine_structure"]
    }
    assert owners == {("A", "layout")}
    assert places[0]["fine_structure"][0]["location_um"] == pytest.approx(
        list(JOG_WALL_UM)
    )
    assert "layout must act" in str(raised.value)


def _covers_every_reach(place) -> bool:
    return all(
        math.dist(place.centre, spot) + reach <= place.radius_um + 1e-12
        for spot, reach in place.spots
    )


def test_a_chain_of_overlapping_reaches_is_one_place_that_covers_them_all():
    """Each needle reaches 0.025 um; at 0.04 um apart they chain, and the place must hold the far end too."""
    bad = tuple(
        bad_at(tag, (x, 0.0, 0.0), edge_min=0.001, edge_max=1.0)
        for tag, x in ((1, 0.0), (2, 0.04), (3, 0.08))
    )
    (place,) = stop_places(bad, (), 0.01, RepairPolicy(), ())
    assert [e.element for e in place.elements] == [1, 2, 3]
    assert _covers_every_reach(place)
    assert place.radius_um == pytest.approx(0.065)


def test_a_reach_that_bridges_two_places_joins_them():
    """Elements 1 and 2 are apart; element 3 overlaps both, so all three are one place."""
    bad = tuple(
        bad_at(tag, (x, 0.0, 0.0), edge_min=0.001, edge_max=1.0)
        for tag, x in ((1, 0.0), (2, 0.1), (3, 0.05))
    )
    (place,) = stop_places(bad, (), 0.01, RepairPolicy(), ())
    assert sorted(e.element for e in place.elements) == [1, 2, 3]
    assert _covers_every_reach(place)


def test_a_place_of_one_element_keeps_its_own_reach():
    """One needle is its own centre and reach, exactly."""
    (place,) = stop_places(
        (bad_at(7, (1.5, -2.0, 0.1), edge_min=0.001, edge_max=1.0),),
        (),
        0.01,
        RepairPolicy(),
        (),
    )
    assert place.centre == (1.5, -2.0, 0.1)
    assert place.radius_um == pytest.approx(0.025)
