"""``scgsim.aedt.EigenmodeSim.simulation_drc`` runs the geometry half of its own AEDT pipeline on a layout, without AEDT.

Every test builds a real SGB model and its STEP (OCP), as ``prepare_handoff()`` does; none starts AEDT.  Where
the check is compared with the simulation, the simulation is set up from the settings a user would write and
its ``prepare_handoff()`` or the import's first step raises the error compared.
"""

from __future__ import annotations

import pytest
from test_simulation_drc import (
    BANDED,
    LAYER_STACK,
    MATERIALS,
    NO_VACUUM_STACK,
    two_pads,
)

pytest.importorskip("OCP")

import scgsim.aedt.drc as drc_module  # noqa: E402
from scgsim.aedt import EigenmodeSim, LumpedRlc  # noqa: E402
from scgsim.aedt.drc import SimulationDrcReport  # noqa: E402
from scgsim.aedt.runtime.families.sgb_boundaries import BoundaryPlanError  # noqa: E402
from scgsim.aedt.runtime.families.sgb_import import planned_import  # noqa: E402
from scgsim.contracts.sgb_sidecar import read_sidecar  # noqa: E402
from scgsim.geometry import GeometryPlan  # noqa: E402
from scgsim.geometry.sgb.output.exchange.guard import read_step_guard  # noqa: E402

MODELINGS = ["solid", "thin_film"]


@pytest.mark.parametrize("modeling", MODELINGS)
def test_the_planned_import_holds_each_guarded_face_once_per_solid_and_each_free_face_as_its_sheet(
    monkeypatch, modeling
):
    """``planned_import`` is the import the guard plans: the copies a Surface-EPR row selects from."""
    seen = []

    def recording(model):
        guard = read_step_guard(model.path("guard"))
        seen.append((guard, planned_import(guard)))

    import scgsim.aedt.eigenmode as eigenmode_module  # noqa: PLC0415 -- the module whose import is replaced

    monkeypatch.setattr(eigenmode_module, "check_eigenmode_model", recording)
    check(two_pads(monkeypatch, gap_um=10.0, port=True), modeling=modeling)
    ((guard, planned),) = seen
    for solid in guard["solids"]:
        for face in solid["faces"]:
            assert solid["aedt_name"] in planned.face_copies[face["face"]]
    assert set(planned.sheets) == {face["face"] for face in guard["free_faces"]}
    for face, name in planned.sheets.items():
        assert planned.face_copies[face] == {name: 0}


def check(component, *, layer_stack=LAYER_STACK, **options):
    """``EigenmodeSim.simulation_drc`` with the stand-in PDK data."""
    return EigenmodeSim.simulation_drc(
        component, layer_stack=layer_stack, material_records=MATERIALS, **options
    )


def handoff_refusal(component, modeling, tmp_path, specs=None):
    """The error ``prepare_handoff()`` raises for the plan and settings ``simulation_drc`` builds."""
    snapshot = GeometryPlan(
        component,
        root_id="chip",
        layer_stack=LAYER_STACK,
        material_records=MATERIALS,
        coupon_padding_um=0.0,
    ).prepare()
    sim = EigenmodeSim(modeling=modeling, aedt_version="2024.2")
    sim.set_plan(snapshot)
    for region in snapshot.geometry_input.port_sheet_regions:
        sim.add_lumped_rlc(
            LumpedRlc(region.port_sheet_id, "parallel", inductance_h=1e-9)
        )
    if specs is not None:
        sim.set_surface_epr(specs)
    sim.set_output_dir(tmp_path / "run")
    sim.set_eigenmode(project_name="check", design_name="eigenmode")
    with pytest.raises(Exception) as error:  # noqa: PT011 -- the type is what the test compares
        sim.prepare_handoff()
    return error.value


@pytest.mark.parametrize("modeling", MODELINGS)
@pytest.mark.parametrize("port", [False, True])
def test_a_layout_that_models_passes_on_both_routes(monkeypatch, modeling, port):
    """Two separate pads, with or without an authored port, build as the Route A and the Route C model."""
    report = check(two_pads(monkeypatch, gap_um=10.0, port=port), modeling=modeling)
    assert isinstance(report, SimulationDrcReport)
    assert report.passed
    assert report.render_text() == "geometry stage: passed\n"


def built_models(monkeypatch, **options):
    """The sidecar of every model ``simulation_drc`` builds, read before its folder is removed."""
    sidecars = []
    real = drc_module.check_eigenmode_model

    def recording(model):
        sidecars.append(read_sidecar(model.path("sidecar")))
        real(model)

    monkeypatch.setattr(drc_module, "check_eigenmode_model", recording)
    import scgsim.aedt.eigenmode as eigenmode_module  # noqa: PLC0415 -- the module whose import is replaced

    monkeypatch.setattr(eigenmode_module, "check_eigenmode_model", recording)
    check(two_pads(monkeypatch, gap_um=10.0, port=True), **options)
    return sidecars


@pytest.mark.parametrize(("modeling", "route"), [("solid", "C"), ("thin_film", "A")])
def test_the_check_builds_the_route_its_modeling_realizes(monkeypatch, modeling, route):
    """``solid`` is checked as the Route C model and ``thin_film`` as the Route A model (decision 7)."""
    (sidecar,) = built_models(monkeypatch, modeling=modeling)
    assert sidecar["model"]["route"] == route


def test_surface_epr_specs_become_bands_and_every_row_is_selected(monkeypatch):
    """With specs, the MA rows carry their bands; without, no row has a band."""
    (plain,) = built_models(monkeypatch, modeling="solid")
    assert all(row["band"] is None for row in plain["surface_epr_row"].values())
    (banded,) = built_models(monkeypatch, modeling="solid", surface_epr_specs=BANDED)
    bands = {
        row["interface_type"]
        for row in banded["surface_epr_row"].values()
        if row["band"] is not None
    }
    assert bands == {"MA"}


@pytest.mark.parametrize("modeling", MODELINGS)
def test_a_refused_layout_raises_the_error_prepare_handoff_raises(
    monkeypatch, tmp_path, modeling
):
    """Overlapping electrodes are refused with the type and message of ``prepare_handoff()``."""
    component = two_pads(monkeypatch, gap_um=-5.0)
    expected = handoff_refusal(component, modeling, tmp_path)
    with pytest.raises(type(expected)) as raised:
        check(component, modeling=modeling)
    assert str(raised.value) == str(expected)


def test_a_layout_without_vacuum_is_checked_in_the_fixed_vacuum(monkeypatch):
    """A stack with no ``VACUUM_REGION`` gets Palace's check vacuum, so AEDT has a Region box to check."""
    (sidecar,) = built_models(
        monkeypatch, modeling="solid", layer_stack=NO_VACUUM_STACK
    )
    vacuum = [
        volume for volume in sidecar["volume"].values() if volume["kind"] == "vacuum"
    ]
    assert vacuum


def test_a_modeling_that_is_not_solid_or_thin_film_is_refused_before_the_layout_is_read(
    monkeypatch,
):
    """The modeling selector is checked first, with the message ``EigenmodeSim`` gives."""

    def layout_read(*_args, **_kwargs):
        raise AssertionError("the layout was read")

    import scgsim.aedt.eigenmode as eigenmode_module  # noqa: PLC0415 -- the module whose import is replaced

    monkeypatch.setattr(eigenmode_module, "layout_snapshot", layout_read)
    with pytest.raises(ValueError, match="explicitly 'solid' or 'thin_film'"):
        check(object(), modeling="route_b")


def test_an_empty_surface_epr_mapping_is_refused_as_set_surface_epr_refuses_it(
    monkeypatch,
):
    """``surface_epr_specs={}`` requests no Surface EPR and is refused as the simulation refuses it."""
    with pytest.raises(ValueError, match="requests no Surface EPR"):
        check(
            two_pads(monkeypatch, gap_um=10.0), modeling="solid", surface_epr_specs={}
        )


def test_nets_that_become_one_aedt_name_are_refused_naming_both(monkeypatch):
    """Two Nets whose IDs sanitise alike are refused by the preparation the check runs."""
    nets = {"chip/A": ("chip/pad",), "chip_A": ("chip/electrode",)}
    with pytest.raises(ValueError, match="chip/A") as raised:
        check(two_pads(monkeypatch, gap_um=10.0), modeling="solid", nets=nets)
    assert "chip_A" in str(raised.value)


def test_the_import_plan_refusal_is_the_checks_refusal(monkeypatch):
    """A refusal of the import's first step is raised by the check unchanged."""

    def refused(*_args, **_kwargs):
        raise BoundaryPlanError("port sheet 'PORT::P1' has no face in the sidecar")

    monkeypatch.setattr(drc_module, "plan_import", refused)
    with pytest.raises(BoundaryPlanError, match="PORT::P1"):
        check(two_pads(monkeypatch, gap_um=10.0, port=True), modeling="solid")
