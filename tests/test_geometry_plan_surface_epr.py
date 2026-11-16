"""``GeometryPlan.set_surface_epr``: the one place the Surface-EPR specs are given (Issue #268).

The plan checks the specs by the shared rules of ``scgsim.geometry.surface_epr.specs``, against its own
material records, when they are set, and ``prepare()`` puts them, normalized against the stack it prepares,
into the snapshot every simulation reads.  The refusals are those a simulation's ``set_surface_epr`` gave at
3.0.0rc3.  Each plan here is the two-pad layout of ``test_simulation_drc``; no test meshes or solves.
"""

from __future__ import annotations

import pytest
from test_simulation_drc import BANDED, FILM, layout_plan, two_pads

from scgsim.geometry.surface_epr.specs import normalize_surface_epr_specs

EMPTY = (
    "set_surface_epr({}) requests no Surface EPR: to have none, do not call plan.set_surface_epr(), "
    "or call plan.set_surface_epr(None) to turn it off again. "
    "Change it in the call to GeometryPlan.set_surface_epr."
)
REFUSED = {
    "not a mapping": ([], TypeError, "surface_epr specs must be a mapping."),
    "text": ("", TypeError, "surface_epr specs must be a mapping."),
    "unknown interface type": (
        {"MA": FILM, "XY": FILM},
        ValueError,
        "surface_epr has unsupported interface types: ['XY'].",
    ),
    "empty mapping": ({}, ValueError, EMPTY),
    "spec not a mapping": (
        {"MA": 2e-9},
        TypeError,
        "Surface EPR MA spec must be a mapping.",
    ),
    "unknown field": (
        {"MA": {**FILM, "margin_um": [0.5]}},
        ValueError,
        "Surface EPR MA has unknown fields: ['margin_um'].",
    ),
    "no thickness": (
        {"MA": {"permittivity": 4}},
        ValueError,
        "Surface EPR MA requires exactly one native thickness or film_thickness_m.",
    ),
    "material not a dielectric of the plan": (
        {"MA": {"film_thickness_m": 2e-9, "material_id": "PEC"}},
        ValueError,
        "Surface EPR MA material_id must resolve an explicit dielectric material.",
    ),
    "margins decreasing": (
        {"MA": {**FILM, "inset_margins_um": [1.0, 0.5]}},
        ValueError,
        "Surface EPR MA inset_margins_um must be strictly increasing.",
    ),
    "margins repeated": (
        {"MA": {**FILM, "inset_margins_um": [0.5, 0.5]}},
        ValueError,
        "Surface EPR MA inset_margins_um must be strictly increasing.",
    ),
}


@pytest.fixture
def plan(monkeypatch):
    """The two-pad plan in the stand-in PDK's vacuum, without Surface EPR."""
    return layout_plan(two_pads(monkeypatch, gap_um=10.0))


@pytest.mark.parametrize("case", REFUSED)
def test_set_surface_epr_refuses_what_the_simulation_refused(plan, case):
    """Wrong type, unknown interface type, empty mapping, invalid spec, margins not strictly increasing."""
    specs, error, message = REFUSED[case]
    with pytest.raises(error) as raised:
        plan.set_surface_epr(specs)
    assert str(raised.value) == message


def test_a_refused_call_leaves_the_earlier_specs(plan):
    """A refusal comes before anything is stored: the plan still prepares the specs it had."""
    plan.set_surface_epr(BANDED)
    with pytest.raises(ValueError, match="strictly increasing"):
        plan.set_surface_epr({"MA": {**FILM, "inset_margins_um": [1.0, 0.5]}})
    assert plan.prepare().surface_epr_specs["MA"]["inset_margins_um"] == [0.5, 1.0]


def test_the_snapshot_carries_the_specs_normalized_against_its_stack(plan):
    """``prepare()`` normalizes the specs as the shared rules do, a material film included."""
    specs = {
        **BANDED,
        "MS": {"film_thickness_m": 3e-9, "material_id": "Si", "face_kinds": ["Top"]},
    }
    plan.set_surface_epr(specs)
    snapshot = plan.prepare()
    assert snapshot.surface_epr_specs == normalize_surface_epr_specs(
        specs, materials=snapshot.stack["materials"]
    )
    assert snapshot.surface_epr_specs["MS"]["permittivity"] == 11.45
    assert snapshot.surface_epr_specs["MS"]["face_kinds"] == ["top"]


def test_a_plan_without_specs_or_turned_off_prepares_none(plan):
    """No call, or ``set_surface_epr(None)`` after one, gives a snapshot without Surface EPR."""
    assert plan.prepare().surface_epr_specs is None
    plan.set_surface_epr(BANDED)
    plan.set_surface_epr(None)
    assert plan.prepare().surface_epr_specs is None


def test_the_specs_are_detached_from_the_caller_and_the_snapshot(plan):
    """Changing the mapping given, or the one a snapshot returns, changes neither the plan nor the snapshot."""
    given = {"MA": {**FILM, "inset_margins_um": [0.5, 1.0]}}
    plan.set_surface_epr(given)
    given["MA"]["inset_margins_um"].append(2.0)
    snapshot = plan.prepare()
    snapshot.surface_epr_specs["MA"]["inset_margins_um"].append(3.0)
    assert snapshot.surface_epr_specs["MA"]["inset_margins_um"] == [0.5, 1.0]
    assert plan.prepare().surface_epr_specs["MA"]["inset_margins_um"] == [0.5, 1.0]
