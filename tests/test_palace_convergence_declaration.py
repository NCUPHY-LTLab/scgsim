"""``set_convergence``: the quantities and criteria, every refusal, and what ``write_config()`` writes from them.

The write-time checks and the plan run on a real Route B SGB build (``palace_convergence_support``): the
plan's columns are read from the index entries the same config writes.  No Gmsh and no solver run.
"""

from __future__ import annotations

import json

import pytest
from palace_convergence_support import eigenmode, electrostatic

from scgsim.contracts.palace_run_folder import CONFIG, CONVERGENCE_PLAN, INDEX_MAP
from scgsim.palace import (
    Capacitance,
    Criterion,
    CriterionGroup,
    EigenmodeSim,
    ElectrostaticSim,
    ErrorIndicator,
    ModeFrequency,
    SurfaceEpr,
)
from scgsim.palace.config.numerical import DECLARED_CONVERGENCE_TOL, NumericalControls
from scgsim.palace.simulation.lifecycle import Stage

# ---- What set_convergence takes -------------------------------------------------------------------------


def test_each_capacitance_selection_has_its_label():
    """The report names a selection the way the user wrote it."""
    assert Capacitance.element("a", "b").label == "C[a][b]"
    assert Capacitance.diagonal().label == "C diagonal"
    assert Capacitance.sub_diagonal(2).label == "C sub-diagonal (offset 2)"
    assert Capacitance.off_diagonal().label == "C off-diagonal"
    assert Capacitance.matrix().label == "C matrix"
    assert Capacitance.elements([("a", "a"), ("a", "b")]).label == "C C[a][a], C[a][b]"
    assert ModeFrequency(modes=[1, 3]).label == "frequency, modes 1, 3"
    assert (
        SurfaceEpr("MA", margin_um=1).label
        == "MA participation, 1 um or more from the gap edge"
    )
    assert SurfaceEpr("SA", modes=[2]).label == "SA participation, mode 2"


def test_each_threshold_keyword_names_its_unit():
    """A relative change, or an absolute change in the unit the quantity is stored in."""
    assert Capacitance.diagonal().within(relative=1e-3).test == "relative_change"
    assert Capacitance.diagonal().within(absolute_f=1e-18).test == "absolute_change"
    assert ModeFrequency().within(absolute_ghz=1e-4).threshold == 1e-4
    assert SurfaceEpr("MA").within(absolute=1e-5).test == "absolute_change"
    criterion = ErrorIndicator().below(1e-2)
    assert (criterion.test, criterion.threshold) == ("at_most", 1e-2)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Capacitance.diagonal().within(), "exactly one threshold"),
        (
            lambda: Capacitance.diagonal().within(relative=1e-3, absolute_f=1e-18),
            "exactly one threshold",
        ),
        (lambda: ModeFrequency().within(relative=0.0), "positive finite number"),
        (lambda: SurfaceEpr("MA").within(absolute=float("nan")), "positive finite"),
        (lambda: ErrorIndicator().below(-1.0), "positive finite number"),
        (lambda: ErrorIndicator().below(True), "positive finite number"),
        (lambda: Capacitance.sub_diagonal(0), "integer of at least 1"),
        (lambda: Capacitance.elements([]), "non-empty list"),
        (lambda: Capacitance.elements([("a", "b"), ("a", "b")]), "each pair once"),
        (lambda: Capacitance.element("a", ""), "two non-empty terminal names"),
        (lambda: ModeFrequency(modes=[0]), "mode numbers from 1"),
        (lambda: ModeFrequency(modes=[1, 1]), "distinct mode numbers"),
        (lambda: SurfaceEpr("MX"), "one of"),
        (lambda: SurfaceEpr("MA", margin_um=0), "positive number of micrometres"),
        (lambda: SurfaceEpr("MA", modes=[1], terminals=["a"]), "at once"),
        (lambda: SurfaceEpr.each(), "at least one of"),
        (lambda: SurfaceEpr.each("MA", "MA"), "each interface kind once"),
    ],
)
def test_an_invalid_quantity_or_threshold_is_refused_at_the_call(build, message):
    """Each refusal names the call to change."""
    with pytest.raises(ValueError, match=message) as refused:
        build()
    assert "Change it in the call to" in str(refused.value)


def test_each_declares_one_criterion_per_kind_with_one_threshold():
    """``SurfaceEpr.each`` is one declaration that becomes a criterion per interface kind."""
    sim = ElectrostaticSim()
    sim.set_convergence(
        SurfaceEpr.each("MA", "MS", "SA", margin_um=1.0).within(relative=2e-2)
    )
    assert sim.convergence is not None
    assert [item.label for item in sim.convergence.criteria] == [
        "MA participation, 1 um or more from the gap edge",
        "MS participation, 1 um or more from the gap edge",
        "SA participation, 1 um or more from the gap edge",
    ]
    assert {item.threshold for item in sim.convergence.criteria} == {2e-2}
    assert sim.convergence.minimum_converged_passes == 1


@pytest.mark.parametrize(
    ("sim_type", "criterion", "message"),
    [
        (
            EigenmodeSim,
            Capacitance.diagonal().within(relative=1e-3),
            "does not fit EigenmodeSim, which takes ModeFrequency, SurfaceEpr, ErrorIndicator",
        ),
        (
            ElectrostaticSim,
            ModeFrequency().within(relative=1e-3),
            "ElectrostaticSim, which takes Capacitance, SurfaceEpr, ErrorIndicator",
        ),
        (
            ElectrostaticSim,
            SurfaceEpr("MA", modes=[1]).within(relative=1e-3),
            "whose rows are terminals",
        ),
        (
            EigenmodeSim,
            SurfaceEpr("MA", terminals=["a"]).within(relative=1e-3),
            "whose rows are modes",
        ),
        (ElectrostaticSim, Capacitance.diagonal(), "takes criteria, a quantity of"),
    ],
)
def test_a_quantity_that_does_not_fit_the_simulation_is_refused(
    sim_type, criterion, message
):
    """A wrong declaration is refused before anything runs, naming what the class takes."""
    with pytest.raises(ValueError, match=message) as refused:
        sim_type().set_convergence(criterion)
    assert "Change it in the call to set_convergence" in str(refused.value)


@pytest.mark.parametrize(
    ("criterion", "message"),
    [
        (
            Criterion(Capacitance.diagonal(), "at_most", 0.001),
            r"tests 'at_most', and Capacitance offers \['relative_change', 'absolute_change'\]",
        ),
        (
            Criterion(ErrorIndicator(), "relative_change", 0.01),
            r"ErrorIndicator offers \['at_most'\]",
        ),
        (
            Criterion(ModeFrequency(), "relative_change", float("nan")),
            "threshold must be a positive finite number",
        ),
        (
            Criterion(SurfaceEpr("MA"), "absolute_change", -1.0),
            "threshold must be a positive finite number",
        ),
        (Criterion(ErrorIndicator(), "at_most", True), "positive finite number"),
    ],
)
def test_a_hand_built_criterion_goes_through_the_same_checks(criterion, message):
    """``set_convergence`` refuses a record the fluent calls could not have built (review round 1, B1).

    A value test on a capacitance would hold at pass 1 and "converge" there; a NaN threshold never compares.
    """
    sim_type = (
        EigenmodeSim
        if isinstance(criterion.quantity, ModeFrequency)
        else ElectrostaticSim
    )
    with pytest.raises(ValueError, match=message) as refused:
        sim_type().set_convergence(criterion)
    assert "Change it in the call to set_convergence" in str(refused.value)


def test_an_empty_group_declares_nothing_and_is_refused():
    """``CriterionGroup(())`` flattens to no criterion, which is refused like ``set_convergence()``."""
    with pytest.raises(ValueError, match="names no criterion"):
        ElectrostaticSim().set_convergence(CriterionGroup(()))


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Capacitance("element"), "names exactly one"),
        (lambda: Capacitance("elements"), "non-empty list"),
        (lambda: Capacitance("sub_diagonal"), "integer of at least 1"),
        (lambda: Capacitance("diagonal", offset=2), "offset must be None"),
        (lambda: Capacitance("rows"), "selection must be one of"),
    ],
)
def test_a_hand_built_capacitance_goes_through_the_same_checks(build, message):
    """The class methods and the constructor share one check of the selection."""
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize("passes", [0, 1.5, True, "2"])
def test_minimum_converged_passes_is_an_integer_of_at_least_one(passes):
    """``minimum_converged_passes`` is refused unless it is an int >= 1."""
    with pytest.raises(ValueError, match="integer of at least 1"):
        ElectrostaticSim().set_convergence(
            ErrorIndicator().below(1e-2), minimum_converged_passes=passes
        )


def test_no_criterion_is_refused_and_none_removes_the_declaration():
    """``set_convergence()`` names the way to remove a declaration; ``None`` alone is that way."""
    sim = ElectrostaticSim()
    with pytest.raises(ValueError, match=r"set_convergence\(None\) to remove"):
        sim.set_convergence()
    sim.set_convergence(ErrorIndicator().below(1e-2))
    with pytest.raises(ValueError, match="alone"):
        sim.set_convergence(None, ErrorIndicator().below(1e-2))
    assert sim.convergence is not None
    sim.set_convergence(None)
    assert sim.convergence is None


def test_a_call_replaces_the_whole_declaration():
    """Calling again replaces every criterion and the pass count."""
    sim = EigenmodeSim()
    sim.set_convergence(
        ModeFrequency().within(relative=1e-4), minimum_converged_passes=3
    )
    sim.set_convergence(ErrorIndicator().below(1e-2))
    assert sim.convergence is not None
    assert [item.label for item in sim.convergence.criteria] == [
        "error indicator (Norm)"
    ]
    assert sim.convergence.minimum_converged_passes == 1


def test_set_convergence_keeps_the_mesh_and_discards_the_config(tmp_path):
    """A problem setting: the simulation falls back to ``MESHED``."""
    sim = electrostatic(tmp_path)
    sim.set_numerical(amr_max_passes=2)
    sim.write_config()
    assert sim._stage is Stage.CONFIG_WRITTEN  # noqa: SLF001
    sim.set_convergence(ErrorIndicator().below(1e-2))
    assert sim._stage is Stage.MESHED  # noqa: SLF001
    assert sim.config_path is None


# ---- What write_config() checks and writes --------------------------------------------------------------


def _plan(sim) -> dict:
    return json.loads((sim.output_dir / CONVERGENCE_PLAN).read_text(encoding="utf-8"))


def _config(sim) -> dict:
    return json.loads((sim.output_dir / CONFIG).read_text(encoding="utf-8"))


def test_the_plan_reads_the_maxwell_matrix_by_terminal_order(tmp_path):
    """Terminals are rows and columns in the order they were added; each pair is read once."""
    sim = electrostatic(tmp_path)
    sim.set_numerical(amr_max_passes=3)
    sim.set_convergence(
        Capacitance.matrix().within(relative=1e-3),
        Capacitance.sub_diagonal().within(absolute_f=1e-18),
        minimum_converged_passes=2,
    )
    sim.write_config()
    plan = _plan(sim)
    assert plan["schema"] == "scgsim.palace.convergence-plan"
    assert plan["problem"] == "Electrostatic"
    assert plan["minimum_converged_passes"] == 2
    matrix, sub_diagonal = plan["criteria"]
    assert [
        (value["label"], value["row"], value["columns"]) for value in matrix["values"]
    ] == [
        ("C[pad][pad]", {"column": "i", "index": 1}, ["C[i][1] (F)"]),
        ("C[plane][plane]", {"column": "i", "index": 2}, ["C[i][2] (F)"]),
        ("C[pad][plane]", {"column": "i", "index": 1}, ["C[i][2] (F)"]),
    ]
    assert sub_diagonal["unit"] == "F"
    assert [value["label"] for value in sub_diagonal["values"]] == ["C[pad][plane]"]
    assert all(value["table"] == "terminal-C.csv" for value in matrix["values"])


def test_the_plan_takes_a_participation_from_the_index_map_entries(tmp_path):
    """Whole-surface MA sums the whole-surface MA rows; MA beyond 1 um sums the margin rows at 1 um."""
    sim = electrostatic(tmp_path)
    sim.set_numerical(amr_max_passes=1)
    sim.set_convergence(
        SurfaceEpr("MA").within(relative=1e-2),
        SurfaceEpr("MA", margin_um=1.0, terminals=["plane"]).within(relative=1e-2),
    )
    sim.write_config()
    entries = json.loads((sim.output_dir / INDEX_MAP).read_text(encoding="utf-8"))[
        "entries"
    ]
    ma = [
        entry
        for entry in entries
        if entry["section"] == "Boundaries.Postprocessing.Dielectric"
        and entry["metadata"]["rows"][0]["interface_type"] == "MA"
    ]
    whole = [f"p_surf[{entry['index']}]" for entry in ma if "inset" not in entry]
    beyond = [
        f"p_surf[{entry['index']}]"
        for entry in ma
        if entry.get("inset", {}).get("margin_um") == 1.0
    ]
    assert whole and beyond and not set(whole) & set(beyond)
    surfaces, margins = _plan(sim)["criteria"]
    assert [value["columns"] for value in surfaces["values"]] == [whole, whole]
    assert [value["label"] for value in surfaces["values"]] == [
        "MA participation, excitation pad",
        "MA participation, excitation plane",
    ]
    assert [(value["row"], value["columns"]) for value in margins["values"]] == [
        ({"column": "i", "index": 2}, beyond)
    ]


def test_an_eigenmode_plan_reads_each_mode(tmp_path):
    """``ModeFrequency()`` reads every computed mode, ``modes=`` the chosen ones, by ``m``."""
    sim = eigenmode(tmp_path, num_modes=3)
    sim.set_numerical(amr_max_passes=2, amr_update_fraction=0.7)
    sim.set_convergence(
        ModeFrequency().within(relative=1e-5),
        SurfaceEpr("SA", modes=[2]).within(relative=1e-2),
    )
    sim.write_config()
    frequency, participation = _plan(sim)["criteria"]
    assert [value["row"]["index"] for value in frequency["values"]] == [1, 2, 3]
    assert {value["columns"][0] for value in frequency["values"]} == {"Re{f} (GHz)"}
    assert frequency["unit"] == "GHz"
    assert [value["row"] for value in participation["values"]] == [
        {"column": "m", "index": 2}
    ]


@pytest.mark.parametrize("problem", ["Electrostatic", "Eigenmode"])
def test_a_declaration_takes_the_place_of_palace_tolerance(problem, tmp_path):
    """``Tol`` is the smallest normal double and every pass is saved; Eigenmode needs no amr_tolerance."""
    if problem == "Electrostatic":
        sim = electrostatic(tmp_path)
        sim.set_numerical(amr_max_passes=4)
    else:
        sim = eigenmode(tmp_path)
        sim.set_numerical(amr_max_passes=4, amr_update_fraction=0.7)
    sim.set_convergence(ErrorIndicator().below(1e-2))
    sim.write_config()
    refinement = _config(sim)["Model"]["Refinement"]
    assert refinement["Tol"] == DECLARED_CONVERGENCE_TOL > 0.0
    assert refinement["SaveAdaptIterations"] is True
    assert refinement["MaxIts"] == 4


def test_without_a_declaration_the_refinement_and_files_are_as_before(tmp_path):
    """No plan file, and Palace's own Tol and SaveAdaptIterations as the controls give them."""
    sim = electrostatic(tmp_path)
    sim.set_numerical(amr_max_passes=4, save_adapt_iterations=False)
    sim.write_config()
    refinement = _config(sim)["Model"]["Refinement"]
    assert refinement["Tol"] == 1e-2
    assert refinement["SaveAdaptIterations"] is False
    assert not (sim.output_dir / CONVERGENCE_PLAN).exists()


def test_removing_the_declaration_removes_the_plan(tmp_path):
    """A config written after ``set_convergence(None)`` leaves no plan of the earlier one."""
    sim = electrostatic(tmp_path)
    sim.set_numerical(amr_max_passes=2)
    sim.set_convergence(ErrorIndicator().below(1e-2))
    sim.write_config()
    assert (sim.output_dir / CONVERGENCE_PLAN).is_file()
    sim.set_convergence(None)
    sim.write_config()
    assert not (sim.output_dir / CONVERGENCE_PLAN).exists()
    assert "SaveAdaptIterations" not in _config(sim)["Model"]["Refinement"]


@pytest.mark.parametrize(
    ("declare", "message", "where"),
    [
        (
            lambda sim: sim.set_convergence(
                Capacitance.element("pad", "gnd").within(relative=1e-3)
            ),
            r"names terminals \['gnd'\] the simulation does not have; it has \['pad', 'plane'\]",
            "set_convergence and add_terminal",
        ),
        (
            lambda sim: sim.set_convergence(
                Capacitance.sub_diagonal(2).within(relative=1e-3)
            ),
            "selects no element of the 2-terminal matrix",
            "set_convergence and add_terminal",
        ),
        (
            lambda sim: sim.set_convergence(
                SurfaceEpr("MA", terminals=["feed"]).within(relative=1e-3)
            ),
            r"selects terminals \['feed'\]",
            "set_convergence and add_terminal",
        ),
        (
            lambda sim: sim.set_convergence(
                SurfaceEpr("SA", margin_um=1.0).within(relative=1e-3)
            ),
            r"margin 1 um, which the SA spec does not declare; it declares no margin",
            "set_convergence and set_surface_epr",
        ),
        (
            lambda sim: sim.set_convergence(
                SurfaceEpr("MA", margin_um=0.75).within(relative=1e-3)
            ),
            r"its margins are \[0.5, 1.0\] um",
            "set_convergence and set_surface_epr",
        ),
    ],
)
def test_a_criterion_the_electrostatic_config_cannot_satisfy_is_refused(
    declare, message, where, tmp_path
):
    """Each write-time refusal names what is missing and the calls that change it; nothing is written."""
    sim = electrostatic(tmp_path)
    sim.set_numerical(amr_max_passes=2)
    declare(sim)
    with pytest.raises(ValueError, match=message) as refused:
        sim.write_config()
    assert f"Change it in the calls to {where}" in str(refused.value)
    assert not (sim.output_dir / CONFIG).exists()
    assert not (sim.output_dir / CONVERGENCE_PLAN).exists()


def test_a_kind_without_surface_epr_is_refused(tmp_path):
    """A participation criterion needs Surface EPR requested for its kind."""
    sim = electrostatic(tmp_path, surface_epr=False)
    sim.set_numerical(amr_max_passes=2)
    sim.set_convergence(SurfaceEpr("MS").within(relative=1e-3))
    with pytest.raises(
        ValueError,
        match=r"Surface EPR is not requested for MS: call set_surface_epr\(\)",
    ):
        sim.write_config()


def test_eigenmode_modes_and_the_epr_selection_are_checked(tmp_path):
    """A mode beyond ``num_modes`` and a kind ``set_epr_request`` leaves out are refused."""
    sim = eigenmode(tmp_path, num_modes=2)
    sim.set_numerical(amr_max_passes=2, amr_update_fraction=0.7)
    sim.set_convergence(ModeFrequency(modes=[3]).within(relative=1e-5))
    with pytest.raises(
        ValueError, match=r"selects modes \[3\], and the run computes 2"
    ) as refused:
        sim.write_config()
    assert "set_convergence and set_eigenmode" in str(refused.value)
    sim.set_convergence(SurfaceEpr("SA").within(relative=1e-2))
    sim.set_epr_request(surface_interfaces=["MA"])
    with pytest.raises(
        ValueError, match=r"set_epr_request\(surface_interfaces=...\) leaves out"
    ) as refused:
        sim.write_config()
    assert "set_convergence and set_epr_request" in str(refused.value)


def test_too_few_passes_for_the_declaration_are_refused(tmp_path):
    """``amr_max_passes`` refinements solve one more time; a change needs one pass more than in a row."""
    sim = electrostatic(tmp_path)
    sim.set_numerical(amr_max_passes=1)
    sim.set_convergence(
        Capacitance.diagonal().within(relative=1e-3), minimum_converged_passes=2
    )
    with pytest.raises(
        ValueError, match=r"allows 2 solve\(s\), and the declaration needs at least 3"
    ) as refused:
        sim.write_config()
    assert r"set_numerical(amr_max_passes=2)" in str(refused.value)
    sim.set_numerical(amr_max_passes=0)
    sim.set_convergence(ErrorIndicator().below(1e-2))
    sim.write_config()


def test_a_stated_amr_tolerance_is_refused_with_the_criterion_to_write(tmp_path):
    """Electrostatic: a stated tolerance is refused, the default one is set aside."""
    sim = electrostatic(tmp_path)
    sim.set_numerical(amr_max_passes=2, amr_tolerance=0.02)
    sim.set_convergence(Capacitance.diagonal().within(relative=1e-3))
    with pytest.raises(
        ValueError, match=r"add the criterion ErrorIndicator\(\)\.below\(0\.02\)"
    ) as refused:
        sim.write_config()
    assert "set_numerical() without amr_tolerance" in str(refused.value)
    sim.set_numerical(amr_max_passes=2)
    sim.write_config()


def test_an_eigenmode_tolerance_kept_by_set_numerical_is_refused(tmp_path):
    """Eigenmode keeps an omitted control, so the refusal says to build controls without it."""
    sim = meshed_eigenmode_with_tolerance(tmp_path)
    sim.set_convergence(ModeFrequency().within(relative=1e-5))
    with pytest.raises(
        ValueError, match="EigenmodeSim.set_numerical keeps an omitted amr_tolerance"
    ):
        sim.write_config()


def meshed_eigenmode_with_tolerance(tmp_path) -> EigenmodeSim:
    """An Eigenmode simulation whose tolerance was stated once and kept by a later ``set_numerical``."""
    sim = eigenmode(tmp_path)
    sim.set_numerical(amr_max_passes=2, amr_tolerance=1e-3, amr_update_fraction=0.7)
    sim.set_numerical(order=2)
    assert sim.numerical.amr_tolerance == 1e-3
    return sim


def test_save_adapt_iterations_false_is_refused(tmp_path):
    """The check reads every pass, so the passes must be saved."""
    sim = electrostatic(tmp_path)
    sim.set_numerical(amr_max_passes=2, save_adapt_iterations=False)
    sim.set_convergence(ErrorIndicator().below(1e-2))
    with pytest.raises(ValueError, match="keeps Palace from saving each pass"):
        sim.write_config()


def test_constructor_controls_count_as_stated():
    """``numerical=NumericalControls(amr_tolerance=...)`` states the tolerance; the default does not."""
    stated = ElectrostaticSim(numerical=NumericalControls(amr_tolerance=0.01))
    assert stated._amr_tolerance_stated  # noqa: SLF001
    assert not ElectrostaticSim()._amr_tolerance_stated  # noqa: SLF001
    assert ElectrostaticSim().numerical.amr_tolerance == 1e-2
