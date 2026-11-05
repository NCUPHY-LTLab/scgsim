"""The repair loop's decisions, with the mesher replaced by a scripted stand-in."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from meshing_support import audit_of, bad_element, features_of, film, outcome_of

from scgsim.meshing import repair
from scgsim.meshing.policy import DELAUNAY, HXT, BoundingBox, RuleSet
from scgsim.meshing.repair import Budgets, RepairInputs, refine_sizes, repair_mesh

# A model whose coarse size is 20 um (ceiling 10), so the ladder is 10, 5, 2.5, 1.25, ...
BOX = BoundingBox((0.0, 0.0, 0.0), (400.0, 0.0, 0.0))
LONG = 100.0  # a curve length well above the 1 um floor


def inputs_for(
    tmp_path: Path, *entries, budgets: Budgets | None = None
) -> RepairInputs:
    """The loop's inputs for a model with the given features and a 20 um coarse size."""
    return RepairInputs(
        xao=tmp_path / "case.xao",
        sidecar=tmp_path / "case.sidecar.json",
        features=features_of(*entries),
        bounding_box=BOX,
        threshold=0.01,
        ruleset=RuleSet(),
        budgets=budgets or Budgets(),
        work_dir=tmp_path,
    )


def script_rounds(monkeypatch, decide):
    """Replace the mesher: ``decide(job)`` returns the audit of the round (or ``None`` for a crash)."""
    jobs = []

    def fake_run_round(job, work_dir):
        jobs.append(job)
        return outcome_of(decide(job))

    monkeypatch.setattr(repair, "run_round", fake_run_round)
    return jobs


def bad_until_finer_than(
    curve: int, size: float, *, local_edge: float = 8.0, tetrahedra: int = 1000
):
    """A mesher that leaves one bad element at ``curve`` until that curve's size is at most ``size``."""

    def decide(job):
        if job.sizes.curve_sizes.get(curve, float("inf")) <= size:
            return audit_of(tetrahedra=tetrahedra)
        return audit_of(
            (bad_element(1, [(curve, local_edge, LONG)]),), tetrahedra=tetrahedra
        )

    return decide


def test_a_clean_first_mesh_stops_at_once_with_the_a_priori_sizes(
    tmp_path, monkeypatch
):
    """A clean first mesh stops at once with the a priori sizes."""
    jobs = script_rounds(monkeypatch, lambda job: audit_of())
    outcome = repair_mesh(inputs_for(tmp_path, film([1, 2])))
    assert (outcome.stop, outcome.rounds, len(jobs)) == ("clean", 0, 1)
    assert dict(jobs[0].sizes.curve_sizes) == {
        1: 5.0,
        2: 5.0,
    }  # 25 x 0.2 = 5 um is exactly the first rung
    assert outcome.curve_sizes == {1: 5.0, 2: 5.0}


def test_an_untouched_curve_is_refined_to_half_the_bad_edge_on_the_ladder(
    tmp_path, monkeypatch
):
    # Bad element at curve 7: local edge 8 um, longest element edge 9 um -> 0.5 x 8 = 4 -> rung 2.5.
    """An untouched curve is refined to half the bad edge on the ladder."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(7, 2.5))
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert (outcome.stop, outcome.rounds) == ("clean", 1)
    assert dict(jobs[1].sizes.curve_sizes)[7] == 2.5
    assert (
        outcome.history[0].refined_curves == 1
        and outcome.history[1].refined_curves == 2
    )


def test_a_curve_with_a_film_size_goes_one_step_finer_even_for_a_long_bad_edge(
    tmp_path, monkeypatch
):
    # Curve 1 starts at the film size 5.  A bad edge of 40 um alone would ask for 20 -> rung 5 again; the
    # curve must still go one step finer than it was.
    """A curve with a film size goes one step finer even for a long bad edge."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(1, 2.5, local_edge=40.0))
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert dict(jobs[1].sizes.curve_sizes)[1] == 2.5
    assert outcome.stop == "clean"


def test_a_curve_that_stays_touched_halves_each_round(tmp_path, monkeypatch):
    """A curve that stays touched halves each round."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(1, 0.3))
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert [dict(j.sizes.curve_sizes)[1] for j in jobs] == [
        5.0,
        2.5,
        1.25,
        0.625,
        0.3125,
        0.15625,
    ]
    assert outcome.stop == "clean" and outcome.rounds == 5


def test_curves_shorter_than_one_micron_are_never_refined(tmp_path, monkeypatch):
    """Curves shorter than one micron are never refined."""
    short_vertical = bad_element(1, [(9, 8.0, 0.2)])  # a 0.2 um vertical wall edge
    jobs = script_rounds(monkeypatch, lambda job: audit_of((short_vertical,)))
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert (outcome.stop, outcome.rounds, len(jobs)) == ("unattributed", 0, 1)


def test_only_the_culprit_curves_of_an_element_are_refined():
    """Only the culprit curves of an element are refined."""
    element = bad_element(1, [(1, 8.0, LONG), (2, 3.0, LONG), (3, 4.0, LONG)])
    refined = refine_sizes({}, (element,), {}, 10.0, RuleSet())
    assert set(refined) == {1, 3}  # local edges 8 and 4 reach half of 8; 3 does not


def test_the_largest_demand_over_the_elements_of_a_curve_wins():
    """The largest demand over the elements of a curve wins."""
    gentle = bad_element(1, [(1, 3.0, LONG)], edge_max=3.0)
    harsh = bad_element(2, [(1, 8.0, LONG)], edge_max=9.0)
    assert refine_sizes({}, (gentle, harsh), {}, 10.0, RuleSet()) == {
        1: 2.5
    }  # limit 8 -> 4 -> rung 2.5


def test_elements_in_a_face_or_the_interior_cannot_be_refined(tmp_path, monkeypatch):
    """Elements in a face or the interior cannot be refined."""
    stuck = bad_element(1, [], location="interior")
    jobs = script_rounds(monkeypatch, lambda job: audit_of((stuck,)))
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert outcome.stop == "unattributed" and len(jobs) == 1
    assert outcome.last_audit is not None and outcome.last_audit.bad_count == 1


def test_the_round_budget_ends_the_loop_after_that_many_refinements(
    tmp_path, monkeypatch
):
    """The round budget ends the loop after that many refinements."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(1, 0.0))
    outcome = repair_mesh(
        inputs_for(tmp_path, film([1]), budgets=Budgets(max_rounds=3))
    )
    assert (outcome.stop, outcome.rounds, len(jobs)) == ("rounds", 3, 4)


def test_the_default_budget_is_eight_rounds_and_two_million_tetrahedra():
    """The default budget is eight rounds and two million tetrahedra."""
    assert (Budgets().max_rounds, Budgets().max_tetrahedra) == (8, 2_000_000)


def test_the_tetrahedron_budget_ends_a_loop_that_is_still_bad(tmp_path, monkeypatch):
    """The tetrahedron budget ends a loop that is still bad."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(1, 0.0, tetrahedra=3000))
    outcome = repair_mesh(
        inputs_for(
            tmp_path, film([1]), budgets=Budgets(max_rounds=8, max_tetrahedra=2000)
        )
    )
    assert (outcome.stop, outcome.rounds, len(jobs)) == ("elements", 0, 1)


def test_a_clean_mesh_over_the_tetrahedron_budget_is_rejected(tmp_path, monkeypatch):
    """The budget caps every accepted mesh: a clean round over it stops with ``elements``, not ``clean``."""
    jobs = script_rounds(monkeypatch, lambda job: audit_of(tetrahedra=5000))
    outcome = repair_mesh(
        inputs_for(tmp_path, film([1]), budgets=Budgets(max_tetrahedra=10))
    )
    assert (outcome.stop, outcome.rounds, len(jobs)) == ("elements", 0, 1)
    assert outcome.last_mesh == jobs[0].msh


def test_a_crash_keeps_the_last_mesh_that_was_made(tmp_path, monkeypatch):
    """A crash keeps the last mesh that was made."""

    def decide(job):
        if len(job.sizes.curve_sizes) > 1:
            return None
        return audit_of((bad_element(1, [(7, 8.0, LONG)]),))

    jobs = script_rounds(monkeypatch, decide)
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert (outcome.stop, outcome.rounds, len(jobs)) == ("mesh_crash", 1, 2)
    assert outcome.last_mesh == jobs[0].msh
    assert outcome.history[-1].failure and outcome.history[-1].tetrahedra is None
    assert set(outcome.curve_sizes) == {1}


def test_a_crash_in_the_first_round_leaves_no_mesh(tmp_path, monkeypatch):
    """A crash in the first round leaves no mesh."""
    script_rounds(monkeypatch, lambda job: None)
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert (outcome.stop, outcome.last_mesh, outcome.last_audit) == (
        "mesh_crash",
        None,
        None,
    )


def test_two_runs_make_the_same_decisions(tmp_path, monkeypatch):
    """Two runs make the same decisions."""

    def run():
        script_rounds(monkeypatch, bad_until_finer_than(7, 0.6))
        return repair_mesh(inputs_for(tmp_path, film([1, 2, 3])))

    first, second = run(), run()
    assert first.history == second.history
    assert first.curve_sizes == second.curve_sizes
    assert first.stop == second.stop == "clean"


def test_every_round_writes_its_own_mesh_file(tmp_path, monkeypatch):
    """Every round writes its own mesh file."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(7, 1.0))
    repair_mesh(inputs_for(tmp_path, film([1])))
    assert [j.msh.name for j in jobs] == [
        f"round_{i:02d}.msh" for i in range(len(jobs))
    ]


def test_the_size_table_never_loosens(tmp_path, monkeypatch):
    """The size table never loosens."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(7, 0.3))
    repair_mesh(inputs_for(tmp_path, film([1])))
    for before, after in zip(jobs, jobs[1:], strict=False):
        for tag, size in before.sizes.curve_sizes.items():
            assert after.sizes.curve_sizes[tag] <= size


@pytest.mark.parametrize("rounds", [0, 2])
def test_a_budget_of_n_rounds_allows_n_plus_one_meshes(tmp_path, monkeypatch, rounds):
    """A budget of n rounds allows n plus one meshes."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(1, 0.0))
    repair_mesh(inputs_for(tmp_path, film([1]), budgets=Budgets(max_rounds=rounds)))
    assert len(jobs) == rounds + 1


def test_the_rule_set_chooses_the_algorithm_of_every_round(tmp_path, monkeypatch):
    """Delaunay by default; HXT when the rule set says so."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(7, 1.0))
    repair_mesh(inputs_for(tmp_path, film([1])))
    assert {job.algorithm_3d for job in jobs} == {DELAUNAY}
    jobs.clear()
    hxt_inputs = replace(
        inputs_for(tmp_path, film([1])), ruleset=RuleSet(algorithm_3d=HXT)
    )
    repair_mesh(hxt_inputs)
    assert {job.algorithm_3d for job in jobs} == {HXT}
