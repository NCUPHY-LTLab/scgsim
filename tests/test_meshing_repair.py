"""The repair loop's decisions, with the mesher replaced by a scripted stand-in."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from meshing_support import audit_of, bad_element, features_of, film, outcome_of

from scgsim.meshing import (
    DELAUNAY,
    HXT,
    FeatureRecipe,
    GlobalSizeRecipe,
    MesherOptions,
    repair,
)
from scgsim.meshing.repair import (
    Budgets,
    RepairInputs,
    RepairPolicy,
    refine_locally,
    repair_mesh,
    stalled,
    wanted_size,
)
from scgsim.meshing.sizing import BoundingBox, LocalSize, SizeRequests

# A model whose coarse size is 20 um (ceiling 10), so the ladder is 10, 5, 2.5, 1.25, ...
BOX = BoundingBox((0.0, 0.0, 0.0), (400.0, 0.0, 0.0))
THRESHOLD = 0.01
SPOT = (1.0, 2.0, 3.0)


def inputs_for(
    tmp_path: Path,
    *entries,
    budgets: Budgets | None = None,
    initial: SizeRequests | None = None,
    **overrides,
) -> RepairInputs:
    """The loop's inputs for a model with the given features and a 20 um coarse size."""
    inputs = RepairInputs(
        xao=tmp_path / "case.xao",
        sidecar=tmp_path / "case.sidecar.json",
        initial=initial or FeatureRecipe().size_requests(features_of(*entries), BOX),
        threshold=THRESHOLD,
        policy=RepairPolicy(),
        budgets=budgets or Budgets(),
        mesher=MesherOptions(),
        work_dir=tmp_path,
    )
    return replace(inputs, **overrides)


def script_rounds(monkeypatch, decide):
    """Replace the mesher: ``decide(job)`` returns the audit of the round (or ``None`` for a crash)."""
    jobs = []

    def fake_run_round(job, work_dir, timeout_seconds=None):
        jobs.append(job)
        return outcome_of(decide(job))

    monkeypatch.setattr(repair, "run_round", fake_run_round)
    return jobs


def local_size_at(job, spot) -> float:
    """The local size a job asks for at ``spot``; infinite when it asks for none there."""
    return min(
        (ball.size for ball in job.sizes.local_sizes if ball.centre == spot),
        default=float("inf"),
    )


def bad_until_finer_than(
    size: float, *, spot=SPOT, quality: float = 0.009, tetrahedra: int = 1000
):
    """A mesher that leaves one bad element at ``spot`` until the local size there is at most ``size``.

    The default quality is just below the threshold, so the element asks for half its longest edge.
    """

    def decide(job):
        if local_size_at(job, spot) <= size:
            return audit_of(tetrahedra=tetrahedra)
        return audit_of(
            (bad_element(1, at=spot, edge_max=8.0, quality=quality),),
            tetrahedra=tetrahedra,
        )

    return decide


def never_clean(*, tetrahedra: int = 1000, start: int = 20):
    """A mesher whose bad count falls by one each round and never reaches zero, so no stall stops it."""
    rounds = []

    def decide(job):
        rounds.append(job)
        count = max(start - len(rounds), 1)
        bad = tuple(
            bad_element(t, at=(10.0 * t, 0.0, 0.0), quality=0.009) for t in range(count)
        )
        return audit_of(bad, tetrahedra=tetrahedra)

    return decide


def test_a_clean_first_mesh_stops_at_once_with_the_recipe_sizes_and_no_local_size(
    tmp_path, monkeypatch
):
    """A clean first mesh stops at once with the a priori sizes."""
    jobs = script_rounds(monkeypatch, lambda job: audit_of())
    outcome = repair_mesh(inputs_for(tmp_path, film([1, 2])))
    assert (outcome.stop, outcome.rounds, len(jobs)) == ("clean", 0, 1)
    # 25 x 0.2 = 5 um is exactly the first rung.
    assert dict(jobs[0].sizes.curve_sizes) == {1: 5.0, 2: 5.0}
    assert jobs[0].sizes.local_sizes == ()
    assert (outcome.curve_sizes, outcome.local_sizes) == ({1: 5.0, 2: 5.0}, ())


def test_a_bad_element_gets_a_local_size_where_it_is_thinnest(tmp_path, monkeypatch):
    """One local size at the midpoint of the element's shortest edge; the recipe's curve sizes stay."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(5.0))
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert (outcome.stop, outcome.rounds) == ("clean", 1)
    # Quality 0.009 is 0.9 of the threshold: 0.5 x 8 x 0.9 = 3.6 -> rung 2.5.
    assert jobs[1].sizes.local_sizes == (LocalSize(SPOT, 2.5),)
    assert dict(jobs[1].sizes.curve_sizes) == dict(jobs[0].sizes.curve_sizes)
    assert [r.local_sizes for r in outcome.history] == [0, 1]
    assert outcome.history[1].smallest_local_size == 2.5


def test_the_size_asked_for_scales_with_the_quality_deficit():
    """Half the long edge near the threshold, finer for a worse element, never below half the short edge."""
    policy = RepairPolicy()
    near = bad_element(1, edge_max=8.0, quality=0.009)
    assert wanted_size(near, THRESHOLD, policy) == pytest.approx(0.5 * 8.0 * 0.9)
    passing = bad_element(1, edge_max=8.0, quality=0.02)
    assert wanted_size(passing, THRESHOLD, policy) == pytest.approx(4.0)
    needle = bad_element(1, edge_max=5.0, edge_min=0.001, quality=0.0005)
    assert wanted_size(needle, THRESHOLD, policy) == pytest.approx(0.5 * 5.0 * 0.05)
    hopeless = bad_element(1, edge_max=5.0, edge_min=0.001, quality=0.000001)
    assert wanted_size(hopeless, THRESHOLD, policy) == pytest.approx(0.5 * 0.001)
    inverted = bad_element(1, edge_max=5.0, edge_min=0.001, quality=-0.3)
    assert wanted_size(inverted, THRESHOLD, policy) == pytest.approx(0.5 * 0.001)


def test_a_local_size_that_still_holds_a_bad_element_goes_at_least_one_step_finer():
    """An element still bad inside the core halves the size even when it asks for a coarser one."""
    element = bad_element(
        1, at=SPOT, edge_max=8.0, quality=0.009
    )  # asks for 3.6 -> 2.5
    local: tuple[LocalSize, ...] = ()
    sizes = []
    for _ in range(4):
        local = refine_locally(local, (element,), 20.0, THRESHOLD, RepairPolicy())
        sizes.append([ball.size for ball in local])
    assert sizes == [[2.5], [1.25], [0.625], [0.3125]]


def test_the_loop_refines_one_spot_again_in_the_next_round(tmp_path, monkeypatch):
    """The same element bad twice gets one local size, one ladder step finer, and the loop goes on."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(1.25))
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert [j.sizes.local_sizes for j in jobs[1:]] == [
        (LocalSize(SPOT, 2.5),),
        (LocalSize(SPOT, 1.25),),
    ]
    assert (outcome.stop, outcome.rounds) == ("clean", 2)


def test_an_element_outside_every_core_gets_its_own_local_size():
    """Claiming is by the fine core only: a nearby element outside it starts a size of its own."""
    first = bad_element(1, at=(0.0, 0.0, 0.0), edge_max=8.0, quality=0.009)
    inside = bad_element(2, at=(1.0, 0.0, 0.0), edge_max=8.0, quality=0.002)
    outside = bad_element(3, at=(3.0, 0.0, 0.0), edge_max=8.0, quality=0.009)
    refined = refine_locally(
        (), (first, inside, outside), 20.0, THRESHOLD, RepairPolicy()
    )
    # The first core is 2.5 wide.  Element 2 (1 um away) is in it and asks 0.5 x 8 x 0.2 = 0.8 -> 0.625,
    # which would shrink that needle core to 0.625 and leave element 2 outside: it gets its own size and
    # the first keeps element 1's 2.5.  Element 3, 3 um away, was never in it.
    assert refined == (
        LocalSize((0.0, 0.0, 0.0), 2.5),
        LocalSize((3.0, 0.0, 0.0), 2.5),
        LocalSize((1.0, 0.0, 0.0), 0.625),
    )


def field_at(local, point, max_size: float = 20.0, gradation: float = 1.0) -> float:
    """The size Gmsh's Ball fields give at ``point``: the finest over the sizes, graded outside each core."""
    return min(
        (
            min(
                max_size,
                ball.size
                + gradation
                * max(0.0, math.dist(ball.centre, point) - ball.core_radius),
            )
            for ball in local
        ),
        default=max_size,
    )


def test_a_needle_that_a_shrinking_core_leaves_gets_its_own_size():
    """The review's case: needles at x = 0 and 1, longest edges 8, qualities 0.009 and 0.002.

    Each location gets at least the size its element asked for, on the ladder: 2.5 at x = 0
    (0.5 x 8 x 0.9 = 3.6) and 0.625 at x = 1 (0.5 x 8 x 0.2 = 0.8).
    """
    at_zero = bad_element(1, at=(0.0, 0.0, 0.0), edge_max=8.0, quality=0.009)
    at_one = bad_element(2, at=(1.0, 0.0, 0.0), edge_max=8.0, quality=0.002)
    for order in ((at_zero, at_one), (at_one, at_zero)):
        refined = refine_locally((), order, 20.0, THRESHOLD, RepairPolicy())
        assert field_at(refined, (0.0, 0.0, 0.0)) <= 2.5
        assert field_at(refined, (1.0, 0.0, 0.0)) <= 0.625
        for element in order:
            spot = element.shortest_edge_midpoint
            assert any(
                math.dist(ball.centre, spot) <= ball.core_radius for ball in refined
            )


def test_an_existing_core_that_shrinks_away_from_its_elements_still_goes_finer():
    """A needle size whose claimed elements all fall outside its halved core halves and lets them go."""
    existing = LocalSize((0.0, 0.0, 0.0), 2.5)
    near = bad_element(1, at=(2.0, 0.0, 0.0), edge_max=8.0, quality=0.009)  # inside 2.5
    refined = refine_locally((existing,), (near,), 20.0, THRESHOLD, RepairPolicy())
    assert refined == (
        LocalSize((0.0, 0.0, 0.0), 1.25),
        LocalSize((2.0, 0.0, 0.0), 2.5),
    )


def co_centred_needle_and_sliver(needle_tag: int, sliver_tag: int):
    """A needle asking 0.078125 and a sliver asking 0.625 over a 3 um core, both at the origin."""
    needle = bad_element(needle_tag, edge_max=5.0, edge_min=0.001, quality=0.0005)
    sliver = bad_element(sliver_tag, edge_max=6.0, edge_min=2.0, quality=0.004)
    return needle, sliver


@pytest.mark.parametrize("needle_first", [True, False])
def test_a_sliver_beside_a_needle_keeps_its_whole_core_refined(needle_first):
    """The review's case: a needle's one-element core must not absorb a co-centred sliver's 3 um core.

    Alone, the sliver gets 0.625 over 3 um.  Whichever element comes first, every point of that range
    must still get 0.625 or finer, through the sliver's own size or one that covers it.
    """
    needle, sliver = co_centred_needle_and_sliver(*((1, 2) if needle_first else (2, 1)))
    refined = refine_locally((), (needle, sliver), 20.0, THRESHOLD, RepairPolicy())
    assert any(
        ball.size <= 0.625
        and math.dist(ball.centre, sliver.centroid) + 3.0 <= ball.core_radius
        for ball in refined
    )
    assert field_at(refined, (3.0, 0.0, 0.0)) <= 0.625
    assert field_at(refined, (0.0, 0.0, 0.0)) <= 0.078125


def test_a_size_whose_core_covers_a_sliver_s_core_is_shared():
    """A sliver whose whole 1 um core lies inside an existing 3 um core refines that size; no new one."""
    existing = LocalSize((0.0, 0.0, 0.0), 1.25, 3.0)
    # 0.5 x 2 x max(0.4, 1 / 2) = 0.5 -> 0.3125, over a 1 um core 1 um from the centre: 1 + 1 <= 3.
    sliver = bad_element(
        1, at=(1.0, 0.0, 0.0), edge_max=2.0, edge_min=1.0, quality=0.004
    )
    refined = refine_locally((existing,), (sliver,), 20.0, THRESHOLD, RepairPolicy())
    assert refined == (LocalSize((0.0, 0.0, 0.0), 0.3125, 3.0),)


def test_a_recipe_s_own_local_sizes_reach_every_round_unchanged(tmp_path, monkeypatch):
    """Local sizes a recipe asks for come first in every job; the loop adds its own after them."""
    recipe_size = LocalSize((5.0, 5.0, 5.0), 0.5, 2.0)
    initial = SizeRequests(20.0, 1.0, MappingProxyType({}), (), (recipe_size,))
    jobs = script_rounds(monkeypatch, bad_until_finer_than(2.5))
    outcome = repair_mesh(inputs_for(tmp_path, initial=initial))
    assert [job.sizes.local_sizes for job in jobs] == [
        (recipe_size,),
        (recipe_size, LocalSize(SPOT, 2.5)),
    ]
    assert outcome.stop == "clean"
    assert outcome.local_sizes == (LocalSize(SPOT, 2.5),)


def test_a_needle_is_refined_at_its_short_edge_and_a_sliver_over_its_extent():
    """An element with an edge shorter than its size gets a one-element core there; a flat one a wide core."""
    needle = replace(
        bad_element(1, edge_max=5.0, edge_min=0.001, quality=0.0005),
        centroid=(2.0, 0.0, 0.0),
        shortest_edge_midpoint=(0.0, 0.0, 0.0),
    )
    sliver = replace(
        bad_element(2, edge_max=6.0, edge_min=2.0, quality=0.004),
        centroid=(50.0, 0.0, 0.0),
        shortest_edge_midpoint=(51.0, 0.0, 0.0),
    )
    first, second = refine_locally(
        (), (needle, sliver), 20.0, THRESHOLD, RepairPolicy()
    )
    # Needle: 0.5 x 5 x 0.05 = 0.125 -> 0.078125, centred on the short edge, core as wide as the size.
    assert first == LocalSize((0.0, 0.0, 0.0), 0.078125)
    assert first.core_radius == 0.078125
    # Sliver: 0.5 x 6 x 0.4 = 1.2 (not above its 2 um shortest edge) -> 0.625, over half its longest edge.
    assert second == LocalSize((50.0, 0.0, 0.0), 0.625, 3.0)
    tightened = refine_locally(
        (first, second), (sliver,), 20.0, THRESHOLD, RepairPolicy()
    )
    assert tightened == (first, LocalSize((50.0, 0.0, 0.0), 0.3125, 3.0))


def test_the_same_bad_elements_always_give_the_same_local_sizes():
    """Elements are taken in tag order whatever order the audit lists them in."""
    elements = [
        bad_element(tag, at=(3.0 * tag, 0.0, 0.0), edge_max=8.0, quality=0.001 * tag)
        for tag in range(1, 6)
    ]
    forward = refine_locally((), tuple(elements), 20.0, THRESHOLD, RepairPolicy())
    backward = refine_locally(
        (), tuple(reversed(elements)), 20.0, THRESHOLD, RepairPolicy()
    )
    assert forward == backward
    assert [ball.centre[0] for ball in forward] == [3.0, 6.0, 9.0, 12.0, 15.0]


@pytest.mark.parametrize("location", ["surface", "interior", "point", "curve"])
def test_bad_elements_are_refined_wherever_they_are(tmp_path, monkeypatch, location):
    """An element that touches only a face, or nothing, is refined like one on a curve."""

    def decide(job):
        if local_size_at(job, SPOT) <= 2.5:
            return audit_of()
        return audit_of(
            (bad_element(1, at=SPOT, edge_max=8.0, quality=0.009, location=location),)
        )

    script_rounds(monkeypatch, decide)
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert (outcome.stop, outcome.rounds) == ("clean", 1)


def test_an_element_at_a_curve_shorter_than_a_micrometre_is_refined(
    tmp_path, monkeypatch
):
    """The 0.2 um vertical edge of a film wall no longer leaves its element unrepaired."""

    def decide(job):
        if local_size_at(job, SPOT) <= 0.1:
            return audit_of()
        return audit_of(
            (
                bad_element(
                    1,
                    [(9, 8.0, 0.2)],
                    at=SPOT,
                    edge_max=8.0,
                    edge_min=0.2,
                    quality=0.0001,
                ),
            )
        )

    jobs = script_rounds(monkeypatch, decide)
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert outcome.stop == "clean"
    # 0.0001 / 0.01 = 0.01, floored at 0.2 / 8 = 0.025: 0.5 x 8 x 0.025 = 0.1 -> rung 0.078125.
    assert local_size_at(jobs[1], SPOT) == 0.078125


def test_the_loop_stops_when_the_bad_count_does_not_fall_for_two_rounds(
    tmp_path, monkeypatch
):
    """Bad counts 5, 3, 3, 4: two rounds in a row without going below 3 stop the loop, with its report."""
    counts = iter([5, 3, 3, 4, 0])

    def decide(job):
        count = next(counts)
        return audit_of(
            tuple(
                bad_element(t, at=(10.0 * t, 0.0, 0.0), quality=0.009)
                for t in range(count)
            )
        )

    jobs = script_rounds(monkeypatch, decide)
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert (outcome.stop, outcome.rounds, len(jobs)) == ("stalled", 3, 4)
    assert outcome.last_audit is not None and outcome.last_audit.bad_count == 4
    assert outcome.last_mesh == jobs[-1].msh


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([5, 3, 3, 4], True),
        ([5, 6, 7], True),
        ([5, 3, 4, 2], False),
        ([5, 5], False),
        ([9, 8, 7, 6], False),
    ],
)
def test_a_stall_is_two_rounds_above_the_fewest_before_them(counts, expected):
    """The comparison is with the fewest earlier count, not with the previous round."""
    assert stalled(counts) is expected


def test_the_time_budget_scales_with_the_footprint_above_four_square_millimetres():
    """1 min target and 3 min limit up to 4 mm2; a 4 mm x 4 mm chip gets 4 and 12 min (Human, issue 124)."""
    budgets = Budgets()
    assert (budgets.time_limit_seconds, budgets.time_target_seconds) == (180.0, 60.0)
    assert budgets.time_reference_area_mm2 == 4.0
    for footprint in (0.5, 4.0):
        small = budgets.time_budget(footprint)
        assert (small.limit_seconds, small.target_seconds) == (180.0, 60.0)
    large = budgets.time_budget(16.0)
    assert (large.footprint_mm2, large.limit_seconds, large.target_seconds) == (
        16.0,
        720.0,
        240.0,
    )


@pytest.mark.parametrize(
    "name", ["time_limit_seconds", "time_target_seconds", "time_reference_area_mm2"]
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_a_time_budget_must_be_a_positive_number(name, value):
    """A zero, negative or non-finite time setting is refused when the budgets are made."""
    with pytest.raises(ValueError, match=name):
        Budgets(**{name: value})


def test_the_footprint_is_the_x_by_y_extent_in_square_millimetres():
    """The z extent does not count."""
    box = BoundingBox((-500.0, -1000.0, -1500.0), (500.0, 1000.0, 1500.0))
    assert box.footprint_mm2 == pytest.approx(2.0)


def test_each_round_is_given_the_time_that_is_left(tmp_path, monkeypatch):
    """The round's limit is the time to the deadline; with no deadline the round has none."""
    seen = []

    def fake_run_round(job, work_dir, timeout_seconds=None):
        seen.append(timeout_seconds)
        return outcome_of(audit_of())

    monkeypatch.setattr(repair, "run_round", fake_run_round)
    repair_mesh(inputs_for(tmp_path, film([1]), deadline=time.perf_counter() + 100.0))
    repair_mesh(inputs_for(tmp_path, film([1])))
    assert 99.0 < seen[0] <= 100.0
    assert seen[1] is None


def test_a_spent_deadline_stops_before_the_next_round(tmp_path, monkeypatch):
    """No round starts after the deadline: the loop stops with ``time_limit`` and the mesh it has."""
    clock = iter([0.0, 10.0, 100.0])
    monkeypatch.setattr(
        repair, "time", SimpleNamespace(perf_counter=lambda: next(clock))
    )
    jobs = script_rounds(monkeypatch, never_clean())
    outcome = repair_mesh(inputs_for(tmp_path, film([1]), deadline=50.0))
    assert (outcome.stop, outcome.rounds, len(jobs)) == ("time_limit", 1, 2)
    assert outcome.last_mesh == jobs[-1].msh


def test_a_deadline_spent_before_the_first_round_leaves_no_mesh(tmp_path, monkeypatch):
    """The probe can use up a tiny budget: the loop stops without meshing and says why."""
    jobs = script_rounds(monkeypatch, lambda job: audit_of())
    outcome = repair_mesh(
        inputs_for(tmp_path, film([1]), deadline=time.perf_counter() - 1)
    )
    assert (outcome.stop, outcome.rounds, outcome.history, jobs) == (
        "time_limit",
        0,
        (),
        [],
    )
    assert (outcome.last_mesh, outcome.last_audit) == (None, None)


def test_a_round_cut_off_at_the_time_limit_keeps_the_last_mesh(tmp_path, monkeypatch):
    """The killed round is in the history as timed out; the mesh of the round before stays the last one."""
    rounds = []

    def fake_run_round(job, work_dir, timeout_seconds=None):
        rounds.append(job)
        if len(rounds) == 1:
            return outcome_of(audit_of((bad_element(1, at=SPOT, quality=0.009),)))
        return replace(outcome_of(None), timed_out=True, failure="time limit")

    monkeypatch.setattr(repair, "run_round", fake_run_round)
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert (outcome.stop, outcome.rounds) == ("time_limit", 1)
    assert [r.timed_out for r in outcome.history] == [False, True]
    assert outcome.last_mesh == rounds[0].msh
    assert outcome.last_audit is not None and outcome.last_audit.bad_count == 1


def test_the_round_budget_ends_the_loop_after_that_many_refinements(
    tmp_path, monkeypatch
):
    """The round budget ends the loop after that many refinements."""
    jobs = script_rounds(monkeypatch, never_clean())
    outcome = repair_mesh(
        inputs_for(tmp_path, film([1]), budgets=Budgets(max_rounds=3))
    )
    assert (outcome.stop, outcome.rounds, len(jobs)) == ("rounds", 3, 4)


def test_the_default_budget_is_eight_rounds_and_two_million_tetrahedra():
    """The default budget is eight rounds and two million tetrahedra."""
    assert (Budgets().max_rounds, Budgets().max_tetrahedra) == (8, 2_000_000)


def test_the_tetrahedron_budget_ends_a_loop_that_is_still_bad(tmp_path, monkeypatch):
    """The tetrahedron budget ends a loop that is still bad."""
    jobs = script_rounds(monkeypatch, never_clean(tetrahedra=3000))
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
        if job.sizes.local_sizes:
            return None
        return audit_of((bad_element(1, at=SPOT, quality=0.009),))

    jobs = script_rounds(monkeypatch, decide)
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert (outcome.stop, outcome.rounds, len(jobs)) == ("mesh_crash", 1, 2)
    assert outcome.last_mesh == jobs[0].msh
    assert outcome.history[-1].failure and outcome.history[-1].tetrahedra is None
    assert outcome.local_sizes == ()


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
        script_rounds(monkeypatch, bad_until_finer_than(1.25))
        return repair_mesh(inputs_for(tmp_path, film([1, 2, 3])))

    first, second = run(), run()
    assert first.history == second.history
    assert first.local_sizes == second.local_sizes
    assert first.stop == second.stop == "clean"


def test_every_round_writes_its_own_mesh_file(tmp_path, monkeypatch):
    """Every round writes its own mesh file."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(1.0))
    repair_mesh(inputs_for(tmp_path, film([1])))
    assert [j.msh.name for j in jobs] == [
        f"round_{i:02d}.msh" for i in range(len(jobs))
    ]


def test_local_sizes_never_loosen(tmp_path, monkeypatch):
    """A local size keeps its place in the list and never gets coarser; the recipe's sizes never change."""
    spots = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0)]

    def decide(job):
        bad = tuple(
            bad_element(t, at=spot, edge_max=8.0, quality=0.009)
            for t, spot in enumerate(spots, start=1)
            if local_size_at(job, spot) > 0.3
        )
        return audit_of(bad)

    jobs = script_rounds(monkeypatch, decide)
    repair_mesh(inputs_for(tmp_path, film([1])))
    for before, after in zip(jobs, jobs[1:], strict=False):
        assert after.sizes.curve_sizes == before.sizes.curve_sizes
        for old, new in zip(
            before.sizes.local_sizes, after.sizes.local_sizes, strict=False
        ):
            assert new.centre == old.centre and new.size <= old.size


@pytest.mark.parametrize("rounds", [0, 2])
def test_a_budget_of_n_rounds_allows_n_plus_one_meshes(tmp_path, monkeypatch, rounds):
    """A budget of n rounds allows n plus one meshes."""
    jobs = script_rounds(monkeypatch, never_clean())
    repair_mesh(inputs_for(tmp_path, film([1]), budgets=Budgets(max_rounds=rounds)))
    assert len(jobs) == rounds + 1


def test_the_mesher_options_choose_the_algorithm_of_every_round(tmp_path, monkeypatch):
    """Delaunay by default; HXT when the mesher options say so."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(1.0))
    repair_mesh(inputs_for(tmp_path, film([1])))
    assert {job.mesher.algorithm_3d for job in jobs} == {DELAUNAY}
    jobs.clear()
    repair_mesh(inputs_for(tmp_path, film([1]), mesher=MesherOptions(algorithm_3d=HXT)))
    assert {job.mesher.algorithm_3d for job in jobs} == {HXT}


def test_the_repair_policy_default_is_half_the_longest_edge():
    """The policy is frozen data with one number."""
    assert RepairPolicy().edge_fraction == 0.5


def test_the_repair_policy_changes_the_size_first_asked_for():
    """A smaller edge fraction asks for a finer first size."""
    element = bad_element(1, at=SPOT, edge_max=8.0, quality=0.02)
    default = refine_locally((), (element,), 10.0, THRESHOLD, RepairPolicy())
    finer = refine_locally(
        (), (element,), 10.0, THRESHOLD, RepairPolicy(edge_fraction=0.2)
    )
    assert default == (LocalSize(SPOT, 2.5),)  # half of 8 -> 4 -> rung 2.5
    assert finer == (LocalSize(SPOT, 1.25),)  # 0.2 x 8 = 1.6 -> rung 1.25


def test_the_loop_repairs_whatever_recipe_chose_the_initial_sizes(
    tmp_path, monkeypatch
):
    """With no a priori sizes at all, the loop still refines around the bad element until the mesh is clean."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(2.5))
    initial = GlobalSizeRecipe().size_requests(features_of(film([1])), BOX)
    outcome = repair_mesh(inputs_for(tmp_path, initial=initial))
    assert dict(jobs[0].sizes.curve_sizes) == {} == dict(jobs[1].sizes.curve_sizes)
    assert jobs[1].sizes.local_sizes == (LocalSize(SPOT, 2.5),)
    assert (outcome.stop, outcome.rounds) == ("clean", 1)


def test_the_loop_keeps_the_fields_and_gradation_of_the_initial_sizes(
    tmp_path, monkeypatch
):
    """Fields and gradation of the recipe's sizes are passed to every round unchanged."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(2.5))
    initial = SizeRequests(20.0, 0.75, MappingProxyType({}), ())
    repair_mesh(inputs_for(tmp_path, initial=initial))
    assert {(job.sizes.max_size, job.sizes.gradation) for job in jobs} == {(20.0, 0.75)}


def test_zero_rounds_mesh_once_and_the_threshold_still_applies(tmp_path, monkeypatch):
    """No repair does not mean no judgement: a bad mesh stops with the rounds budget, a clean one is accepted."""
    jobs = script_rounds(monkeypatch, never_clean())
    rejected = repair_mesh(
        inputs_for(tmp_path, film([1]), budgets=Budgets(max_rounds=0))
    )
    assert (rejected.stop, rejected.rounds, len(jobs)) == ("rounds", 0, 1)
    assert rejected.last_audit is not None and rejected.last_audit.bad_count == 19
    jobs.clear()
    script_rounds(monkeypatch, lambda job: audit_of())
    accepted = repair_mesh(
        inputs_for(tmp_path, film([1]), budgets=Budgets(max_rounds=0))
    )
    assert (accepted.stop, accepted.rounds) == ("clean", 0)


def test_a_linear_mesh_job_asks_for_no_elevation(tmp_path, monkeypatch):
    """The default order is 1: no round is asked to write an elevated mesh."""
    jobs = script_rounds(monkeypatch, lambda job: audit_of())
    outcome = repair_mesh(inputs_for(tmp_path, film([1])))
    assert jobs[0].elevation is None
    assert (outcome.elevated_mesh, outcome.geometry_order) == (None, 1)


def test_a_higher_order_job_carries_the_elevation_target_and_the_tetrahedron_budget(
    tmp_path, monkeypatch
):
    """Every round may elevate; only the one the loop accepts does."""
    jobs = script_rounds(monkeypatch, bad_until_finer_than(2.5))
    outcome = repair_mesh(
        inputs_for(
            tmp_path,
            film([1]),
            mesher=MesherOptions(geometry_order=2),
            budgets=Budgets(max_tetrahedra=5000),
        )
    )
    assert [j.elevation.msh.name for j in jobs if j.elevation] == [
        f"round_{i:02d}.elevated.msh" for i in range(len(jobs))
    ]
    assert {j.elevation.max_tetrahedra for j in jobs if j.elevation} == {5000}
    assert {j.mesher.geometry_order for j in jobs} == {2}
    assert outcome.stop == "clean"


def test_the_elevated_mesh_of_the_accepted_round_is_the_one_to_deliver(
    tmp_path, monkeypatch
):
    """A clean stop with elevation delivers the elevated file of that round, whose audit is of the linear mesh."""
    elevated_rounds = []

    def fake_run_round(job, work_dir, timeout_seconds=None):
        elevated = job.elevation is not None and bool(job.sizes.local_sizes)
        elevated_rounds.append(elevated)
        audit = (
            audit_of()
            if elevated
            else audit_of((bad_element(1, at=SPOT, quality=0.009),))
        )
        return replace(outcome_of(audit), elevated=elevated)

    monkeypatch.setattr(repair, "run_round", fake_run_round)
    outcome = repair_mesh(
        inputs_for(tmp_path, film([1]), mesher=MesherOptions(geometry_order=2))
    )
    assert (outcome.stop, elevated_rounds) == ("clean", [False, True])
    assert outcome.last_mesh == tmp_path / "round_01.msh"
    assert outcome.elevated_mesh == tmp_path / "round_01.elevated.msh"
    assert outcome.geometry_order == 2


def test_a_clean_mesh_whose_order_could_not_be_raised_is_not_delivered(
    tmp_path, monkeypatch
):
    """The caller asked for a mesh of that order: the loop stops with its own reason and keeps the linear mesh."""
    monkeypatch.setattr(
        repair,
        "run_round",
        lambda job, work_dir, timeout_seconds=None: replace(
            outcome_of(audit_of()), elevation_error="setOrder failed"
        ),
    )
    outcome = repair_mesh(
        inputs_for(tmp_path, film([1]), mesher=MesherOptions(geometry_order=3))
    )
    assert outcome.stop == "elevation_failed"
    assert (outcome.elevated_mesh, outcome.geometry_order) == (None, 1)
    assert outcome.last_mesh == tmp_path / "round_00.msh"
    assert outcome.history[-1].elevation_error == "setOrder failed"
