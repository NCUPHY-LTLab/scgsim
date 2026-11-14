"""Keep genuine native refinement rounds even when their matrices are equal."""

import json
from pathlib import Path

import pytest

from scgsim.palace.results.snapshots import collect_amr_passes


def _snapshot(path: Path, iteration: int | None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "terminal-C.csv").write_text("i,C[i][1] (F)\n1,1e-13\n")
    problem = {} if iteration is None else {"Iteration": iteration}
    (path / "palace.json").write_text(json.dumps({"Problem": problem}))


@pytest.mark.parametrize("final_iteration", [2, 4])
def test_equal_capacitance_keeps_distinct_native_iterations(tmp_path, final_iteration):
    """An unchanged matrix can belong to a different genuine refinement round."""
    _snapshot(tmp_path / "results/palace/iteration1", 1)
    _snapshot(tmp_path / "results/palace", final_iteration)
    observed = collect_amr_passes(tmp_path, "Electrostatic", None, None)
    assert tuple(item.native_iteration for item in observed.passes) == (
        1,
        final_iteration,
    )
    assert (
        observed.passes[0].capacitance_matrix_f
        == observed.passes[1].capacitance_matrix_f
    )


def test_a_top_level_copy_of_an_iteration_is_that_pass_read_from_the_iteration(
    tmp_path,
):
    """A top level that records iteration 1 with iteration1's very bytes is pass 1 once, read from iteration1.

    A tool that follows links turns a stopped run's linked top level into such copies; its palace.json can
    describe the next solve, so the iteration's own record is the one read.
    """
    _snapshot(tmp_path / "results/palace/iteration1", 1)
    _snapshot(tmp_path / "results/palace", 1)
    observed = collect_amr_passes(tmp_path, "Electrostatic", None, None)
    assert len(observed.passes) == 1
    assert observed.passes[0].native_iteration == 1
    assert observed.passes[0].source == "iteration1"
    assert observed.final_snapshot_status == "unreadable"


def test_a_folder_that_records_no_iteration_is_no_pass(tmp_path):
    """Without its own Problem.Iteration a folder has no established pass number, so none is invented."""
    _snapshot(tmp_path / "results/palace/iteration1", None)
    _snapshot(tmp_path / "results/palace", None)
    observed = collect_amr_passes(tmp_path, "Electrostatic", None, None)
    assert observed.passes == ()
    assert [item.reason for item in observed.set_aside] == [
        "iteration1 has a palace.json that records no Problem.Iteration",
        "the final result has a palace.json that records no Problem.Iteration",
    ]
