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
    observed = collect_amr_passes(
        tmp_path, "Electrostatic", None, None, failed_attempt=False
    )
    assert tuple(item.native_iteration for item in observed.passes) == (
        1,
        final_iteration,
    )
    assert (
        observed.passes[0].capacitance_matrix_f
        == observed.passes[1].capacitance_matrix_f
    )


def test_same_known_native_iteration_replaces_archived_copy(tmp_path):
    """A parent copy of the same native round replaces its archived copy."""
    _snapshot(tmp_path / "results/palace/iteration1", 1)
    _snapshot(tmp_path / "results/palace", 1)
    observed = collect_amr_passes(
        tmp_path, "Electrostatic", None, None, failed_attempt=False
    )
    assert len(observed.passes) == 1
    assert observed.passes[0].native_iteration == 1
    assert observed.passes[0].source == "final"


def test_missing_native_identity_does_not_infer_round_from_equal_matrix(tmp_path):
    """Unknown identity remains unknown instead of deduplicating by values."""
    _snapshot(tmp_path / "results/palace/iteration1", None)
    _snapshot(tmp_path / "results/palace", None)
    observed = collect_amr_passes(
        tmp_path, "Electrostatic", None, None, failed_attempt=False
    )
    assert len(observed.passes) == 2
    assert all(item.native_iteration is None for item in observed.passes)
