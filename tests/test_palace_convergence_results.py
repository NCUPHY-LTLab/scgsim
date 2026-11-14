"""The declared convergence of a returned run: the plan sealed by the handoff, result reading and the report.

The run folder is the one ``write_config()`` and ``prepare_handoff()`` write for a real Route B SGB build
(``palace_convergence_support``).  The run script is executed for real; the ``palace`` it starts is a
stand-in that copies pass folders written in Palace's formats into ``results/palace`` (four passes: three
``iteration<N>`` folders and the final result), and the copied recorder writes the receipt.  No solver runs.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from palace_convergence_support import electrostatic, write_electrostatic_pass

from scgsim.contracts.palace_run_folder import (
    CONVERGENCE_PLAN,
    HANDOFF_METADATA,
    INDEX_MAP,
    OUTPUT_DIR,
    RETURNED_RECEIPT,
    read_json,
)
from scgsim.palace import (
    Capacitance,
    ErrorIndicator,
    SurfaceEpr,
    inspect_run_trustworthiness,
    resolve_palace_result,
)
from scgsim.palace.reports.convergence import verdict_sentence
from scgsim.palace.reports.plotting import plotly
from scgsim.palace.results import convergence as convergence_module
from scgsim.palace.results.convergence import read_plan
from scgsim.palace.results.snapshots import collect_amr_passes
from scgsim.palace.results.surface_readers import (
    read_surface_bindings,
    read_surface_margin_bindings,
)
from scgsim.palace.results.tables import count_index_entries

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the run script is a POSIX shell script"
)

# The two diagonal elements (fF) and the error-indicator norm of each pass: C[pad][pad] settles at pass 3,
# C[plane][plane] at pass 2, and the norm falls below 0.05 at pass 2.
_DIAGONALS_FF = [(100.0, 300.0), (101.0, 300.06), (101.05, 300.07), (101.06, 300.07)]
_NORMS = [0.2, 0.04, 0.03, 0.02]

_FAKE_PALACE = """#!/usr/bin/env bash
# A stand-in for Palace: it reports the supported release, accepts the dry run, and copies the pass folders
# of a solve into the output folder.
if [ "$1" = "--version" ]; then
  printf 'Palace version: 0.18.1\\nSchema version: 1-7-0\\n'
  exit 0
fi
if [ "$1" = "--dry-run" ]; then
  exit 0
fi
cp -R "{solve}"/. results/palace/
"""


@contextmanager
def _captured_notebook_display():
    """Collect what the report displays, in a stand-in ``IPython.display`` whose ``HTML`` keeps its text."""
    # Plotly reads IPython when it is first imported; import it first so it never caches the stand-in.
    plotly()
    # Only the two IPython entries are replaced and restored: modules Plotly imports while the report
    # builds its figures stay loaded, so their classes stay the ones later tests use.
    displayed: list[object] = []
    module = types.ModuleType("IPython.display")
    module.HTML = lambda data: types.SimpleNamespace(data=data)  # type: ignore[attr-defined]
    module.display = displayed.append  # type: ignore[attr-defined]
    package = types.ModuleType("IPython")
    package.display = module  # type: ignore[attr-defined]
    saved = {name: sys.modules.get(name) for name in ("IPython", "IPython.display")}
    sys.modules.update({"IPython": package, "IPython.display": module})
    try:
        yield displayed
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _pass_tables(folder: Path, number: int, counts: dict[str, int]) -> None:
    """Every Electrostatic table of one pass, in Palace's headers, for two terminals."""
    write_electrostatic_pass(
        folder, number, counts, _DIAGONALS_FF[number - 1], _NORMS[number - 1]
    )


def _isolated_python(directory: Path) -> Path:
    """A ``python3`` on PATH that sees only the standard library, as on a solver host."""
    directory.mkdir()
    wrapper = directory / "python3"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "$@"\n')
    wrapper.chmod(0o755)
    return directory


_DECLARED = (
    Capacitance.diagonal().within(percent=0.1),
    SurfaceEpr.each("MA", "MS", margin_um=1.0).within(percent=2),
    ErrorIndicator().below(0.05),
)


def _returned_run(
    tmp_path: Path, *, declare: bool = True, criteria=_DECLARED
) -> SimpleNamespace:
    """Write the config and handoff of a four-pass run, run its script, and return the run folder and id."""
    sim = electrostatic(tmp_path / "model")
    sim.set_numerical(amr_max_passes=len(_DIAGONALS_FF) - 1)
    if declare:
        sim.set_convergence(*criteria, minimum_converged_passes=2)
    sim.write_config()
    counts = count_index_entries(read_json(sim.output_dir / INDEX_MAP))
    solve = tmp_path / "solve"
    for number in range(1, len(_DIAGONALS_FF)):
        _pass_tables(solve / f"iteration{number}", number, counts)
    _pass_tables(solve, len(_DIAGONALS_FF), counts)
    palace = tmp_path / "palace"
    palace.write_text(_FAKE_PALACE.replace("{solve}", str(solve)), encoding="utf-8")
    palace.chmod(0o755)
    plan = sim.prepare_handoff(
        profile="direct-local",
        executable=str(palace),
        resources={"processes": 1, "threads": 1},
    )
    python = _isolated_python(tmp_path / "bin")
    completed = subprocess.run(
        ["bash", str(plan.script_path)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{python}{os.pathsep}{os.environ['PATH']}"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert read_json(plan.returned_receipt_path)["status"] == "completed"
    return SimpleNamespace(root=plan.run_dir, handoff_id=plan.handoff_id, counts=counts)


@pytest.fixture(scope="module")
def returned(tmp_path_factory):
    """One declared four-pass run, returned and recorded."""
    return _returned_run(tmp_path_factory.mktemp("declared"))


def test_the_handoff_seals_the_plan_with_the_other_inputs(returned):
    """The plan is hashed after the index map, and the receipt carries the same hashes."""
    handoff = read_json(returned.root / HANDOFF_METADATA)
    paths = [entry["path"] for entry in handoff["hashes"]]
    assert paths[-2:] == [INDEX_MAP, CONVERGENCE_PLAN]
    receipt = read_json(returned.root / RETURNED_RECEIPT)
    assert receipt["input_hashes"] == handoff["hashes"]


def test_result_reading_gives_the_convergence_of_every_pass(returned):
    """``ResolvedPalaceResult.convergence``: the plan, four passes, converged at pass 3."""
    result = resolve_palace_result(
        returned.root, expected_handoff_id=returned.handoff_id
    )
    check = result.convergence
    assert check.plan is not None
    assert [item.number for item in check.passes] == [1, 2, 3, 4]
    assert [item.holds for item in check.passes] == [False, False, True, True]
    verdict = check.verdict
    assert (verdict.state, verdict.pass_number, verdict.passes_read) == (
        "converged",
        4,
        4,
    )
    assert verdict.last_to_hold == ("C diagonal",)
    assert verdict.last_to_hold_from == 3
    assert verdict.passes_read_after == 0
    assert verdict_sentence(check).startswith(
        "Converged at pass 4, of the 4 passes read (passes 1 to 4): every criterion held at passes 3 to 4, "
        "2 in a row."
    )


def test_the_check_reads_the_numbers_the_snapshot_readers_read(returned):
    """Each planned number equals the value the snapshot readers bind for the same pass and surfaces."""
    result = resolve_palace_result(
        returned.root, expected_handoff_id=returned.handoff_id
    )
    index_map = result.provenance.index_map
    snapshots = collect_amr_passes(
        returned.root,
        "Electrostatic",
        read_surface_bindings(index_map),
        read_surface_margin_bindings(index_map),
    ).passes
    assert len(snapshots) == len(result.convergence.passes) == 4
    for snapshot, check in zip(snapshots, result.convergence.passes, strict=True):
        diagonal, ma, ms, error = check.criteria
        matrix = snapshot.capacitance_matrix_f
        assert matrix is not None
        assert [value.value for value in diagonal.values] == [
            matrix[0][0],
            matrix[1][1],
        ]
        margins = snapshot.surface_margin_epr or ()
        for criterion, kind in ((ma, "MA"), (ms, "MS")):
            expected = [
                sum(
                    record.participation
                    for record in series.records
                    if record.interface_type == kind and record.margin_um == 1.0
                )
                for series in margins
            ]
            assert [value.value for value in criterion.values] == pytest.approx(
                expected, rel=1e-12
            )
        assert error.values[0].value == snapshot.error_norm


def test_whole_surface_participation_matches_the_whole_surface_records(
    tmp_path, returned
):
    """A whole-surface criterion sums exactly the records the snapshot reader marks as whole surfaces."""
    plan = read_json(returned.root / CONVERGENCE_PLAN)
    index_map = read_json(returned.root / INDEX_MAP)
    whole = read_surface_bindings(index_map) or ()
    margin_columns = {
        f"p_surf[{binding.index}]"
        for binding in read_surface_margin_bindings(index_map) or ()
    }
    for criterion in plan["criteria"][1:3]:
        for value in criterion["values"]:
            assert set(value["columns"]) <= margin_columns
            assert not set(value["columns"]) & {f"p_surf[{b.index}]" for b in whole}


def _shown(report) -> tuple[str, list]:
    """The HTML a report displays in a notebook, joined, and its figures in order."""
    with (
        _captured_notebook_display() as displayed,
        patch("scgsim.palace.reports.trust.show_figure", displayed.append),
    ):
        report._ipython_display_()  # noqa: SLF001 -- the notebook display hook
    html = "".join(item.data for item in displayed if isinstance(item.data, str))
    return html, [item for item in displayed if hasattr(item, "layout")]


def test_the_trust_report_shows_the_run_the_table_and_each_quantity_with_its_change(
    returned,
):
    """What a reader of a run needs: which run, the table of the criteria per pass, and the figures.

    The identity is five cards.  The table heads a criterion by its name and its threshold, a group's name
    once over its criteria.  Each quantity's figure is followed by the figure of its declared change.
    """
    result = resolve_palace_result(
        returned.root, expected_handoff_id=returned.handoff_id
    )
    trust = result.show_run_trustworthiness()
    assert trust.convergence == result.convergence
    html, figures = _shown(trust)
    cards = html.split("<h3>Convergence</h3>")[0]
    for label in ("Problem", "Route", "Profile", "AMR passes recorded", "MPI × OpenMP"):
        assert f">{label}</div>" in cards
    for label in ("Final snapshot", "Selection integrity", "AMR Tol", "Receipt"):
        assert f">{label}</div>" not in html
    assert "Converged at pass 4" not in html and "Numerical Evidence" not in html
    assert ">Pass</th>" in html and ">Solved elements</th>" in html
    assert ">C diagonal<br>" in html and ">≤ 0.1 %</span>" in html
    # MA and MS at one inset stand under the inset's name, said once, with the threshold they share.
    assert html.count("Surface EPR, inset 1 um") == 1
    assert "colspan='2'" in html and ">MA</th>" in html and ">MS</th>" in html
    assert ">N/A</td>" in html
    titles = [figure.layout.title.text for figure in figures]
    value, change = (
        titles.index("C[pad][pad] vs AMR pass"),
        titles.index("C diagonal Delta % vs AMR pass"),
    )
    assert value < change < titles.index("AMR error indicator vs AMR pass")
    assert not [title for title in titles if title.startswith("|Δ")]
    assert "Excitation 1: MA participation vs AMR pass" in titles
    by_kind = figures[titles.index("Excitation 1: MA participation vs AMR pass")]
    # Every inset the run records is a line, declared or not.
    assert [trace.name for trace in by_kind.data] == [
        "whole surface",
        "inset 0.5 um",
        "inset 1 um",
    ]
    shared = figures[
        titles.index("Surface EPR, inset 1 um Max Percent Delta vs AMR pass")
    ]
    assert [trace.name for trace in shared.data] == [
        "MA, excitation pad",
        "MA, excitation plane",
        "MS, excitation pad",
        "MS, excitation plane",
    ]
    assert shared.layout.yaxis.title.text == "Surface EPR Max Percent Delta"
    assert [shape.y0 for shape in shared.layout.shapes] == [2.0]
    # The label sits on the line: on a log axis an annotation is placed by the exponent.
    label = shared.layout.annotations[0]
    assert (label.text, label.y) == ("≤ 2 %", math.log10(2.0))
    error_figure = figures[titles.index("AMR error indicator vs AMR pass")]
    assert [shape.y0 for shape in error_figure.layout.shapes] == [0.05]
    assert error_figure.layout.annotations[0].text == "≤ 0.05"


def test_the_trust_report_with_details_adds_the_records_and_the_verdict(returned):
    """``show_details`` adds what the run folder records and the verdict as a sentence."""
    result = resolve_palace_result(
        returned.root, expected_handoff_id=returned.handoff_id
    )
    html, _ = _shown(result.show_run_trustworthiness(show_details=True))
    for label in ("Final snapshot", "Selection integrity", "AMR Tol", "Receipt"):
        assert f">{label}</div>" in html
    assert "set aside: convergence declared" in html
    assert "Converged at pass 4, of the 4 passes read" in html


def test_the_physics_view_opens_with_the_numbers_the_run_was_solved_for(returned):
    """The capacitance matrix of the last pass read, its rows and columns named by terminal."""
    result = resolve_palace_result(
        returned.root, expected_handoff_id=returned.handoff_id
    )
    with (
        _captured_notebook_display() as displayed,
        patch("scgsim.palace.reports.views.show_figure", lambda figure: None),
    ):
        result.show_physics_quantities()._ipython_display_()  # noqa: SLF001 -- the notebook display hook
    html = "".join(item.data for item in displayed if isinstance(item.data, str))
    assert "Values of pass 4, the last pass read." in html
    assert "Maxwell capacitance matrix (fF)" in html
    assert html.count(">pad</th>") == 2 and html.count(">plane</th>") == 2


def test_all_results_are_shown_in_the_order_a_reader_asks(returned):
    """Which run and did it converge, then its numbers, then what it cost."""
    result = resolve_palace_result(
        returned.root, expected_handoff_id=returned.handoff_id
    )
    with _captured_notebook_display() as displayed:
        result.show_all_results()
    assert [type(item).__name__ for item in displayed] == [
        "PalaceTrustReport",
        "PhysicsQuantitiesReport",
        "SimulationBenchmarkReport",
    ]


def test_a_partly_returned_run_shows_the_passes_it_holds(tmp_path):
    """Without the final result or the receipt, the trust report checks the iteration folders there are."""
    run = _returned_run(tmp_path)
    (run.root / RETURNED_RECEIPT).unlink()
    results = run.root / OUTPUT_DIR
    for child in results.iterdir():
        if child.is_file():
            child.unlink()
    (results / "iteration3").rename(results / "unfinished")
    report = inspect_run_trustworthiness(run.root)
    check = report.convergence
    assert [item.number for item in check.passes] == [1, 2]
    assert check.verdict.state == "not_converged"
    assert verdict_sentence(check).startswith(
        "Not converged in the 2 passes read (passes 1 to 2). At pass 2:"
    )
    assert "C diagonal (C[pad][pad]): changed by 0.99 %, above the declared 0.1 %" in (
        verdict_sentence(check)
    )


def test_a_changed_plan_is_refused(tmp_path):
    """The handoff sealed the plan: result reading and the trust report refuse one changed since."""
    run = _returned_run(tmp_path)
    plan_path = run.root / CONVERGENCE_PLAN
    payload = read_json(plan_path)
    payload["criteria"][0]["threshold"] = 0.5
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    with pytest.raises(ValueError, match=f"mismatch for {CONVERGENCE_PLAN}"):
        inspect_run_trustworthiness(run.root)
    # Without a receipt to compare with, the handoff's own hash of the plan refuses it.
    (run.root / RETURNED_RECEIPT).unlink()
    with pytest.raises(ValueError, match="changed after the handoff"):
        inspect_run_trustworthiness(run.root)


def test_a_run_without_a_declaration_is_not_declared_and_refuses_an_added_plan(
    tmp_path,
):
    """No plan: ``not declared``.  A plan put into the folder afterwards was never sealed and is refused."""
    run = _returned_run(tmp_path, declare=False)
    result = resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    assert result.convergence.verdict.state == "not_declared"
    assert result.convergence.plan is None
    handoff = read_json(run.root / HANDOFF_METADATA)
    assert CONVERGENCE_PLAN not in [entry["path"] for entry in handoff["hashes"]]
    (run.root / CONVERGENCE_PLAN).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="is not among the inputs"):
        read_plan(run.root, handoff, "Electrostatic")


def test_without_a_declared_indicator_the_indicator_figure_draws_no_line(tmp_path):
    """The written Tol only keeps Palace from stopping, so it is not drawn as a threshold."""
    run = _returned_run(tmp_path, criteria=_DECLARED[:1])
    trust = inspect_run_trustworthiness(run.root)
    figures: list = []
    with (
        _captured_notebook_display(),
        patch("scgsim.palace.reports.trust.show_figure", figures.append),
    ):
        trust._ipython_display_()  # noqa: SLF001 -- the notebook display hook
    indicator = next(
        figure
        for figure in figures
        if figure.layout.title.text == "AMR error indicator vs AMR pass"
    )
    assert indicator.layout.shapes == ()


def test_a_partly_returned_run_counts_the_passes_it_read(tmp_path):
    """Passes 1, 4 and 5 are three passes read; none is "after convergence" by Palace's numbers (round 1, B3).

    The folder keeps iteration1, iteration4 and the final result of pass 5, each recording its own pass; the
    criterion is on the indicator, which holds at passes 4 and 5.
    """
    run = _returned_run(tmp_path, criteria=(ErrorIndicator().below(0.035),))
    results = run.root / OUTPUT_DIR
    (run.root / RETURNED_RECEIPT).unlink()
    # The solves of a longer run, as a partial copy of it leaves them: passes 1, 4 and 5.
    shutil.rmtree(results / "iteration2")
    (results / "iteration3").rename(results / "iteration4")
    _record_iteration(results / "iteration4", 4)
    _record_iteration(results, 5)
    check = inspect_run_trustworthiness(run.root).convergence
    assert [item.number for item in check.passes] == [1, 4, 5]
    verdict = check.verdict
    assert (verdict.state, verdict.pass_number) == ("converged", 5)
    assert verdict.passes_read == 3
    assert verdict.passes_read_after == 0
    sentence = verdict_sentence(check)
    assert sentence.startswith(
        "Converged at pass 5, of the 3 passes read (passes 1, 4 and 5)"
    )
    assert "-" not in sentence.split("Last to hold")[1]
    assert "No pass was read after it." in sentence
    assert sentence.endswith(
        "Passes 2 and 3 were not read, so the run may have converged earlier."
    )


def _record_iteration(folder: Path, iteration: int) -> None:
    """Make ``folder``'s ``palace.json`` record ``iteration`` as its ``Problem.Iteration``."""
    payload = read_json(folder / "palace.json")
    payload["Problem"]["Iteration"] = iteration
    (folder / "palace.json").write_text(json.dumps(payload), encoding="utf-8")


def test_a_run_without_a_plan_reads_no_pass_folder(tmp_path, monkeypatch):
    """Resolving an undeclared run never collects its passes, so it costs what it did before (round 1, R1)."""
    run = _returned_run(tmp_path, declare=False)

    def refuse(*_args, **_kwargs):
        raise AssertionError("an undeclared run must not collect its passes")

    monkeypatch.setattr(convergence_module, "returned_passes", refuse)
    result = resolve_palace_result(run.root, expected_handoff_id=run.handoff_id)
    assert result.convergence.verdict.state == "not_declared"
