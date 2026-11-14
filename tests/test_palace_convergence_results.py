"""The declared convergence of a returned run: the plan sealed by the handoff, result reading and the report.

The run folder is the one ``write_config()`` and ``prepare_handoff()`` write for a real Route B SGB build
(``palace_convergence_support``).  The run script is executed for real; the ``palace`` it starts is a
stand-in that copies pass folders written in Palace's formats into ``results/palace`` (four passes: three
``iteration<N>`` folders and the final result), and the copied recorder writes the receipt.  No solver runs.
"""

from __future__ import annotations

import json
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
from palace_convergence_support import electrostatic

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


def _matrix(path: Path, prefix: str, unit: str, matrix: list[list[float]]) -> None:
    columns = ",".join(f"{prefix}{j}] ({unit})" for j in range(1, len(matrix) + 1))
    rows = "".join(
        f"{i},{','.join(f'{value:+.12e}' for value in row)}\n"
        for i, row in enumerate(matrix, start=1)
    )
    path.write_text(f"i,{columns}\n{rows}", encoding="utf-8")


def _pass_tables(folder: Path, number: int, counts: dict[str, int]) -> None:
    """Every Electrostatic table of one pass, in Palace's headers, for two terminals."""
    folder.mkdir(parents=True, exist_ok=True)
    c11, c22 = (value * 1e-15 for value in _DIAGONALS_FF[number - 1])
    c12 = -5e-15
    maxwell = [[c11, c12], [c12, c22]]
    _matrix(folder / "terminal-C.csv", "C[i][", "F", maxwell)
    _matrix(folder / "terminal-Cm.csv", "C_m[i][", "F", maxwell)
    _matrix(
        folder / "terminal-Cinv.csv", "C⁻¹[i][", "1/F", [[1e13, 1e11], [1e11, 3e12]]
    )
    (folder / "terminal-V.csv").write_text(
        "i,V_inc[i] (V)\n1,1.0\n2,1.0\n", encoding="utf-8"
    )
    domains = counts["domain"]
    domain_headers = "".join(
        f",E_elec[{k}] (J),p_elec[{k}],E_mag[{k}] (J),p_mag[{k}]"
        for k in range(1, domains + 1)
    )
    domain_cells = ",1e-20,0.5,0.0,0.0" * domains
    (folder / "domain-E.csv").write_text(
        f"i,E_elec (J),E_mag (J),E_cap (J),E_ind (J){domain_headers}\n"
        + "".join(f"{i},1e-20,0.0,0.0,0.0{domain_cells}\n" for i in (1, 2)),
        encoding="utf-8",
    )
    surfaces = counts["surface"]
    surface_headers = "".join(
        f",p_surf[{k}],Q_surf[{k}]" for k in range(1, surfaces + 1)
    )
    (folder / "surface-Q.csv").write_text(
        f"i{surface_headers}\n"
        + "".join(
            f"{i}"
            + "".join(
                f",{1e-3 * k * (1 + 0.01 / number) * i:.12e},+inf"
                for k in range(1, surfaces + 1)
            )
            + "\n"
            for i in (1, 2)
        ),
        encoding="utf-8",
    )
    norm = _NORMS[number - 1]
    (folder / "error-indicators.csv").write_text(
        f"Norm,Minimum,Maximum,Mean\n{norm},1e-9,{norm},{norm / 10}\n", encoding="utf-8"
    )
    (folder / "palace.json").write_text(
        json.dumps(
            {
                "Problem": {
                    "DegreesOfFreedom": 1000 * number,
                    "MeshElements": 500 * number,
                    "MPISize": 1,
                    "OpenMPThreads": 1,
                    "Iteration": number,
                },
                "ElapsedTime": {
                    "Counts": {"Total": 1},
                    "Durations": {"Total": 2.0 * number, "Solve": 1.0 * number},
                },
                "PeakMemoryMegabytes": {
                    "Average": 10,
                    "Max": 10,
                    "Min": 10,
                    "Total": 10,
                },
                "PeakNodeMemoryMegabytes": {
                    "Average": 10,
                    "Max": 10,
                    "Min": 10,
                    "Total": 10,
                },
                "LinearSolver": {"TotalSolves": 2},
                "GitTag": "v0.18.1",
            }
        ),
        encoding="utf-8",
    )


def _isolated_python(directory: Path) -> Path:
    """A ``python3`` on PATH that sees only the standard library, as on a solver host."""
    directory.mkdir()
    wrapper = directory / "python3"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "$@"\n')
    wrapper.chmod(0o755)
    return directory


_DECLARED = (
    Capacitance.diagonal().within(relative=1e-3),
    SurfaceEpr.each("MA", "MS", margin_um=1.0).within(relative=2e-2),
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


def test_the_trust_report_shows_the_verdict_the_table_and_the_thresholds(returned):
    """The verdict sentence, the criterion-by-pass table and one change figure per criterion."""
    result = resolve_palace_result(
        returned.root, expected_handoff_id=returned.handoff_id
    )
    trust = result.show_run_trustworthiness()
    assert trust.convergence == result.convergence
    with (
        _captured_notebook_display() as displayed,
        patch("scgsim.palace.reports.trust.show_figure", displayed.append),
    ):
        trust._ipython_display_()  # noqa: SLF001 -- the notebook display hook
    html = "".join(item.data for item in displayed if isinstance(item.data, str))
    assert "Declared convergence" in html
    assert "Converged at pass 4, of the 4 passes read" in html
    assert "Pass 1</th>" in html and "Pass 4</th>" in html
    assert "A dashed line is the threshold declared with set_convergence()" in html
    assert "set aside: convergence declared" in html
    assert "not a newly invented acceptance Gate" not in html
    titles = [
        figure.layout.title.text
        for figure in displayed
        if hasattr(figure, "layout") and figure.layout.title.text
    ]
    assert "C diagonal: relative change vs AMR pass" in titles
    assert "error indicator (Norm): value vs AMR pass" in titles
    error_figure = next(
        figure
        for figure in displayed
        if hasattr(figure, "layout")
        and figure.layout.title.text == "AMR error indicator vs AMR pass"
    )
    assert [shape.y0 for shape in error_figure.layout.shapes] == [0.05]


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
    assert "C diagonal (C[pad][pad]): relative change 0.0099" in verdict_sentence(check)


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
