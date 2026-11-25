"""The views of a Palace Driven run, on the real runs of issue 187 (``palace_driven_runs_support``).

Each view answers one question, in the order a reader asks them.  The trust view: which run, and did its
numbers converge (the sweep, the FEM order, the mesh, a declared convergence, the adaptive sweep's samples per
port), with the reciprocity and passivity of the S-matrix recorded and not judged.  The physics view: |S| and
its phase against frequency per entry, then the Surface EPR of each interface type with each port driven.  The
benchmark: the full solves, their time and memory.  A returned run folder read as it came back, complete or
failed, has the identity, the failure and the sweep, and no S-parameters.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import shutil
import sys
import types
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from palace_driven_runs_support import (
    ADAPTIVE_RUN,
    INDUCTOR_RUN,
    LOG,
    RESISTOR_ADAPTIVE_RUN,
    RESISTOR_RUN,
    UNIFORM_RUN,
    copied_run,
    file_record,
    fixture_result,
    handoff_id,
    write_handoff,
)

from scgsim.contracts.palace_run_folder import OUTPUT_DIR, RETURNED_RECEIPT
from scgsim.palace import (
    DrivenPhysicsReport,
    DrivenTrustReport,
    Linear,
    Log,
    Points,
    SimulationBenchmarkReport,
    inspect_run_trustworthiness,
    show_epr,
)
from scgsim.palace.config.driven_sweep import DrivenSweep
from scgsim.palace.reports.driven_figures import (
    s_parameter_figures,
    surface_epr_figures,
)
from scgsim.palace.reports.plotting import plotly
from scgsim.palace.results.inspection_records import SurfaceMarginEprRecord

# What E1 and round 2 of slice 5 measured on these runs (issue 187), and what each run's palace.json records.
FULL_SOLVES = {
    UNIFORM_RUN: 10,  # 5 frequencies, 2 ports
    ADAPTIVE_RUN: 8,  # 4 samples for each of 2 ports
    RESISTOR_RUN: 5,  # 5 frequencies, 1 port
    INDUCTOR_RUN: 5,
    RESISTOR_ADAPTIVE_RUN: 4,  # 4 samples, 1 port
}


def resolved(name: str, tmp_path: Path):
    """The resolved Driven run of fixture ``name``, in its folder with its handoff record, as the report reads it."""
    root = copied_run(name, tmp_path)
    write_handoff(root)
    return fixture_result(root)


@contextlib.contextmanager
def captured_display():
    """What a view displays, in order: each HTML block's text and each figure, in a stand-in ``IPython.display``."""
    # Plotly reads IPython when it is first imported; import it first so it never caches the stand-in.
    plotly()
    displayed: list[object] = []
    module = types.ModuleType("IPython.display")
    module.HTML = lambda data: types.SimpleNamespace(data=data)  # type: ignore[attr-defined]
    module.display = displayed.append  # type: ignore[attr-defined]
    package = types.ModuleType("IPython")
    package.display = module  # type: ignore[attr-defined]
    saved = {name: sys.modules.get(name) for name in ("IPython", "IPython.display")}
    sys.modules.update({"IPython": package, "IPython.display": module})
    try:
        with (
            patch("scgsim.palace.reports.driven.show_figure", displayed.append),
            patch("scgsim.palace.reports.views.show_figure", displayed.append),
        ):
            yield displayed
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def plain(text: str) -> str:
    """``text`` with the style of its table cells left out, so a row reads ``<td>a</td><td>b</td>``."""
    return re.sub(r"<(t[dh]) style='[^']*'>", r"<\1>", text)


def shown(view) -> tuple[str, list]:
    """The HTML text ``view`` displays, and its figures."""
    with captured_display() as displayed:
        view._ipython_display_()  # noqa: SLF001 -- the notebook display hook
    text = "".join(
        item.data for item in displayed if isinstance(getattr(item, "data", None), str)
    )
    return text, [item for item in displayed if hasattr(item, "layout")]


# ---- Did it converge? -------------------------------------------------------------------------------------


def test_the_trust_view_names_the_run_and_how_each_port_s_sweep_converged(tmp_path):
    """E1's order-2 run: five identity cards, the sweep and the mesh, then the samples of each port."""
    trust = resolved(ADAPTIVE_RUN, tmp_path).show_run_trustworthiness()
    assert isinstance(trust, DrivenTrustReport)
    text, figures = shown(trust)
    for label, value in (
        ("Problem", "Driven"),
        ("Route", "B"),
        ("Profile", "direct-local"),
        ("AMR passes recorded", "1"),
        ("MPI × OpenMP", "8 MPI × 1 OMP"),
        ("Sweep", "2–10 GHz, 81 points"),
        ("Sampling", "adaptive, tolerance 0.001"),
        ("FEM order", "2"),
        ("Mesh elements", "204000"),
        ("Convergence", "not declared"),
    ):
        assert f">{label}</div>" in text
        assert f">{value}</div>" in text
    assert ">Estimated error ≤ 0.001</th>" in text
    # Palace's log: 4 samples each, at the same four frequencies, in the order Palace added them.
    for row in (
        ("1: chip/o_lumped_left", "4", "1.22e-06", "2, 10, 6.142, 3.744"),
        ("2: chip/o_lumped_right", "4", "4.57e-06", "2, 10, 6.142, 3.744"),
    ):
        assert "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" in plain(
            text
        )
    assert "Completeness" not in text and "Provenance" not in text
    assert figures == []


def test_a_uniform_sweep_solves_every_frequency_and_has_no_samples(tmp_path):
    """E1's order-1 run: a uniform sweep says so, and has no table of adaptive samples."""
    text, _ = shown(resolved(UNIFORM_RUN, tmp_path).show_run_trustworthiness())
    assert ">uniform</div>" in text and ">2–10 GHz, 5 points</div>" in text
    assert ">1</div>" in text  # FEM order 1
    assert "Samples" not in text


@pytest.mark.parametrize(
    ("ranges", "card"),
    [
        (
            [Linear(4e9, 9e9, points=501), Linear(7.30e9, 7.34e9, points=4001)],
            "4–9 GHz, 501 points; 7.3–7.34 GHz, 4,001 points; 4,497 frequencies",
        ),
        (
            [Log(1e8, 1e10, points=201), Points([7.3e9, 7.32e9])],
            "0.1–10 GHz, 201 points, log-spaced; 7.3, 7.32 GHz; 203 frequencies",
        ),
        ([Points([7.3e9, 7.4e9, 7.35e9, 7.31e9])], "4 points, 7.3–7.4 GHz"),
        ([Points([7.3e9, 7.3e9])], "7.3, 7.3 GHz; 1 frequency"),
    ],
    ids=["band and window", "log and points", "many points", "a repeated point"],
)
def test_the_sweep_card_names_each_range_and_the_frequencies_they_merge_into(
    tmp_path, ranges, card
):
    """The sweep card lists each range, then the frequencies Palace solves at when one range alone does not say it.

    The sweep is set on the report of a real run, to show the card alone: no Palace run of these ranges is read.
    """
    trust = resolved(UNIFORM_RUN, tmp_path).show_run_trustworthiness()
    text = replace(trust, sweep=DrivenSweep(ranges=ranges)).convergence_html()
    assert f">{card}</div>" in text


def test_the_full_record_is_behind_show_details(tmp_path):
    """``show_details`` adds the receipt, Palace's tag and the handoff, then the provenance."""
    trust = resolved(ADAPTIVE_RUN, tmp_path).show_run_trustworthiness(show_details=True)
    text, _ = shown(trust)
    assert ">completed, solver_ended / exit 0</div>" in text
    assert ">0dc74cd</div>" in text
    assert "<h3>Provenance</h3>" in text


def _largest_singular_value_of_two_ports(result) -> float:
    """The largest singular value of each 2 x 2 S, in closed form, the largest over the frequencies."""
    largest = 0.0
    for s11, s21, s12, s22 in zip(
        result.s(1, 1), result.s(2, 1), result.s(1, 2), result.s(2, 2), strict=True
    ):
        frobenius = abs(s11) ** 2 + abs(s21) ** 2 + abs(s12) ** 2 + abs(s22) ** 2
        determinant = abs(s11 * s22 - s12 * s21)
        largest = max(
            largest,
            math.sqrt((frobenius + math.sqrt(frobenius**2 - 4 * determinant**2)) / 2),
        )
    return largest


@pytest.mark.parametrize(
    ("run", "reciprocity"),
    [(UNIFORM_RUN, 3.5e-6), (ADAPTIVE_RUN, 7.7e-15)],
)
def test_reciprocity_and_passivity_are_the_largest_over_the_band(
    tmp_path, run, reciprocity
):
    """Recorded, not judged: the largest |S12 - S21| and the largest singular value of S over the sweep.

    The reciprocity values are slice 5's, measured on these runs (issue 187); the singular value is checked
    against the closed form of a 2 x 2 matrix.  The near-lossless box keeps it just below 1.
    """
    result = resolved(run, tmp_path)
    trust = result.show_run_trustworthiness()
    detached = result.driven_result()
    assert trust.reciprocity_error == max(
        abs(a - b) for a, b in zip(detached.s(1, 2), detached.s(2, 1), strict=True)
    )
    assert trust.reciprocity_error == pytest.approx(reciprocity, rel=0.02)
    # The closed form takes a square root of a near-cancelling difference, so it holds 9 digits, not 15.
    assert trust.largest_singular_value == pytest.approx(
        _largest_singular_value_of_two_ports(detached), rel=1e-9
    )
    assert 0.9999 < trust.largest_singular_value < 1.0
    text, _ = shown(trust)
    assert "<h3>Reciprocity and passivity</h3>" in text
    assert f">{trust.reciprocity_error:.2g}</div>" in text
    assert f">{trust.largest_singular_value:.6g}</div>" in text


@pytest.mark.parametrize("run", [RESISTOR_RUN, INDUCTOR_RUN, RESISTOR_ADAPTIVE_RUN])
def test_one_port_has_no_reciprocity_and_its_passivity_is_its_reflection(tmp_path, run):
    """A one-port S is |S11|: the largest singular value is its largest magnitude over the sweep."""
    trust = resolved(run, tmp_path).show_run_trustworthiness()
    assert trust.reciprocity_error is None
    assert trust.largest_singular_value == pytest.approx(
        max(abs(value) for value in trust.result.s(1, 1)), rel=1e-12
    )
    text, _ = shown(trust)
    assert "<h3>Passivity</h3>" in text
    assert "Reciprocity" not in text


def test_a_sweep_that_reached_its_maximum_is_reported_and_not_read(tmp_path):
    """The trust view says why ``driven_result()`` refuses the run; the physics view draws nothing."""
    root = copied_run(ADAPTIVE_RUN, tmp_path)
    log = root / LOG
    text = log.read_text(encoding="utf-8")
    assert text.count("Adaptive sampling converged with 4 frequency samples:") == 2
    log.write_text(
        text.replace(
            "Adaptive sampling converged with 4 frequency samples:",
            "Adaptive sampling reached maximum 4 frequency samples:",
            1,
        ),
        encoding="utf-8",
    )
    write_handoff(root)
    result = fixture_result(root)
    trust = result.show_run_trustworthiness()
    assert trust.result is None
    assert "reached its maximum of 4 frequency samples" in trust.readback_refusal
    trust_text, _ = shown(trust)
    assert "<strong>Not read</strong>" in trust_text
    assert "port 1 (chip/o_lumped_left)" in trust_text
    assert "Reciprocity" not in trust_text
    physics_text, figures = shown(result.show_physics_quantities())
    assert "No S-parameters were read." in physics_text
    assert figures == []


# ---- What are the numbers? --------------------------------------------------------------------------------


def test_the_physics_view_draws_each_entry_against_frequency_with_the_ports_named(
    tmp_path,
):
    """The ports with the plan's names, reflection, transmission and phase, then MA, MS and SA."""
    result = resolved(ADAPTIVE_RUN, tmp_path)
    physics = result.show_physics_quantities()
    assert isinstance(physics, DrivenPhysicsReport)
    text, figures = shown(physics)
    assert ">chip/o_lumped_left</td>" in text and ">chip/o_lumped_right</td>" in text
    assert ">Impedance (Ω)</th>" in text
    assert [figure.layout.title.text for figure in figures] == [
        "Reflection",
        "Transmission",
        "Phase",
        "MA participation",
        "MS participation",
        "SA participation",
    ]
    detached = result.driven_result()
    frequencies = [frequency / 1e9 for frequency in detached.frequencies_hz]
    reflection, transmission, phase = figures[:3]
    assert [trace.name for trace in reflection.data] == ["S11", "S22"]
    assert [trace.name for trace in transmission.data] == ["S21", "S12"]
    assert [trace.name for trace in phase.data] == ["S11", "S21", "S12", "S22"]
    for trace in transmission.data:
        output, excited = int(trace.name[1]), int(trace.name[2])
        assert list(trace.x) == frequencies
        assert list(trace.y) == [
            m[output - 1][excited - 1] for m in detached.magnitude_db
        ]
        # A line takes its excited port's style: S12 is dashed over S21, which it covers.
        assert trace.line.dash == ("solid" if excited == 1 else "dash")
    for trace in phase.data:
        output, excited = int(trace.name[1]), int(trace.name[2])
        assert list(trace.y) == [p[output - 1][excited - 1] for p in detached.phase_deg]
    assert reflection.layout.yaxis.title.text == "Magnitude (dB)"
    assert phase.layout.yaxis.title.text == "Phase (degrees)"
    assert reflection.layout.xaxis.title.text == "Frequency (GHz)"


def test_a_one_port_run_has_reflection_and_phase_only(tmp_path):
    """One port, one entry: no transmission figure, and no legend for a figure's single line."""
    detached = resolved(RESISTOR_RUN, tmp_path).driven_result()
    figures = s_parameter_figures(detached, "light")
    assert [figure.layout.title.text for figure in figures] == ["Reflection", "Phase"]
    assert all([trace.name for trace in figure.data] == ["S11"] for figure in figures)
    assert not figures[0].layout.showlegend


def test_the_surface_epr_of_a_type_is_its_rows_summed_for_each_driven_port(tmp_path):
    """Each port is a legend group; its line is the sum of the type's whole-surface rows at each frequency."""
    detached = resolved(ADAPTIVE_RUN, tmp_path).driven_result()
    figures = surface_epr_figures(detached, "light")
    assert [figure.layout.title.text for figure in figures] == [
        "MA participation",
        "MS participation",
        "SA participation",
    ]
    for figure, interface_type in zip(figures, ("MA", "MS", "SA"), strict=True):
        assert [trace.legendgroup for trace in figure.data] == [
            "Port 1 driven",
            "Port 2 driven",
        ]
        assert figure.layout.yaxis.title.text == "Participation"
        for trace, port in zip(figure.data, (1, 2), strict=True):
            records = [r for r in detached.surface_epr if r.excited_port == port]
            assert trace.name == "whole surface"
            assert list(trace.x) == [r.frequency_hz / 1e9 for r in records]
            assert list(trace.y) == [
                sum(
                    row.participation
                    for row in record.surfaces
                    if row.interface_type == interface_type
                )
                for record in records
            ]


def test_each_inset_is_a_line_of_its_port(tmp_path):
    """Margin rows at 0.5 and 1 um: each inset is a line of its port, in its port's colour, never added up.

    E1 had no bands, so the margin rows here are made from its whole-surface rows; the figure is what is tested.
    """
    detached = resolved(ADAPTIVE_RUN, tmp_path).driven_result()
    banded = replace(
        detached,
        surface_epr=tuple(
            replace(
                record,
                margins=tuple(
                    SurfaceMarginEprRecord(
                        index=100 * margin_index + row.index,
                        baseline_index=row.index,
                        margin_index=margin_index,
                        margin_um=margin_um,
                        bands=(margin_index,),
                        interface_type=row.interface_type,
                        name=row.name,
                        rows=row.rows,
                        face_kinds=row.face_kinds,
                        entities=row.entities,
                        nets=row.nets,
                        attributes=row.attributes,
                        participation=share * row.participation,
                        quality_factor=row.quality_factor / share,
                        loss_tangent=row.loss_tangent,
                    )
                    for margin_index, (margin_um, share) in enumerate(
                        ((0.5, 0.6), (1.0, 0.3)), start=1
                    )
                    for row in record.surfaces
                ),
            )
            for record in detached.surface_epr
        ),
    )
    ma = surface_epr_figures(banded, "light")[0]
    assert [(trace.legendgroup, trace.name) for trace in ma.data] == [
        ("Port 1 driven", "whole surface"),
        ("Port 1 driven", "inset 0.5 µm"),
        ("Port 1 driven", "inset 1 µm"),
        ("Port 2 driven", "whole surface"),
        ("Port 2 driven", "inset 0.5 µm"),
        ("Port 2 driven", "inset 1 µm"),
    ]
    whole, half, one = ma.data[:3]
    assert list(half.y) == pytest.approx([0.6 * value for value in whole.y])
    assert list(one.y) == pytest.approx([0.3 * value for value in whole.y])
    assert half.line.color == whole.line.color != ma.data[3].line.color
    assert (whole.line.dash, half.line.dash, one.line.dash) == ("solid", "dash", "dot")


def test_show_epr_names_the_view_of_a_driven_run_s_surface_epr(tmp_path):
    """``show_epr`` draws one Eigenmode mode; a Driven result is refused, naming the view that draws it."""
    detached = resolved(UNIFORM_RUN, tmp_path).driven_result()
    with pytest.raises(TypeError, match=r"drawn by show_physics_quantities\(\)"):
        show_epr(detached)


# ---- What did it cost? ------------------------------------------------------------------------------------


@pytest.mark.parametrize("run", list(FULL_SOLVES))
def test_the_benchmark_counts_the_full_solves_and_their_time(tmp_path, run):
    """Palace's linear solves: one per port and solved frequency; the wall time per solve is the total over them."""
    benchmark = resolved(run, tmp_path).show_simulation_benchmark()
    assert isinstance(benchmark, SimulationBenchmarkReport)
    assert benchmark.data["full_solves"] == FULL_SOLVES[run]
    palace_json = json.loads(
        (tmp_path / run / OUTPUT_DIR / "palace.json").read_text(encoding="utf-8")
    )
    total = palace_json["ElapsedTime"]["Durations"]["Total"]
    assert benchmark.data["durations"]["Total"] == total
    text, figures = shown(benchmark)
    per_solve = total / FULL_SOLVES[run]
    assert ">Wall time per solve</div>" in text
    assert f">{per_solve:.3g} s</div>" in text
    assert f">{FULL_SOLVES[run]}</div>" in text
    assert ">Peak node memory</div>" in text and "GiB</div>" in text
    assert "Cost by AMR pass" not in text
    assert [figure.layout.title.text for figure in figures] == [
        "Elapsed time by Palace timer"
    ]


def test_the_adaptive_sweep_s_reduced_model_has_its_timers(tmp_path):
    """r3's timers include the reduced model's construction and evaluation, PROMConstruction and PROMSolve."""
    _, figures = shown(resolved(ADAPTIVE_RUN, tmp_path).show_simulation_benchmark())
    assert {"PROMConstruction", "PROMSolve"} <= set(figures[0].data[0].y)


def test_every_view_takes_the_theme_of_its_figures(tmp_path):
    """Light by default, dark when asked, for each view and its figures."""
    result = resolved(ADAPTIVE_RUN, tmp_path)
    assert result.show_simulation_benchmark(theme="dark").trust.theme == "dark"
    assert result.show_physics_quantities(theme="dark").trust.theme == "dark"
    _, figures = shown(result.show_physics_quantities(theme="dark"))
    assert {figure.layout.paper_bgcolor for figure in figures} == {"#0d1117"}


def test_all_results_are_shown_in_the_order_a_reader_asks(tmp_path):
    """Which run and did it converge, then its numbers, then what it cost."""
    with captured_display() as displayed:
        resolved(ADAPTIVE_RUN, tmp_path).show_all_results()
    assert [type(item).__name__ for item in displayed] == [
        "DrivenTrustReport",
        "DrivenPhysicsReport",
        "SimulationBenchmarkReport",
    ]


# ---- A run folder read as it came back -------------------------------------------------------------------


def returned_folder(name: str, tmp_path: Path, *, status: str | None) -> Path:
    """Fixture run ``name`` as a returned run folder: its handoff, and a receipt of ``status`` unless None.

    The handoff seals the config, as ``prepare_handoff()`` does.  A failed run was killed before Palace wrote a
    table, as a run out of memory is: no ``results/palace``, a log that says so, exit code 137.
    """
    root = copied_run(name, tmp_path)
    if status == "failed":
        shutil.rmtree(root / OUTPUT_DIR)
        (root / LOG).write_text("solver killed\n", encoding="utf-8")
    hashes = write_handoff(root)
    if status is None:
        return root
    exit_code = 137 if status == "failed" else 0
    outputs = [f"{OUTPUT_DIR}/{table}" for table in ("port-S.csv", "palace.json")]
    (root / RETURNED_RECEIPT).write_text(
        json.dumps(
            {
                "schema": "palace-returned-run-receipt.v2",
                "schema_version": 2,
                "handoff_id": handoff_id(root),
                "route": "B",
                "problem": "Driven",
                "status": status,
                "exit_code": exit_code,
                "solver_exit_code": exit_code,
                "tee_exit_code": 0,
                "ending": "solver_ended",
                "identity_verified": status == "completed",
                "input_hashes": hashes,
                "output_files": [file_record(root, path) for path in outputs],
                "log": file_record(root, LOG),
                "convergence_decision": None,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_a_folder_without_its_receipt_is_partial_with_its_sweep_and_cost(tmp_path):
    """No receipt yet: partial, the identity, the sweep the config requests and the cost; no S-parameters."""
    trust = inspect_run_trustworthiness(
        returned_folder(ADAPTIVE_RUN, tmp_path, status=None)
    )
    assert isinstance(trust, DrivenTrustReport)
    assert trust.completeness == "partial" and trust.failure is None
    assert trust.result is None
    assert trust.full_solves == FULL_SOLVES[ADAPTIVE_RUN]
    text, _ = shown(trust)
    assert ">2–10 GHz, 81 points</div>" in text and ">partial</div>" in text
    assert "Samples" not in text and "Reciprocity" not in text
    physics_text, figures = shown(trust.show_physics_quantities())
    assert "No S-parameters were read." in physics_text
    assert "resolve_palace_result()" in physics_text
    assert figures == []


def test_a_failed_driven_run_says_why_and_keeps_its_sweep(tmp_path):
    """E1's order-2 config, killed before Palace wrote a table: the failure, the sweep, no cost."""
    trust = inspect_run_trustworthiness(
        returned_folder(ADAPTIVE_RUN, tmp_path, status="failed")
    )
    assert trust.completeness == "partial"
    assert trust.failure is not None and trust.failure.exit_code == 137
    text, _ = shown(trust)
    assert "Run failed" in text
    assert ">2–10 GHz, 81 points</div>" in text and ">2</div>" in text
    benchmark = trust.show_simulation_benchmark()
    assert benchmark.data["full_solves"] is None
    assert benchmark.data["run_state"]["failure"]["exit_code"] == 137


def test_a_complete_driven_folder_read_as_it_came_back_is_complete(tmp_path):
    """A completed receipt and a final solve of its own: complete, with its cost, and still no S-parameters."""
    trust = inspect_run_trustworthiness(
        returned_folder(UNIFORM_RUN, tmp_path, status="completed")
    )
    assert trust.completeness == "complete" and trust.failure is None
    assert trust.full_solves == FULL_SOLVES[UNIFORM_RUN]
    assert trust.result is None
    text, _ = shown(trust)
    assert "Completeness" not in text and "Run failed" not in text
